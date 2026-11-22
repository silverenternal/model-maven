use crate::{
    adapters::worker_assets::stage_worker_package,
    ports::{ArtifactExporter, TrainingExportRequest},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tempfile::Builder;

pub struct FileArtifactExporter;

fn sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_file(path: &Path, contents: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut file = File::create(path).map_err(|error| error.to_string())?;
    file.write_all(contents)
        .map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut contents = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    contents.push(b'\n');
    write_file(path, &contents)
}

fn append_bundle_file_hashes(
    root: &Path,
    directory: &Path,
    files: &mut Map<String, Value>,
) -> Result<(), String> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        if file_type.is_symlink() {
            return Err(format!("导出支持文件不能包含符号链接：{}", path.display()));
        }
        if file_type.is_dir() {
            append_bundle_file_hashes(root, &path, files)?;
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|error| error.to_string())?
                .to_string_lossy()
                .replace('\\', "/");
            files.insert(relative, json!(sha256(&path)?));
        }
    }
    Ok(())
}

fn input_projection(inputs: &Value) -> Map<String, Value> {
    const FIELDS: &[&str] = &[
        "model",
        "model_params",
        "model_injections",
        "batch",
        "batch_params",
        "batch_binding",
        "training_step",
        "training_params",
        "training_mode",
        "optimizer",
        "lr",
        "weight_decay",
        "seed",
        "redacted_fields",
    ];
    FIELDS
        .iter()
        .filter_map(|field| {
            inputs
                .get(*field)
                .filter(|value| !value.is_null())
                .map(|value| ((*field).to_owned(), value.clone()))
        })
        .collect()
}

fn result_projection(report: &Value) -> Value {
    const FIELDS: &[&str] = &[
        "training_mode",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "requested_steps",
        "completed_steps",
        "optimizer_updates",
        "optimizer_updates_by_group",
        "mean_loss",
        "last_loss",
        "mean_losses",
        "mean_metrics",
        "checkpoint_completed_steps",
        "model_state_sha256",
    ];
    let Some(data) = report.get("data").and_then(Value::as_object) else {
        return Value::Object(Map::new());
    };
    FIELDS
        .iter()
        .filter_map(|field| {
            data.get(*field)
                .map(|value| ((*field).into(), value.clone()))
        })
        .collect::<Map<_, _>>()
        .into()
}

fn environment_runtime_comparison(
    environment: &Value,
    recorded_python: Option<&str>,
    recorded_torch: Option<&str>,
) -> (String, Value) {
    let mut comparison = json!({
        "python_version":{"recorded":recorded_python,"snapshot":environment["python"]["version"]},
        "pytorch_version":{"recorded":recorded_torch,"snapshot":environment["pytorch_version"]}
    });
    if environment["schema_version"] != 1 {
        return (
            environment["status"]
                .as_str()
                .unwrap_or("unavailable")
                .to_owned(),
            comparison,
        );
    }

    let python_matches = recorded_python
        .is_some_and(|expected| environment["python"]["version"].as_str() == Some(expected));
    let torch_matches = recorded_torch
        .is_some_and(|expected| environment["pytorch_version"].as_str() == Some(expected));
    comparison["python_version"]["status"] = json!(if python_matches {
        "matches"
    } else {
        "differs_or_unknown"
    });
    comparison["pytorch_version"]["status"] = json!(if torch_matches {
        "matches"
    } else {
        "differs_or_unknown"
    });
    let status = if python_matches && torch_matches {
        "captured"
    } else {
        "runtime_mismatch"
    };
    (status.to_owned(), comparison)
}

fn relative_source_path(root: &Path, source: &Path) -> Result<PathBuf, String> {
    let canonical = source
        .canonicalize()
        .map_err(|error| format!("源码不可读取 {}：{error}", source.display()))?;
    let relative = canonical
        .strip_prefix(root)
        .map_err(|_| format!("源码位于项目目录之外：{}", source.display()))?;
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("源码路径无效：{}", source.display()));
    }
    let is_python = relative.extension().and_then(|value| value.to_str()) == Some("py");
    let is_component_manifest = relative.extension().and_then(|value| value.to_str())
        == Some("json")
        && relative.starts_with(Path::new(".model-maven/components"));
    if !is_python && !is_component_manifest {
        return Err(format!("源码清单含未支持的文件：{}", source.display()));
    }
    Ok(relative.to_owned())
}

fn collect_component_source(
    root: &Path,
    reference: &Value,
    sources: &mut Map<String, Value>,
) -> Result<(), String> {
    let id = reference["id"]
        .as_str()
        .ok_or("组件锁缺少组件 ID，无法导出恢复清单。")?;
    let version = reference["version"]
        .as_str()
        .ok_or("组件锁缺少精确版本，无法导出恢复清单。")?;
    let digest = reference["sha256"]
        .as_str()
        .ok_or("组件锁缺少清单 SHA-256，无法导出恢复清单。")?;
    let (namespace, name) = id
        .split_once('/')
        .ok_or("组件锁 ID 格式无效，无法导出恢复清单。")?;
    let valid_segment = |value: &str| {
        !value.is_empty()
            && value.len() <= 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    };
    if !valid_segment(namespace)
        || !valid_segment(name)
        || name.contains('/')
        || version.split('.').count() != 3
        || version
            .split('.')
            .any(|part| part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()))
        || digest.len() != 64
        || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(format!("组件锁格式无效，无法导出 {id}@{version}。"));
    }
    let relative = PathBuf::from(".model-maven")
        .join("components")
        .join(namespace)
        .join(name)
        .join(format!("{version}.json"));
    let relative_name = relative.to_string_lossy().replace('\\', "/");
    if reference["path"].as_str() != Some(relative_name.as_str()) {
        return Err(format!("组件锁路径与 ID/版本不匹配：{id}@{version}。"));
    }
    let source = root.join(&relative);
    let verified_relative = relative_source_path(root, &source)?;
    if verified_relative != relative {
        return Err(format!("组件清单解析到意外路径：{id}@{version}。"));
    }
    let key = source.to_string_lossy().into_owned();
    if let Some(previous) = sources.insert(key, json!(digest))
        && previous.as_str() != Some(digest)
    {
        return Err(format!("同一组件清单出现冲突指纹：{id}@{version}。"));
    }
    if let Some(dependencies) = reference["dependencies"].as_array() {
        for dependency in dependencies {
            collect_component_source(root, dependency, sources)?;
        }
    }
    Ok(())
}

fn collect_injection_component_sources(
    root: &Path,
    injections: &Value,
    sources: &mut Map<String, Value>,
) -> Result<(), String> {
    if let Some(injections) = injections.as_object() {
        for injection in injections.values() {
            if let Some(reference) = injection.get("component_ref") {
                collect_component_source(root, reference, sources)?;
            }
        }
    }
    Ok(())
}

fn collect_edit_component_sources(
    root: &Path,
    edit: &Value,
    sources: &mut Map<String, Value>,
) -> Result<(), String> {
    if edit["action"] == "transaction" {
        if let Some(operations) = edit["operations"].as_array() {
            for operation in operations {
                collect_edit_component_sources(root, operation, sources)?;
            }
        }
    } else if let Some(reference) = edit.get("component_ref") {
        collect_component_source(root, reference, sources)?;
    }
    Ok(())
}

fn add_component_sources(
    root: &Path,
    run: &Value,
    sources: &mut Map<String, Value>,
) -> Result<(), String> {
    collect_injection_component_sources(root, &run["inputs"]["model_injections"], sources)?;
    let variant = &run["inputs"]["variant"];
    if variant.is_object() {
        collect_injection_component_sources(root, &variant["origin"]["model_injections"], sources)?;
        if let Some(edits) = variant["edits"].as_array() {
            let cursor = variant["cursor"].as_u64().unwrap_or(edits.len() as u64) as usize;
            for edit in edits.iter().take(cursor.min(edits.len())) {
                collect_edit_component_sources(root, edit, sources)?;
            }
        }
    }
    Ok(())
}

struct VerifiedExtensionFile {
    source: PathBuf,
    relative: PathBuf,
    sha256: String,
    size_bytes: u64,
}

struct VerifiedExtensionPackage {
    distribution: String,
    version: String,
    role: String,
    components: Vec<String>,
    required_by: Vec<String>,
    bundle_path: PathBuf,
    files: Vec<VerifiedExtensionFile>,
    omitted_files: Vec<Value>,
    size_bytes: u64,
}

fn relative_bundle_path(value: &str) -> Result<PathBuf, String> {
    if value.is_empty()
        || value.contains('\\')
        || Path::new(value).is_absolute()
        || Path::new(value)
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("扩展包含不安全的相对路径：{value}"));
    }
    Ok(PathBuf::from(value))
}

fn verify_extension_package_plans(
    plans: &[Value],
) -> Result<Vec<VerifiedExtensionPackage>, String> {
    if plans.len() > 64 {
        return Err("单次导出的扩展 distribution 超过 64 个。".into());
    }
    let mut packages = Vec::new();
    let mut total_file_count = 0usize;
    let mut total_bytes = 0u64;
    for plan in plans {
        let distribution = plan["distribution"]
            .as_str()
            .ok_or("扩展包计划缺少 distribution 名称。")?
            .to_owned();
        let version = plan["version"]
            .as_str()
            .ok_or("扩展包计划缺少 distribution 版本。")?
            .to_owned();
        let role = plan["role"]
            .as_str()
            .filter(|role| matches!(*role, "extension" | "runtime_dependency"))
            .ok_or("扩展包计划缺少有效的包角色。")?
            .to_owned();
        let source_root = Path::new(
            plan["source_root"]
                .as_str()
                .ok_or("扩展包计划缺少来源目录。")?,
        )
        .canonicalize()
        .map_err(|error| format!("无法读取扩展包来源目录：{error}"))?;
        if !source_root.is_dir() {
            return Err(format!("扩展包来源不是目录：{}", source_root.display()));
        }
        let components = plan["components"]
            .as_array()
            .ok_or("扩展包计划缺少组件列表。")?
            .iter()
            .map(|component| {
                component
                    .as_str()
                    .map(str::to_owned)
                    .ok_or("扩展包组件 ID 无效。")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let required_by = plan["required_by"]
            .as_array()
            .ok_or("扩展包计划缺少依赖来源列表。")?
            .iter()
            .map(|item| item.as_str().map(str::to_owned).ok_or("依赖来源无效。"))
            .collect::<Result<Vec<_>, _>>()?;
        let omitted_files = plan["omitted_files"]
            .as_array()
            .cloned()
            .ok_or("扩展包计划缺少未纳入文件列表。")?;
        let file_plans = plan["files"].as_array().ok_or("扩展包计划缺少文件列表。")?;
        if file_plans.is_empty() || file_plans.len() > 20_000 {
            return Err(format!("扩展 {distribution} 文件数为空或超过 20000。"));
        }
        let mut files = Vec::with_capacity(file_plans.len());
        let mut names = std::collections::BTreeSet::new();
        let mut package_bytes = 0u64;
        for file in file_plans {
            let relative = relative_bundle_path(
                file["relative_path"]
                    .as_str()
                    .ok_or("扩展包文件缺少相对路径。")?,
            )?;
            if !names.insert(relative.clone()) {
                return Err(format!("扩展包重复列出文件：{}", relative.display()));
            }
            let source = Path::new(
                file["source_path"]
                    .as_str()
                    .ok_or("扩展包文件缺少来源路径。")?,
            );
            if !source.is_absolute() {
                return Err("扩展包文件来源路径必须是绝对路径。".into());
            }
            let mut path = source_root.clone();
            for component in relative.components() {
                let Component::Normal(component) = component else {
                    return Err("扩展包文件路径含特殊路径项。".into());
                };
                path.push(component);
                if fs::symlink_metadata(&path)
                    .is_ok_and(|metadata| metadata.file_type().is_symlink())
                {
                    return Err(format!("拒绝归档扩展包符号链接：{}", path.display()));
                }
            }
            let canonical_source = source
                .canonicalize()
                .map_err(|error| format!("扩展包文件不可读取：{error}"))?;
            let canonical_expected = path
                .canonicalize()
                .map_err(|error| format!("扩展包文件缺失：{error}"))?;
            if canonical_source != canonical_expected || !canonical_source.starts_with(&source_root)
            {
                return Err(format!(
                    "扩展包文件来源与相对路径不一致：{}",
                    relative.display()
                ));
            }
            let metadata = fs::metadata(&canonical_source).map_err(|error| error.to_string())?;
            if !metadata.is_file() || metadata.len() > 256 * 1024 * 1024 {
                return Err(format!(
                    "扩展包文件不是普通文件或超过 256 MiB：{}",
                    relative.display()
                ));
            }
            let expected_size = file["size_bytes"].as_u64().ok_or("扩展包文件大小无效。")?;
            if metadata.len() != expected_size {
                return Err(format!(
                    "扩展包文件在计划后发生变化：{}",
                    relative.display()
                ));
            }
            let expected_hash = file["sha256"]
                .as_str()
                .filter(|value| {
                    value.len() == 64
                        && value
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
                .ok_or("扩展包文件 SHA-256 无效。")?
                .to_owned();
            if sha256(&canonical_source)? != expected_hash {
                return Err(format!(
                    "扩展包文件在计划后发生变化：{}",
                    relative.display()
                ));
            }
            package_bytes = package_bytes.saturating_add(expected_size);
            files.push(VerifiedExtensionFile {
                source: canonical_source,
                relative,
                sha256: expected_hash,
                size_bytes: expected_size,
            });
        }
        if package_bytes != plan["size_bytes"].as_u64().unwrap_or(u64::MAX) {
            return Err(format!("扩展 {distribution} 的文件总大小与计划不一致。"));
        }
        total_file_count = total_file_count.saturating_add(files.len());
        total_bytes = total_bytes.saturating_add(package_bytes);
        if total_file_count > 20_000 || total_bytes > 2 * 1024 * 1024 * 1024 {
            return Err("扩展包归档超过 20000 个文件或 2 GiB 上限。".into());
        }
        let mut content_hash = Sha256::new();
        content_hash.update(distribution.as_bytes());
        content_hash.update([0]);
        content_hash.update(version.as_bytes());
        for file in &files {
            content_hash.update(file.relative.to_string_lossy().as_bytes());
            content_hash.update([0]);
            content_hash.update(file.sha256.as_bytes());
        }
        let bundle_path =
            PathBuf::from("vendor/packages").join(format!("{:x}", content_hash.finalize()));
        packages.push(VerifiedExtensionPackage {
            distribution,
            version,
            role,
            components,
            required_by,
            bundle_path,
            files,
            omitted_files,
            size_bytes: package_bytes,
        });
    }
    Ok(packages)
}

fn verify_runtime_requirements(requirements: &[Value]) -> Result<(), String> {
    if requirements.len() > 64 {
        return Err("导出运行时依赖超过 64 项。".into());
    }
    let mut names = std::collections::BTreeSet::new();
    for requirement in requirements {
        let distribution = requirement["distribution"]
            .as_str()
            .filter(|name| {
                !name.is_empty()
                    && name.len() <= 128
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
            })
            .ok_or("运行时依赖 distribution 名称无效。")?;
        let key = distribution
            .chars()
            .map(|character| {
                if matches!(character, '-' | '_' | '.') {
                    '-'
                } else {
                    character.to_ascii_lowercase()
                }
            })
            .collect::<String>();
        if !names.insert(key) {
            return Err(format!("运行时依赖重复：{distribution}。"));
        }
        requirement["version"]
            .as_str()
            .filter(|version| !version.is_empty() && version.len() <= 128)
            .ok_or("运行时依赖缺少安装版本。")?;
        let constraints = requirement["constraints"]
            .as_array()
            .ok_or("运行时依赖缺少约束列表。")?;
        if constraints.len() > 128
            || constraints
                .iter()
                .any(|value| value.as_str().is_none_or(|text| text.len() > 512))
        {
            return Err("运行时依赖约束列表无效。".into());
        }
        let required_by = requirement["required_by"]
            .as_array()
            .ok_or("运行时依赖缺少来源列表。")?;
        if required_by.is_empty()
            || required_by.len() > 256
            || required_by.iter().any(|value| {
                value
                    .as_str()
                    .is_none_or(|text| text.is_empty() || text.len() > 256)
            })
        {
            return Err("运行时依赖来源列表无效。".into());
        }
    }
    Ok(())
}

fn python_restore_script() -> &'static str {
    r#"from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path, PurePosixPath

import torch

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

for requirement in manifest.get("runtime_requirements", []):
    distribution = requirement.get("distribution")
    expected_version = requirement.get("version")
    try:
        actual_version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise SystemExit(f"缺少导出所需运行时依赖：{distribution}=={expected_version}") from error
    if actual_version != expected_version:
        raise SystemExit(
            f"运行时依赖版本不匹配：{distribution} 需要 {expected_version}，当前为 {actual_version}"
        )

def sha256_path(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def checked_relative(base, relative, directory=False):
    if not isinstance(relative, str) or "\\\\" in relative:
        raise SystemExit("导出清单含无效的相对路径")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(f"导出路径越界：{relative}")
    candidate = base.joinpath(*path.parts).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        raise SystemExit(f"导出路径越界：{relative}")
    if directory and not candidate.is_dir():
        raise SystemExit(f"导出目录缺失：{relative}")
    if not directory and not candidate.is_file():
        raise SystemExit(f"导出文件缺失：{relative}")
    return candidate

for relative, expected in manifest.get("bundle_files", {}).items():
    path = checked_relative(HERE, relative)
    if sha256_path(path) != expected:
        raise SystemExit(f"导出支持文件 SHA-256 校验失败：{relative}")

for package in reversed(manifest.get("extension_packages", [])):
    bundle_root = checked_relative(HERE, package.get("bundle_path"), directory=True)
    for relative, expected in package.get("files", {}).items():
        path = checked_relative(bundle_root, relative)
        actual = sha256_path(path)
        if actual != expected:
            raise SystemExit(f"扩展包文件 SHA-256 校验失败：{relative}")
    sys.path.insert(0, str(bundle_root))

SOURCE = HERE / "source"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))
import model_maven_worker as worker
worker.torch = torch
for relative, expected in manifest["source_files"].items():
    path = (SOURCE / relative).resolve()
    try:
        path.relative_to(SOURCE.resolve())
    except ValueError:
        raise SystemExit(f"导出源码路径越界：{relative}")
    if not path.is_file():
        raise SystemExit(f"导出源码缺失或路径越界：{relative}")
    actual = sha256_path(path)
    if actual != expected:
        raise SystemExit(f"导出源码校验失败：{relative}")

inputs = manifest["inputs"]
variant_path = HERE / "variant.json"
variant = json.loads(variant_path.read_text(encoding="utf-8")) if variant_path.exists() else None

def contains_redacted(value):
    if isinstance(value, str):
        return value == "[REDACTED]"
    if isinstance(value, dict):
        return any(contains_redacted(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_redacted(item) for item in value)
    return False

if contains_redacted(inputs.get("model_params", {})) or contains_redacted(inputs.get("model_injections", {})) or contains_redacted(variant):
    raise SystemExit("模型入口含已脱敏参数。请先在 manifest.json 或 variant.json 中填回本地凭证，再重新运行。")

if variant:
    previous_root = Path(variant["origin"]["project_root"])
    variant["origin"]["project_root"] = str(SOURCE.resolve())
    rewritten = {}
    for original, digest in variant["source_manifest"].items():
        relative = Path(original).resolve().relative_to(previous_root.resolve())
        rewritten[str((SOURCE / relative).resolve())] = digest
    variant["source_manifest"] = rewritten
    variant["revision"] = worker.digest({key: value for key, value in variant.items() if key != "revision"})

checkpoint_info = manifest["checkpoint"]
checkpoint_path = HERE / checkpoint_info["file"]
actual_checkpoint_hash = sha256_path(checkpoint_path)
if actual_checkpoint_hash != checkpoint_info["sha256"]:
    raise SystemExit("checkpoint SHA-256 校验失败")

request = {
    "project_root": str(SOURCE.resolve()),
    "model": inputs.get("model"),
    "model_params": inputs.get("model_params", {}),
    "model_injections": inputs.get("model_injections", {}),
    "variant": variant,
    "seed": inputs.get("seed"),
}
model, _, _, _ = worker.load_model(request)
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
if not isinstance(payload, dict) or payload.get("schema_version") != 1:
    raise SystemExit("不支持此 checkpoint 格式")
if not isinstance(payload.get("model_state"), dict):
    raise SystemExit("checkpoint 不包含有效的模型状态")
if type(payload.get("completed_steps")) is not int or payload["completed_steps"] < 0:
    raise SystemExit("checkpoint 的训练步数无效")
model.load_state_dict(payload["model_state"], strict=True)
model.eval()
state_hash = worker.model_state_digest(model)
if state_hash != checkpoint_info["model_state_sha256"]:
    raise SystemExit("恢复后的模型权重指纹与导出清单不一致")
training_state = payload

def load_training_state():
    """Return the verified, weights-only checkpoint payload for caller-managed recovery."""
    return training_state

_managed_resume_used = False

def resume_managed_steps(additional_steps, *, checkpoint_directory=None, from_checkpoint=None):
    """Continue a supported CPU managed-step run once, optionally saving a new checkpoint.

    The exported checkpoint remains unchanged. A failed call may have partially
    updated in-memory state; reload this module before attempting it again.
    """
    global _managed_resume_used
    if _managed_resume_used:
        raise RuntimeError("恢复接口只能调用一次；重新加载导出包以从原 checkpoint 再试。")
    if type(additional_steps) is not int or not 1 <= additional_steps <= 1000:
        raise ValueError("additional_steps 必须是 1 到 1000 的整数。")
    source_state = training_state
    bundle_digest = sha256_path(HERE / "manifest.json")
    if from_checkpoint is not None:
        directory = Path(from_checkpoint)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("续跑 checkpoint 目录不存在或为符号链接。")
        latest = directory / "latest.json"
        if latest.is_symlink() or not latest.is_file():
            raise ValueError("续跑 checkpoint 缺少安全清单。")
        info = json.loads(latest.read_text(encoding="utf-8"))
        filename = info.get("file") if isinstance(info, dict) else None
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.startswith("checkpoint-"):
            raise ValueError("续跑 checkpoint 路径无效。")
        source_state, _ = worker.load_verified_checkpoint_payload(directory / filename)
        continuation = source_state.get("export_continuation")
        if not isinstance(continuation, dict) or continuation.get("base_manifest_sha256") != bundle_digest or continuation.get("base_checkpoint_sha256") != checkpoint_info["sha256"]:
            raise ValueError("续跑 checkpoint 不属于此导出包。")
        if source_state.get("run_signature") != training_state.get("run_signature") or source_state.get("source_manifest") != training_state.get("source_manifest") or source_state.get("runtime") != training_state.get("runtime"):
            raise ValueError("续跑 checkpoint 的训练签名、源码或运行时不匹配。")
        if type(source_state.get("completed_steps")) is not int or source_state["completed_steps"] < training_state["completed_steps"]:
            raise ValueError("续跑 checkpoint 的步数无效。")
    destination = None
    if checkpoint_directory is not None:
        destination = Path(checkpoint_directory).expanduser().absolute()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("续跑输出目录已存在；不会覆盖。")
        if not destination.parent.is_dir():
            raise ValueError("续跑输出目录的父目录不存在。")
    signature = source_state.get("run_signature")
    if source_state.get("training_mode") != "managed-step" or not isinstance(signature, dict):
        raise ValueError("只支持带完整签名的 managed-step checkpoint。")
    if signature.get("training_mode") != "managed-step" or signature.get("apply_supervision") is not False or signature.get("train_augment") != []:
        raise ValueError("此接口不支持监督规则、数据增强或缺少完整签名的运行。")
    if not isinstance(source_state.get("runtime"), dict) or source_state["runtime"].get("device") != "cpu":
        raise ValueError("此接口仅支持原运行设备为 CPU 的 checkpoint。")
    expected_python = str(source_state["runtime"].get("python_version", "")).split(".")[:2]
    if expected_python != [str(sys.version_info.major), str(sys.version_info.minor)]:
        raise ValueError("Python 主次版本与 checkpoint 不一致。")
    if not isinstance(source_state.get("engine_state"), dict) or not isinstance(source_state["engine_state"].get("optimizer_state"), dict):
        raise ValueError("checkpoint 缺少可恢复的 AdamW 状态。")
    for key in ("batch", "batch_params", "training_step", "training_params", "training_mode", "lr", "weight_decay"):
        if key not in inputs or contains_redacted(inputs[key]) or inputs[key] != signature.get(key):
            raise ValueError(f"导出输入 {key} 缺失、脱敏或与 checkpoint 不一致。")
    if not isinstance(signature.get("batch"), str) or not isinstance(signature.get("training_step"), str):
        raise ValueError("checkpoint 缺少批次或训练步骤入口。")
    _managed_resume_used = True
    if source_state is not training_state:
        model.load_state_dict(source_state["model_state"], strict=True)
        if worker.model_state_digest(model) != source_state.get("model_state_sha256"):
            raise ValueError("续跑 checkpoint 的模型状态指纹不匹配。")
    batch_fn = worker.entrypoint(signature["batch"])
    step_fn = worker.entrypoint(signature["training_step"])
    worker.restore_callable_checkpoint(batch_fn, source_state.get("batch_state"), "batch 工厂", False)
    worker.restore_callable_checkpoint(step_fn, source_state.get("step_state"), "training_step", False)
    worker.restore_rng_state(source_state.get("rng_state"))
    request = {
        "operation": "train", "training_mode": "managed-step", "optimizer": "adamw",
        "batch": signature["batch"], "batch_params": signature["batch_params"],
        "training_step": signature["training_step"], "training_params": signature["training_params"],
        "steps": source_state["completed_steps"] + additional_steps,
        "lr": signature["lr"], "weight_decay": signature["weight_decay"],
        "seed": signature.get("seed"), "train_augment": [], "apply_supervision": False,
        "_resolved_device": "cpu",
    }
    final_engine = {}
    def capture(completed, engine):
        if completed == request["steps"]:
            final_engine["value"] = engine
        return False
    result = worker.run_managed_steps(model, request, restore=source_state,
                                      checkpoint_callback=capture if destination is not None else None)
    if destination is not None:
        if "value" not in final_engine:
            raise RuntimeError("训练未报告最终状态；没有写入续跑 checkpoint。")
        next_state = {
            "schema_version": 1, "training_mode": "managed-step", "completed_steps": request["steps"],
            "run_signature": signature, "source_manifest": source_state.get("source_manifest"),
            "runtime": source_state.get("runtime"), "environment_inventory": source_state.get("environment_inventory"),
            "model_state": model.state_dict(), "model_state_sha256": worker.model_state_digest(model),
            "model_state_alias_groups": worker.model_state_alias_groups(model), "engine_state": final_engine["value"],
            "batch_state": worker.callable_checkpoint_hooks(batch_fn, "batch 工厂"),
            "step_state": worker.callable_checkpoint_hooks(step_fn, "training_step"),
            "loop_state": None, "rng_state": worker.checkpoint_rng_state(),
            "export_continuation": {"base_manifest_sha256": bundle_digest, "base_checkpoint_sha256": checkpoint_info["sha256"]},
        }
        destination.mkdir(mode=0o700)
        saved = worker.save_training_checkpoint(destination, next_state)
        result["continuation_checkpoint"] = {"directory": str(destination), **saved}
    return result

print(f"模型已恢复：{manifest['run']['id']} · step {checkpoint_info['completed_steps']} · {state_hash}")
print("模型位于变量 model，经过哈希验证的训练状态位于 load_training_state()。")
"#
}

fn export_readme(manifest: &Value) -> String {
    let run_id = manifest["run"]["id"].as_str().unwrap_or("unknown");
    let completed = manifest["checkpoint"]["completed_steps"]
        .as_u64()
        .unwrap_or_default();
    let python = manifest["runtime"]["python_version"]
        .as_str()
        .unwrap_or("unknown");
    let torch = manifest["runtime"]["pytorch_version"]
        .as_str()
        .unwrap_or("unknown");
    let environment_status = manifest["environment"]["status"]
        .as_str()
        .unwrap_or("unavailable");
    let capture_timing = manifest["environment"]["capture_timing"]
        .as_str()
        .unwrap_or("unknown");
    format!(
        "# Model-Maven training export\n\nRun: `{run_id}`  \nCheckpoint: step `{completed}`  \nRuntime recorded: Python `{python}`, PyTorch `{torch}`  \nEnvironment inventory: `{environment_status}`\n\n## Restore the trained model\n\nUse a compatible Python/PyTorch environment and review `manifest.json` → `runtime_requirements`; the restore script checks those exact external runtime versions before loading the model. Registered extensions and their installed Python distribution dependencies are included under `vendor/` and loaded from the export. Then run:\n\n```sh\npython restore_model.py\n```\n\nThe script verifies copied project sources, bundled package files, external runtime versions, and the checkpoint before reconstructing the model. It exposes the restored model as `model` and the verified `weights_only` checkpoint payload through `load_training_state()`, including completed steps and any saved optimizer, RNG, batch, step, or loop state. For CPU managed-step AdamW runs without supervision or augmentation, `resume_managed_steps(n)` restores the recorded callable/optimizer/RNG state and executes up to 1000 further steps once per process. Pass checkpoint_directory to save a new verified checkpoint into a new directory; in a fresh process, pass from_checkpoint to continue from that directory. Existing destinations are never overwritten. A failed save can leave an incomplete new directory. A failed call may partially modify memory; reload the module before retrying. Other modes require a caller-owned recovery routine. External data and environment must be made available separately; exact continuation is not guaranteed for arbitrary user code.\n\nTo use the values from another Python script in this export directory:\n\n```python\nimport restore_model\nmodel = restore_model.model\ntraining_state = restore_model.load_training_state()\n# For a supported managed CPU run, save a continuation in a new directory:\nresult = restore_model.resume_managed_steps(1, checkpoint_directory='continued-1')\n# In a fresh process, continue that checkpoint into another new directory:\n# result = restore_model.resume_managed_steps(1, from_checkpoint='continued-1', checkpoint_directory='continued-2')\n```\n\n## Included files\n\n- `source/`: Python files observed by the selected run and registered local component manifests, copied with SHA-256 verification.\n- `vendor/packages/`: exact installed files for registered extension distributions and their Python dependency closure, copied with SHA-256 verification. Each package records its role and dependency sources in `manifest.json`.\n- `artifacts/model-maven-checkpoint.pt`: latest atomic training checkpoint.\n- `environment.json`: selected interpreter details and installed distribution inventory, with package source categories and known omissions.\n- `requirements.snapshot.txt`: sorted name/version snapshot for installed distributions; review before using it to create another environment.\n- `variant.json`: redacted, replayable structural variant when the run used one.\n- `manifest.json`: run provenance, redacted inputs, runtime, external runtime requirements, environment snapshot hashes, and file hashes.\n- `model_maven_worker.py`: versioned Model-Maven model/variant reconstruction helpers.\n\nDatasets, external model weights, private index configuration, operating-system shared libraries, accelerator drivers, and non-Python project files are not copied. Review the bundle before sharing: run inputs are redacted, but the checkpoint can contain unredacted training parameters, callable state, or sample-derived information, and source code may contain credentials. Native Python extensions may still rely on ABI-compatible system libraries.\n"
    )
    .replace(
        "## Restore the trained model",
        &format!(
            "{} Python and PyTorch versions are compared with the run record; check `environment.json` for capture time and comparison details.\n\n## Restore the trained model",
            match capture_timing {
                "training_start" => "The environment inventory was captured at the start of training.",
                "export_time" => "The environment inventory was captured at export time, after the recorded run; other installed packages may have changed since training.",
                _ => "The environment inventory capture time is unavailable or comes from an older run record.",
            }
        ),
    )
}

fn normalize_path(path: &Path) -> Result<PathBuf, String> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err("导出路径越过文件系统根目录。".into());
                }
            }
            Component::Normal(value) => normalized.push(value),
        }
    }
    if !normalized.is_absolute() {
        return Err("导出路径必须能够解析为绝对路径。".into());
    }
    Ok(normalized)
}

fn resolve_future_parent(parent: &Path) -> Result<PathBuf, String> {
    let mut ancestor = parent.to_owned();
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        let component = ancestor.file_name().ok_or("无法解析导出目录的父路径。")?;
        if !matches!(
            Path::new(component).components().next(),
            Some(Component::Normal(_))
        ) {
            return Err("导出父目录包含无效路径项。".into());
        }
        suffix.push(component.to_owned());
        ancestor = ancestor
            .parent()
            .ok_or("无法解析导出目录的父路径。")?
            .to_owned();
    }
    let mut resolved = ancestor.canonicalize().map_err(|error| error.to_string())?;
    for component in suffix.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn resolve_destination(root: &Path, destination: &Path) -> Result<(PathBuf, PathBuf), String> {
    let requested = normalize_path(
        if destination.is_absolute() {
            destination.to_owned()
        } else {
            root.join(destination)
        }
        .as_path(),
    )?;
    let name = requested
        .file_name()
        .ok_or("导出目录必须是项目外的明确目录名。")?;
    if !matches!(
        Path::new(name).components().next(),
        Some(Component::Normal(_))
    ) || Path::new(name).components().count() != 1
    {
        return Err("导出目录名称不能是 .、.. 或包含路径分隔的特殊项。".into());
    }
    let parent = requested
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(root);
    let future_parent = resolve_future_parent(parent)?;
    let future_target = future_parent.join(name);
    if future_target.starts_with(root) || root.starts_with(&future_target) {
        return Err("为保护工作区文件，导出目录必须位于项目目录之外。".into());
    }
    if fs::symlink_metadata(&future_target).is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        return Err("拒绝覆盖符号链接导出目标。".into());
    }
    if fs::symlink_metadata(&future_target).is_ok_and(|metadata| !metadata.is_dir()) {
        return Err("导出目标已存在且不是目录。".into());
    }
    Ok((future_parent, future_target))
}

impl ArtifactExporter for FileArtifactExporter {
    fn export_training_run(&self, request: TrainingExportRequest<'_>) -> Result<Value, String> {
        let TrainingExportRequest {
            project_root,
            destination,
            run,
            checkpoint,
            environment,
            extension_packages: extension_package_plans,
            runtime_requirements,
            overwrite,
        } = request;
        let root = project_root
            .canonicalize()
            .map_err(|error| error.to_string())?;
        let (mut parent, mut target) = resolve_destination(&root, destination)?;
        if target.exists() && !overwrite {
            return Err(format!(
                "导出目标已存在：{}；如确认替换，请使用 --overwrite。",
                target.display()
            ));
        }
        if run["record"]["operation"] != "train" || run["inputs"]["operation"] != "train" {
            return Err("只能导出正式训练运行；try 试跑不含模型 checkpoint。".into());
        }
        let report_source_manifest = run["report"]["data"]["source_manifest"]
            .as_object()
            .ok_or("运行记录没有可导出的源码指纹。")?;
        if report_source_manifest.is_empty() {
            return Err("运行记录的源码清单为空，无法生成可恢复工程。".into());
        }
        let mut source_manifest = report_source_manifest.clone();
        if let Some(variant_manifest) = run["inputs"]["variant"]["source_manifest"].as_object() {
            for (path, digest) in variant_manifest {
                let path = Path::new(path);
                if path.starts_with(root.join(".model-maven/components"))
                    && path.extension().and_then(|value| value.to_str()) == Some("json")
                {
                    source_manifest.insert(path.to_string_lossy().into_owned(), digest.clone());
                }
            }
        }
        add_component_sources(&root, run, &mut source_manifest)?;
        let checkpoint_manifest = run["checkpoint"]
            .as_object()
            .ok_or("该运行没有可用 checkpoint。")?;
        let expected_checkpoint_hash = checkpoint_manifest["sha256"]
            .as_str()
            .ok_or("checkpoint 清单缺少 SHA-256。")?;
        if checkpoint_manifest["file"].as_str()
            != checkpoint.file_name().and_then(|name| name.to_str())
        {
            return Err("checkpoint 文件与运行清单不一致。".into());
        }
        if sha256(checkpoint)? != expected_checkpoint_hash {
            return Err("checkpoint SHA-256 校验失败，拒绝导出。".into());
        }

        let mut verified_sources = Vec::with_capacity(source_manifest.len());
        for (source_name, expected) in &source_manifest {
            let source = Path::new(source_name);
            let relative = relative_source_path(&root, source)?;
            let expected_hash = expected
                .as_str()
                .ok_or("源码清单中包含无效 SHA-256。")?
                .to_owned();
            if sha256(source)? != expected_hash {
                return Err(format!(
                    "运行源码已变化：{}；为保持可复现性，拒绝导出。",
                    source.display()
                ));
            }
            verified_sources.push((source.to_owned(), relative, expected_hash));
        }
        let verified_extensions = verify_extension_package_plans(extension_package_plans)?;
        verify_runtime_requirements(runtime_requirements)?;

        fs::create_dir_all(&parent).map_err(|error| error.to_string())?;
        parent = parent.canonicalize().map_err(|error| error.to_string())?;
        let target_name = target.file_name().ok_or("导出目标名称无效。")?.to_owned();
        target = parent.join(target_name);
        if target.starts_with(&root) || root.starts_with(&target) {
            return Err("为保护工作区文件，导出目录必须位于项目目录之外。".into());
        }
        if fs::symlink_metadata(&target).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return Err("拒绝覆盖符号链接导出目标。".into());
        }
        if fs::symlink_metadata(&target).is_ok_and(|metadata| !metadata.is_dir()) {
            return Err("导出目标已存在且不是目录。".into());
        }
        if target.exists() && !overwrite {
            return Err(format!(
                "导出目标已存在：{}；如确认替换，请使用 --overwrite。",
                target.display()
            ));
        }

        let staging = Builder::new()
            .prefix(".m-model-export-")
            .tempdir_in(&parent)
            .map_err(|error| error.to_string())?;
        let stage = staging.path();
        let source_dir = stage.join("source");
        let artifact_dir = stage.join("artifacts");
        fs::create_dir(&source_dir).map_err(|error| error.to_string())?;
        fs::create_dir(&artifact_dir).map_err(|error| error.to_string())?;

        let mut files = Map::new();
        let mut copied_bytes = 0u64;
        for (source, relative, expected_hash) in verified_sources {
            let output = source_dir.join(&relative);
            if let Some(directory) = output.parent() {
                fs::create_dir_all(directory).map_err(|error| error.to_string())?;
            }
            fs::copy(source, &output).map_err(|error| error.to_string())?;
            if sha256(&output)? != expected_hash {
                return Err(format!("复制后的源码校验失败：{}", relative.display()));
            }
            copied_bytes = copied_bytes.saturating_add(
                fs::metadata(&output)
                    .map_err(|error| error.to_string())?
                    .len(),
            );
            files.insert(
                relative.to_string_lossy().replace('\\', "/"),
                json!(expected_hash),
            );
        }

        let mut extension_package_records = Vec::with_capacity(verified_extensions.len());
        let mut extension_file_count = 0usize;
        let mut extension_bytes = 0u64;
        for package in &verified_extensions {
            let bundle_relative = relative_bundle_path(
                package
                    .bundle_path
                    .to_str()
                    .ok_or("扩展包归档路径不是有效文本。")?,
            )?;
            let mut package_files = Map::new();
            for file in &package.files {
                let output = stage.join(&bundle_relative).join(&file.relative);
                if let Some(directory) = output.parent() {
                    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
                }
                fs::copy(&file.source, &output).map_err(|error| error.to_string())?;
                if sha256(&output)? != file.sha256
                    || fs::metadata(&output)
                        .map_err(|error| error.to_string())?
                        .len()
                        != file.size_bytes
                {
                    return Err(format!(
                        "复制后的扩展包文件校验失败：{}/{}",
                        package.distribution,
                        file.relative.display()
                    ));
                }
                copied_bytes = copied_bytes.saturating_add(file.size_bytes);
                extension_bytes = extension_bytes.saturating_add(file.size_bytes);
                extension_file_count += 1;
                package_files.insert(
                    file.relative.to_string_lossy().replace('\\', "/"),
                    json!(file.sha256),
                );
            }
            extension_package_records.push(json!({
                "distribution":package.distribution,
                "version":package.version,
                "role":package.role,
                "components":package.components,
                "required_by":package.required_by,
                "bundle_path":bundle_relative.to_string_lossy().replace('\\', "/"),
                "file_count":package.files.len(),
                "size_bytes":package.size_bytes,
                "files":package_files,
                "omitted_files":package.omitted_files
            }));
        }

        let checkpoint_output = artifact_dir.join("model-maven-checkpoint.pt");
        fs::copy(checkpoint, &checkpoint_output).map_err(|error| error.to_string())?;
        if sha256(&checkpoint_output)? != expected_checkpoint_hash {
            return Err("复制后的 checkpoint 校验失败。".into());
        }
        copied_bytes = copied_bytes.saturating_add(
            fs::metadata(&checkpoint_output)
                .map_err(|error| error.to_string())?
                .len(),
        );
        stage_worker_package(stage, true).map_err(|error| error.to_string())?;
        write_file(
            &stage.join("restore_model.py"),
            python_restore_script().as_bytes(),
        )?;

        let report = run.get("report").unwrap_or(&Value::Null);
        let runtime = report.get("data").and_then(|data| data.get("runtime"));
        let recorded_python = runtime.and_then(|value| value["python_version"].as_str());
        let recorded_torch = runtime.and_then(|value| value["pytorch_version"].as_str());
        let mut environment_document = environment.clone();
        let (environment_status, runtime_comparison) =
            environment_runtime_comparison(&environment_document, recorded_python, recorded_torch);
        let requirements_snapshot = environment_document
            .get_mut("requirements_snapshot")
            .and_then(|value| value.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| {
                "# Environment package inventory could not be captured.\n# Use a compatible environment and check manifest.json for details.\n".into()
            });
        if let Some(object) = environment_document.as_object_mut() {
            object.remove("requirements_snapshot");
            object.insert("runtime_comparison".into(), runtime_comparison);
            object
                .entry("capture_timing")
                .or_insert_with(|| json!("export_time"));
            object.insert("capture_status".into(), json!(environment_status.clone()));
        }
        write_json(&stage.join("environment.json"), &environment_document)?;
        write_file(
            &stage.join("requirements.snapshot.txt"),
            requirements_snapshot.as_bytes(),
        )?;
        let environment_sha256 = sha256(&stage.join("environment.json"))?;
        let requirements_sha256 = sha256(&stage.join("requirements.snapshot.txt"))?;

        let inputs = &run["inputs"];
        let projected_inputs = input_projection(inputs);
        let variant = inputs.get("variant").filter(|value| value.is_object());
        if let Some(variant) = variant {
            write_json(&stage.join("variant.json"), variant)?;
        }
        let mut bundle_files = Map::new();
        for relative in [
            "environment.json",
            "requirements.snapshot.txt",
            "model_maven_worker.py",
        ] {
            bundle_files.insert(relative.into(), json!(sha256(&stage.join(relative))?));
        }
        if variant.is_some() {
            bundle_files.insert(
                "variant.json".into(),
                json!(sha256(&stage.join("variant.json"))?),
            );
        }
        append_bundle_file_hashes(stage, &stage.join("model_maven_worker"), &mut bundle_files)?;
        let run_summary = json!({
            "id":run["record"]["id"],
            "status":run["record"]["status"],
            "parent_run_id":run["record"]["parent_run_id"],
            "started_at_ms":run["record"]["started_at_ms"],
            "finished_at_ms":run["record"]["finished_at_ms"],
            "summary":run["record"]["summary"],
            "result":result_projection(report)
        });
        let exported_checkpoint = json!({
            "file":"artifacts/model-maven-checkpoint.pt",
            "sha256":expected_checkpoint_hash,
            "completed_steps":checkpoint_manifest["completed_steps"],
            "training_mode":checkpoint_manifest["training_mode"],
            "model_state_sha256":checkpoint_manifest["model_state_sha256"]
        });
        let source_file_count = files.len();
        let manifest = json!({
            "schema_version":1,
            "product":"Model-Maven",
            "run":run_summary,
            "inputs":projected_inputs,
            "variant_file":variant.is_some(),
            "runtime":{
                "python_version":runtime.and_then(|value| value.get("python_version")),
                "pytorch_version":runtime.and_then(|value| value.get("pytorch_version"))
            },
            "environment":{
                "status":environment_status.clone(),
                "capture_timing":environment_document["capture_timing"],
                "captured_at_utc":environment_document["captured_at_utc"],
                "file":"environment.json",
                "sha256":environment_sha256,
                "requirements_snapshot_file":"requirements.snapshot.txt",
                "requirements_snapshot_sha256":requirements_sha256,
                "distribution_count":environment_document["distribution_count"]
            },
            "source_files":files,
            "bundle_files":bundle_files,
            "extension_packages":extension_package_records,
            "runtime_requirements":runtime_requirements,
            "checkpoint":exported_checkpoint,
            "omitted_inputs":["datasets and non-Python files", "external model weights", "unregistered packages", "private index configuration", "OS-level shared libraries and accelerator drivers", "resolver-generated lockfile"]
        });
        write_json(&stage.join("manifest.json"), &manifest)?;
        write_file(
            &stage.join("README.md"),
            export_readme(&manifest).as_bytes(),
        )?;
        File::open(stage)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| error.to_string())?;

        let backup = if target.exists() {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let base = target
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("导出目标名称不是有效文本。")?;
            let mut candidate =
                parent.join(format!(".{base}.backup-{}-{timestamp}", std::process::id()));
            let mut suffix = 0u32;
            while candidate.exists() {
                suffix = suffix.saturating_add(1);
                candidate = parent.join(format!(
                    ".{base}.backup-{}-{timestamp}-{suffix}",
                    std::process::id()
                ));
            }
            fs::rename(&target, &candidate).map_err(|error| error.to_string())?;
            Some(candidate)
        } else {
            None
        };
        if let Err(error) = fs::rename(stage, &target) {
            if let Some(backup) = &backup {
                let _ = fs::rename(backup, &target);
            }
            return Err(format!("无法提交导出目录：{error}"));
        }
        File::open(&parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| error.to_string())?;
        drop(staging);

        let file_count =
            source_file_count + extension_file_count + 7 + usize::from(variant.is_some());
        Ok(json!({
            "path":target,
            "run_id":run["record"]["id"],
            "checkpoint_completed_steps":checkpoint_manifest["completed_steps"],
            "source_file_count":source_file_count,
            "extension_package_count":verified_extensions.iter().filter(|package| package.role == "extension").count(),
            "dependency_package_count":verified_extensions.iter().filter(|package| package.role == "runtime_dependency").count(),
            "extension_file_count":extension_file_count,
            "extension_bytes":extension_bytes,
            "file_count":file_count,
            "verified_bytes":copied_bytes,
            "environment_snapshot_status":environment_status,
            "environment_snapshot_timing":environment_document["capture_timing"],
            "environment_distribution_count":environment_document["distribution_count"],
            "checkpoint_sha256":expected_checkpoint_hash,
            "backup_path":backup
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::environment_runtime_comparison;
    use serde_json::{Value, json};

    #[test]
    fn environment_capture_requires_matching_recorded_python_and_torch() {
        let environment = json!({
            "schema_version":1,
            "python":{"version":"3.12.1"},
            "pytorch_version":"2.6.0"
        });
        let (status, comparison) =
            environment_runtime_comparison(&environment, Some("3.12.1"), Some("2.6.0"));
        assert_eq!(status, "captured");
        assert_eq!(comparison["python_version"]["status"], "matches");
        assert_eq!(comparison["pytorch_version"]["status"], "matches");

        let (status, comparison) =
            environment_runtime_comparison(&environment, Some("3.12.1"), Some("2.5.0"));
        assert_eq!(status, "runtime_mismatch");
        assert_eq!(comparison["python_version"]["status"], "matches");
        assert_eq!(
            comparison["pytorch_version"]["status"],
            "differs_or_unknown"
        );
    }

    #[test]
    fn unavailable_environment_keeps_its_failure_status() {
        let environment = json!({
            "status":"unavailable",
            "diagnostics":["selected interpreter exited"]
        });
        let (status, comparison) = environment_runtime_comparison(&environment, None, None);
        assert_eq!(status, "unavailable");
        assert_eq!(comparison["python_version"]["snapshot"], Value::Null);
    }
}
