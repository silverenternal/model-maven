use crate::ports::ComponentScaffolder;
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
};

pub struct FileComponentScaffolder;

impl ComponentScaffolder for FileComponentScaffolder {
    fn scaffold(
        &self,
        project_root: &Path,
        component_id: &str,
        kind: &str,
        version: &str,
        output: &Path,
    ) -> Result<Value, String> {
        let (namespace, name) = component_id
            .split_once('/')
            .ok_or("组件 ID 必须使用 namespace/name 格式。")?;
        let valid_segment = |value: &str| {
            !value.is_empty()
                && value.len() <= 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        };
        if !valid_segment(namespace) || !valid_segment(name) || name.contains('/') {
            return Err(
                "组件 ID 只能包含一个斜线，namespace/name 两段各限字母、数字、-、_。".into(),
            );
        }
        if !matches!(kind, "layer" | "layer-edit-provider") {
            return Err("扩展类型必须是 layer 或 layer-edit-provider。".into());
        }
        semver::Version::parse(version)
            .map_err(|_| "组件版本必须使用 MAJOR.MINOR.PATCH SemVer 格式。")?;
        let relative = safe_relative_output(output)?;
        let root = project_root
            .canonicalize()
            .map_err(|error| format!("无法解析项目目录：{error}"))?;
        let target = ensure_output_parent(&root, &relative)?;
        fs::create_dir(&target).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                format!("输出目录已存在，未覆盖：{}", target.display())
            } else {
                format!("无法创建输出目录 {}：{error}", target.display())
            }
        })?;

        let module = format!(
            "model_maven_{}_{}",
            python_identifier(namespace),
            python_identifier(name)
        );
        let package_name = format!("model-maven-{namespace}-{name}").replace('_', "-");
        let entry_name = name.replace('_', "-");
        let (description, entrypoint, params_schema, capabilities, fixture, python_source) = if kind
            == "layer"
        {
            (
                    format!("A PyTorch Layer component for {component_id}.") ,
                    format!("{module}.layer:make_layer"),
                    json!({"type":"object","properties":{"inplace":{"type":"boolean","default":false,"description":"Whether the Layer may modify its input in place."}},"required":[],"additionalProperties":false}),
                    json!({"construct":{"state":"supported"}}),
                    None,
                    "\"\"\"Layer factory entry point for the Model-Maven component scaffold.\"\"\"\n\nfrom torch import nn\n\n\ndef make_layer(inplace: bool = False) -> nn.Module:\n    \"\"\"Build a SiLU layer; replace this body with your own module.\"\"\"\n    return nn.SiLU(inplace=inplace)\n".to_owned(),
                )
        } else {
            (
                format!("A PyTorch Layer Edit Provider for {component_id}."),
                format!("{module}.provider:NamedBranchProvider"),
                json!({"type":"object","properties":{"target":{"type":"string","default":"encoder","description":"Dot-separated model path to a Sequential module."},"name":{"type":"string","default":"silu","description":"Name for the appended activation."}},"required":[],"additionalProperties":false}),
                json!({"layer_edit":{"state":"supported"}}),
                Some(format!("{module}.provider:provider_fixture")),
                provider_source(),
            )
        };

        let mut manifest = json!({
            "schema_version":1,
            "namespace":namespace,
            "component_id":component_id,
            "component_version":version,
            "extension_api_version":1,
            "kind":kind,
            "entrypoint":entrypoint,
            "description":description,
            "params_schema":params_schema,
            "capabilities":capabilities
        });
        if let Some(fixture) = fixture {
            manifest["conformance_fixture"] = Value::String(fixture);
        }
        let manifest_text = serde_json::to_string_pretty(&manifest)
            .map_err(|error| format!("无法生成组件清单：{error}"))?;
        let pyproject = format!(
            "[build-system]\nrequires = [\"setuptools>=68\", \"wheel\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"{package_name}\"\nversion = \"{version}\"\ndescription = \"Model-Maven extension component {component_id}\"\nrequires-python = \">=3.10\"\ndependencies = [\"torch>=2.1\"]\n\n[project.entry-points.\"model_maven.components\"]\n\"{entry_name}\" = \"{entrypoint}\"\n\n[tool.setuptools]\npackage-dir = {{\"\" = \"src\"}}\n\n[tool.setuptools.packages.find]\nwhere = [\"src\"]\n\n[tool.setuptools.package-data]\n\"{module}\" = [\"__model_maven__/*.json\"]\n"
        );
        let (source_file, manifest_file) = if kind == "layer" {
            ("layer.py", format!("{entry_name}.json"))
        } else {
            ("provider.py", format!("{entry_name}.json"))
        };
        let source_relative = PathBuf::from("src").join(&module).join(source_file);
        let manifest_relative = PathBuf::from("src")
            .join(&module)
            .join("__model_maven__")
            .join(manifest_file);
        let files = [
            (PathBuf::from("pyproject.toml"), pyproject),
            (
                PathBuf::from("README.md"),
                package_readme(component_id, kind, &entrypoint),
            ),
            (
                PathBuf::from(".gitignore"),
                "__pycache__/\n*.egg-info/\ndist/\nbuild/\n.venv/\n".into(),
            ),
            (
                PathBuf::from("src").join(&module).join("__init__.py"),
                "\"\"\"Model-Maven extension package.\"\"\"\n".into(),
            ),
            (source_relative, python_source),
            (manifest_relative, format!("{manifest_text}\n")),
        ];

        let result = (|| {
            for (relative_path, contents) in &files {
                let path = target.join(relative_path);
                let parent = path.parent().expect("generated file has a parent");
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(|error| format!("无法创建 {}：{error}", path.display()))?;
                file.write_all(contents.as_bytes())
                    .map_err(|error| format!("无法写入 {}：{error}", path.display()))?;
                file.sync_all()
                    .map_err(|error| format!("无法同步 {}：{error}", path.display()))?;
            }
            Ok(json!({
                "component_id":component_id,
                "version":version,
                "kind":kind,
                "directory":target,
                "entrypoint":entrypoint,
                "manifest":manifest,
                "next_steps":[
                "在生成包目录中实现或调整 Layer/Provider，并保持 manifest Schema 与入口参数一致。",
                "从模型项目根目录运行 m-model component install <包目录>，构建并登记本地扩展。",
                "运行 m-model component test <namespace/name> 检查组件。"
                ]
            }))
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&target);
            return result;
        }
        result
    }
}

fn safe_relative_output(output: &Path) -> Result<PathBuf, String> {
    if output.as_os_str().is_empty() || output.is_absolute() {
        return Err("--output 必须是项目内的相对目录。".into());
    }
    let mut normalized = PathBuf::new();
    for component in output.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            _ => return Err("--output 不接受 .、..、根路径或平台前缀。".into()),
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err("--output 必须指定一个新目录。".into());
    }
    Ok(normalized)
}

fn ensure_output_parent(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    let mut current = root.to_path_buf();
    let parts = relative.components().collect::<Vec<_>>();
    for part in &parts[..parts.len() - 1] {
        let Component::Normal(name) = part else {
            unreachable!()
        };
        current.push(name);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "输出路径包含符号链接，拒绝写入：{}",
                    current.display()
                ));
            }
            Ok(metadata) if !metadata.is_dir() => {
                return Err(format!("输出路径父项不是目录：{}", current.display()));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&current)
                    .map_err(|error| format!("无法创建目录 {}：{error}", current.display()))?;
            }
            Err(error) => return Err(format!("无法访问 {}：{error}", current.display())),
        }
    }
    let Component::Normal(name) = parts.last().expect("relative output is non-empty") else {
        unreachable!()
    };
    current.push(name);
    if fs::symlink_metadata(&current).is_ok() {
        return Err(format!("输出目录已存在，未覆盖：{}", current.display()));
    }
    Ok(current)
}

fn python_identifier(value: &str) -> String {
    value.replace('-', "_")
}

fn provider_source() -> String {
    r###"
"""Layer Edit Provider scaffold. Replace the sample operation with your editor."""

from collections import OrderedDict

import torch
from torch import nn

MODEL_MAVEN_LAYER_EDIT_API = 2


def _resolve(model: nn.Module, path: str) -> nn.Module:
    normalized = path.removeprefix("model.")
    if normalized == "model" or not normalized:
        return model
    return model.get_submodule(normalized)


def _parent(model: nn.Module, path: str):
    normalized = path if path.startswith("model.") else f"model.{path}"
    parent_path, _, name = normalized.rpartition(".")
    parent = _resolve(model, parent_path)
    if name not in parent._modules:
        raise ValueError(f"{normalized} is not a registered child module")
    return normalized, parent, name


class NamedBranchProvider:
    """Append a Layer under a named Sequential branch with replayable intent."""

    def __init__(self):
        self._previous = None

    def describe(self, model: nn.Module, operation: dict) -> dict:
        return {"name": "named-sequential-layer-insertion", "targets": "registered module paths"}

    def can_edit(self, model: nn.Module, operation: dict) -> dict:
        payload = operation.get("payload", {})
        target, name = payload.get("target"), payload.get("name")
        if not isinstance(target, str) or not isinstance(name, str) or not name.isidentifier():
            return {"state": "unsupported", "reason": "payload requires a named Sequential path and identifier"}
        try:
            _, parent, child = _parent(model, target)
            if not isinstance(parent._modules[child], nn.Sequential):
                return {"state": "unsupported", "reason": f"{target} must be nn.Sequential"}
            if name in parent._modules[child]._modules:
                return {"state": "unsupported", "reason": f"{target}.{name} already exists"}
        except (AttributeError, KeyError, ValueError) as error:
            return {"state": "unsupported", "reason": str(error)}
        return {"state": "supported", "reason": "named branch and new Layer name are available"}

    def plan(self, model: nn.Module, operation: dict) -> dict:
        target, name = operation["payload"]["target"], operation["payload"]["name"]
        normalized, _, _ = _parent(model, target)
        return {"target": normalized, "name": name, "layer": "torch.nn.SiLU"}

    def candidate(self, model: nn.Module, plan: dict) -> dict:
        _, parent, name = _parent(model, plan["target"])
        container = parent._modules[name]
        if not isinstance(container, nn.Sequential):
            raise TypeError(f"{plan['target']} must be nn.Sequential")
        self._previous = (parent, name, container)
        children = OrderedDict(container.named_children())
        children[plan["name"]] = nn.SiLU()
        parent._modules[name] = nn.Sequential(children)
        return {"added": f"{plan['target']}.{plan['name']}", "type": "SiLU"}

    def persist(self, plan: dict) -> dict:
        return {"schema_version": 1, **plan}

    def rollback(self, model: nn.Module, plan: dict) -> dict:
        if self._previous is not None:
            parent, name, previous = self._previous
            parent._modules[name] = previous
            self._previous = None
        return {"restored": True, "target": plan["target"]}


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([("projection", nn.Linear(8, 8))]))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)


def provider_fixture() -> dict:
    return {
        "model": FixtureModel(),
        "expected_result": {"added": "model.encoder.silu", "type": "SiLU"},
        "expected_modules": {"encoder.silu": "SiLU"},
    }
"###
    .to_owned()
}

fn package_readme(component_id: &str, kind: &str, entrypoint: &str) -> String {
    format!(
        "# {component_id}\n\nGenerated Model-Maven {kind} extension package.\n\n## Develop\n\nEdit the Python entry point `{entrypoint}` and its static manifest under `src/`. Keep the manifest parameter schema aligned with the callable.\n\n## Build, install, and check\n\nFrom the model project root, run:\n\n```sh\nm-model component install <project-relative-package-directory>\nm-model component test {component_id}\n```\n\nThe installer builds a wheel and copies it into `.model-maven/extensions/` without changing the selected Python environment. Add runtime dependencies to the model project's requirements and run `m-model env setup` first; component installation never installs dependencies. The build backend executes as your user, so install only trusted source packages.\n"
    )
}
