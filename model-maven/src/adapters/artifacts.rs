use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

const MANIFEST_SCHEMA_VERSION: u64 = 2;

pub(super) fn safe_relative_path(value: &str) -> Result<PathBuf, String> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!(
            "managed artifact 路径必须是运行目录内的普通相对路径：{value}"
        ));
    }
    Ok(path.to_path_buf())
}

pub(super) fn artifact_file(run_directory: &Path, relative: &str) -> Result<(u64, String), String> {
    let relative = safe_relative_path(relative)?;
    let target = run_directory.join(&relative);
    let mut parent = run_directory.to_path_buf();
    let parts = relative.components().collect::<Vec<_>>();
    for component in parts.iter().take(parts.len().saturating_sub(1)) {
        let Component::Normal(name) = component else {
            return Err("managed artifact 路径无效。".into());
        };
        parent.push(name);
        let metadata = fs::symlink_metadata(&parent).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(format!(
                "managed artifact 父路径不是可信目录：{}",
                parent.display()
            ));
        }
    }
    let metadata = fs::symlink_metadata(&target).map_err(|error| error.to_string())?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "managed artifact 必须是普通文件：{}",
            relative.display()
        ));
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options.open(&target).map_err(|error| error.to_string())?;
    let opened = file.metadata().map_err(|error| error.to_string())?;
    if !opened.is_file() || opened.len() != metadata.len() {
        return Err(format!(
            "managed artifact 在登记时发生变化：{}",
            relative.display()
        ));
    }
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let after = file.metadata().map_err(|error| error.to_string())?;
    if after.len() != opened.len() {
        return Err(format!(
            "managed artifact 在登记期间发生变化：{}",
            relative.display()
        ));
    }
    Ok((after.len(), format!("{:x}", digest.finalize())))
}

fn validate_label(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.len() > 160 || value.chars().any(char::is_control) {
        return Err(format!(
            "managed artifact 的 {label} 必须是 1 到 160 个可见字符。"
        ));
    }
    Ok(())
}

/// Builds a run-local manifest only from files explicitly declared by a producer.
/// Unknown file types are never inferred as disposable artifacts.
pub(super) fn build_run_manifest(
    run_directory: &Path,
    run_id: &str,
    terminal_status: &str,
    report: &Value,
) -> Result<Value, String> {
    let declarations = report["data"]["managed_artifacts"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut artifacts = Vec::new();
    let mut ids = std::collections::BTreeSet::new();
    let mut paths = std::collections::BTreeSet::new();
    let generic_checkpoint = report["data"]["training_mode"] == "generic-external-loop"
        || run_directory.join("checkpoints/latest.generic.json").exists();
    let latest_path = if generic_checkpoint {
        run_directory.join("checkpoints/latest.generic.json")
    } else {
        run_directory.join("checkpoints/latest.json")
    };
    let latest = match fs::symlink_metadata(&latest_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.to_string()),
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            return Err("latest checkpoint 清单不是普通文件；拒绝登记产物。".into());
        }
        Ok(_) => Some(
            serde_json::from_slice::<Value>(
                &fs::read(&latest_path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("latest checkpoint 清单无效：{error}"))?,
        ),
    };
    if let Some(latest) = latest {
        let filename = latest["file"]
            .as_str()
            .ok_or("latest checkpoint 清单缺少文件名。")?;
        let relative = format!("checkpoints/{filename}");
        let (size_bytes, sha256) = artifact_file(run_directory, &relative)?;
        if latest["sha256"].as_str() != Some(sha256.as_str()) {
            return Err("latest checkpoint 的内容摘要与文件不匹配；拒绝登记产物。".into());
        }
        ids.insert("training-checkpoint".to_owned());
        paths.insert(relative.clone());
        artifacts.push(json!({
            "id":"training-checkpoint",
            "owner":if generic_checkpoint {"model_maven_worker.train.generic_checkpoint"} else {"model_maven_worker.train.checkpoint"},
            "kind":"checkpoint",
            "complete":true,
            "declared_complete":true,
            "availability":"present",
            "references":["latest-checkpoint"],
            "consumes":report_consumes(report, run_id),
            "files":[{"path":relative,"size_bytes":size_bytes,"sha256":sha256,"available":true}]
        }));
    }
    for declaration in declarations {
        let owner = declaration["owner"]
            .as_str()
            .ok_or("managed artifact 缺少 owner；未登记任何产物。")?;
        let artifact_id = declaration["id"]
            .as_str()
            .ok_or("managed artifact 缺少 id；未登记任何产物。")?;
        validate_label(owner, "owner")?;
        validate_label(artifact_id, "id")?;
        if !ids.insert(artifact_id.to_owned()) {
            return Err(format!("managed artifact id 重复：{artifact_id}"));
        }
        let declared_files = declaration["files"]
            .as_array()
            .ok_or_else(|| format!("managed artifact {artifact_id} 缺少 files 数组。"))?;
        if declared_files.is_empty() {
            return Err(format!(
                "managed artifact {artifact_id} 必须登记至少一个文件。"
            ));
        }
        let references = declaration["references"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut reference_ids = Vec::new();
        for reference in references {
            let value = reference
                .as_str()
                .ok_or_else(|| format!("managed artifact {artifact_id} 的引用必须是字符串。"))?;
            validate_label(value, "reference")?;
            reference_ids.push(value.to_owned());
        }
        reference_ids.sort();
        reference_ids.dedup();
        let consumes = match declaration.get("consumes") {
            None => Vec::new(),
            Some(value) => value.as_array().cloned().ok_or_else(|| {
                format!("managed artifact {artifact_id} 的 consumes 必须是数组。")
            })?,
        };
        let mut typed_consumes = Vec::new();
        for consumed in consumes {
            let producer_run_id = consumed["run_id"].as_str().ok_or_else(|| {
                format!("managed artifact {artifact_id} 的 consumes 缺少 run_id。")
            })?;
            let producer_artifact_id = consumed["artifact_id"].as_str().ok_or_else(|| {
                format!("managed artifact {artifact_id} 的 consumes 缺少 artifact_id。")
            })?;
            validate_label(producer_run_id, "consumed run_id")?;
            validate_label(producer_artifact_id, "consumed artifact_id")?;
            if producer_run_id == run_id {
                return Err(format!(
                    "managed artifact {artifact_id} 不能消费自身运行的产物。"
                ));
            }
            typed_consumes
                .push(json!({"run_id":producer_run_id,"artifact_id":producer_artifact_id}));
        }
        typed_consumes.sort_by(|left, right| {
            (left["run_id"].as_str(), left["artifact_id"].as_str())
                .cmp(&(right["run_id"].as_str(), right["artifact_id"].as_str()))
        });
        typed_consumes.dedup();
        let declared_complete = declaration["complete"].as_bool().unwrap_or(false);
        let complete = declared_complete && terminal_status == "completed";
        let kind = declaration["kind"].as_str().unwrap_or("other");
        validate_label(kind, "kind")?;
        if !matches!(kind, "checkpoint" | "other") {
            return Err(format!(
                "managed artifact {artifact_id} 的 kind 必须是 checkpoint 或 other。"
            ));
        }
        let mut files = Vec::new();
        for file in declared_files {
            let relative = file.as_str().ok_or_else(|| {
                format!("managed artifact {artifact_id} 的文件路径必须是字符串。")
            })?;
            let path = safe_relative_path(relative)?;
            let normalized = path.to_string_lossy().into_owned();
            if !normalized.starts_with("artifacts/") {
                return Err(format!(
                    "自定义 managed artifact 只能登记 artifacts/ 下的文件：{normalized}"
                ));
            }
            if !paths.insert(normalized.clone()) {
                return Err(format!(
                    "同一运行内重复登记 managed artifact 文件：{normalized}"
                ));
            }
            let (size_bytes, sha256) = artifact_file(run_directory, &normalized)?;
            files.push(json!({
                "path":normalized,
                "size_bytes":size_bytes,
                "sha256":sha256,
                "available":true
            }));
        }
        artifacts.push(json!({
            "id":artifact_id,
            "owner":owner,
            "kind":kind,
            "complete":complete,
            "declared_complete":declared_complete,
            "availability":"present",
            "references":reference_ids,
            "consumes":typed_consumes,
            "files":files
        }));
    }
    artifacts.sort_by(|left, right| left["id"].as_str().cmp(&right["id"].as_str()));
    Ok(json!({
        "schema_version":MANIFEST_SCHEMA_VERSION,
        "run_id":run_id,
        "status":"complete",
        "recorded_at_ms":super::runs::FileRunStore::now_ms_for_artifact_manifest(),
        "artifacts":artifacts
    }))
}

fn report_consumes(report: &Value, run_id: &str) -> Vec<Value> {
    let parent = report["data"]["parent_run_id"]
        .as_str()
        .or_else(|| report["data"]["repeat_of"].as_str());
    parent
        .filter(|parent| *parent != run_id)
        .map(|parent| vec![json!({"run_id":parent,"artifact_id":"training-checkpoint"})])
        .unwrap_or_default()
}

pub(super) fn plan_managed_checkpoint_releases(
    run_directory: &Path,
    run_id: &str,
    selected_artifact: Option<&str>,
    project_consumers: &std::collections::BTreeMap<(String, String), Vec<String>>,
) -> (Vec<Value>, Vec<Value>) {
    let path = run_directory.join("artifacts.json");
    let manifest = match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return (Vec::new(), Vec::new());
        }
        Err(error) => {
            return (
                Vec::new(),
                vec![json!({"run_id":run_id,"reason":error.to_string()})],
            );
        }
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            return (
                Vec::new(),
                vec![
                    json!({"run_id":run_id,"reason":"产物清单不安全；自定义 checkpoint 保持保护"}),
                ],
            );
        }
        Ok(_) => match fs::read(&path)
            .map_err(|error| error.to_string())
            .and_then(|bytes| {
                serde_json::from_slice::<Value>(&bytes).map_err(|error| error.to_string())
            }) {
            Ok(manifest) if validate_manifest(&manifest, run_id).is_ok() => manifest,
            Ok(_) => {
                return (
                    Vec::new(),
                    vec![
                        json!({"run_id":run_id,"reason":"产物清单身份无效；自定义 checkpoint 保持保护"}),
                    ],
                );
            }
            Err(error) => {
                return (
                    Vec::new(),
                    vec![json!({"run_id":run_id,"reason":format!("产物清单无法读取：{error}")} )],
                );
            }
        },
    };
    let mut items = Vec::new();
    let mut skipped = Vec::new();
    for artifact in manifest["artifacts"].as_array().into_iter().flatten() {
        if selected_artifact.is_some_and(|selected| artifact["id"] != selected) {
            continue;
        }
        let id = artifact["id"].as_str().unwrap_or("unknown");
        if artifact["kind"] != "checkpoint" {
            if selected_artifact.is_some() {
                skipped.push(json!({"run_id":run_id,"artifact_id":id,"reason":"指定产物不是 checkpoint，保持保护"}));
            }
            continue;
        }
        if artifact["availability"] != "present" {
            if selected_artifact.is_some() {
                skipped.push(json!({"run_id":run_id,"artifact_id":id,"reason":"指定 checkpoint 当前不可用或处于释放恢复状态"}));
            }
            continue;
        }
        if artifact["complete"] != true
            || !artifact["references"].as_array().is_some_and(Vec::is_empty)
        {
            skipped.push(json!({"run_id":run_id,"artifact_id":id,"reason":"自定义 checkpoint 不完整或仍有登记引用，保持保护"}));
            continue;
        }
        let consumers = project_consumers
            .get(&(run_id.to_owned(), id.to_owned()))
            .cloned()
            .unwrap_or_default();
        if !consumers.is_empty() {
            skipped.push(json!({
                "run_id":run_id,
                "artifact_id":id,
                "consumers":consumers,
                "reason":"项目级 typed consumer 仍依赖此 checkpoint，保持保护"
            }));
            continue;
        }
        let Some(files) = artifact["files"].as_array() else {
            skipped.push(json!({"run_id":run_id,"artifact_id":id,"reason":"自定义 checkpoint 文件集合无效，保持保护"}));
            continue;
        };
        if files.is_empty() {
            skipped.push(json!({"run_id":run_id,"artifact_id":id,"reason":"自定义 checkpoint 文件集合为空，保持保护"}));
            continue;
        }
        let mut planned_files = Vec::new();
        let mut total = 0_u64;
        let mut invalid = None;
        for file in files {
            let Some(relative) = file["path"].as_str() else {
                invalid = Some("自定义 checkpoint 路径缺失".to_owned());
                break;
            };
            if safe_relative_path(relative).is_err() || !relative.starts_with("artifacts/") {
                invalid = Some("自定义 checkpoint 路径不安全".to_owned());
                break;
            }
            let target = run_directory.join(relative);
            let metadata = match fs::symlink_metadata(&target) {
                Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                    invalid = Some(format!("文件不再是普通文件：{relative}"));
                    break;
                }
                Ok(metadata) => metadata,
                Err(error) => {
                    invalid = Some(format!("文件不可用：{relative}（{error}）"));
                    break;
                }
            };
            if super::runs::FileRunStore::has_shared_links_for_artifact(&metadata) {
                invalid = Some(format!("文件存在硬链接共享引用：{relative}"));
                break;
            }
            let (size_bytes, sha256) = match artifact_file(run_directory, relative) {
                Ok(value) => value,
                Err(error) => {
                    invalid = Some(format!("文件无法校验：{relative}（{error}）"));
                    break;
                }
            };
            if file["size_bytes"].as_u64() != Some(size_bytes)
                || file["sha256"].as_str() != Some(sha256.as_str())
            {
                invalid = Some(format!("文件内容与登记清单不同：{relative}"));
                break;
            }
            total = total.saturating_add(size_bytes);
            planned_files.push(json!({"path":relative,"size_bytes":size_bytes,"sha256":sha256}));
        }
        if let Some(reason) = invalid {
            skipped.push(json!({"run_id":run_id,"artifact_id":id,"reason":reason}));
            continue;
        }
        items.push(json!({
            "run_id":run_id,
            "kind":"managed_checkpoint_release",
            "artifact_id":id,
            "owner":artifact["owner"],
            "files":planned_files,
            "size_bytes":total,
            "sha256":artifact_files_digest(&planned_files),
            "release_checkpoint":true,
            "reason":format!("显式释放由 {} 登记的多文件 checkpoint；此操作会失去该恢复点。",artifact["owner"].as_str().unwrap_or("未知生产者"))
        }));
    }
    if let Some(selected) = selected_artifact
        && !manifest["artifacts"]
            .as_array()
            .is_some_and(|items| items.iter().any(|item| item["id"] == selected))
    {
        skipped.push(
            json!({"run_id":run_id,"artifact_id":selected,"reason":"指定 artifact ID 不存在"}),
        );
    }
    (items, skipped)
}

pub(super) fn project_artifact_consumers(
    project_root: &Path,
) -> Result<std::collections::BTreeMap<(String, String), Vec<String>>, String> {
    let runs_root = project_root.join(".model-maven/runs");
    let metadata = match fs::symlink_metadata(&runs_root) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(std::collections::BTreeMap::new());
        }
        Err(error) => return Err(error.to_string()),
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err("项目运行目录不可信；managed checkpoint 保持保护。".into());
        }
        Ok(metadata) => metadata,
    };
    let _ = metadata;
    let mut entries = fs::read_dir(&runs_root)
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    entries.sort_by_key(|entry| entry.file_name());
    let mut consumers = std::collections::BTreeMap::<(String, String), Vec<String>>::new();
    let mut artifact_identities = std::collections::BTreeSet::<(String, String)>::new();
    let mut declared_edges = Vec::<((String, String), String)>::new();
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let run_directory = entry.path();
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        if file_type.is_symlink() || !file_type.is_dir() {
            return Err(format!(
                "项目运行目录包含不可信条目 {name}；checkpoint 保持保护。"
            ));
        }
        let run_metadata_path = run_directory.join("run.json");
        let run_metadata = fs::read(&run_metadata_path)
            .map_err(|error| format!("无法读取运行 {name} 的身份：{error}"))
            .and_then(|bytes| {
                serde_json::from_slice::<Value>(&bytes)
                    .map_err(|error| format!("运行 {name} 的身份 JSON 无效：{error}"))
            })?;
        if run_metadata["id"].as_str() != Some(name.as_str()) {
            return Err(format!(
                "运行目录 {name} 的身份不匹配；checkpoint 保持保护。"
            ));
        }
        let status = run_metadata["status"].as_str().unwrap_or("unknown");
        if matches!(status, "running" | "stopping" | "unknown") {
            return Err(format!(
                "运行 {name} 尚未形成稳定产物图（状态 {status}）；checkpoint 保持保护。"
            ));
        }
        let manifest_path = run_directory.join("artifacts.json");
        let manifest = fs::read(&manifest_path)
            .map_err(|error| format!("运行 {name} 缺少可解析的产物图：{error}"))
            .and_then(|bytes| {
                serde_json::from_slice::<Value>(&bytes)
                    .map_err(|error| format!("运行 {name} 的产物清单无效：{error}"))
            })?;
        validate_manifest(&manifest, &name)?;
        if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION || manifest["status"] != "complete"
        {
            return Err(format!(
                "运行 {name} 的产物清单缺少完整 typed identity；checkpoint 保持保护。"
            ));
        }
        for artifact in manifest["artifacts"].as_array().into_iter().flatten() {
            let consumer_id = artifact["id"].as_str().ok_or_else(|| {
                format!("运行 {name} 的产物缺少 artifact id；checkpoint 保持保护。")
            })?;
            artifact_identities.insert((name.clone(), consumer_id.to_owned()));
            for edge in artifact["consumes"].as_array().into_iter().flatten() {
                let producer_run = edge["run_id"].as_str().ok_or_else(|| {
                    format!("运行 {name} 的 typed consumer identity 无效；checkpoint 保持保护。")
                })?;
                let producer_artifact = edge["artifact_id"].as_str().ok_or_else(|| {
                    format!("运行 {name} 的 typed consumer identity 无效；checkpoint 保持保护。")
                })?;
                declared_edges.push((
                    (producer_run.to_owned(), producer_artifact.to_owned()),
                    format!("{name}/{consumer_id}"),
                ));
            }
        }
    }
    for (producer, consumer) in declared_edges {
        if !artifact_identities.contains(&producer) {
            return Err(format!(
                "consumer {consumer} 声明的 producer {}/{} 不存在；checkpoint 保持保护。",
                producer.0, producer.1
            ));
        }
        consumers.entry(producer).or_default().push(consumer);
    }
    Ok(consumers)
}

pub(super) fn artifact_files_digest(files: &[Value]) -> String {
    let digest = Sha256::digest(serde_json::to_vec(files).unwrap_or_default());
    format!("{digest:x}")
}

pub(super) fn validate_manifest(value: &Value, run_id: &str) -> Result<(), String> {
    if !matches!(value["schema_version"].as_u64(), Some(1 | 2))
        || value["run_id"] != run_id
        || !matches!(value["status"].as_str(), Some("complete" | "incomplete"))
        || !value["artifacts"].is_array()
        || (value["status"] == "incomplete"
            && (value["artifacts"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
                || value["unregistered_files_protected"] != true))
    {
        return Err(format!("运行 {run_id} 的产物清单身份或格式无效。"));
    }
    if value["schema_version"] == 2 {
        let mut artifact_ids = std::collections::BTreeSet::new();
        for artifact in value["artifacts"].as_array().into_iter().flatten() {
            let artifact_id = artifact["id"]
                .as_str()
                .ok_or_else(|| format!("运行 {run_id} 的 artifact 缺少 id。"))?;
            validate_label(artifact_id, "artifact id")?;
            if !artifact_ids.insert(artifact_id) {
                return Err(format!(
                    "运行 {run_id} 的 artifact id 重复：{artifact_id}。"
                ));
            }
            if !artifact["references"].is_array() || !artifact["files"].is_array() {
                return Err(format!("运行 {run_id} 的 artifact 引用或文件列表无效。"));
            }
            let consumes = artifact["consumes"]
                .as_array()
                .ok_or_else(|| format!("运行 {run_id} 的 typed consumes 不是数组。"))?;
            let mut unique_consumes = std::collections::BTreeSet::new();
            for edge in consumes {
                let producer_run = edge["run_id"]
                    .as_str()
                    .ok_or_else(|| format!("运行 {run_id} 的 typed artifact identity 无效。"))?;
                let producer_artifact = edge["artifact_id"]
                    .as_str()
                    .ok_or_else(|| format!("运行 {run_id} 的 typed artifact identity 无效。"))?;
                validate_label(producer_run, "consumed run_id")?;
                validate_label(producer_artifact, "consumed artifact_id")?;
                if !unique_consumes.insert((producer_run, producer_artifact)) {
                    return Err(format!("运行 {run_id} 的 typed consumes 存在重复身份。"));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::build_run_manifest;
    use serde_json::json;
    use std::fs;

    #[test]
    fn explicit_managed_artifact_records_owner_references_and_content_hash() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("artifacts")).unwrap();
        fs::write(root.path().join("artifacts/weights.bin"), b"weights").unwrap();
        let report = json!({"data":{"managed_artifacts":[{
            "id":"custom-model",
            "owner":"project.trainer.v1",
            "complete":true,
            "references":["evaluation:run-2","evaluation:run-2"],
            "files":["artifacts/weights.bin"]
        }]}});
        let manifest = build_run_manifest(root.path(), "run-1", "completed", &report).unwrap();
        let artifact = &manifest["artifacts"][0];
        assert_eq!(artifact["owner"], "project.trainer.v1");
        assert_eq!(artifact["complete"], true);
        assert_eq!(artifact["references"], json!(["evaluation:run-2"]));
        assert_eq!(artifact["files"][0]["size_bytes"], 7);
        assert_eq!(
            artifact["files"][0]["sha256"],
            "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
        );
    }

    #[test]
    fn failed_run_never_marks_declared_artifact_complete() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("artifacts")).unwrap();
        fs::write(root.path().join("artifacts/weights.bin"), b"weights").unwrap();
        let report = json!({"data":{"managed_artifacts":[{
            "id":"custom-model","owner":"trainer","complete":true,"files":["artifacts/weights.bin"]
        }]}});
        let manifest = build_run_manifest(root.path(), "run-1", "failed", &report).unwrap();
        assert_eq!(manifest["artifacts"][0]["complete"], false);
        assert_eq!(manifest["artifacts"][0]["declared_complete"], true);
    }

    #[test]
    fn resumed_run_records_typed_parent_checkpoint_consumption() {
        let report = json!({"data":{"parent_run_id":"run-parent-1"}});
        assert_eq!(
            super::report_consumes(&report, "run-child-2"),
            vec![json!({"run_id":"run-parent-1","artifact_id":"training-checkpoint"})]
        );
    }

    #[test]
    fn traversal_duplicate_and_symlink_artifacts_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("artifacts")).unwrap();
        fs::write(root.path().join("artifacts/weights.bin"), b"weights").unwrap();
        let traversal = json!({"data":{"managed_artifacts":[{
            "id":"unsafe","owner":"trainer","complete":true,"files":["../weights.bin"]
        }]}});
        assert!(build_run_manifest(root.path(), "run-1", "completed", &traversal).is_err());

        let duplicate = json!({"data":{"managed_artifacts":[
            {"id":"one","owner":"trainer","complete":true,"files":["artifacts/weights.bin"]},
            {"id":"two","owner":"trainer","complete":true,"files":["artifacts/weights.bin"]}
        ]}});
        assert!(build_run_manifest(root.path(), "run-1", "completed", &duplicate).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            symlink("weights.bin", root.path().join("artifacts/alias.bin")).unwrap();
            let linked = json!({"data":{"managed_artifacts":[{
                "id":"linked","owner":"trainer","complete":true,"files":["artifacts/alias.bin"]
            }]}});
            assert!(build_run_manifest(root.path(), "run-1", "completed", &linked).is_err());
        }
    }
}
