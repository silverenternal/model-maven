use crate::{
    domain::input::content_revision,
    ports::{ProjectConfigStore, ProjectConfigUpdate, ProjectConfigUpdateError},
};
use fs2::FileExt;
use serde_json::Value;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

const CONFIG_PATH: &str = ".model-maven/project.json";
const MAX_CONFIG_BYTES: u64 = 64 * 1024;
const HISTORY_PATH: &str = ".model-maven/project.history.json";
const MAX_HISTORY_BYTES: u64 = 2 * 1024 * 1024;
const MAX_HISTORY_ENTRIES: usize = 20;

pub struct FileProjectConfigStore;

fn entrypoint_is_valid(value: &Value) -> bool {
    let Some(value) = value.as_str() else {
        return false;
    };
    let Some((module, callable)) = value.split_once(':') else {
        return false;
    };
    !module.trim().is_empty() && !callable.trim().is_empty() && !callable.contains(':')
}

fn identifier_is_valid(value: &str) -> bool {
    let mut chars = value.chars();
    chars.next().is_some_and(|ch| ch == '_' || unicode_ident::is_xid_start(ch))
        && chars.all(|ch| unicode_ident::is_xid_continue(ch))
}

fn injection_entrypoint_is_valid(value: &Value) -> bool {
    let Some(value) = value.as_str() else {
        return false;
    };
    let Some((module, callable)) = value.split_once(':') else {
        return false;
    };
    !module.is_empty()
        && !callable.is_empty()
        && !value.contains(char::is_whitespace)
        && module.split('.').all(identifier_is_valid)
        && callable.split('.').all(identifier_is_valid)
}

fn validate(config: &Value) -> Result<(), String> {
    if config["schema_version"] != 1 {
        return Err(format!(
            "不支持项目配置 schema_version={}；当前版本只读取 schema_version=1。",
            config["schema_version"]
        ));
    }
    if let Some(python) = config.get("runtime_python").filter(|value| !value.is_null()) {
        if python.as_str().is_none_or(|value| value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control)) {
            return Err("runtime_python 必须是非空解释器路径或命令，且不能包含控制字符。".into());
        }
    }
    if !entrypoint_is_valid(&config["model"])
        || !(config["batch"].is_null() || entrypoint_is_valid(&config["batch"]))
    {
        return Err("项目配置格式无效；model/batch 必须是 module:callable 入口。".into());
    }
    if !matches!(
        config["batch_binding"].as_str(),
        None | Some("envelope" | "single" | "single-raw" | "args" | "kwargs")
    ) || (config["batch"].is_null() && !config["batch_binding"].is_null())
    {
        return Err(
            "batch_binding 必须是 envelope/single/single-raw/args/kwargs，且只能和 batch 一起保存。".into(),
        );
    }
    if let Some(params) = config.get("batch_params") {
        if !params.is_object()
            || (config["batch"].is_null() && !params.as_object().unwrap().is_empty())
        {
            return Err("batch_params 必须是对象，且非空参数需要批次入口。".into());
        }
    }
    if config
        .get("model_params")
        .is_some_and(|params| !params.is_object())
    {
        return Err("model_params 必须是对象。".into());
    }
    if let Some(paths) = config.get("runtime_paths").filter(|paths| !paths.is_null()) {
        let object = paths.as_object().ok_or("runtime_paths 必须是对象。")?;
        if object.keys().any(|key| key != "working_directory" && key != "project_import_roots") {
            return Err("runtime_paths 包含未知字段。".into());
        }
        if let Some(directory) = object.get("working_directory").filter(|value| !value.is_null()) {
            if directory.as_str().is_none_or(|value| value.is_empty() || !Path::new(value).is_absolute() || value.contains('\0')) {
                return Err("runtime_paths.working_directory 必须是绝对路径或 null。".into());
            }
        }
        if let Some(roots) = object.get("project_import_roots").filter(|value| !value.is_null()) {
            let roots = roots.as_array().ok_or("project_import_roots 必须是路径数组。")?;
            let mut seen = std::collections::HashSet::new();
            if roots.len() > 32 || roots.iter().any(|value| {
                let Some(value) = value.as_str() else { return true; };
                value.is_empty() || value.len() > 1024 || value.contains('\\') || value.contains('\0')
                    || value.split('/').any(|part| part.is_empty() || part == "." || part == "..")
                    || Path::new(value).is_absolute()
                    || Path::new(value).components().any(|part| !matches!(part, std::path::Component::Normal(_)))
                    || !seen.insert(value)
            }) {
                return Err("project_import_roots 必须是不重复的项目相对路径数组。".into());
            }
        }
    }
    if let Some(injections) = config.get("model_injections") {
        let injections = injections
            .as_object()
            .ok_or("model_injections 必须是 SLOT 到注入工厂或已锁定组件的对象。")?;
        if injections.len() > 32 {
            return Err("model_injections 最多声明 32 个接缝。".into());
        }
        for (slot, specification) in injections {
            if !identifier_is_valid(slot) {
                return Err(format!(
                    "模型接缝名无效：{slot}；接缝名须是 Python 标识符。"
                ));
            }
            if specification.is_string() {
                if !injection_entrypoint_is_valid(specification) {
                    return Err(format!("模型接缝 {slot} 的 Python 工厂入口无效。"));
                }
                continue;
            }
            if specification.get("component_ref").is_none() {
                if !injection_entrypoint_is_valid(&specification["factory"])
                    || !specification["params"].is_object()
                    || specification.as_object().is_none_or(|object| object.keys().any(|key| key != "factory" && key != "params"))
                {
                    return Err(format!("模型接缝 {slot} 的直接工厂或参数无效。"));
                }
                continue;
            }
            let reference = &specification["component_ref"];
            let sha256 = reference["sha256"].as_str().unwrap_or_default();
            let path = reference["path"].as_str().unwrap_or_default();
            if !injection_entrypoint_is_valid(&specification["factory"])
                || !specification["params"].is_object()
                || reference["id"].as_str().is_none_or(str::is_empty)
                || reference["version"].as_str().is_none_or(str::is_empty)
                || sha256.len() != 64
                || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                || !path.starts_with(".model-maven/components/")
                || path.split('/').any(|part| part == "..")
            {
                return Err(format!("模型接缝 {slot} 的注册组件参数或版本锁无效。"));
            }
        }
    }
    Ok(())
}

fn config_path(root: &Path) -> Result<(PathBuf, PathBuf), String> {
    let root = root.canonicalize().map_err(|error| error.to_string())?;
    if !root.is_dir() {
        return Err("项目根路径不是目录。".into());
    }
    let directory = root.join(".model-maven");
    let path = root.join(CONFIG_PATH);
    match fs::symlink_metadata(&directory) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err("项目元数据目录不能通过符号链接重定向。".into());
        }
        Ok(metadata) if !metadata.is_dir() => {
            return Err("项目元数据路径不是目录。".into());
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    Ok((directory, path))
}

fn read_config(path: &Path) -> Result<Option<Value>, String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err("拒绝通过符号链接读取项目配置。".into());
        }
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    if !metadata.is_file() || metadata.len() > MAX_CONFIG_BYTES {
        return Err("项目配置必须是小于 64 KiB 的普通文件。".into());
    }
    let config: Value = serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
        .map_err(|error| format!("项目配置 JSON 无法读取：{error}"))?;
    validate(&config)?;
    Ok(Some(config))
}

fn history_path(root: &Path) -> Result<PathBuf, String> {
    let (directory, _) = config_path(root)?;
    Ok(directory.join(Path::new(HISTORY_PATH).file_name().unwrap()))
}

fn read_history(path: &Path) -> Result<Vec<Value>, String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err("拒绝通过符号链接读取项目配置历史。".into());
        }
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.to_string()),
    };
    if !metadata.is_file() || metadata.len() > MAX_HISTORY_BYTES {
        return Err("项目配置历史必须是小于 2 MiB 的普通文件。".into());
    }
    let value: Value = serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
        .map_err(|error| format!("项目配置历史 JSON 无法读取：{error}"))?;
    value
        .as_array()
        .cloned()
        .ok_or_else(|| "项目配置历史格式无效。".into())
}

fn write_json_atomic(path: &Path, value: &Value) -> Result<(), String> {
    let directory = path
        .parent()
        .ok_or_else(|| "项目配置历史路径无效。".to_string())?;
    let mut temporary = NamedTempFile::new_in(directory).map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(&mut temporary, value).map_err(|error| error.to_string())?;
    temporary
        .write_all(b"\n")
        .map_err(|error| error.to_string())?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    temporary
        .persist(path)
        .map_err(|error| error.error.to_string())?;
    crate::adapters::sync_directory(directory)
        .map_err(|error| error.to_string())
}

impl ProjectConfigStore for FileProjectConfigStore {
    fn load(&self, root: &Path) -> Result<Option<Value>, String> {
        let (_, path) = config_path(root)?;
        read_config(&path)
    }

    fn create(&self, root: &Path, config: &Value) -> Result<PathBuf, String> {
        validate(config)?;
        let (directory, path) = config_path(root)?;
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        if fs::symlink_metadata(&directory).is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err("项目元数据目录不能通过符号链接重定向。".into());
        }
        let mut file = NamedTempFile::new_in(&directory).map_err(|error| error.to_string())?;
        serde_json::to_writer_pretty(&mut file, config).map_err(|error| error.to_string())?;
        file.write_all(b"\n").map_err(|error| error.to_string())?;
        file.as_file()
            .sync_all()
            .map_err(|error| error.to_string())?;
        match file.persist_noclobber(&path) {
            Ok(_) => {}
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err("项目配置已存在；不会覆盖现有默认入口。".into());
            }
            Err(error) => return Err(error.error.to_string()),
        }
        crate::adapters::sync_directory(&directory)
            .map_err(|error| error.to_string())?;
        Ok(path)
    }

    fn update(&self, root: &Path, expected: &Value, config: &Value) -> Result<PathBuf, String> {
        validate(config)?;
        let (directory, path) = config_path(root)?;
        if !path.exists() {
            return Err("项目配置不存在；先运行 m-model init。".into());
        }
        if fs::symlink_metadata(&directory).is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err("项目元数据目录不能通过符号链接重定向。".into());
        }
        let lock_path = directory.join("project.json.lock");
        if fs::symlink_metadata(&lock_path).is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err("项目配置锁不能是符号链接。".into());
        }
        let mut lock_options = OpenOptions::new();
        lock_options
            .create(true)
            .truncate(false)
            .read(true)
            .write(true);
        #[cfg(unix)]
        lock_options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        let lock = lock_options
            .open(&lock_path)
            .map_err(|error| error.to_string())?;
        FileExt::try_lock_exclusive(&lock)
            .map_err(|_| "另一项装配编辑正在更新项目配置，请稍后重试。".to_string())?;
        let current = read_config(&path)?;
        if current.as_ref() != Some(expected) {
            return Err("项目配置在装配候选验证期间发生变化；没有覆盖新的设置。".into());
        }
        let previous_revision = content_revision(expected)?;
        let history = history_path(root)?;
        if fs::symlink_metadata(&history).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return Err("项目配置历史不能是符号链接。".into());
        }
        let mut entries = read_history(&history)?;
        if expected != config {
            entries.retain(|entry| entry["revision"].as_str() != Some(previous_revision.as_str()));
            entries.insert(
                0,
                serde_json::json!({"revision":previous_revision,"config":expected}),
            );
            entries.truncate(MAX_HISTORY_ENTRIES);
            write_json_atomic(&history, &Value::Array(entries))?;
        }
        let mut temporary = NamedTempFile::new_in(&directory).map_err(|error| error.to_string())?;
        serde_json::to_writer_pretty(&mut temporary, config).map_err(|error| error.to_string())?;
        temporary
            .write_all(b"\n")
            .map_err(|error| error.to_string())?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|error| error.to_string())?;
        temporary
            .persist(&path)
            .map_err(|error| error.to_string())?;
        crate::adapters::sync_directory(&directory)
            .map_err(|error| error.to_string())?;
        Ok(path)
    }

    fn update_versioned(
        &self,
        root: &Path,
        expected_revision: &str,
        config: &Value,
    ) -> Result<ProjectConfigUpdate, ProjectConfigUpdateError> {
        let current = self
            .load(root)
            .map_err(ProjectConfigUpdateError::Failed)?
            .ok_or_else(|| ProjectConfigUpdateError::Failed("项目配置不存在。".into()))?;
        let current_revision =
            content_revision(&current).map_err(ProjectConfigUpdateError::Failed)?;
        if current_revision != expected_revision {
            return Err(ProjectConfigUpdateError::Conflict {
                expected_revision: expected_revision.into(),
                current_revision,
            });
        }
        let path = match self.update(root, &current, config) {
            Ok(path) => path,
            Err(message) if message.contains("在装配候选验证期间发生变化") => {
                let current_revision = self
                    .load(root)
                    .ok()
                    .flatten()
                    .and_then(|current| content_revision(&current).ok())
                    .unwrap_or_default();
                return Err(ProjectConfigUpdateError::Conflict {
                    expected_revision: expected_revision.into(),
                    current_revision,
                });
            }
            Err(message) => return Err(ProjectConfigUpdateError::Failed(message)),
        };
        let revision = content_revision(config).map_err(ProjectConfigUpdateError::Failed)?;
        Ok(ProjectConfigUpdate { path, revision })
    }

    fn history(&self, root: &Path) -> Result<Vec<Value>, String> {
        read_history(&history_path(root)?)
    }
}

#[cfg(test)]
mod tests {
    use super::{FileProjectConfigStore, ProjectConfigStore, ProjectConfigUpdateError};
    use crate::domain::input::content_revision;
    use serde_json::json;

    #[test]
    fn project_config_is_created_without_overwriting_and_rejects_bad_entries() {
        let root = tempfile::tempdir().unwrap();
        let store = FileProjectConfigStore;
        let config = json!({
            "schema_version":1,
            "model":"my_model:build_model",
            "batch":"my_data:example_batch",
            "batch_binding":"envelope"
        });
        let path = store.create(root.path(), &config).unwrap();
        assert!(path.is_file());
        assert_eq!(store.load(root.path()).unwrap(), Some(config.clone()));
        assert!(store.create(root.path(), &config).is_err());
        let updated = json!({
            "schema_version":1,
            "model":"my_model:build_model",
            "batch":"my_data:example_batch",
            "batch_binding":"envelope",
            "model_injections":{"encoder":"my_layers:build_encoder"}
        });
        assert_eq!(store.update(root.path(), &config, &updated).unwrap(), path);
        assert_eq!(store.load(root.path()).unwrap(), Some(updated));
        assert!(store.update(root.path(), &config, &config).is_err());

        let invalid = json!({
            "schema_version":1,
            "model":"not-a-callable",
            "batch":null,
            "batch_binding":null
        });
        assert!(store.create(root.path(), &invalid).is_err());

        let injection_root = tempfile::tempdir().unwrap();
        let invalid_injection = json!({
            "schema_version":1,
            "model":"my_model:build_model",
            "batch":null,
            "batch_binding":null,
            "model_injections":{"not-a-slot":"my_layers:build_encoder"}
        });
        assert!(
            store
                .create(injection_root.path(), &invalid_injection)
                .is_err()
        );

        let version_root = tempfile::tempdir().unwrap();
        let version_directory = version_root.path().join(".model-maven");
        std::fs::create_dir_all(&version_directory).unwrap();
        let unsupported_path = version_directory.join("project.json");
        let unsupported_config = json!({
            "schema_version":2,
            "model":"my_model:build_model",
            "batch":null,
            "batch_binding":null
        });
        std::fs::write(&unsupported_path, unsupported_config.to_string()).unwrap();
        let error = store.load(version_root.path()).unwrap_err();
        assert!(error.contains("schema_version=2"));
        assert_eq!(
            std::fs::read_to_string(unsupported_path).unwrap(),
            unsupported_config.to_string()
        );
    }

    #[test]
    fn versioned_update_preserves_history_and_returns_structured_stale_revision() {
        let root = tempfile::tempdir().unwrap();
        let store = FileProjectConfigStore;
        let first = json!({
            "schema_version":1,
            "model":"my_model:build_model",
            "batch":null,
            "batch_binding":null
        });
        store.create(root.path(), &first).unwrap();
        let first_revision = content_revision(&first).unwrap();
        let second = json!({
            "schema_version":1,
            "model":"my_model:build_model",
            "batch":"my_data:example_batch",
            "batch_binding":"envelope"
        });
        let update = store
            .update_versioned(root.path(), &first_revision, &second)
            .unwrap();
        assert_eq!(update.revision, content_revision(&second).unwrap());
        assert_eq!(store.load(root.path()).unwrap(), Some(second.clone()));

        let history = store.history(root.path()).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0]["revision"], first_revision);
        assert_eq!(history[0]["config"], first);

        let conflict = store
            .update_versioned(root.path(), &first_revision, &first)
            .unwrap_err();
        assert_eq!(
            conflict,
            ProjectConfigUpdateError::Conflict {
                expected_revision: first_revision,
                current_revision: content_revision(&second).unwrap()
            }
        );
        assert_eq!(store.load(root.path()).unwrap(), Some(second));
    }
}
