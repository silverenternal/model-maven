//! Persistent request generations and publication guard for data previews.

use fs2::FileExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::{
    fs::{self, DirBuilder, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const PREVIEW_STATE_VERSION: u64 = 1;
static PREVIEW_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// File-backed state for coalesced preview requests and atomic current publication.
pub struct FilePreviewStore;

impl FilePreviewStore {
    /// Start the latest request. A verified successful preview with the same signature is reused.
    pub fn begin(project_root: &Path, input_signature: &str) -> Result<Value, String> {
        Self::validate_signature(input_signature)?;
        let root = Self::root(project_root, true)?;
        let _lock = Self::lock(&root)?;
        let mut state = Self::read_state(&root)?;
        let generation = state["generation"]
            .as_u64()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or("预览 generation 已耗尽。")?;
        let request_id = Self::request_id(generation);

        if let Some(previous) = state["current"].as_object()
            && matches!(
                previous.get("status").and_then(Value::as_str),
                Some("pending" | "running")
            )
        {
            let previous_id = previous
                .get("request_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !previous_id.is_empty() {
                Self::write_request_status(
                    &root,
                    previous_id,
                    &json!({
                        "status":"superseded",
                        "reason":"A newer preview request was submitted.",
                        "updated_at_unix_ms":Self::now_ms()
                    }),
                )?;
            }
        }

        let last_success_exists = state["last_success"].is_object();
        let last_success_matches = state["last_success"]["request_signature"] == input_signature
            || state["last_success"]["input_signature"] == input_signature;
        let cache_is_valid = if last_success_exists {
            state["last_success"]["scope"]["stable_sample_ids"] == true
                && Self::publication_is_valid(&root, &state["last_success"])?
        } else {
            false
        };
        if last_success_matches && cache_is_valid {
            let mut publication = state["last_success"].clone();
            publication["cached"] = json!(true);
            state["generation"] = json!(generation);
            state["current"] = json!({
                "request_id":request_id,
                "generation":generation,
                "request_signature":input_signature,
                "input_signature":input_signature,
                "status":"succeeded",
                "cached":true,
                "updated_at_unix_ms":Self::now_ms()
            });
            state["publication"] = publication.clone();
            state["last_success_stale"] = json!(false);
            state["updated_at_unix_ms"] = json!(Self::now_ms());
            Self::atomic_json(&root.join("current.json"), &state)?;
            return Ok(json!({
                "request_id":request_id,
                "generation":generation,
                "request_signature":input_signature,
                "input_signature":input_signature,
                "status":"succeeded",
                "cached":true,
                "publication":publication,
                "request_directory":Value::Null
            }));
        }

        let request_directory = root.join(&request_id);
        Self::create_private_dir(&request_directory)?;
        Self::atomic_json(
            &request_directory.join("request.json"),
            &json!({
                "schema_version":PREVIEW_STATE_VERSION,
                "request_id":request_id,
                "generation":generation,
                "request_signature":input_signature,
                "input_signature":input_signature,
                "created_at_unix_ms":Self::now_ms()
            }),
        )?;
        Self::atomic_json(
            &request_directory.join("status.json"),
            &json!({"status":"pending","updated_at_unix_ms":Self::now_ms()}),
        )?;
        state["generation"] = json!(generation);
        state["current"] = json!({
            "request_id":request_id,
            "generation":generation,
            "request_signature":input_signature,
            "input_signature":input_signature,
            "status":"pending",
            "cached":false,
            "updated_at_unix_ms":Self::now_ms()
        });
        state["last_success_stale"] =
            json!(last_success_exists && (!last_success_matches || !cache_is_valid));
        state["updated_at_unix_ms"] = json!(Self::now_ms());
        Self::atomic_json(&root.join("current.json"), &state)?;
        Ok(json!({
            "request_id":request_id,
            "generation":generation,
            "input_signature":input_signature,
            "status":"pending",
            "cached":false,
            "request_directory":request_directory
        }))
    }

    /// Publish a Python renderer completion only if its generation is still current.
    pub fn complete(
        project_root: &Path,
        request_id: &str,
        generation: u64,
        input_signature: &str,
    ) -> Result<Value, String> {
        Self::validate_request_id(request_id)?;
        Self::validate_signature(input_signature)?;
        let root = Self::root(project_root, false)?;
        let request_directory = Self::request_directory(&root, request_id)?;
        let manifest_path = request_directory.join("complete.json");
        let manifest = Self::read_json(&manifest_path)?;
        let final_signature = manifest["input_signature"]
            .as_str()
            .ok_or("预览完成清单缺少 input_signature。")?;
        Self::validate_signature(final_signature)?;
        if manifest["schema_version"] != PREVIEW_STATE_VERSION
            || manifest["request_id"] != request_id
            || manifest["generation"] != generation
            || manifest["base_signature"] != input_signature
        {
            return Err("预览完成清单与请求 ID、generation 或 base signature 不匹配。".into());
        }
        let artifacts = Self::verify_artifacts(&root, request_id, &manifest["artifacts"])?;
        if artifacts.is_empty() {
            return Err("预览完成清单没有经过验证的产物。".into());
        }
        let publication = json!({
            "request_id":request_id,
            "generation":generation,
            "request_signature":input_signature,
            "input_signature":final_signature,
            "renderer":manifest["renderer"],
            "renderer_version":manifest["renderer_version"],
            "scope":manifest["scope"],
            "created_at":manifest["created_at"],
            "sample_count":manifest["sample_count"],
            "rendered_image_count":manifest["rendered_image_count"],
            "truncated":manifest["truncated"],
            "artifacts":artifacts,
            "cached":false
        });
        let _lock = Self::lock(&root)?;
        let mut state = Self::read_state(&root)?;
        let is_current = state["current"]["request_id"] == request_id
            && state["current"]["generation"] == generation
            && state["current"]["request_signature"] == input_signature
            && matches!(
                state["current"]["status"].as_str(),
                Some("pending" | "running")
            );
        if !is_current {
            let status = if state["current"]["request_id"] == request_id
                && state["current"]["generation"] == generation
                && state["current"]["status"] == "cancelled"
            {
                "cancelled"
            } else {
                "superseded"
            };
            Self::write_request_status(
                &root,
                request_id,
                &json!({"status":status,"updated_at_unix_ms":Self::now_ms()}),
            )?;
            return Ok(json!({"published":false,"status":status,"publication":publication}));
        }
        state["current"]["status"] = json!("succeeded");
        state["current"]["input_signature"] = json!(final_signature);
        state["current"]["updated_at_unix_ms"] = json!(Self::now_ms());
        state["publication"] = publication.clone();
        state["last_success"] = publication.clone();
        state["last_success_stale"] = json!(false);
        state["updated_at_unix_ms"] = json!(Self::now_ms());
        Self::write_request_status(
            &root,
            request_id,
            &json!({"status":"succeeded","input_signature":input_signature,"updated_at_unix_ms":Self::now_ms()}),
        )?;
        Self::atomic_json(&root.join("current.json"), &state)?;
        Ok(json!({"published":true,"status":"succeeded","publication":publication}))
    }

    /// Record failure while retaining the last successful publication as an explicitly stale result.
    pub fn fail(
        project_root: &Path,
        request_id: &str,
        generation: u64,
        message: &str,
    ) -> Result<Value, String> {
        Self::finish_non_success(project_root, request_id, generation, "failed", message)
    }

    /// Cancel one preview request; this store has no ownership of training jobs.
    pub fn cancel(project_root: &Path, request_id: &str, generation: u64) -> Result<Value, String> {
        Self::finish_non_success(
            project_root,
            request_id,
            generation,
            "cancelled",
            "Cancelled by user.",
        )
    }

    /// Record that the isolated renderer started without changing the user's training state.
    pub fn mark_running(
        project_root: &Path,
        request_id: &str,
        generation: u64,
    ) -> Result<bool, String> {
        Self::validate_request_id(request_id)?;
        let root = Self::root(project_root, false)?;
        let _lock = Self::lock(&root)?;
        let mut state = Self::read_state(&root)?;
        let is_current = state["current"]["request_id"] == request_id
            && state["current"]["generation"] == generation
            && state["current"]["status"] == "pending";
        if is_current {
            state["current"]["status"] = json!("running");
            state["current"]["updated_at_unix_ms"] = json!(Self::now_ms());
            state["updated_at_unix_ms"] = json!(Self::now_ms());
            Self::write_request_status(
                &root,
                request_id,
                &json!({"status":"running","updated_at_unix_ms":Self::now_ms()}),
            )?;
            Self::atomic_json(&root.join("current.json"), &state)?;
        }
        Ok(is_current)
    }

    /// Read current state and publication without inferring that a remote artifact was opened.
    pub fn current(project_root: &Path) -> Result<Value, String> {
        let root = Self::root(project_root, false)?;
        let _lock = Self::lock(&root)?;
        Self::read_state(&root)
    }

    fn finish_non_success(
        project_root: &Path,
        request_id: &str,
        generation: u64,
        status: &str,
        message: &str,
    ) -> Result<Value, String> {
        Self::validate_request_id(request_id)?;
        let root = Self::root(project_root, false)?;
        let _lock = Self::lock(&root)?;
        let mut state = Self::read_state(&root)?;
        let is_current = state["current"]["request_id"] == request_id
            && state["current"]["generation"] == generation;
        let is_active = is_current
            && matches!(
                state["current"]["status"].as_str(),
                Some("pending" | "running")
            );
        if is_active {
            Self::write_request_status(
                &root,
                request_id,
                &json!({"status":status,"message":message,"updated_at_unix_ms":Self::now_ms()}),
            )?;
            state["current"]["status"] = json!(status);
            state["current"]["message"] = json!(message);
            state["current"]["updated_at_unix_ms"] = json!(Self::now_ms());
            state["updated_at_unix_ms"] = json!(Self::now_ms());
            Self::atomic_json(&root.join("current.json"), &state)?;
        } else if !is_current {
            Self::write_request_status(
                &root,
                request_id,
                &json!({"status":"superseded","message":message,"updated_at_unix_ms":Self::now_ms()}),
            )?;
        }
        let final_status = if is_active {
            status
        } else if is_current {
            state["current"]["status"].as_str().unwrap_or("unknown")
        } else {
            "superseded"
        };
        Ok(json!({"updated":is_active,"status":final_status,"last_success":state["last_success"]}))
    }

    fn root(project_root: &Path, create: bool) -> Result<PathBuf, String> {
        let project_root = project_root
            .canonicalize()
            .map_err(|error| format!("项目目录不可用：{error}"))?;
        if !project_root.is_dir() {
            return Err("项目根路径不是目录。".into());
        }
        let metadata = project_root.join(".model-maven");
        let previews = metadata.join("previews");
        Self::ensure_directory(&metadata, create)?;
        Self::ensure_directory(&previews, create)?;
        Ok(previews)
    }

    fn ensure_directory(path: &Path, create: bool) -> Result<(), String> {
        match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                Err(format!("拒绝通过符号链接访问预览存储：{}", path.display()))
            }
            Ok(metadata) if !metadata.is_dir() => {
                Err(format!("预览存储路径不是目录：{}", path.display()))
            }
            Ok(_) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && create => {
                Self::create_private_dir(path)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err("预览存储尚未初始化。".into())
            }
            Err(error) => Err(error.to_string()),
        }
    }

    fn create_private_dir(path: &Path) -> Result<(), String> {
        let mut builder = DirBuilder::new();
        #[cfg(unix)]
        builder.mode(0o700);
        builder.create(path).map_err(|error| error.to_string())
    }

    fn lock(root: &Path) -> Result<File, String> {
        let path = root.join("preview.lock");
        if fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return Err("拒绝通过符号链接访问预览锁。".into());
        }
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options.open(path).map_err(|error| error.to_string())?;
        file.lock_exclusive().map_err(|error| error.to_string())?;
        Ok(file)
    }

    fn read_state(root: &Path) -> Result<Value, String> {
        let path = root.join("current.json");
        match Self::read_json_optional(&path)? {
            Some(state) if state["schema_version"] == PREVIEW_STATE_VERSION => Ok(state),
            Some(state) => Err(format!(
                "不支持预览状态版本 {:?}。",
                state["schema_version"]
            )),
            None => Ok(json!({
                "schema_version":PREVIEW_STATE_VERSION,
                "generation":0,
                "current":Value::Null,
                "publication":Value::Null,
                "last_success":Value::Null,
                "last_success_stale":false,
                "updated_at_unix_ms":Self::now_ms()
            })),
        }
    }

    fn read_json_optional(path: &Path) -> Result<Option<Value>, String> {
        match fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.to_string()),
            Ok(metadata) if metadata.file_type().is_symlink() => {
                Err(format!("拒绝读取符号链接：{}", path.display()))
            }
            Ok(metadata) if !metadata.is_file() => {
                Err(format!("JSON 状态不是普通文件：{}", path.display()))
            }
            Ok(_) => {
                let contents = fs::read(path).map_err(|error| error.to_string())?;
                serde_json::from_slice(&contents)
                    .map(Some)
                    .map_err(|error| format!("预览状态损坏：{error}"))
            }
        }
    }

    fn read_json(path: &Path) -> Result<Value, String> {
        Self::read_json_optional(path)?.ok_or_else(|| format!("预览文件不存在：{}", path.display()))
    }

    fn atomic_json(path: &Path, value: &Value) -> Result<(), String> {
        let parent = path.parent().ok_or("预览文件缺少父目录。")?;
        let mut temporary =
            tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
        serde_json::to_writer_pretty(&mut temporary, value).map_err(|error| error.to_string())?;
        temporary
            .write_all(b"\n")
            .map_err(|error| error.to_string())?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|error| error.to_string())?;
        temporary.persist(path).map_err(|error| error.to_string())?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| error.to_string())
    }

    fn request_directory(root: &Path, request_id: &str) -> Result<PathBuf, String> {
        Self::validate_request_id(request_id)?;
        let path = root.join(request_id);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                Err("预览请求目录不能是符号链接。".into())
            }
            Ok(metadata) if !metadata.is_dir() => Err("预览请求路径不是目录。".into()),
            Ok(_) => Ok(path),
            Err(error) => Err(error.to_string()),
        }
    }

    fn write_request_status(root: &Path, request_id: &str, value: &Value) -> Result<(), String> {
        let directory = Self::request_directory(root, request_id)?;
        Self::atomic_json(&directory.join("status.json"), value)
    }

    fn verify_artifacts(
        root: &Path,
        request_id: &str,
        artifacts: &Value,
    ) -> Result<Vec<Value>, String> {
        let artifacts = artifacts.as_array().ok_or("预览 artifacts 必须是数组。")?;
        let request_directory = Self::request_directory(root, request_id)?;
        let mut verified = Vec::new();
        for artifact in artifacts {
            let relative = artifact["path"]
                .as_str()
                .ok_or("预览 artifact 缺少相对 path。")?;
            let relative_path = Path::new(relative);
            if relative_path.is_absolute()
                || relative_path.components().any(|part| {
                    matches!(
                        part,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
                || relative_path.components().count() == 0
            {
                return Err("预览 artifact 路径必须位于请求目录内。".into());
            }
            let path = request_directory.join(relative_path);
            Self::reject_symlink_chain(&request_directory, relative_path)?;
            if relative == "complete.json" {
                return Err("complete.json 不能作为预览 artifact。".into());
            }
            let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
            if !metadata.is_file() {
                return Err(format!("预览 artifact 不是普通文件：{relative}"));
            }
            let expected = artifact["sha256"]
                .as_str()
                .ok_or("预览 artifact 缺少 sha256。")?;
            let mut file = File::open(&path).map_err(|error| error.to_string())?;
            let mut digest = Sha256::new();
            let mut buffer = [0_u8; 1024 * 1024];
            loop {
                let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
            }
            let actual = format!("{:x}", digest.finalize());
            if expected != actual || artifact["size_bytes"].as_u64() != Some(metadata.len()) {
                return Err(format!("预览 artifact 完整性检查失败：{relative}"));
            }
            let kind = artifact["kind"]
                .as_str()
                .ok_or("预览 artifact 缺少 kind。")?;
            let media_type = artifact["media_type"]
                .as_str()
                .ok_or("预览 artifact 缺少 media_type。")?;
            verified.push(json!({
                "artifact_id":format!("{request_id}:{kind}"),
                "kind":kind,
                "uri":format!(".model-maven/previews/{request_id}/{}",relative_path.to_string_lossy()),
                "media_type":media_type,
                "sha256":actual,
                "owner_operation_id":format!("preview:{request_id}"),
                "size_bytes":metadata.len()
            }));
        }
        Ok(verified)
    }

    fn publication_is_valid(root: &Path, publication: &Value) -> Result<bool, String> {
        let artifacts = match publication["artifacts"].as_array() {
            Some(artifacts) if !artifacts.is_empty() => artifacts,
            _ => return Ok(false),
        };
        for artifact in artifacts {
            let uri = artifact["uri"].as_str().ok_or("缓存 artifact 缺少 uri。")?;
            let path = Path::new(uri);
            if path.is_absolute()
                || path.components().any(|part| {
                    matches!(
                        part,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
            {
                return Ok(false);
            }
            let project_root = root
                .parent()
                .and_then(Path::parent)
                .ok_or("预览缓存缺少项目根目录。")?;
            Self::reject_symlink_chain(project_root, path)?;
            let full_path = project_root.join(path);
            let expected = artifact["sha256"].as_str().unwrap_or_default();
            let bytes = match fs::read(&full_path) {
                Ok(bytes) => bytes,
                Err(_) => return Ok(false),
            };
            let actual = format!("{:x}", Sha256::digest(&bytes));
            if actual != expected || artifact["size_bytes"].as_u64() != Some(bytes.len() as u64) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn reject_symlink_chain(root: &Path, relative: &Path) -> Result<(), String> {
        let mut current = root.to_path_buf();
        for component in relative.components() {
            if let Component::Normal(name) = component {
                current.push(name);
                if fs::symlink_metadata(&current)
                    .is_ok_and(|metadata| metadata.file_type().is_symlink())
                {
                    return Err(format!(
                        "拒绝通过符号链接读取预览产物：{}",
                        current.display()
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_signature(signature: &str) -> Result<(), String> {
        if signature.len() != 64 || !signature.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("预览 input_signature 必须是 SHA-256 十六进制摘要。".into());
        }
        Ok(())
    }

    fn validate_request_id(request_id: &str) -> Result<(), String> {
        if request_id.is_empty()
            || request_id.len() > 96
            || !request_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err("预览 request_id 格式无效。".into());
        }
        Ok(())
    }

    fn request_id(generation: u64) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let sequence = PREVIEW_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        format!("preview-{generation}-{nanos:x}-{sequence:x}")
    }

    fn now_ms() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }
}
