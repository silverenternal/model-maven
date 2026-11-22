use crate::ports::RunRepository;
use fs2::FileExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::{
    fs::{self, DirBuilder, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(test)]
thread_local! {
    static ARTIFACT_RELEASE_FAIL_AFTER: std::cell::Cell<i64> = const { std::cell::Cell::new(-1) };
}

pub struct FileRunStore;
const RUN_METADATA_SCHEMA_VERSION: u64 = 1;
static RUN_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static CLEAN_SEQUENCE: AtomicU64 = AtomicU64::new(0);

impl FileRunStore {
    fn create_private_dir(path: &Path) -> Result<(), String> {
        let mut builder = DirBuilder::new();
        #[cfg(unix)]
        builder.mode(0o700);
        builder.create(path).map_err(|error| error.to_string())
    }

    fn roots(root: &Path, create: bool) -> Result<PathBuf, String> {
        let metadata = root.join(".model-maven");
        let runs = metadata.join("runs");
        for path in [&metadata, &runs] {
            match fs::symlink_metadata(path) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    return Err(format!("拒绝通过符号链接访问运行记录：{}", path.display()));
                }
                Ok(meta) if !meta.is_dir() => {
                    return Err(format!("运行记录路径不是目录：{}", path.display()));
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound && create => {
                    Self::create_private_dir(path)?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(runs.clone());
                }
                Err(error) => return Err(error.to_string()),
            }
        }
        Ok(runs)
    }

    fn validate_id(id: &str) -> Result<(), String> {
        if !id.starts_with("run-")
            || id.len() > 96
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err("运行 ID 格式无效。".into());
        }
        Ok(())
    }

    fn validate_name(name: &str) -> Result<String, String> {
        let name = name.trim();
        if name.is_empty() || name.chars().count() > 80 || name.chars().any(char::is_control) {
            return Err("运行名称须为 1 到 80 个可见字符。".into());
        }
        Ok(name.to_owned())
    }

    fn atomic_json(path: &Path, value: &Value) -> Result<(), String> {
        let parent = path.parent().ok_or("运行记录缺少父目录。")?;
        let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        serde_json::to_writer_pretty(&mut file, value).map_err(|e| e.to_string())?;
        file.write_all(b"\n").map_err(|e| e.to_string())?;
        file.as_file().sync_all().map_err(|e| e.to_string())?;
        file.persist(path).map_err(|e| e.to_string())?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|e| e.to_string())
    }

    fn append_event(directory: &Path, event: &Value) -> Result<(), String> {
        let path = directory.join("events.jsonl");
        if fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err("拒绝通过符号链接写入运行事件。".into());
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        serde_json::to_writer(&mut file, event).map_err(|e| e.to_string())?;
        file.write_all(b"\n").map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())
    }

    fn read_json(path: &Path) -> Result<Value, String> {
        if fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err(format!("拒绝通过符号链接读取运行记录：{}", path.display()));
        }
        let contents = fs::read(path).map_err(|e| e.to_string())?;
        serde_json::from_slice(&contents).map_err(|e| format!("运行记录损坏：{e}"))
    }

    fn read_run_metadata(path: &Path) -> Result<Value, String> {
        let metadata = Self::read_json(path)?;
        if metadata["schema_version"] != RUN_METADATA_SCHEMA_VERSION {
            return Err(format!(
                "不支持运行记录 schema_version={}；当前版本只读取 schema_version={}。保留原记录，使用兼容版本查看或导出后再迁移。",
                metadata["schema_version"], RUN_METADATA_SCHEMA_VERSION
            ));
        }
        Ok(metadata)
    }

    fn read_events(directory: &Path) -> Result<Vec<Value>, String> {
        let path = directory.join("events.jsonl");
        if fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err("拒绝通过符号链接读取运行事件。".into());
        }
        let contents = fs::read_to_string(path).map_err(|e| e.to_string())?;
        contents
            .lines()
            .map(|line| serde_json::from_str(line).map_err(|e| format!("运行事件损坏：{e}")))
            .collect()
    }

    fn tail_bytes(path: &Path, max_bytes: usize) -> Result<Option<Vec<u8>>, String> {
        match fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.to_string()),
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(format!("拒绝通过符号链接读取运行文件：{}", path.display()));
            }
            Ok(meta) if !meta.is_file() => {
                return Err(format!("运行文件不是普通文件：{}", path.display()));
            }
            Ok(_) => {}
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW);
        let mut file = options.open(path).map_err(|error| error.to_string())?;
        let length = file.metadata().map_err(|error| error.to_string())?.len();
        let start = length.saturating_sub(max_bytes as u64);
        file.seek(SeekFrom::Start(start))
            .map_err(|error| error.to_string())?;
        let mut bytes = Vec::with_capacity(length.saturating_sub(start) as usize);
        file.take(max_bytes as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        Ok(Some(bytes))
    }

    fn lock(directory: &Path) -> Result<File, String> {
        let path = directory.join("run.lock");
        if fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err("拒绝通过符号链接锁定运行记录。".into());
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|error| error.to_string())?;
        lock.lock_exclusive().map_err(|error| error.to_string())?;
        Ok(lock)
    }

    /// Serializes run admission and state transitions for the whole project.
    /// Keeping this lock separate from per-run locks makes concurrent `create`
    /// calls observe one another before either publishes a running record.
    fn activity_lock(runs: &Path) -> Result<File, String> {
        let path = runs.join(".activity.lock");
        if fs::symlink_metadata(&path).is_ok_and(|meta| meta.file_type().is_symlink()) {
            return Err("拒绝通过符号链接锁定项目活动任务。".into());
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|error| error.to_string())?;
        lock.lock_exclusive().map_err(|error| error.to_string())?;
        Ok(lock)
    }

    fn ensure_no_active_run(runs: &Path, request_id: Option<&str>) -> Result<(), String> {
        let entries = fs::read_dir(runs).map_err(|error| error.to_string())?;
        let mut active = None;
        for entry in entries {
            let entry = entry.map_err(|error| error.to_string())?;
            let file_type = entry.file_type().map_err(|error| error.to_string())?;
            if !file_type.is_dir() || file_type.is_symlink() {
                continue;
            }
            let directory = entry.path();
            if directory
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with('.'))
            {
                continue;
            }
            let metadata_path = directory.join("run.json");
            let metadata = match Self::read_run_metadata(&metadata_path) {
                Ok(metadata) => metadata,
                Err(_error) if !metadata_path.exists() => continue,
                Err(error) => {
                    return Err(format!(
                        "RUN_STATE_UNKNOWN:无法确认已有运行是否结束（{}）：{error}。为避免重复启动，拒绝新任务。",
                        directory.file_name().unwrap_or_default().to_string_lossy()
                    ));
                }
            };
            if request_id.is_some() && metadata["request_id"].as_str() == request_id {
                let id = metadata["id"].as_str().unwrap_or("unknown");
                return Err(format!("RUN_REQUEST_ID_EXISTS:{id}"));
            }
            if matches!(metadata["status"].as_str(), Some("running" | "stopping")) {
                let id = metadata["id"].as_str().unwrap_or("unknown");
                let name = metadata["name"].as_str().unwrap_or("未命名任务");
                active.get_or_insert_with(|| format!(
                    "RUN_ALREADY_ACTIVE:已有未结束任务 {name}（{id}，状态 {}）；请先查看或处理该任务。",
                    metadata["status"].as_str().unwrap_or("unknown")
                ));
            }
        }
        if let Some(error) = active {
            return Err(error);
        }
        Ok(())
    }

    fn directory(root: &Path, id: &str) -> Result<PathBuf, String> {
        Self::validate_id(id)?;
        let directory = Self::roots(root, false)?.join(id);
        match fs::symlink_metadata(&directory) {
            Ok(meta) if meta.file_type().is_symlink() => Err("拒绝访问符号链接运行记录。".into()),
            Ok(meta) if meta.is_dir() => Ok(directory),
            Ok(_) => Err("运行记录路径不是目录。".into()),
            Err(error) => Err(error.to_string()),
        }
    }

    fn clean_plans(root: &Path, create: bool) -> Result<PathBuf, String> {
        let path = Self::roots(root, create)?.join(".clean-plans");
        match fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                Err("拒绝通过符号链接访问清理计划。".into())
            }
            Ok(meta) if !meta.is_dir() => Err("清理计划路径不是目录。".into()),
            Ok(_) => Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && create => {
                Self::create_private_dir(&path)?;
                Ok(path)
            }
            Err(error) => Err(error.to_string()),
        }
    }

    fn file_sha256(path: &Path) -> Result<String, String> {
        let mut file = File::open(path).map_err(|error| error.to_string())?;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn validate_plan_id(id: &str) -> Result<(), String> {
        if !id.starts_with("clean-")
            || id.len() > 96
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err("清理计划 ID 格式无效。".into());
        }
        Ok(())
    }

    fn validate_checkpoint_filename(filename: &str) -> Result<(), String> {
        let stem = filename
            .strip_prefix("checkpoint-")
            .and_then(|value| value.strip_suffix(".pt"))
            .ok_or("checkpoint 文件名无效。")?;
        let mut parts = stem.split('-');
        let step = parts.next().unwrap_or_default();
        let nonce = parts.next();
        let has_extra = parts.next().is_some();
        let valid_step = step.len() == 9 && step.bytes().all(|byte| byte.is_ascii_digit());
        let valid_nonce = nonce.is_none_or(|value| {
            value.len() == 32 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        });
        if !valid_step || !valid_nonce || has_extra {
            return Err("checkpoint 文件名无效。".into());
        }
        Ok(())
    }

    fn json_sha256(value: &Value) -> Result<String, String> {
        let encoded = serde_json::to_vec(value).map_err(|error| error.to_string())?;
        let digest = Sha256::digest(encoded);
        Ok(format!("{digest:x}"))
    }

    fn checkpoints(directory: &Path, create: bool) -> Result<PathBuf, String> {
        let path = directory.join("checkpoints");
        match fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                Err("拒绝通过符号链接访问 checkpoint 目录。".into())
            }
            Ok(meta) if !meta.is_dir() => Err("checkpoint 路径不是目录。".into()),
            Ok(_) => Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && create => {
                Self::create_private_dir(&path)?;
                Ok(path)
            }
            Err(error) => Err(error.to_string()),
        }
    }

    // These explicit accumulators are shared across the recursive filesystem walk.
    #[allow(clippy::too_many_arguments)]
    fn scan_storage(
        path: &Path,
        root: &Path,
        totals: &mut [u64; 5],
        files: &mut u64,
        skipped: &mut Vec<String>,
        managed_paths: &std::collections::HashSet<String>,
        seen_files: &mut std::collections::HashSet<String>,
        deduplicated_files: &mut u64,
    ) -> Result<(), String> {
        for entry in fs::read_dir(path).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let entry_path = entry.path();
            let file_type = entry.file_type().map_err(|error| error.to_string())?;
            if file_type.is_symlink() {
                if skipped.len() < 100 {
                    skipped.push(
                        entry_path
                            .strip_prefix(root)
                            .unwrap_or(&entry_path)
                            .display()
                            .to_string(),
                    );
                }
                continue;
            }
            if file_type.is_dir() {
                Self::scan_storage(
                    &entry_path,
                    root,
                    totals,
                    files,
                    skipped,
                    managed_paths,
                    seen_files,
                    deduplicated_files,
                )?;
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let metadata = fs::symlink_metadata(&entry_path).map_err(|error| error.to_string())?;
            let identity = Self::file_identity(&metadata, &entry_path);
            if !seen_files.insert(identity) {
                *deduplicated_files = deduplicated_files.saturating_add(1);
                continue;
            }
            let relative = entry_path.strip_prefix(root).unwrap_or(&entry_path);
            let relative_string = relative.to_string_lossy().into_owned();
            let category = if relative
                .components()
                .next()
                .is_some_and(|part| part.as_os_str() == "checkpoints")
            {
                3
            } else if managed_paths.contains(&relative_string) {
                4
            } else if relative == Path::new("worker.log") {
                1
            } else if relative == Path::new("progress.jsonl") {
                2
            } else {
                0
            };
            totals[category] = totals[category].saturating_add(metadata.len());
            *files = files.saturating_add(1);
        }
        Ok(())
    }

    fn storage_entry(
        root: &Path,
        id: &str,
        seen_files: &mut std::collections::HashSet<String>,
    ) -> Result<Value, String> {
        let directory = Self::directory(root, id)?;
        let metadata = Self::read_run_metadata(&directory.join("run.json"))?;
        let mut totals = [0_u64; 5];
        let mut files = 0_u64;
        let mut skipped = Vec::new();
        let mut deduplicated_files = 0_u64;
        let manifest_path = directory.join("artifacts.json");
        let managed_paths = match fs::symlink_metadata(&manifest_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::collections::HashSet::new()
            }
            Err(error) => return Err(error.to_string()),
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                return Err("产物清单不是普通文件。".into());
            }
            Ok(_) => {
                let manifest = Self::read_json(&manifest_path)?;
                super::artifacts::validate_manifest(&manifest, id)?;
                manifest["artifacts"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .flat_map(|artifact| artifact["files"].as_array().into_iter().flatten())
                    .filter_map(|file| file["path"].as_str().map(str::to_owned))
                    .collect()
            }
        };
        Self::scan_storage(
            &directory,
            &directory,
            &mut totals,
            &mut files,
            &mut skipped,
            &managed_paths,
            seen_files,
            &mut deduplicated_files,
        )?;
        let total = totals.iter().copied().fold(0_u64, u64::saturating_add);
        Ok(json!({
            "id":id,
            "name":metadata["name"],
            "kept":metadata["kept"],
            "status":metadata["status"],
            "total_bytes":total,
            "file_count":files,
            "deduplicated_file_count":deduplicated_files,
            "categories":{
                "metadata":totals[0],
                "logs":totals[1],
                "progress":totals[2],
                "checkpoints":totals[3],
                "artifacts":totals[4]
            },
            "skipped_symlinks":skipped
        }))
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u64::MAX as u128) as u64
    }

    fn file_identity(metadata: &fs::Metadata, _path: &Path) -> String {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            format!("{}:{}", metadata.dev(), metadata.ino())
        }
        #[cfg(not(unix))]
        {
            let _ = metadata;
            _path
                .canonicalize()
                .unwrap_or_else(|_| _path.to_path_buf())
                .to_string_lossy()
                .into_owned()
        }
    }

    fn has_shared_links(metadata: &fs::Metadata) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            metadata.nlink() > 1
        }
        #[cfg(not(unix))]
        {
            let _ = metadata;
            false
        }
    }

    pub(super) fn has_shared_links_for_artifact(metadata: &fs::Metadata) -> bool {
        Self::has_shared_links(metadata)
    }

    pub(super) fn now_ms_for_artifact_manifest() -> u64 {
        Self::now_ms()
    }
}

impl FileRunStore {
    fn apply_managed_checkpoint_release(root: &Path, item: &Value) -> Result<Value, String> {
        let id = item["run_id"].as_str().ok_or("计划项目缺少 run_id。")?;
        let artifact_id = item["artifact_id"]
            .as_str()
            .ok_or("自定义 checkpoint 计划缺少 artifact_id。")?;
        if item["kind"] != "managed_checkpoint_release" || item["release_checkpoint"] != true {
            return Err("自定义 checkpoint 计划类型无效。".into());
        }
        let directory = Self::directory(root, id)?;
        let _lock = Self::lock(&directory)?;
        let metadata = Self::read_run_metadata(&directory.join("run.json"))?;
        let status = metadata["status"].as_str().unwrap_or("unknown");
        if metadata["id"] != id || !matches!(status, "completed" | "failed" | "interrupted") {
            return Err(format!("运行状态 {status} 已变化或受保护。"));
        }
        if metadata["kept"] == true {
            return Err("运行已标记保留。".into());
        }
        let manifest_path = directory.join("artifacts.json");
        let mut manifest = Self::read_json(&manifest_path)?;
        super::artifacts::validate_manifest(&manifest, id)?;
        let artifacts = manifest["artifacts"]
            .as_array_mut()
            .ok_or("产物清单缺少 artifacts 数组。")?;
        let Some(index) = artifacts
            .iter()
            .position(|artifact| artifact["id"] == artifact_id)
        else {
            return Err("自定义 checkpoint 已从产物清单移除。".into());
        };
        let artifact = artifacts[index].clone();
        if artifact["owner"] != item["owner"]
            || artifact["kind"] != "checkpoint"
            || artifact["complete"] != true
            || artifact["availability"] != "present"
            || !artifact["references"].as_array().is_some_and(Vec::is_empty)
        {
            return Err("自定义 checkpoint 的 owner、引用、状态或可用性已变化。".into());
        }
        let manifest_files = artifact["files"]
            .as_array()
            .ok_or("自定义 checkpoint 文件集合无效。")?;
        let planned_files = item["files"]
            .as_array()
            .ok_or("固定计划缺少自定义 checkpoint 文件集合。")?;
        if manifest_files.len() != planned_files.len() || manifest_files.is_empty() {
            return Err("自定义 checkpoint 文件集合与固定计划不同。".into());
        }
        for (stored, planned) in manifest_files.iter().zip(planned_files) {
            if stored["path"] != planned["path"]
                || stored["size_bytes"] != planned["size_bytes"]
                || stored["sha256"] != planned["sha256"]
                || stored["available"] == false
            {
                return Err("自定义 checkpoint 文件集合与固定计划不同。".into());
            }
            let relative = stored["path"].as_str().ok_or("产物路径无效。")?;
            if super::artifacts::safe_relative_path(relative).is_err()
                || !relative.starts_with("artifacts/")
            {
                return Err("自定义 checkpoint 路径不安全。".into());
            }
            let target = directory.join(relative);
            let file_metadata = fs::symlink_metadata(&target).map_err(|error| error.to_string())?;
            if file_metadata.file_type().is_symlink()
                || !file_metadata.is_file()
                || Self::has_shared_links(&file_metadata)
            {
                return Err(format!("自定义 checkpoint 文件已共享或不安全：{relative}"));
            }
            let (size_bytes, sha256) = super::artifacts::artifact_file(&directory, relative)?;
            if stored["size_bytes"].as_u64() != Some(size_bytes)
                || stored["sha256"].as_str() != Some(sha256.as_str())
            {
                return Err(format!("自定义 checkpoint 文件内容已变化：{relative}"));
            }
        }
        if item["sha256"] != super::artifacts::artifact_files_digest(planned_files) {
            return Err("自定义 checkpoint 固定计划摘要无效。".into());
        }

        artifacts[index]["availability"] = json!("releasing");
        Self::atomic_json(&manifest_path, &manifest)?;
        let mut removed = 0_usize;
        let mut released_bytes = 0_u64;
        let mut release_error = None;
        let release_files = manifest["artifacts"][index]["files"]
            .as_array()
            .cloned()
            .ok_or("产物清单文件列表无效。")?;
        for (file_index, stored) in release_files.iter().enumerate() {
            let relative = stored["path"].as_str().ok_or("产物路径无效。")?.to_owned();
            let target = directory.join(&relative);
            let current = match fs::symlink_metadata(&target) {
                Ok(metadata)
                    if metadata.is_file()
                        && !metadata.file_type().is_symlink()
                        && !Self::has_shared_links(&metadata) =>
                {
                    metadata
                }
                Ok(_) => {
                    release_error = Some(format!("文件变成共享或不安全目标：{relative}"));
                    break;
                }
                Err(error) => {
                    release_error = Some(format!("文件无法读取：{relative}（{error}）"));
                    break;
                }
            };
            let (size_bytes, sha256) = match super::artifacts::artifact_file(&directory, &relative)
            {
                Ok(value) => value,
                Err(error) => {
                    release_error = Some(error);
                    break;
                }
            };
            if stored["size_bytes"].as_u64() != Some(size_bytes)
                || stored["sha256"].as_str() != Some(sha256.as_str())
            {
                release_error = Some(format!("文件内容在应用期间发生变化：{relative}"));
                break;
            }
            if let Err(error) = fs::remove_file(&target) {
                release_error = Some(format!("无法释放文件 {relative}：{error}"));
                break;
            }
            manifest["artifacts"][index]["files"][file_index]["available"] = json!(false);
            removed += 1;
            released_bytes = released_bytes.saturating_add(current.len());
            if let Err(error) = Self::atomic_json(&manifest_path, &manifest) {
                release_error = Some(format!("文件已移除，但更新产物清单失败：{error}"));
                break;
            }
            #[cfg(test)]
            if ARTIFACT_RELEASE_FAIL_AFTER.with(|fail_after| fail_after.get()) == removed as i64 {
                release_error = Some("测试注入：模拟组内文件释放中断".into());
                break;
            }
        }
        if let Some(error) = release_error {
            manifest["artifacts"][index]["availability"] = json!(if removed == 0 {
                "changed"
            } else {
                "partial_release"
            });
            manifest["artifacts"][index]["complete"] = json!(false);
            Self::atomic_json(&manifest_path, &manifest)?;
            let _ = Self::append_event(
                &directory,
                &json!({"event":"custom_checkpoint_release_incomplete","at_ms":Self::now_ms(),"artifact_id":artifact_id,"removed_files":removed,"error":error}),
            );
            return Err(format!(
                "自定义 checkpoint 释放不完整；已移除 {removed} 个文件，其余文件保持保护：{error}"
            ));
        }
        manifest["artifacts"][index]["availability"] = json!("released");
        manifest["artifacts"][index]["complete"] = json!(false);
        Self::atomic_json(&manifest_path, &manifest)?;
        Self::append_event(
            &directory,
            &json!({"event":"custom_checkpoint_released","at_ms":Self::now_ms(),"artifact_id":artifact_id,"owner":artifact["owner"],"files":removed,"size_bytes":released_bytes}),
        )?;
        Ok(json!({
            "item_id":item["item_id"],
            "run_id":id,
            "artifact_id":artifact_id,
            "owner":artifact["owner"],
            "kind":"managed_checkpoint_release",
            "release_checkpoint":true,
            "files":manifest["artifacts"][index]["files"],
            "size_bytes":released_bytes,
            "availability":"released"
        }))
    }
}

impl RunRepository for FileRunStore {
    fn create(&self, root: &Path, inputs: &Value) -> Result<String, String> {
        let runs = Self::roots(root, true)?;
        let _activity = Self::activity_lock(&runs)?;
        Self::ensure_no_active_run(&runs, inputs["request_id"].as_str())?;
        let now = Self::now_ms();
        let pid = std::process::id();
        let sequence = RUN_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let id = format!("run-{now}-{pid}-{sequence}");
        let directory = runs.join(&id);
        let staging = runs.join(format!(".{id}.tmp"));
        Self::create_private_dir(&staging)?;

        let initialized = (|| {
            Self::atomic_json(&staging.join("inputs.json"), inputs)?;
            Self::append_event(&staging, &json!({"event":"started","at_ms":now,"pid":pid}))?;
            let metadata = json!({
                "schema_version": RUN_METADATA_SCHEMA_VERSION,
                "id": id,
                "name": inputs["run_name"],
                "request_id": inputs["request_id"],
                "kept": false,
                "operation": inputs["operation"],
                "training_mode": inputs["training_mode"],
                "requested_steps": inputs["steps"],
                "status": "running",
                "started_at_ms": now,
                "pid": pid,
                "supervisor": null,
                "parent_run_id": inputs["parent_run_id"]
                    .as_str()
                    .or_else(|| inputs["repeat_of"].as_str()),
                "inputs_file": "inputs.json"
            });
            Self::atomic_json(&staging.join("run.json"), &metadata)?;
            Self::create_private_dir(&staging.join("checkpoints"))?;
            fs::rename(&staging, &directory).map_err(|e| e.to_string())?;
            File::open(&runs)
                .and_then(|directory| directory.sync_all())
                .map_err(|e| e.to_string())
        })();
        if let Err(error) = initialized {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        Ok(id)
    }

    fn finish(&self, root: &Path, id: &str, report: &Value) -> Result<(), String> {
        let runs = Self::roots(root, false)?;
        let _activity = Self::activity_lock(&runs)?;
        let directory = Self::directory(root, id)?;
        let _lock = Self::lock(&directory)?;
        let metadata_path = directory.join("run.json");
        let mut metadata = Self::read_run_metadata(&metadata_path)?;
        if metadata["id"] != id
            || !matches!(metadata["status"].as_str(), Some("running" | "stopping"))
        {
            return Err("运行记录身份不匹配或已结束，拒绝覆盖。".into());
        }
        let status = if report["status"] == "success" {
            "completed"
        } else if matches!(
            report["diagnostics"][0]["code"].as_str(),
            Some("USER_INTERRUPTED" | "RUN_PROCESS_LOST")
        ) {
            "interrupted"
        } else {
            "failed"
        };
        let artifact_manifest = super::artifacts::build_run_manifest(
            &directory, id, status, report,
        )
        .unwrap_or_else(|error| {
            json!({
                "schema_version":2,
                "run_id":id,
                "status":"incomplete",
                "recorded_at_ms":Self::now_ms(),
                "artifacts":[],
                "registration_error":error,
                "unregistered_files_protected":true
            })
        });
        Self::atomic_json(&directory.join("artifacts.json"), &artifact_manifest)?;
        let now = Self::now_ms();
        let latest_checkpoint = Self::read_json(&directory.join("checkpoints/latest.json")).ok();
        let inputs = Self::read_json(&directory.join("inputs.json")).unwrap_or(Value::Null);
        let summary = json!({
            "training_mode": report["data"]["training_mode"].as_str()
                .or_else(|| inputs["training_mode"].as_str()),
            "requested_steps": report["data"]["requested_steps"].as_u64()
                .or_else(|| inputs["steps"].as_u64()),
            "completed_steps": report["data"]["completed_steps"].as_u64()
                .or_else(|| latest_checkpoint.as_ref().and_then(|value| value["completed_steps"].as_u64())),
            "mean_loss": report["data"]["mean_loss"],
            "optimizer_updates": report["data"]["optimizer_updates"],
            "checkpoint_saved": report["data"]["checkpoint_saved"] == true || latest_checkpoint.is_some(),
            "checkpoint_completed_steps": report["data"]["checkpoint_completed_steps"].as_u64()
                .or_else(|| latest_checkpoint.as_ref().and_then(|value| value["completed_steps"].as_u64())),
            "resume_checkpoint": report["data"]["resume_checkpoint"]
        });
        Self::atomic_json(&directory.join("report.json"), report)?;
        if report["data"]["checkpoint_saved"] == true {
            Self::append_event(
                &directory,
                &json!({
                    "event":"checkpoint_saved",
                    "at_ms":now,
                    "completed_steps":report["data"]["checkpoint_completed_steps"],
                    "sha256":report["data"]["checkpoint"]["sha256"]
                }),
            )?;
        } else if let Some(checkpoint) = latest_checkpoint {
            Self::append_event(
                &directory,
                &json!({
                    "event":"checkpoint_preserved",
                    "at_ms":now,
                    "completed_steps":checkpoint["completed_steps"],
                    "sha256":checkpoint["sha256"]
                }),
            )?;
        }
        Self::append_event(
            &directory,
            &json!({"event":"finished","at_ms":now,"status":status}),
        )?;
        let _ = fs::remove_file(directory.join("stop.request"));
        metadata["status"] = json!(status);
        metadata["finished_at_ms"] = json!(now);
        metadata["summary"] = summary;
        Self::atomic_json(&metadata_path, &metadata)
    }

    fn checkpoint_dir(&self, root: &Path, id: &str, create: bool) -> Result<PathBuf, String> {
        let directory = Self::directory(root, id)?;
        Self::checkpoints(&directory, create)
    }

    fn run_directory(&self, root: &Path, id: &str) -> Result<PathBuf, String> {
        Self::directory(root, id)
    }

    fn set_supervisor(&self, root: &Path, id: &str, pid: u32, token: &str) -> Result<(), String> {
        let directory = Self::directory(root, id)?;
        let _lock = Self::lock(&directory)?;
        let metadata_path = directory.join("run.json");
        let mut metadata = Self::read_run_metadata(&metadata_path)?;
        if metadata["id"] != id || metadata["status"] != "running" {
            return Err("运行已结束或身份不匹配，拒绝登记监督进程。".into());
        }
        if metadata["supervisor"].is_object() {
            return Err("该运行已经登记了监督进程。".into());
        }
        metadata["supervisor"] = json!({
            "pid":pid,
            "start_token":token,
            "kind":if pid == std::process::id() { "foreground" } else { "detached" }
        });
        Self::atomic_json(&metadata_path, &metadata)?;
        Self::append_event(
            &directory,
            &json!({"event":"supervisor_started","at_ms":Self::now_ms(),"pid":pid}),
        )
    }

    fn record_heartbeat(&self, root: &Path, id: &str, pid: u32, token: &str) -> Result<(), String> {
        let directory = Self::directory(root, id)?;
        let _lock = Self::lock(&directory)?;
        let metadata = Self::read_run_metadata(&directory.join("run.json"))?;
        if metadata["id"] != id
            || !matches!(metadata["status"].as_str(), Some("running" | "stopping"))
            || metadata["supervisor"]["pid"] != pid
            || metadata["supervisor"]["start_token"] != token
            || metadata["supervisor"]["kind"] != "detached"
        {
            return Err("运行身份或状态已变化，拒绝写入监督心跳。".into());
        }
        Self::atomic_json(
            &directory.join("heartbeat.json"),
            &json!({"pid":pid,"start_token":token,"at_ms":Self::now_ms()}),
        )
    }

    fn request_stop(&self, root: &Path, id: &str) -> Result<(), String> {
        let directory = Self::directory(root, id)?;
        let _lock = Self::lock(&directory)?;
        let metadata_path = directory.join("run.json");
        let mut metadata = Self::read_run_metadata(&metadata_path)?;
        if metadata["id"] != id || metadata["status"] != "running" {
            return Err(format!(
                "运行 {id} 当前状态为 {}，无法请求停止。",
                metadata["status"].as_str().unwrap_or("unknown")
            ));
        }
        if !metadata["supervisor"].is_object() {
            return Err("该运行没有可验证的后台监督进程，不能安全发送停止请求。".into());
        }
        let stop_path = directory.join("stop.request");
        let mut request = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&stop_path)
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    "该运行已经收到停止请求。".to_owned()
                } else {
                    error.to_string()
                }
            })?;
        let write_result = (|| {
            serde_json::to_writer(
                &mut request,
                &json!({"requested_at_ms":Self::now_ms(),"requested_by_pid":std::process::id()}),
            )
            .map_err(|error| error.to_string())?;
            request
                .write_all(b"\n")
                .map_err(|error| error.to_string())?;
            request.sync_all().map_err(|error| error.to_string())
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&stop_path);
            return Err(error);
        }
        metadata["status"] = json!("stopping");
        metadata["stop_requested_at_ms"] = json!(Self::now_ms());
        if let Err(error) = Self::append_event(
            &directory,
            &json!({"event":"stop_requested","at_ms":Self::now_ms()}),
        ) {
            let _ = fs::remove_file(&stop_path);
            return Err(error);
        }
        if let Err(error) = Self::atomic_json(&metadata_path, &metadata) {
            let _ = fs::remove_file(stop_path);
            return Err(error);
        }
        Ok(())
    }

    fn progress(&self, root: &Path, id: &str) -> Result<Vec<Value>, String> {
        let directory = Self::directory(root, id)?;
        let path = directory.join("progress.jsonl");
        let Some(bytes) = Self::tail_bytes(&path, 2 * 1024 * 1024)? else {
            return Ok(Vec::new());
        };
        let contents = String::from_utf8_lossy(&bytes);
        let mut records = contents
            .lines()
            .skip(usize::from(bytes.len() == 2 * 1024 * 1024))
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect::<Vec<Value>>();
        if records.len() > 2_000 {
            records.drain(..records.len() - 2_000);
        }
        Ok(records)
    }

    fn log_tail(&self, root: &Path, id: &str, max_bytes: usize) -> Result<String, String> {
        let directory = Self::directory(root, id)?;
        let path = directory.join("worker.log");
        Ok(Self::tail_bytes(&path, max_bytes)?
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default())
    }

    fn storage(&self, root: &Path, id: Option<&str>) -> Result<Value, String> {
        let ids = if let Some(id) = id {
            vec![id.to_owned()]
        } else {
            self.list(root)?
                .into_iter()
                .filter_map(|record| record["id"].as_str().map(str::to_owned))
                .collect()
        };
        let mut seen_files = std::collections::HashSet::new();
        let runs = ids
            .iter()
            .map(|id| Self::storage_entry(root, id, &mut seen_files))
            .collect::<Result<Vec<_>, _>>()?;
        let totals = runs.iter().fold([0_u64; 6], |mut sum, run| {
            sum[0] = sum[0].saturating_add(run["total_bytes"].as_u64().unwrap_or(0));
            for (index, key) in ["metadata", "logs", "progress", "checkpoints", "artifacts"]
                .iter()
                .enumerate()
            {
                sum[index + 1] =
                    sum[index + 1].saturating_add(run["categories"][*key].as_u64().unwrap_or(0));
            }
            sum
        });
        let deduplicated_file_count = runs
            .iter()
            .map(|run| run["deduplicated_file_count"].as_u64().unwrap_or(0))
            .sum::<u64>();
        let run_root = Self::roots(root, false)?;
        let filesystem_path = if run_root.is_dir() {
            run_root.as_path()
        } else {
            root
        };
        let filesystem_space = match (
            fs2::available_space(filesystem_path),
            fs2::total_space(filesystem_path),
        ) {
            (Ok(available_bytes), Ok(total_bytes)) => json!({
                "status":"available",
                "available_bytes":available_bytes,
                "total_bytes":total_bytes,
                "scope":"filesystem containing the project run directory"
            }),
            (available, total) => json!({
                "status":"unknown",
                "reason":available.err().or_else(|| total.err()).map(|error| error.to_string()),
                "scope":"filesystem containing the project run directory"
            }),
        };
        Ok(json!({
            "runs":runs,
            "total_bytes":totals[0],
            "deduplicated_file_count":deduplicated_file_count,
            "filesystem_space":filesystem_space,
            "categories":{
                "metadata":totals[1],
                "logs":totals[2],
                "progress":totals[3],
                "checkpoints":totals[4],
                "artifacts":totals[5]
            },
            "scope":"only files inside .model-maven/runs; source code and datasets are not copied into run records"
        }))
    }

    fn rename(&self, root: &Path, id: &str, name: &str) -> Result<(), String> {
        let name = Self::validate_name(name)?;
        let directory = Self::directory(root, id)?;
        let _lock = Self::lock(&directory)?;
        let path = directory.join("run.json");
        let mut metadata = Self::read_run_metadata(&path)?;
        if metadata["id"] != id {
            return Err("运行记录目录与内部 ID 不匹配。".into());
        }
        let previous = metadata["name"].as_str().unwrap_or("").to_owned();
        if previous == name {
            return Ok(());
        }
        metadata["name"] = json!(name);
        Self::atomic_json(&path, &metadata)?;
        Self::append_event(
            &directory,
            &json!({"event":"renamed","at_ms":Self::now_ms(),"from":previous,"to":name}),
        )
    }

    fn set_kept(&self, root: &Path, id: &str, kept: bool) -> Result<(), String> {
        let directory = Self::directory(root, id)?;
        let _lock = Self::lock(&directory)?;
        let path = directory.join("run.json");
        let mut metadata = Self::read_run_metadata(&path)?;
        if metadata["id"] != id {
            return Err("运行记录目录与内部 ID 不匹配。".into());
        }
        if metadata["kept"].as_bool().unwrap_or(false) == kept {
            return Ok(());
        }
        metadata["kept"] = json!(kept);
        Self::atomic_json(&path, &metadata)?;
        Self::append_event(
            &directory,
            &json!({"event":if kept {"kept"} else {"unkept"},"at_ms":Self::now_ms()}),
        )
    }

    fn plan_clean(
        &self,
        root: &Path,
        run_id: Option<&str>,
        include_logs: bool,
        include_progress: bool,
        include_checkpoints: bool,
        artifact_id: Option<&str>,
    ) -> Result<Value, String> {
        if include_checkpoints && run_id.is_none() {
            return Err("释放 checkpoint 必须明确指定一个 run。".into());
        }
        let ids = if let Some(id) = run_id {
            vec![id.to_owned()]
        } else {
            self.list(root)?
                .into_iter()
                .filter_map(|record| record["id"].as_str().map(str::to_owned))
                .collect()
        };
        let project_consumers = if include_checkpoints {
            Some(super::artifacts::project_artifact_consumers(root))
        } else {
            None
        };
        let mut items = Vec::new();
        let mut skipped = Vec::new();
        for id in ids {
            let directory = Self::directory(root, &id)?;
            let metadata = Self::read_run_metadata(&directory.join("run.json"))?;
            let status = metadata["status"].as_str().unwrap_or("unknown");
            if !matches!(status, "completed" | "failed" | "interrupted") {
                skipped.push(json!({"run_id":id,"reason":format!("状态 {status} 受保护") }));
                continue;
            }
            if metadata["kept"] == true {
                skipped.push(json!({"run_id":id,"reason":"运行已标记保留"}));
                continue;
            }
            for (include, filename, kind, reason) in [
                (include_logs, "worker.log", "run_log", "清理运行日志"),
                (
                    include_progress,
                    "progress.jsonl",
                    "run_progress",
                    "清理训练进度记录",
                ),
            ] {
                if !include {
                    continue;
                }
                let path = directory.join(filename);
                let file_meta = match fs::symlink_metadata(&path) {
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(error) => return Err(error.to_string()),
                    Ok(meta) if meta.file_type().is_symlink() => {
                        skipped
                            .push(json!({"run_id":id,"path":filename,"reason":"符号链接已跳过"}));
                        continue;
                    }
                    Ok(meta) if meta.is_file() => meta,
                    Ok(_) => {
                        skipped
                            .push(json!({"run_id":id,"path":filename,"reason":"目标不是普通文件"}));
                        continue;
                    }
                };
                if Self::has_shared_links(&file_meta) {
                    skipped.push(json!({"run_id":id,"path":filename,"reason":"文件存在硬链接共享引用，保持保护"}));
                    continue;
                }
                items.push(json!({
                    "run_id":id,
                    "path":filename,
                    "kind":kind,
                    "size_bytes":file_meta.len(),
                    "sha256":Self::file_sha256(&path)?,
                    "release_checkpoint":false,
                    "reason":reason
                }));
            }
            let checkpoint_path = directory.join("checkpoints");
            match fs::symlink_metadata(&checkpoint_path) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.to_string()),
                Ok(meta) if meta.file_type().is_symlink() => {
                    skipped.push(json!({"run_id":id,"reason":"checkpoint 目录是符号链接"}));
                    continue;
                }
                Ok(meta) if !meta.is_dir() => {
                    skipped.push(json!({"run_id":id,"reason":"checkpoint 路径不是目录"}));
                    continue;
                }
                Ok(_) => {}
            }
            let checkpoint_dir = Self::checkpoints(&directory, false)?;
            let latest_path = checkpoint_dir.join("latest.json");
            let latest = match fs::symlink_metadata(&latest_path) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.to_string()),
                Ok(meta) if meta.file_type().is_symlink() => {
                    skipped.push(json!({"run_id":id,"reason":"latest.json 是符号链接"}));
                    None
                }
                Ok(_) => Some(Self::read_json(&latest_path)?),
            };
            let candidate_latest_file = latest.as_ref().and_then(|manifest| {
                let filename = manifest["file"].as_str()?;
                Self::validate_checkpoint_filename(filename)
                    .is_ok()
                    .then(|| filename.to_owned())
            });
            let latest_file = candidate_latest_file.filter(|filename| {
                let path = checkpoint_dir.join(filename);
                let Ok(meta) = fs::symlink_metadata(&path) else {
                    return false;
                };
                !meta.file_type().is_symlink()
                    && meta.is_file()
                    && latest
                        .as_ref()
                        .and_then(|manifest| manifest["sha256"].as_str())
                        .is_some_and(|expected| {
                            Self::file_sha256(&path).is_ok_and(|actual| actual == expected)
                        })
            });
            if latest.is_some() && latest_file.is_none() {
                skipped.push(json!({"run_id":id,"reason":"latest.json 指向缺失、无效或校验失败的权重；所有 checkpoint 权重保持受保护"}));
            }
            let mut entries = fs::read_dir(&checkpoint_dir)
                .map_err(|error| error.to_string())?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?;
            entries.sort_by_key(|entry| entry.file_name());
            for entry in entries {
                let file_name = entry.file_name().to_string_lossy().into_owned();
                let path = entry.path();
                let file_type = entry.file_type().map_err(|error| error.to_string())?;
                if file_type.is_symlink() {
                    skipped.push(json!({"run_id":id,"path":file_name,"reason":"符号链接已跳过"}));
                    continue;
                }
                if !file_type.is_file() || Some(file_name.as_str()) == latest_file.as_deref() {
                    continue;
                }
                let temporary = file_name.starts_with(".checkpoint-");
                let old_checkpoint =
                    latest_file.is_some() && Self::validate_checkpoint_filename(&file_name).is_ok();
                if Self::validate_checkpoint_filename(&file_name).is_ok() && latest_file.is_none() {
                    skipped.push(json!({"run_id":id,"path":file_name,"reason":"没有有效 latest.json 引用，权重保持受保护"}));
                    continue;
                }
                if !temporary && !old_checkpoint {
                    continue;
                }
                let size = entry.metadata().map_err(|error| error.to_string())?.len();
                let digest = Self::file_sha256(&path)?;
                items.push(json!({
                    "run_id":id,
                    "path":file_name,
                    "kind":if temporary {"checkpoint_temporary"} else {"unreferenced_checkpoint"},
                    "size_bytes":size,
                    "sha256":digest,
                    "release_checkpoint":false,
                    "reason":if temporary {"崩溃遗留的 checkpoint 临时文件"} else {"不被 latest.json 引用的旧 checkpoint"}
                }));
            }
            if include_checkpoints {
                match project_consumers
                    .as_ref()
                    .expect("loaded for checkpoint cleanup")
                {
                    Ok(consumers) => {
                        let (managed_items, managed_skipped) =
                            super::artifacts::plan_managed_checkpoint_releases(
                                &directory,
                                &id,
                                artifact_id.filter(|selected| *selected != "training-checkpoint"),
                                consumers,
                            );
                        items.extend(managed_items);
                        skipped.extend(managed_skipped);
                    }
                    Err(error) => skipped.push(json!({
                        "run_id":id,
                        "reason":format!("{error} 项目产物图不完整。")
                    })),
                }
            }
            if include_checkpoints {
                let graph_error = project_consumers
                    .as_ref()
                    .and_then(|result| result.as_ref().err());
                let checkpoint_consumers = project_consumers
                    .as_ref()
                    .and_then(|result| result.as_ref().ok())
                    .and_then(|consumers| {
                        consumers.get(&(id.clone(), "training-checkpoint".to_owned()))
                    });
                if let Some(error) = graph_error {
                    skipped.push(json!({
                        "run_id":id,
                        "reason":format!("{error} 项目产物图不完整，latest checkpoint 保持保护。")
                    }));
                } else if let Some(consumers) =
                    checkpoint_consumers.filter(|items| !items.is_empty())
                {
                    skipped.push(json!({
                        "run_id":id,
                        "artifact_id":"training-checkpoint",
                        "consumers":consumers,
                        "reason":"项目级 typed consumer 仍依赖此 checkpoint，保持保护"
                    }));
                } else if metadata["checkpoint_released"] == true {
                    skipped.push(json!({"run_id":id,"reason":"checkpoint 已释放"}));
                } else if artifact_id.is_none_or(|selected| selected == "training-checkpoint")
                    && let Some(file_name) = latest_file.as_deref()
                {
                    let path = checkpoint_dir.join(file_name);
                    let meta = match fs::symlink_metadata(&path) {
                        Ok(meta) if meta.file_type().is_symlink() => {
                            skipped.push(json!({"run_id":id,"path":file_name,"reason":"latest checkpoint 是符号链接"}));
                            continue;
                        }
                        Ok(meta) if meta.is_file() => meta,
                        Ok(_) => {
                            skipped.push(json!({"run_id":id,"path":file_name,"reason":"latest checkpoint 不是普通文件"}));
                            continue;
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                            skipped.push(json!({"run_id":id,"path":file_name,"reason":"latest checkpoint 文件已缺失"}));
                            continue;
                        }
                        Err(error) => return Err(error.to_string()),
                    };
                    if Self::has_shared_links(&meta) {
                        skipped.push(json!({"run_id":id,"path":file_name,"reason":"latest checkpoint 存在硬链接共享引用，保持保护"}));
                        continue;
                    }
                    items.push(json!({
                        "run_id":id,
                        "path":file_name,
                        "kind":"latest_checkpoint_release",
                        "size_bytes":meta.len(),
                        "sha256":Self::file_sha256(&path)?,
                        "release_checkpoint":true,
                        "expected_latest_file":latest_file,
                        "reason":"移除该文件会让此运行失去权重恢复和续训能力"
                    }));
                } else if artifact_id.is_none_or(|selected| selected == "training-checkpoint") {
                    skipped.push(json!({"run_id":id,"reason":"没有 latest checkpoint 可释放"}));
                }
            }
        }
        for (index, item) in items.iter_mut().enumerate() {
            item["item_id"] = json!(format!("item-{:04}", index + 1));
        }
        let plan_id = format!(
            "clean-{}-{}-{}",
            Self::now_ms(),
            std::process::id(),
            CLEAN_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        );
        let now = Self::now_ms();
        let total_bytes = items
            .iter()
            .map(|item| item["size_bytes"].as_u64().unwrap_or(0))
            .fold(0_u64, u64::saturating_add);
        let mut plan = json!({
            "schema_version":1,
            "id":plan_id.clone(),
            "status":"planned",
            "created_at_ms":now,
            "expires_at_ms":now.saturating_add(7 * 24 * 60 * 60 * 1000),
            "run_id":run_id,
            "include_logs":include_logs,
            "include_progress":include_progress,
            "include_checkpoints":include_checkpoints,
            "artifact_id":artifact_id,
            "items":items,
            "skipped":skipped,
            "total_bytes":total_bytes
        });
        let plan_hash = Self::json_sha256(&plan)?;
        plan["plan_sha256"] = json!(plan_hash);
        let plans = Self::clean_plans(root, true)?;
        Self::atomic_json(&plans.join(format!("{plan_id}.json")), &plan)?;
        Ok(plan)
    }
    fn apply_clean(
        &self,
        root: &Path,
        plan_id: &str,
        release_checkpoints: bool,
    ) -> Result<Value, String> {
        Self::validate_plan_id(plan_id)?;
        let runs_root = Self::roots(root, false)?;
        let _activity = Self::activity_lock(&runs_root)?;
        let plans = Self::clean_plans(root, false)?;
        let _plan_lock = Self::lock(&plans)?;
        let plan_path = plans.join(format!("{plan_id}.json"));
        let mut plan = Self::read_json(&plan_path)?;
        if plan["id"] != plan_id || plan["schema_version"] != 1 || plan["status"] != "planned" {
            return Err("清理计划身份无效、已使用或格式不支持。".into());
        }
        let expected_hash = plan["plan_sha256"]
            .as_str()
            .ok_or("清理计划缺少完整性摘要。")?;
        let mut unsigned_plan = plan.clone();
        unsigned_plan
            .as_object_mut()
            .ok_or("清理计划格式无效。")?
            .remove("plan_sha256");
        if Self::json_sha256(&unsigned_plan)? != expected_hash {
            return Err("清理计划内容已变化；请重新预览。".into());
        }
        if plan["expires_at_ms"]
            .as_u64()
            .is_none_or(|expires| expires < Self::now_ms())
        {
            return Err("清理计划已过期；请重新预览当前文件。".into());
        }
        let items = plan["items"]
            .as_array()
            .ok_or("清理计划缺少候选文件。")?
            .clone();
        let contains_release = items.iter().any(|item| item["release_checkpoint"] == true);
        if contains_release && !release_checkpoints {
            return Err("计划包含恢复权重；执行时必须显式提供 --release-checkpoints。".into());
        }
        if !contains_release && release_checkpoints {
            return Err("该计划没有包含恢复权重；请不要传入 --release-checkpoints。".into());
        }
        let project_consumers = if contains_release {
            Some(super::artifacts::project_artifact_consumers(root))
        } else {
            None
        };
        let mut applied = Vec::new();
        let mut skipped = Vec::new();
        let mut warnings = Vec::new();
        for item in items {
            let Some(id) = item["run_id"].as_str() else {
                skipped.push(json!({"item_id":item["item_id"],"reason":"计划项目缺少 run_id"}));
                continue;
            };
            if item["kind"] == "managed_checkpoint_release" {
                let consumers = match project_consumers.as_ref().expect("release graph loaded") {
                    Ok(consumers) => consumers,
                    Err(error) => {
                        skipped.push(json!({"item_id":item["item_id"],"reason":format!("{error} 项目产物图不完整，checkpoint 保持保护。")}));
                        continue;
                    }
                };
                if consumers
                    .get(&(
                        id.to_owned(),
                        item["artifact_id"].as_str().unwrap_or("").to_owned(),
                    ))
                    .is_some_and(|consumers| !consumers.is_empty())
                {
                    skipped.push(json!({"item_id":item["item_id"],"reason":"预览后出现 typed consumer；checkpoint 保持保护。"}));
                    continue;
                }
                match Self::apply_managed_checkpoint_release(root, &item) {
                    Ok(result) => applied.push(result),
                    Err(error) => skipped.push(json!({"item_id":item["item_id"],"reason":error})),
                }
                continue;
            }
            let Some(filename) = item["path"].as_str() else {
                skipped.push(json!({"item_id":item["item_id"],"reason":"计划项目缺少路径"}));
                continue;
            };
            let mut components = Path::new(filename).components();
            if !matches!(components.next(), Some(std::path::Component::Normal(_)))
                || components.next().is_some()
            {
                skipped.push(json!({"item_id":item["item_id"],"reason":"计划路径越界"}));
                continue;
            }
            let release_checkpoint = item["release_checkpoint"] == true;
            if release_checkpoint {
                match project_consumers.as_ref().expect("release graph loaded") {
                    Ok(consumers) => {
                        if consumers
                            .get(&(id.to_owned(), "training-checkpoint".to_owned()))
                            .is_some_and(|consumers| !consumers.is_empty())
                        {
                            skipped.push(json!({"item_id":item["item_id"],"reason":"预览后出现 typed consumer；latest checkpoint 保持保护。"}));
                            continue;
                        }
                    }
                    Err(error) => {
                        skipped.push(json!({"item_id":item["item_id"],"reason":format!("{error} 项目产物图不完整，latest checkpoint 保持保护。")}));
                        continue;
                    }
                }
            }
            let allowed_log = filename == "worker.log" && item["kind"] == "run_log";
            let allowed_progress = filename == "progress.jsonl" && item["kind"] == "run_progress";
            let allowed_temporary =
                filename.starts_with(".checkpoint-") && item["kind"] == "checkpoint_temporary";
            let allowed_old_checkpoint = Self::validate_checkpoint_filename(filename).is_ok()
                && item["kind"] == "unreferenced_checkpoint";
            let allowed_release = release_checkpoint
                && item["kind"] == "latest_checkpoint_release"
                && Self::validate_checkpoint_filename(filename).is_ok()
                && item["expected_latest_file"].as_str() == Some(filename);
            if !(allowed_log
                || allowed_progress
                || allowed_temporary
                || allowed_old_checkpoint
                || allowed_release)
            {
                skipped.push(json!({"item_id":item["item_id"],"reason":"计划项目类型或 checkpoint 文件名不安全"}));
                continue;
            }
            let directory = Self::directory(root, id)?;
            let _lock = Self::lock(&directory)?;
            let metadata_path = directory.join("run.json");
            let mut metadata = Self::read_run_metadata(&metadata_path)?;
            let status = metadata["status"].as_str().unwrap_or("unknown");
            if metadata["id"] != id || !matches!(status, "completed" | "failed" | "interrupted") {
                skipped.push(json!({"item_id":item["item_id"],"reason":format!("运行状态 {status} 已变化或受保护") }));
                continue;
            }
            if metadata["kept"] == true {
                skipped.push(json!({"item_id":item["item_id"],"reason":"运行已标记保留"}));
                continue;
            }
            if allowed_log || allowed_progress {
                let path = directory.join(filename);
                let file_meta = match fs::symlink_metadata(&path) {
                    Ok(meta) if meta.file_type().is_symlink() => {
                        skipped
                            .push(json!({"item_id":item["item_id"],"reason":"目标变成符号链接"}));
                        continue;
                    }
                    Ok(meta) if meta.is_file() => meta,
                    Ok(_) => {
                        skipped
                            .push(json!({"item_id":item["item_id"],"reason":"目标不再是普通文件"}));
                        continue;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        skipped.push(json!({"item_id":item["item_id"],"reason":"目标已不存在"}));
                        continue;
                    }
                    Err(error) => return Err(error.to_string()),
                };
                if file_meta.len() != item["size_bytes"].as_u64().unwrap_or(u64::MAX)
                    || Self::file_sha256(&path)? != item["sha256"].as_str().unwrap_or("")
                {
                    skipped
                        .push(json!({"item_id":item["item_id"],"reason":"文件内容与预览时不同"}));
                    continue;
                }
                fs::remove_file(&path).map_err(|error| error.to_string())?;
                if let Err(error) = Self::append_event(
                    &directory,
                    &json!({"event":"run_artifact_cleaned","at_ms":Self::now_ms(),"file":filename,"kind":item["kind"],"size_bytes":file_meta.len()}),
                ) {
                    warnings.push(json!({"item_id":item["item_id"],"reason":format!("文件已删除，但事件日志写入失败：{error}")}));
                }
                applied.push(json!({"item_id":item["item_id"],"run_id":id,"path":filename,"kind":item["kind"],"release_checkpoint":false,"size_bytes":file_meta.len()}));
                continue;
            }
            if release_checkpoint && metadata["checkpoint_released"] == true {
                skipped.push(json!({"item_id":item["item_id"],"reason":"checkpoint 已经释放"}));
                continue;
            }
            let checkpoint_dir = Self::checkpoints(&directory, false)?;
            if release_checkpoint {
                let latest = Self::read_json(&checkpoint_dir.join("latest.json"))?;
                if latest["file"].as_str() != item["expected_latest_file"].as_str()
                    || latest["file"].as_str() != Some(filename)
                    || latest["sha256"].as_str() != item["sha256"].as_str()
                {
                    skipped.push(
                        json!({"item_id":item["item_id"],"reason":"latest checkpoint 指针已变化"}),
                    );
                    continue;
                }
            } else if allowed_old_checkpoint {
                let latest_path = checkpoint_dir.join("latest.json");
                match fs::symlink_metadata(&latest_path) {
                    Ok(meta) if meta.file_type().is_symlink() => {
                        skipped.push(json!({"item_id":item["item_id"],"reason":"latest.json 是符号链接，无法确认文件引用"}));
                        continue;
                    }
                    Ok(meta) if meta.is_file() => meta,
                    Ok(_) => {
                        skipped.push(
                            json!({"item_id":item["item_id"],"reason":"latest.json 不是普通文件"}),
                        );
                        continue;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        skipped.push(json!({"item_id":item["item_id"],"reason":"latest.json 已缺失，无法确认权重引用"}));
                        continue;
                    }
                    Err(error) => return Err(error.to_string()),
                };
                let latest = Self::read_json(&latest_path)?;
                let Some(latest_filename) = latest["file"].as_str() else {
                    skipped.push(
                        json!({"item_id":item["item_id"],"reason":"latest.json 缺少权重文件名"}),
                    );
                    continue;
                };
                if Self::validate_checkpoint_filename(latest_filename).is_err() {
                    skipped.push(json!({"item_id":item["item_id"],"reason":"latest.json 文件名无效，无法确认权重引用"}));
                    continue;
                }
                let latest_target = checkpoint_dir.join(latest_filename);
                let latest_file_meta = match fs::symlink_metadata(&latest_target) {
                    Ok(meta) if !meta.file_type().is_symlink() && meta.is_file() => meta,
                    _ => {
                        skipped.push(json!({"item_id":item["item_id"],"reason":"latest.json 指向的权重缺失或不安全，无法确认文件引用"}));
                        continue;
                    }
                };
                let latest_hash = latest["sha256"].as_str().unwrap_or("");
                if latest_hash.is_empty()
                    || Self::file_sha256(&latest_target)? != latest_hash
                    || latest_file_meta.len() == 0
                {
                    skipped.push(json!({"item_id":item["item_id"],"reason":"latest.json 权重校验失败，无法确认文件引用"}));
                    continue;
                }
                if latest_filename == filename {
                    skipped.push(json!({"item_id":item["item_id"],"reason":"该 checkpoint 仍被 latest.json 引用"}));
                    continue;
                }
            }
            let path = checkpoint_dir.join(filename);
            let file_meta = match fs::symlink_metadata(&path) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    skipped.push(json!({"item_id":item["item_id"],"reason":"目标变成符号链接"}));
                    continue;
                }
                Ok(meta) if meta.is_file() => meta,
                Ok(_) => {
                    skipped.push(json!({"item_id":item["item_id"],"reason":"目标不再是普通文件"}));
                    continue;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    skipped.push(json!({"item_id":item["item_id"],"reason":"目标已不存在"}));
                    continue;
                }
                Err(error) => return Err(error.to_string()),
            };
            if Self::has_shared_links(&file_meta) {
                skipped.push(
                    json!({"item_id":item["item_id"],"reason":"文件存在硬链接共享引用，保持保护"}),
                );
                continue;
            }
            if file_meta.len() != item["size_bytes"].as_u64().unwrap_or(u64::MAX)
                || Self::file_sha256(&path)? != item["sha256"].as_str().unwrap_or("")
            {
                skipped.push(json!({"item_id":item["item_id"],"reason":"文件内容与预览时不同"}));
                continue;
            }
            if release_checkpoint {
                let now = Self::now_ms();
                let original_metadata = metadata.clone();
                metadata["checkpoint_released"] = json!(true);
                metadata["checkpoint_released_at_ms"] = json!(now);
                Self::atomic_json(&metadata_path, &metadata).map_err(|error| {
                    format!("无法记录 checkpoint 释放状态，权重文件保持不变：{error}")
                })?;
                if let Err(error) = fs::remove_file(&path) {
                    let rollback = Self::atomic_json(&metadata_path, &original_metadata);
                    return Err(match rollback {
                        Ok(()) => format!("无法移除 checkpoint，释放状态已回滚：{error}"),
                        Err(rollback_error) => format!(
                            "无法移除 checkpoint：{error}；释放状态回滚也失败：{rollback_error}"
                        ),
                    });
                }
                if let Err(error) = Self::append_event(
                    &directory,
                    &json!({"event":"checkpoint_released","at_ms":now,"file":filename,"size_bytes":file_meta.len()}),
                ) {
                    warnings.push(json!({"item_id":item["item_id"],"reason":format!("权重已释放，但事件日志写入失败：{error}")}));
                }
            } else {
                fs::remove_file(&path).map_err(|error| error.to_string())?;
                if let Err(error) = Self::append_event(
                    &directory,
                    &json!({"event":"checkpoint_artifact_cleaned","at_ms":Self::now_ms(),"file":filename,"size_bytes":file_meta.len()}),
                ) {
                    warnings.push(json!({"item_id":item["item_id"],"reason":format!("文件已删除，但事件日志写入失败：{error}")}));
                }
            }
            applied.push(json!({"item_id":item["item_id"],"run_id":id,"path":filename,"kind":item["kind"],"release_checkpoint":release_checkpoint,"size_bytes":file_meta.len()}));
        }
        let now = Self::now_ms();
        plan["status"] = json!("applied");
        plan["applied_at_ms"] = json!(now);
        plan["applied_items"] = json!(applied);
        plan["apply_skipped"] = json!(skipped);
        plan["apply_warnings"] = json!(warnings);
        plan.as_object_mut()
            .ok_or("清理计划格式无效。")?
            .remove("plan_sha256");
        let plan_hash = Self::json_sha256(&plan)?;
        plan["plan_sha256"] = json!(plan_hash);
        Self::atomic_json(&plan_path, &plan)?;
        Ok(plan)
    }

    fn latest_checkpoint(&self, root: &Path, id: &str) -> Result<PathBuf, String> {
        let run_directory = Self::directory(root, id)?;
        let _lock = Self::lock(&run_directory)?;
        let metadata = Self::read_run_metadata(&run_directory.join("run.json"))?;
        if metadata["checkpoint_released"] == true {
            return Err("该运行的 checkpoint 已按清理计划释放；当前不能续训或导出权重。".into());
        }
        let directory = self.checkpoint_dir(root, id, false)?;
        let manifest = Self::read_json(&directory.join("latest.json"))?;
        let filename = manifest["file"]
            .as_str()
            .ok_or("checkpoint 清单缺少文件名。")?;
        Self::validate_checkpoint_filename(filename)?;
        let path = directory.join(filename);
        match fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                Err("拒绝读取符号链接 checkpoint。".into())
            }
            Ok(meta) if meta.is_file() => Ok(path),
            Ok(_) => Err("checkpoint 不是普通文件。".into()),
            Err(error) => Err(error.to_string()),
        }
    }

    fn find_request(&self, root: &Path, request_id: &str) -> Result<Value, String> {
        let runs = Self::roots(root, false)?;
        let entries = match fs::read_dir(&runs) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(json!({"matches":[],"complete":true,"unreadable":0,"truncated":false}));
            }
            Err(error) => return Err(error.to_string()),
        };
        let mut matches = Vec::new();
        let mut unreadable = 0_u64;
        let mut truncated = false;
        for entry in entries {
            let entry = entry.map_err(|error| error.to_string())?;
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if Self::validate_id(&name).is_err() {
                continue;
            }
            let kind = entry.file_type().map_err(|error| error.to_string())?;
            if !kind.is_dir() || kind.is_symlink() {
                unreadable += 1;
                continue;
            }
            let metadata_path = entry.path().join("run.json");
            if !fs::symlink_metadata(&metadata_path).is_ok_and(|metadata| {
                metadata.is_file()
                    && !metadata.file_type().is_symlink()
                    && metadata.len() <= 1024 * 1024
            }) {
                unreadable += 1;
                continue;
            }
            let metadata = match Self::read_run_metadata(&metadata_path) {
                Ok(metadata)
                    if metadata["id"].as_str() == Some(&name)
                        && metadata["operation"].as_str().is_some()
                        && metadata["status"].as_str().is_some() =>
                {
                    metadata
                }
                _ => {
                    unreadable += 1;
                    continue;
                }
            };
            if metadata["request_id"].as_str() == Some(request_id) {
                if matches.len() == 64 {
                    truncated = true;
                    continue;
                }
                matches.push(
                    json!({"id":name,"request_id":request_id,"operation":metadata["operation"],
                    "status":metadata["status"],"status_scope":"recorded_not_live"}),
                );
            }
        }
        Ok(
            json!({"matches":matches,"complete":unreadable == 0 && !truncated,
                  "unreadable":unreadable,"truncated":truncated}),
        )
    }

    fn list(&self, root: &Path) -> Result<Vec<Value>, String> {
        let runs = Self::roots(root, false)?;
        if !runs.is_dir() {
            return Ok(Vec::new());
        }
        let mut records = Vec::new();
        for entry in fs::read_dir(runs).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            Self::validate_id(&name)?;
            if entry.file_type().map_err(|e| e.to_string())?.is_symlink() {
                return Err(format!("运行目录含符号链接：{name}"));
            }
            let metadata = Self::read_run_metadata(&entry.path().join("run.json"))?;
            if metadata["id"] != name {
                return Err(format!("运行记录目录与内部 ID 不匹配：{name}"));
            }
            let checkpoints_path = entry.path().join("checkpoints");
            let checkpoint = if metadata["checkpoint_released"] == true {
                None
            } else {
                match fs::symlink_metadata(&checkpoints_path) {
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => return Err(error.to_string()),
                    Ok(meta) if meta.file_type().is_symlink() => {
                        return Err(format!("运行 {name} 的 checkpoint 目录是符号链接。"));
                    }
                    Ok(meta) if !meta.is_dir() => {
                        return Err(format!("运行 {name} 的 checkpoint 路径不是目录。"));
                    }
                    Ok(_) => {
                        let manifest = checkpoints_path.join("latest.json");
                        match fs::symlink_metadata(&manifest) {
                            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                            Err(error) => return Err(error.to_string()),
                            Ok(_) => Some(Self::read_json(&manifest)?),
                        }
                    }
                }
            };
            let heartbeat_path = entry.path().join("heartbeat.json");
            let heartbeat = match fs::symlink_metadata(&heartbeat_path) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.to_string()),
                Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => {
                    return Err(format!("运行 {name} 的监督心跳不是普通文件。"));
                }
                Ok(_) => Some(Self::read_json(&heartbeat_path)?),
            };
            records.push(json!({
                "id": metadata["id"],
                "name": metadata["name"],
                "request_id": metadata["request_id"],
                "kept": metadata["kept"],
                "operation": metadata["operation"],
                "status": metadata["status"],
                "training_mode": metadata["training_mode"],
                "requested_steps": metadata["requested_steps"],
                "supervisor": metadata["supervisor"],
                "heartbeat": heartbeat,
                "parent_run_id": metadata["parent_run_id"],
                "checkpoint": checkpoint,
                "started_at_ms": metadata["started_at_ms"],
                "finished_at_ms": metadata["finished_at_ms"],
                "summary": metadata["summary"]
            }));
        }
        records.sort_by(|left, right| {
            let rank = |status: Option<&str>| match status {
                Some("running" | "stopping") => 0,
                Some("unknown") => 1,
                Some("failed" | "interrupted") => 2,
                Some("completed") => 3,
                _ => 4,
            };
            rank(left["status"].as_str())
                .cmp(&rank(right["status"].as_str()))
                .then_with(|| {
                    right["started_at_ms"]
                        .as_u64()
                        .unwrap_or(0)
                        .cmp(&left["started_at_ms"].as_u64().unwrap_or(0))
                })
                .then_with(|| right["id"].as_str().cmp(&left["id"].as_str()))
        });
        Ok(records)
    }

    fn show(&self, root: &Path, id: &str) -> Result<Value, String> {
        let directory = Self::directory(root, id)?;
        let _lock = Self::lock(&directory)?;
        let metadata = Self::read_run_metadata(&directory.join("run.json"))?;
        if metadata["id"] != id {
            return Err("运行记录目录与内部 ID 不匹配。".into());
        }
        let inputs_file = metadata["inputs_file"]
            .as_str()
            .ok_or("运行记录缺少输入快照路径。")?;
        if inputs_file != "inputs.json" {
            return Err("运行记录输入快照路径无效。".into());
        }
        let inputs = Self::read_json(&directory.join(inputs_file))?;
        let heartbeat_path = directory.join("heartbeat.json");
        let heartbeat = match fs::symlink_metadata(&heartbeat_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.to_string()),
            Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => {
                return Err("监督心跳不是普通文件。".into());
            }
            Ok(_) => Some(Self::read_json(&heartbeat_path)?),
        };
        let report_path = directory.join("report.json");
        let report = if report_path.exists() {
            Some(Self::read_json(&report_path)?)
        } else {
            None
        };
        let checkpoint_path = directory.join("checkpoints");
        let checkpoint = if metadata["checkpoint_released"] == true {
            None
        } else {
            match fs::symlink_metadata(&checkpoint_path) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(error.to_string()),
                Ok(meta) if meta.file_type().is_symlink() => {
                    return Err("运行记录的 checkpoint 目录是符号链接。".into());
                }
                Ok(meta) if !meta.is_dir() => {
                    return Err("运行记录的 checkpoint 路径不是目录。".into());
                }
                Ok(_) => {
                    let manifest = checkpoint_path.join("latest.json");
                    match fs::symlink_metadata(&manifest) {
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                        Err(error) => return Err(error.to_string()),
                        Ok(_) => Some(Self::read_json(&manifest)?),
                    }
                }
            }
        };
        let artifact_manifest_path = directory.join("artifacts.json");
        let artifact_manifest = match fs::symlink_metadata(&artifact_manifest_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.to_string()),
            Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => {
                return Err("产物清单不是普通文件。".into());
            }
            Ok(_) => {
                let manifest = Self::read_json(&artifact_manifest_path)?;
                super::artifacts::validate_manifest(&manifest, id)?;
                Some(manifest)
            }
        };
        let events = Self::read_events(&directory)?;
        Ok(json!({
            "record": metadata,
            "inputs": inputs,
            "heartbeat": heartbeat,
            "events": events,
            "report": report,
            "checkpoint": checkpoint,
            "artifacts": artifact_manifest
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{ARTIFACT_RELEASE_FAIL_AFTER, FileRunStore};
    use crate::ports::RunRepository;
    use serde_json::json;
    use std::{
        fs,
        sync::{Arc, Barrier},
    };

    #[test]
    fn request_identity_prevents_duplicate_runs_after_completion() {
        let root = tempfile::tempdir().unwrap();
        let store = FileRunStore;
        let request = json!({"operation":"train", "request_id":"unique-request-123456"});
        let id = store.create(root.path(), &request).unwrap();
        assert_eq!(
            store.list(root.path()).unwrap()[0]["request_id"],
            "unique-request-123456"
        );
        assert_eq!(
            store.create(root.path(), &request).unwrap_err(),
            format!("RUN_REQUEST_ID_EXISTS:{id}")
        );
        store
            .finish(root.path(), &id, &json!({"status":"success","data":{}}))
            .unwrap();
        assert_eq!(
            store.create(root.path(), &request).unwrap_err(),
            format!("RUN_REQUEST_ID_EXISTS:{id}")
        );
        assert!(
            store
                .create(
                    root.path(),
                    &json!({"operation":"train", "request_id":"another-request-12345"})
                )
                .is_ok()
        );
    }

    #[test]
    fn request_lookup_ignores_checkpoint_damage_but_marks_unreadable_records() {
        let root = tempfile::tempdir().unwrap();
        let store = FileRunStore;
        let id = store
            .create(
                root.path(),
                &json!({"operation":"train", "request_id":"unique-request-123456"}),
            )
            .unwrap();
        store
            .finish(root.path(), &id, &json!({"status":"success","data":{}}))
            .unwrap();
        let runs = root.path().join(".model-maven/runs");
        fs::write(runs.join(&id).join("checkpoints/latest.json"), b"broken").unwrap();
        assert!(store.list(root.path()).is_err());
        let result = store
            .find_request(root.path(), "unique-request-123456")
            .unwrap();
        assert_eq!(result["complete"], true);
        assert_eq!(result["matches"][0]["id"], id);
        fs::create_dir(runs.join("run-123-other")).unwrap();
        fs::write(runs.join("run-123-other/run.json"), b"broken").unwrap();
        let result = store
            .find_request(root.path(), "unique-request-123456")
            .unwrap();
        assert_eq!(result["complete"], false);
        assert_eq!(result["unreadable"], 1);
        assert_eq!(result["matches"][0]["id"], id);
        fs::create_dir(runs.join("run-124-large")).unwrap();
        fs::write(
            runs.join("run-124-large/run.json"),
            vec![b' '; 1024 * 1024 + 1],
        )
        .unwrap();
        let result = store
            .find_request(root.path(), "unique-request-123456")
            .unwrap();
        assert_eq!(result["complete"], false);
        assert_eq!(result["unreadable"], 2);
    }

    #[test]
    fn concurrent_identical_request_is_admitted_once() {
        let root = tempfile::tempdir().unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                let root = root.path().to_owned();
                std::thread::spawn(move || {
                    barrier.wait();
                    FileRunStore.create(
                        &root,
                        &json!({"operation":"train", "request_id":"same-request-123456"}),
                    )
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(results.iter().any(|result| {
            result
                .as_ref()
                .is_err_and(|error| error.starts_with("RUN_REQUEST_ID_EXISTS:"))
        }));
        assert_eq!(FileRunStore.list(root.path()).unwrap().len(), 1);
    }

    #[test]
    fn project_admits_only_one_active_run_and_reopens_after_finish() {
        let root = tempfile::tempdir().unwrap();
        let store = FileRunStore;
        let id = store
            .create(
                root.path(),
                &json!({"operation":"train","run_name":"first"}),
            )
            .unwrap();
        assert_eq!(store.list(root.path()).unwrap().len(), 1);
        assert_eq!(
            store.show(root.path(), &id).unwrap()["record"]["status"],
            "running"
        );

        let error = store
            .create(
                root.path(),
                &json!({"operation":"training_probe","run_name":"second"}),
            )
            .unwrap_err();
        assert!(error.contains(&id));
        assert!(error.contains("first"));

        store
            .finish(root.path(), &id, &json!({"status":"success","data":{}}))
            .unwrap();
        assert!(
            store
                .create(
                    root.path(),
                    &json!({"operation":"training_probe","run_name":"second"}),
                )
                .is_ok()
        );
    }

    #[test]
    fn concurrent_run_creation_has_a_single_winner() {
        let root = tempfile::tempdir().unwrap();
        let root = Arc::new(root.path().to_path_buf());
        let barrier = Arc::new(Barrier::new(8));
        let handles = (0..8)
            .map(|index| {
                let root = Arc::clone(&root);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    FileRunStore.create(
                        root.as_path(),
                        &json!({"operation":"train","run_name":format!("run-{index}")}),
                    )
                })
            })
            .collect::<Vec<_>>();
        let results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 7);
    }

    #[test]
    fn unclosed_run_after_process_loss_blocks_ambiguous_relaunch() {
        let root = tempfile::tempdir().unwrap();
        let store = FileRunStore;
        let id = store
            .create(
                root.path(),
                &json!({"operation":"train","run_name":"possibly alive"}),
            )
            .unwrap();
        let error = FileRunStore
            .create(
                root.path(),
                &json!({"operation":"train","run_name":"replacement"}),
            )
            .unwrap_err();
        assert!(error.contains(&id));
        assert!(error.contains("未结束"));
    }

    #[test]
    fn invalid_artifact_declaration_is_visible_and_never_strands_finished_run() {
        let root = tempfile::tempdir().unwrap();
        let store = FileRunStore;
        let id = store
            .create(root.path(), &json!({"operation":"train"}))
            .unwrap();
        store
            .finish(
                root.path(),
                &id,
                &json!({
                    "status":"success",
                    "data":{"managed_artifacts":[{
                        "id":"unsafe","owner":"user-loop","complete":true,
                        "files":["../outside.bin"]
                    }]}
                }),
            )
            .unwrap();
        let record = store.show(root.path(), &id).unwrap();
        assert_eq!(record["record"]["status"], "completed");
        assert_eq!(record["artifacts"]["status"], "incomplete");
        assert_eq!(record["artifacts"]["unregistered_files_protected"], true);
        assert!(
            record["artifacts"]["registration_error"]
                .as_str()
                .unwrap()
                .contains("相对路径")
        );
        assert!(
            store
                .create(root.path(), &json!({"operation":"train"}))
                .is_ok()
        );
    }

    #[test]
    fn fixed_plan_releases_explicit_final_custom_checkpoint_set_and_keeps_run_summary() {
        let root = tempfile::tempdir().unwrap();
        let store = FileRunStore;
        let id = store
            .create(
                root.path(),
                &json!({"operation":"train","run_name":"custom model"}),
            )
            .unwrap();
        let directory = store.run_directory(root.path(), &id).unwrap();
        fs::create_dir(directory.join("artifacts")).unwrap();
        fs::write(directory.join("artifacts/model.bin"), b"model-state").unwrap();
        fs::write(
            directory.join("artifacts/optimizer.bin"),
            b"optimizer-state",
        )
        .unwrap();
        store
            .finish(
                root.path(),
                &id,
                &json!({
                    "status":"success",
                    "data":{
                        "completed_steps":4,
                        "managed_artifacts":[{
                            "id":"full-state",
                            "owner":"project.loop.v1",
                            "kind":"checkpoint",
                            "complete":true,
                            "references":[],
                            "files":["artifacts/model.bin","artifacts/optimizer.bin"]
                        }]
                    }
                }),
            )
            .unwrap();
        let plan = store
            .plan_clean(root.path(), Some(&id), false, false, true, None)
            .unwrap();
        assert_eq!(plan["items"].as_array().unwrap().len(), 1);
        assert_eq!(plan["items"][0]["kind"], "managed_checkpoint_release");
        assert_eq!(plan["items"][0]["files"].as_array().unwrap().len(), 2);
        let plan_id = plan["id"].as_str().unwrap();
        let applied = store.apply_clean(root.path(), plan_id, true).unwrap();
        assert_eq!(
            applied["applied_items"].as_array().unwrap().len(),
            1,
            "{applied:#}"
        );
        assert!(!directory.join("artifacts/model.bin").exists());
        assert!(!directory.join("artifacts/optimizer.bin").exists());
        let shown = store.show(root.path(), &id).unwrap();
        assert_eq!(shown["record"]["status"], "completed");
        assert_eq!(shown["record"]["summary"]["completed_steps"], 4);
        assert_eq!(
            shown["artifacts"]["artifacts"][0]["availability"],
            "released"
        );
        assert_eq!(shown["artifacts"]["artifacts"][0]["complete"], false);
        assert_eq!(
            shown["events"].as_array().unwrap().last().unwrap()["event"],
            "custom_checkpoint_released"
        );
    }

    #[test]
    fn partial_custom_checkpoint_release_is_recorded_and_remaining_file_is_protected() {
        let root = tempfile::tempdir().unwrap();
        let store = FileRunStore;
        let id = store
            .create(root.path(), &json!({"operation":"train"}))
            .unwrap();
        let directory = store.run_directory(root.path(), &id).unwrap();
        fs::create_dir(directory.join("artifacts")).unwrap();
        fs::write(directory.join("artifacts/model.bin"), b"model").unwrap();
        fs::write(directory.join("artifacts/optimizer.bin"), b"optimizer").unwrap();
        store
            .finish(
                root.path(),
                &id,
                &json!({"status":"success","data":{"managed_artifacts":[{
                    "id":"partial-state","owner":"test-loop","kind":"checkpoint",
                    "complete":true,"references":[],
                    "files":["artifacts/model.bin","artifacts/optimizer.bin"]
                }]}}),
            )
            .unwrap();
        let plan = store
            .plan_clean(
                root.path(),
                Some(&id),
                false,
                false,
                true,
                Some("partial-state"),
            )
            .unwrap();
        let plan_id = plan["id"].as_str().unwrap();

        struct ResetReleaseFault;
        impl Drop for ResetReleaseFault {
            fn drop(&mut self) {
                ARTIFACT_RELEASE_FAIL_AFTER.with(|fail_after| fail_after.set(-1));
            }
        }
        let _reset = ResetReleaseFault;
        ARTIFACT_RELEASE_FAIL_AFTER.with(|fail_after| fail_after.set(1));
        let applied = store.apply_clean(root.path(), plan_id, true).unwrap();
        let skipped = &applied["apply_skipped"][0]["reason"];
        assert!(skipped.as_str().unwrap().contains("释放不完整"));
        assert!(!directory.join("artifacts/model.bin").exists());
        assert!(directory.join("artifacts/optimizer.bin").exists());
        let shown = store.show(root.path(), &id).unwrap();
        let artifact = &shown["artifacts"]["artifacts"][0];
        assert_eq!(artifact["availability"], "partial_release");
        assert_eq!(artifact["complete"], false);
        assert_eq!(artifact["files"][0]["available"], false);
        assert_eq!(artifact["files"][1]["available"], true);
        assert_eq!(
            shown["events"].as_array().unwrap().last().unwrap()["event"],
            "custom_checkpoint_release_incomplete"
        );
        let retry = store
            .plan_clean(
                root.path(),
                Some(&id),
                false,
                false,
                true,
                Some("partial-state"),
            )
            .unwrap();
        assert!(retry["items"].as_array().unwrap().is_empty());
    }

    #[test]
    fn detached_supervisor_heartbeat_is_separate_and_identity_scoped() {
        let root = tempfile::tempdir().unwrap();
        let store = FileRunStore;
        let id = store
            .create(
                root.path(),
                &json!({"operation":"train","run_name":"heartbeat"}),
            )
            .unwrap();
        store
            .set_supervisor(root.path(), &id, 42_424_242, "boot:started-at")
            .unwrap();
        store
            .record_heartbeat(root.path(), &id, 42_424_242, "boot:started-at")
            .unwrap();

        let shown = store.show(root.path(), &id).unwrap();
        assert_eq!(shown["heartbeat"]["pid"], 42_424_242);
        assert_eq!(shown["heartbeat"]["start_token"], "boot:started-at");
        assert_eq!(shown["events"].as_array().unwrap().len(), 2);
        assert!(store.list(root.path()).unwrap()[0]["heartbeat"]["at_ms"].is_number());
        assert!(
            store
                .record_heartbeat(root.path(), &id, 42_424_243, "boot:started-at")
                .is_err()
        );
    }

    #[test]
    fn unknown_run_metadata_version_blocks_reads_and_mutations_without_rewriting_record() {
        let root = tempfile::tempdir().unwrap();
        let store = FileRunStore;
        let id = store
            .create(
                root.path(),
                &json!({"operation":"train","run_name":"compatibility"}),
            )
            .unwrap();
        let metadata_path = root
            .path()
            .join(".model-maven/runs")
            .join(&id)
            .join("run.json");
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
        metadata["schema_version"] = json!(2);
        let unsupported_bytes = serde_json::to_vec(&metadata).unwrap();
        std::fs::write(&metadata_path, &unsupported_bytes).unwrap();

        let show_error = store.show(root.path(), &id).unwrap_err();
        assert!(show_error.contains("schema_version=2"));
        assert!(show_error.contains("兼容版本"));
        let list_error = store.list(root.path()).unwrap_err();
        assert!(list_error.contains("schema_version=2"));
        let rename_error = store.rename(root.path(), &id, "new name").unwrap_err();
        assert!(rename_error.contains("schema_version=2"));
        assert_eq!(std::fs::read(metadata_path).unwrap(), unsupported_bytes);
    }

    #[test]
    fn storage_reports_filesystem_capacity_for_the_run_directory() {
        let root = tempfile::tempdir().unwrap();
        let report = FileRunStore.storage(root.path(), None).unwrap();
        assert_eq!(report["filesystem_space"]["status"], "available");
        let available = report["filesystem_space"]["available_bytes"]
            .as_u64()
            .unwrap();
        let total = report["filesystem_space"]["total_bytes"].as_u64().unwrap();
        assert!(total >= available);
        assert!(report["filesystem_space"]["scope"].is_string());
    }
}
