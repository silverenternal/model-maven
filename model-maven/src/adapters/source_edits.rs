use crate::ports::{SourceEditLock, SourceEditRepository};
use fs2::FileExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const STATE_LIMIT: u64 = 4 * 1024 * 1024;
const SOURCE_LIMIT: u64 = 16 * 1024 * 1024;
const BATCH_SOURCE_LIMIT: u64 = 64 * 1024 * 1024;
const HISTORY_LIMIT: usize = 256;

pub struct FileSourceEditStore;

struct ProjectLock {
    _file: File,
}

struct PendingFileWrite {
    file: Value,
    actual: String,
    expected: String,
    target: String,
    contents: Vec<u8>,
}

fn ensure_directory(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(format!("源码编辑目录不能是符号链接：{}", path.display()))
        }
        Ok(metadata) if !metadata.is_dir() => {
            Err(format!("源码编辑路径不是目录：{}", path.display()))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(path).map_err(|error| error.to_string())?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(path, fs::Permissions::from_mode(0o700))
                    .map_err(|error| error.to_string())?;
            }
            Ok(())
        }
        Err(error) => Err(error.to_string()),
    }
}

fn create_private_directory(path: &Path) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn files_digest(files: &[Value], side: &str) -> Result<String, String> {
    let key = format!("{side}_sha256");
    let material: Vec<Value> = files
        .iter()
        .map(|file| json!([file["path"], file[&key]]))
        .collect();
    let encoded = serde_json::to_vec(&material).map_err(|error| error.to_string())?;
    Ok(digest(&encoded))
}

fn checked_source(root: &Path, relative: &str) -> Result<(PathBuf, Vec<u8>), String> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path.components().any(|component| {
            !matches!(component, Component::Normal(_))
                || component.as_os_str().to_string_lossy().contains('\\')
        })
        || relative_path.extension().and_then(|value| value.to_str()) != Some("py")
        || relative_path.starts_with(".model-maven")
    {
        return Err("源码编辑只接受项目内、非隐藏目录中的相对 .py 文件路径。".into());
    }
    let mut current = root.to_path_buf();
    for component in relative_path.components() {
        current.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&current).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() {
            return Err(format!("源码路径不能经过符号链接：{}", current.display()));
        }
    }
    let metadata = fs::metadata(&current).map_err(|error| error.to_string())?;
    if !metadata.is_file() {
        return Err(format!("源码目标不是普通文件：{}", current.display()));
    }
    if metadata.len() > SOURCE_LIMIT {
        return Err(format!(
            "源码文件超过 {} MiB，暂不支持自动编辑。",
            SOURCE_LIMIT / 1024 / 1024
        ));
    }
    let bytes = fs::read(&current).map_err(|error| error.to_string())?;
    Ok((current, bytes))
}

fn store_directory(root: &Path, create: bool) -> Result<Option<PathBuf>, String> {
    let metadata = root.join(".model-maven");
    match fs::symlink_metadata(&metadata) {
        Ok(info) if info.file_type().is_symlink() => {
            return Err(".model-maven 不能是符号链接。".into());
        }
        Ok(info) if !info.is_dir() => return Err(".model-maven 不是目录。".into()),
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !create => return Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(&metadata).map_err(|error| error.to_string())?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&metadata, fs::Permissions::from_mode(0o700))
                    .map_err(|error| error.to_string())?;
            }
        }
        Err(error) => return Err(error.to_string()),
    }
    let store = metadata.join("source-edits");
    if create {
        ensure_directory(&store)?;
        ensure_directory(&store.join("transactions"))?;
    } else {
        match fs::symlink_metadata(&store) {
            Ok(info) if info.file_type().is_symlink() => {
                return Err("源码编辑存储目录不能是符号链接。".into());
            }
            Ok(info) if !info.is_dir() => return Err("源码编辑存储路径不是目录。".into()),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(Some(store))
}

fn read_json(path: &Path) -> Result<Option<Value>, String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("拒绝读取非普通源码编辑记录：{}", path.display()));
    }
    if metadata.len() > STATE_LIMIT {
        return Err("源码编辑记录超过 4 MiB。".into());
    }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| format!("源码编辑记录 JSON 无效：{error}"))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    if fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(format!("拒绝覆盖符号链接：{}", path.display()));
    }
    let parent = path.parent().ok_or("源码编辑记录缺少父目录。")?;
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
    temporary
        .persist(path)
        .map_err(|error| error.error.to_string())?;
    crate::adapters::sync_directory(parent)
        .map_err(|error| error.to_string())
}

fn exchange_paths(left: &Path, right: &Path) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::ffi::OsStrExt;

        let left = CString::new(left.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "源码路径包含 NUL。"))?;
        let right = CString::new(right.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "源码路径包含 NUL。"))?;
        // Atomically return the destination that was displaced to the staged
        // path, where its bytes can be compared with the approved base.
        let result = unsafe {
            libc::syscall(
                libc::SYS_renameat2,
                libc::AT_FDCWD,
                left.as_ptr(),
                libc::AT_FDCWD,
                right.as_ptr(),
                libc::RENAME_EXCHANGE,
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::ffi::OsStrExt;

        let left = CString::new(left.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "源码路径包含 NUL。"))?;
        let right = CString::new(right.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "源码路径包含 NUL。"))?;
        let result = unsafe {
            libc::renameatx_np(
                libc::AT_FDCWD,
                left.as_ptr(),
                libc::AT_FDCWD,
                right.as_ptr(),
                libc::RENAME_SWAP,
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (left, right);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "当前平台没有可用的原子源码交换操作；源码未提交。",
        ))
    }
}

fn preserve_staged_file(mut temporary: tempfile::NamedTempFile, reason: &str) -> String {
    temporary.disable_cleanup(true);
    format!("{reason} 保留的冲突版本：{}", temporary.path().display())
}

fn regular_file_digest(path: &Path) -> Option<String> {
    let metadata = fs::symlink_metadata(path).ok()?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return None;
    }
    fs::read(path).ok().map(|bytes| digest(&bytes))
}

fn write_source(
    root: &Path,
    relative: &str,
    expected: &str,
    contents: &[u8],
) -> Result<String, String> {
    let (path, current) = checked_source(root, relative)?;
    if digest(&current) != expected {
        return Err(format!("源码已变化，拒绝覆盖：{relative}"));
    }
    let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
    let parent = path.parent().ok_or("源码文件缺少父目录。")?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
    temporary
        .write_all(contents)
        .map_err(|error| error.to_string())?;
    temporary
        .as_file()
        .set_permissions(metadata.permissions())
        .map_err(|error| error.to_string())?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    let (_, latest) = checked_source(root, relative)?;
    if digest(&latest) != expected {
        return Err(format!("源码在提交前再次变化，拒绝覆盖：{relative}"));
    }
    let expected_target = digest(contents);
    exchange_paths(&path, temporary.path())
        .map_err(|error| format!("无法安全比较并替换源码（未覆盖现有文件）：{error}"))?;

    // Inspect the exact destination displaced by the atomic exchange. A
    // pathname read before rename cannot close the external-writer race.
    let displaced_metadata = match fs::symlink_metadata(temporary.path()) {
        Ok(metadata) => metadata,
        Err(error) => {
            return Err(preserve_staged_file(
                temporary,
                &format!("无法核验被替换的源码：{error}。"),
            ));
        }
    };
    let displaced_digest = if displaced_metadata.file_type().is_symlink()
        || !displaced_metadata.is_file()
    {
        None
    } else {
        match fs::read(temporary.path()) {
            Ok(bytes) => Some(digest(&bytes)),
            Err(error) => {
                return Err(preserve_staged_file(
                    temporary,
                    &format!("无法核验被替换的源码：{error}。"),
                ));
            }
        }
    };
    if displaced_digest.as_deref() != Some(expected) {
        let current_digest = regular_file_digest(&path);
        if current_digest.as_deref() == Some(expected_target.as_str())
            && exchange_paths(&path, temporary.path()).is_ok()
        {
            // Preserve a newer external replacement if it raced with rollback.
            let moved_digest = regular_file_digest(temporary.path());
            if moved_digest.as_deref() != Some(expected_target.as_str()) {
                if exchange_paths(&path, temporary.path()).is_err() {
                    return Err(preserve_staged_file(
                        temporary,
                        "检测到并发源码修改；回滚时又有外部替换。",
                    ));
                }
                return Err(preserve_staged_file(
                    temporary,
                    "检测到并发源码修改；已恢复外部文件，保留被替换版本。",
                ));
            }
            return Err(format!("源码在原子替换时已变化，已恢复外部版本：{relative}"));
        }
        return Err(preserve_staged_file(
            temporary,
            &format!(
                "源码在原子替换时已变化，目标摘要为 {}。",
                displaced_digest.as_deref().unwrap_or("非普通文件")
            ),
        ));
    }
    crate::adapters::sync_directory(parent)
        .map_err(|error| error.to_string())?;
    let (_, persisted) = checked_source(root, relative)?;
    if digest(&persisted) != expected_target {
        return Err(format!("源码在原子替换后再次变化：{relative}"));
    }
    Ok(expected_target)
}

fn default_state() -> Value {
    json!({"schema_version":1,"cursor":0,"records":[]})
}

fn state_path(store: &Path) -> PathBuf {
    store.join("history.json")
}

fn pending_path(store: &Path) -> PathBuf {
    store.join("pending.json")
}

fn transaction_path(store: &Path, id: &str, side: &str) -> PathBuf {
    store
        .join("transactions")
        .join(id)
        .join(format!("{side}.py"))
}

fn transaction_file_path(store: &Path, id: &str, side: &str, index: usize) -> PathBuf {
    store
        .join("transactions")
        .join(id)
        .join(format!("{side}-{index}.py"))
}

fn read_state(store: &Path) -> Result<Value, String> {
    let Some(state) = read_json(&state_path(store))? else {
        return Ok(default_state());
    };
    if state["schema_version"] != 1
        || state["cursor"]
            .as_u64()
            .is_none_or(|cursor| cursor > state["records"].as_array().map_or(0, Vec::len) as u64)
        || !state["records"].is_array()
    {
        return Err("源码编辑历史版本或内容无效；文件保持不变。".into());
    }
    Ok(state)
}

fn complete_transition(store: &Path, pending: &Value) -> Result<(), String> {
    let mut state = read_state(store)?;
    let current_cursor = state["cursor"]
        .as_u64()
        .ok_or("源码编辑历史 cursor 无效。")? as usize;
    let records = state["records"]
        .as_array_mut()
        .ok_or("源码编辑历史缺少 records。")?;
    let cursor = pending["cursor"]
        .as_u64()
        .ok_or("源码编辑事务缺少 cursor。")? as usize;
    let action = pending["action"]
        .as_str()
        .ok_or("源码编辑事务缺少 action。")?;
    let record = pending
        .get("record")
        .filter(|value| value.is_object())
        .ok_or("源码编辑事务缺少记录。")?
        .clone();
    match action {
        "commit" => {
            if current_cursor == cursor + 1 && records.get(cursor) == Some(&record) {
                // The state file was synced before a crash prevented pending.json removal.
            } else if current_cursor == cursor {
                records.truncate(cursor);
                records.push(record);
                state["cursor"] = json!(cursor + 1);
            } else {
                return Err("源码提交历史与待恢复事务游标冲突；保留 pending.json。".into());
            }
        }
        "undo" => {
            let target = cursor.saturating_sub(1);
            if current_cursor == cursor {
                state["cursor"] = json!(target);
            } else if current_cursor != target {
                return Err("源码撤销历史与待恢复事务游标冲突；保留 pending.json。".into());
            }
        }
        "redo" => {
            let target = cursor + 1;
            if current_cursor == cursor {
                state["cursor"] = json!(target);
            } else if current_cursor != target {
                return Err("源码重做历史与待恢复事务游标冲突；保留 pending.json。".into());
            }
        }
        "rollback" => {
            if current_cursor == cursor + 1 && records.get(cursor) == Some(&record) {
                records.truncate(cursor);
                state["cursor"] = json!(cursor);
            } else if current_cursor != cursor {
                return Err("源码回滚历史与待恢复事务游标冲突；保留 pending.json。".into());
            }
        }
        _ => return Err(format!("未知源码编辑事务：{action}")),
    }
    write_json(&state_path(store), &state)?;
    #[cfg(test)]
    pause_source_edit_test_failpoint("after_history_state_write");
    fs::remove_file(pending_path(store)).map_err(|error| error.to_string())?;
    crate::adapters::sync_directory(store)
        .map_err(|error| error.to_string())
}

fn recover_pending(root: &Path, store: &Path) -> Result<(), String> {
    let Some(pending) = read_json(&pending_path(store))? else {
        return Ok(());
    };
    if pending["files"].is_array() {
        apply_pending_files(root, store, &pending)?;
        return complete_transition(store, &pending);
    }
    let relative = pending["path"].as_str().ok_or("待恢复事务缺少源码路径。")?;
    let (_, current) = checked_source(root, relative)?;
    let current_hash = digest(&current);
    if current_hash == pending["target_sha256"].as_str().unwrap_or("") {
        return complete_transition(store, &pending);
    }
    if current_hash == pending["expected_sha256"].as_str().unwrap_or("") {
        fs::remove_file(pending_path(store)).map_err(|error| error.to_string())?;
        return Ok(());
    }
    Err(format!(
        "发现未完成的源码编辑事务，且文件又被修改：{relative}。保留 pending.json 供人工恢复。"
    ))
}

fn read_transaction_bytes(store: &Path, record: &Value, side: &str) -> Result<Vec<u8>, String> {
    let id = record["id"].as_str().ok_or("源码编辑历史记录缺少 ID。")?;
    if id.is_empty() || !id.bytes().all(|byte| byte.is_ascii_digit() || byte == b'-') {
        return Err("源码编辑历史记录 ID 无效。".into());
    }
    let transactions = store.join("transactions");
    let transactions_metadata =
        fs::symlink_metadata(&transactions).map_err(|error| error.to_string())?;
    if transactions_metadata.file_type().is_symlink() || !transactions_metadata.is_dir() {
        return Err("源码编辑 transactions 路径必须是普通目录。".into());
    }
    let transaction = transactions.join(id);
    let transaction_metadata =
        fs::symlink_metadata(&transaction).map_err(|error| error.to_string())?;
    if transaction_metadata.file_type().is_symlink() || !transaction_metadata.is_dir() {
        return Err("源码编辑事务快照目录必须是普通目录。".into());
    }
    let path = transaction_path(store, id, side);
    let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > SOURCE_LIMIT {
        return Err(format!("拒绝读取无效源码编辑快照：{}", path.display()));
    }
    fs::read(path).map_err(|error| error.to_string())
}

fn read_transaction_file_bytes(
    store: &Path,
    record: &Value,
    file: &Value,
    side: &str,
) -> Result<Vec<u8>, String> {
    let Some(index) = file["snapshot_index"].as_u64() else {
        return read_transaction_bytes(store, record, side);
    };
    let id = record["id"].as_str().ok_or("源码编辑历史记录缺少 ID。")?;
    if id.is_empty() || !id.bytes().all(|byte| byte.is_ascii_digit() || byte == b'-') {
        return Err("源码编辑历史记录 ID 无效。".into());
    }
    let transactions = store.join("transactions");
    let transaction = transactions.join(id);
    for directory in [&transactions, &transaction] {
        let metadata = fs::symlink_metadata(directory).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(format!("源码编辑事务快照目录无效：{}", directory.display()));
        }
    }
    let path = transaction_file_path(store, id, side, index as usize);
    let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > SOURCE_LIMIT {
        return Err(format!("拒绝读取无效源码编辑快照：{}", path.display()));
    }
    fs::read(path).map_err(|error| error.to_string())
}

fn history_files(record: &Value) -> Result<Vec<Value>, String> {
    if let Some(files) = record["files"].as_array() {
        if files.is_empty() || files.len() > 64 {
            return Err("源码编辑历史文件列表无效。".into());
        }
        Ok(files.clone())
    } else {
        Ok(vec![record.clone()])
    }
}

fn preflight_pending_files(
    root: &Path,
    store: &Path,
    pending: &Value,
) -> Result<Vec<PendingFileWrite>, String> {
    let files = pending["files"]
        .as_array()
        .ok_or("源码事务缺少文件列表。")?;
    if files.is_empty() || files.len() > 64 {
        return Err("源码事务文件列表无效。".into());
    }
    let record = pending.get("record").ok_or("源码事务缺少历史记录。")?;
    let target_side = pending["target_side"]
        .as_str()
        .ok_or("源码事务缺少目标快照方向。")?;
    if !matches!(target_side, "before" | "after") {
        return Err("源码事务目标快照方向无效。".into());
    }
    let recorded_files = history_files(record)?;
    if !record["files"].is_array() || files.len() != recorded_files.len() {
        return Err("源码事务待恢复清单不完整。".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut current = Vec::with_capacity(files.len());
    let mut snapshot_bytes = 0u64;
    for file in files {
        let relative = file["path"].as_str().ok_or("源码事务文件缺少路径。")?;
        if !seen.insert(relative.to_owned()) {
            return Err(format!("源码事务重复包含文件：{relative}"));
        }
        let recorded = recorded_files
            .iter()
            .find(|item| item["path"] == relative)
            .ok_or_else(|| format!("源码事务记录与待恢复清单不匹配：{relative}"))?;
        let (_, bytes) = checked_source(root, relative)?;
        let actual = digest(&bytes);
        let expected = file["expected_sha256"]
            .as_str()
            .ok_or("源码事务缺少预期指纹。")?;
        let target = file["target_sha256"]
            .as_str()
            .ok_or("源码事务缺少目标指纹。")?;
        let expected_side = if target_side == "after" {
            "before"
        } else {
            "after"
        };
        if file["snapshot_index"] != recorded["snapshot_index"]
            || recorded[format!("{expected_side}_sha256")] != expected
            || recorded[format!("{target_side}_sha256")] != target
        {
            return Err(format!("源码事务指纹与快照记录不匹配：{relative}"));
        }
        if actual != expected && actual != target {
            return Err(format!(
                "源码事务文件已发生外部修改，不能安全继续：{relative}。保留 pending.json 供人工恢复。"
            ));
        }
        let target_contents = read_transaction_file_bytes(store, record, recorded, target_side)?;
        if digest(&target_contents) != target {
            return Err(format!("源码事务目标快照指纹不匹配：{relative}"));
        }
        snapshot_bytes = snapshot_bytes
            .checked_add(target_contents.len() as u64)
            .ok_or("源码事务快照大小溢出。")?;
        if snapshot_bytes > BATCH_SOURCE_LIMIT {
            return Err("源码事务目标快照总量超过 64 MiB。".into());
        }
        current.push(PendingFileWrite {
            file: file.clone(),
            actual,
            expected: expected.to_owned(),
            target: target.to_owned(),
            contents: target_contents,
        });
    }
    Ok(current)
}

fn apply_pending_files(root: &Path, store: &Path, pending: &Value) -> Result<(), String> {
    let current = preflight_pending_files(root, store, pending)?;
    for item in current {
        if item.actual == item.target {
            continue;
        }
        let relative = item.file["path"].as_str().ok_or("源码事务文件缺少路径。")?;
        write_source(root, relative, &item.expected, &item.contents)?;
        #[cfg(test)]
        pause_source_edit_test_failpoint("after_file_replacement");
    }
    // A later file replacement may race with an external edit to an earlier
    // file. Recheck the whole batch at the commit boundary before recording
    // the history transition as complete.
    for file in pending["files"]
        .as_array()
        .ok_or("源码事务缺少文件列表。")?
    {
        let relative = file["path"].as_str().ok_or("源码事务文件缺少路径。")?;
        let target = file["target_sha256"]
            .as_str()
            .ok_or("源码事务缺少目标指纹。")?;
        let (_, persisted) = checked_source(root, relative)?;
        if digest(&persisted) != target {
            return Err(format!("源码事务在完成前再次变化：{relative}"));
        }
    }
    #[cfg(test)]
    pause_source_edit_test_failpoint("after_all_file_replacements");
    Ok(())
}

#[cfg(test)]
fn pause_source_edit_test_failpoint(phase: &str) {
    if std::env::var("MODEL_MAVEN_SOURCE_EDIT_TEST_FAILPOINT")
        .ok()
        .as_deref()
        != Some(phase)
    {
        return;
    }
    let Some(marker) = std::env::var_os("MODEL_MAVEN_SOURCE_EDIT_TEST_MARKER") else {
        return;
    };
    fs::write(marker, phase).expect("write source edit test failpoint marker");
    loop {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn apply_history_transition(
    root: &Path,
    store: &Path,
    record: Value,
    action: &str,
    expected: &str,
    target_side: &str,
    cursor: usize,
) -> Result<Value, String> {
    if record["files"].is_array() {
        let files = history_files(&record)?;
        let mut pending_files = Vec::with_capacity(files.len());
        let mut paths = Vec::with_capacity(files.len());
        for file in files {
            let relative = file["path"].as_str().ok_or("源码编辑历史缺少路径。")?;
            let expected_sha = file[expected]
                .as_str()
                .ok_or("源码编辑历史缺少预期指纹。")?;
            let target_sha = file[format!("{target_side}_sha256")]
                .as_str()
                .ok_or("源码编辑历史缺少目标指纹。")?;
            let target = read_transaction_file_bytes(store, &record, &file, target_side)?;
            if digest(&target) != target_sha {
                return Err(format!("源码编辑快照指纹不匹配：{relative}"));
            }
            pending_files.push(json!({
                "path":relative,
                "expected_sha256":expected_sha,
                "target_sha256":target_sha,
                "snapshot_index":file["snapshot_index"]
            }));
            paths.push(relative.to_owned());
        }
        let pending = json!({
            "schema_version":1,
            "action":action,
            "target_side":target_side,
            "files":pending_files,
            "cursor":cursor,
            "record":record
        });
        preflight_pending_files(root, store, &pending)?;
        write_json(&pending_path(store), &pending)?;
        apply_pending_files(root, store, &pending)?;
        complete_transition(store, &pending)?;
        let state = read_state(store)?;
        return Ok(json!({
            "action":action,
            "path":if paths.len() == 1 { paths[0].to_owned() } else { format!("{} files", paths.len()) },
            "paths":paths,
            "cursor":state["cursor"],
            "revision":record["after_sha256"]
        }));
    }
    let relative = record["path"].as_str().ok_or("源码编辑历史缺少路径。")?;
    let target = read_transaction_bytes(store, &record, target_side)?;
    let target_hash = digest(&target);
    let expected_sha = record[expected]
        .as_str()
        .ok_or("源码编辑历史缺少校验指纹。")?;
    let recorded_target = if target_side == "before" {
        record["before_sha256"].as_str().unwrap_or("")
    } else {
        record["after_sha256"].as_str().unwrap_or("")
    };
    if target_hash != recorded_target {
        return Err(format!("源码编辑快照指纹不匹配：{relative}"));
    }
    let pending = json!({
        "schema_version":1,"action":action,"path":relative,
        "expected_sha256":expected_sha,"target_sha256":target_hash,
        "cursor":cursor,"record":record
    });
    write_json(&pending_path(store), &pending)?;
    write_source(root, relative, expected_sha, &target)?;
    complete_transition(store, &pending)?;
    let state = read_state(store)?;
    Ok(
        json!({"action":action,"path":relative,"cursor":state["cursor"],"revision":record["after_sha256"]}),
    )
}

impl SourceEditRepository for FileSourceEditStore {
    fn lock(&self, root: &Path) -> Result<Box<dyn SourceEditLock>, String> {
        let metadata = root.join(".model-maven");
        if !metadata.exists() {
            ensure_directory(&metadata)?;
        }
        if fs::symlink_metadata(&metadata).is_ok_and(|info| info.file_type().is_symlink()) {
            return Err(".model-maven 不能是符号链接。".into());
        }
        let lock_path = metadata.join("source-edits.lock");
        if fs::symlink_metadata(&lock_path).is_ok_and(|info| info.file_type().is_symlink()) {
            return Err("源码编辑锁不能是符号链接。".into());
        }
        let mut options = OpenOptions::new();
        options.create(true).truncate(false).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let file = options
            .open(&lock_path)
            .map_err(|error| error.to_string())?;
        FileExt::try_lock_exclusive(&file)
            .map_err(|_| "另一源码编辑事务正在运行，请稍后重试。".to_string())?;
        Ok(Box::new(ProjectLock { _file: file }))
    }

    fn commit_batch(&self, root: &Path, patches: &[Value]) -> Result<Value, String> {
        if patches.is_empty() || patches.len() > 64 {
            return Err("源码事务需要 1..64 个文件补丁。".into());
        }
        let store = store_directory(root, true)?.expect("created store");
        recover_pending(root, &store)?;
        let state = read_state(&store)?;
        let cursor = state["cursor"]
            .as_u64()
            .ok_or("源码编辑历史 cursor 无效。")? as usize;
        if cursor >= HISTORY_LIMIT {
            return Err("源码编辑历史达到 256 项上限；请先归档项目或移除旧历史。".into());
        }
        let mut ordered = patches.to_vec();
        ordered.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
        let mut seen = std::collections::BTreeSet::new();
        let mut prepared = Vec::with_capacity(ordered.len());
        let mut total_bytes = 0u64;
        for patch in ordered {
            let relative_path = patch["path"].as_str().ok_or("源码补丁缺少相对路径。")?;
            if !seen.insert(relative_path.to_owned()) {
                return Err(format!("源码事务重复包含文件：{relative_path}"));
            }
            let expected_sha256 = patch["before_sha256"]
                .as_str()
                .ok_or("源码补丁缺少原文件指纹。")?;
            let contents = patch["contents"].as_str().ok_or("源码补丁缺少候选内容。")?;
            let (_, before) = checked_source(root, relative_path)?;
            let before_sha = digest(&before);
            if before_sha != expected_sha256 {
                return Err(format!("源码已变化，编辑计划过期：{relative_path}"));
            }
            let after = contents.as_bytes().to_vec();
            if after.len() as u64 > SOURCE_LIMIT {
                return Err(format!("编辑后的源码超过 16 MiB：{relative_path}"));
            }
            std::str::from_utf8(&after).map_err(|error| error.to_string())?;
            let after_sha = digest(&after);
            if after_sha == before_sha {
                continue;
            }
            total_bytes = total_bytes
                .checked_add(before.len() as u64)
                .and_then(|value| value.checked_add(after.len() as u64))
                .ok_or("源码事务快照大小溢出。")?;
            if total_bytes > BATCH_SOURCE_LIMIT {
                return Err("源码事务快照总量超过 64 MiB。".into());
            }
            prepared.push((
                relative_path.to_owned(),
                before,
                after,
                before_sha,
                after_sha,
            ));
        }
        if prepared.is_empty() {
            return Err("源码编辑没有产生任何变更。".into());
        }
        let id = format!(
            "{}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos(),
            std::process::id()
        );
        let transaction_dir = store.join("transactions").join(&id);
        create_private_directory(&transaction_dir)?;
        let mut file_records = Vec::with_capacity(prepared.len());
        let mut pending_files = Vec::with_capacity(prepared.len());
        let mut paths = Vec::with_capacity(prepared.len());
        for (index, (relative_path, before, after, before_sha, after_sha)) in
            prepared.iter().enumerate()
        {
            let before_path = transaction_file_path(&store, &id, "before", index);
            let after_path = transaction_file_path(&store, &id, "after", index);
            fs::write(&before_path, before).map_err(|error| error.to_string())?;
            fs::write(&after_path, after).map_err(|error| error.to_string())?;
            File::open(&before_path)
                .and_then(|file| file.sync_all())
                .map_err(|error| error.to_string())?;
            File::open(&after_path)
                .and_then(|file| file.sync_all())
                .map_err(|error| error.to_string())?;
            file_records.push(json!({
                "path":relative_path,
                "before_sha256":before_sha,
                "after_sha256":after_sha,
                "snapshot_index":index
            }));
            pending_files.push(json!({
                "path":relative_path,
                "expected_sha256":before_sha,
                "target_sha256":after_sha,
                "snapshot_index":index
            }));
            paths.push(relative_path.clone());
        }
        crate::adapters::sync_directory(&transaction_dir)
            .map_err(|error| error.to_string())?;
        crate::adapters::sync_directory(&store.join("transactions"))
            .map_err(|error| error.to_string())?;
        let before_revision = files_digest(&file_records, "before")?;
        let after_revision = files_digest(&file_records, "after")?;
        let record = json!({
            "id":id,
            "files":file_records,
            "paths":paths,
            "before_sha256":before_revision,
            "after_sha256":after_revision
        });
        let pending = json!({
            "schema_version":1,"action":"commit","target_side":"after",
            "files":pending_files,
            "cursor":cursor,"record":record
        });
        preflight_pending_files(root, &store, &pending)?;
        write_json(&pending_path(&store), &pending)?;
        apply_pending_files(root, &store, &pending).map_err(|error| {
            format!("源码事务未能全部完成：{error}；保留事务日志，后续源码操作会尝试恢复。")
        })?;
        complete_transition(&store, &pending)?;
        let receipt_files: Vec<Value> = prepared
            .iter()
            .enumerate()
            .map(|(index, (path, _, _, before_sha, after_sha))| {
                json!({
                    "path":path,
                    "before_sha256":before_sha,
                    "after_sha256":after_sha,
                    "backup":transaction_file_path(&store, &id, "before", index)
                })
            })
            .collect();
        let path_label = if paths.len() == 1 {
            paths[0].clone()
        } else {
            format!("{} files", paths.len())
        };
        let mut receipt = json!({
            "id":id,
            "path":path_label,
            "paths":paths,
            "files":receipt_files,
            "before_sha256":before_revision,
            "after_sha256":after_revision,
            "committed":true
        });
        if prepared.len() == 1 {
            receipt["before_sha256"] = json!(prepared[0].3);
            receipt["after_sha256"] = json!(prepared[0].4);
            receipt["backup"] = json!(transaction_file_path(&store, &id, "before", 0));
        }
        Ok(receipt)
    }

    fn rollback(&self, root: &Path, receipt: &Value) -> Result<Value, String> {
        let store = store_directory(root, false)?.ok_or("源码编辑历史目录不存在。")?;
        recover_pending(root, &store)?;
        let state = read_state(&store)?;
        let cursor = state["cursor"]
            .as_u64()
            .ok_or("源码编辑历史 cursor 无效。")? as usize;
        let last = state["records"]
            .as_array()
            .and_then(|records| records.get(cursor.saturating_sub(1)))
            .ok_or("刚提交的源码编辑记录不存在，无法安全回滚。")?;
        if last["files"].is_array() {
            let recorded_files = history_files(last)?;
            let receipt_files = receipt["files"]
                .as_array()
                .ok_or("源码事务回执缺少文件列表，无法安全回滚。")?;
            let matches = recorded_files.len() == receipt_files.len()
                && recorded_files
                    .iter()
                    .zip(receipt_files)
                    .all(|(recorded, returned)| {
                        recorded["path"] == returned["path"]
                            && recorded["before_sha256"] == returned["before_sha256"]
                            && recorded["after_sha256"] == returned["after_sha256"]
                    });
            if last["id"] != receipt["id"] || !matches {
                return Err("源码编辑历史在验证前已变化，拒绝自动回滚。".into());
            }
            return apply_history_transition(
                root,
                &store,
                last.clone(),
                "rollback",
                "after_sha256",
                "before",
                cursor - 1,
            );
        }
        if last["id"] != receipt["id"] || last["after_sha256"] != receipt["after_sha256"] {
            return Err("源码编辑历史在验证前已变化，拒绝自动回滚。".into());
        }
        let relative = last["path"].as_str().ok_or("源码编辑历史缺少路径。")?;
        let before = read_transaction_bytes(&store, last, "before")?;
        let before_sha = last["before_sha256"].as_str().ok_or("缺少原源码指纹。")?;
        let after_sha = last["after_sha256"].as_str().ok_or("缺少新源码指纹。")?;
        if digest(&before) != before_sha {
            return Err(format!("源码回滚快照指纹不匹配：{relative}"));
        }
        let pending = json!({
            "schema_version":1,"action":"rollback","path":relative,
            "expected_sha256":after_sha,"target_sha256":before_sha,
            "cursor":cursor-1,"record":last
        });
        write_json(&pending_path(&store), &pending)?;
        write_source(root, relative, after_sha, &before)?;
        complete_transition(&store, &pending)?;
        Ok(json!({"rolled_back":true,"path":relative,"cursor":cursor-1}))
    }

    fn undo(&self, root: &Path) -> Result<Value, String> {
        let store = store_directory(root, false)?.ok_or("没有源码编辑历史。")?;
        recover_pending(root, &store)?;
        let state = read_state(&store)?;
        let cursor = state["cursor"]
            .as_u64()
            .ok_or("源码编辑历史 cursor 无效。")? as usize;
        if cursor == 0 {
            return Err("没有可撤销的源码编辑。".into());
        }
        let record = state["records"][cursor - 1].clone();
        apply_history_transition(
            root,
            &store,
            record,
            "undo",
            "after_sha256",
            "before",
            cursor,
        )
    }

    fn redo(&self, root: &Path) -> Result<Value, String> {
        let store = store_directory(root, false)?.ok_or("没有源码编辑历史。")?;
        recover_pending(root, &store)?;
        let state = read_state(&store)?;
        let cursor = state["cursor"]
            .as_u64()
            .ok_or("源码编辑历史 cursor 无效。")? as usize;
        let records = state["records"]
            .as_array()
            .ok_or("源码编辑历史缺少 records。")?;
        if cursor >= records.len() {
            return Err("没有可重做的源码编辑。".into());
        }
        let record = records[cursor].clone();
        apply_history_transition(
            root,
            &store,
            record,
            "redo",
            "before_sha256",
            "after",
            cursor,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    fn stage_pending_batch(root: &Path) -> (PathBuf, Value) {
        let originals = [
            (
                "encoder.py",
                b"width = 8\n".as_slice(),
                b"width = 16\n".as_slice(),
            ),
            (
                "head.py",
                b"inputs = 8\n".as_slice(),
                b"inputs = 16\n".as_slice(),
            ),
        ];
        for (path, before, _) in originals {
            fs::write(root.join(path), before).unwrap();
        }
        let store = store_directory(root, true).unwrap().unwrap();
        let id = "1700000000-1";
        let transaction_dir = store.join("transactions").join(id);
        create_private_directory(&transaction_dir).unwrap();
        let mut records = Vec::new();
        let mut pending_files = Vec::new();
        let mut paths = Vec::new();
        for (index, (path, before, after)) in originals.into_iter().enumerate() {
            fs::write(transaction_file_path(&store, id, "before", index), before).unwrap();
            fs::write(transaction_file_path(&store, id, "after", index), after).unwrap();
            let before_sha = digest(before);
            let after_sha = digest(after);
            records.push(json!({
                "path":path,
                "before_sha256":before_sha,
                "after_sha256":after_sha,
                "snapshot_index":index
            }));
            pending_files.push(json!({
                "path":path,
                "expected_sha256":before_sha,
                "target_sha256":after_sha,
                "snapshot_index":index
            }));
            paths.push(path);
        }
        let before_revision = files_digest(&records, "before").unwrap();
        let after_revision = files_digest(&records, "after").unwrap();
        let record = json!({
            "id":id,
            "paths":paths,
            "files":records,
            "before_sha256":before_revision,
            "after_sha256":after_revision
        });
        let pending = json!({
            "schema_version":1,
            "action":"commit",
            "target_side":"after",
            "files":pending_files,
            "cursor":0,
            "record":record
        });
        write_json(&pending_path(&store), &pending).unwrap();
        (store, pending)
    }

    #[test]
    fn multi_file_commit_undo_redo_and_validation_rollback_share_one_history_item() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("encoder.py"), "width = 8\n").unwrap();
        fs::write(directory.path().join("head.py"), "inputs = 8\n").unwrap();
        let repository = FileSourceEditStore;
        let patches = [
            json!({
                "path":"encoder.py",
                "before_sha256":digest(b"width = 8\n"),
                "contents":"width = 16\n"
            }),
            json!({
                "path":"head.py",
                "before_sha256":digest(b"inputs = 8\n"),
                "contents":"inputs = 16\n"
            }),
        ];

        let receipt = repository.commit_batch(directory.path(), &patches).unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("encoder.py")).unwrap(),
            "width = 16\n"
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("head.py")).unwrap(),
            "inputs = 16\n"
        );
        assert_eq!(
            read_state(&store_directory(directory.path(), false).unwrap().unwrap()).unwrap()["cursor"],
            1
        );

        repository.undo(directory.path()).unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("encoder.py")).unwrap(),
            "width = 8\n"
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("head.py")).unwrap(),
            "inputs = 8\n"
        );
        repository.redo(directory.path()).unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("encoder.py")).unwrap(),
            "width = 16\n"
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("head.py")).unwrap(),
            "inputs = 16\n"
        );

        repository.rollback(directory.path(), &receipt).unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("encoder.py")).unwrap(),
            "width = 8\n"
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("head.py")).unwrap(),
            "inputs = 8\n"
        );
        let state =
            read_state(&store_directory(directory.path(), false).unwrap().unwrap()).unwrap();
        assert_eq!(state["cursor"], 0);
        assert!(state["records"].as_array().unwrap().is_empty());
    }

    #[test]
    fn recovery_finishes_a_partially_replaced_batch() {
        let directory = tempfile::tempdir().unwrap();
        let (store, pending) = stage_pending_batch(directory.path());
        let first = &pending["files"][0];
        let first_target = read_transaction_file_bytes(
            &store,
            &pending["record"],
            &pending["record"]["files"][0],
            "after",
        )
        .unwrap();
        write_source(
            directory.path(),
            first["path"].as_str().unwrap(),
            first["expected_sha256"].as_str().unwrap(),
            &first_target,
        )
        .unwrap();

        recover_pending(directory.path(), &store).unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("encoder.py")).unwrap(),
            "width = 16\n"
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("head.py")).unwrap(),
            "inputs = 16\n"
        );
        assert_eq!(read_state(&store).unwrap()["cursor"], 1);
        assert!(!pending_path(&store).exists());
    }

    #[test]
    fn recovery_preserves_all_files_and_journal_when_a_target_has_a_conflict() {
        let directory = tempfile::tempdir().unwrap();
        let (store, pending) = stage_pending_batch(directory.path());
        let first = &pending["files"][0];
        let first_target = read_transaction_file_bytes(
            &store,
            &pending["record"],
            &pending["record"]["files"][0],
            "after",
        )
        .unwrap();
        write_source(
            directory.path(),
            first["path"].as_str().unwrap(),
            first["expected_sha256"].as_str().unwrap(),
            &first_target,
        )
        .unwrap();
        fs::write(directory.path().join("head.py"), "inputs = 32\n").unwrap();

        assert!(recover_pending(directory.path(), &store).is_err());
        assert_eq!(
            fs::read_to_string(directory.path().join("encoder.py")).unwrap(),
            "width = 16\n"
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("head.py")).unwrap(),
            "inputs = 32\n"
        );
        assert_eq!(read_state(&store).unwrap()["cursor"], 0);
        assert!(pending_path(&store).exists());
    }

    #[test]
    fn source_edit_commit_child_pauses_at_requested_failpoint() {
        let (Ok(root), Ok(_marker)) = (
            std::env::var("MODEL_MAVEN_SOURCE_EDIT_TEST_ROOT"),
            std::env::var("MODEL_MAVEN_SOURCE_EDIT_TEST_MARKER"),
        ) else {
            return;
        };
        let root = PathBuf::from(root);
        let patches = [
            json!({
                "path":"encoder.py",
                "before_sha256":digest(b"width = 8\n"),
                "contents":"width = 16\n"
            }),
            json!({
                "path":"head.py",
                "before_sha256":digest(b"inputs = 8\n"),
                "contents":"inputs = 16\n"
            }),
        ];
        FileSourceEditStore
            .commit_batch(&root, &patches)
            .expect("child transaction should reach its test failpoint");
        panic!("child transaction passed the failpoint without pausing");
    }

    fn kill_source_commit_at(phase: &str) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("encoder.py"), "width = 8\n").unwrap();
        fs::write(directory.path().join("head.py"), "inputs = 8\n").unwrap();
        let marker = directory.path().join("test-failpoint-ready");
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "adapters::source_edits::tests::source_edit_commit_child_pauses_at_requested_failpoint",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("MODEL_MAVEN_SOURCE_EDIT_TEST_ROOT", directory.path())
            .env("MODEL_MAVEN_SOURCE_EDIT_TEST_MARKER", &marker)
            .env("MODEL_MAVEN_SOURCE_EDIT_TEST_FAILPOINT", phase)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !marker.exists() {
            if let Some(status) = child.try_wait().unwrap() {
                panic!("child exited before reaching the journal boundary: {status}");
            }
            assert!(
                Instant::now() < deadline,
                "child did not reach the commit boundary"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        child.kill().unwrap();
        let status = child.wait().unwrap();
        assert!(!status.success(), "the child must be terminated mid-commit");
        directory
    }

    #[test]
    fn operating_system_kill_after_first_replacement_recovers_on_restart() {
        let directory = kill_source_commit_at("after_file_replacement");
        let store = store_directory(directory.path(), false).unwrap().unwrap();
        assert!(
            pending_path(&store).exists(),
            "durable journal must survive process death"
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("encoder.py")).unwrap(),
            "width = 16\n"
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("head.py")).unwrap(),
            "inputs = 8\n"
        );

        // A fresh repository operation recovers the journal before proceeding.
        recover_pending(directory.path(), &store).unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("encoder.py")).unwrap(),
            "width = 16\n"
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("head.py")).unwrap(),
            "inputs = 16\n"
        );
        assert_eq!(read_state(&store).unwrap()["cursor"], 1);
        assert!(!pending_path(&store).exists());
    }

    #[test]
    fn operating_system_kill_after_all_replacements_recovers_history_on_restart() {
        let directory = kill_source_commit_at("after_all_file_replacements");
        let store = store_directory(directory.path(), false).unwrap().unwrap();
        assert!(pending_path(&store).exists());
        for (path, contents) in [("encoder.py", "width = 16\n"), ("head.py", "inputs = 16\n")] {
            assert_eq!(
                fs::read_to_string(directory.path().join(path)).unwrap(),
                contents
            );
        }
        assert_eq!(read_state(&store).unwrap()["cursor"], 0);

        recover_pending(directory.path(), &store).unwrap();
        let state = read_state(&store).unwrap();
        assert_eq!(state["cursor"], 1);
        assert_eq!(state["records"].as_array().unwrap().len(), 1);
        assert!(!pending_path(&store).exists());
    }

    #[test]
    fn operating_system_kill_after_history_write_cleans_pending_journal_on_restart() {
        let directory = kill_source_commit_at("after_history_state_write");
        let store = store_directory(directory.path(), false).unwrap().unwrap();
        assert!(pending_path(&store).exists());
        let state_before_recovery = read_state(&store).unwrap();
        assert_eq!(state_before_recovery["cursor"], 1);
        assert_eq!(
            state_before_recovery["records"].as_array().unwrap().len(),
            1
        );

        recover_pending(directory.path(), &store).unwrap();
        let state_after_recovery = read_state(&store).unwrap();
        assert_eq!(state_after_recovery["cursor"], 1);
        assert_eq!(state_after_recovery["records"].as_array().unwrap().len(), 1);
        assert!(!pending_path(&store).exists());
    }
}
