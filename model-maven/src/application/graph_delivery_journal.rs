//! Durable, project-local state for a staged graph delivery.
//!
//! The journal is not a lock on the user's files. Each stage must still use
//! its own conditional commit and recognize both its pre- and post-state.
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const JOURNAL_LIMIT: u64 = 512 * 1024;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DeliveryJournal {
    pub version: u32,
    pub project_root: PathBuf,
    pub graph_file: String,
    pub graph_before: String,
    pub graph_after: String,
    pub graph_contents: String,
    pub source_file: String,
    pub source_before: String,
    pub source_after: String,
    pub config_before: String,
    pub config_after: String,
    pub include_data: bool,
    pub reviewed_plan: String,
}

fn digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn revision(value: &str) -> bool { value == "absent" || digest(value) }
fn root_file(value: &str, suffix: &str) -> bool {
    value.strip_suffix(suffix).is_some_and(|name| !name.is_empty() && name.len() <= 128
        && name.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')))
}
impl DeliveryJournal {
    fn validate(&self, root: &Path) -> Result<(), String> {
        if self.version != 1 || self.project_root != root || !root_file(&self.graph_file, ".json")
            || !root_file(&self.source_file, ".py") || !revision(&self.graph_before)
            || !revision(&self.source_before) || !revision(&self.config_before)
            || !digest(&self.graph_after) || !digest(&self.source_after) || !digest(&self.config_after)
            || !digest(&self.reviewed_plan) || self.graph_contents.len() > 128 * 1024
            || super::scaffold::sha256_bytes(self.graph_contents.as_bytes()) != self.graph_after {
            return Err("图交付记录无效；保留记录，不修改项目文件。".into());
        }
        Ok(())
    }
}

pub(super) struct DeliveryJournalStore {
    directory: PathBuf,
    root: PathBuf,
    _lock: File,
}
impl DeliveryJournalStore {
    pub(super) fn lock(root: &Path) -> Result<Self, String> {
        let directory = root.join(".model-maven");
        match fs::symlink_metadata(&directory) {
            Ok(meta) if meta.is_dir() && !meta.file_type().is_symlink() => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&directory).map_err(|error| error.to_string())?;
                #[cfg(unix)] {
                    use std::os::unix::fs::PermissionsExt;
                    fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).map_err(|error| error.to_string())?;
                }
            }
            _ => return Err("项目元数据目录无效或为符号链接。".into()),
        }
        let mut options = OpenOptions::new();
        options.create(true).truncate(false).read(true).write(true);
        #[cfg(unix)] {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
        }
        let lock = options.open(directory.join("graph-delivery.lock")).map_err(|error| error.to_string())?;
        lock.try_lock_exclusive().map_err(|_| "另一个图交付正在进行。".to_owned())?;
        Ok(Self { directory, root: root.to_path_buf(), _lock: lock })
    }
    fn path(&self) -> PathBuf { self.directory.join("graph-delivery.pending.json") }
    pub(super) fn load(&self) -> Result<Option<DeliveryJournal>, String> {
        let path = self.path();
        match fs::symlink_metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= JOURNAL_LIMIT => (),
            _ => return Err("图交付记录不是受支持的普通文件；保留记录。".into()),
        }
        let mut options = OpenOptions::new(); options.read(true);
        #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt; options.custom_flags(libc::O_NOFOLLOW); }
        let file = options.open(path).map_err(|error| error.to_string())?;
        let mut bytes = Vec::new();
        file.take(JOURNAL_LIMIT + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
        if bytes.len() as u64 > JOURNAL_LIMIT { return Err("图交付记录超过大小限制。".into()); }
        let journal: DeliveryJournal = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        journal.validate(&self.root)?;
        Ok(Some(journal))
    }
    pub(super) fn create(&self, journal: &DeliveryJournal) -> Result<(), String> {
        journal.validate(&self.root)?;
        if self.load()?.is_some() { return Err("已有未完成的图交付；请先恢复或核查。".into()); }
        let bytes = serde_json::to_vec(journal).map_err(|error| error.to_string())?;
        if bytes.len() as u64 > JOURNAL_LIMIT { return Err("图交付记录超过大小限制。".into()); }
        let mut file = tempfile::NamedTempFile::new_in(&self.directory).map_err(|error| error.to_string())?;
        file.write_all(&bytes).and_then(|_| file.as_file().sync_all()).map_err(|error| error.to_string())?;
        file.persist_noclobber(self.path()).map_err(|error| error.to_string())?;
        File::open(&self.directory).and_then(|directory| directory.sync_all()).map_err(|error| error.to_string())?;
        Ok(())
    }
    pub(super) fn archive(&self, journal: &DeliveryJournal) -> Result<String, String> {
        journal.validate(&self.root)?;
        let current = self.load()?.ok_or("图交付记录已消失；请核查结果。")?;
        let bytes = serde_json::to_vec(journal).map_err(|error| error.to_string())?;
        if serde_json::to_vec(&current).map_err(|error| error.to_string())? != bytes {
            return Err("图交付记录已变化；保留记录。".into());
        }
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?.as_nanos();
        let name = format!("graph-delivery.archived-{stamp}-{}.json", std::process::id());
        let mut file = tempfile::NamedTempFile::new_in(&self.directory).map_err(|error| error.to_string())?;
        file.write_all(&bytes).and_then(|_| file.as_file().sync_all()).map_err(|error| error.to_string())?;
        file.persist_noclobber(self.directory.join(&name)).map_err(|error| error.to_string())?;
        File::open(&self.directory).and_then(|directory| directory.sync_all()).map_err(|error| error.to_string())?;
        self.finish(journal).map_err(|error| format!("已保存归档 {name}，但待恢复记录未清除：{error}"))?;
        Ok(format!(".model-maven/{name}"))
    }
    pub(super) fn finish(&self, journal: &DeliveryJournal) -> Result<(), String> {
        let current = self.load()?.ok_or("图交付记录已消失；请核查结果。")?;
        if serde_json::to_vec(&current).map_err(|error| error.to_string())? != serde_json::to_vec(journal).map_err(|error| error.to_string())? {
            return Err("图交付记录已变化；保留记录。".into());
        }
        fs::remove_file(self.path()).map_err(|error| error.to_string())?;
        File::open(&self.directory).and_then(|directory| directory.sync_all()).map_err(|error| error.to_string())?;
        Ok(())
    }
}
