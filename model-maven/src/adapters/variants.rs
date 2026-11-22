use fs2::FileExt;
use serde_json::{Value, json};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use crate::ports::{VariantLock, VariantRepository};

pub struct FileVariantStore;

struct VariantFile {
    path: PathBuf,
}

impl VariantFile {
    fn check_parents(&self) -> Result<(), String> {
        for path in self
            .path
            .parent()
            .into_iter()
            .flat_map(|p| p.ancestors())
            .take(2)
        {
            if fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
                return Err("变体目录不能通过符号链接重定向。".into());
            }
        }
        Ok(())
    }
    fn new(root: &Path, name: &str) -> Result<Self, String> {
        if name.is_empty()
            || name.len() > 64
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err("变体名只能包含字母、数字、-、_，长度 1..64。".into());
        }
        Ok(Self {
            path: root
                .join(".model-maven/variants")
                .join(format!("{name}.json")),
        })
    }

    fn load(&self) -> Result<Option<Value>, String> {
        self.check_parents()?;
        match fs::symlink_metadata(&self.path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err("拒绝通过符号链接读取/写入变体。".into());
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.to_string()),
        }
        let text = fs::read_to_string(&self.path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| format!("变体文件损坏：{e}"))
    }

    fn lock(&self) -> Result<File, String> {
        self.check_parents()?;
        let parent = self.path.parent().expect("variant parent");
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let lock_path = self.path.with_extension("lock");
        if fs::symlink_metadata(&lock_path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err("拒绝符号链接变体锁。".into());
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)
            .map_err(|e| e.to_string())?;
        FileExt::try_lock_exclusive(&lock)
            .map_err(|_| "另一编辑正在使用该变体，请稍后重试。".to_string())?;
        Ok(lock)
    }

    fn save(&self, value: &Value) -> Result<(), String> {
        let parent = self.path.parent().expect("variant parent");
        let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        serde_json::to_writer_pretty(&mut file, value).map_err(|e| e.to_string())?;
        file.write_all(b"\n").map_err(|e| e.to_string())?;
        file.as_file().sync_all().map_err(|e| e.to_string())?;
        file.persist(&self.path).map_err(|e| e.to_string())?;
        File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())
    }

    fn create(&self, value: &Value) -> Result<(), String> {
        self.check_parents()?;
        let parent = self.path.parent().expect("variant parent");
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        if fs::symlink_metadata(&self.path).is_ok() {
            return Err("目标变体已存在；复制不会覆盖已有版本。".into());
        }
        let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        serde_json::to_writer_pretty(&mut file, value).map_err(|e| e.to_string())?;
        file.write_all(b"\n").map_err(|e| e.to_string())?;
        file.as_file().sync_all().map_err(|e| e.to_string())?;
        match file.persist_noclobber(&self.path) {
            Ok(_) => {}
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err("目标变体已存在；复制不会覆盖已有版本。".into());
            }
            Err(error) => return Err(error.error.to_string()),
        }
        File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())
    }
}

impl VariantRepository for FileVariantStore {
    fn path(&self, root: &Path, name: &str) -> Result<PathBuf, String> {
        Ok(VariantFile::new(root, name)?.path)
    }

    fn load(&self, root: &Path, name: &str) -> Result<Option<Value>, String> {
        VariantFile::new(root, name)?.load()
    }

    fn list(&self, root: &Path) -> Result<Vec<Value>, String> {
        let directory = root.join(".model-maven/variants");
        match fs::symlink_metadata(&directory) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.to_string()),
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err("拒绝通过符号链接列出模型变体。".into());
            }
            Ok(metadata) if !metadata.is_dir() => {
                return Err("变体存储路径不是目录。".into());
            }
            Ok(_) => {}
        }
        VariantFile::new(root, "listing")?.check_parents()?;
        let mut rows = Vec::new();
        for entry in fs::read_dir(&directory).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let name = entry
                .path()
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_owned();
            let file_type = entry.file_type().map_err(|error| error.to_string())?;
            if file_type.is_symlink() {
                rows.push(json!({"name":name,"readable":false,"error":"拒绝符号链接变体。"}));
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let bytes = match fs::read(entry.path()) {
                Ok(bytes) => bytes,
                Err(error) => {
                    rows.push(json!({"name":name,"readable":false,"error":error.to_string()}));
                    continue;
                }
            };
            match serde_json::from_slice::<Value>(&bytes) {
                Ok(value) => rows.push(json!({
                    "name":name,
                    "readable":true,
                    "schema_version":value.get("schema_version"),
                    "model":value.pointer("/origin/model"),
                    "edit_count":value.get("edits").and_then(Value::as_array).map(Vec::len),
                    "cursor":value.get("cursor"),
                    "revision":value.get("revision")
                })),
                Err(error) => rows.push(json!({"name":name,"readable":false,"error":format!("变体 JSON 无效：{error}")})),
            }
        }
        rows.sort_by(|left, right| left["name"].as_str().cmp(&right["name"].as_str()));
        Ok(rows)
    }

    fn lock(&self, root: &Path, name: &str) -> Result<Box<dyn VariantLock>, String> {
        Ok(Box::new(VariantFile::new(root, name)?.lock()?))
    }

    fn save(&self, root: &Path, name: &str, value: &Value) -> Result<(), String> {
        VariantFile::new(root, name)?.save(value)
    }

    fn create(&self, root: &Path, name: &str, value: &Value) -> Result<(), String> {
        VariantFile::new(root, name)?.create(value)
    }
}
