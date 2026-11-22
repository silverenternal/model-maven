use serde_json::Value;
use std::path::{Path, PathBuf};

/// Keeps an edit lock alive until the application finishes candidate validation and commit.
pub trait VariantLock {}

impl<T> VariantLock for T {}

/// Keeps a source-edit transaction serialized through candidate revalidation.
pub trait SourceEditLock {}

impl<T> SourceEditLock for T {}

/// Applies recoverable, conflict-checked edits to project-owned Python files.
pub trait SourceEditRepository {
    fn lock(&self, root: &Path) -> Result<Box<dyn SourceEditLock>, String>;
    fn commit_batch(&self, root: &Path, patches: &[Value]) -> Result<Value, String>;
    fn rollback(&self, root: &Path, receipt: &Value) -> Result<Value, String>;
    fn undo(&self, root: &Path) -> Result<Value, String>;
    fn redo(&self, root: &Path) -> Result<Value, String>;
}

/// Persistence boundary used by edit use cases; the application does not know about files.
pub trait VariantRepository {
    fn path(&self, root: &Path, name: &str) -> Result<PathBuf, String>;
    fn load(&self, root: &Path, name: &str) -> Result<Option<Value>, String>;
    fn list(&self, root: &Path) -> Result<Vec<Value>, String>;
    fn lock(&self, root: &Path, name: &str) -> Result<Box<dyn VariantLock>, String>;
    fn save(&self, root: &Path, name: &str, value: &Value) -> Result<(), String>;
    fn create(&self, root: &Path, name: &str, value: &Value) -> Result<(), String>;
}
