use serde_json::Value;
use std::path::{Path, PathBuf};

/// Persists foreground training probes and exposes their immutable inputs/results.
pub trait RunRepository {
    fn create(&self, root: &Path, inputs: &Value) -> Result<String, String>;
    fn finish(&self, root: &Path, id: &str, report: &Value) -> Result<(), String>;
    fn checkpoint_dir(&self, root: &Path, id: &str, create: bool) -> Result<PathBuf, String>;
    fn latest_checkpoint(&self, root: &Path, id: &str) -> Result<PathBuf, String>;
    fn run_directory(&self, root: &Path, id: &str) -> Result<PathBuf, String>;
    fn set_supervisor(&self, root: &Path, id: &str, pid: u32, token: &str) -> Result<(), String>;
    fn record_heartbeat(&self, root: &Path, id: &str, pid: u32, token: &str) -> Result<(), String>;
    fn request_stop(&self, root: &Path, id: &str) -> Result<(), String>;
    fn progress(&self, root: &Path, id: &str) -> Result<Vec<Value>, String>;
    fn log_tail(&self, root: &Path, id: &str, max_bytes: usize) -> Result<String, String>;
    fn storage(&self, root: &Path, id: Option<&str>) -> Result<Value, String>;
    fn rename(&self, root: &Path, id: &str, name: &str) -> Result<(), String>;
    fn set_kept(&self, root: &Path, id: &str, kept: bool) -> Result<(), String>;
    fn plan_clean(
        &self,
        root: &Path,
        run_id: Option<&str>,
        include_logs: bool,
        include_progress: bool,
        include_checkpoints: bool,
        artifact_id: Option<&str>,
    ) -> Result<Value, String>;
    fn apply_clean(
        &self,
        root: &Path,
        plan_id: &str,
        release_checkpoints: bool,
    ) -> Result<Value, String>;
    fn list(&self, root: &Path) -> Result<Vec<Value>, String>;
    fn find_request(&self, root: &Path, request_id: &str) -> Result<Value, String>;
    fn show(&self, root: &Path, id: &str) -> Result<Value, String>;
}

/// Owns detached process identity and OS signal delivery, separate from run persistence.
pub trait RunSupervisor {
    fn take_user_interrupt(&self) -> bool {
        false
    }
    fn identify(&self, pid: u32) -> Option<String>;
    fn launch(
        &self,
        executable: &Path,
        project_root: &Path,
        run_id: &str,
        request: &Value,
        log_path: &Path,
        register_identity: &mut dyn FnMut(u32, &str) -> Result<(), String>,
    ) -> Result<(), String>;
    fn is_alive(&self, pid: u32, token: &str) -> bool;
    fn interrupt(&self, pid: u32, token: &str) -> Result<(), String>;
}
