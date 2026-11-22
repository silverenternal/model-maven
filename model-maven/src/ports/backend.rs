use crate::domain::Report;
use serde_json::Value;

/// The application exchanges descriptors and reports, never framework objects.
pub trait Backend {
    fn execute(&self, request: &Value) -> Report;
}

/// Optional control for backends that can switch the project runtime interpreter.
pub trait PythonRuntimeControl: Backend {
    fn select_python(&mut self, executable: String);
}
