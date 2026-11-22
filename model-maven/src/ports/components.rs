use serde_json::{Map, Value};
use std::path::Path;

/// Resolves static project component descriptions without importing user Python code.
pub trait ComponentResolver {
    fn list(&self, root: &Path, query: Option<&str>) -> Result<Vec<Value>, String>;
    fn inspect(&self, root: &Path, id: &str) -> Result<Value, String>;
    fn resolve_dependencies(&self, root: &Path, id: &str) -> Result<Value, String>;
    fn register(&self, root: &Path, manifest: &Value) -> Result<Value, String>;
    fn resolve_layer(
        &self,
        root: &Path,
        id: &str,
        params: &Map<String, Value>,
    ) -> Result<Value, String>;
    fn resolve_layer_edit_provider(
        &self,
        root: &Path,
        id: &str,
        payload: &Map<String, Value>,
    ) -> Result<Value, String>;
    fn resolve_transform(
        &self,
        root: &Path,
        id: &str,
        params: &Map<String, Value>,
    ) -> Result<Value, String>;
    fn resolve_component_for_test(
        &self,
        root: &Path,
        id: &str,
        kind: &str,
        params: &Map<String, Value>,
    ) -> Result<Value, String>;
    fn component_verification(
        &self,
        root: &Path,
        id: &str,
        version: &str,
    ) -> Result<Option<Value>, String>;
    fn record_component_verification(
        &self,
        root: &Path,
        id: &str,
        version: &str,
        verification: &Value,
    ) -> Result<(), String>;
}

/// Creates a starter package for an independently buildable Python extension.
pub trait ComponentScaffolder {
    fn scaffold(
        &self,
        project_root: &Path,
        component_id: &str,
        kind: &str,
        version: &str,
        output: &Path,
    ) -> Result<Value, String>;
}
