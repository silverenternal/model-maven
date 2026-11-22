use serde_json::Value;
use std::path::Path;

/// Builds a portable, verifiable handoff bundle from an immutable training run.
pub struct TrainingExportRequest<'a> {
    pub project_root: &'a Path,
    pub destination: &'a Path,
    pub run: &'a Value,
    pub checkpoint: &'a Path,
    pub environment: &'a Value,
    pub extension_packages: &'a [Value],
    pub runtime_requirements: &'a [Value],
    pub overwrite: bool,
}

pub trait ArtifactExporter {
    fn export_training_run(&self, request: TrainingExportRequest<'_>) -> Result<Value, String>;
}
