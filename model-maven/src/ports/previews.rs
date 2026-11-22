use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewTransforms {
    pub steps: Vec<Value>,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewFingerprints {
    pub project_python_sha256: String,
    pub environment_lock_sha256: Option<String>,
    pub source_entrypoint_is_local: bool,
    pub renderer_entrypoint_is_local: Option<bool>,
    pub custom_transform_entrypoints_are_local: Vec<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreviewRepositoryError {
    ProjectUnavailable(String),
    TransformsPathInvalid(String),
    TransformsReadFailed(String),
    TransformsInvalid(String),
    SourceFingerprintFailed(String),
    EnvironmentReadFailed(String),
}

impl PreviewRepositoryError {
    pub fn message(&self) -> &str {
        match self {
            Self::ProjectUnavailable(message)
            | Self::TransformsPathInvalid(message)
            | Self::TransformsReadFailed(message)
            | Self::TransformsInvalid(message)
            | Self::SourceFingerprintFailed(message)
            | Self::EnvironmentReadFailed(message) => message,
        }
    }
}

/// Owns project-local preview inputs and persisted preview generations.
/// Application policy receives fingerprints and lifecycle results, not filesystem access.
pub trait PreviewRepository {
    fn project_root(&self) -> Result<PathBuf, PreviewRepositoryError>;
    fn read_transforms(
        &self,
        path: Option<&Path>,
    ) -> Result<PreviewTransforms, PreviewRepositoryError>;
    fn project_fingerprints(
        &self,
        root: &Path,
        source_entrypoint: &str,
        renderer_entrypoint: Option<&str>,
        transforms: &[Value],
    ) -> Result<PreviewFingerprints, PreviewRepositoryError>;

    fn begin_preview(&self, root: &Path, signature: &str) -> Result<Value, String>;
    fn mark_preview_running(
        &self,
        root: &Path,
        request_id: &str,
        generation: u64,
    ) -> Result<bool, String>;
    fn cancel_preview(&self, root: &Path, request_id: &str, generation: u64) -> Result<(), String>;
    fn fail_preview(
        &self,
        root: &Path,
        request_id: &str,
        generation: u64,
        message: &str,
    ) -> Result<(), String>;
    fn complete_preview(
        &self,
        root: &Path,
        request_id: &str,
        generation: u64,
        signature: &str,
    ) -> Result<Value, String>;
    fn current_preview(&self, root: &Path) -> Result<Value, String>;
}
