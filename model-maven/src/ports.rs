//! Compatibility facade for application-owned persistence and runtime ports.
//! Domain-specific contracts live in focused modules and remain re-exported here.

mod artifacts;
mod backend;
mod components;
mod edit;
mod previews;
mod project_config;
mod project_source;
mod runs;
pub mod services;

pub use artifacts::{ArtifactExporter, TrainingExportRequest};
pub use backend::{Backend, PythonRuntimeControl};
pub use components::{ComponentResolver, ComponentScaffolder};
pub use edit::{SourceEditLock, SourceEditRepository, VariantLock, VariantRepository};
pub use previews::{
    PreviewFingerprints, PreviewRepository, PreviewRepositoryError, PreviewTransforms,
};
pub use project_config::{ProjectConfigStore, ProjectConfigUpdate, ProjectConfigUpdateError};
pub use project_source::ProjectSourceWriter;
pub use runs::{RunRepository, RunSupervisor};
