//! Compatibility re-export for callers while the application use cases move into modules.
#[path = "application/mod.rs"]
mod application;

pub(crate) use application::classification;
pub(crate) use application::{scaffold_encoder_layer_catalog, normalize_training_command};
pub use application::{Application, ExecutionContext};
