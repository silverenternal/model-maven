//! Compatibility re-export for callers while the application use cases move into modules.
#[path = "application/mod.rs"]
mod application;

pub(crate) use application::classification;
pub use application::{Application, ExecutionContext};
