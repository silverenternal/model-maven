//! Proposed typed application service ports. Existing `Backend` and repository traits
//! remain the implemented interfaces until CORE-03 migrates use cases to these ports.
#![allow(dead_code)] // Proposed ports become active when CORE-03 migrates application use cases.

use crate::domain::{
    Report,
    operation::{ArtifactRef, OperationEvent, OperationPlan, OperationRequest},
};

pub trait OperationPlanner {
    fn plan(&self, request: &OperationRequest) -> Result<OperationPlan, String>;
}

/// Executes a previously assessed operation plan through application-owned use cases.
/// Implementations must not infer missing inputs or silently downgrade unknown capability.
pub trait OperationExecutor {
    fn execute(&self, plan: &OperationPlan) -> Report;
}

/// Read-only event query; event absence is not proof that an operation did not run.
pub trait OperationEventSource {
    fn read_after(
        &self,
        operation_id: &str,
        after_sequence: Option<u64>,
    ) -> Result<Vec<OperationEvent>, String>;
}

/// Resolves an artifact reference only when the owning service can verify its identity.
pub trait ArtifactReferenceResolver {
    fn resolve(&self, artifact_id: &str) -> Result<Option<ArtifactRef>, String>;
}

/// Receives user-facing execution lifecycle feedback without coupling the application to a terminal.
pub trait ExecutionProgress {
    fn worker_starting(&self, operation: &str, timeout_seconds: u64);
}
