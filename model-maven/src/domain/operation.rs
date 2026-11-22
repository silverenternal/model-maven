//! Versioned operation/readiness contracts shared by the application and presentation.
//!
//! These are data contracts only. They do not imply that an operation or capability is
//! implemented by the current application.
#![allow(dead_code)] // Contracts are consumed incrementally by CORE-03 and domain modules.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const OPERATION_CONTRACT_VERSION: u32 = 2;

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OperationRef(pub String);

impl OperationRef {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactState {
    Available,
    Missing,
    Unsupported,
    Conditional,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessState {
    Ready,
    NeedsInput,
    Conditional,
    Unsupported,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputSourceKind {
    ProjectConfig,
    CommandLine,
    UserSession,
    SavedRun,
    ModelVariant,
    DataAdapter,
    UserCallable,
    Generated,
    External,
    Unknown,
}

/// Provenance for a value used by an operation. `revision` is scoped to the source and
/// must not be interpreted as a global or permanent identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputSource {
    pub kind: InputSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct InputBinding {
    pub key: String,
    pub state: FactState,
    pub source: InputSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    pub required: bool,
    /// Sensitive values must be redacted from diagnostics, logs, and handoff evidence.
    pub sensitive: bool,
}

impl std::fmt::Debug for InputBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = if self.sensitive {
            self.value
                .as_ref()
                .map(|_| Value::String("[REDACTED]".into()))
        } else {
            self.value.clone()
        };
        formatter
            .debug_struct("InputBinding")
            .field("key", &self.key)
            .field("state", &self.state)
            .field("source", &self.source)
            .field("value", &value)
            .field("required", &self.required)
            .field("sensitive", &self.sensitive)
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReadinessFact {
    pub key: String,
    pub state: FactState,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<InputSource>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionAvailability {
    Available,
    Conditional,
    Unsupported,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SuggestedAction {
    pub id: String,
    pub label: String,
    pub availability: ActionAvailability,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityState {
    pub id: String,
    pub state: FactState,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

/// A point-in-time answer for one selected operation; unrelated project gaps do not block it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Readiness {
    pub state: ReadinessState,
    pub facts: Vec<ReadinessFact>,
    pub capabilities: Vec<CapabilityState>,
    pub conditions: Vec<String>,
    pub actions: Vec<SuggestedAction>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationRequest {
    pub contract_version: u32,
    pub operation: OperationRef,
    pub inputs: Vec<InputBinding>,
    pub parameters: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationPlan {
    pub contract_version: u32,
    pub plan_id: String,
    pub operation: OperationRef,
    pub readiness: Readiness,
    pub inputs: Vec<InputBinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_revision: Option<String>,
}

impl OperationPlan {
    /// Reject plans that claim readiness while a required fact or capability is not known
    /// to be available. This is a structural guard, not a domain-specific readiness engine.
    pub fn validate(&self) -> Result<(), String> {
        if self.contract_version != OPERATION_CONTRACT_VERSION {
            return Err(format!(
                "unsupported operation contract version: {}",
                self.contract_version
            ));
        }
        if self.plan_id.trim().is_empty() || self.operation.as_str().trim().is_empty() {
            return Err("plan_id and operation must not be empty".into());
        }
        if !unique_nonempty(self.inputs.iter().map(|input| input.key.as_str()))
            || !unique_nonempty(self.readiness.facts.iter().map(|fact| fact.key.as_str()))
            || !unique_nonempty(
                self.readiness
                    .capabilities
                    .iter()
                    .map(|capability| capability.id.as_str()),
            )
            || !unique_nonempty(
                self.readiness
                    .actions
                    .iter()
                    .map(|action| action.id.as_str()),
            )
        {
            return Err(
                "input, fact, capability, and action identifiers must be non-empty and unique"
                    .into(),
            );
        }
        let has_required_state = |state| {
            self.readiness
                .facts
                .iter()
                .any(|fact| fact.required && fact.state == state)
                || self
                    .inputs
                    .iter()
                    .any(|input| input.required && input.state == state)
                || self
                    .readiness
                    .capabilities
                    .iter()
                    .any(|capability| capability.required && capability.state == state)
        };
        match self.readiness.state {
            ReadinessState::Ready
                if [
                    FactState::Missing,
                    FactState::Unsupported,
                    FactState::Conditional,
                    FactState::Unknown,
                ]
                .into_iter()
                .any(has_required_state) =>
            {
                return Err("ready plan contains a required fact, input, or capability that is not available".into());
            }
            ReadinessState::NeedsInput if !has_required_state(FactState::Missing) => {
                return Err(
                    "needs_input plan must identify a missing required fact or input".into(),
                );
            }
            ReadinessState::Unsupported if !has_required_state(FactState::Unsupported) => {
                return Err(
                    "unsupported plan must identify an unsupported required fact or capability"
                        .into(),
                );
            }
            ReadinessState::Conditional if self.readiness.conditions.is_empty() => {
                return Err("conditional plan must explain its condition".into());
            }
            ReadinessState::Conditional
                if self
                    .readiness
                    .conditions
                    .iter()
                    .any(|condition| condition.trim().is_empty()) =>
            {
                return Err("conditional plan must provide non-empty conditions".into());
            }
            _ => {}
        }
        Ok(())
    }
}

fn unique_nonempty<'a>(mut values: impl Iterator<Item = &'a str>) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    values.all(|value| !value.trim().is_empty() && seen.insert(value))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateDisposition {
    Preserved,
    Reset,
    Migrated,
    NotApplicable,
    Unknown,
}

/// State effects belong to EDIT/TRAIN adapters; the envelope requires explicit answers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateImpact {
    pub model_weights: StateDisposition,
    pub optimizer_state: StateDisposition,
    pub random_state: StateDisposition,
    pub data_cursor: StateDisposition,
    pub scheduler_state: StateDisposition,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditPlanState {
    Draft,
    NeedsDecision,
    ReadyToApply,
    Applied,
    Rejected,
    Stale,
    Unknown,
}

/// Stable envelope only; EDIT owns the operation-specific edit payload and algorithms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditPlanContract {
    pub plan_id: String,
    pub base_revision: String,
    pub state: EditPlanState,
    pub operations: Vec<Value>,
    pub impact: StateImpact,
}

impl EditPlanContract {
    pub fn validate(&self) -> Result<(), String> {
        if self.plan_id.trim().is_empty() || self.base_revision.trim().is_empty() {
            return Err("edit plan id and base revision must not be empty".into());
        }
        if self.state == EditPlanState::ReadyToApply && self.operations.is_empty() {
            return Err("ready-to-apply edit plan must contain at least one operation".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_ref: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_revision: Option<String>,
}

/// Invocation identity is valid only inside the named forward/probe execution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModuleInvocationIdentity {
    pub probe_id: String,
    pub forward_id: String,
    pub batch_id: String,
    pub module_instance_id: String,
    pub invocation_index: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_invocation_id: Option<String>,
    pub alias_paths: Vec<String>,
    pub input_output_paths: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callsite: Option<String>,
}

impl ModuleInvocationIdentity {
    pub fn validate(&self) -> Result<(), String> {
        if self.probe_id.trim().is_empty()
            || self.forward_id.trim().is_empty()
            || self.batch_id.trim().is_empty()
            || self.module_instance_id.trim().is_empty()
        {
            return Err(
                "invocation identity requires probe, forward, batch, and module ids".into(),
            );
        }
        if self.invocation_index == 0 {
            return Err("invocation_index is one-based".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub artifact_id: String,
    pub kind: String,
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    pub owner_operation_id: String,
}

impl ArtifactRef {
    pub fn validate(&self) -> Result<(), String> {
        if self.artifact_id.trim().is_empty()
            || self.kind.trim().is_empty()
            || self.uri.trim().is_empty()
            || self.owner_operation_id.trim().is_empty()
        {
            return Err("artifact identity, kind, uri, and owner must not be empty".into());
        }
        if self
            .sha256
            .as_ref()
            .is_some_and(|digest| !is_sha256(digest))
        {
            return Err("artifact sha256 must contain 64 hexadecimal characters".into());
        }
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationEvent {
    pub operation_id: String,
    pub sequence: u64,
    pub kind: String,
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    pub artifacts: Vec<ArtifactRef>,
}

impl OperationEvent {
    pub fn validate(&self) -> Result<(), String> {
        if self.operation_id.trim().is_empty()
            || self.kind.trim().is_empty()
            || self.state.trim().is_empty()
        {
            return Err("operation id, event kind, and state must not be empty".into());
        }
        if self
            .progress
            .is_some_and(|progress| !progress.is_finite() || !(0.0..=1.0).contains(&progress))
        {
            return Err("event progress must be a finite value between 0 and 1".into());
        }
        for artifact in &self.artifacts {
            artifact.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn plan(state: ReadinessState, facts: Vec<ReadinessFact>) -> OperationPlan {
        OperationPlan {
            contract_version: OPERATION_CONTRACT_VERSION,
            plan_id: "plan-1".into(),
            operation: OperationRef::new("forward_probe"),
            readiness: Readiness {
                state,
                facts,
                capabilities: vec![],
                conditions: vec![],
                actions: vec![],
            },
            inputs: vec![],
            base_revision: Some("source:abc".into()),
        }
    }

    #[test]
    fn ready_operation_requires_all_required_facts_to_be_available() {
        let valid = plan(
            ReadinessState::Ready,
            vec![ReadinessFact {
                key: "model".into(),
                state: FactState::Available,
                required: true,
                reason: None,
                source: None,
            }],
        );
        assert!(valid.validate().is_ok());

        let invalid = plan(
            ReadinessState::Ready,
            vec![ReadinessFact {
                key: "batch".into(),
                state: FactState::Missing,
                required: true,
                reason: Some("forward probe requires an example batch".into()),
                source: None,
            }],
        );
        assert!(invalid.validate().unwrap_err().contains("required fact"));
    }

    #[test]
    fn optional_unknown_capability_does_not_block_a_ready_operation() {
        let mut plan = plan(ReadinessState::Ready, vec![]);
        plan.readiness.capabilities.push(CapabilityState {
            id: "backward_pass".into(),
            state: FactState::Unknown,
            required: false,
            reason: Some("not requested".into()),
            provider: None,
        });
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn required_unknown_capability_and_missing_input_block_ready_plans() {
        let mut unsupported = plan(ReadinessState::Ready, vec![]);
        unsupported.readiness.capabilities.push(CapabilityState {
            id: "forward_execution".into(),
            state: FactState::Unknown,
            required: true,
            reason: Some("provider has not reported this capability".into()),
            provider: None,
        });
        assert!(unsupported.validate().unwrap_err().contains("capability"));

        let mut missing_input = plan(ReadinessState::Ready, vec![]);
        missing_input.inputs.push(InputBinding {
            key: "batch".into(),
            state: FactState::Missing,
            source: InputSource {
                kind: InputSourceKind::Unknown,
                reference: None,
                revision: None,
            },
            value: None,
            required: true,
            sensitive: false,
        });
        assert!(
            missing_input
                .validate()
                .unwrap_err()
                .contains("not available")
        );
    }

    #[test]
    fn unsupported_contract_version_is_rejected_explicitly() {
        let mut plan = plan(ReadinessState::Unknown, vec![]);
        plan.contract_version += 1;
        assert!(plan.validate().unwrap_err().contains("contract version"));
    }

    #[test]
    fn incomplete_operations_and_unknown_facts_are_valid_non_ready_plans() {
        let plan = plan(
            ReadinessState::Unknown,
            vec![ReadinessFact {
                key: "batch".into(),
                state: FactState::Unknown,
                required: true,
                reason: Some("not inspected".into()),
                source: None,
            }],
        );
        assert!(plan.validate().is_ok());
        let encoded = serde_json::to_value(&plan).unwrap();
        assert_eq!(encoded["readiness"]["state"], "unknown");
        assert_eq!(encoded["readiness"]["facts"][0]["state"], "unknown");
    }

    #[test]
    fn needs_input_and_conditional_states_must_explain_the_blocker() {
        let needs_input = plan(
            ReadinessState::NeedsInput,
            vec![ReadinessFact {
                key: "batch".into(),
                state: FactState::Missing,
                required: true,
                reason: Some("provide an example batch".into()),
                source: None,
            }],
        );
        assert!(needs_input.validate().is_ok());

        let conditional = plan(ReadinessState::Conditional, vec![]);
        assert!(conditional.validate().unwrap_err().contains("condition"));
    }

    #[test]
    fn operation_event_rejects_out_of_range_progress() {
        let event = OperationEvent {
            operation_id: "run-1".into(),
            sequence: 1,
            kind: "progress".into(),
            state: "running".into(),
            progress: Some(1.1),
            message: None,
            artifacts: vec![],
        };
        assert!(event.validate().unwrap_err().contains("between 0 and 1"));
    }

    #[test]
    fn artifact_and_invocation_identities_reject_invalid_scoped_ids() {
        let artifact = ArtifactRef {
            artifact_id: "weights-1".into(),
            kind: "checkpoint".into(),
            uri: "runs/run-1/checkpoint.pt".into(),
            media_type: None,
            sha256: Some("bad-digest".into()),
            owner_operation_id: "run-1".into(),
        };
        assert!(artifact.validate().unwrap_err().contains("64 hexadecimal"));

        let invocation = ModuleInvocationIdentity {
            probe_id: "probe-1".into(),
            forward_id: "forward-1".into(),
            batch_id: "batch-1".into(),
            module_instance_id: "module-1".into(),
            invocation_index: 0,
            parent_invocation_id: None,
            alias_paths: vec!["model.encoder".into()],
            input_output_paths: vec![],
            callsite: None,
        };
        assert!(invocation.validate().unwrap_err().contains("one-based"));
    }

    #[test]
    fn sensitive_input_values_roundtrip_but_are_marked_for_redaction() {
        let input = InputBinding {
            key: "token".into(),
            state: FactState::Available,
            source: InputSource {
                kind: InputSourceKind::CommandLine,
                reference: None,
                revision: None,
            },
            value: Some(json!("secret")),
            required: true,
            sensitive: true,
        };
        let value = serde_json::to_value(&input).unwrap();
        assert_eq!(value["sensitive"], true);
        assert!(!format!("{input:?}").contains("secret"));
        assert!(format!("{input:?}").contains("[REDACTED]"));
        assert_eq!(
            serde_json::from_value::<InputBinding>(value).unwrap(),
            input
        );
    }

    #[test]
    fn operation_references_preserve_future_extension_ids() {
        let value = json!("custom.task.inspect");
        let operation: OperationRef = serde_json::from_value(value).unwrap();
        assert_eq!(operation.as_str(), "custom.task.inspect");
    }
}
