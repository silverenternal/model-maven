//! Per-operation readiness evaluation; unrelated missing project settings do not block work.

use super::operation::{
    CapabilityState, FactState, InputBinding, OPERATION_CONTRACT_VERSION, OperationPlan,
    OperationRef, Readiness, ReadinessFact, ReadinessState, SuggestedAction,
};

pub struct ReadinessInput {
    pub key: String,
    pub state: FactState,
    pub required: bool,
    pub reason: Option<String>,
    pub source: Option<super::operation::InputSource>,
}

pub struct ReadinessCapability {
    pub id: String,
    pub state: FactState,
    pub required: bool,
    pub reason: Option<String>,
    pub provider: Option<String>,
}

/// Create a validated, operation-scoped plan from facts actually known by the caller.
/// Missing optional training fields therefore do not block inspect/forward operations.
// A plan is assembled from independent operation facts, conditions and actions.
#[allow(clippy::too_many_arguments)]
pub fn plan(
    plan_id: impl Into<String>,
    operation: impl Into<String>,
    inputs: Vec<InputBinding>,
    facts: Vec<ReadinessInput>,
    capabilities: Vec<ReadinessCapability>,
    mut conditions: Vec<String>,
    actions: Vec<SuggestedAction>,
    base_revision: Option<String>,
) -> Result<OperationPlan, String> {
    for fact in facts
        .iter()
        .filter(|fact| fact.required && fact.state == FactState::Conditional)
    {
        conditions.push(format!(
            "{}：{}",
            fact.key,
            fact.reason.as_deref().unwrap_or("此条件尚未确认。")
        ));
    }
    for capability in capabilities
        .iter()
        .filter(|capability| capability.required && capability.state == FactState::Conditional)
    {
        conditions.push(format!(
            "{}：{}",
            capability.id,
            capability
                .reason
                .as_deref()
                .unwrap_or("此能力仅在声明的条件满足时可用。")
        ));
    }
    for fact in facts
        .iter()
        .filter(|fact| fact.required && fact.state == FactState::Unknown)
    {
        push_condition(
            &mut conditions,
            format!(
                "{}：{}",
                fact.key,
                fact.reason
                    .as_deref()
                    .filter(|reason| !reason.trim().is_empty())
                    .unwrap_or("此项状态未知；检查原始诊断并显式确认后重试。")
            ),
        );
    }
    for capability in capabilities
        .iter()
        .filter(|capability| capability.required && capability.state == FactState::Unknown)
    {
        push_condition(
            &mut conditions,
            format!(
                "{}：{}",
                capability.id,
                capability
                    .reason
                    .as_deref()
                    .filter(|reason| !reason.trim().is_empty())
                    .unwrap_or("运行时未能确认此能力；检查环境或协议诊断后重试。")
            ),
        );
    }
    let facts = facts
        .into_iter()
        .map(|fact| ReadinessFact {
            key: fact.key,
            state: fact.state,
            required: fact.required,
            reason: fact.reason,
            source: fact.source,
        })
        .collect::<Vec<_>>();
    let capabilities = capabilities
        .into_iter()
        .map(|capability| CapabilityState {
            id: capability.id,
            state: capability.state,
            required: capability.required,
            reason: capability.reason,
            provider: capability.provider,
        })
        .collect::<Vec<_>>();
    let state = derive_state(&inputs, &facts, &capabilities, &conditions);
    let result = OperationPlan {
        contract_version: OPERATION_CONTRACT_VERSION,
        plan_id: plan_id.into(),
        operation: OperationRef::new(operation),
        readiness: Readiness {
            state,
            facts,
            capabilities,
            conditions,
            actions,
        },
        inputs,
        base_revision,
    };
    result.validate()?;
    Ok(result)
}

fn push_condition(conditions: &mut Vec<String>, condition: String) {
    if !conditions.iter().any(|existing| existing == &condition) {
        conditions.push(condition);
    }
}

fn derive_state(
    inputs: &[InputBinding],
    facts: &[ReadinessFact],
    capabilities: &[CapabilityState],
    conditions: &[String],
) -> ReadinessState {
    let required_state = |state| {
        inputs
            .iter()
            .any(|input| input.required && input.state == state)
            || facts
                .iter()
                .any(|fact| fact.required && fact.state == state)
            || capabilities
                .iter()
                .any(|capability| capability.required && capability.state == state)
    };
    if required_state(FactState::Unsupported) {
        ReadinessState::Unsupported
    } else if required_state(FactState::Missing) {
        ReadinessState::NeedsInput
    } else if required_state(FactState::Unknown) {
        ReadinessState::Unknown
    } else if required_state(FactState::Conditional) || !conditions.is_empty() {
        ReadinessState::Conditional
    } else {
        ReadinessState::Ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::operation::{ActionAvailability, InputSource, InputSourceKind};
    use serde_json::{Value, json};

    fn input(key: &str, value: Option<Value>, required: bool) -> InputBinding {
        InputBinding {
            key: key.into(),
            state: if value.is_some() {
                FactState::Available
            } else if required {
                FactState::Missing
            } else {
                FactState::Unknown
            },
            source: InputSource {
                kind: InputSourceKind::ProjectConfig,
                reference: Some(".model-maven/project.json".into()),
                revision: None,
            },
            value,
            required,
            sensitive: false,
        }
    }

    fn action() -> SuggestedAction {
        SuggestedAction {
            id: "configure_batch".into(),
            label: "选择示例批次".into(),
            availability: ActionAvailability::Available,
            reason: None,
        }
    }

    #[test]
    fn inspect_does_not_require_unrelated_training_or_batch_inputs() {
        let plan = plan(
            "inspect-1",
            "inspect_model",
            vec![
                input("model", Some(json!("project:model")), true),
                input("batch", None, false),
            ],
            vec![],
            vec![],
            vec![],
            vec![action()],
            Some("source-revision-1".into()),
        )
        .unwrap();
        assert_eq!(plan.readiness.state, ReadinessState::Ready);
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn missing_forward_batch_is_operation_scoped_and_actionable() {
        let plan = plan(
            "probe-1",
            "forward_probe",
            vec![
                input("model", Some(json!("project:model")), true),
                input("batch", None, true),
            ],
            vec![],
            vec![],
            vec![],
            vec![action()],
            None,
        )
        .unwrap();
        assert_eq!(plan.readiness.state, ReadinessState::NeedsInput);
        assert_eq!(plan.inputs[1].key, "batch");
        assert_eq!(plan.readiness.actions[0].id, "configure_batch");
    }

    #[test]
    fn unknown_required_capability_is_not_reported_as_ready() {
        let result = plan(
            "train-1",
            "managed_train",
            vec![input("model", Some(json!("project:model")), true)],
            vec![],
            vec![ReadinessCapability {
                id: "training_progress".into(),
                state: FactState::Unknown,
                required: true,
                reason: Some("backend has not declared progress support".into()),
                provider: None,
            }],
            vec![],
            vec![],
            None,
        )
        .unwrap();
        assert_eq!(result.readiness.state, ReadinessState::Unknown);
    }

    #[test]
    fn conditional_capability_remains_distinct_from_unknown() {
        let result = plan(
            "conditional-1",
            "layer_edit",
            vec![input("model", Some(json!("project:model")), true)],
            vec![],
            vec![ReadinessCapability {
                id: "model.layer_edit".into(),
                state: FactState::Conditional,
                required: true,
                reason: Some("Requires a model-specific adapter.".into()),
                provider: Some("component-registry".into()),
            }],
            vec!["model.layer_edit: Requires a model-specific adapter.".into()],
            vec![],
            None,
        )
        .unwrap();
        assert_eq!(result.readiness.state, ReadinessState::Conditional);
        assert_eq!(
            result.readiness.capabilities[0].state,
            FactState::Conditional
        );
        assert!(result.validate().is_ok());
    }
}
