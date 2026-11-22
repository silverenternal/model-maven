//! Structured diagnostics and recovery suggestions. They describe advice and do not
//! execute recovery actions by themselves.
#![allow(dead_code)] // Optional detail fields are consumed as domain modules migrate.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
    Fatal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticEvidenceState {
    Complete,
    Partial,
    Unknown,
    NotApplicable,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticScope {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_path: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryAvailability {
    Available,
    Conditional,
    Unsupported,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RecoveryAction {
    pub id: String,
    pub label: String,
    pub availability: RecoveryAvailability,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub requires_confirmation: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticDetails {
    pub severity: DiagnosticSeverity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<DiagnosticScope>,
    pub recovery_actions: Vec<RecoveryAction>,
    /// Unknown means the system has not established the evidence boundary.
    pub evidence_state: DiagnosticEvidenceState,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<DiagnosticDetails>,
}

impl Diagnostic {
    pub fn error(
        code: impl Into<String>,
        message: impl Into<String>,
        hint: Option<String>,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            hint,
            details: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_diagnostic_shape_remains_compatible() {
        let legacy = json!({
            "code":"FORWARD_FAILED",
            "message":"shape mismatch",
            "hint":"inspect the last call"
        });
        let parsed: Diagnostic = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(parsed.code, "FORWARD_FAILED");
        assert_eq!(serde_json::to_value(parsed).unwrap(), legacy);
    }

    #[test]
    fn recovery_actions_are_advice_with_explicit_availability() {
        let diagnostic = Diagnostic {
            code: "STALE_PLAN".into(),
            message: "source changed".into(),
            hint: None,
            details: Some(DiagnosticDetails {
                severity: DiagnosticSeverity::Error,
                scope: None,
                recovery_actions: vec![RecoveryAction {
                    id: "replan".into(),
                    label: "Create a new plan".into(),
                    availability: RecoveryAvailability::Available,
                    reason: None,
                    requires_confirmation: false,
                }],
                evidence_state: DiagnosticEvidenceState::Partial,
            }),
        };
        let encoded = serde_json::to_value(diagnostic).unwrap();
        assert_eq!(
            encoded["details"]["recovery_actions"][0]["availability"],
            "available"
        );
    }
}
