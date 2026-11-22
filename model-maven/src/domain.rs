use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub mod capability;
pub mod comparison;
pub mod diagnostic;
pub mod input;
pub mod operation;
pub mod readiness;
pub mod worker;

pub use diagnostic::Diagnostic;

pub const PROTOCOL_VERSION: u32 = capability::WORKER_PROTOCOL_VERSION;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub protocol_version: u32,
    pub operation: String,
    pub status: String,
    #[serde(default)]
    pub data: Value,
    #[serde(default)]
    pub diagnostics: Vec<Diagnostic>,
}

impl Report {
    pub fn ok(operation: &str, data: Value) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            operation: operation.into(),
            status: "success".into(),
            data,
            diagnostics: Vec::new(),
        }
    }

    pub fn error(operation: &str, code: &str, message: impl Into<String>, hint: &str) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            operation: operation.into(),
            status: "error".into(),
            data: json!({}),
            diagnostics: vec![Diagnostic::error(
                code,
                message,
                (!hint.is_empty()).then(|| hint.into()),
            )],
        }
    }

    pub fn success(&self) -> bool {
        self.status == "success"
    }

    pub fn exit_code(&self) -> i32 {
        if self.status == "inconclusive" {
            return 5;
        }
        if self.success() {
            return 0;
        }
        match self.diagnostics.first().map(|d| d.code.as_str()) {
            Some("USER_INTERRUPTED") => 130,
            Some("WORKER_TIMEOUT") => 6,
            Some("PYTHON_UNAVAILABLE" | "TORCH_UNAVAILABLE" | "ENVIRONMENT_MISMATCH") => 3,
            Some(
                "FORWARD_FAILED"
                | "EDIT_VALIDATION_FAILED"
                | "BACKWARD_FAILED"
                | "LOSS_NONFINITE"
                | "LOSS_INVALID"
                | "GRADIENT_MISSING"
                | "GRADIENT_NONFINITE"
                | "STEP_FAILED"
                | "STEP_RESULT_INVALID"
                | "CUSTOM_EDIT_FAILED"
                | "CUSTOM_EDIT_RESULT_INVALID",
            ) => 4,
            Some("WORKER_FAILED" | "PROTOCOL_ERROR" | "INTERNAL_ERROR") => 1,
            _ => 2,
        }
    }
}
