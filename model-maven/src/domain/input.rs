//! Input provenance and deterministic resolution shared by configuration and operations.

use serde_json::Value;

use super::operation::{FactState, InputBinding, InputSource, InputSourceKind};

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub value: Value,
    pub source: InputSource,
}

/// Select the effective value for an operation.
///
/// Explicit command values win. A saved run or variant is the next fixed source,
/// followed by values explicitly selected in the current UI session, project defaults,
/// and finally a product default. A direct CLI invocation never receives a UI draft
/// unless the caller passes it as the current session candidate.
// Each candidate source is independently optional and participates in a fixed precedence.
#[allow(clippy::too_many_arguments)]
pub fn resolve(
    key: impl Into<String>,
    required: bool,
    sensitive: bool,
    command_line: Option<Candidate>,
    saved_run_or_variant: Option<Candidate>,
    user_session: Option<Candidate>,
    project_config: Option<Candidate>,
    product_default: Option<Candidate>,
) -> InputBinding {
    let value = command_line
        .or(saved_run_or_variant)
        .or(user_session)
        .or(project_config)
        .or(product_default);
    match value {
        Some(candidate) => InputBinding {
            key: key.into(),
            state: FactState::Available,
            source: candidate.source,
            value: Some(candidate.value),
            required,
            sensitive,
        },
        None => InputBinding {
            key: key.into(),
            state: if required {
                FactState::Missing
            } else {
                FactState::Unknown
            },
            source: InputSource {
                kind: InputSourceKind::Unknown,
                reference: None,
                revision: None,
            },
            value: None,
            required,
            sensitive,
        },
    }
}

/// A stable, source-scoped digest for the content of a saved configuration.
/// This digest identifies configuration content only; it is not a model/data/runtime ID.
pub fn content_revision(value: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let digest = sha2::Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Hash integer/string-only payloads with the same sorted-key JSON encoding used by the
/// Python worker's `environment.digest`. Floating-point payloads are deferred to runtime,
/// because Python and Rust may choose different valid decimal spellings for the same value.
pub fn worker_payload_digest(value: &Value) -> Result<String, String> {
    let mut encoded = String::new();
    encode_worker_json(value, &mut encoded)?;
    let digest = sha2::Sha256::digest(encoded.as_bytes());
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn encode_worker_json(value: &Value, output: &mut String) -> Result<(), String> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) if value.is_i64() || value.is_u64() => {
            output.push_str(&value.to_string());
        }
        Value::Number(_) => {
            return Err(
                "floating-point payload digest must be confirmed by the Python worker".into(),
            );
        }
        Value::String(value) => {
            output.push_str(&serde_json::to_string(value).map_err(|error| error.to_string())?)
        }
        Value::Array(values) => {
            output.push('[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                encode_worker_json(item, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            for (index, (key, item)) in values.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                output.push_str(&serde_json::to_string(key).map_err(|error| error.to_string())?);
                output.push_str(": ");
                encode_worker_json(item, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

use sha2::Digest;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn candidate(value: Value, kind: InputSourceKind) -> Candidate {
        Candidate {
            value,
            source: InputSource {
                kind,
                reference: None,
                revision: None,
            },
        }
    }

    #[test]
    fn explicit_command_value_wins_over_fixed_and_session_values() {
        let binding = resolve(
            "model",
            true,
            false,
            Some(candidate(json!("cli:model"), InputSourceKind::CommandLine)),
            Some(candidate(json!("saved:model"), InputSourceKind::SavedRun)),
            Some(candidate(
                json!("draft:model"),
                InputSourceKind::UserSession,
            )),
            Some(candidate(
                json!("project:model"),
                InputSourceKind::ProjectConfig,
            )),
            None,
        );
        assert_eq!(binding.value, Some(json!("cli:model")));
        assert_eq!(binding.source.kind, InputSourceKind::CommandLine);
    }

    #[test]
    fn session_draft_is_explicit_and_never_silently_beats_a_direct_value() {
        let direct = resolve(
            "lr",
            false,
            false,
            None,
            None,
            Some(candidate(json!(0.2), InputSourceKind::UserSession)),
            Some(candidate(json!(0.1), InputSourceKind::ProjectConfig)),
            None,
        );
        assert_eq!(direct.value, Some(json!(0.2)));
        assert_eq!(direct.source.kind, InputSourceKind::UserSession);
    }

    #[test]
    fn config_revision_is_stable_and_changes_with_content() {
        assert_eq!(
            content_revision(&json!({"model":"pkg:build","schema_version":1})).unwrap(),
            content_revision(&json!({"schema_version":1,"model":"pkg:build"})).unwrap()
        );
        assert_ne!(
            content_revision(&json!({"model":"pkg:build"})).unwrap(),
            content_revision(&json!({"model":"pkg:other"})).unwrap()
        );
    }

    #[test]
    fn worker_payload_digest_matches_sorted_python_json_and_defers_floats() {
        let payload = json!({"scale": 2, "name": "silu"});
        let digest = worker_payload_digest(&payload).unwrap();
        assert_eq!(
            digest,
            "73f122819efc96dcffbc6fd36b5b5ac997e605d764b95c33e9d85d3c0321d854"
        );
        assert!(worker_payload_digest(&json!({"scale": 0.25})).is_err());
    }
}
