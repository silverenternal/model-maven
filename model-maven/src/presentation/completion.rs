use clap_complete::engine::CompletionCandidate;
use std::{
    collections::BTreeSet,
    ffi::{OsStr, OsString},
    io::Write,
    path::Path,
};

use crate::{adapters::components::FileComponentRegistry, ports::ComponentResolver};

pub fn generate_script(shell_name: &str, output: &mut impl Write) -> Result<(), String> {
    let setup = match shell_name {
        "bash" => "source <(COMPLETE=bash m-model)",
        "elvish" => "eval (E:COMPLETE=elvish m-model | slurp)",
        "fish" => "COMPLETE=fish m-model | source",
        "powershell" => {
            "$env:COMPLETE = \"powershell\"; m-model | Out-String | Invoke-Expression; Remove-Item Env:\\COMPLETE"
        }
        "zsh" => "source <(COMPLETE=zsh m-model)",
        _ => return Err(format!("不支持的 shell：{shell_name}")),
    };
    writeln!(output, "{setup}").map_err(|error| error.to_string())?;
    Ok(())
}

/// Return project component IDs and exact versions without importing project code.
pub fn component_candidates_at(
    root: &Path,
    current: &OsStr,
    include_unversioned: bool,
) -> Vec<CompletionCandidate> {
    let Some(current) = current.to_str() else {
        return Vec::new();
    };
    let Ok(descriptors) = FileComponentRegistry.list(root, None) else {
        return Vec::new();
    };
    let mut values = BTreeSet::new();
    for descriptor in descriptors {
        let Some(id) = descriptor["manifest"]["component_id"].as_str() else {
            continue;
        };
        let Some(version) = descriptor["manifest"]["component_version"].as_str() else {
            continue;
        };
        if include_unversioned && id.starts_with(current) {
            values.insert(id.to_owned());
        }
        let versioned = format!("{id}@{version}");
        if versioned.starts_with(current) {
            values.insert(versioned);
        }
    }
    values.into_iter().map(CompletionCandidate::new).collect()
}

fn component_candidates(current: &OsStr, include_unversioned: bool) -> Vec<CompletionCandidate> {
    let Ok(root) = std::env::current_dir() else {
        return Vec::new();
    };
    component_candidates_at(&root, current, include_unversioned)
}

pub fn component_id_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    component_candidates(current, true)
}

pub fn exact_component_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    component_candidates(current, false)
}

/// Complete declared parameter names and simple enum/const values for registered components.
/// Candidate data comes only from the project's static component manifests.
pub fn component_param_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    let args = std::env::args_os().collect::<Vec<_>>();
    let Some(component_ref) = component_schema_ref(&args) else {
        return Vec::new();
    };
    let Ok(root) = std::env::current_dir() else {
        return Vec::new();
    };
    component_param_candidates_at(&root, &component_ref, current)
}

/// Complete parameters for manually registered component injections (`SLOT.NAME=JSON`).
pub fn injection_param_completer(current: &OsStr) -> Vec<CompletionCandidate> {
    let args = std::env::args_os().collect::<Vec<_>>();
    let words = args
        .iter()
        .map(|arg| arg.to_str())
        .collect::<Option<Vec<_>>>();
    let Some(words) = words else {
        return Vec::new();
    };
    let Some(current_text) = current.to_str() else {
        return Vec::new();
    };
    let Some((slot, _)) = current_text.split_once('.') else {
        return Vec::new();
    };
    let mut injections = words.iter().enumerate().filter_map(|(index, word)| {
        if let Some(value) = word.strip_prefix("--inject=") {
            return Some(value);
        }
        (*word == "--inject")
            .then(|| words.get(index + 1).copied())
            .flatten()
    });
    let component_ref = injections.find_map(|injection| {
        let (injected_slot, binding) = injection.split_once('=')?;
        (injected_slot == slot)
            .then(|| binding.strip_prefix("component:"))
            .flatten()
    });
    let Some(component_ref) = component_ref else {
        return Vec::new();
    };
    let Ok(root) = std::env::current_dir() else {
        return Vec::new();
    };
    schema_parameter_candidates_at(&root, component_ref, current, &format!("{slot}."))
}

fn component_schema_ref(args: &[OsString]) -> Option<String> {
    let words = args
        .iter()
        .map(|arg| arg.to_str())
        .collect::<Option<Vec<_>>>()?;
    let from_test = words
        .windows(3)
        .find(|window| window[0] == "component" && window[1] == "test")
        .and_then(|window| window.get(2))
        .filter(|value| !value.starts_with('-'))
        .map(|value| (*value).to_owned());
    if from_test.is_some() {
        return from_test;
    }
    words.iter().enumerate().rev().find_map(|(index, word)| {
        if let Some(value) = word.strip_prefix("--component=") {
            return (!value.is_empty()).then(|| value.to_owned());
        }
        (*word == "--component")
            .then(|| words.get(index + 1).copied())
            .flatten()
            .filter(|value| !value.starts_with('-'))
            .map(str::to_owned)
    })
}

/// Resolve parameter candidates for a component reference without importing its entrypoint.
pub fn component_param_candidates_at(
    root: &Path,
    component_ref: &str,
    current: &OsStr,
) -> Vec<CompletionCandidate> {
    schema_parameter_candidates_at(root, component_ref, current, "")
}

fn schema_parameter_candidates_at(
    root: &Path,
    component_ref: &str,
    current: &OsStr,
    name_prefix: &str,
) -> Vec<CompletionCandidate> {
    let Some(current) = current.to_str() else {
        return Vec::new();
    };
    let Ok(descriptors) = FileComponentRegistry.list(root, None) else {
        return Vec::new();
    };
    let (component_id, requested_version) = component_ref
        .split_once('@')
        .map_or((component_ref, None), |(id, version)| (id, Some(version)));
    let descriptor = if let Some(version) = requested_version {
        descriptors.iter().find(|item| {
            item["manifest"]["component_id"].as_str() == Some(component_id)
                && item["manifest"]["component_version"].as_str() == Some(version)
        })
    } else {
        descriptors
            .iter()
            .filter(|item| item["manifest"]["component_id"].as_str() == Some(component_id))
            .max_by(|left, right| {
                let left = left["manifest"]["component_version"]
                    .as_str()
                    .unwrap_or("0.0.0");
                let right = right["manifest"]["component_version"]
                    .as_str()
                    .unwrap_or("0.0.0");
                semver::Version::parse(left)
                    .ok()
                    .cmp(&semver::Version::parse(right).ok())
            })
    };
    let Some(properties) =
        descriptor.and_then(|item| item["manifest"]["params_schema"]["properties"].as_object())
    else {
        return Vec::new();
    };

    let mut values = BTreeSet::new();
    if let Some((name, partial)) = current.split_once('=') {
        let Some(field_name) = name.strip_prefix(name_prefix) else {
            return Vec::new();
        };
        let Some(schema) = properties.get(field_name) else {
            return Vec::new();
        };
        for value in schema_value_examples(schema) {
            let Ok(encoded) = serde_json::to_string(&value) else {
                continue;
            };
            let candidate = format!("{name_prefix}{field_name}={encoded}");
            if candidate.starts_with(current) && encoded.starts_with(partial) {
                values.insert(candidate);
            }
        }
    } else {
        for name in properties.keys() {
            let candidate = format!("{name_prefix}{name}=");
            if candidate.starts_with(current) {
                values.insert(candidate);
            }
        }
    }
    values.into_iter().map(CompletionCandidate::new).collect()
}

/// Produce bounded, schema-valid whole JSON values for static completion.
///
/// This intentionally does not introspect Python factories or emit partial JSON
/// fragments. For nested objects it composes required child values from their
/// declared enum/default values; fields without a safe declared value remain
/// for the user to enter. Arrays can be started as `[]` because the supported
/// schema has no minimum-length keyword.
fn schema_value_examples(schema: &serde_json::Value) -> Vec<serde_json::Value> {
    const MAX_CANDIDATES: usize = 32;

    if let Some(values) = schema.get("enum").and_then(serde_json::Value::as_array) {
        return values.iter().take(MAX_CANDIDATES).cloned().collect();
    }
    if let Some(default) = schema.get("default") {
        return vec![default.clone()];
    }
    match schema.get("type").and_then(serde_json::Value::as_str) {
        Some("array") => vec![serde_json::Value::Array(Vec::new())],
        Some("object") => {
            let Some(properties) = schema
                .get("properties")
                .and_then(serde_json::Value::as_object)
            else {
                return Vec::new();
            };
            let required = schema
                .get("required")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>();
            let mut partials = vec![serde_json::Map::new()];
            for name in required {
                let Some(child_schema) = properties.get(name) else {
                    return Vec::new();
                };
                let child_values = schema_value_examples(child_schema);
                if child_values.is_empty() {
                    return Vec::new();
                }
                let mut next = Vec::new();
                for partial in &partials {
                    for child in &child_values {
                        let mut value = partial.clone();
                        value.insert(name.to_owned(), child.clone());
                        next.push(value);
                        if next.len() >= MAX_CANDIDATES {
                            break;
                        }
                    }
                    if next.len() >= MAX_CANDIDATES {
                        break;
                    }
                }
                partials = next;
            }
            partials
                .into_iter()
                .map(serde_json::Value::Object)
                .collect()
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{component_candidates_at, component_param_candidates_at, generate_script};
    use std::{ffi::OsStr, fs};
    use tempfile::tempdir;

    #[test]
    fn supported_shells_get_dynamic_completion_setup() {
        for (shell, expected) in [
            ("bash", "COMPLETE=bash m-model"),
            ("elvish", "COMPLETE=elvish m-model"),
            ("fish", "COMPLETE=fish m-model"),
            ("powershell", r#"COMPLETE = "powershell""#),
            ("zsh", "COMPLETE=zsh m-model"),
        ] {
            let mut script = Vec::new();
            generate_script(shell, &mut script).unwrap();
            let script = String::from_utf8(script).unwrap();
            assert!(!script.trim().is_empty(), "{shell}");
            assert!(script.contains(expected), "{shell}: {script}");
        }
    }

    #[test]
    fn component_candidates_are_prefix_filtered_and_read_static_manifests_only() {
        let directory = tempdir().unwrap();
        let manifest_directory = directory
            .path()
            .join(".model-maven/components/example/encoder");
        fs::create_dir_all(&manifest_directory).unwrap();
        for version in ["1.0.0", "2.1.0"] {
            fs::write(
                manifest_directory.join(format!("{version}.json")),
                serde_json::json!({
                    "schema_version": 1,
                    "namespace": "example",
                    "component_id": "example/encoder",
                    "component_version": version,
                    "extension_api_version": 1,
                    "kind": "layer",
                    "entrypoint": "must_not_be_imported:factory",
                    "description": "test component",
                    "params_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": false},
                    "capabilities": {"construct": "supported"}
                })
                .to_string(),
            )
            .unwrap();
        }
        let candidates =
            component_candidates_at(directory.path(), OsStr::new("example/encoder@2"), true)
                .into_iter()
                .map(|candidate| candidate.get_value().to_string_lossy().into_owned())
                .collect::<Vec<_>>();
        assert_eq!(candidates, ["example/encoder@2.1.0"]);

        let ids = component_candidates_at(directory.path(), OsStr::new("example/enc"), true)
            .into_iter()
            .map(|candidate| candidate.get_value().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "example/encoder",
                "example/encoder@1.0.0",
                "example/encoder@2.1.0"
            ]
        );

        let exact = component_candidates_at(directory.path(), OsStr::new("example/enc"), false);
        let exact = exact
            .into_iter()
            .map(|candidate| candidate.get_value().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(exact, ["example/encoder@1.0.0", "example/encoder@2.1.0"]);
    }

    #[test]
    fn component_candidates_treat_spaces_and_shell_metacharacters_as_plain_input() {
        let directory = tempdir().unwrap();
        let manifest_directory = directory
            .path()
            .join(".model-maven/components/example/encoder");
        fs::create_dir_all(&manifest_directory).unwrap();
        fs::write(
            manifest_directory.join("1.2.3.json"),
            serde_json::json!({
                "schema_version": 1,
                "namespace": "example",
                "component_id": "example/encoder",
                "component_version": "1.2.3",
                "extension_api_version": 1,
                "kind": "layer",
                "entrypoint": "must_not_be_imported:factory",
                "description": "test component",
                "params_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": false},
                "capabilities": {"construct": "supported"}
            })
            .to_string(),
        )
        .unwrap();

        let sentinel = directory.path().join("completion-side-effect");
        let partials = [
            "example/encoder with space".to_owned(),
            format!("example/encoder;touch {}", sentinel.display()),
            format!("$(touch {})", sentinel.display()),
        ];
        for partial in &partials {
            let candidates = component_candidates_at(directory.path(), OsStr::new(partial), true);
            assert!(
                candidates.is_empty(),
                "partial input was echoed: {partial:?}"
            );
        }
        assert!(!sentinel.exists());
        assert_eq!(
            fs::read_dir(
                directory
                    .path()
                    .join(".model-maven/components/example/encoder")
            )
            .unwrap()
            .count(),
            1,
            "completion must not write into the component registry"
        );
    }

    #[test]
    fn invalid_shell_metacharacter_component_manifest_fails_closed() {
        let directory = tempdir().unwrap();
        let manifest_directory = directory
            .path()
            .join(".model-maven/components/example/encoder");
        fs::create_dir_all(&manifest_directory).unwrap();
        fs::write(
            manifest_directory.join("1.2.3.json"),
            serde_json::json!({
                "schema_version": 1,
                "namespace": "example",
                "component_id": "example/encoder;touch",
                "component_version": "1.2.3",
                "extension_api_version": 1,
                "kind": "layer",
                "entrypoint": "must_not_be_imported:factory",
                "description": "test component",
                "params_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": false},
                "capabilities": {"construct": "supported"}
            })
            .to_string(),
        )
        .unwrap();
        assert!(component_candidates_at(directory.path(), OsStr::new("example/"), true).is_empty());
        assert_eq!(
            fs::read_dir(&manifest_directory).unwrap().count(),
            1,
            "completion must not repair or rewrite a manifest"
        );
    }

    #[test]
    fn unsupported_shell_is_rejected_without_writing_a_script() {
        let mut script = Vec::new();
        assert!(generate_script("unknown", &mut script).is_err());
        assert!(script.is_empty());
    }

    #[test]
    fn component_parameter_candidates_follow_static_schema_for_names_and_enum_values() {
        let directory = tempdir().unwrap();
        let manifest_directory = directory
            .path()
            .join(".model-maven/components/example/optimizer");
        fs::create_dir_all(&manifest_directory).unwrap();
        fs::write(
            manifest_directory.join("1.2.0.json"),
            serde_json::json!({
                "schema_version": 1,
                "namespace": "example",
                "component_id": "example/optimizer",
                "component_version": "1.2.0",
                "extension_api_version": 1,
                "kind": "layer",
                "entrypoint": "must_not_be_imported:factory",
                "description": "test component",
                "params_schema": {
                    "type": "object",
                    "properties": {
                        "optimizer": {"type":"string", "enum":["adamw", "sgd"]},
                        "lr": {"type":"number", "default":0.001}
                    },
                    "required": [],
                    "additionalProperties": false
                },
                "capabilities": {"construct": "supported"}
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            manifest_directory.join("2.0.0.json"),
            serde_json::json!({
                "schema_version": 1,
                "namespace": "example",
                "component_id": "example/optimizer",
                "component_version": "2.0.0",
                "extension_api_version": 1,
                "kind": "layer",
                "entrypoint": "must_not_be_imported:factory",
                "description": "test component",
                "params_schema": {
                    "type": "object",
                    "properties": {"optimizer": {"type":"string", "enum":["lion"]}},
                    "required": [],
                    "additionalProperties": false
                },
                "capabilities": {"construct": "supported"}
            })
            .to_string(),
        )
        .unwrap();

        let names =
            component_param_candidates_at(directory.path(), "example/optimizer", OsStr::new("op"))
                .into_iter()
                .map(|candidate| candidate.get_value().to_string_lossy().into_owned())
                .collect::<Vec<_>>();
        assert_eq!(names, ["optimizer="]);

        let values = component_param_candidates_at(
            directory.path(),
            "example/optimizer@1.2.0",
            OsStr::new("optimizer=\"a"),
        )
        .into_iter()
        .map(|candidate| candidate.get_value().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
        assert_eq!(values, ["optimizer=\"adamw\""]);

        let latest_values = component_param_candidates_at(
            directory.path(),
            "example/optimizer",
            OsStr::new("optimizer=\"l"),
        )
        .into_iter()
        .map(|candidate| candidate.get_value().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
        assert_eq!(latest_values, ["optimizer=\"lion\""]);

        let unknown_version = component_param_candidates_at(
            directory.path(),
            "example/optimizer@9.9.9",
            OsStr::new(""),
        );
        assert!(unknown_version.is_empty());
    }

    #[test]
    fn component_parameter_completion_builds_bounded_nested_json_values() {
        let directory = tempdir().unwrap();
        let manifest_directory = directory
            .path()
            .join(".model-maven/components/example/model");
        fs::create_dir_all(&manifest_directory).unwrap();
        fs::write(
            manifest_directory.join("1.0.0.json"),
            serde_json::json!({
                "schema_version": 1,
                "namespace": "example",
                "component_id": "example/model",
                "component_version": "1.0.0",
                "extension_api_version": 1,
                "kind": "model",
                "entrypoint": "must_not_be_imported:factory",
                "description": "nested completion fixture",
                "params_schema": {
                    "type": "object",
                    "properties": {
                        "settings": {
                            "type": "object",
                            "required": ["optimizer", "schedule"],
                            "additionalProperties": false,
                            "properties": {
                                "optimizer": {"type":"string", "enum":["adamw", "sgd"]},
                                "schedule": {
                                    "type":"object",
                                    "required":["warmup"],
                                    "additionalProperties":false,
                                    "properties":{"warmup":{"type":"integer", "default":3}}
                                }
                            }
                        },
                        "stages": {"type":"array", "items":{"type":"string"}, "enum":[["stem", "head"], []]},
                        "enabled": {"type":"boolean", "enum":[false, true]}
                    },
                    "required": [],
                    "additionalProperties": false
                },
                "capabilities": {"construct":"supported"}
            })
            .to_string(),
        )
        .unwrap();

        let complete = |partial: &str| {
            component_param_candidates_at(
                directory.path(),
                "example/model@1.0.0",
                OsStr::new(partial),
            )
            .into_iter()
            .map(|candidate| candidate.get_value().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
        };

        assert_eq!(
            complete("settings={\"optimizer\":\"s"),
            ["settings={\"optimizer\":\"sgd\",\"schedule\":{\"warmup\":3}}"]
        );
        assert_eq!(
            complete("stages="),
            ["stages=[\"stem\",\"head\"]", "stages=[]"]
        );
        assert_eq!(complete("enabled=t"), ["enabled=true"]);
        assert!(complete("settings={\"optimizer\":\"l").is_empty());
    }
}
