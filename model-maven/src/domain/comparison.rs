use serde_json::{Value, json};

/// Pick the newest record by creation time, independent of the presentation sort
/// used by `runs list` (which groups active and failed runs first).
pub fn latest_run<'a>(runs: &'a [Value], operation: Option<&str>) -> Option<&'a Value> {
    runs.iter()
        .filter(|run| operation.is_none_or(|expected| run["operation"] == expected))
        .max_by(|left, right| {
            left["started_at_ms"]
                .as_u64()
                .unwrap_or(0)
                .cmp(&right["started_at_ms"].as_u64().unwrap_or(0))
                .then_with(|| left["id"].as_str().cmp(&right["id"].as_str()))
        })
}

fn value_at<'a>(root: &'a Value, paths: &[&[&str]]) -> Option<&'a Value> {
    paths
        .iter()
        .find_map(|path| path.iter().try_fold(root, |value, key| value.get(*key)))
        .filter(|value| !value.is_null())
}

fn contains_redacted(value: &Value) -> bool {
    match value {
        Value::String(value) => value == "[REDACTED]",
        Value::Array(values) => values.iter().any(contains_redacted),
        Value::Object(values) => values.values().any(contains_redacted),
        _ => false,
    }
}

fn compare_dimension(name: &str, left: Option<&Value>, right: Option<&Value>) -> Value {
    let status = match (left, right) {
        (Some(left), Some(right)) if contains_redacted(left) || contains_redacted(right) => "unknown",
        (Some(left), Some(right)) if left == right => "same",
        (Some(_), Some(_)) => "changed",
        _ => "unknown",
    };
    json!({
        "name": name,
        "status": status,
        "redacted": left.is_some_and(contains_redacted) || right.is_some_and(contains_redacted),
        "left": left.cloned().unwrap_or(Value::Null),
        "right": right.cloned().unwrap_or(Value::Null)
    })
}

fn compare_input_dimension(name: &str, left: &Value, right: &Value, keys: &[&str]) -> Value {
    compare_input_values(name, left, right, keys)
}

fn compare_input_values(name: &str, left: &Value, right: &Value, keys: &[&str]) -> Value {
    let collect = |inputs: &Value| {
        let mut values = serde_json::Map::new();
        let mut missing = Vec::new();
        for key in keys {
            if let Some(value) = inputs.get(*key).filter(|value| !value.is_null()) {
                values.insert((*key).to_owned(), value.clone());
            } else {
                missing.push(*key);
            }
        }
        (Value::Object(values), missing)
    };
    let (left_values, left_missing) = collect(left);
    let (right_values, right_missing) = collect(right);
    let redacted_fields = keys.iter().filter(|key| {
        left.get(**key).is_some_and(contains_redacted) || right.get(**key).is_some_and(contains_redacted)
    }).copied().collect::<Vec<_>>();
    let changed_fields = keys
        .iter()
        .filter(|key| {
            left.get(**key).is_some_and(|value| !value.is_null())
                && right.get(**key).is_some_and(|value| !value.is_null())
                && !left.get(**key).is_some_and(contains_redacted)
                && !right.get(**key).is_some_and(contains_redacted)
                && left.get(**key) != right.get(**key)
        })
        .copied()
        .collect::<Vec<_>>();
    let no_left_values = left_values
        .as_object()
        .is_none_or(serde_json::Map::is_empty);
    let no_right_values = right_values
        .as_object()
        .is_none_or(serde_json::Map::is_empty);
    let status = if !changed_fields.is_empty() {
        "changed"
    } else if no_left_values
        || no_right_values
        || !left_missing.is_empty()
        || !right_missing.is_empty()
        || !redacted_fields.is_empty()
    {
        "unknown"
    } else {
        "same"
    };
    json!({
        "name": name,
        "status": status,
        "left": left_values,
        "right": right_values,
        "left_missing_fields": left_missing,
        "right_missing_fields": right_missing,
        "redacted_fields": redacted_fields,
        "changed_fields": changed_fields
    })
}

fn compare_data(left: &Value, right: &Value) -> Value {
    let mut result = compare_input_dimension(
        "data",
        left,
        right,
        &["batch", "batch_binding", "batch_params", "data_fingerprint"],
    );
    let declared_fingerprint = |inputs: &Value| {
        inputs["data_fingerprint"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned)
    };
    let version_status = match (declared_fingerprint(left), declared_fingerprint(right)) {
        (Some(left), Some(right)) if left == right => "same",
        (Some(_), Some(_)) => "changed",
        _ => "unknown",
    };
    result["settings_status"] = result["status"].clone();
    result["declared_version_status"] = json!(version_status);
    result["content_identity"] = json!("unknown");
    if result["status"] == "same" {
        result["status"] = json!("unknown");
        result["reason"] = json!(
            "Matching factory settings or declared versions do not establish identical data contents."
        );
    } else if result["status"] == "changed" {
        result["reason"] = json!(
            "Data settings or declared versions differ; actual data contents were not independently compared."
        );
    } else {
        result["reason"] = json!(
            "Data settings or declarations are incomplete; actual data contents were not independently compared."
        );
    }
    result
}

// An omitted optimizer_params is the worker protocol's empty-parameters
// default when a factory is present. An absent factory in older records does
// not establish whether a user-managed optimizer was involved.
fn compare_optimizer_factory(left: &Value, right: &Value) -> Value {
    fn factory(inputs: &Value) -> Option<Value> {
        let name = inputs.get("optimizer_factory")?.as_str().filter(|name| !name.is_empty())?;
        let params = match inputs.get("optimizer_params") {
            None => json!({}),
            Some(value) if value.is_object() => value.clone(),
            _ => return None,
        };
        Some(json!({"factory": name, "params": params}))
    }
    let left = factory(left);
    let right = factory(right);
    let mut result = compare_dimension("optimizer_factory", left.as_ref(), right.as_ref());
    result["reason"] = json!("Compares declared factory and parameters, not optimizer implementation or state; absence may also mean a user-managed optimizer.");
    result
}

fn compare_supervision(left: &Value, right: &Value) -> Value {
    fn declaration(inputs: &Value) -> Option<Value> {
        match inputs.get("apply_supervision")?.as_bool()? {
            false => Some(json!({"enabled": false})),
            true => {
                let revision = inputs.get("expected_supervision_revision")?.as_str()?;
                if revision.len() != 64 || !revision.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                    return None;
                }
                Some(json!({"enabled": true, "revision": revision}))
            }
        }
    }
    let left = declaration(left);
    let right = declaration(right);
    let mut result = compare_dimension("supervision", left.as_ref(), right.as_ref());
    result["reason"] = json!("Compares declared activation and recorded rule revision; it does not prove equivalent runtime coverage or bindings.");
    result
}

// Compare recorded declarations only. Missing fields in older records are not
// evidence that the default was used, and paths do not identify loaded code.
fn compare_runtime_paths(left: &Value, right: &Value) -> Value {
    fn recorded(inputs: &Value, key: &str) -> Option<Value> {
        let value = inputs.get("runtime")?.get(key)?;
        let valid = if value.is_null() {
            true
        } else if key == "working_directory" {
            value.as_str().is_some_and(|s| !s.is_empty() && std::path::Path::new(s).is_absolute())
        } else {
            value.as_array().is_some_and(|items| items.iter().all(|item| {
                item.as_str().is_some_and(|s| !s.is_empty())
            }))
        };
        valid.then(|| value.clone())
    }
    let keys = ["working_directory", "project_import_roots"];
    let left_values: Vec<_> = keys.iter().map(|key| recorded(left, key)).collect();
    let right_values: Vec<_> = keys.iter().map(|key| recorded(right, key)).collect();
    let changed_fields: Vec<_> = keys.iter().enumerate().filter_map(|(i, key)| {
        match (&left_values[i], &right_values[i]) {
            (Some(a), Some(b)) if a != b => Some(*key),
            _ => None,
        }
    }).collect();
    let status = if !changed_fields.is_empty() {
        "changed"
    } else if left_values.iter().any(Option::is_none) || right_values.iter().any(Option::is_none) {
        "unknown"
    } else {
        "same"
    };
    json!({
        "name": "runtime_paths",
        "status": status,
        "changed_fields": changed_fields,
        "left_missing_fields": keys.iter().enumerate().filter_map(|(i, key)| left_values[i].is_none().then_some(*key)).collect::<Vec<_>>(),
        "right_missing_fields": keys.iter().enumerate().filter_map(|(i, key)| right_values[i].is_none().then_some(*key)).collect::<Vec<_>>(),
        "reason": "Compares recorded declarations, not effective paths, imported modules or environment identity."
    })
}

fn metric_values(report: &Value) -> Option<Value> {
    let mut metrics = serde_json::Map::new();
    if let Some(reported) = value_at(
        report,
        &[&["data", "metrics"], &["data", "evaluation", "metrics"]],
    ).and_then(Value::as_object)
    {
        for (name, value) in reported {
            if !value.is_null() {
                metrics.insert(name.clone(), value.clone());
            }
        }
    }
    let data = report.get("data")?;
    let keys = [
        "loss",
        "mean_loss",
        "accuracy",
        "f1",
        "f1_score",
        "precision",
        "recall",
        "mAP",
    ];
    for key in keys {
        if let Some(value) = data.get(key).filter(|value| !value.is_null()) {
            metrics.entry(key.to_owned()).or_insert_with(|| value.clone());
        }
    }
    (!metrics.is_empty()).then_some(Value::Object(metrics))
}

fn training_mean_metric_values(report: &Value) -> Option<Value> {
    let means = report.get("data")?.get("mean_metrics")?.as_object()?;
    let metrics = means.iter().filter(|(_, value)| !value.is_null())
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<serde_json::Map<_, _>>();
    Some(Value::Object(metrics))
}

fn sample_identity(report: &Value) -> Option<Value> {
    let identity = report.get("data")?.get("sample_identity")?;
    if identity.get("status")?.as_str()? != "reported"
        || identity.get("algorithm")?.as_str()?.is_empty()
        || !identity.get("sample_ids_sha256")?.as_str()?.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || identity.get("sample_ids_sha256")?.as_str()?.len() != 64
        || identity.get("sample_count")?.as_u64().is_none()
    {
        return None;
    }
    Some(json!({
        "algorithm": identity["algorithm"],
        "sample_count": identity["sample_count"],
        "sample_ids_sha256": identity["sample_ids_sha256"]
    }))
}

fn source_file_changes(left: &Value, right: &Value) -> Value {
    let left_manifest = value_at(left, &[&["data", "source_manifest"]]);
    let right_manifest = value_at(right, &[&["data", "source_manifest"]]);
    let (Some(left_manifest), Some(right_manifest)) = (left_manifest, right_manifest) else {
        return json!({"status":"unknown","changed_files":[],"added_files":[],"removed_files":[]});
    };
    let (Some(left_files), Some(right_files)) =
        (left_manifest.as_object(), right_manifest.as_object())
    else {
        return json!({"status":"unknown","changed_files":[],"added_files":[],"removed_files":[]});
    };
    let changed_files = left_files
        .keys()
        .filter(|path| {
            right_files
                .get(*path)
                .is_some_and(|right| *right != left_files[*path])
        })
        .cloned()
        .collect::<Vec<_>>();
    let removed_files = left_files
        .keys()
        .filter(|path| !right_files.contains_key(*path))
        .cloned()
        .collect::<Vec<_>>();
    let added_files = right_files
        .keys()
        .filter(|path| !left_files.contains_key(*path))
        .cloned()
        .collect::<Vec<_>>();
    let status = if changed_files.is_empty() && removed_files.is_empty() && added_files.is_empty() {
        "same"
    } else {
        "changed"
    };
    json!({
        "status": status,
        "changed_files": changed_files,
        "added_files": added_files,
        "removed_files": removed_files
    })
}

fn observed_call_summary(report: &Value) -> (Option<&Vec<Value>>, Option<bool>) {
    report.get("data").map_or((None, None), |data| {
        (data.get("calls").and_then(Value::as_array), data.get("trace_truncated").and_then(Value::as_bool))
    })
}

/// Compare only calls present in both persisted forward reports. The worker
/// trace is invocation based, so repeated calls to a shared module remain
/// distinct through invocation_index while aliases stay grouped in paths.
fn compare_observed_forwards(left: &Value, right: &Value) -> Value {
    let left_report = left.get("report").unwrap_or(&Value::Null);
    let right_report = right.get("report").unwrap_or(&Value::Null);
    let (left_calls, left_truncated) = observed_call_summary(left_report);
    let (right_calls, right_truncated) = observed_call_summary(right_report);
    let calls_available = left_calls.is_some() && right_calls.is_some();
    let mut left_locations = serde_json::Map::new();
    let mut right_locations = serde_json::Map::new();
    let mut malformed = false;
    let mut collect = |calls: Option<&Vec<Value>>, locations: &mut serde_json::Map<String, Value>| {
        if let Some(calls) = calls {
            for call in calls {
                let paths = call.get("paths").and_then(Value::as_array);
                let module_type = call.get("module_type").and_then(Value::as_str);
                let invocation = call.get("invocation_index").and_then(Value::as_u64);
                let (Some(paths), Some(module_type), Some(invocation)) = (paths, module_type, invocation) else {
                    malformed = true;
                    continue;
                };
                if invocation == 0 || module_type.is_empty() || paths.is_empty() || paths.iter().any(|path| path.as_str().is_none()) {
                    malformed = true;
                    continue;
                }
                let key = json!({"paths":paths,"module_type":module_type,"invocation_index":invocation}).to_string();
                let implementation = call.get("implementation").and_then(Value::as_object)
                    .and_then(|implementation| {
                        let file = implementation.get("source_file")?.as_str()?;
                        let digest = implementation.get("source_sha256")?.as_str()?;
                        (digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
                            .then(|| json!({
                                "source_file":file,
                                "source_sha256":digest,
                                "declaration_line":implementation.get("declaration_line").and_then(Value::as_u64)
                            }))
                    }).unwrap_or(Value::Null);
                locations.insert(key, json!({"paths":paths,"module_type":module_type,
                    "invocation_index":invocation,"implementation":implementation}));
            }
        }
    };
    collect(left_calls, &mut left_locations);
    collect(right_calls, &mut right_locations);
    let left_keys = left_locations.keys().cloned().collect::<std::collections::BTreeSet<_>>();
    let right_keys = right_locations.keys().cloned().collect::<std::collections::BTreeSet<_>>();
    let common = left_keys.intersection(&right_keys).cloned().collect::<Vec<_>>();
    let only_left = left_keys.difference(&right_keys).cloned().collect::<Vec<_>>();
    let only_right = right_keys.difference(&left_keys).cloned().collect::<Vec<_>>();
    let complete = calls_available && left_truncated == Some(false) && right_truncated == Some(false) && !malformed;
    let call_comparison_status = if !calls_available || left_truncated.is_none() || right_truncated.is_none() || malformed {
        "unknown"
    } else if !complete {
        "partial"
    } else if only_left.is_empty() && only_right.is_empty() {
        "same_observed_locations"
    } else {
        "changed_observed_locations"
    };
    // The compact TUI comparison renderer understands only same/changed/unknown.
    // Keep the detailed trace state separately for the full report.
    let status = match call_comparison_status {
        "same_observed_locations" => "same",
        "changed_observed_locations" => "changed",
        _ => "unknown",
    };
    let mut common_implementations = Vec::new();
    let mut changed_implementations = Vec::new();
    let mut implementation_unknown = 0usize;
    for key in &common {
        let a = &left_locations[key]["implementation"];
        let b = &right_locations[key]["implementation"];
        if a.is_null() || b.is_null() {
            implementation_unknown += 1;
        } else {
            common_implementations.push(json!({"location":left_locations[key],"left":a,"right":b}));
            if a != b { changed_implementations.push(left_locations[key].clone()); }
        }
    }
    let left_data = left_report.get("data").unwrap_or(&Value::Null);
    let right_data = right_report.get("data").unwrap_or(&Value::Null);
    let inputs = compare_dimension(
        "input_metadata",
        value_at(left_data, &[&["inputs"]]),
        value_at(right_data, &[&["inputs"]]),
    );
    let outputs = compare_dimension(
        "output_metadata",
        value_at(left_data, &[&["outputs"]]),
        value_at(right_data, &[&["outputs"]]),
    );
    let input_digest = |data: &Value| {
        let content = data.get("batch_content")?;
        let digest = content.get("digest")?.as_str()?;
        (content.get("status")?.as_str()? == "available"
            && content.get("algorithm")?.as_str()? == "sha256-forward-input-v1"
            && digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
            .then(|| digest.to_owned())
    };
    let left_digest = input_digest(left_data);
    let right_digest = input_digest(right_data);
    let input_content = json!({
        "name":"input_content",
        "status":match (left_digest.as_deref(), right_digest.as_deref()) {
            (Some(a), Some(b)) if a == b => "same",
            (Some(_), Some(_)) => "changed",
            _ => "unknown",
        },
        "left_digest":left_digest,
        "right_digest":right_digest,
        "reason":"Only matching supported batch-content digests establish reported input-content identity."
    });
    let left_inputs = left.get("inputs").unwrap_or(&Value::Null);
    let right_inputs = right.get("inputs").unwrap_or(&Value::Null);
    let model = compare_input_dimension("model_declaration", left_inputs, right_inputs, &["model", "model_params"]);
    let weights = compare_dimension(
        "weight_identity",
        value_at(left, &[&["checkpoint", "sha256"], &["report", "data", "checkpoint", "sha256"]]),
        value_at(right, &[&["checkpoint", "sha256"], &["report", "data", "checkpoint", "sha256"]]),
    );
    let environment = compare_input_values(
        "reported_environment",
        left_data.get("runtime").unwrap_or(&Value::Null),
        right_data.get("runtime").unwrap_or(&Value::Null),
        &["python_executable", "python_version", "pytorch_version", "probe_device"],
    );
    let random_state = compare_input_values("random_state_declaration", left_data, right_data, &["seed", "seed_policy"]);
    json!({
        "name":"observed_forward",
        "status":status,
        "call_comparison_status":call_comparison_status,
        "coverage":"recorded_calls_only",
        "left_trace_truncated":left_truncated,
        "right_trace_truncated":right_truncated,
        "calls_available":calls_available,
        "malformed_call_details":malformed,
        "common_invocations":common.iter().map(|key| left_locations[key].clone()).collect::<Vec<_>>(),
        "left_only_invocations":if complete { json!(only_left.iter().map(|key| left_locations[key].clone()).collect::<Vec<_>>()) } else { Value::Null },
        "right_only_invocations":if complete { json!(only_right.iter().map(|key| right_locations[key].clone()).collect::<Vec<_>>()) } else { Value::Null },
        "missing_locations":"unknown_when_trace_incomplete",
        "shared_aliases_are_grouped_in_paths":true,
        "repeated_calls_are_distinguished_by_invocation_index":true,
        "common_implementation_comparisons":common_implementations,
        "changed_common_implementations":changed_implementations,
        "implementation_identity_unknown_count":implementation_unknown,
        "inputs":inputs,
        "input_content":input_content,
        "output_metadata":outputs,
        "comparability":{
            "model_declaration":model,
            "weights":weights,
            "reported_environment":environment,
            "full_environment":"unknown",
            "random_state_declaration":random_state,
            "effective_random_state":"unknown",
            "reason":"Matching declarations do not establish identical loaded model state, full environment or effective random state."
        },
        "input_identity":"reported batch_content is required for content identity; shapes and factory declarations are insufficient",
        "reason":"Observed call differences describe these reports only. Dynamic branches not observed in either report remain unknown; this is not a complete model graph or coverage claim."
    })
}

/// Compare persisted run evidence without rerunning work or inferring causality.
pub fn compare_runs(left: &Value, right: &Value) -> Value {
    let left_inputs = &left["inputs"];
    let right_inputs = &right["inputs"];
    let left_report = &left["report"];
    let right_report = &right["report"];
    let mut dimensions = vec![
        compare_input_dimension(
            "inputs",
            left_inputs,
            right_inputs,
            &[
                "operation",
                "model",
                "model_params",
                "model_injections",
                "batch",
                "batch_params",
                "training_mode",
                "training_step",
                "training_loop",
                "training_params",
                "apply_supervision",
                "expected_supervision_revision",
                "loss",
                "optimizer",
                "lr",
                "weight_decay",
                "seed",
            ],
        ),
        compare_dimension(
            "source_code",
            value_at(left_report, &[&["data", "source_manifest"]]),
            value_at(right_report, &[&["data", "source_manifest"]]),
        ),
        compare_data(left_inputs, right_inputs),
        compare_optimizer_factory(left_inputs, right_inputs),
        compare_supervision(left_inputs, right_inputs),
        compare_input_values(
            "samples",
            &sample_identity(left_report).unwrap_or(Value::Null),
            &sample_identity(right_report).unwrap_or(Value::Null),
            &["algorithm", "sample_count", "sample_ids_sha256"],
        ),
        compare_input_dimension(
            "budget",
            left_inputs,
            right_inputs,
            &["steps", "requested_steps", "training_params"],
        ),
        compare_runtime_paths(left_inputs, right_inputs),
        compare_dimension(
            "requested_device",
            value_at(
                left_inputs,
                &[&["runtime", "device"], &["runtime", "accelerator"]],
            ),
            value_at(
                right_inputs,
                &[&["runtime", "device"], &["runtime", "accelerator"]],
            ),
        ),
        compare_input_values(
            "runtime_device",
            value_at(left_report, &[&["data", "runtime"], &["data"]]).unwrap_or(&Value::Null),
            value_at(right_report, &[&["data", "runtime"], &["data"]]).unwrap_or(&Value::Null),
            &["device", "device_backend", "device_name"],
        ),
        compare_dimension(
            "weights",
            value_at(
                left,
                &[
                    &["checkpoint", "sha256"],
                    &["report", "data", "checkpoint", "sha256"],
                ],
            ),
            value_at(
                right,
                &[
                    &["checkpoint", "sha256"],
                    &["report", "data", "checkpoint", "sha256"],
                ],
            ),
        ),
        compare_dimension(
            "metrics",
            metric_values(left_report).as_ref(),
            metric_values(right_report).as_ref(),
        ),
        compare_dimension(
            "training_mean_metrics",
            training_mean_metric_values(left_report).as_ref(),
            training_mean_metric_values(right_report).as_ref(),
        ),
        compare_observed_forwards(left, right),
    ];
    let left_snapshot = value_at(left_report, &[&["data", "source_snapshot", "sha256"]]);
    let right_snapshot = value_at(right_report, &[&["data", "source_snapshot", "sha256"]]);
    if left_snapshot.is_some() || right_snapshot.is_some() {
        dimensions.push(compare_dimension("source_snapshot", left_snapshot, right_snapshot));
    }
    if left_inputs["training_mode"] == "generic-external-loop" || right_inputs["training_mode"] == "generic-external-loop" {
        let interface = |inputs: &Value| {
            (inputs["training_mode"] == "generic-external-loop")
                .then(|| inputs.get("loop_interface").cloned().unwrap_or_else(|| json!("model-context")))
        };
        dimensions.push(compare_dimension("user_loop_interface", interface(left_inputs).as_ref(), interface(right_inputs).as_ref()));
        dimensions.push(compare_dimension(
            "user_reported_last_metrics",
            value_at(left_report, &[&["data", "last_metrics"]]).filter(|value| value.is_object()),
            value_at(right_report, &[&["data", "last_metrics"]]).filter(|value| value.is_object()),
        ));
    }
    let changed = dimensions
        .iter()
        .filter(|item| item["status"] == "changed")
        .count();
    let unknown = dimensions
        .iter()
        .filter(|item| item["status"] == "unknown" || item["status"] == "partial")
        .count();
    let left_protocol = left_report["protocol_version"].as_u64();
    let right_protocol = right_report["protocol_version"].as_u64();
    let protocol_status = match (left_protocol, right_protocol) {
        (Some(left), Some(right)) if left == right => "same",
        (Some(_), Some(_)) => "changed",
        _ => "unknown",
    };
    json!({
        "read_only": true,
        "causal_attribution": false,
        "left_run_id": left["record"]["id"],
        "right_run_id": right["record"]["id"],
        "protocol": {
            "left": left_report["protocol_version"],
            "right": right_report["protocol_version"],
            "status": protocol_status
        },
        "source_file_changes": source_file_changes(left_report, right_report),
        "dimensions": dimensions,
        "summary": {
            "changed_dimensions": changed,
            "unknown_dimensions": unknown,
            "interpretation": "Observed differences are descriptive; this comparison does not attribute metric changes to a cause."
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{compare_runs, latest_run};
    use serde_json::json;

    #[test]
    fn matching_batch_settings_without_data_fingerprint_do_not_claim_same_data() {
        let inputs = json!({"batch":"data:batch","batch_binding":null,"batch_params":{},"data_fingerprint":null});
        let result = super::compare_data(&inputs, &inputs);
        assert_eq!(result["status"], "unknown");
        assert!(
            result["reason"]
                .as_str()
                .unwrap()
                .contains("do not establish")
        );
    }

    #[test]
    fn matching_declared_versions_do_not_prove_data_content_identity() {
        let inputs = json!({"batch":"data:batch","batch_binding":"envelope","batch_params":{},"data_fingerprint":"v1"});
        let result = super::compare_data(&inputs, &inputs);
        assert_eq!(result["status"], "unknown");
        assert_eq!(result["settings_status"], "same");
        assert_eq!(result["declared_version_status"], "same");
        assert_eq!(result["content_identity"], "unknown");
        let mut changed = inputs.clone();
        changed["data_fingerprint"] = json!("v2");
        let result = super::compare_data(&inputs, &changed);
        assert_eq!(result["status"], "changed");
        assert_eq!(result["declared_version_status"], "changed");
        assert_eq!(result["content_identity"], "unknown");
    }

    #[test]
    fn latest_is_based_on_timestamp_not_status_priority() {
        let runs = vec![
            json!({"id":"run-new-failed","operation":"train","status":"failed","started_at_ms":20}),
            json!({"id":"run-old-complete","operation":"train","status":"completed","started_at_ms":10}),
            json!({"id":"run-new-probe","operation":"training_probe","status":"completed","started_at_ms":30}),
        ];
        assert_eq!(
            latest_run(&runs, Some("train")).unwrap()["id"],
            "run-new-failed"
        );
        assert_eq!(latest_run(&runs, None).unwrap()["id"], "run-new-probe");
    }

    #[test]
    fn comparison_labels_missing_evidence_unknown_and_never_claims_causality() {
        let result = compare_runs(
            &json!({"record":{"id":"left"},"inputs":{"batch":"a","data_fingerprint":"v1"},"report":{"protocol_version":1,"data":{"mean_loss":0.4}}}),
            &json!({"record":{"id":"right"},"inputs":{"batch":"b","data_fingerprint":"v1"},"report":{"protocol_version":1,"data":{"mean_loss":0.2}}}),
        );
        assert_eq!(result["read_only"], true);
        assert_eq!(result["causal_attribution"], false);
        assert_eq!(result["protocol"]["status"], "same");
        let dimensions = result["dimensions"].as_array().unwrap();
        assert!(
            dimensions
                .iter()
                .any(|item| item["name"] == "source_code" && item["status"] == "unknown")
        );
        assert!(
            dimensions
                .iter()
                .any(|item| item["name"] == "data" && item["status"] == "changed")
        );
        assert!(
            dimensions
                .iter()
                .any(|item| item["name"] == "metrics" && item["status"] == "changed")
        );
        assert!(
            dimensions
                .iter()
                .any(|item| item["name"] == "samples" && item["status"] == "unknown")
        );
    }

    #[test]
    fn comparison_detects_reported_sample_identity_changes_without_raw_ids() {
        let result = compare_runs(
            &json!({
                "record":{"id":"left"},"inputs":{},
                "report":{"protocol_version":1,"data":{"sample_identity":{
                    "status":"reported","algorithm":"ordered-step-sample-ids-sha256-v1",
                    "sample_count":4,"sample_ids_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                }}}
            }),
            &json!({
                "record":{"id":"right"},"inputs":{},
                "report":{"protocol_version":1,"data":{"sample_identity":{
                    "status":"reported","algorithm":"ordered-step-sample-ids-sha256-v1",
                    "sample_count":4,"sample_ids_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                }}}
            }),
        );
        let samples = result["dimensions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["name"] == "samples")
            .unwrap();
        assert_eq!(samples["status"], "changed");
        assert_eq!(samples["left"]["sample_count"], 4);
        assert!(
            samples["changed_fields"]
                .as_array()
                .unwrap()
                .iter()
                .any(|field| field == "sample_ids_sha256")
        );
        assert_eq!(result["causal_attribution"], false);
    }

    #[test]
    fn comparison_marks_missing_metrics_and_different_protocols_explicitly() {
        let result = compare_runs(
            &json!({"record":{"id":"left"},"inputs":{},"report":{"protocol_version":1,"data":{"metrics":{"accuracy":0.9}}}}),
            &json!({"record":{"id":"right"},"inputs":{},"report":{"protocol_version":2,"data":{}}}),
        );
        assert_eq!(result["protocol"]["status"], "changed");
        assert!(
            result["dimensions"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| { item["name"] == "metrics" && item["status"] == "unknown" })
        );
    }
}
