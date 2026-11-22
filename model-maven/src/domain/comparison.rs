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

fn compare_dimension(name: &str, left: Option<&Value>, right: Option<&Value>) -> Value {
    let status = match (left, right) {
        (Some(left), Some(right)) if left == right => "same",
        (Some(_), Some(_)) => "changed",
        _ => "unknown",
    };
    json!({
        "name": name,
        "status": status,
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
            if let Some(value) = inputs.get(*key) {
                values.insert((*key).to_owned(), value.clone());
            } else {
                missing.push(*key);
            }
        }
        (Value::Object(values), missing)
    };
    let (left_values, left_missing) = collect(left);
    let (right_values, right_missing) = collect(right);
    let changed_fields = keys
        .iter()
        .filter(|key| {
            left.get(**key).is_some()
                && right.get(**key).is_some()
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

fn metric_values(report: &Value) -> Option<Value> {
    if let Some(metrics) = value_at(
        report,
        &[&["data", "metrics"], &["data", "evaluation", "metrics"]],
    ) && metrics.is_object()
    {
        return Some(metrics.clone());
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
    let mut metrics = serde_json::Map::new();
    for key in keys {
        if let Some(value) = data.get(key).filter(|value| !value.is_null()) {
            metrics.insert(key.to_owned(), value.clone());
        }
    }
    (!metrics.is_empty()).then_some(Value::Object(metrics))
}

fn sample_identity(report: &Value) -> Option<Value> {
    let identity = report.get("data")?.get("sample_identity")?;
    if identity.get("status")?.as_str()? != "reported"
        || identity.get("algorithm")?.as_str()?.is_empty()
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

/// Compare persisted run evidence without rerunning work or inferring causality.
pub fn compare_runs(left: &Value, right: &Value) -> Value {
    let left_inputs = &left["inputs"];
    let right_inputs = &right["inputs"];
    let left_report = &left["report"];
    let right_report = &right["report"];
    let dimensions = vec![
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
                "training_params",
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
        compare_dimension(
            "runtime_device",
            value_at(
                left_inputs,
                &[&["runtime", "device"], &["runtime", "accelerator"]],
            )
            .or_else(|| value_at(left_report, &[&["data", "device"]])),
            value_at(
                right_inputs,
                &[&["runtime", "device"], &["runtime", "accelerator"]],
            )
            .or_else(|| value_at(right_report, &[&["data", "device"]])),
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
    ];
    let changed = dimensions
        .iter()
        .filter(|item| item["status"] == "changed")
        .count();
    let unknown = dimensions
        .iter()
        .filter(|item| item["status"] == "unknown")
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
