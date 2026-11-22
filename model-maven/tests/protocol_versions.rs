use serde_json::Value;
use std::{path::Path, process::Command};

fn validate(program: &str) -> Value {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = Command::new(python)
        .arg("-c")
        .arg(program)
        .env("PYTHONPATH", root.join("python"))
        .output()
        .expect("start worker protocol compatibility fixture");
    assert!(
        output.status.success(),
        "Python fixture failed: {}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("Python fixture emits JSON")
}

#[test]
fn worker_accepts_current_optional_versions_features_and_state_formats() {
    let result = validate(
        r#"
import json
from model_maven_worker.protocol import validate_request, compatibility_report
request = {
    "protocol_version": 1,
    "operation": "probe",
    "required_features": ["model.forward_probe", "data.dataset_probe"],
    "required_state_formats": {"training_checkpoint": 1},
}
validate_request(request)
print(json.dumps({"valid": True, "compatibility": compatibility_report()}, sort_keys=True))
"#,
    );
    assert_eq!(result["valid"], true);
    assert_eq!(result["compatibility"]["worker_protocol_version"], 1);
    assert_eq!(result["compatibility"]["worker_api_version"], 1);
    assert_eq!(
        result["compatibility"]["state_formats"]["training_checkpoint"],
        1
    );
    assert_eq!(
        result["compatibility"]["operations"]["probe"],
        serde_json::json!(["model.forward_probe", "data.dataset_probe"])
    );
}

#[test]
fn worker_rejects_unregistered_operations_before_dispatch() {
    let result = validate(
        r#"
import json
from model_maven_worker.protocol import UserError, validate_request
try:
    validate_request({"protocol_version": 1, "operation": "future_operation"})
except UserError as error:
    print(json.dumps({"code": error.code, "data": error.data}, sort_keys=True))
"#,
    );
    assert_eq!(result["code"], "OPERATION_UNSUPPORTED");
    assert_eq!(result["data"]["operation"], "future_operation");
}

#[test]
fn doctor_capabilities_derive_from_registered_operations_and_runtime_conditions() {
    let result = validate(
        r#"
import json
from model_maven_worker.protocol import compatibility_report
from model_maven_worker.runtime.capabilities import capability_catalog
compatibility = compatibility_report()
catalog = capability_catalog(torch_available=False, platform="linux")
windows = capability_catalog(torch_available=True, platform="win32")
print(json.dumps({
    "features": compatibility["required_features"],
    "catalog": catalog,
    "windows": windows,
}, sort_keys=True))
"#,
    );
    let catalog = result["catalog"].as_array().unwrap();
    let ids = catalog
        .iter()
        .map(|capability| capability["id"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    let features = result["features"]
        .as_array()
        .unwrap()
        .iter()
        .map(|feature| feature.as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(ids, features);
    for capability in catalog {
        if capability["state"] != "supported" {
            assert!(
                capability["reason"]
                    .as_str()
                    .is_some_and(|reason| !reason.is_empty())
            );
        }
    }
    assert_eq!(
        result["windows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|capability| capability["id"] == "training.detached")
            .unwrap()["state"],
        "unsupported"
    );
}

#[test]
fn legacy_and_future_checkpoint_or_environment_lock_versions_fail_with_next_steps() {
    let result = validate(
        r#"
import hashlib
import json
from pathlib import Path
import tempfile
from model_maven_worker.environment import validate_environment_lock
from model_maven_worker.protocol import UserError
from model_maven_worker.train.checkpoint import load_training_checkpoint

class FakeTorch:
    @staticmethod
    def load(path, map_location, weights_only):
        return {"schema_version": 2}

errors = []
try:
    validate_environment_lock({"schema_version": 0}, b"")
except UserError as error:
    errors.append({"code": error.code, "hint": error.hint})

with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "checkpoint-000000001.pt"
    checkpoint.write_bytes(b"fixture")
    (Path(directory) / "latest.json").write_text(json.dumps({
        "file": checkpoint.name,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    try:
        load_training_checkpoint(FakeTorch, checkpoint, None, {}, {})
    except UserError as error:
        errors.append({"code": error.code, "hint": error.hint})
print(json.dumps(errors, sort_keys=True))
"#,
    );
    assert_eq!(result[0]["code"], "ENVIRONMENT_LOCK_INVALID");
    assert!(result[0]["hint"].as_str().unwrap().contains("--refresh"));
    assert_eq!(result[1]["code"], "CHECKPOINT_VERSION_UNSUPPORTED");
    assert!(result[1]["hint"].as_str().unwrap().contains("不会猜测"));
}

#[test]
fn worker_rejects_unknown_features_and_future_state_formats_before_dispatch() {
    let result = validate(
        r#"
import json
from model_maven_worker.protocol import UserError, validate_request
results = []
for request in [
    {"protocol_version": 1, "operation": "probe", "required_features": ["future.magic"]},
            {"protocol_version": 1, "operation": "inspect", "required_features": ["training.manual_step"]},
    {"protocol_version": 1, "operation": "train", "required_state_formats": {"training_checkpoint": 2}},
    {"protocol_version": 1, "worker_api_version": 2, "operation": "probe"},
]:
    try:
        validate_request(request)
    except UserError as error:
        results.append({"code": error.code, "data": error.data})
print(json.dumps(results, sort_keys=True))
"#,
    );
    assert_eq!(result[0]["code"], "FEATURE_UNAVAILABLE");
    assert_eq!(result[0]["data"]["unsupported_features"][0], "future.magic");
    assert_eq!(result[1]["code"], "FEATURE_UNAVAILABLE");
    assert_eq!(
        result[1]["data"]["unsupported_features"][0],
        "training.manual_step"
    );
    assert_eq!(result[2]["code"], "STATE_FORMAT_UNSUPPORTED");
    assert_eq!(result[3]["code"], "WORKER_API_UNSUPPORTED");
}

#[test]
fn required_feature_names_must_be_unique_strings() {
    let result = validate(
        r#"
import json
from model_maven_worker.protocol import UserError, validate_request
try:
    validate_request({"protocol_version": 1, "operation": "inspect", "required_features": ["model.inspect", "model.inspect"]})
except UserError as error:
    print(json.dumps({"code": error.code, "message": str(error)}))
"#,
    );
    assert_eq!(result["code"], "PROTOCOL_ERROR");
}
