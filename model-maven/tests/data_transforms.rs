use serde_json::Value;
use std::{path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn run_python(program: &str) -> Value {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let package = project.join("python");
    let output = Command::new(python())
        .arg("-c")
        .arg(program)
        .env("PYTHONPATH", package)
        .output()
        .expect("start Python transform fixture");
    assert!(
        output.status.success(),
        "Python fixture failed: {}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("fixture emits JSON")
}

#[test]
fn valid_pipeline_orders_keep_declared_sequence_and_observe_real_ports() {
    let has_pillow = Command::new(python())
        .args(["-c", "from PIL import Image"])
        .status()
        .is_ok_and(|status| status.success());
    if !has_pillow {
        eprintln!("skipped image transform order: Pillow is not installed in the test interpreter");
        return;
    }
    let result = run_python(
        r#"
import json
from PIL import Image
from model_maven_worker.data.transforms import run_transform_pipeline

image = Image.new("RGB", (5, 3), color=(128, 64, 32))
resize_first = run_transform_pipeline([
    {"type": "Resize", "size": [4, 6]},
    {"type": "ToTensor"},
], image)
tensor_first = run_transform_pipeline([
    {"type": "ToTensor"},
    {"type": "Resize", "size": [4, 6]},
], image)
print(json.dumps({
  "resize_order": [step["type"] for step in resize_first["steps"]],
  "resize_types": [step["output"]["kind"] for step in resize_first["steps"]],
  "tensor_order": [step["type"] for step in tensor_first["steps"]],
  "tensor_types": [step["output"]["kind"] for step in tensor_first["steps"]],
  "shape": tensor_first["output_port"]["shape"],
  "dtype": tensor_first["output_port"]["dtype"],
}, sort_keys=True))
"#,
    );
    assert_eq!(
        result["resize_order"],
        serde_json::json!(["Resize", "ToTensor"])
    );
    assert_eq!(
        result["resize_types"],
        serde_json::json!(["image", "tensor"])
    );
    assert_eq!(
        result["tensor_order"],
        serde_json::json!(["ToTensor", "Resize"])
    );
    assert_eq!(
        result["tensor_types"],
        serde_json::json!(["tensor", "tensor"])
    );
    assert_eq!(result["shape"], serde_json::json!([3, 4, 6]));
    assert_eq!(result["dtype"], "float32");
}

#[test]
fn actual_input_and_output_contract_failures_identify_the_exact_step_and_port() {
    let result = run_python(
        r#"
import json
import torch
from model_maven_worker.data.transforms import run_transform_pipeline
from model_maven_worker.protocol import UserError

sample = torch.zeros((3, 5, 7), dtype=torch.float32)
try:
    run_transform_pipeline([{
      "type": "Resize", "size": 4,
      "input_contract": {"kind": "tensor", "shape": [3, 5, 7]},
      "output_contract": {"kind": "tensor", "shape": [3, 99, 99]},
    }], sample)
except UserError as error:
    print(json.dumps({"code": error.code, **error.data}, sort_keys=True))
"#,
    );
    assert_eq!(result["code"], "TRANSFORM_PORT_MISMATCH");
    assert_eq!(result["step"], 0);
    assert_eq!(result["port"], "output");
    assert_eq!(
        result["mismatches"]["shape"]["actual"],
        serde_json::json!([3, 4, 4])
    );
}

#[test]
fn runtime_shape_errors_are_attributed_to_the_transform_that_received_them() {
    let result = run_python(
        r#"
import json
import torch
from model_maven_worker.data.transforms import run_transform_pipeline
from model_maven_worker.protocol import UserError

try:
    run_transform_pipeline([{"type": "Resize", "size": [4, 4]}], torch.ones((4, 4)))
except UserError as error:
    print(json.dumps({"code": error.code, **error.data}, sort_keys=True))
"#,
    );
    assert_eq!(result["code"], "TRANSFORM_STEP_FAILED");
    assert_eq!(result["step"], 0);
    assert_eq!(result["type"], "Resize");
    assert!(result["input"]["shape"].is_array());
}

#[test]
fn custom_python_transform_and_seeded_random_transform_are_replayable() {
    let result = run_python(
        r#"
import json
import torch
from model_maven_worker.data.transforms import run_transform_pipeline, describe_transform_registry

def resolve(reference):
    assert reference == "fixture:scale"
    return lambda factor=1: (lambda value: value * factor)

sample = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
custom = run_transform_pipeline([
  {"type": "custom", "entrypoint": "fixture:scale", "parameters": {"factor": 2},
   "input_contract": {"kind": "tensor", "shape": [1, 3, 4]},
   "output_contract": {"kind": "tensor", "dtype": "float32"}}
], sample, resolve_entrypoint=resolve)
spec = [{"type": "RandomHorizontalFlip", "p": 0.5}]
first = run_transform_pipeline(spec, sample, seed=21)["output"]
again = run_transform_pipeline(spec, sample, seed=21)["output"]
print(json.dumps({
  "custom_sum": float(custom["output"].sum()),
  "custom_steps": custom["step_count"],
  "same_seed_equal": bool(torch.equal(first, again)),
  "registry_has_fields": "size" in describe_transform_registry()["transforms"]["Resize"]["fields"],
  "registry_version": describe_transform_registry()["version"],
}, sort_keys=True))
"#,
    );
    assert_eq!(result["custom_sum"], 132.0);
    assert_eq!(result["custom_steps"], 1);
    assert_eq!(result["same_seed_equal"], true);
    assert_eq!(result["registry_has_fields"], true);
    assert_eq!(result["registry_version"], 1);
}

#[test]
fn random_resized_crop_is_seeded_and_reports_output_shape() {
    let result = run_python(
        r#"
import json
import torch
from model_maven_worker.data.transforms import run_transform_pipeline

sample = torch.arange(3 * 12 * 16, dtype=torch.float32).reshape(3, 12, 16)
spec = [{"type": "RandomResizedCrop", "size": [8, 10], "scale": [0.4, 0.8]}]
first = run_transform_pipeline(spec, sample, seed=31)
again = run_transform_pipeline(spec, sample, seed=31)
print(json.dumps({
  "same_seed_equal": bool(torch.equal(first["output"], again["output"])),
  "shape": first["output_port"]["shape"],
  "step": first["steps"][0]["type"],
}))
"#,
    );
    assert_eq!(result["same_seed_equal"], true);
    assert_eq!(result["shape"], serde_json::json!([3, 8, 10]));
    assert_eq!(result["step"], "RandomResizedCrop");
}

#[test]
fn invalid_transform_parameters_are_rejected_without_execution_or_reordering() {
    let result = run_python(
        r#"
import json
from model_maven_worker.data.transforms import validate_transform_pipeline
from model_maven_worker.protocol import UserError

try:
    validate_transform_pipeline([
      {"type": "ToTensor"},
      {"type": "Normalize", "mean": [0.5], "std": [0]},
    ])
except UserError as error:
    print(json.dumps({"code": error.code, "data": error.data, "message": str(error)}, sort_keys=True))
"#,
    );
    assert_eq!(result["code"], "TRANSFORM_DEFINITION_INVALID");
    assert_eq!(result["data"]["location"], "transforms[1]");
    assert!(result["message"].as_str().unwrap().contains("std"));
}
