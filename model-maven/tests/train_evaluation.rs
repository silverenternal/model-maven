use serde_json::Value;
use std::{path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

fn run_python(program: &str, directory: &Path) -> Value {
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let output = Command::new(python())
        .arg("-c")
        .arg(program)
        .arg(directory)
        .env("PYTHONPATH", package)
        .output()
        .expect("start evaluation fixture");
    assert!(
        output.status.success(),
        "evaluation fixture failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("fixture emits JSON")
}

#[test]
fn evaluation_accepts_custom_multi_output_metrics_and_records_provenance() {
    if !has_torch() {
        eprintln!("skipped evaluation integration: Python needs PyTorch");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(
        directory.path().join("evaluation_fixture.py"),
        r#"
import torch
from torch import nn

class MultiOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2)
    def forward(self, values):
        output = self.shared(values)
        return {"classification": output, "embedding": output.mean(dim=0)}

def build_model(): return MultiOutput()
def build_data(): return {"values": torch.ones(3, 2), "truth": ["a", "b", "a"]}

def evaluate(model, data, context):
    output = model(data["values"])
    assert set(output) == {"classification", "embedding"}
    assert not model.training
    context.report_samples(3)
    context.report_prediction("sample-0", output["classification"][0].tolist(), data["truth"][0])
    return {
        "metrics": {"task/class_macro_f1": 0.75, "retrieval/recall_at_5": 0.5},
        "artifacts": {"heads": ["classification", "embedding"]},
    }

def invalid_metrics(model, data, context):
    return {"metrics": {"macro_f1": float("nan")}}

def missing_metrics(model, data, context):
    return {"samples": 3}

def missing_sample_count(model, data, context):
    return {"metrics": {"macro_f1": 0.75}}

def over_limit(model, data, context):
    context.report_samples(3)
    return {"metrics": {"custom": 1.0}}
def preview_overflow(model, data, context):
    for index in range(context.max_preview_samples + 1):
        context.report_prediction(str(index), {"scores": [0.1, 0.9]})
    return {"metrics": {"custom": 1.0}}
def unsolicited_preview(model, data, context):
    context.report_prediction("sample-0", [0.1, 0.9])
    return {"metrics": {"custom": 1.0}}
def duplicate_preview(model, data, context):
    context.report_prediction("sample-0", [0.1, 0.9])
    context.report_prediction("sample-0", [0.2, 0.8])
    return {"metrics": {"custom": 1.0}}
def nonfinite_preview(model, data, context):
    context.report_prediction("sample-0", [float("nan")])
    return {"metrics": {"custom": 1.0}}
"#,
    )
    .unwrap();

    let output = run_python(
        r#"
import hashlib, json, sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from evaluation_fixture import build_model
from model_maven_worker import entrypoint, verify_manifest
from model_maven_worker.protocol import UserError
from model_maven_worker.train.state import TrainingStateRuntime, scalar_value as state_scalar_value
from model_maven_worker.train.checkpoint import model_state_digest, save_training_checkpoint
from model_maven_worker.train.evaluation import EvaluationRuntime, run_evaluation

model = build_model()
source = {str(root / "evaluation_fixture.py"): hashlib.sha256((root / "evaluation_fixture.py").read_bytes()).hexdigest()}
checkpoint_dir = root / "checkpoints"
checkpoint_dir.mkdir()
payload = {
    "schema_version": 1, "training_mode": "managed-step", "completed_steps": 7,
    "model_state": model.state_dict(), "model_state_sha256": model_state_digest(torch, model),
}
manifest = save_training_checkpoint(torch, checkpoint_dir, payload)
runtime = EvaluationRuntime(torch, entrypoint, verify_manifest,
    lambda value, field: state_scalar_value(TrainingStateRuntime(torch), value, field),
    lambda _root: {"evaluator_loaded_after_execution": str("evaluation_fixture" in sys.modules)})
request = {
    "checkpoint_input": str(checkpoint_dir / manifest["file"]),
    "evaluation_entrypoint": "evaluation_fixture:evaluate",
    "evaluation_data": "evaluation_fixture:build_data",
    "evaluation_data_params": {}, "evaluation_params": {},
    "evaluation_split": "validation", "evaluation_samples": 4,
    "evaluation_preview_samples": 2,
    "source_run_id": "run-source-123", "project_root": str(root),
    "protocol_version": 1, "worker_api_version": 1,
}
result = run_evaluation(runtime, model, request, source)
request["evaluation_entrypoint"] = "evaluation_fixture:missing_sample_count"
missing_count = run_evaluation(runtime, model, request, source)
codes = {}
for entrypoint in ("invalid_metrics", "missing_metrics", "over_limit", "preview_overflow", "unsolicited_preview", "duplicate_preview", "nonfinite_preview"):
    request["evaluation_entrypoint"] = "evaluation_fixture:" + entrypoint
    request["evaluation_samples"] = 2 if entrypoint == "over_limit" else 4
    request["evaluation_preview_samples"] = 0 if entrypoint == "unsolicited_preview" else 2
    try:
        run_evaluation(runtime, model, request, source)
    except UserError as error:
        codes[entrypoint] = error.code
print(json.dumps({
    "metrics": result["evaluation"]["metrics"],
    "samples": result["evaluation"]["samples_evaluated"],
    "sample_state": result["coverage"]["samples"],
    "missing_count": missing_count["coverage"]["samples"],
    "has_provenance": result["provenance"]["data_identity_sha256"] != "",
    "evaluator_source_observed": result["provenance"]["evaluation_source_manifest"]["evaluator_loaded_after_execution"],
    "checkpoint_steps": result["checkpoint"]["completed_steps"],
    "source_run_id": result["source_run_id"],
    "codes": codes,
    "previews": result["evaluation"]["sample_previews"],
}, sort_keys=True))
"#,
        directory.path(),
    );
    assert_eq!(output["metrics"]["task/class_macro_f1"], 0.75);
    assert_eq!(output["metrics"]["retrieval/recall_at_5"], 0.5);
    assert_eq!(output["samples"], 3);
    assert_eq!(output["sample_state"], "observed");
    assert_eq!(output["missing_count"], "unknown");
    assert_eq!(output["has_provenance"], true);
    assert_eq!(output["evaluator_source_observed"], "True");
    assert_eq!(output["checkpoint_steps"], 7);
    assert_eq!(output["source_run_id"], "run-source-123");
    assert_eq!(output["previews"]["state"], "reported");
    assert_eq!(output["previews"]["limit"], 2);
    assert_eq!(output["previews"]["reported"], 1);
    assert_eq!(output["previews"]["samples"][0]["sample_id"], "sample-0");
    assert_eq!(output["previews"]["samples"][0]["target"], "a");
    assert_eq!(
        output["previews"]["samples"][0]["prediction"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        output["codes"]["invalid_metrics"],
        "EVALUATION_METRIC_INVALID"
    );
    assert_eq!(
        output["codes"]["missing_metrics"],
        "EVALUATION_METRICS_MISSING"
    );
    assert_eq!(
        output["codes"]["over_limit"],
        "EVALUATION_SAMPLE_LIMIT_EXCEEDED"
    );
    assert_eq!(
        output["codes"]["preview_overflow"],
        "EVALUATION_PREVIEW_LIMIT_EXCEEDED"
    );
    assert_eq!(
        output["codes"]["unsolicited_preview"],
        "EVALUATION_PREVIEW_NOT_REQUESTED"
    );
    assert_eq!(
        output["codes"]["duplicate_preview"],
        "EVALUATION_PREVIEW_DUPLICATE"
    );
    assert_eq!(
        output["codes"]["nonfinite_preview"],
        "EVALUATION_PREVIEW_INVALID"
    );
}

#[test]
fn evaluation_rejects_missing_dataset_and_checkpoint_integrity_failure() {
    if !has_torch() {
        eprintln!("skipped evaluation integration: Python needs PyTorch");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let output = run_python(
        r#"
import json, sys
from pathlib import Path
import torch
from torch import nn
from model_maven_worker import entrypoint, verify_manifest
from model_maven_worker.protocol import UserError
from model_maven_worker.train.state import TrainingStateRuntime, scalar_value as state_scalar_value
from model_maven_worker.train.checkpoint import model_state_digest, save_training_checkpoint
from model_maven_worker.train.evaluation import EvaluationRuntime, run_evaluation

root = Path(sys.argv[1])
model = nn.Linear(2, 1)
directory = root / "checkpoints"
directory.mkdir()
payload = {"schema_version":1,"training_mode":"managed-step","completed_steps":1,
    "model_state":model.state_dict(),"model_state_sha256":model_state_digest(torch,model)}
manifest = save_training_checkpoint(torch, directory, payload)
request = {"checkpoint_input":str(directory / manifest["file"]),
    "evaluation_entrypoint":"unavailable:evaluate","evaluation_split":"test"}
runtime = EvaluationRuntime(torch, entrypoint, verify_manifest,
    lambda value, field: state_scalar_value(TrainingStateRuntime(torch), value, field))
try:
    run_evaluation(runtime, model, request, {})
except UserError as error:
    dataset_code = error.code
path = directory / manifest["file"]
path.write_bytes(path.read_bytes() + b"tampered")
request["evaluation_data"] = "unavailable:data"
try:
    run_evaluation(runtime, model, request, {})
except UserError as error:
    integrity_code = error.code
print(json.dumps({"dataset_code":dataset_code,"integrity_code":integrity_code}, sort_keys=True))
"#,
        directory.path(),
    );
    assert_eq!(output["dataset_code"], "EVALUATION_DATA_REQUIRED");
    assert_eq!(output["integrity_code"], "CHECKPOINT_CORRUPT");
}
