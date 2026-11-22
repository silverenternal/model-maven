use serde_json::Value;
use std::{path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn run_python(program: &str) -> Value {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(python())
        .arg("-c")
        .arg(program)
        .env("PYTHONPATH", project.join("python"))
        .output()
        .expect("start Python preset fixture");
    assert!(
        output.status.success(),
        "Python preset fixture failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("fixture emits JSON")
}

#[test]
fn preset_registry_exposes_tiny_cnn_and_resnet_without_loading_optional_ml_libraries() {
    let result = run_python(
        r#"
import json, sys
from model_maven_worker.data.presets import describe_presets

catalog = describe_presets()
assert "torch" not in sys.modules
assert "torchvision" not in sys.modules
print(json.dumps({"version": catalog["version"], "presets": catalog["presets"]}, sort_keys=True))
"#,
    );
    assert_eq!(result["version"], 1);
    assert_eq!(
        result["presets"]["tiny-cnn"]["task"],
        "image-classification"
    );
    assert_eq!(
        result["presets"]["tiny-cnn"]["model"],
        "model_maven_worker.data.presets:build_tiny_cnn"
    );
    assert_eq!(
        result["presets"]["resnet18"]["task"],
        "image-classification"
    );
    assert_eq!(
        result["presets"]["resnet18"]["weights"],
        serde_json::json!(["random", "imagenet1k-v1"])
    );
    assert_eq!(result["presets"]["resnet18"]["download_is_explicit"], true);
    assert_eq!(
        result["presets"]["resnet18"]["weight_sha256"],
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    );
}

#[test]
fn tiny_cnn_uses_real_class_folder_data_and_the_shared_managed_step_engine() {
    let result = run_python(
        r#"
import json, tempfile
from pathlib import Path
import torch
from PIL import Image
from model_maven_worker.data.presets import build_tiny_cnn, classification_batch
from model_maven_worker.model.tensor_tree import tree
from model_maven_worker.runtime.entrypoints import entrypoint
from model_maven_worker.train.managed import run_managed_steps
from model_maven_worker.train.runtime import TrainingRuntime
from model_maven_worker.train.state import (
    StepContext, TrainingStateRuntime, add_finite_total, scalar_map, scalar_value,
)

torch.set_num_threads(1)
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp) / "images"
    for label, color in (("cats", (210, 40, 30)), ("dogs", (20, 90, 220))):
        directory = root / label
        directory.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (20 + index, 18), color).save(directory / f"{index}.png")
    batch = classification_batch(
        root, batch_size=4, split="train", seed=19,
        transforms=[{"type": "Resize", "size": 16}, {"type": "ToTensor"}],
    )
    cursor = classification_batch.state_dict()
    next_batch = classification_batch(
        root, batch_size=4, split="train", seed=19,
        transforms=[{"type": "Resize", "size": 16}, {"type": "ToTensor"}],
    )
    classification_batch.load_state_dict(cursor)
    model = build_tiny_cnn(num_classes=len(batch["dataset"]["class_to_idx"]), width=4)
    logits = model(batch["inputs"])
    request = {
        "operation": "training_probe",
        "training_mode": "managed-step",
        "steps": 2,
        "lr": 0.001,
        "weight_decay": 0.0,
        "batch": "model_maven_worker.data.presets:classification_batch",
        "batch_params": {
            "root": str(root), "batch_size": 4, "split": "train", "seed": 19,
            "transforms": [{"type": "Resize", "size": 16}, {"type": "ToTensor"}],
        },
        "training_step": "model_maven_worker.data.presets:classification_step",
    }
    state_runtime = TrainingStateRuntime(torch=torch)
    runtime = TrainingRuntime(
        torch=torch,
        entrypoint=entrypoint,
        tree=lambda value: tree(torch, value),
        step_context=StepContext,
        scalar_value=lambda value, field: scalar_value(state_runtime, value, field),
        scalar_map=lambda value, field: scalar_map(state_runtime, value, field),
        add_finite_total=add_finite_total,
    )
    report = run_managed_steps(runtime, model, request)
    print(json.dumps({
        "logits_shape": list(logits.shape),
        "batch_shape": list(batch["inputs"].shape),
        "target_shape": list(batch["targets"].shape),
        "sample_ids": batch["dataset"]["sample_ids"],
        "next_sample_ids": next_batch["dataset"]["sample_ids"],
        "batch_state_version": classification_batch.state_dict()["version"],
        "dataset_fingerprint": batch["dataset"]["dataset_fingerprint"],
        "completed_steps": report["completed_steps"],
        "loss_finite": torch.isfinite(torch.tensor(report["mean_loss"])).item(),
        "metric_names": sorted(report["mean_metrics"]),
        "gradient_observations": report["gradient_observations"],
    }, sort_keys=True))
"#,
    );
    assert_eq!(result["logits_shape"], serde_json::json!([4, 2]));
    assert_eq!(result["batch_shape"], serde_json::json!([4, 3, 16, 16]));
    assert_eq!(result["target_shape"], serde_json::json!([4]));
    assert_eq!(result["sample_ids"].as_array().unwrap().len(), 4);
    assert_eq!(result["next_sample_ids"].as_array().unwrap().len(), 2);
    let first = result["sample_ids"].as_array().unwrap();
    let next = result["next_sample_ids"].as_array().unwrap();
    assert!(first.iter().all(|sample| !next.contains(sample)));
    assert_eq!(result["batch_state_version"], 1);
    assert!(result["dataset_fingerprint"].as_str().is_some());
    assert_eq!(result["completed_steps"], 2);
    assert_eq!(result["loss_finite"], true);
    assert_eq!(result["metric_names"], serde_json::json!(["accuracy"]));
    assert!(result["gradient_observations"].as_u64().unwrap() > 0);
}

#[test]
fn resnet_pretrained_plan_never_downloads_without_explicit_opt_in_and_reports_cache_source() {
    let result = run_python(
        r#"
import json, tempfile
from pathlib import Path
from model_maven_worker.data.presets import (
    RESNET18_WEIGHT_FILENAME, RESNET18_WEIGHT_SHA256,
    RESNET18_WEIGHT_SHA256_PREFIX,
    RESNET18_WEIGHT_URL, build_resnet18, describe_presets, resnet18_weights_status,
)
from model_maven_worker.protocol import UserError

with tempfile.TemporaryDirectory() as temp:
    cache = Path(temp)
    status = resnet18_weights_status(cache)
    try:
        build_resnet18(num_classes=2, weights="imagenet1k-v1", cache_dir=cache)
    except UserError as error:
        result = {
            "code": error.code,
            "download_flag_required": error.data.get("download_flag_required"),
            "url": error.data.get("source"),
            "cache_path": error.data.get("cache_path"),
            "still_missing": not (cache / RESNET18_WEIGHT_FILENAME).exists(),
            "status_cached": status["cached"],
            "expected_sha256": status["expected_sha256"],
            "pinned_sha256": RESNET18_WEIGHT_SHA256,
            "hash_prefix": RESNET18_WEIGHT_SHA256_PREFIX,
            "listed_weight_id": describe_presets()["presets"]["resnet18"]["weights"][1],
            "source_constant": RESNET18_WEIGHT_URL,
        }
    else:
        raise AssertionError("uncached pretrained weights must not download implicitly")
    print(json.dumps(result, sort_keys=True))
"#,
    );
    assert_eq!(result["code"], "PRETRAINED_WEIGHTS_NOT_CACHED");
    assert_eq!(result["download_flag_required"], true);
    assert_eq!(
        result["url"],
        "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    );
    assert_eq!(result["still_missing"], true);
    assert_eq!(result["status_cached"], false);
    assert_eq!(result["expected_sha256"], result["pinned_sha256"]);
    assert_eq!(
        result["pinned_sha256"],
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    );
    assert_eq!(result["hash_prefix"], "f37072fd");
    assert_eq!(result["listed_weight_id"], "imagenet1k-v1");
    assert_eq!(result["source_constant"], result["url"]);
}

#[test]
fn invalid_resnet_cache_fails_closed_without_replacing_user_file() {
    let result = run_python(
        r#"
import json, tempfile
from pathlib import Path
from model_maven_worker.data.presets import (
    RESNET18_WEIGHT_FILENAME, build_resnet18, resnet18_weights_status,
)
from model_maven_worker.protocol import UserError

with tempfile.TemporaryDirectory() as temp:
    cache = Path(temp)
    checkpoint = cache / RESNET18_WEIGHT_FILENAME
    checkpoint.write_bytes(b"not the pinned official checkpoint")
    status = resnet18_weights_status(cache)
    try:
        build_resnet18(num_classes=2, weights="imagenet1k-v1", cache_dir=cache)
    except UserError as error:
        print(json.dumps({
            "code": error.code,
            "cache_error": error.data.get("cache_error"),
            "reported_sha256": status.get("sha256"),
            "source": error.data.get("source"),
            "download_required": error.data.get("download_required"),
            "file_preserved": checkpoint.read_bytes() == b"not the pinned official checkpoint",
        }, sort_keys=True))
    else:
        raise AssertionError("unverified checkpoint cache must be rejected")
"#,
    );
    assert_eq!(result["code"], "PRETRAINED_WEIGHTS_NOT_CACHED");
    assert_eq!(result["cache_error"], "sha256_mismatch");
    assert_eq!(
        result["source"],
        "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    );
    assert_eq!(result["download_required"], true);
    assert_eq!(result["file_preserved"], true);
    assert_eq!(result["reported_sha256"].as_str().unwrap().len(), 64);
}

#[test]
fn resnet_without_torchvision_reports_dependency_missing_without_weight_download() {
    let result = run_python(
        r#"
import builtins, json
import torch
from model_maven_worker.data.presets import build_resnet18
from model_maven_worker.protocol import UserError

original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "torchvision" or name.startswith("torchvision."):
        raise ModuleNotFoundError("simulated missing torchvision")
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import

def forbidden_download(*args, **kwargs):
    raise AssertionError("random ResNet18 construction must never download weights")
torch.hub.load_state_dict_from_url = forbidden_download
try:
    build_resnet18(num_classes=2, weights="random")
except UserError as error:
    assert error.code == "PRESET_DEPENDENCY_MISSING", error.code
    assert error.data["dependency"] == "torchvision"
    print(json.dumps({"code": error.code, "dependency": error.data["dependency"]}))
else:
    raise AssertionError("missing torchvision must be reported before a model is returned")
"#,
    );
    assert_eq!(result["code"], "PRESET_DEPENDENCY_MISSING");
    assert_eq!(result["dependency"], "torchvision");
}

#[test]
fn embedded_cli_can_construct_the_tiny_cnn_preset_without_project_copies() {
    let project = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "--json",
            "model",
            "inspect",
            "--model",
            "model_maven_worker.data.presets:build_tiny_cnn",
            "--model-param",
            "num_classes=2",
        ])
        .current_dir(project.path())
        .output()
        .expect("start embedded CLI preset inspection");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["operation"], "inspect");
    assert_eq!(report["status"], "success");
    let modules = report["data"]["modules"].as_array().expect("model modules");
    assert!(
        modules.iter().any(|module| module["type"] == "Linear"),
        "{report}"
    );
}

#[test]
fn human_cli_lists_and_searches_presets_without_project_configuration() {
    let project = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python(), "--json", "data", "presets"])
        .current_dir(project.path())
        .output()
        .expect("start preset catalog command");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "success");
    assert!(report["data"]["presets"].get("tiny-cnn").is_some());
    assert_eq!(
        report["data"]["presets"]["resnet18"]["download_is_explicit"],
        true
    );

    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python(), "--json", "data", "presets", "resnet"])
        .current_dir(project.path())
        .output()
        .expect("search preset catalog");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["data"]["query"], "resnet");
    assert_eq!(report["data"]["presets"].as_object().unwrap().len(), 1);
    assert!(report["data"]["presets"].get("resnet18").is_some());

    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "--no-interactive",
            "data",
            "presets",
            "resnet",
        ])
        .current_dir(project.path())
        .output()
        .expect("render human preset catalog");
    assert!(output.status.success());
    let rendered = String::from_utf8_lossy(&output.stdout);
    assert!(rendered.contains("resnet18"), "{rendered}");
    assert!(rendered.contains("权重选项"), "{rendered}");
}

#[test]
fn embedded_cli_runs_real_class_folder_images_through_managed_preset_training() {
    let project = tempfile::tempdir().unwrap();
    std::fs::write(
        project.path().join("project_entry.py"),
        "def marker():\n    return 'fixture'\n",
    )
    .unwrap();
    let create_images = Command::new(python())
        .arg("-c")
        .arg(
            r#"
from pathlib import Path
from PIL import Image
for name, color in (("cats", (200, 30, 20)), ("dogs", (20, 50, 200))):
    folder = Path("images") / name
    folder.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (20, 18), color).save(folder / f"{index}.png")
"#,
        )
        .current_dir(project.path())
        .output()
        .expect("create temporary class-folder images");
    assert!(
        create_images.status.success(),
        "{}",
        String::from_utf8_lossy(&create_images.stderr)
    );

    let python = python();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            python.as_str(),
            "--json",
            "try",
            "--model",
            "model_maven_worker.data.presets:build_tiny_cnn",
            "--model-param",
            "num_classes=2",
            "--model-param",
            "width=4",
            "--batch",
            "model_maven_worker.data.presets:classification_batch",
            "--batch-param",
            "root=\"images\"",
            "--batch-param",
            "batch_size=4",
            "--batch-param",
            "seed=19",
            "--batch-param",
            "transforms=[{\"type\":\"Resize\",\"size\":16},{\"type\":\"ToTensor\"}]",
            "--step",
            "model_maven_worker.data.presets:classification_step",
            "--training-mode",
            "managed-step",
            "--steps",
            "2",
            "--lr",
            "0.001",
        ])
        .current_dir(project.path())
        .output()
        .expect("run the built-in classification preset through the CLI");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["operation"], "training_probe");
    assert_eq!(report["status"], "success");
    assert_eq!(report["data"]["completed_steps"], 2);
    assert!(report["data"]["mean_loss"].as_f64().is_some());
    assert!(
        report["data"]["mean_metrics"]["accuracy"]
            .as_f64()
            .is_some()
    );
}

#[test]
fn invalid_preset_parameters_return_actionable_diagnostics() {
    let result = run_python(
        r#"
import json
from model_maven_worker.data.presets import build_tiny_cnn
from model_maven_worker.protocol import UserError
try:
    build_tiny_cnn(num_classes=1)
except UserError as error:
    print(json.dumps({"code": error.code, "message": str(error)}))
"#,
    );
    assert_eq!(result["code"], "PRESET_MODEL_PARAMETER_INVALID");
    assert!(result["message"].as_str().unwrap().contains("num_classes"));
}
