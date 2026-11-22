use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    static PYTORCH_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PYTORCH_AVAILABLE.get_or_init(|| {
        Command::new(python())
            .args(["-c", "import torch"])
            .status()
            .is_ok_and(|status| status.success())
    })
}

fn invoke(root: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--verbose", "--python", &python()])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report = serde_json::from_slice(stdout.trim().as_bytes()).unwrap_or_else(|error| {
        panic!(
            "expected one JSON report, got {stdout:?}: {error}; status={:?}; stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

fn assert_success(output: &std::process::Output, report: &Value) {
    assert!(
        output.status.success(),
        "unexpected failure: {report:#}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(report["status"], "success", "{report:#}");
}

#[test]
fn external_loop_can_train_and_resume_without_model_or_batch_factory() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("external_loop.py"),
        r#"
def run_loop(model, context):
    assert model is None
    for step in range(context.remaining_steps):
        context.report_sample_ids([f"external:{context.completed_steps + 1}"])
        if not context.report_step(0.5, {"updates": 1}):
            break
    return {"finished": True}

run_loop.MODEL_MAVEN_STATELESS = True
"#,
    )
    .unwrap();

    let (output, result) = invoke(
        project.path(),
        &[
            "train",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop:run_loop",
            "--steps",
            "2",
            "--checkpoint-every",
            "1",
        ],
    );
    assert_success(&output, &result);
    assert_eq!(result["data"]["training_mode"], "external-loop");
    assert_eq!(result["data"]["completed_steps"], 2);
    assert_eq!(result["data"]["checkpoint_saved"], true);
    assert_eq!(
        result["data"]["capabilities"]["parameter_finiteness"],
        "not_applicable_no_model"
    );
    assert_eq!(
        result["data"]["capabilities"]["checkpoint_model"],
        "not_applicable_no_model"
    );
    assert_eq!(result["data"]["origin"]["kind"], "user_external_loop");

    let (resumed_output, resumed) = invoke(
        project.path(),
        &[
            "train",
            "--resume",
            result["data"]["run_id"].as_str().unwrap(),
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    );
    assert_success(&resumed_output, &resumed);
    assert_eq!(resumed["data"]["training_mode"], "external-loop");
    assert_eq!(resumed["data"]["resumed_from_steps"], 2);
    assert_eq!(resumed["data"]["completed_steps"], 3);
    assert_eq!(resumed["data"]["sample_identity"]["status"], "reported");
    assert_eq!(resumed["data"]["sample_identity"]["sample_count"], 3);
}

#[test]
fn saved_train_augmentation_targets_the_selected_batch_input_during_real_training() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        r#"
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 1)
    def forward(self, values):
        return self.projection(values.flatten(1))

def build_model():
    return Model()
"#,
    )
    .unwrap();
    fs::write(
        project.path().join("data.py"),
        r#"
import torch

def build_batch():
    images = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8) / 64
    targets = torch.tensor([0, 1])
    return {"args": [images], "kwargs": {}, "loss_inputs": {"targets": targets}}

def build_eval_batch():
    images = torch.ones((2, 1, 8, 8), dtype=torch.float32)
    return {"args": [images], "kwargs": {}}
"#,
    )
    .unwrap();
    fs::write(
        project.path().join("training.py"),
        r#"
def step(model, batch, context):
    images = batch["args"][0]
    prediction = model(images)
    return {"loss": prediction.square().mean(), "metrics": {"input_height": float(images.shape[-2])}}

def manual_step(model, batch, context):
    images = batch["args"][0]
    optimizer = context.optimizer("main", model.parameters())
    optimizer.zero_grad()
    loss = model(images).square().mean()
    context.backward(loss)
    optimizer.step()
    return {"loss": loss, "metrics": {"input_height": float(images.shape[-2])}}

def run_loop(model, context):
    import torch
    while context.remaining_steps:
        batch = context.transform_batch({"args": [torch.ones((2, 1, 8, 8))], "kwargs": {}})
        images = batch["args"][0]
        loss = model(images).square().mean()
        if not context.report_step(loss, {"input_height": float(images.shape[-2])}):
            break

def evaluate(model, data, context):
    batch = context.transform_batch(data)
    images = batch["args"][0]
    prediction = model(images)
    context.report_samples(len(images))
    return {"metrics": {"input_height": float(images.shape[-2]), "prediction_mean": float(prediction.mean())}}
"#,
    ).unwrap();

    let (init_output, init) = invoke(project.path(), &["init", "--model", "model:build_model"]);
    assert_success(&init_output, &init);
    let (augment_output, augment) = invoke(
        project.path(),
        &[
            "augment",
            "add",
            "RandomResizedCrop",
            "--id",
            "crop",
            "--target",
            "args.0",
            "--param",
            "size=4",
            "--scale",
            "1,1",
        ],
    );
    assert_success(&augment_output, &augment);
    let (missing_target_output, missing_target) = invoke(
        project.path(),
        &[
            "augment",
            "add",
            "Resize",
            "--id",
            "missing-target",
            "--param",
            "size=4",
        ],
    );
    assert_success(&missing_target_output, &missing_target);
    let (blocked_output, blocked) = invoke(
        project.path(),
        &[
            "train",
            "--model",
            "model:build_model",
            "--batch",
            "data:build_batch",
            "--step",
            "training:step",
            "--steps",
            "1",
        ],
    );
    assert_eq!(blocked["status"], "error", "{blocked:#}");
    assert_eq!(
        blocked["diagnostics"][0]["code"], "AUGMENT_TARGET_REQUIRED",
        "{blocked:#}"
    );
    assert_eq!(blocked_output.status.code(), Some(2));
    let (target_output, target_report) = invoke(
        project.path(),
        &["augment", "set", "missing-target", "--target", "args.0"],
    );
    assert_success(&target_output, &target_report);
    let (remove_output, removed) = invoke(project.path(), &["augment", "remove", "missing-target"]);
    assert_success(&remove_output, &removed);
    let (output, report) = invoke(
        project.path(),
        &[
            "train",
            "--model",
            "model:build_model",
            "--batch",
            "data:build_batch",
            "--step",
            "training:step",
            "--steps",
            "1",
        ],
    );
    assert_success(&output, &report);
    assert_eq!(report["data"]["completed_steps"], 1, "{report:#}");
    assert_eq!(
        report["data"]["mean_metrics"]["input_height"], 4.0,
        "{report:#}"
    );
    assert_eq!(
        report["data"]["augmentation"]["status"], "applied",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["augmentation"]["transformed_batches"], 1,
        "{report:#}"
    );
    assert_eq!(
        report["data"]["augmentation"]["last_batch"][0]["target"],
        "args.0"
    );
    assert!(report["data"]["input_sources"]["train_augment"]["revision"].is_string());

    let (manual_output, manual) = invoke(
        project.path(),
        &[
            "try",
            "--training-mode",
            "manual-step",
            "--model",
            "model:build_model",
            "--batch",
            "data:build_batch",
            "--step",
            "training:manual_step",
            "--steps",
            "1",
        ],
    );
    assert_success(&manual_output, &manual);
    assert_eq!(
        manual["data"]["mean_metrics"]["input_height"], 4.0,
        "{manual:#}"
    );
    assert_eq!(
        manual["data"]["augmentation"]["status"], "applied",
        "{manual:#}"
    );

    let (loop_output, loop_report) = invoke(
        project.path(),
        &[
            "train",
            "--training-mode",
            "external-loop",
            "--model",
            "model:build_model",
            "--loop",
            "training:run_loop",
            "--steps",
            "1",
        ],
    );
    assert_success(&loop_output, &loop_report);
    assert_eq!(
        loop_report["data"]["mean_metrics"]["input_height"], 4.0,
        "{loop_report:#}"
    );
    assert_eq!(
        loop_report["data"]["augmentation"]["status"], "applied",
        "{loop_report:#}"
    );

    let (eval_augment_output, eval_augment) = invoke(
        project.path(),
        &[
            "augment",
            "add",
            "RandomResizedCrop",
            "--split",
            "eval",
            "--id",
            "eval-crop",
            "--target",
            "args.0",
            "--param",
            "size=4",
            "--scale",
            "1,1",
        ],
    );
    assert_success(&eval_augment_output, &eval_augment);
    let (evaluation_output, evaluation) = invoke(
        project.path(),
        &[
            "evaluate",
            "--run",
            report["data"]["run_id"].as_str().unwrap(),
            "--evaluator",
            "training:evaluate",
            "--data",
            "data:build_eval_batch",
            "--split",
            "validation",
            "--samples",
            "2",
        ],
    );
    assert_success(&evaluation_output, &evaluation);
    assert_eq!(
        evaluation["data"]["evaluation"]["metrics"]["input_height"], 4.0,
        "{evaluation:#}"
    );
    assert_eq!(
        evaluation["data"]["evaluation"]["augmentation"]["status"], "applied",
        "{evaluation:#}"
    );
    assert_eq!(
        evaluation["data"]["evaluation"]["augmentation"]["transformed_batches"],
        1
    );
}

#[test]
fn workspace_edits_during_a_run_do_not_change_frozen_code() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("external_loop.py"),
        r##"
from pathlib import Path

def run_loop(model, context, source_path):
    Path(__file__).with_name("ran_from_snapshot").touch()
    Path(source_path).write_text("# edited workspace source\\n")
    context.report_step(0.5, {"updates": 1})
    return {"finished": True}

run_loop.MODEL_MAVEN_STATELESS = True
"##,
    )
    .unwrap();
    let source_before = fs::read_to_string(project.path().join("external_loop.py")).unwrap();

    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop:run_loop",
            "--steps",
            "1",
            "--loop-param",
            &format!(
                "source_path={}",
                serde_json::to_string(&project.path().join("external_loop.py").to_string_lossy())
                    .unwrap()
            ),
        ],
    );
    assert_success(&output, &report);
    assert_eq!(
        report["data"]["source_changed_during_run"], false,
        "{report:#}"
    );
    let run_dir = project
        .path()
        .join(".model-maven/runs")
        .join(report["data"]["run_id"].as_str().unwrap());
    let frozen_source = run_dir.join("snapshot/code/external_loop.py");
    assert!(frozen_source.is_file());
    assert!(run_dir.join("snapshot/code/ran_from_snapshot").is_file());
    assert_eq!(report["data"]["code_snapshot"]["status"], "frozen");
    assert_eq!(report["data"]["code_snapshot"]["files"], 1);
    assert_eq!(fs::read_to_string(frozen_source).unwrap(), source_before);
    assert!(
        fs::read_to_string(project.path().join("external_loop.py"))
            .unwrap()
            .starts_with("# edited workspace source")
    );
}

#[test]
fn zero_step_external_try_is_inconclusive_and_formal_training_is_failed() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("external_loop.py"),
        r#"
def run_loop(model, context):
    assert model is None
    return {"reason": "no update was requested"}

run_loop.MODEL_MAVEN_STATELESS = True
"#,
    )
    .unwrap();

    let (try_output, try_report) = invoke(
        project.path(),
        &[
            "try",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop:run_loop",
            "--steps",
            "3",
        ],
    );
    assert_eq!(try_report["status"], "inconclusive", "{try_report:#}");
    assert_eq!(
        try_report["diagnostics"][0]["code"],
        "EXTERNAL_LOOP_NO_STEPS"
    );
    assert_eq!(try_report["data"]["completed_steps"], 0);
    assert_eq!(try_report["data"]["total_steps"], 3);
    assert_eq!(try_output.status.code(), Some(5));

    let (train_output, train_report) = invoke(
        project.path(),
        &[
            "train",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop:run_loop",
            "--steps",
            "3",
            "--checkpoint-every",
            "1",
        ],
    );
    assert_eq!(train_report["status"], "error", "{train_report:#}");
    assert_eq!(
        train_report["diagnostics"][0]["code"],
        "EXTERNAL_LOOP_NO_STEPS"
    );
    assert_eq!(train_output.status.code(), Some(2));
}

#[test]
fn managed_step_checks_loss_and_allows_parameters_unused_by_this_step() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        r#"
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.active = nn.Linear(2, 1)
        self.optional_route = nn.Linear(2, 1)
    def forward(self, values):
        return self.active(values)

def build_model():
    return Network()
"#,
    )
    .unwrap();
    fs::write(
        project.path().join("training.py"),
        r#"
import torch

def finite_loss(model, batch, context):
    return {"loss": model(batch).square().mean()}

def nonfinite_loss(model, batch, context):
    return {"loss": model(batch).sum() * torch.tensor(float("nan"))}
"#,
    )
    .unwrap();
    fs::write(
        project.path().join("data.py"),
        "import torch\ndef batch():\n    return torch.ones((2, 2))\n",
    )
    .unwrap();

    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--step",
            "training:finite_loss",
            "--steps",
            "1",
        ],
    );
    assert_success(&output, &report);
    assert_eq!(report["data"]["completed_steps"], 1);
    assert_eq!(report["data"]["sample_identity"]["status"], "unknown");
    assert_eq!(report["data"]["parameters_with_gradient"], 2);
    assert!(
        report["data"]["gradient_parameters"]
            .as_array()
            .unwrap()
            .iter()
            .all(|name| !name.as_str().unwrap().starts_with("optional_route."))
    );

    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--step",
            "training:nonfinite_loss",
            "--steps",
            "1",
        ],
    );
    assert!(!output.status.success());
    assert_eq!(report["diagnostics"][0]["code"], "STEP_FAILED");
    assert_eq!(report["data"]["cause_code"], "STEP_RESULT_INVALID");
    assert_eq!(report["data"]["completed_steps"], 0);
}

#[test]
fn managed_step_sample_identity_is_hashed_and_used_by_run_comparison() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "from torch import nn\ndef build_model():\n    return nn.Linear(2, 1)\n",
    )
    .unwrap();
    fs::write(
        project.path().join("data.py"),
        "import torch\ndef batch():\n    return torch.ones((2, 2))\n",
    )
    .unwrap();
    fs::write(
        project.path().join("training.py"),
        r#"
def report_ids(model, batch, context, sample_ids):
    context.report_sample_ids(sample_ids)
    return {"loss": model(batch).square().mean()}
"#,
    )
    .unwrap();

    let mut run_ids = Vec::new();
    for ids in [r#"["sample-a","sample-b"]"#, r#"["sample-c","sample-d"]"#] {
        let (output, report) = invoke(
            project.path(),
            &[
                "try",
                "--model",
                "model:build_model",
                "--batch",
                "data:batch",
                "--step",
                "training:report_ids",
                "--step-param",
                &format!("sample_ids={ids}"),
                "--steps",
                "1",
            ],
        );
        assert_success(&output, &report);
        let identity = &report["data"]["sample_identity"];
        assert_eq!(identity["status"], "reported");
        assert_eq!(identity["sample_count"], 2);
        assert_eq!(identity["reported_steps"], 1);
        assert!(identity["sample_ids_sha256"].as_str().unwrap().len() == 64);
        assert!(
            !serde_json::to_string(identity)
                .unwrap()
                .contains("sample-a")
        );
        run_ids.push(report["data"]["run_id"].as_str().unwrap().to_owned());
    }

    let (comparison_output, comparison) = invoke(
        project.path(),
        &["runs", "compare", &run_ids[0], &run_ids[1]],
    );
    assert_success(&comparison_output, &comparison);
    let samples = comparison["data"]["dimensions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|dimension| dimension["name"] == "samples")
        .unwrap();
    assert_eq!(samples["status"], "changed");
    assert!(
        samples["changed_fields"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "sample_ids_sha256")
    );

    let human = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "runs",
            "compare",
            &run_ids[0],
            &run_ids[1],
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human.status.success());
    assert!(
        String::from_utf8_lossy(&human.stdout).contains("样本身份（训练代码报告）：有差异"),
        "{}",
        String::from_utf8_lossy(&human.stdout)
    );
}

#[test]
fn managed_step_sample_identity_is_unknown_when_any_step_does_not_report_ids() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "from torch import nn\ndef build_model():\n    return nn.Linear(2, 1)\n",
    )
    .unwrap();
    fs::write(
        project.path().join("data.py"),
        "import torch\ndef batch():\n    return torch.ones((2, 2))\n",
    )
    .unwrap();
    fs::write(
        project.path().join("training.py"),
        r#"
def report_some_ids(model, batch, context):
    if context.step == 1:
        context.report_sample_ids(["sample-a", "sample-b"])
    return {"loss": model(batch).square().mean()}
"#,
    )
    .unwrap();

    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--step",
            "training:report_some_ids",
            "--steps",
            "2",
        ],
    );
    assert_success(&output, &report);
    let identity = &report["data"]["sample_identity"];
    assert_eq!(identity["status"], "unknown");
    assert_eq!(identity["reported_steps"], 1);
    assert_eq!(identity["completed_steps"], 2);
    assert!(identity.get("sample_ids_sha256").is_none());
}

#[test]
fn manual_step_context_can_report_stable_sample_ids() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "from torch import nn\ndef build_model():\n    return nn.Linear(2, 1)\n",
    )
    .unwrap();
    fs::write(
        project.path().join("data.py"),
        "import torch\ndef batch():\n    return torch.ones((2, 2))\n",
    )
    .unwrap();
    fs::write(
        project.path().join("training.py"),
        r#"
def manual_step(model, batch, context):
    context.report_sample_ids(["manual-a", "manual-b"])
    optimizer = context.optimizer("main", model.parameters())
    optimizer.zero_grad()
    loss = model(batch).square().mean()
    context.backward(loss)
    optimizer.step()
    return {"loss": loss}
"#,
    )
    .unwrap();

    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--training-mode",
            "manual-step",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--step",
            "training:manual_step",
            "--steps",
            "1",
        ],
    );
    assert_success(&output, &report);
    assert_eq!(report["data"]["sample_identity"]["status"], "reported");
    assert_eq!(report["data"]["sample_identity"]["sample_count"], 2);
}

#[test]
fn manual_step_without_registered_optimizer_update_is_rejected() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "from torch import nn\ndef build_model():\n    return nn.Linear(2, 1)\n",
    )
    .unwrap();
    fs::write(
        project.path().join("data.py"),
        "import torch\ndef batch():\n    return torch.ones((2, 2))\n",
    )
    .unwrap();
    fs::write(
        project.path().join("training.py"),
        r#"
def no_update(model, batch, context):
    return {"loss": model(batch).square().mean()}
"#,
    )
    .unwrap();

    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--training-mode",
            "manual-step",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--step",
            "training:no_update",
            "--steps",
            "1",
        ],
    );
    assert!(!output.status.success());
    assert_eq!(report["diagnostics"][0]["code"], "STEP_FAILED");
    assert_eq!(report["data"]["cause_code"], "MANUAL_STEP_NO_UPDATE");
    assert_eq!(report["data"]["completed_steps"], 0);
}
