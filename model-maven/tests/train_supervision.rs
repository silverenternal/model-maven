use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

fn invoke(root: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", &python()])
        .args(args)
        .current_dir(root)
        .output()
        .expect("start supervision training command");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report, got {:?}: {error}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

fn project() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::write(
        root.path().join("model.py"),
        r#"import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, values):
        return {"features": values * 2 * self.scale}

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, values, targets=None):
        return self.encoder(values)

def build_model(): return Model()
"#,
    )
    .unwrap();
    fs::write(
        root.path().join("data.py"),
        "import torch\ndef batch():\n    return torch.tensor([[2.0], [-3.0]])\nbatch.MODEL_MAVEN_STATELESS = True\ndef batch_with_target():\n    return {\"args\":[torch.tensor([[2.0],[-3.0]])],\"kwargs\":{\"targets\":torch.tensor([0,1,2])}}\nbatch_with_target.MODEL_MAVEN_STATELESS = True\ndef batch_with_matching_target():\n    return {\"args\":[torch.tensor([[2.0],[-3.0]])],\"kwargs\":{\"targets\":torch.tensor([0,1])}}\nbatch_with_matching_target.MODEL_MAVEN_STATELESS = True\n",
    )
    .unwrap();
    fs::write(
        root.path().join("training.py"),
        r#"import torch

def step(model, batch, context):
    output = model(batch)
    return {"loss": output["features"].sum() * 0}

step.MODEL_MAVEN_STATELESS = True

def manual_step(model, batch, context):
    optimizer = context.optimizer("main", model.parameters())
    optimizer.zero_grad()
    output = model(batch)
    loss = output["features"].sum() * 0
    context.backward(loss)
    optimizer.step()
    return {"loss": loss}

manual_step.MODEL_MAVEN_STATELESS = True

def alignment_step(model, batch, context):
    output = model(batch["args"][0], targets=batch["kwargs"]["targets"])
    return {"loss": output["features"].sum() * 0}

alignment_step.MODEL_MAVEN_STATELESS = True

def alignment_manual_step(model, batch, context):
    optimizer = context.optimizer("main", model.parameters())
    optimizer.zero_grad()
    output = model(batch["args"][0], targets=batch["kwargs"]["targets"])
    loss = output["features"].sum() * 0
    context.backward(loss)
    optimizer.step()
    return {"loss": loss}

alignment_manual_step.MODEL_MAVEN_STATELESS = True

def run_loop(model, context):
    context.report_step(0.0)
    return {"finished": True}

run_loop.MODEL_MAVEN_STATELESS = True

def supervised_loop(model, context):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    context.register_optimizer(optimizer)
    for step in range(context.remaining_steps):
        values = torch.tensor([[0.25], [0.5]])
        batch = {"args": [values], "metadata": {
            "stable_sample_ids": True,
            "sample_ids": [f"external:{step}:0", f"external:{step}:1"],
        }}
        optimizer.zero_grad(set_to_none=True)
        with context.supervised_step(batch):
            output = model(values)
            loss = output["features"].sum() * 0
            total_loss = context.backward(loss)
            optimizer.step()
        if not context.report_step(total_loss.detach()):
            break

supervised_loop.MODEL_MAVEN_STATELESS = True
"#,
    )
    .unwrap();
    root
}

fn mark(root: &Path) {
    let (output, report) = invoke(
        root,
        &[
            "model",
            "mark",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--path",
            "model.encoder",
            "--port",
            "outputs.features",
            "--role",
            "embedding",
            "--meaning",
            "Representation constrained before downstream use.",
            "--id",
            "embedding",
        ],
    );
    assert!(output.status.success(), "{report:#}");
}

fn add_rule(root: &Path, mode: &str) {
    let mut args = vec![
        "model",
        "supervise",
        "rule",
        "add",
        "--mark",
        "embedding",
        "--kind",
        "range",
        "--mode",
        mode,
        "--min",
        "-1",
        "--max",
        "1",
        "--id",
        mode,
    ];
    if mode == "penalty" {
        args.extend(["--weight", "0.5"]);
    }
    let (output, report) = invoke(root, &args);
    assert!(output.status.success(), "{report:#}");
}

#[test]
fn require_rejects_before_gradient_or_parameter_update() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let root = project();
    mark(root.path());
    add_rule(root.path(), "require");
    let (output, report) = invoke(
        root.path(),
        &[
            "try",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--step",
            "training:step",
            "--steps",
            "1",
            "--apply-supervision",
        ],
    );
    assert!(!output.status.success(), "{report:#}");
    assert_eq!(
        report["diagnostics"][0]["code"], "STEP_FAILED",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["cause_code"], "SUPERVISION_REQUIRE_FAILED",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["cause_data"]["rule_id"], "require",
        "{report:#}"
    );
    assert_eq!(report["data"]["cause_data"]["step"], 1, "{report:#}");
    assert_eq!(
        report["data"]["cause_data"]["violation_count"], 2,
        "{report:#}"
    );
    assert_eq!(report["data"]["completed_steps"], 0, "{report:#}");
}

#[test]
fn manually_bound_batch_alignment_rejects_mismatched_targets_before_managed_or_manual_update() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let root = project();
    mark(root.path());
    let (rule_output, rule) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "embedding",
            "--kind",
            "batch-alignment",
            "--mode",
            "require",
            "--target-path",
            "batch.kwargs.targets",
            "--axis",
            "0",
            "--id",
            "batch-shape",
        ],
    );
    assert!(rule_output.status.success(), "{rule:#}");

    for (mode, step) in [
        ("managed-step", "training:alignment_step"),
        ("manual-step", "training:alignment_manual_step"),
    ] {
        let (output, report) = invoke(
            root.path(),
            &[
                "try",
                "--model",
                "model:build_model",
                "--batch",
                "data:batch_with_target",
                "--step",
                step,
                "--steps",
                "1",
                "--training-mode",
                mode,
                "--apply-supervision",
            ],
        );
        assert!(!output.status.success(), "{report:#}");
        assert_eq!(
            report["data"]["cause_code"], "SUPERVISION_REQUIRE_FAILED",
            "{report:#}"
        );
        assert_eq!(
            report["data"]["cause_data"]["rule_id"], "batch-shape",
            "{report:#}"
        );
        assert_eq!(
            report["data"]["cause_data"]["target_path"], "batch.kwargs.targets",
            "{report:#}"
        );
        assert_eq!(
            report["data"]["cause_data"]["shape"],
            serde_json::json!([2, 1]),
            "{report:#}"
        );
        assert_eq!(
            report["data"]["cause_data"]["target_shape"],
            serde_json::json!([3]),
            "{report:#}"
        );
        assert_eq!(report["data"]["completed_steps"], 0, "{mode}: {report:#}");
    }
}

#[test]
fn manually_bound_batch_alignment_persists_measured_target_evidence_in_training_report() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let root = project();
    mark(root.path());
    let (rule_output, rule) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "embedding",
            "--kind",
            "batch-alignment",
            "--mode",
            "require",
            "--target-path",
            "batch.kwargs.targets",
            "--axis",
            "0",
            "--id",
            "batch-shape",
        ],
    );
    assert!(rule_output.status.success(), "{rule:#}");

    let (output, report) = invoke(
        root.path(),
        &[
            "train",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch_with_matching_target",
            "--step",
            "training:alignment_step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
            "--apply-supervision",
        ],
    );
    assert!(output.status.success(), "{report:#}");
    assert_eq!(
        report["data"]["supervision"]["coverage_status"], "partial",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["supervision"]["sample_localization"]["status"], "insufficient",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["supervision"]["sample_localization"]["stable_ids_available"], false,
        "{report:#}"
    );
    let evidence = &report["data"]["supervision"]["rules"][0]["last_target_evidence"];
    assert_eq!(
        evidence["target_path"], "batch.kwargs.targets",
        "{report:#}"
    );
    assert_eq!(evidence["axis"], 0, "{report:#}");
    assert_eq!(
        evidence["model_shape"],
        serde_json::json!([2, 1]),
        "{report:#}"
    );
    assert_eq!(
        evidence["target_shape"],
        serde_json::json!([2]),
        "{report:#}"
    );
    assert_eq!(evidence["matched"], true, "{report:#}");
    assert_eq!(evidence["step"], 1, "{report:#}");
}

#[test]
fn explicit_penalty_changes_managed_loss_and_is_reported() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let root = project();
    mark(root.path());
    add_rule(root.path(), "penalty");
    let (default_output, default_report) = invoke(
        root.path(),
        &[
            "try",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--step",
            "training:step",
            "--steps",
            "1",
        ],
    );
    assert!(default_output.status.success(), "{default_report:#}");
    assert!(default_report["data"].get("supervision").is_none());
    assert_eq!(default_report["data"]["mean_loss"], 0.0);
    let (output, report) = invoke(
        root.path(),
        &[
            "try",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--step",
            "training:step",
            "--steps",
            "1",
            "--apply-supervision",
        ],
    );
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["data"]["objective_applied"], true, "{report:#}");
    assert_eq!(
        report["data"]["mean_losses"]["supervision/penalty"], 8.5,
        "{report:#}"
    );
    assert_eq!(
        report["data"]["supervision"]["rules"][0]["violation_count"],
        2
    );

    let (formal_output, formal) = invoke(
        root.path(),
        &[
            "train",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--step",
            "training:step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
            "--apply-supervision",
        ],
    );
    assert!(formal_output.status.success(), "{formal:#}");
    assert_eq!(formal["data"]["checkpoint_saved"], true, "{formal:#}");
    assert_eq!(
        formal["data"]["supervision"]["status"], "applied",
        "{formal:#}"
    );
    let run_id = formal["data"]["run_id"].as_str().unwrap();
    let (show_output, show) = invoke(root.path(), &["runs", "show", run_id]);
    assert!(show_output.status.success(), "{show:#}");
    assert_eq!(show["data"]["inputs"]["apply_supervision"], true);
    assert_eq!(
        show["data"]["report"]["data"]["supervision"]["status"],
        "applied"
    );
    let (resume_output, resumed) = invoke(
        root.path(),
        &[
            "train",
            "--resume",
            run_id,
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
            "--apply-supervision",
        ],
    );
    assert!(resume_output.status.success(), "{resumed:#}");
    assert_eq!(resumed["data"]["completed_steps"], 2, "{resumed:#}");
    assert_eq!(resumed["data"]["supervision"]["status"], "applied");

    let (rule_output, rule_report) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "embedding",
            "--kind",
            "finite",
            "--mode",
            "observe",
            "--id",
            "new-observation",
        ],
    );
    assert!(rule_output.status.success(), "{rule_report:#}");
    let resumed_run_id = resumed["data"]["run_id"].as_str().unwrap();
    let (changed_output, changed) = invoke(
        root.path(),
        &[
            "train",
            "--resume",
            resumed_run_id,
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
            "--apply-supervision",
        ],
    );
    assert!(!changed_output.status.success(), "{changed:#}");
    assert_eq!(
        changed["diagnostics"][0]["code"], "RESUME_INPUTS_INCOMPATIBLE",
        "{changed:#}"
    );
}

#[test]
fn manual_step_applies_rules_through_context_backward_and_tracked_optimizer() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let root = project();
    mark(root.path());
    add_rule(root.path(), "require");
    let (output, report) = invoke(
        root.path(),
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
            "--apply-supervision",
        ],
    );
    assert!(!output.status.success(), "{report:#}");
    assert_eq!(
        report["diagnostics"][0]["code"], "STEP_FAILED",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["cause_code"], "SUPERVISION_REQUIRE_FAILED",
        "{report:#}"
    );
    assert_eq!(report["data"]["completed_steps"], 0, "{report:#}");

    let penalty_root = project();
    mark(penalty_root.path());
    add_rule(penalty_root.path(), "penalty");
    let (penalty_output, penalty_report) = invoke(
        penalty_root.path(),
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
            "--apply-supervision",
        ],
    );
    assert!(penalty_output.status.success(), "{penalty_report:#}");
    assert_eq!(
        penalty_report["data"]["objective_applied"], true,
        "{penalty_report:#}"
    );
    assert_eq!(
        penalty_report["data"]["mean_losses"]["supervision/penalty"],
        8.5
    );
    assert_eq!(
        penalty_report["data"]["supervision"]["training_mode"],
        "manual-step"
    );
}

#[test]
fn external_loop_can_explicitly_apply_and_report_supervision() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let root = project();
    mark(root.path());
    let (rule_output, rule_report) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "embedding",
            "--kind",
            "range",
            "--mode",
            "observe",
            "--max",
            "0.75",
            "--axis",
            "0",
            "--id",
            "external-range",
        ],
    );
    assert!(rule_output.status.success(), "{rule_report:#}");
    let (penalty_output, penalty_report) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "embedding",
            "--kind",
            "range",
            "--mode",
            "penalty",
            "--max",
            "0.75",
            "--weight",
            "1",
            "--axis",
            "0",
            "--id",
            "external-penalty",
        ],
    );
    assert!(penalty_output.status.success(), "{penalty_report:#}");
    let (output, report) = invoke(
        root.path(),
        &[
            "try",
            "--training-mode",
            "external-loop",
            "--loop",
            "training:supervised_loop",
            "--model",
            "model:build_model",
            "--steps",
            "2",
            "--apply-supervision",
        ],
    );
    assert!(output.status.success(), "{report:#}");
    assert_eq!(
        report["data"]["supervision"]["training_mode"], "external-loop",
        "{report:#}"
    );
    let coverage = &report["data"]["supervision"]["rules"][0]["sample_coverage"];
    assert_eq!(coverage["localized_batches"], 2, "{report:#}");
    assert_eq!(coverage["sample_occurrences"], 4, "{report:#}");
    assert_eq!(coverage["unique_sample_count_lower_bound"], 4, "{report:#}");
    assert!(
        report["data"]["mean_metrics"]["supervision/external-penalty"]
            .as_f64()
            .is_some_and(|value| value > 0.0),
        "{report:#}"
    );

    let (require_output, require_report) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "embedding",
            "--kind",
            "range",
            "--mode",
            "require",
            "--max",
            "0.75",
            "--axis",
            "0",
            "--id",
            "external-require",
        ],
    );
    assert!(require_output.status.success(), "{require_report:#}");
    let (rejected_output, rejected_report) = invoke(
        root.path(),
        &[
            "try",
            "--training-mode",
            "external-loop",
            "--loop",
            "training:supervised_loop",
            "--model",
            "model:build_model",
            "--steps",
            "1",
            "--apply-supervision",
        ],
    );
    assert!(!rejected_output.status.success(), "{rejected_report:#}");
    assert_eq!(
        rejected_report["diagnostics"][0]["code"], "SUPERVISION_REQUIRE_FAILED",
        "{rejected_report:#}"
    );
    assert_eq!(
        rejected_report["data"]["sample_localization"]["violating_samples"][0]["sample_id"],
        "external:0:1",
        "{rejected_report:#}"
    );
}
