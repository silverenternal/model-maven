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
        .expect("start custom supervision command");
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
        return {"features": values * self.scale}
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
    def forward(self, values):
        return self.encoder(values)
def build_model(): return Model()
"#,
    )
    .unwrap();
    fs::write(
        root.path().join("data.py"),
        "import torch\ndef batch(): return {\"args\":[torch.tensor([[2.0],[-3.0]])],\"kwargs\":{},\"metadata\":{\"stable_sample_ids\":True,\"sample_ids\":[\"sample:1\",\"sample:2\"]}}\nbatch.MODEL_MAVEN_STATELESS = True\n",
    )
    .unwrap();
    fs::write(
        root.path().join("training.py"),
        r#"def step(model, batch, context):
    output = model(batch["args"][0])
    return {"loss": output["features"].sum() * 0}
step.MODEL_MAVEN_STATELESS = True
"#,
    )
    .unwrap();
    fs::write(
        root.path().join("constraints.py"),
        r#"def bounded(value, batch, params):
    maximum = float(value.detach().abs().max().item())
    violations = value.detach().abs() > params["limit"]
    return {"passed": not bool(violations.any()), "summary": {"max_abs": maximum}, "violations": violations}
def energy(value, batch, params):
    return (value - params["target"]).square().mean()
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
            "representation",
            "--meaning",
            "Representation constrained by project validator",
            "--id",
            "representation",
        ],
    );
    assert!(output.status.success(), "{report:#}");
}

#[test]
fn custom_validator_and_differentiable_penalty_preview_and_train() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let root = project();
    mark(root.path());
    let (validator_output, validator_report) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "representation",
            "--kind",
            "custom-validator",
            "--mode",
            "observe",
            "--plugin",
            "constraints:bounded",
            "--plugin-param",
            "limit=4.0",
            "--axis",
            "0",
            "--id",
            "bounded-validator",
        ],
    );
    assert!(validator_output.status.success(), "{validator_report:#}");
    let (penalty_output, penalty_report) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "representation",
            "--kind",
            "custom-penalty",
            "--mode",
            "penalty",
            "--plugin",
            "constraints:energy",
            "--plugin-param",
            "target=0.0",
            "--weight",
            "0.25",
            "--id",
            "energy-penalty",
        ],
    );
    assert!(penalty_output.status.success(), "{penalty_report:#}");

    let (preview_output, preview_report) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "preview",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
        ],
    );
    assert!(preview_output.status.success(), "{preview_report:#}");
    let results = preview_report["data"]["results"].as_array().unwrap();
    assert!(
        results
            .iter()
            .any(|item| item["rule_id"] == "bounded-validator" && item["status"] == "observed"),
        "{preview_report:#}"
    );
    let penalty = results
        .iter()
        .find(|item| item["rule_id"] == "energy-penalty")
        .unwrap();
    assert_eq!(penalty["status"], "penalty_preview_only");
    assert_eq!(penalty["objective_applied"], false);
    assert_eq!(
        results[0]["observations"][0]["sample_localization"]["status"],
        "localized"
    );
    assert!(
        penalty["observations"][0]["weighted_penalty_preview"]
            .as_f64()
            .unwrap()
            > 0.0
    );

    let (train_output, train_report) = invoke(
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
    assert!(train_output.status.success(), "{train_report:#}");
    assert!(
        train_report["data"]["mean_losses"]["supervision/energy-penalty"]
            .as_f64()
            .unwrap()
            > 0.0,
        "{train_report:#}"
    );
}

#[test]
fn custom_require_rejects_before_update_and_plugin_source_drift_fails_closed() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let root = project();
    mark(root.path());
    let (add_output, add_report) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "representation",
            "--kind",
            "custom-validator",
            "--mode",
            "require",
            "--plugin",
            "constraints:bounded",
            "--plugin-param",
            "limit=1.0",
            "--axis",
            "0",
            "--id",
            "bounded-require",
        ],
    );
    assert!(add_output.status.success(), "{add_report:#}");
    let (try_output, try_report) = invoke(
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
    assert!(!try_output.status.success(), "{try_report:#}");
    assert_eq!(
        try_report["diagnostics"][0]["code"], "STEP_FAILED",
        "{try_report:#}"
    );
    assert_eq!(
        try_report["data"]["cause_code"], "SUPERVISION_REQUIRE_FAILED",
        "{try_report:#}"
    );
    assert_eq!(try_report["data"]["completed_steps"], 0);
    assert_eq!(
        try_report["data"]["cause_data"]["sample_localization"]["violating_samples"][0]["sample_id"],
        "sample:1"
    );

    let mut source = fs::read_to_string(root.path().join("constraints.py")).unwrap();
    source.push_str("\n# semantic implementation changed\n");
    fs::write(root.path().join("constraints.py"), source).unwrap();
    let (preview_output, preview_report) = invoke(
        root.path(),
        &[
            "model",
            "supervise",
            "preview",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
        ],
    );
    assert!(preview_output.status.success(), "{preview_report:#}");
    assert_eq!(preview_report["data"]["results"][0]["status"], "stale");
}
