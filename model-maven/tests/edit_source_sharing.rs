use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}
fn invoke(project: &Path, args: &[&str]) -> (Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--no-interactive", "--python", &python()])
        .args(args)
        .current_dir(project)
        .output()
        .unwrap();
    let report = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&output.stderr)));
    (output, report)
}

#[test]
fn source_edit_checks_shared_identity_in_fresh_worker_and_rolls_back_divergence() {
    if !Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let project = root.path();
    let file = project.join("fixture.py");
    let source = r#"import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.right.weight = self.left.weight
        self.layers = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.layers[1].weight = nn.Parameter(self.layers[0].weight.detach())
    def forward(self, x): return self.layers(self.right(self.left(x)))
def model(): return Net()
def batch(): return torch.ones(2, 4)
"#;
    fs::write(&file, source).unwrap();
    let (init, report) = invoke(
        project,
        &[
            "init",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:batch",
        ],
    );
    assert!(init.status.success(), "{report:#}");
    let (variant_preview, variant_report) = invoke(
        project,
        &[
            "layer",
            "set",
            "model.left",
            "--param",
            "bias=false",
            "--batch",
            "fixture:batch",
            "--dry-run",
        ],
    );
    assert!(variant_preview.status.success(), "{variant_report:#}");
    assert_eq!(
        variant_report["data"]["edit_effects"][0]["old_state_not_copied"],
        json!(["bias"])
    );
    let set = [
        "layer",
        "set",
        "model.left",
        "--target",
        "source",
        "--param",
        "bias=false",
        "--require-forward",
    ];
    let mut dry = set.to_vec();
    dry.push("--dry-run");
    let (preview, report) = invoke(project, &dry);
    assert!(preview.status.success(), "{report:#}");
    assert_eq!(
        report["data"]["edit_effects"][0]["preserved"],
        json!(["weight"])
    );
    assert_eq!(
        report["data"]["edit_effects"][0]["old_state_not_copied"],
        json!(["bias"])
    );
    assert_eq!(
        report["data"]["shared_state_evidence"]["scope"],
        "in_memory_candidate_object_storage_v2"
    );
    assert_eq!(
        report["data"]["edit_effects"][0]["candidate_external_state_sharing"][0]["external_paths"]
            [0],
        "model.right.weight"
    );
    let digest = report["data"]["source_plan"]["digest"].as_str().unwrap();
    let mut apply = set.to_vec();
    apply.extend(["--if-source-plan", digest]);
    let (saved, report) = invoke(project, &apply);
    assert!(saved.status.success(), "{report:#}");
    assert_eq!(
        report["data"]["source_validation"]["shared_state"]["status"],
        "passed"
    );
    assert_eq!(
        report["data"]["source_validation"]["shared_state"]["group_count"],
        3
    );
    let after_set = fs::read(&file).unwrap();
    assert_ne!(after_set, source.as_bytes());
    let replace = [
        "layer",
        "replace",
        "model.layers.0",
        "--target",
        "source",
        "--type",
        "Linear",
        "--param",
        "in_features=4",
        "--param",
        "out_features=4",
        "--require-forward",
    ];
    let mut dry = replace.to_vec();
    dry.push("--dry-run");
    let (preview, report) = invoke(project, &dry);
    assert!(preview.status.success(), "{report:#}");
    assert_eq!(
        report["data"]["edit_effects"][0]["old_state_not_copied"],
        json!(["weight", "bias"])
    );
    assert_eq!(
        report["data"]["edit_effects"][0]["removed_candidate_external_state_sharing"][0]["external_paths"]
            [0],
        "model.layers.1.weight"
    );
    let digest = report["data"]["source_plan"]["digest"].as_str().unwrap();
    let mut apply = replace.to_vec();
    apply.extend(["--if-source-plan", digest]);
    let (failed, report) = invoke(project, &apply);
    assert!(!failed.status.success());
    assert_eq!(
        report["diagnostics"][0]["code"],
        "SOURCE_EDIT_VALIDATION_FAILED"
    );
    assert_eq!(
        report["data"]["verification"]["diagnostics"][0]["code"],
        "SOURCE_SHARED_STATE_MISMATCH"
    );
    assert_eq!(fs::read(&file).unwrap(), after_set);
}
