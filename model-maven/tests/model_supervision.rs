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

fn fixture() -> tempfile::TempDir {
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("fixture.py"),
        r#"import torch
from torch import nn

class Encoder(nn.Module):
    def forward(self, values):
        return {"features": values * 2}

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
    def forward(self, values, targets=None):
        return self.encoder(values)

def build_model(): return Model()
def batch(): return {"args": [torch.tensor([[2.0], [-3.0]])], "kwargs": {}}
def mismatched_target_batch():
    return {"args": [torch.tensor([[2.0], [-3.0]])], "kwargs": {"targets": torch.tensor([0, 1, 2])}}
def matched_target_batch():
    return {"args": [torch.tensor([[2.0], [-3.0]])], "kwargs": {"targets": torch.tensor([0, 1])}}
"#,
    )
    .unwrap();
    project
}

fn invoke(project: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", &python()])
        .args(args)
        .current_dir(project)
        .output()
        .expect("start supervision command");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report, got {:?}: {error}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

#[test]
fn rule_preview_uses_real_marked_ports_and_never_applies_the_objective() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    let (marked_output, marked) = invoke(
        project.path(),
        &[
            "model",
            "mark",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:batch",
            "--path",
            "model.encoder",
            "--port",
            "outputs.features",
            "--role",
            "latent_state",
            "--meaning",
            "Representation consumed by the next model block.",
            "--id",
            "latent",
        ],
    );
    assert!(marked_output.status.success(), "{marked:#}");

    for (id, kind, mode, extra) in [
        (
            "bounded",
            "range",
            "require",
            vec!["--min", "-1", "--max", "1"],
        ),
        ("finite", "finite", "observe", vec![]),
        (
            "penalty",
            "range",
            "penalty",
            vec!["--min", "-1", "--max", "1", "--weight", "0.1"],
        ),
    ] {
        let mut args = vec![
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "latent",
            "--kind",
            kind,
            "--mode",
            mode,
            "--id",
            id,
        ];
        args.extend(extra);
        let (output, report) = invoke(project.path(), &args);
        assert!(output.status.success(), "{report:#}");
    }

    let (preview_output, preview) = invoke(
        project.path(),
        &[
            "model",
            "supervise",
            "preview",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:batch",
        ],
    );
    assert!(preview_output.status.success(), "{preview:#}");
    assert_eq!(preview["data"]["status"], "observed-runtime");
    assert_eq!(preview["data"]["coverage_status"], "partial");
    assert_eq!(preview["data"]["objective_applied"], false);
    assert_eq!(
        preview["data"]["sample_localization"]["status"],
        "insufficient"
    );
    assert_eq!(
        preview["data"]["sample_localization"]["stable_ids_available"],
        false
    );
    assert!(
        preview["data"]["coverage"]
            .as_str()
            .unwrap()
            .contains("没有足够的稳定样本 ID")
    );
    let results = preview["data"]["results"].as_array().unwrap();
    let by_id: std::collections::BTreeMap<_, _> = results
        .iter()
        .map(|result| (result["rule_id"].as_str().unwrap(), result))
        .collect();
    assert_eq!(by_id["bounded"]["status"], "would_reject", "{preview:#}");
    assert_eq!(by_id["bounded"]["violation_count"], 2);
    assert_eq!(by_id["finite"]["status"], "observed");
    assert_eq!(by_id["penalty"]["status"], "penalty_preview_only");
    assert_eq!(by_id["penalty"]["objective_applied"], false);
    assert!(
        (by_id["penalty"]["observations"][0]["weighted_penalty_preview"]
            .as_f64()
            .unwrap()
            - 1.7)
            .abs()
            < 1e-6
    );
    assert!(preview.to_string().contains("weighted_penalty_preview"));
    assert!(!preview.to_string().contains("\"2.0\""));
}

#[test]
fn rule_preview_refuses_stale_mark_bindings_without_running_forward() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    let (mark_output, mark) = invoke(
        project.path(),
        &[
            "model",
            "mark",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:batch",
            "--path",
            "model.encoder",
            "--port",
            "outputs.features",
            "--role",
            "latent",
            "--meaning",
            "Latent feature.",
            "--id",
            "latent",
        ],
    );
    assert!(mark_output.status.success(), "{mark:#}");
    let (rule_output, rule) = invoke(
        project.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "latent",
            "--kind",
            "finite",
            "--mode",
            "require",
            "--id",
            "finite",
        ],
    );
    assert!(rule_output.status.success(), "{rule:#}");
    let mut source = fs::read_to_string(project.path().join("fixture.py")).unwrap();
    source.push_str("# implementation changed\n");
    fs::write(project.path().join("fixture.py"), source).unwrap();
    let (preview_output, preview) = invoke(
        project.path(),
        &[
            "model",
            "supervise",
            "preview",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:batch",
        ],
    );
    assert!(preview_output.status.success(), "{preview:#}");
    assert_eq!(preview["data"]["status"], "stale");
    assert_eq!(preview["data"]["results"][0]["status"], "stale");
    assert!(
        preview["data"]["results"][0]["observations"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn manually_bound_batch_target_shape_is_checked_against_a_real_model_port() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    let (mark_output, mark) = invoke(
        project.path(),
        &[
            "model",
            "mark",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:mismatched_target_batch",
            "--path",
            "model.encoder",
            "--port",
            "outputs.features",
            "--role",
            "logits",
            "--meaning",
            "Model scores aligned with manually selected batch targets.",
            "--id",
            "logits",
        ],
    );
    assert!(mark_output.status.success(), "{mark:#}");
    let (rule_output, rule) = invoke(
        project.path(),
        &[
            "model",
            "supervise",
            "rule",
            "add",
            "--mark",
            "logits",
            "--kind",
            "batch-alignment",
            "--mode",
            "require",
            "--target-path",
            "batch.kwargs.targets",
            "--axis",
            "0",
            "--id",
            "target-axis",
        ],
    );
    assert!(rule_output.status.success(), "{rule:#}");

    let (bad_output, bad) = invoke(
        project.path(),
        &[
            "model",
            "supervise",
            "preview",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:mismatched_target_batch",
        ],
    );
    assert!(bad_output.status.success(), "{bad:#}");
    let result = &bad["data"]["results"][0];
    assert_eq!(result["status"], "would_reject", "{bad:#}");
    assert_eq!(result["expected"]["target_path"], "batch.kwargs.targets");
    assert_eq!(
        result["observations"][0]["shape"],
        serde_json::json!([2, 1])
    );
    assert_eq!(
        result["observations"][0]["target_shape"],
        serde_json::json!([3])
    );
    assert_eq!(result["observations"][0]["axis"], 0);

    let (good_output, good) = invoke(
        project.path(),
        &[
            "model",
            "supervise",
            "preview",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:matched_target_batch",
        ],
    );
    assert!(good_output.status.success(), "{good:#}");
    assert_eq!(good["data"]["results"][0]["status"], "passed", "{good:#}");
}
