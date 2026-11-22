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

class Projection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(4, width)
    def forward(self, values):
        return {"features": self.linear(values)}

class Model(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = Projection(width)
    def forward(self, values):
        return self.encoder(values)

def build_model(width=3): return Model(width)
def batch(): return {"args": [torch.ones(2, 4)], "kwargs": {}}
"#,
    )
    .unwrap();
    project
}

fn mark(project: &Path) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
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
            "embedding",
            "--meaning",
            "Learned representation for downstream retrieval.",
            "--id",
            "retrieval-embedding",
        ])
        .current_dir(project)
        .output()
        .expect("start semantic mark command");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report, got {:?}: {error}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

fn store_command(project: &Path, command: &str, id: Option<&str>) -> (std::process::Output, Value) {
    let mut args = vec!["--json", "--python", "python3", "model"];
    args.push(command);
    if let Some(id) = id {
        args.extend(["--id", id]);
    }
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(args)
        .current_dir(project)
        .output()
        .expect("start semantic mark store command");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report, got {:?}: {error}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

fn retag(project: &Path) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "model",
            "retag",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:batch",
            "--id",
            "retrieval-embedding",
            "--path",
            "model.encoder",
            "--port",
            "outputs.features",
            "--role",
            "latent_state",
            "--meaning",
            "Updated meaning after reviewing the changed implementation.",
        ])
        .current_dir(project)
        .output()
        .expect("start semantic rebind command");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report, got {:?}: {error}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

fn audit_with_width(project: &Path, width: &str) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "model",
            "marks",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:batch",
            "--model-param",
            &format!("width={width}"),
        ])
        .current_dir(project)
        .output()
        .expect("start semantic binding audit");
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
fn cli_mark_binds_only_a_real_probe_port_and_persists_project_source_identity() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    let (output, report) = mark(project.path());
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["data"]["status"], "current");
    assert_eq!(report["data"]["mark"]["module_path"], "model.encoder");
    assert_eq!(report["data"]["mark"]["port"], "outputs.features");
    assert_eq!(report["data"]["mark"]["role"], "embedding");
    assert!(report["data"]["observation"]["probe_id"].is_string());

    let sidecar = project.path().join(".model-maven/semantic-marks.json");
    let saved: Value = serde_json::from_slice(&fs::read(sidecar).unwrap()).unwrap();
    let mark = &saved["marks"][0];
    assert_eq!(mark["anchor"]["source_file"], "fixture.py");
    assert!(mark["anchor"]["source_sha256"].as_str().unwrap().len() == 64);
    assert_eq!(mark["observation"]["batch_identity_stable"], false);

    let (list_output, listed) = store_command(project.path(), "marks", None);
    assert!(list_output.status.success(), "{listed:#}");
    assert_eq!(listed["data"]["marks"][0]["binding_status"], "current");
    let (audit_output, audited) = audit_with_width(project.path(), "5");
    assert!(audit_output.status.success(), "{audited:#}");
    assert_eq!(audited["data"]["marks"][0]["binding_status"], "stale");
    assert!(
        audited["data"]["marks"][0]["stale_reason"]
            .as_str()
            .unwrap()
            .contains("参数")
    );
    let mut source = fs::read_to_string(project.path().join("fixture.py")).unwrap();
    source.push_str("# changed after semantic binding\n");
    fs::write(project.path().join("fixture.py"), source).unwrap();
    let (stale_output, stale) = store_command(project.path(), "marks", None);
    assert!(stale_output.status.success(), "{stale:#}");
    assert_eq!(stale["data"]["marks"][0]["binding_status"], "stale");
    let (retag_output, retagged) = retag(project.path());
    assert!(retag_output.status.success(), "{retagged:#}");
    assert_eq!(retagged["data"]["status"], "rebound");
    assert_eq!(retagged["data"]["mark"]["role"], "latent_state");
    let (fresh_output, fresh) = store_command(project.path(), "marks", None);
    assert!(fresh_output.status.success(), "{fresh:#}");
    assert_eq!(fresh["data"]["marks"][0]["binding_status"], "current");
    let (remove_output, removed) =
        store_command(project.path(), "unmark", Some("retrieval-embedding"));
    assert!(remove_output.status.success(), "{removed:#}");
    assert_eq!(removed["data"]["removed"], "retrieval-embedding");
    let (empty_output, empty) = store_command(project.path(), "marks", None);
    assert!(empty_output.status.success(), "{empty:#}");
    assert_eq!(empty["data"]["count"], 0);
}

#[test]
fn cli_mark_rejects_a_port_that_the_selected_implementation_did_not_emit() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "model",
            "mark",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:batch",
            "--path",
            "model.encoder",
            "--port",
            "outputs.not_real",
            "--role",
            "embedding",
            "--meaning",
            "This port does not exist.",
        ])
        .current_dir(project.path())
        .output()
        .expect("start semantic mark command");
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(!output.status.success(), "{report:#}");
    assert_eq!(
        report["diagnostics"][0]["code"],
        "SEMANTIC_PORT_NOT_OBSERVED"
    );
    assert!(
        !project
            .path()
            .join(".model-maven/semantic-marks.json")
            .exists()
    );
}
