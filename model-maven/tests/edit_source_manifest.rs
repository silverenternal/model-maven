use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}
fn run(project: &Path, args: &[&str]) -> (Output, Value) {
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
fn source_edit_guards_observed_dependency_files_and_rolls_back_revalidation_race() {
    if !Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path();
    let helper = project.join("helper.py");
    let original_helper = "FACTOR = 1\n";
    fs::write(&helper, original_helper).unwrap();
    let source = r#"from pathlib import Path
import json
import torch
from torch import nn
import helper
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)
    def forward(self, x): return self.head(x * helper.FACTOR)
def model(): return Net()
def batch():
    race = Path('race.json')
    if race.exists():
        state = json.loads(race.read_text())
        state['count'] += 1
        race.write_text(json.dumps(state))
        if state['count'] == state['change_at']:
            Path('helper.py').write_text('FACTOR = 2\n')
    return torch.ones(2, 4)
"#;
    let file = project.join("fixture.py");
    fs::write(&file, source).unwrap();
    let (out, report) = run(
        project,
        &[
            "init",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:batch",
        ],
    );
    assert!(out.status.success(), "{report:#}");
    let edit = [
        "layer",
        "set",
        "model.head",
        "--target",
        "source",
        "--param",
        "bias=false",
        "--require-forward",
    ];
    let mut dry = edit.to_vec();
    dry.push("--dry-run");
    let (out, preview) = run(project, &dry);
    assert!(out.status.success(), "{preview:#}");
    let manifest = &preview["data"]["source_manifest"];
    assert!(
        manifest[helper.to_str().unwrap()].is_string(),
        "{preview:#}"
    );
    assert_eq!(
        preview["data"]["source_manifest_evidence"]["scope"],
        "worker-recorded-project-sources"
    );
    let source_digest = preview["data"]["source_manifest_evidence"]["digest"]
        .as_str()
        .unwrap();
    let config = preview["data"]["config_revision"].as_str().unwrap();
    let plan = preview["data"]["source_plan"]["digest"].as_str().unwrap();
    let input = preview["data"]["validation"]["batch_content"]["digest"]
        .as_str()
        .unwrap();
    let mut apply = edit.to_vec();
    apply.extend([
        "--if-source-config",
        config,
        "--if-source-plan",
        plan,
        "--if-input-content",
        input,
        "--if-source-manifest",
        source_digest,
    ]);
    fs::write(&helper, "FACTOR = 2\n").unwrap();
    let (out, rejected) = run(project, &apply);
    assert!(!out.status.success());
    assert_eq!(
        rejected["diagnostics"][0]["code"],
        "SOURCE_MANIFEST_CONFLICT"
    );
    assert_eq!(fs::read_to_string(&file).unwrap(), source);
    fs::write(&helper, original_helper).unwrap();
    fs::write(project.join("race.json"), r#"{"count":0,"change_at":1}"#).unwrap();
    let (out, rejected) = run(project, &apply);
    assert!(!out.status.success(), "{rejected:#}");
    assert_eq!(
        rejected["diagnostics"][0]["code"],
        "SOURCE_MANIFEST_CONFLICT"
    );
    assert_eq!(fs::read_to_string(&file).unwrap(), source);
    fs::write(&helper, original_helper).unwrap();
    fs::write(project.join("race.json"), r#"{"count":0,"change_at":2}"#).unwrap();
    let (out, rejected) = run(project, &apply);
    assert!(!out.status.success(), "{rejected:#}");
    assert_eq!(
        rejected["diagnostics"][0]["code"],
        "SOURCE_EDIT_VALIDATION_FAILED"
    );
    assert!(
        rejected["data"]["source_manifest_error"]
            .as_str()
            .unwrap()
            .contains("helper.py")
    );
    assert_eq!(fs::read_to_string(&file).unwrap(), source);
    assert_ne!(fs::read_to_string(&helper).unwrap(), original_helper);
    fs::remove_file(project.join("race.json")).unwrap();
    fs::write(&helper, original_helper).unwrap();
    let (out, committed) = run(project, &apply);
    assert!(out.status.success(), "{committed:#}");
    assert_eq!(
        committed["data"]["source_validation"]["source_manifest_status"],
        "passed"
    );
    assert_eq!(
        committed["data"]["source_validation"]["source_manifest_digest"],
        source_digest
    );
    assert!(fs::read_to_string(&file).unwrap().contains("bias=False"));
}
