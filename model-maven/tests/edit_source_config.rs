use serde_json::{Value, json};
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
fn source_configuration_guard_detects_changes_before_and_after_writing() {
    if !Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path();
    let source = r#"from pathlib import Path
import json
import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 2)
    def forward(self, x): return self.head(x)
def model(): return Net()
def batch():
    race = Path('race.json')
    if race.exists():
        state = json.loads(race.read_text())
        state['count'] += 1
        race.write_text(json.dumps(state))
        if state['count'] == state['change_at']:
            path = Path('.model-maven/project.json')
            config = json.loads(path.read_text())
            config['concurrent_marker'] = 'preserve-me'
            path.write_text(json.dumps(config))
    return torch.ones(2, 4)
"#;
    let source_path = project.join("fixture.py");
    fs::write(&source_path, source).unwrap();
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
    let config_path = project.join(".model-maven/project.json");
    let original = fs::read(&config_path).unwrap();
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
    let revision = preview["data"]["config_revision"].as_str().unwrap();
    let plan = preview["data"]["source_plan"]["digest"].as_str().unwrap();
    let input = preview["data"]["validation"]["batch_content"]["digest"]
        .as_str()
        .unwrap();
    let mut apply = edit.to_vec();
    apply.extend([
        "--if-source-config",
        revision,
        "--if-source-plan",
        plan,
        "--if-input-content",
        input,
    ]);
    let mut changed: Value = serde_json::from_slice(&original).unwrap();
    changed["concurrent_marker"] = json!("preserve-me");
    fs::write(&config_path, serde_json::to_vec(&changed).unwrap()).unwrap();
    let (out, rejected) = run(project, &apply);
    assert!(!out.status.success());
    assert_eq!(
        rejected["diagnostics"][0]["code"],
        "PROJECT_CONFIG_CONFLICT"
    );
    assert_eq!(fs::read_to_string(&source_path).unwrap(), source);
    fs::write(&config_path, &original).unwrap();
    fs::write(project.join("race.json"), r#"{"count":0,"change_at":1}"#).unwrap();
    let (out, rejected) = run(project, &apply);
    assert!(!out.status.success());
    assert_eq!(
        rejected["diagnostics"][0]["code"],
        "PROJECT_CONFIG_CONFLICT"
    );
    assert_eq!(fs::read_to_string(&source_path).unwrap(), source);
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(&config_path).unwrap()).unwrap()["concurrent_marker"],
        "preserve-me"
    );
    fs::write(&config_path, &original).unwrap();
    fs::write(project.join("race.json"), r#"{"count":0,"change_at":2}"#).unwrap();
    let (out, rejected) = run(project, &apply);
    assert!(!out.status.success(), "{rejected:#}");
    assert_eq!(
        rejected["diagnostics"][0]["code"],
        "SOURCE_EDIT_VALIDATION_FAILED"
    );
    assert!(
        rejected["data"]["configuration_error"]
            .as_str()
            .unwrap()
            .contains("配置")
    );
    assert_eq!(fs::read_to_string(&source_path).unwrap(), source);
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(&config_path).unwrap()).unwrap()["concurrent_marker"],
        "preserve-me"
    );
}
