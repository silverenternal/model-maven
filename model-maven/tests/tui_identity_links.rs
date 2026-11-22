use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn probe(root: &Path, model: &str) -> Value {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--no-interactive",
            "--python",
            &python,
            "model",
            "probe",
            "--model",
            model,
            "--batch",
            "fixture:batch",
        ])
        .current_dir(root)
        .output()
        .unwrap();
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("{error}: {}", String::from_utf8_lossy(&output.stderr)))
}

#[test]
fn actual_forward_exposes_only_observed_same_object_links() {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(python)
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        eprintln!("PyTorch unavailable; skipping real CLI integration");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("fixture.py"),
        r#"import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.src = nn.Linear(4, 4)
        self.identity = nn.Identity()
        self.branches = nn.ModuleList([nn.Linear(4, 4) for _ in range(32)])
        self.head = nn.Linear(32, 1)
    def forward(self, x):
        value = self.identity(self.src(x))
        for branch in self.branches: branch(value)
        if self.mode == 'mutate': value.add_(1)
        if self.mode == 'clone': value = value.clone()
        return self.head(value)
def stable(): return Model('stable')
def mutating(): return Model('mutate')
def cloned(): return Model('clone')
def batch(): return torch.ones(2, 4)
"#,
    )
    .unwrap();
    for (factory, expected) in [
        ("fixture:stable", 2),
        ("fixture:mutating", 0),
        ("fixture:cloned", 0),
    ] {
        let report = probe(project.path(), factory);
        assert_eq!(
            report["diagnostics"][0]["code"], "FORWARD_FAILED",
            "{report:#}"
        );
        assert_eq!(
            report["data"]["tensor_identity_trace"]["kind"],
            "same_tensor_object_and_version"
        );
        assert_eq!(report["data"]["tensor_identity_trace"]["scope"], "partial");
        let calls = report["data"]["calls"].as_array().unwrap();
        let call = |path: &str| {
            calls
                .iter()
                .find(|call| {
                    call["alias_paths"]
                        .as_array()
                        .is_some_and(|paths| paths.iter().any(|p| p == path))
                })
                .unwrap()
        };
        let src = call("model.src")["invocation_id"].as_str().unwrap();
        let identity = call("model.identity")["invocation_id"].as_str().unwrap();
        for index in 0..32 {
            let branch = call(&format!("model.branches.{index}"));
            let relation = &branch["input_tensor_matches"][0];
            assert_eq!(relation["input_path"], "inputs.args[0]");
            let prior = relation["prior_outputs"].as_array().unwrap();
            assert_eq!(prior.len(), 2, "{branch:#}");
            assert_eq!(prior[0]["invocation_id"], src);
            assert_eq!(prior[1]["invocation_id"], identity);
            assert!(prior.iter().all(|item| item["output_path"] == "outputs"));
        }
        let head = call("model.head");
        assert_eq!(
            head["input_tensor_matches"][0]["prior_outputs"]
                .as_array()
                .unwrap()
                .len(),
            expected,
            "{head:#}"
        );
        assert_eq!(
            report["data"]["failure_location"]["last_observed_module_call"]["invocation_id"],
            head["invocation_id"]
        );
    }
}
