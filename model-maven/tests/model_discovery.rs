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

fn invoke(root: &Path, model: &str) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--verbose",
            "--python",
            &python(),
            "model",
            "inspect",
            "--model",
            model,
        ])
        .current_dir(root)
        .output()
        .unwrap();
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
fn root_model_inspection_is_training_free_and_provenance_is_scoped_to_live_modules() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }

    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("discovery.py"),
        r#"from collections import OrderedDict
from torch import nn

class CustomSequence(nn.Sequential):
    def forward(self, values, scale=1.0):
        return super().forward(values) * scale

class Root(nn.Module):
    def __init__(self):
        super().__init__()
        candidates = [nn.Linear(4, 4) for _ in range(2)]
        shared = candidates[-1]
        self.encoder = CustomSequence(OrderedDict([
            ("projection", shared),
            ("activation", nn.ReLU()),
        ]))
        self.left = shared
        self.right = shared

    def forward(self, values, extra=None):
        encoded = self.encoder(values)
        return encoded if extra is None else encoded + self.right(extra)

def build_model():
    return Root()

def build_invalid():
    return object()

def build_failing():
    raise RuntimeError("factory exploded")
"#,
    )
    .unwrap();

    let (output, report) = invoke(project.path(), "discovery:build_model");
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["status"], "success", "{report:#}");
    assert!(!project.path().join(".model-maven").exists());

    let modules = report["data"]["modules"].as_array().unwrap();
    let shared = modules
        .iter()
        .find(|module| module["path"] == "model.encoder.projection")
        .unwrap();
    assert_eq!(
        shared["aliases"],
        serde_json::json!(["model.encoder.projection", "model.left", "model.right"])
    );
    assert!(shared["instance_id"].as_str().is_some());
    assert_eq!(
        shared["construction_site"]["file"],
        project
            .path()
            .join("discovery.py")
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .as_ref()
    );
    assert!(shared["construction_site"]["line"].as_u64().unwrap() > 0);
    assert_eq!(shared["can_source_set"], true);

    let root = modules
        .iter()
        .find(|module| module["type"] == "Root")
        .unwrap();
    assert_eq!(
        root["source_definition"]["file"],
        project
            .path()
            .join("discovery.py")
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .as_ref()
    );

    let (invalid_output, invalid_report) = invoke(project.path(), "discovery:build_invalid");
    assert!(!invalid_output.status.success());
    assert_eq!(
        invalid_report["diagnostics"][0]["code"],
        "MODEL_TYPE_INVALID"
    );

    let (failed_output, failed_report) = invoke(project.path(), "discovery:build_failing");
    assert!(!failed_output.status.success());
    assert_eq!(
        failed_report["diagnostics"][0]["code"],
        "MODEL_CONSTRUCTION_FAILED"
    );
}
