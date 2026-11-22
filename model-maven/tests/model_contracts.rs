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

fn probe(
    project: &Path,
    model: &str,
    contract: &str,
    batch: &str,
) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "model",
            "probe",
            "--model",
            model,
            "--batch",
            batch,
            "--contract",
            contract,
        ])
        .current_dir(project)
        .output()
        .expect("start model contract probe");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report, got {:?}: {error}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

fn fixture() -> tempfile::TempDir {
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("fixture.py"),
        r#"import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3)
    def forward(self, values):
        return {"features": self.projection(values)}

class DynamicModel(nn.Module):
    def forward(self, values):
        if values.sum().item() > 0:
            return values + 1
        return values - 1

class StaticMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(5, 3)
    def forward(self, values):
        return self.projection(values)

class UnsupportedMetaModel(nn.Module):
    def forward(self, values):
        return torch.unique(values)

def build_model():
    return Model()

def build_dynamic():
    return DynamicModel()

def build_static_mismatch():
    return StaticMismatch()

def build_unsupported_meta():
    return UnsupportedMetaModel()

def batch():
    return {"args": [torch.ones(2, 4)], "kwargs": {}}

def batch_dynamic():
    return {"args": [torch.ones(2, 4)], "kwargs": {}}

def batch_unique():
    return {"args": [torch.tensor([1., 2., 2.])], "kwargs": {}}
"#,
    )
    .unwrap();
    project
}

#[test]
fn declared_shape_and_symbol_contracts_are_checked_without_training_fields() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    fs::write(
        project.path().join("contract.json"),
        r#"{
  "version": 1,
  "inputs": {"args[0]": {"rank": 2, "dtype": "torch.float32", "shape": ["B", 4]}},
  "outputs": {"features": {"rank": 2, "shape": ["B", 3]}},
  "symbols": {"B": {"min": 1, "max": 8}},
  "static": true
}"#,
    )
    .unwrap();
    let (output, report) = probe(
        project.path(),
        "fixture:build_model",
        "contract.json",
        "fixture:batch",
    );
    assert!(output.status.success(), "{report:#}");
    let data = &report["data"];
    assert_eq!(data["contract_report"]["status"], "observed-runtime");
    assert_eq!(data["contract_report"]["symbol_bindings"]["B"], 2);
    assert!(
        data["shape_inference"]["status"] == "inferred-static"
            || data["shape_inference"]["status"] == "unsupported"
    );
    assert!(
        data["coverage"]
            .as_str()
            .unwrap()
            .contains("不证明所有分支")
    );
}

#[test]
fn known_contract_mismatch_reports_path_expected_actual_and_batch_coverage() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    fs::write(
        project.path().join("contract.json"),
        r#"{"inputs":{"args[0]":{"shape":["B",5]}},"symbols":{"B":{"min":1}}}"#,
    )
    .unwrap();
    let (output, report) = probe(
        project.path(),
        "fixture:build_model",
        "contract.json",
        "fixture:batch",
    );
    assert!(!output.status.success(), "{report:#}");
    assert_eq!(report["diagnostics"][0]["code"], "MODEL_CONTRACT_FAILED");
    let checks = report["data"]["contract_report"]["checks"]
        .as_array()
        .unwrap();
    assert_eq!(checks[0]["status"], "failed");
    assert_eq!(checks[0]["path"], "inputs.args[0]");
    assert_eq!(checks[0]["mismatches"][0]["expected"], 5);
    assert_eq!(checks[0]["mismatches"][0]["actual"], 4);
    assert!(
        checks[0]["coverage"]
            .as_str()
            .unwrap()
            .contains("真实前向观测")
    );
}

#[test]
fn meta_fx_reports_static_layer_dimension_mismatch_before_real_forward() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    fs::write(
        project.path().join("contract.json"),
        r#"{"inputs":{"args[0]":{"rank":2,"shape":["B",4]}},"symbols":{"B":{"min":1}}}"#,
    )
    .unwrap();
    let (output, report) = probe(
        project.path(),
        "fixture:build_static_mismatch",
        "contract.json",
        "fixture:batch",
    );
    assert!(!output.status.success(), "{report:#}");
    assert_eq!(report["diagnostics"][0]["code"], "MODEL_CONTRACT_FAILED");
    let diagnostic = &report["data"]["shape_inference"]["diagnostics"][0];
    assert_eq!(diagnostic["path"], "projection");
    assert_eq!(diagnostic["expected"], 5);
    assert_eq!(diagnostic["actual"], 4);
    assert!(
        report["data"]["shape_inference"]["coverage"]
            .as_str()
            .unwrap()
            .contains("真实前向")
    );
}

#[test]
fn layer_port_contracts_are_checked_per_observed_invocation() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    fs::write(
        project.path().join("contract.json"),
        r#"{
  "inputs": {"args[0]": {"shape": ["B", 4]}},
  "symbols": {"B": {"min": 1}},
  "modules": {
    "model.projection": {
      "inputs": {"args[0]": {"shape": ["B", 4]}},
      "outputs": {"outputs": {"shape": ["B", 5]}}
    }
  }
}"#,
    )
    .unwrap();
    let (output, report) = probe(
        project.path(),
        "fixture:build_model",
        "contract.json",
        "fixture:batch",
    );
    assert!(!output.status.success(), "{report:#}");
    assert_eq!(report["diagnostics"][0]["code"], "MODEL_CONTRACT_FAILED");
    let checks = report["data"]["contract_report"]["checks"]
        .as_array()
        .unwrap();
    let layer_output = checks
        .iter()
        .find(|check| {
            check["path"].as_str().is_some_and(|path| {
                path.starts_with("model.projection#") && path.ends_with("outputs")
            })
        })
        .expect("per-invocation module output check");
    assert_eq!(layer_output["status"], "failed");
    assert_eq!(layer_output["mismatches"][0]["expected"], 5);
    assert_eq!(layer_output["mismatches"][0]["actual"], 3);
    assert!(layer_output["invocation_id"].as_str().is_some());
}

#[test]
fn dynamic_control_flow_and_meta_unsupported_ops_remain_observable() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    fs::write(
        project.path().join("contract.json"),
        r#"{"inputs":{"args[0]":{"rank":2,"shape":["B",4]}},"symbols":{"B":{"min":1}}}"#,
    )
    .unwrap();
    let (dynamic_output, dynamic_report) = probe(
        project.path(),
        "fixture:build_dynamic",
        "contract.json",
        "fixture:batch_dynamic",
    );
    assert!(dynamic_output.status.success(), "{dynamic_report:#}");
    assert_eq!(
        dynamic_report["data"]["contract_report"]["status"],
        "observed-runtime"
    );
    assert_eq!(
        dynamic_report["data"]["shape_inference"]["status"],
        "unsupported"
    );

    fs::write(
        project.path().join("contract.json"),
        r#"{"inputs":{"args[0]":{"rank":1,"shape":["N"]}},"symbols":{"N":{"min":1}}}"#,
    )
    .unwrap();
    let (meta_output, meta_report) = probe(
        project.path(),
        "fixture:build_unsupported_meta",
        "contract.json",
        "fixture:batch_unique",
    );
    assert!(meta_output.status.success(), "{meta_report:#}");
    assert_eq!(
        meta_report["data"]["shape_inference"]["status"],
        "unsupported"
    );
    assert_eq!(
        meta_report["data"]["contract_report"]["status"],
        "observed-runtime"
    );
}
