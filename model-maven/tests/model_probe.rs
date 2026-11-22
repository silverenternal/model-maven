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

fn probe(project: &Path, batch: &str) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "model",
            "probe",
            "--model",
            "fixture:build_model",
            "--batch",
            batch,
        ])
        .current_dir(project)
        .output()
        .expect("start model probe");
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
fn shared_calls_have_forward_scoped_identity_and_structural_signatures() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("fixture.py"),
        r#"import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
    def forward(self, left, right):
        return self.shared(left), self.shared(right)

def build_model():
    return Model()

def batch_a():
    return {"args": [torch.ones(2, 4), torch.ones(3, 4)], "kwargs": {}}

def batch_b():
    return {"args": [torch.ones(3, 4), torch.ones(4, 4)], "kwargs": {}}

def batch_failing():
    return {"args": [torch.ones(2, 4), torch.ones(3, 5)], "kwargs": {}}
"#,
    )
    .unwrap();

    let (first_output, first_report) = probe(project.path(), "fixture:batch_a");
    assert!(first_output.status.success(), "{first_report:#}");
    let first = &first_report["data"];
    let first_calls = first["calls"].as_array().unwrap();
    let shared: Vec<_> = first_calls
        .iter()
        .filter(|call| {
            call["paths"]
                .as_array()
                .is_some_and(|paths| paths.contains(&Value::String("model.shared".into())))
        })
        .collect();
    assert_eq!(shared.len(), 2);
    assert_eq!(shared[0]["invocation_index"], 1);
    assert_eq!(shared[1]["invocation_index"], 2);
    assert_eq!(
        shared[0]["inputs"]["items"]["args"]["items"][0]["shape"],
        serde_json::json!([2, 4])
    );
    assert_eq!(
        shared[1]["inputs"]["items"]["args"]["items"][0]["shape"],
        serde_json::json!([3, 4])
    );
    assert_ne!(shared[0]["input_signature"], shared[1]["input_signature"]);
    assert_eq!(shared[0]["probe_id"], first["probe_id"]);
    assert_eq!(shared[0]["forward_id"], first["forward_id"]);
    assert_eq!(shared[0]["batch_id"], first["batch_id"]);
    assert_eq!(shared[0]["alias_paths"], shared[0]["paths"]);
    assert_eq!(
        shared[0]["parent_invocation_id"],
        first_calls[0]["invocation_id"]
    );
    assert_ne!(shared[0]["invocation_id"], shared[1]["invocation_id"]);
    assert!(
        shared[0]["input_output_paths"]
            .as_array()
            .unwrap()
            .iter()
            .any(|path| path == "inputs.args[0]")
    );
    assert!(
        shared[0]["input_output_paths"]
            .as_array()
            .unwrap()
            .iter()
            .any(|path| path == "outputs")
    );
    assert_eq!(first["batch_identity_stable"], false);

    let (second_output, second_report) = probe(project.path(), "fixture:batch_b");
    assert!(second_output.status.success(), "{second_report:#}");
    let second = &second_report["data"];
    let second_calls = second["calls"].as_array().unwrap();
    let second_shared = second_calls
        .iter()
        .find(|call| call["paths"][0] == "model.shared")
        .unwrap();
    assert_ne!(first["probe_id"], second["probe_id"]);
    assert_ne!(first["forward_id"], second["forward_id"]);
    assert_ne!(first["batch_id"], second["batch_id"]);
    assert_ne!(shared[0]["invocation_id"], second_shared["invocation_id"]);
    assert_ne!(first["input_signature"], second["input_signature"]);

    let (repeat_output, repeat_report) = probe(project.path(), "fixture:batch_a");
    assert!(repeat_output.status.success(), "{repeat_report:#}");
    let repeat = &repeat_report["data"];
    assert_eq!(first["input_signature"], repeat["input_signature"]);
    assert_ne!(first["forward_id"], repeat["forward_id"]);

    let (failure_output, failure_report) = probe(project.path(), "fixture:batch_failing");
    assert!(!failure_output.status.success());
    assert_eq!(failure_report["diagnostics"][0]["code"], "FORWARD_FAILED");
    let failure_data = &failure_report["data"];
    let last_call = &failure_data["failure_location"]["last_observed_module_call"];
    assert_eq!(last_call["paths"][0], "model.shared");
    assert_eq!(last_call["probe_id"], failure_data["probe_id"]);
    assert_eq!(last_call["forward_id"], failure_data["forward_id"]);
    assert_eq!(last_call["batch_id"], failure_data["batch_id"]);
    assert!(last_call["invocation_id"].as_str().is_some());
    assert!(last_call["input_signature"].as_str().is_some());
}
