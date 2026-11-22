use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn invoke(root: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--verbose", "--python", &python()])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid CLI JSON: {error}; stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

#[test]
fn registered_layer_state_roundtrip_runs_extension_hooks_through_component_test() {
    let torch = Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .expect("start selected Python");
    if !torch.success() {
        eprintln!("skipped extension component state test: selected Python needs PyTorch");
        return;
    }

    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("stateful_layer.py"),
        r#"import torch
from torch import nn

_next_marker = 0

class StatefulLayer(nn.Module):
    def __init__(self, marker):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.marker = marker

    def forward(self, values):
        return values * self.weight + self.marker

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_extension_marker"] = self.marker
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state = state_dict.copy()
        self.marker = state.pop("_extension_marker")
        return super().load_state_dict(state, strict=strict, assign=assign)

def make_stateful():
    global _next_marker
    _next_marker += 1
    return StatefulLayer(float(_next_marker))

def state_fixture():
    return {"args": (torch.tensor([1.0, 3.0]),), "kwargs": {}}
"#,
    )
    .unwrap();

    let (registered, registration) = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/stateful-layer",
            "--kind",
            "layer",
            "--entrypoint",
            "stateful_layer:make_stateful",
            "--version",
            "1.0.0",
            "--capabilities",
            r#"{"construct":"supported","state_roundtrip":"supported"}"#,
            "--conformance-fixture",
            "stateful_layer:state_fixture",
        ],
    );
    assert!(registered.status.success(), "{registration:#}");

    let (tested, report) = invoke(
        project.path(),
        &["component", "test", "local/stateful-layer"],
    );
    assert!(
        tested.status.success(),
        "{report:#}\nstderr: {}",
        String::from_utf8_lossy(&tested.stderr)
    );
    assert_eq!(
        report["data"]["capability_results"]["state_roundtrip"]["status"], "passed",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["capability_results"]["state_roundtrip"]["state_entries"],
        2
    );
    assert_eq!(
        report["data"]["capability_results"]["state_roundtrip"]["tensor_values"],
        1
    );
    assert_eq!(report["data"]["coverage"], "complete_for_declared_checks");
}

#[test]
fn layer_component_test_preserves_conditional_claims_and_rejects_source_drift() {
    let torch = Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .expect("start selected Python");
    if !torch.success() {
        eprintln!("skipped extension component conformance test: selected Python needs PyTorch");
        return;
    }

    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("guarded_layer.py"),
        r#"import torch
from torch import nn

class Scaled(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, values):
        return values * self.weight

def make_layer():
    return Scaled()

def batch():
    return torch.ones(2, 3)

def state_fixture():
    return {"args": (torch.ones(2, 3),), "kwargs": {}}
"#,
    )
    .unwrap();

    let (registered, registration) = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/source-guard",
            "--kind",
            "layer",
            "--entrypoint",
            "guarded_layer:make_layer",
            "--version",
            "1.0.0",
            "--conformance-fixture",
            "guarded_layer:state_fixture",
            "--capabilities",
            r#"{"construct":"supported","forward_probe":{"state":"conditional","reason":"Needs a task-specific batch","requires":["task_contract"]},"state_roundtrip":{"state":"conditional","reason":"Needs this task's state schema","requires":["task_contract"]}}"#,
        ],
    );
    assert!(registered.status.success(), "{registration:#}");

    let (conditional, conditional_report) = invoke(
        project.path(),
        &[
            "component",
            "test",
            "local/source-guard",
            "--batch",
            "guarded_layer:batch",
        ],
    );
    assert!(conditional.status.success(), "{conditional_report:#}");
    assert_eq!(
        conditional_report["data"]["capability_results"]["forward_probe"]["status"], "conditional",
        "{conditional_report:#}"
    );
    assert_eq!(
        conditional_report["data"]["capability_results"]["forward_probe"]["observed_status"],
        "passed",
        "{conditional_report:#}"
    );
    assert_eq!(
        conditional_report["data"]["capability_results"]["state_roundtrip"]["status"],
        "conditional",
        "{conditional_report:#}"
    );
    assert_eq!(
        conditional_report["data"]["capability_results"]["state_roundtrip"]["observed_status"],
        "passed",
        "{conditional_report:#}"
    );
    assert_eq!(
        conditional_report["data"]["coverage"], "partial",
        "{conditional_report:#}"
    );

    let (unsupported, unsupported_report) = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/no-construct",
            "--kind",
            "layer",
            "--entrypoint",
            "guarded_layer:make_layer",
            "--version",
            "1.0.0",
            "--capabilities",
            r#"{"construct":{"state":"unsupported","reason":"No factory is exposed"}}"#,
        ],
    );
    assert!(unsupported.status.success(), "{unsupported_report:#}");
    let (rejected_output, rejected) =
        invoke(project.path(), &["component", "test", "local/no-construct"]);
    assert!(!rejected_output.status.success(), "{rejected:#}");
    assert_eq!(
        rejected["diagnostics"][0]["code"], "COMPONENT_CAPABILITY_CONTRADICTION",
        "{rejected:#}"
    );

    fs::write(
        project.path().join("drifting_layer.py"),
        r##"from pathlib import Path
import torch
from torch import nn

class Layer(nn.Module):
    def forward(self, values):
        return values

def make_layer():
    path = Path(__file__)
    source = path.read_text()
    path.write_text(source + "\n# source changed by component factory\n")
    return Layer()
"##,
    )
    .unwrap();
    let (drift_registered, drift_registration) = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/source-drift",
            "--kind",
            "layer",
            "--entrypoint",
            "drifting_layer:make_layer",
            "--version",
            "1.0.0",
        ],
    );
    assert!(drift_registered.status.success(), "{drift_registration:#}");
    let (drift_output, drift) =
        invoke(project.path(), &["component", "test", "local/source-drift"]);
    assert!(!drift_output.status.success(), "{drift:#}");
    assert_eq!(
        drift["diagnostics"][0]["code"], "COMPONENT_SOURCE_CHANGED_DURING_TEST",
        "{drift:#}"
    );

    fs::write(
        project.path().join("drifting_batch.py"),
        r##"from pathlib import Path
import torch

def batch():
    path = Path(__file__)
    source = path.read_text()
    path.write_text(source + "\n# source changed by batch factory\n")
    return torch.ones(2, 3)
"##,
    )
    .unwrap();
    let (batch_registered, batch_registration) = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/batch-drift",
            "--kind",
            "layer",
            "--entrypoint",
            "guarded_layer:make_layer",
            "--version",
            "1.0.0",
            "--capabilities",
            r#"{"construct":"supported","forward_probe":"supported"}"#,
        ],
    );
    assert!(batch_registered.status.success(), "{batch_registration:#}");
    let (batch_output, batch_report) = invoke(
        project.path(),
        &[
            "component",
            "test",
            "local/batch-drift",
            "--batch",
            "drifting_batch:batch",
        ],
    );
    assert!(!batch_output.status.success(), "{batch_report:#}");
    assert_eq!(
        batch_report["diagnostics"][0]["code"], "COMPONENT_SOURCE_CHANGED_DURING_TEST",
        "{batch_report:#}"
    );
}
