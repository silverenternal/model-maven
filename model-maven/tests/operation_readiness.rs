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

fn invoke(root: &Path, args: &[&str]) -> (std::process::Output, Value) {
    invoke_with_python(root, &python(), args)
}

fn invoke_with_python(root: &Path, python: &str, args: &[&str]) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", python])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report = serde_json::from_slice(stdout.trim().as_bytes()).unwrap_or_else(|error| {
        panic!(
            "expected one JSON report, got {stdout:?}: {error}; status={:?}; stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

#[test]
fn operation_report_includes_runtime_capabilities_and_redacted_input_plan() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        r#"
import torch
from torch import nn

def build():
    return nn.Sequential(nn.Linear(4, 2))
"#,
    )
    .unwrap();

    let (output, report) = invoke(
        project.path(),
        &["model", "inspect", "--model", "model:build"],
    );
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["data"]["readiness"]["operation"], "inspect_model");
    assert_eq!(report["data"]["readiness"]["readiness"]["state"], "ready");
    let capability = report["data"]["readiness"]["readiness"]["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == "model.inspect")
        .unwrap();
    assert_eq!(capability["state"], "available");
    assert_eq!(capability["required"], true);
    assert_eq!(capability["provider"], "python-worker");
    assert!(
        report["data"]["readiness"]["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|input| input.get("value").is_none())
    );
}

#[test]
fn missing_operation_input_does_not_claim_unchecked_runtime_capability() {
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "def build():\n    raise AssertionError('model worker must not start')\n",
    )
    .unwrap();

    let (output, report) = invoke(
        project.path(),
        &["model", "probe", "--model", "model:build"],
    );
    assert!(!output.status.success());
    assert_eq!(report["diagnostics"][0]["code"], "OPERATION_NEEDS_INPUT");
    let readiness = &report["data"]["readiness"]["readiness"];
    assert_eq!(readiness["state"], "needs_input");
    let capability = readiness["capabilities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == "model.forward_probe")
        .unwrap();
    assert_eq!(capability["state"], "unknown");
    assert_eq!(capability["required"], false);
    assert!(
        readiness["conditions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|condition| condition.as_str().unwrap().contains("补齐必需输入后再检测"))
    );
    assert!(!project.path().join(".model-maven/runs").exists());
}

#[cfg(unix)]
#[test]
fn unsupported_runtime_capability_stops_the_target_operation() {
    use std::os::unix::fs::PermissionsExt;

    let project = tempfile::tempdir().unwrap();
    let fake_python = project.path().join("fake-python");
    fs::write(
        &fake_python,
        r##"#!/bin/sh
cat >/dev/null
cat <<'JSON'
{"protocol_version":1,"operation":"doctor","status":"success","data":{"compatibility":{"worker_protocol_version":1,"worker_api_version":1,"state_formats":{}},"capabilities":[{"id":"model.inspect","state":"unsupported","reason":"test runtime has no model inspection support","provider":"fake-runtime"}]},"diagnostics":[]}
JSON
"##,
    )
    .unwrap();
    fs::set_permissions(&fake_python, fs::Permissions::from_mode(0o755)).unwrap();

    let (output, report) = invoke_with_python(
        project.path(),
        fake_python.to_str().unwrap(),
        &["model", "inspect", "--model", "fixture:build"],
    );
    assert!(!output.status.success());
    assert_eq!(
        report["diagnostics"][0]["code"],
        "OPERATION_CAPABILITY_UNAVAILABLE"
    );
    let readiness = &report["data"]["readiness"]["readiness"];
    assert_eq!(readiness["state"], "unsupported");
    assert_eq!(readiness["capabilities"][0]["state"], "unsupported");
    assert_eq!(readiness["capabilities"][0]["provider"], "fake-runtime");
}

#[cfg(unix)]
#[test]
fn interrupt_during_capability_check_cancels_instead_of_running_the_operation() {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Stdio;

    let project = tempfile::tempdir().unwrap();
    let fake_python = project.path().join("slow-python");
    fs::write(&fake_python, "#!/bin/sh\ncat >/dev/null\nsleep 10\n").unwrap();
    fs::set_permissions(&fake_python, fs::Permissions::from_mode(0o755)).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            fake_python.to_str().unwrap(),
            "model",
            "inspect",
            "--model",
            "does_not_exist:build",
        ])
        .current_dir(project.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert_eq!(
        unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) },
        0
    );
    let output = child.wait_with_output().unwrap();
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(output.status.code(), Some(130));
    assert_eq!(report["diagnostics"][0]["code"], "USER_INTERRUPTED");
}

#[test]
fn component_check_reports_static_requirements_without_starting_worker() {
    let project = tempfile::tempdir().unwrap();
    let (added, add_report) = invoke(
        project.path(),
        &[
            "component",
            "add",
            "demo/ready-layer",
            "--kind",
            "layer",
            "--entrypoint",
            "layers:make_layer",
            "--version",
            "1.0.0",
        ],
    );
    assert!(added.status.success(), "{add_report:#}");

    let (checked, report) = invoke(project.path(), &["component", "check", "demo/ready-layer"]);
    assert!(checked.status.success(), "{report:#}");
    let readiness = &report["data"]["readiness"];
    assert_eq!(readiness["operation"], "component_check");
    assert_eq!(readiness["readiness"]["state"], "ready");
    assert!(
        readiness["readiness"]["facts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|fact| fact["key"] == "component_dependencies" && fact["state"] == "available")
    );
    assert!(
        readiness["readiness"]["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|capability| capability["id"] == "component.static_metadata")
    );
}

#[cfg(unix)]
#[test]
fn unsupported_layer_and_training_capabilities_block_before_target_dispatch() {
    use std::os::unix::fs::PermissionsExt;

    let project = tempfile::tempdir().unwrap();
    for (capability, args, expected_operation) in [
        (
            "model.layer_edit",
            vec![
                "layer",
                "add",
                "--parent",
                "model",
                "--name",
                "act",
                "--type",
                "ReLU",
                "--model",
                "fixture:build",
                "--dry-run",
            ],
            "plan_layer_edit",
        ),
        (
            "training.managed_step",
            vec![
                "try",
                "--model",
                "fixture:build",
                "--batch",
                "fixture:batch",
                "--step",
                "fixture:step",
                "--steps",
                "1",
            ],
            "training_probe",
        ),
    ] {
        let marker = project.path().join(format!("{capability}.dispatched"));
        let fake_python = project.path().join(format!("fake-{capability}"));
        let doctor = format!(
            r#"{{"protocol_version":1,"operation":"doctor","status":"success","data":{{"compatibility":{{"worker_protocol_version":1,"worker_api_version":1,"state_formats":{{}}}},"capabilities":[{{"id":"{capability}","state":"unsupported","reason":"disabled for readiness test","provider":"fake-runtime"}}]}},"diagnostics":[]}}"#
        );
        let script = format!(
            "#!/bin/sh\nrequest=$(cat)\ncase \"$request\" in\n  *doctor*) cat <<'JSON'\n{doctor}\nJSON\n    ;;\n  *) touch '{}'\n     cat <<'JSON'\n{{\"protocol_version\":1,\"operation\":\"inspect\",\"status\":\"success\",\"data\":{{}},\"diagnostics\":[]}}\nJSON\n    ;;\nesac\n",
            marker.display()
        );
        fs::write(&fake_python, script).unwrap();
        fs::set_permissions(&fake_python, fs::Permissions::from_mode(0o755)).unwrap();

        let string_args = args.into_iter().collect::<Vec<_>>();
        let (output, report) =
            invoke_with_python(project.path(), fake_python.to_str().unwrap(), &string_args);
        assert!(!output.status.success(), "{report:#}");
        assert_eq!(
            report["diagnostics"][0]["code"],
            "OPERATION_CAPABILITY_UNAVAILABLE"
        );
        assert_eq!(report["data"]["readiness"]["operation"], expected_operation);
        assert!(!marker.exists(), "target worker was dispatched");
    }
}

#[cfg(unix)]
#[test]
fn malformed_doctor_catalog_stays_unknown_and_target_runtime_can_validate() {
    use std::os::unix::fs::PermissionsExt;

    let project = tempfile::tempdir().unwrap();
    let marker = project.path().join("target-dispatched");
    let fake_python = project.path().join("fake-python");
    fs::write(
        &fake_python,
        format!(
            "#!/bin/sh\nrequest=$(cat)\ncase \"$request\" in\n  *doctor*) cat <<'JSON'\n{{\"protocol_version\":1,\"operation\":\"doctor\",\"status\":\"success\",\"data\":{{\"compatibility\":{{\"worker_protocol_version\":1,\"worker_api_version\":1,\"state_formats\":{{}}}}}},\"diagnostics\":[]}}\nJSON\n    ;;\n  *) touch '{}'\n     cat <<'JSON'\n{{\"protocol_version\":1,\"operation\":\"inspect\",\"status\":\"success\",\"data\":{{}},\"diagnostics\":[]}}\nJSON\n    ;;\nesac\n",
            marker.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake_python, fs::Permissions::from_mode(0o755)).unwrap();

    let (output, report) = invoke_with_python(
        project.path(),
        fake_python.to_str().unwrap(),
        &["model", "inspect", "--model", "fixture:build"],
    );
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["data"]["readiness"]["readiness"]["state"], "unknown");
    assert!(
        report["data"]["readiness"]["readiness"]["conditions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|condition| condition.as_str().unwrap().contains("能力未能确认"))
    );
    assert!(
        marker.is_file(),
        "target runtime should validate the operation"
    );
}
