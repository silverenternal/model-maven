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
fn registered_trainer_checks_real_progress_events_and_cooperative_stop() {
    let torch = Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .expect("start selected Python");
    if !torch.success() {
        eprintln!("skipped trainer component test: selected Python needs PyTorch");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("trainer_fixture.py"),
        r#"import time

def make_trainer():
    def train(_model, context):
        if context.stop_requested:
            return {"stopped": True}
        while context.remaining_steps:
            if context.timeout_requested:
                break
            if context.remaining_time_seconds is not None:
                time.sleep(0.35)
            context.log("learning_rate", 0.01)
            if not context.report_step(0.25, {"accuracy": 0.75}):
                break
        return {"completed": context.completed_steps}
    return train

def conformance_fixture():
    return {
        "events": {
            "steps": 3,
            "expected_mean_loss": 0.25,
            "expected_metrics": {"learning_rate": 0.01, "accuracy": 0.75},
        },
        "stop": {"steps": 3},
        "timeout": {"steps": 8, "timeout_seconds": 1},
    }
"#,
    )
    .unwrap();

    let (registered, registration) = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/sample-trainer",
            "--kind",
            "trainer",
            "--entrypoint",
            "trainer_fixture:make_trainer",
            "--version",
            "1.0.0",
            "--capabilities",
            r#"{"events":"supported","stop":"supported","timeout":"supported"}"#,
            "--conformance-fixture",
            "trainer_fixture:conformance_fixture",
        ],
    );
    assert!(registered.status.success(), "{registration:#}");

    let (tested, report) = invoke(
        project.path(),
        &["component", "test", "local/sample-trainer"],
    );
    assert!(
        tested.status.success(),
        "{report:#}\nstderr: {}",
        String::from_utf8_lossy(&tested.stderr)
    );
    assert_eq!(
        report["data"]["capability_results"]["events"]["status"], "passed",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["capability_results"]["events"]["completed_steps"],
        3
    );
    assert_eq!(
        report["data"]["capability_results"]["stop"]["status"], "passed",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["capability_results"]["timeout"]["status"], "passed",
        "timeout support requires a passing cooperative deadline fixture"
    );
    assert_eq!(
        report["data"]["capability_results"]["timeout"]["termination_reason"],
        "timeout"
    );
    assert!(
        report["data"]["capability_results"]["timeout"]["completed_steps"]
            .as_u64()
            .unwrap()
            < 8
    );
    assert_eq!(report["data"]["coverage"], "complete_for_declared_checks");

    let (conditional_registered, conditional_registration) = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/conditional-trainer",
            "--kind",
            "trainer",
            "--entrypoint",
            "trainer_fixture:make_trainer",
            "--version",
            "1.0.0",
            "--capabilities",
            r#"{"events":{"state":"conditional","reason":"Requires the task's objective schema","requires":["objective_schema"]}}"#,
            "--conformance-fixture",
            "trainer_fixture:conformance_fixture",
        ],
    );
    assert!(
        conditional_registered.status.success(),
        "{conditional_registration:#}"
    );
    let (conditional_test, conditional_report) = invoke(
        project.path(),
        &["component", "test", "local/conditional-trainer"],
    );
    assert!(conditional_test.status.success(), "{conditional_report:#}");
    assert_eq!(
        conditional_report["data"]["capability_results"]["events"]["status"], "conditional",
        "{conditional_report:#}"
    );
    assert_eq!(
        conditional_report["data"]["capability_results"]["events"]["observed_status"], "passed",
        "{conditional_report:#}"
    );
    assert_eq!(conditional_report["data"]["coverage"], "partial");
}

#[test]
fn trainer_that_ignores_deadline_boundary_fails_timeout_conformance() {
    let torch = Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .expect("start selected Python");
    if !torch.success() {
        eprintln!("skipped trainer timeout test: selected Python needs PyTorch");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("bad_trainer.py"),
        r#"import time

def make_trainer():
    def train(_model, context):
        while context.remaining_steps:
            time.sleep(0.4)
            context.report_step(0.25)
        return {"completed": context.completed_steps}
    return train

def conformance_fixture():
    return {"timeout": {"steps": 8, "timeout_seconds": 1}}
"#,
    )
    .unwrap();

    let (registered, registration) = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/ignores-deadline",
            "--kind",
            "trainer",
            "--entrypoint",
            "bad_trainer:make_trainer",
            "--version",
            "1.0.0",
            "--capabilities",
            r#"{"timeout":"supported"}"#,
            "--conformance-fixture",
            "bad_trainer:conformance_fixture",
        ],
    );
    assert!(registered.status.success(), "{registration:#}");

    let (tested, report) = invoke(
        project.path(),
        &["component", "test", "local/ignores-deadline"],
    );
    assert!(!tested.status.success(), "{report:#}");
    assert_eq!(report["status"], "error", "{report:#}");
    assert_eq!(
        report["diagnostics"][0]["code"], "TRAINER_TIMEOUT_IGNORED",
        "{report:#}"
    );
}
