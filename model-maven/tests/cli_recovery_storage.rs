use std::{fs, path::Path, process::Command, time::Duration};

#[cfg(unix)]
#[path = "cli_ui_support/mod.rs"]
mod cli_ui_support;

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(unix)]
#[test]
fn checkpoint_failure_can_be_cleaned_then_retrained_in_the_same_project() {
    if !has_torch() {
        eprintln!("skipped E5 recovery: PyTorch is required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let marker = project.path().join(".checkpoint-failure-injected");
    fs::write(
        project.path().join("checkpoint_recovery.py"),
        format!(
            r#"
from pathlib import Path
import torch
from torch import nn

MARKER = Path(r"{}")

def build_model(): return nn.Linear(2, 2)
def batch(): return torch.ones(2, 2)
def step(model, batch, context):
    if not MARKER.exists():
        MARKER.write_text("injected once")
        def no_space(*args, **kwargs): raise OSError("fixture disk full, first run only")
        torch.save = no_space
    return {{"loss": model(batch).square().mean()}}
"#,
            marker.display()
        ),
    )
    .unwrap();
    let init = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            "checkpoint_recovery:build_model",
            "--batch",
            "checkpoint_recovery:batch",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );

    let input = b"3\n3\n1\ncheckpoint_recovery:step\n1\n1\n\n\n\n\nno\nstart\nclean\nCLEAN\nback\n3\n1\ncheckpoint_recovery:step\n1\n1\n\n\n\n\nno\nstart\n11\n8\n";
    let run = cli_ui_support::run_tty(
        Path::new(env!("CARGO_BIN_EXE_m-model")),
        project.path(),
        &["--python", &python()],
        input,
        Duration::from_secs(120),
    )
    .unwrap();
    assert!(!run.timed_out, "{}", run.transcript);
    assert_eq!(run.exit_code, Some(0), "{}", run.transcript);
    assert!(
        run.transcript.contains("CHECKPOINT_WRITE_FAILED"),
        "{}",
        run.transcript
    );
    assert!(
        run.transcript.contains("清理计划预览 · 尚未执行"),
        "{}",
        run.transcript
    );
    assert!(
        run.transcript.contains("清理计划已执行"),
        "{}",
        run.transcript
    );
    assert!(run.transcript.contains("训练完成"), "{}", run.transcript);
    assert_eq!(fs::read_to_string(&marker).unwrap(), "injected once");

    let runs_root = project.path().join(".model-maven/runs");
    let runs = fs::read_dir(&runs_root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("run-"))
        })
        .collect::<Vec<_>>();
    assert_eq!(runs.len(), 2, "{}", run.transcript);
    let statuses = runs
        .iter()
        .map(|path| {
            serde_json::from_slice::<serde_json::Value>(&fs::read(path.join("run.json")).unwrap())
                .unwrap()["status"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert!(statuses.iter().any(|status| status == "failed"));
    assert!(statuses.iter().any(|status| status == "completed"));
    let successful = runs
        .iter()
        .find(|path| {
            serde_json::from_slice::<serde_json::Value>(&fs::read(path.join("run.json")).unwrap())
                .unwrap()["status"]
                == "completed"
        })
        .unwrap();
    assert!(successful.join("checkpoints/latest.json").is_file());
    assert!(successful.join("checkpoints").is_dir());
    let failed = runs
        .iter()
        .find(|path| {
            serde_json::from_slice::<serde_json::Value>(&fs::read(path.join("run.json")).unwrap())
                .unwrap()["status"]
                == "failed"
        })
        .unwrap();
    assert!(!failed.join("worker.log").exists());
    assert!(!failed.join("progress.jsonl").exists());
    assert!(
        marker.is_file(),
        "cleanup must preserve unrelated project files"
    );
}
