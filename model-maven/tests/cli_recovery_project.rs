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

fn initialize(project: &Path, model: &str, batch: &str) {
    let result = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            model,
            "--batch",
            batch,
        ])
        .current_dir(project)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[cfg(unix)]
#[test]
fn project_switch_keeps_background_preview_recovery_bound_to_its_origin() {
    if !has_torch() {
        eprintln!("skipped cross-project preview recovery: PyTorch is required");
        return;
    }
    let parent = tempfile::tempdir().unwrap();
    let project_a = parent.path().join("project-a");
    let project_b = parent.path().join("project-b");
    fs::create_dir_all(&project_a).unwrap();
    fs::create_dir_all(&project_b).unwrap();
    fs::write(
        project_a.join("a_model.py"),
        "import torch\nfrom torch import nn\ndef build_model(): return nn.Linear(2, 2)\n",
    )
    .unwrap();
    fs::write(
        project_a.join("a_data.py"),
        r#"
from pathlib import Path
import time
import torch
from model_maven_worker.protocol import UserError

def build_batch():
    time.sleep(8)
    marker = Path('.model-maven/retry-this-preview-once')
    if marker.exists():
        marker.unlink()
        raise UserError('IMAGE_DECODER_UNAVAILABLE', 'fixture decoder unavailable once', 'retry from the originating project')
    return torch.ones(1, 2)
"#,
    )
    .unwrap();
    fs::write(
        project_b.join("b_model.py"),
        r#"
import torch
from torch import nn
from pathlib import Path

def build_model(): return nn.Linear(100, 2)
def build_step(model, batch, context):
    Path('b-training-step-called').write_text('called')
    return {'loss': model(batch).square().mean()}
"#,
    )
    .unwrap();
    fs::write(
        project_b.join("b_data.py"),
        "import torch\ndef build_batch(): return torch.ones(1, 100)\n",
    )
    .unwrap();
    initialize(&project_a, "a_model:build_model", "a_data:build_batch");
    initialize(&project_b, "b_model:build_model", "b_data:build_batch");
    fs::write(
        project_a.join(".model-maven/retry-this-preview-once"),
        "fail initial request only",
    )
    .unwrap();

    // A's first worker sleeps while the workbench opens B. The second request
    // succeeds because the first attempt consumed the one-shot marker.
    let input = format!(
        "6\n4\nbatch\na_data:build_batch\n\n\n\n\n\n\n\nstart\n3\n7\n3\n1\n{}\n6\n4\nwait\nretry\nstart\n4\nwait\n3\nq\n",
        project_b.display()
    );
    let run = cli_ui_support::run_tty(
        Path::new(env!("CARGO_BIN_EXE_m-model")),
        &project_a,
        &[],
        input.as_bytes(),
        Duration::from_secs(120),
    )
    .unwrap();
    assert!(!run.timed_out, "{}", run.transcript);
    assert_eq!(run.exit_code, Some(0), "{}", run.transcript);
    for expected in [
        "打开/切换项目目录",
        "打开项目",
        "IMAGE_DECODER_UNAVAILABLE",
        "后台预览重试已启动",
        "当前预览 · 本次请求已发布",
        "此预览属于另一项目目录；结果已展示，但不会与当前项目模型组合试跑。",
    ] {
        assert!(
            run.transcript.contains(expected),
            "missing {expected}: {}",
            run.transcript
        );
    }
    assert!(
        !run.transcript.contains(
            "可将当前模型 `b_model:build_model` 与预览使用的批次工厂 `a_data:build_batch`"
        ),
        "the A batch must never be offered with B's model: {}",
        run.transcript
    );
    assert!(!project_b.join("b-training-step-called").exists());
    assert!(
        !project_b
            .join(".model-maven/previews/current.json")
            .exists()
    );

    let state_a: serde_json::Value = serde_json::from_slice(
        &fs::read(project_a.join(".model-maven/previews/current.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(state_a["generation"], 2, "{state_a:#}");
    assert_eq!(state_a["current"]["generation"], 2, "{state_a:#}");
    assert_eq!(state_a["current"]["status"], "succeeded", "{state_a:#}");
    assert_eq!(
        state_a["current"]["request_id"], state_a["last_success"]["request_id"],
        "{state_a:#}"
    );
    assert_eq!(
        state_a["current"]["request_signature"], state_a["last_success"]["request_signature"],
        "{state_a:#}"
    );
    assert!(
        !project_a
            .join(".model-maven/retry-this-preview-once")
            .exists()
    );
}
