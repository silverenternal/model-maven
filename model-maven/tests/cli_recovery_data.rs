use std::{fs, path::Path, process::Command, time::Duration};

#[cfg(unix)]
#[path = "cli_ui_support/mod.rs"]
mod cli_ui_support;

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_classification_runtime() -> bool {
    Command::new(python())
        .args(["-c", "import torch; from PIL import Image"])
        .status()
        .is_ok_and(|status| status.success())
}

fn create_class_images(project: &Path) {
    let script = r#"from pathlib import Path
from PIL import Image
root = Path('data')
for index, label in enumerate(('cats', 'dogs')):
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (8, 8), (index * 100, 30, 60)).save(folder / 'sample.png')
"#;
    let status = Command::new(python())
        .args(["-c", script])
        .current_dir(project)
        .status()
        .unwrap();
    assert!(status.success());
    fs::write(
        project.join("recovery_data.py"),
        "from model_maven_worker.data.presets import classification_batch\n\ndef build_data(root='data', selected_split='train'):\n    return classification_batch(root=root, split=selected_split, split_ratios={'train': 1.0})\n",
    )
    .unwrap();
}

fn initialize_project(project: &Path) {
    fs::write(
        project.join("sample_model.py"),
        "import torch\nfrom torch import nn\ndef build_model(): return nn.Linear(2, 2)\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            "sample_model:build_model",
        ])
        .current_dir(project)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[cfg(unix)]
fn run_workbench(project: &Path, input: &str) -> String {
    let run = cli_ui_support::run_tty(
        Path::new(env!("CARGO_BIN_EXE_m-model")),
        project,
        &["--python", &python()],
        input.as_bytes(),
        Duration::from_secs(90),
    )
    .unwrap();
    assert!(!run.timed_out, "{}", run.transcript);
    assert_eq!(run.exit_code, Some(0), "{}", run.transcript);
    run.transcript
}

#[cfg(unix)]
fn preview_input(
    initial_root: &str,
    initial_selected_split: &str,
    retry_parameter: Option<(&str, &str)>,
) -> String {
    let mut input = format!(
        "6\n4\nbatch\nrecovery_data:build_data\nroot={initial_root}\nselected_split={initial_selected_split}\n\ntrain\n\n\n\n\n\nstart\n返回首页\n6\n4\nwait\nedit\n\n\n"
    );
    if let Some((name, value)) = retry_parameter {
        input.push_str(&format!("{name}={value}\n"));
    }
    input.push_str("\ntrain\n\n\n\n\n\nstart\n返回首页\n6\n4\nwait\n返回首页\nq\n");
    input
}

#[cfg(unix)]
#[test]
fn class_dataset_root_failure_can_be_edited_and_retried() {
    if !has_classification_runtime() {
        eprintln!("skipped CLI E1 recovery: PyTorch and Pillow are required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    create_class_images(project.path());
    initialize_project(project.path());
    let transcript = run_workbench(
        project.path(),
        &preview_input("\"./missing\"", "\"train\"", Some(("root", "\"./data\""))),
    );
    assert!(
        transcript.contains("CLASS_DATASET_ROOT_INVALID"),
        "{transcript}"
    );
    assert!(
        transcript.contains("预览未成功：edit 修改设置后重试"),
        "{transcript}"
    );
    assert!(transcript.contains("后台预览重试已启动"), "{transcript}");
    assert!(
        transcript.contains("当前预览 · 本次请求已发布"),
        "{transcript}"
    );
    let state: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/previews/current.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(state["current"]["status"], "succeeded", "{state:#}");
}

#[cfg(unix)]
#[test]
fn empty_split_failure_can_be_edited_to_train_and_retried() {
    if !has_classification_runtime() {
        eprintln!("skipped CLI E2 recovery: PyTorch and Pillow are required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    create_class_images(project.path());
    initialize_project(project.path());
    let transcript = run_workbench(
        project.path(),
        &preview_input(
            "\"./data\"",
            "\"validation\"",
            Some(("selected_split", "\"train\"")),
        ),
    );
    assert!(transcript.contains("DATA_SPLIT_EMPTY"), "{transcript}");
    assert!(
        transcript.contains("预览未成功：edit 修改设置后重试"),
        "{transcript}"
    );
    assert!(transcript.contains("后台预览重试已启动"), "{transcript}");
    assert!(
        transcript.contains("当前预览 · 本次请求已发布"),
        "{transcript}"
    );
    let state: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/previews/current.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(state["current"]["status"], "succeeded", "{state:#}");
}
