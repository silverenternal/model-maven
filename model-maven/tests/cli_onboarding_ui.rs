use std::{fs, path::Path, process::Command};

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

fn has_classification_runtime() -> bool {
    Command::new(python())
        .args(["-c", "import torch; from PIL import Image"])
        .status()
        .is_ok_and(|status| status.success())
}

fn class_folders(root: &Path) {
    for label in ["cats", "dogs"] {
        let directory = root.join("data").join(label);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("sample.png"), b"fixture").unwrap();
    }
}

fn classification_images(root: &Path) {
    class_folders(root);
    for label in ["cats", "dogs"] {
        fs::remove_file(root.join("data").join(label).join("sample.png")).unwrap();
    }
    let script = r#"from pathlib import Path
from PIL import Image
import sys
root = Path(sys.argv[1])
for index, label in enumerate(("cats", "dogs")):
    Image.new("RGB", (8, 8), (index * 100, 30, 60)).save(root / "data" / label / "sample.png")
"#;
    let status = Command::new(python())
        .args(["-c", script])
        .arg(root)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "cannot create classification image fixtures"
    );
}

fn model_fixture(root: &Path) {
    fs::write(
        root.join("sample_model.py"),
        r#"
import torch
from torch import nn

def build_model():
    return nn.Linear(2, 2)

def example_batch():
    return torch.ones(1, 2)

def training_step(model, batch, context):
    return {"loss": model(batch).square().mean()}

def slow_batch():
    import time
    time.sleep(5)
    return torch.ones(1, 2)
"#,
    )
    .unwrap();
}

#[cfg(unix)]
#[test]
fn completed_training_run_can_be_evaluated_and_its_previews_browsed_in_the_workbench() {
    if !has_torch() {
        eprintln!("skipped workbench evaluation: PyTorch is required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("interactive_eval.py"),
        r#"
import torch
from torch import nn

def build_model(): return nn.Linear(2, 2)
def batch(): return torch.ones(2, 2), torch.zeros(2, 2)
def step(model, batch, context):
    values, target = batch
    return {"loss": (model(values) - target).square().mean()}
def validation_data(): return {"values": torch.ones(2, 2), "truth": [0, 1]}
def evaluate(model, data, context):
    output = model(data["values"])
    context.report_samples(2)
    if context.max_preview_samples:
        context.report_prediction("sample-0", output[0].tolist(), data["truth"][0])
    return {"metrics": {"fixture/score": 0.5}}
"#,
    )
    .unwrap();
    let trained = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "train",
            "--model",
            "interactive_eval:build_model",
            "--batch",
            "interactive_eval:batch",
            "--step",
            "interactive_eval:step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        trained.status.success(),
        "{}",
        String::from_utf8_lossy(&trained.stderr)
    );
    let trained: serde_json::Value = serde_json::from_slice(&trained.stdout).unwrap();
    assert_eq!(trained["status"], "success");

    let transcript = workbench(
        project.path(),
        b"5\n3\n5\n1\nevaluate\ninteractive_eval:evaluate\ninteractive_eval:validation_data\n\nvalidation\n2\n1\n\n\n\ny\n1\nback\nback\nback\nq\n",
    );
    assert!(
        transcript.contains("evaluate 配置并运行评估"),
        "{transcript}"
    );
    assert!(transcript.contains("评估结果"), "{transcript}");
    assert!(transcript.contains("fixture/score"), "{transcript}");
    assert!(transcript.contains("样本预览 1–1 / 1"), "{transcript}");
    assert!(transcript.contains("sample-0"), "{transcript}");

    let transcript = workbench(
        project.path(),
        b"1\n5\n1\nsamples\n1\nback\nback\nback\nq\n",
    );
    assert!(
        transcript.contains("samples 浏览已报告的样本预览"),
        "{transcript}"
    );
    assert!(transcript.contains("sample-0"), "{transcript}");
    assert!(transcript.contains("fixture/score：0.5"), "{transcript}");
}

#[cfg(unix)]
fn workbench(root: &Path, input: &[u8]) -> String {
    workbench_with_python(root, &python(), input)
}

#[cfg(unix)]
fn workbench_with_python(root: &Path, python: &str, input: &[u8]) -> String {
    let run = cli_ui_support::run_tty(
        Path::new(env!("CARGO_BIN_EXE_m-model")),
        root,
        &["--python", python],
        input,
        std::time::Duration::from_secs(60),
    )
    .unwrap_or_else(|error| panic!("onboarding PTY failed: {error}"));
    assert!(!run.timed_out, "onboarding timed out:\n{}", run.transcript);
    assert_eq!(run.exit_code, Some(0), "{}", run.transcript);
    run.transcript
}

#[cfg(unix)]
#[test]
fn new_project_can_probe_existing_model_temporarily_without_saving_training_configuration() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    model_fixture(project.path());
    let transcript = workbench(
        project.path(),
        b"1\nsample_model:build_model\nsample_model:example_batch\n1\n2\nback\nq\n",
    );
    assert!(transcript.contains("接入已有 Python 模型"), "{transcript}");
    assert!(transcript.contains("本次临时入口"), "{transcript}");
    assert!(transcript.contains("前向证据"), "{transcript}");
    assert!(!transcript.contains("损失函数入口"), "{transcript}");
    assert!(!project.path().join(".model-maven/project.json").exists());
    assert!(!project.path().join(".model-maven/ui-session.json").exists());
}

#[cfg(unix)]
#[test]
fn explicit_save_creates_only_model_and_optional_batch_defaults() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    model_fixture(project.path());
    let transcript = workbench(
        project.path(),
        b"1\nsample_model:build_model\nsample_model:example_batch\n2\n1\nq\n",
    );
    assert!(transcript.contains("已保存默认入口"), "{transcript}");
    assert!(transcript.contains("模型结构"), "{transcript}");
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/project.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(config["model"], "sample_model:build_model");
    assert_eq!(config["batch"], "sample_model:example_batch");
    assert!(config.get("loss").is_none());
    assert!(config.get("optimizer").is_none());
    assert!(config.get("trainer").is_none());
    assert!(!project.path().join(".model-maven/ui-session.json").exists());
}

#[cfg(unix)]
#[test]
fn tiny_cnn_onboarding_creates_editable_project_and_probes_real_images() {
    if !has_classification_runtime() {
        eprintln!("skipped Tiny CNN onboarding: PyTorch and Pillow are required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    classification_images(project.path());
    let transcript = workbench(project.path(), b"3\n1\ndata\n1\n1\nback\nq\n");
    assert!(transcript.contains("检测到 2 个类别目录"), "{transcript}");
    assert!(
        transcript.contains("已创建可编辑的 Tiny CNN 分类预设 classification.py"),
        "{transcript}"
    );
    assert!(transcript.contains("前向证据"), "{transcript}");
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/project.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(config["model"], "classification:model");
    assert_eq!(config["batch"], "classification:batch");
    let adapter = fs::read_to_string(project.path().join("classification.py")).unwrap();
    assert!(adapter.contains("NUM_CLASSES = 2"));
    assert!(adapter.contains("DATA_ROOT = \"data\""));
    assert!(project.path().join("data/cats/sample.png").is_file());

    let python = python();
    let trial = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            python.as_str(),
            "--json",
            "try",
            "--step",
            "classification:step",
            "--steps",
            "1",
            "--training-mode",
            "managed-step",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        trial.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&trial.stdout),
        String::from_utf8_lossy(&trial.stderr)
    );
    let trial: serde_json::Value = serde_json::from_slice(&trial.stdout).unwrap();
    assert_eq!(trial["status"], "success");
    assert_eq!(trial["data"]["completed_steps"], 1);
    assert!(trial["data"]["mean_loss"].as_f64().is_some());
}

#[cfg(unix)]
#[test]
fn resnet18_onboarding_checks_torchvision_before_writing_project_files() {
    if !has_torch() {
        eprintln!("skipped ResNet18 onboarding preflight: PyTorch is required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let transcript = workbench(project.path(), b"3\n2\nq\n");
    if transcript.contains("ResNet18 环境检查未通过") {
        assert!(
            transcript.contains("PRESET_DEPENDENCY_MISSING"),
            "{transcript}"
        );
        assert!(!project.path().join("classification.py").exists());
        assert!(!project.path().join(".model-maven/project.json").exists());
    } else {
        assert!(transcript.contains("分类数据目录"), "{transcript}");
        assert!(transcript.contains("不会自动下载"), "{transcript}");
    }
}

#[cfg(unix)]
#[test]
fn workbench_background_preview_returns_to_editing_then_displays_real_result() {
    if !has_classification_runtime() {
        eprintln!("skipped background preview: PyTorch and Pillow are required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    model_fixture(project.path());
    let python = python();
    let init = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            python.as_str(),
            "init",
            "--model",
            "sample_model:build_model",
            "--batch",
            "sample_model:slow_batch",
            "--batch-binding",
            "single",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&init.stdout),
        String::from_utf8_lossy(&init.stderr)
    );

    let input = "6\n4\nbatch\nsample_model:slow_batch\n\n\n\n\n\n\n\nstart\n返回首页\n2\n返回首页\n6\n4\nwait\nsample_model:training_step\nstart\n3\n8\n";
    let transcript = workbench(project.path(), input.as_bytes());
    assert!(transcript.contains("后台预览已启动"), "{transcript}");
    assert!(transcript.contains("数据预览、产物与清理"), "{transcript}");
    assert!(transcript.contains("编辑模型"), "{transcript}");
    assert!(transcript.contains("后台数据预览运行中"), "{transcript}");
    assert!(
        transcript.contains("后台预览仍在运行") || transcript.contains("数据预览结果"),
        "{transcript}"
    );
    assert!(
        transcript.contains("数据预览") || transcript.contains("预览结果"),
        "{transcript}"
    );
    let state: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/previews/current.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(state["current"]["status"], "succeeded", "{state:#}");
    assert!(transcript.contains("运行计划"), "{transcript}");
    let runs = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", &python, "runs", "list"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        runs.status.success(),
        "{}",
        String::from_utf8_lossy(&runs.stderr)
    );
    let runs: serde_json::Value = serde_json::from_slice(&runs.stdout).unwrap();
    assert!(
        runs["data"]["runs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|run| { run["operation"] == "training_probe" && run["status"] == "completed" }),
        "{runs:#}"
    );
}

#[cfg(unix)]
#[test]
fn failed_background_preview_can_edit_and_retry_without_blocking_the_workbench() {
    if !has_torch() {
        eprintln!("skipped background preview recovery: PyTorch is required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("recovery_model.py"),
        r#"
import time
import torch

def build_model():
    from torch import nn
    return nn.Linear(2, 2)
def example_batch(): return torch.ones(2, 2)
def failing_batch():
    time.sleep(2)
    raise RuntimeError("fixture preview failure")
def slow_batch():
    time.sleep(2)
    return torch.ones(2, 2)
"#,
    )
    .unwrap();
    let init = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            "recovery_model:build_model",
            "--batch",
            "recovery_model:example_batch",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&init.stdout),
        String::from_utf8_lossy(&init.stderr)
    );
    let transcript = workbench(
        project.path(),
        "6\n4\nbatch\nrecovery_model:failing_batch\n\n\n\n\n\n\n\nstart\n4\nwait\nedit\n\nrecovery_model:slow_batch\n\n\n\n\n\n\n\nstart\n4\nwait\nback\n返回首页\nq\n".as_bytes(),
    );
    assert!(
        transcript.contains("fixture preview failure"),
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
fn environment_dependency_failure_routes_to_confirmed_project_setup() {
    if !has_torch() {
        eprintln!("skipped environment recovery: PyTorch is required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(project.path().join("requirements.txt"), "").unwrap();
    fs::write(
        project.path().join("recovery_model.py"),
        r#"
import time
import torch
from torch import nn
from model_maven_worker.protocol import UserError

def build_model(): return nn.Linear(2, 2)
def failing_batch():
    time.sleep(1)
    raise UserError("IMAGE_DECODER_UNAVAILABLE", "fixture dependency unavailable", "prepare the project environment")
"#,
    )
    .unwrap();
    let init = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            "recovery_model:build_model",
            "--batch",
            "recovery_model:failing_batch",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );
    let transcript = workbench(
        project.path(),
        "6\n4\nbatch\nrecovery_model:failing_batch\n\n\n\n\n\n\n\nstart\n4\nwait\nenv\n\nsetup\nback\nback\n返回首页\nq\n".as_bytes(),
    );
    assert!(
        transcript.contains("IMAGE_DECODER_UNAVAILABLE"),
        "{transcript}"
    );
    assert!(transcript.contains("env 准备项目环境"), "{transcript}");
    assert!(
        transcript.contains("环境准备可能下载并安装依赖"),
        "{transcript}"
    );
    assert!(
        transcript.contains("ENVIRONMENT_RESOLUTION_EMPTY"),
        "{transcript}"
    );
    assert!(!project.path().join(".venv").exists());
    let state: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/previews/current.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(state["current"]["status"], "failed", "{state:#}");
}

#[cfg(unix)]
#[test]
fn checkpoint_write_failure_opens_storage_and_safe_cleanup_preview() {
    if !has_torch() {
        eprintln!("skipped checkpoint recovery: PyTorch is required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("checkpoint_failure.py"),
        r#"
import torch
from torch import nn

def build_model(): return nn.Linear(2, 2)
def batch(): return torch.ones(2, 2)
def step(model, batch, context):
    def no_space(*args, **kwargs): raise OSError("fixture disk full")
    torch.save = no_space
    return {"loss": model(batch).square().mean()}
"#,
    )
    .unwrap();
    let init = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            "checkpoint_failure:build_model",
            "--batch",
            "checkpoint_failure:batch",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );
    let transcript = workbench(
        project.path(),
        b"3\n3\n1\ncheckpoint_failure:step\n1\n1\n\n\n\n\nno\nstart\nstorage\nclean\nno\nback\n11\n8\n",
    );
    assert!(
        transcript.contains("CHECKPOINT_WRITE_FAILED"),
        "{transcript}"
    );
    assert!(
        transcript.contains("Checkpoint 保存失败：storage 查看运行占用"),
        "{transcript}"
    );
    assert!(transcript.contains("运行记录占用"), "{transcript}");
    assert!(
        transcript.contains("清理计划预览 · 尚未执行"),
        "{transcript}"
    );
    assert!(transcript.contains("已取消；文件未改变"), "{transcript}");
    let runs = fs::read_dir(project.path().join(".model-maven/runs"))
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
    assert_eq!(runs.len(), 1);
    assert!(runs[0].join("worker.log").is_file());
    let latest = runs[0].join("checkpoints/latest.json");
    assert!(
        !latest.exists(),
        "failed save must not publish a checkpoint"
    );
}

#[cfg(unix)]
#[test]
fn damaged_class_images_can_be_explicitly_excluded_and_are_not_deleted() {
    if !has_classification_runtime() {
        eprintln!("skipped damaged-image CLI flow: PyTorch and Pillow are required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    classification_images(project.path());
    let damaged = project.path().join("data/cats/sample.png");
    fs::write(&damaged, b"not an image").unwrap();
    let init = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            "fixture:build_model",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );
    let python = python();
    let json_report = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            python.as_str(),
            "data",
            "index",
            "--root",
            "data",
            "--verify",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&json_report.stdout).unwrap();
    assert_eq!(json["diagnostics"][0]["code"], "CLASS_SAMPLES_DAMAGED");
    assert!(
        damaged.is_file(),
        "JSON inspection must not mutate source files"
    );
    assert!(
        !project
            .path()
            .join(".model-maven/data-exclusions.json")
            .exists(),
        "JSON inspection must not apply exclusions"
    );
    let run = cli_ui_support::run_tty(
        Path::new(env!("CARGO_BIN_EXE_m-model")),
        project.path(),
        &[
            "--python",
            python.as_str(),
            "data",
            "index",
            "--root",
            "data",
            "--verify",
        ],
        b"1\nEXCLUDE\n",
        std::time::Duration::from_secs(60),
    )
    .unwrap();
    assert!(!run.timed_out, "{}", run.transcript);
    assert_eq!(run.exit_code, Some(0), "{}", run.transcript);
    assert!(
        run.transcript.contains("CLASS_SAMPLES_DAMAGED"),
        "{}",
        run.transcript
    );
    assert!(run.transcript.contains("cat"), "{}", run.transcript);
    assert!(
        run.transcript.contains("已排除（不删除原文件）：1 项"),
        "{}",
        run.transcript
    );
    assert!(damaged.is_file(), "the user file must remain in place");
    let protocol: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/data-exclusions.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        protocol["excluded_files"],
        serde_json::json!(["cats/sample.png"])
    );

    let python_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let output = Command::new(&python)
        .arg("-c")
        .arg("import json; from model_maven_worker.data.presets import classification_batch; result=classification_batch(root='data',batch_size=8,split_ratios={'train':1.0},transforms=[{'type':'Resize','size':2},{'type':'ToTensor'}]); print(json.dumps(result['dataset']))")
        .env("PYTHONPATH", &python_path)
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let batch: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let sample_ids = batch["sample_ids"].as_array().unwrap();
    assert!(!sample_ids.iter().any(|sample| sample == "cats/sample.png"));
    assert_eq!(sample_ids.len(), 1);

    let repaired = Command::new(&python)
        .args([
            "-c",
            "from PIL import Image; Image.new('RGB',(3,3),(10,20,30)).save('data/cats/sample.png')",
        ])
        .current_dir(project.path())
        .status()
        .unwrap();
    assert!(repaired.success());
    let cleared = cli_ui_support::run_tty(
        Path::new(env!("CARGO_BIN_EXE_m-model")),
        project.path(),
        &[
            "--python",
            python.as_str(),
            "data",
            "index",
            "--root",
            "data",
            "--verify",
            "--clear-exclusions",
        ],
        b"CLEAR\n",
        std::time::Duration::from_secs(60),
    )
    .unwrap();
    assert!(!cleared.timed_out, "{}", cleared.transcript);
    assert_eq!(cleared.exit_code, Some(0), "{}", cleared.transcript);
    assert!(
        cleared.transcript.contains("已清除项目排除协议"),
        "{}",
        cleared.transcript
    );
    assert!(
        !project
            .path()
            .join(".model-maven/data-exclusions.json")
            .exists()
    );
    let output = Command::new(&python)
        .arg("-c")
        .arg("import json; from model_maven_worker.data.presets import classification_batch; result=classification_batch(root='data',batch_size=8,split_ratios={'train':1.0},transforms=[{'type':'Resize','size':2},{'type':'ToTensor'}]); print(json.dumps(result['dataset']))")
        .env("PYTHONPATH", python_path)
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let batch: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let sample_ids = batch["sample_ids"].as_array().unwrap();
    assert!(sample_ids.iter().any(|sample| sample == "cats/sample.png"));
    assert_eq!(sample_ids.len(), 2);
}

#[cfg(unix)]
#[test]
fn workbench_background_preview_cancel_marks_request_cancelled() {
    if !has_classification_runtime() {
        eprintln!("skipped background preview cancellation: PyTorch and Pillow are required");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("sample_model.py"),
        r#"
import time
import torch
from torch import nn

def build_model():
    return nn.Linear(2, 2)

def slow_batch():
    time.sleep(30)
    return torch.ones(1, 2)
"#,
    )
    .unwrap();
    let python = python();
    let init = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            python.as_str(),
            "init",
            "--model",
            "sample_model:build_model",
            "--batch",
            "sample_model:slow_batch",
            "--batch-binding",
            "single",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );

    let input =
        "6\n4\nbatch\nsample_model:slow_batch\n\n\n\n\n\n\n\nstart\n4\ncancel\n返回首页\nq\n";
    let transcript = workbench(project.path(), input.as_bytes());
    assert!(transcript.contains("后台预览已启动"), "{transcript}");
    assert!(
        transcript.contains("取消") || transcript.contains("USER_INTERRUPTED"),
        "{transcript}"
    );
    let state: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/previews/current.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(state["current"]["status"], "cancelled", "{state:#}");
}

#[cfg(unix)]
#[test]
fn preset_cancel_and_existing_adapter_conflict_leave_project_files_untouched() {
    let cancelled = tempfile::tempdir().unwrap();
    class_folders(cancelled.path());
    let transcript = workbench(cancelled.path(), b"3\n1\ndata\n1\n2\nq\n");
    assert!(transcript.contains("已取消"), "{transcript}");
    assert!(!cancelled.path().join("classification.py").exists());
    assert!(!cancelled.path().join(".model-maven/project.json").exists());

    let conflict = tempfile::tempdir().unwrap();
    class_folders(conflict.path());
    let existing = conflict.path().join("classification.py");
    fs::write(&existing, "# keep this file\n").unwrap();
    let transcript = workbench(conflict.path(), b"3\n1\ndata\n1\n1\nq\n");
    assert!(
        transcript.contains("classification.py 已存在"),
        "{transcript}"
    );
    assert_eq!(fs::read_to_string(existing).unwrap(), "# keep this file\n");
    assert!(!conflict.path().join(".model-maven/project.json").exists());

    let symlinked = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    class_folders(symlinked.path());
    std::os::unix::fs::symlink(external.path(), symlinked.path().join(".model-maven")).unwrap();
    let transcript = workbench(symlinked.path(), b"3\n1\ndata\n1\n1\nq\n");
    assert!(
        transcript.contains(".model-maven 必须是普通项目目录"),
        "{transcript}"
    );
    assert!(!symlinked.path().join("classification.py").exists());
    assert!(!external.path().join("project.json").exists());
}

#[cfg(unix)]
#[test]
fn failed_onboarding_can_be_restored_after_restart_without_rerunning_then_discarded() {
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("failing_model.py"),
        "def build_model():\n    raise RuntimeError('intentional onboarding failure')\n",
    )
    .unwrap();
    let first = workbench(project.path(), b"1\nfailing_model:build_model\n\n1\n1\nq\n");
    assert!(first.contains("intentional onboarding failure"), "{first}");
    let session_path = project.path().join(".model-maven/ui-session.json");
    assert!(session_path.is_file());

    let restored = workbench(project.path(), b"1\nq\n");
    assert!(restored.contains("恢复上次接入"), "{restored}");
    assert!(restored.contains("failing_model:build_model"), "{restored}");
    assert!(restored.contains("恢复的临时入口"), "{restored}");
    assert!(
        !restored.contains("intentional onboarding failure"),
        "{restored}"
    );
    assert!(session_path.is_file());

    let discarded = workbench(project.path(), b"2\nq\n");
    assert!(discarded.contains("已丢弃接入会话"), "{discarded}");
    assert!(!session_path.exists());
}

#[cfg(unix)]
#[test]
fn onboarding_restore_requires_revalidation_after_transitive_local_model_source_changes() {
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("failing_model.py"),
        "from failing_model_helpers import build_model_impl\ndef build_model():\n    return build_model_impl()\n",
    )
    .unwrap();
    let dependency = project.path().join("failing_model_helpers.py");
    fs::write(
        &dependency,
        "def build_model_impl():\n    raise RuntimeError('first version')\n",
    )
    .unwrap();
    let first = workbench(project.path(), b"1\nfailing_model:build_model\n\n1\n1\nq\n");
    assert!(first.contains("first version"), "{first}");
    assert!(
        project
            .path()
            .join(".model-maven/ui-session.json")
            .is_file()
    );

    fs::write(
        &dependency,
        "def build_model_impl():\n    raise RuntimeError('changed version')\n",
    )
    .unwrap();
    let restored = workbench(project.path(), b"1\nq\n");
    assert!(restored.contains("恢复上次接入"), "{restored}");
    assert!(restored.contains("模型源码已变化"), "{restored}");
    assert!(restored.contains("重新检查模型入口"), "{restored}");
    assert!(!restored.contains("changed version"), "{restored}");
    assert!(
        project
            .path()
            .join(".model-maven/ui-session.json")
            .is_file()
    );
}

#[cfg(unix)]
#[test]
fn project_open_switches_directory_and_uses_the_saved_project_defaults() {
    let parent = tempfile::tempdir().unwrap();
    let source = parent.path().join("source");
    let target = parent.path().join("target");
    fs::create_dir_all(&source).unwrap();
    fs::create_dir_all(&target).unwrap();
    let init = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            "sample_model:build_model",
        ])
        .current_dir(&target)
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );
    let transcript = workbench(
        &source,
        format!("2\n1\n{}\n7\n1\n\nq\n", target.display()).as_bytes(),
    );
    assert!(transcript.contains("打开项目"), "{transcript}");
    assert!(
        transcript.contains("sample_model:build_model"),
        "{transcript}"
    );
    assert!(transcript.contains("项目默认配置"), "{transcript}");
    assert!(transcript.contains("配置与输入来源"), "{transcript}");
    assert!(
        transcript.contains("默认模型：sample_model:build_model"),
        "{transcript}"
    );
    assert!(transcript.contains("配置版本："), "{transcript}");
}

#[cfg(unix)]
#[test]
fn workbench_page_selection_can_be_resumed_or_discarded_after_restart() {
    let project = tempfile::tempdir().unwrap();
    let page_path = project.path().join(".model-maven/ui-page-session.json");

    let first = workbench(project.path(), b"5\n4\nq\n");
    assert!(first.contains("模型工作台"), "{first}");
    assert!(page_path.is_file());
    let draft: serde_json::Value = serde_json::from_slice(&fs::read(&page_path).unwrap()).unwrap();
    assert_eq!(draft["page_id"], "components");
    let canonical_root = project
        .path()
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        draft["project_root"].as_str(),
        Some(canonical_root.as_str())
    );

    let resumed = workbench(project.path(), b"1\nq\n");
    assert!(resumed.contains("继续上次的工作台位置"), "{resumed}");
    assert!(resumed.contains("上次离开时停在：组件"), "{resumed}");
    assert!(resumed.contains("不会重放模型检查"), "{resumed}");
    assert!(resumed.contains("组件"), "{resumed}");
    assert!(resumed.contains("检查已登记组件"), "{resumed}");
    assert!(page_path.is_file());

    let discarded = workbench(project.path(), b"2\nq\n");
    assert!(discarded.contains("已丢弃页面恢复点"), "{discarded}");
    assert!(!page_path.exists());
}

#[cfg(unix)]
#[test]
fn invalid_preset_path_is_explained_and_workbench_remains_available() {
    let project = tempfile::tempdir().unwrap();
    let transcript = workbench(project.path(), b"3\n1\nmissing-data\n5\nq\n");
    assert!(transcript.contains("无法打开所选数据目录"), "{transcript}");
    assert!(transcript.contains("模型工作台"), "{transcript}");
}

#[cfg(unix)]
#[test]
fn missing_python_is_reported_with_recovery_actions_before_model_entry() {
    let project = tempfile::tempdir().unwrap();
    let transcript =
        workbench_with_python(project.path(), "/model-maven-test/missing-python", b"q\n");
    assert!(
        transcript.contains("找不到 Python 可执行文件"),
        "{transcript}"
    );
    let normalized = transcript.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(normalized.contains("运行 m-model doctor"), "{transcript}");
    assert!(
        normalized.contains("--python 指定已有解释器"),
        "{transcript}"
    );
}
