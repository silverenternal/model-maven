use serde_json::Value;
use std::{fs, path::Path, process::Command, sync::Mutex};

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    static PYTORCH_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PYTORCH_AVAILABLE.get_or_init(|| {
        Command::new(python())
            .args(["-c", "import torch"])
            .status()
            .is_ok_and(|status| status.success())
    })
}

fn gpu_count() -> usize {
    static AVAILABLE_GPUS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *AVAILABLE_GPUS.get_or_init(|| {
        let output = Command::new(python())
            .args([
                "-c",
                "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)",
            ])
            .output();
        output
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|output| output.trim().parse().ok())
            .unwrap_or(0)
    })
}

#[test]
fn auto_device_falls_back_to_cpu_when_runtime_has_no_gpu() {
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let output = Command::new(python())
        .args([
            "-c",
            r#"
from types import SimpleNamespace
from model_maven_worker.train.device import resolve_device
class FakeCuda:
    def is_available(self): return False
    def device_count(self): return 0
class FakeDevice:
    type = "cpu"
class FakeTorch:
    cuda = FakeCuda()
    version = SimpleNamespace(hip=None)
    def device(self, name):
        assert name == "cpu"
        return FakeDevice()
selection = resolve_device(FakeTorch(), "auto")
assert selection.report() == {"device":"cpu", "device_requested":"auto", "device_backend":"cpu", "device_name":"CPU"}
"#,
        ])
        .env("PYTHONPATH", package)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "auto fallback fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn invoke(root: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--verbose", "--python", &python()])
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

fn assert_success(output: &std::process::Output, report: &Value) {
    assert!(
        output.status.success(),
        "unexpected failure: {report:#}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(report["status"], "success", "{report:#}");
}

fn write_fixture(root: &Path) {
    fs::write(
        root.join("device_fixture.py"),
        r#"
import torch
from torch import nn

def build_model():
    return nn.Linear(2, 1)

def build_batch():
    return {"inputs": (torch.tensor([[1.0, 2.0]]),), "target": torch.tensor([[1.0]])}

def train_step(model, batch, context):
    x, = batch["inputs"]
    assert x.device == model.weight.device
    assert batch["target"].device == model.weight.device
    prediction = model(x)
    loss = ((prediction - batch["target"]) ** 2).mean()
    return {"loss": loss, "outputs": {"device": str(x.device)}}

def oom_step(model, batch, context):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected for diagnostic test)")

def manual_step(model, batch, context):
    x, = batch["inputs"]
    optimizer = context.optimizer("main", model.parameters())
    optimizer.zero_grad(set_to_none=True)
    loss = ((model(x) - batch["target"]) ** 2).mean()
    context.backward(loss)
    optimizer.step()
    return {"loss": loss, "outputs": {"device": str(x.device)}}

build_batch.MODEL_MAVEN_STATELESS = True
train_step.MODEL_MAVEN_STATELESS = True

def run_loop(model, context):
    batch = context.to_device(build_batch())
    x, = batch["inputs"]
    assert x.device == model.weight.device
    for _ in range(context.remaining_steps):
        loss = ((model(x) - batch["target"]) ** 2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in model.parameters():
            parameter.data -= 0.01 * parameter.grad
        if not context.report_step(loss):
            break

run_loop.MODEL_MAVEN_STATELESS = True
"#,
    )
    .unwrap();
}

#[test]
fn cpu_is_an_explicit_device_and_reports_measured_process_memory() {
    if !has_torch() {
        eprintln!("skipped device integration: Python needs PyTorch");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    write_fixture(project.path());
    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--device",
            "cpu",
            "--model",
            "device_fixture:build_model",
            "--batch",
            "device_fixture:build_batch",
            "--step",
            "device_fixture:train_step",
            "--steps",
            "1",
        ],
    );
    assert_success(&output, &report);
    assert_eq!(report["data"]["runtime"]["device"], "cpu");
    assert_eq!(report["data"]["runtime"]["device_backend"], "cpu");
    assert!(
        report["data"]["resources"]["process_peak_rss_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        report["data"]["resources"]["peak_allocated_bytes"],
        Value::Null
    );

    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let (auto_output, auto_report) = invoke(
        project.path(),
        &[
            "try",
            "--device",
            "auto",
            "--model",
            "device_fixture:build_model",
            "--batch",
            "device_fixture:build_batch",
            "--step",
            "device_fixture:train_step",
            "--steps",
            "1",
        ],
    );
    assert_success(&auto_output, &auto_report);
    assert_eq!(auto_report["data"]["runtime"]["device_requested"], "auto");
    assert_eq!(
        auto_report["data"]["runtime"]["device"],
        if gpu_count() > 0 { "cuda:0" } else { "cpu" }
    );
}

#[test]
fn unavailable_gpu_is_rejected_before_running_a_step() {
    if !has_torch() {
        eprintln!("skipped device integration: Python needs PyTorch");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    write_fixture(project.path());
    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--device",
            "cuda:999",
            "--model",
            "device_fixture:build_model",
            "--batch",
            "device_fixture:build_batch",
            "--step",
            "device_fixture:train_step",
            "--steps",
            "1",
        ],
    );
    assert!(!output.status.success());
    assert_eq!(report["status"], "error");
    assert_eq!(report["diagnostics"][0]["code"], "DEVICE_UNAVAILABLE");
}

#[test]
fn a_gpu_selection_is_required_when_multiple_devices_are_visible() {
    if !has_torch() || gpu_count() < 2 {
        eprintln!("skipped GPU ambiguity check: fewer than two CUDA/ROCm devices are visible");
        return;
    }
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let project = tempfile::tempdir().unwrap();
    write_fixture(project.path());
    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--device",
            "cuda",
            "--model",
            "device_fixture:build_model",
            "--batch",
            "device_fixture:build_batch",
            "--step",
            "device_fixture:train_step",
            "--steps",
            "1",
        ],
    );
    assert!(!output.status.success());
    assert_eq!(report["diagnostics"][0]["code"], "DEVICE_AMBIGUOUS");
}

#[test]
fn gpu_out_of_memory_reports_device_phase_and_measured_resources() {
    if !has_torch() || gpu_count() == 0 {
        eprintln!("skipped GPU OOM diagnostic: no CUDA/ROCm device is visible");
        return;
    }
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let project = tempfile::tempdir().unwrap();
    write_fixture(project.path());
    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--device",
            "cuda:0",
            "--model",
            "device_fixture:build_model",
            "--batch",
            "device_fixture:build_batch",
            "--step",
            "device_fixture:oom_step",
            "--steps",
            "1",
        ],
    );
    assert!(!output.status.success());
    let diagnostic = &report["diagnostics"][0];
    assert_eq!(diagnostic["code"], "TRAINING_OUT_OF_MEMORY");
    assert!(diagnostic["message"].as_str().unwrap().contains("cuda:0"));
    assert!(
        diagnostic["hint"]
            .as_str()
            .unwrap()
            .contains("最近测得可用 GPU 内存")
    );
    assert_eq!(report["data"]["device"], "cuda:0");
    assert_eq!(
        report["data"]["phase"],
        "training step (forward/backward/update)"
    );
    assert!(report["data"]["resources"]["total_bytes"].as_u64().unwrap() > 0);
}

#[test]
fn single_gpu_train_checkpoint_and_resume_use_the_selected_device() {
    if !has_torch() {
        eprintln!("skipped device integration: Python needs PyTorch");
        return;
    }
    if gpu_count() == 0 {
        eprintln!("skipped GPU acceptance: no CUDA/ROCm device is visible");
        return;
    }
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let project = tempfile::tempdir().unwrap();
    write_fixture(project.path());
    let (output, first) = invoke(
        project.path(),
        &[
            "train",
            "--device",
            "auto",
            "--model",
            "device_fixture:build_model",
            "--batch",
            "device_fixture:build_batch",
            "--step",
            "device_fixture:train_step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    );
    assert_success(&output, &first);
    assert_eq!(first["data"]["runtime"]["device"], "cuda:0");
    assert_eq!(first["data"]["runtime"]["device_requested"], "auto");
    assert!(matches!(
        first["data"]["runtime"]["device_backend"].as_str(),
        Some("cuda" | "rocm")
    ));
    assert!(
        first["data"]["resources"]["peak_allocated_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(first["data"]["checkpoint_saved"].as_bool().unwrap());

    let run_id = first["data"]["run_id"].as_str().unwrap();
    let (resume_output, resumed) = invoke(
        project.path(),
        &[
            "train",
            "--resume",
            run_id,
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    );
    assert_success(&resume_output, &resumed);
    assert_eq!(resumed["data"]["runtime"]["device"], "cuda:0");
    assert_eq!(resumed["data"]["resumed_from_steps"], 1);
    assert_eq!(resumed["data"]["completed_steps"], 2);

    let (mismatch_output, mismatch) = invoke(
        project.path(),
        &[
            "train",
            "--resume",
            run_id,
            "--additional-steps",
            "1",
            "--device",
            "cpu",
        ],
    );
    assert!(!mismatch_output.status.success());
    assert_eq!(
        mismatch["diagnostics"][0]["code"],
        "RESUME_DEVICE_INCOMPATIBLE"
    );
}

#[test]
fn a_user_owned_loop_can_move_nested_batches_through_its_context() {
    if !has_torch() || gpu_count() == 0 {
        eprintln!("skipped external-loop GPU coverage: no CUDA/ROCm device is visible");
        return;
    }
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let project = tempfile::tempdir().unwrap();
    write_fixture(project.path());
    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--device",
            "cuda:0",
            "--training-mode",
            "external-loop",
            "--loop",
            "device_fixture:run_loop",
            "--model",
            "device_fixture:build_model",
            "--steps",
            "1",
        ],
    );
    assert_success(&output, &report);
    assert_eq!(report["data"]["runtime"]["device"], "cuda:0");
    assert_eq!(report["data"]["completed_steps"], 1);
}

#[test]
fn manual_step_moves_nested_batches_and_updates_on_one_gpu() {
    if !has_torch() || gpu_count() == 0 {
        eprintln!("skipped manual-step GPU coverage: no CUDA/ROCm device is visible");
        return;
    }
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let project = tempfile::tempdir().unwrap();
    write_fixture(project.path());
    let (output, report) = invoke(
        project.path(),
        &[
            "try",
            "--device",
            "cuda:0",
            "--training-mode",
            "manual-step",
            "--model",
            "device_fixture:build_model",
            "--batch",
            "device_fixture:build_batch",
            "--step",
            "device_fixture:manual_step",
            "--steps",
            "1",
        ],
    );
    assert_success(&output, &report);
    assert_eq!(report["data"]["runtime"]["device"], "cuda:0");
    assert_eq!(report["data"]["optimizer_updates"], 1);
    assert!(
        report["data"]["resources"]["peak_allocated_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
}
