use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Output},
    sync::OnceLock,
};
#[cfg(unix)]
use std::{io::Read, os::fd::FromRawFd, process::Stdio};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    static PYTORCH_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *PYTORCH_AVAILABLE.get_or_init(|| {
        Command::new(python())
            .args(["-c", "import torch"])
            .status()
            .is_ok_and(|status| status.success())
    })
}

fn fixture(root: &Path) {
    fs::write(
        root.join("fixture.py"),
        r#"
from collections import OrderedDict
import torch
from torch import nn

class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, left, right):
        return {"left": self.shared(left), "right": self.shared(right)}

def build_shared():
    print("factory output belongs on stderr, not the JSON protocol")
    return SharedModel()

def shared_batch():
    return {"args": [torch.randn(2, 4), torch.randn(3, 4)], "kwargs": {}}

def keyword_batch():
    return {"left": torch.randn(2, 4), "right": torch.randn(3, 4)}

class AssemblyModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, values):
        return self.head(self.encoder(values))

def build_assembly_model(encoder, head):
    return AssemblyModel(encoder, head)

def build_assembly_encoder():
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())

def build_assembly_head():
    return nn.Linear(4, 2)

def build_param_activation(negative_slope=0.01):
    return nn.LeakyReLU(negative_slope=negative_slope)

def build_assembly_unused_model(encoder):
    return SharedModel()

def assembly_batch():
    return torch.randn(2, 4)

def assembly_training_step(model, batch, context):
    outputs = model(batch)
    targets = torch.tensor([0, 1])
    return {"loss": nn.functional.cross_entropy(outputs, targets)}

assembly_batch.MODEL_MAVEN_STATELESS = True
assembly_training_step.MODEL_MAVEN_STATELESS = True

def slow_batch():
    import os
    import time
    marker = os.environ.get("MODEL_MAVEN_TIMEOUT_MARKER")
    if marker:
        with open(marker, "w", encoding="utf-8") as output:
            output.write("started")
    time.sleep(20)
    return {"args": [torch.randn(2, 4), torch.randn(3, 4)], "kwargs": {}}

class EditableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ("proj", nn.Linear(8, 8)),
            ("relu", nn.ReLU()),
        ]))
        self.head = nn.Linear(8, 4)

    def forward(self, values):
        return self.head(self.encoder(values))

def build_editable():
    return EditableModel()

def editable_batch():
    return torch.randn(2, 8)

class CustomSequential(nn.Sequential):
    def forward(self, values):
        return super().forward(values)

class CustomSequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CustomSequential(OrderedDict([
            ("proj", nn.Linear(8, 8)),
            ("relu", nn.ReLU()),
        ]))
        self.head = nn.Linear(8, 4)

    def forward(self, values):
        return self.head(self.encoder(values))

def build_custom_sequential():
    return CustomSequentialModel()

def custom_sequential_batch():
    return torch.randn(2, 8)

class CustomModuleList(nn.ModuleList):
    pass

class CustomModuleListModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CustomModuleList([nn.Linear(8, 8), nn.ReLU()])

    def forward(self, values):
        for layer in self.encoder:
            values = layer(values)
        return values

def build_custom_module_list():
    return CustomModuleListModel()

def custom_module_list_batch():
    return torch.randn(2, 8)

class CustomModuleDict(nn.ModuleDict):
    pass

class CustomModuleDictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CustomModuleDict(OrderedDict([
            ("proj", nn.Linear(8, 8)),
            ("relu", nn.ReLU()),
        ]))

    def forward(self, values):
        for layer in self.encoder.values():
            values = layer(values)
        return values

def build_custom_module_dict():
    return CustomModuleDictModel()

def custom_module_dict_batch():
    return torch.randn(2, 8)

class ContractBreakModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_encoder = nn.Linear(8, 48)
        self.fusion_projection = nn.Linear(32, 16)

    def forward(self, frames):
        return self.fusion_projection(self.video_encoder(frames))

def build_contract_break():
    return ContractBreakModel()

def incompatible_batch():
    return torch.randn(2, 8)

def small_batch():
    return torch.randn(2, 4)

class TruncatedContractBreakModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.fusion_projection = nn.Linear(3, 2)

    def forward(self, frames):
        for _ in range(260):
            frames = self.encoder(frames)
        return self.fusion_projection(frames)

def build_truncated_contract_break():
    return TruncatedContractBreakModel()

def truncated_incompatible_batch():
    return torch.randn(2, 4)

def editable_training_step(model, batch, context):
    outputs = model(batch)
    targets = torch.tensor([0, 1])
    return {"loss": nn.functional.cross_entropy(outputs, targets)}

editable_batch.MODEL_MAVEN_STATELESS = True
editable_training_step.MODEL_MAVEN_STATELESS = True

def build_custom_activation():
    return nn.SiLU()

class FlexibleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict(OrderedDict([
            ("proj", nn.Linear(8, 8)),
            ("relu", nn.ReLU()),
        ]))

    def forward(self, values):
        for layer in self.layers.values():
            values = layer(values)
        return values

class FlexibleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FlexibleEncoder()
        self.head = nn.Linear(8, 4)

    def forward(self, values):
        return self.head(self.encoder(values))

def build_flexible():
    return FlexibleModel()

MODEL_MAVEN_LAYER_EDIT_API = 1

def add_custom_activation(model, operation):
    name = operation["payload"]["name"]
    model.encoder.layers[name] = nn.SiLU()
    return {"added": "model.encoder.layers." + name, "type": "SiLU"}

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, values):
        return self.proj(values)

def build_classifier():
    return Classifier()

def classification_batch():
    return {
        "args": [torch.randn(2, 4)],
        "kwargs": {},
        "loss_inputs": {"targets": torch.tensor([0, 2])},
    }

def classification_loss(outputs, batch):
    return nn.functional.cross_entropy(outputs, batch["loss_inputs"]["targets"])

def classification_training_step(model, batch, context, label_smoothing=0.0, api_token=None):
    outputs = model(*batch["args"], **batch["kwargs"])
    targets = batch["loss_inputs"]["targets"]
    loss = nn.functional.cross_entropy(outputs, targets, label_smoothing=label_smoothing)
    context.log("accuracy", (outputs.detach().argmax(dim=-1) == targets).float().mean())
    return {"loss": loss, "losses": {"classification": loss.detach()}, "outputs": outputs}

def failing_training_step(model, batch, context):
    if context.step == 2:
        raise ValueError("intentional step failure")
    outputs = model(*batch["args"], **batch["kwargs"])
    loss = nn.functional.cross_entropy(outputs, batch["loss_inputs"]["targets"])
    return {"loss": loss, "outputs": outputs}

def nonfinite_loss(outputs, batch):
    return torch.tensor(float("nan"), requires_grad=True)
"#,
    )
    .unwrap();
}

fn copy_complex_fixture(root: &Path) {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/complex_acceptance");
    let destination = root.join("complex_acceptance");
    fs::create_dir_all(&destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            fs::copy(entry.path(), destination.join(entry.file_name())).unwrap();
        }
    }
}

fn invoke(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--verbose", "--python", &python()])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap()
}

#[cfg(unix)]
fn run_workbench_pty(root: &Path, input: &[u8]) -> String {
    let (mut master_fd, mut slave_fd) = (-1, -1);
    let open_result = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(open_result, 0, "openpty failed");
    let slave = unsafe { fs::File::from_raw_fd(slave_fd) };
    let stdout = slave.try_clone().unwrap();
    let stderr = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python()])
        .current_dir(root)
        .stdin(Stdio::from(slave))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap();
    let mut terminal = unsafe { fs::File::from_raw_fd(master_fd) };
    let mut scripted_input = Vec::new();
    if !root.join(".model-maven/project.json").is_file() {
        scripted_input.extend_from_slice(b"5\n");
    }
    scripted_input.extend_from_slice(input);
    terminal.write_all(&scripted_input).unwrap();
    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut poll_fd = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut poll_fd, 1, 60_000) };
        if ready <= 0 {
            let _ = child.kill();
            panic!(
                "workbench interaction timed out:\n{}",
                String::from_utf8_lossy(&transcript)
            );
        }
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read interactive terminal: {error}"),
        }
    }
    let status = child.wait().unwrap();
    assert!(status.success(), "interactive CLI exited with {status}");
    String::from_utf8_lossy(&transcript).into_owned()
}

#[cfg(unix)]
fn run_command_pty(root: &Path, args: &[&str], input: &[u8]) -> String {
    let (mut master_fd, mut slave_fd) = (-1, -1);
    let open_result = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(open_result, 0, "openpty failed");
    let slave = unsafe { fs::File::from_raw_fd(slave_fd) };
    let stdout = slave.try_clone().unwrap();
    let stderr = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python()])
        .args(args)
        .current_dir(root)
        .stdin(Stdio::from(slave))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap();
    let mut terminal = unsafe { fs::File::from_raw_fd(master_fd) };
    terminal.write_all(input).unwrap();
    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut poll_fd = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut poll_fd, 1, 60_000) };
        if ready <= 0 {
            let _ = child.kill();
            panic!(
                "command interaction timed out:\n{}",
                String::from_utf8_lossy(&transcript)
            );
        }
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read interactive terminal: {error}"),
        }
    }
    let status = child.wait().unwrap();
    assert!(
        status.success(),
        "interactive command exited with {status}:\n{}",
        String::from_utf8_lossy(&transcript)
    );
    String::from_utf8_lossy(&transcript).into_owned()
}

fn report(output: &Output) -> Value {
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_slice(stdout.trim().as_bytes()).unwrap_or_else(|error| {
        panic!(
            "expected one JSON report, got {stdout:?}: {error}; status={:?}; stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[cfg(unix)]
#[test]
fn workbench_environment_action_renders_human_summary() {
    let project = tempfile::tempdir().unwrap();
    assert_success(&invoke(
        project.path(),
        &["init", "--model", "fixture:build_model"],
    ));

    let transcript = run_workbench_pty(project.path(), b"5\n1\nback\nq\n");
    assert!(transcript.contains("Python 环境"), "{transcript}");
    assert!(transcript.contains("解释器："), "{transcript}");
    assert!(transcript.contains("状态："), "{transcript}");
}

#[cfg(unix)]
#[test]
fn workbench_environment_setup_rejects_empty_requirements_without_creating_environment() {
    let project = tempfile::tempdir().unwrap();
    fs::write(project.path().join("requirements.txt"), "").unwrap();
    assert_success(&invoke(
        project.path(),
        &["init", "--model", "fixture:build_model"],
    ));

    let transcript = run_workbench_pty(project.path(), b"5\n2\n1\n\nback\n8\n");
    assert!(
        transcript.contains("ENVIRONMENT_RESOLUTION_EMPTY"),
        "{transcript}"
    );
    assert!(!project.path().join(".venv").exists());
}

#[cfg(unix)]
#[test]
fn supervision_guide_manually_marks_a_real_block_adds_rule_and_previews_it() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let mut source = fs::read_to_string(project.path().join("fixture.py")).unwrap();
    source.push_str(
        "\nclass ManualMarkedBlock(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.projection = nn.Linear(4, 4)\n    def forward(self, values):\n        return self.projection(values)\nclass ManualMarkedModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.embedding = ManualMarkedBlock()\n    def forward(self, values, targets=None):\n        return self.embedding(values)\ndef build_manual_marked():\n    return ManualMarkedModel()\ndef manual_marked_batch():\n    return {\"args\": [torch.ones(2, 4)], \"kwargs\": {}}\ndef manual_marked_target_batch():\n    return {\"args\":[torch.ones(2, 4)],\"kwargs\":{\"targets\":torch.tensor([0,1])}}\n",
    );
    fs::write(project.path().join("fixture.py"), source).unwrap();
    assert_success(&invoke(
        project.path(),
        &[
            "init",
            "--model",
            "fixture:build_manual_marked",
            "--batch",
            "fixture:manual_marked_batch",
        ],
    ));
    let transcript = run_command_pty(
        project.path(),
        &["model", "supervise", "guide"],
        b"3\nembedding\nLearned embedding for downstream model heads.\n3\n-1\n1\n0.05\n",
    );
    assert!(transcript.contains("请选择要手动赋予语义"), "{transcript}");
    assert!(transcript.contains("model.embedding"), "{transcript}");
    assert!(
        transcript.contains("含义：Learned embedding for downstream model heads."),
        "{transcript}"
    );
    assert!(
        transcript.contains("第 4 步 · 对当前代表批次做只读预览"),
        "{transcript}"
    );
    assert!(
        transcript.contains("覆盖状态：observed-runtime"),
        "{transcript}"
    );
    assert!(
        transcript.contains("实现来源 fixture.py · SHA-256"),
        "{transcript}"
    );
    assert!(transcript.contains("加权 penalty 预览值"), "{transcript}");
    assert!(
        transcript.contains("只有显式添加 --apply-supervision"),
        "{transcript}"
    );

    let marks: Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/semantic-marks.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(marks["marks"][0]["role"], "embedding");
    assert_eq!(
        marks["marks"][0]["meaning"],
        "Learned embedding for downstream model heads."
    );
    assert_eq!(marks["marks"][0]["module_path"], "model.embedding");
    assert_eq!(marks["marks"][0]["rules"][0]["mode"], "penalty");
    assert_eq!(marks["marks"][0]["rules"][0]["kind"], "range");
    assert_eq!(marks["marks"][0]["rules"][0]["min"], -1.0);
    assert_eq!(marks["marks"][0]["rules"][0]["max"], 1.0);
    assert_eq!(marks["marks"][0]["rules"][0]["weight"], 0.05);

    let target_transcript = run_command_pty(
        project.path(),
        &[
            "model",
            "supervise",
            "guide",
            "--batch",
            "fixture:manual_marked_target_batch",
        ],
        b"3\nlogits\nScores aligned with manually selected targets.\n2\n3\nbatch.kwargs.targets\n\n",
    );
    assert!(
        target_transcript.contains("batch-alignment"),
        "{target_transcript}"
    );
    assert!(
        target_transcript.contains("目标 batch.kwargs.targets"),
        "{target_transcript}"
    );
    let marks: Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/semantic-marks.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(marks["marks"][1]["rules"][0]["kind"], "batch-alignment");
    assert_eq!(
        marks["marks"][1]["rules"][0]["target_path"],
        "batch.kwargs.targets"
    );
    assert_eq!(marks["marks"][1]["rules"][0]["axis"], 0);
}

#[test]
fn supervision_guide_json_mode_returns_actionable_noninteractive_diagnostic() {
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let output = invoke(
        project.path(),
        &[
            "model",
            "supervise",
            "guide",
            "--model",
            "fixture:build_shared",
            "--batch",
            "fixture:shared_batch",
        ],
    );
    let report = report(&output);
    assert!(!output.status.success(), "{report:#}");
    assert_eq!(report["diagnostics"][0]["code"], "GUIDED_FLOW_REQUIRES_TTY");
    assert!(
        report["diagnostics"][0]["hint"]
            .as_str()
            .unwrap()
            .contains("model mark")
    );
}

fn assert_success(output: &Output) -> Value {
    let value = report(output);
    assert!(
        output.status.success(),
        "unexpected failure: {value:#}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(value["status"], "success", "{value:#}");
    value
}

#[test]
fn run_comparison_uses_persisted_evidence_and_latest_means_newest_timestamp() {
    let project = tempfile::tempdir().unwrap();
    let runs = project.path().join(".model-maven/runs");
    for (id, status, started_at_ms, model, loss) in [
        ("run-old-complete", "completed", 10, "fixture:old", 0.8),
        ("run-new-failed", "failed", 20, "fixture:new", 0.3),
    ] {
        let directory = runs.join(id);
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("run.json"),
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "id": id,
                "name": null,
                "kept": false,
                "operation": "training_probe",
                "status": status,
                "started_at_ms": started_at_ms,
                "finished_at_ms": started_at_ms + 1,
                "inputs_file": "inputs.json"
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            directory.join("inputs.json"),
            serde_json::to_vec(&serde_json::json!({
                "operation": "training_probe",
                "project_root": project.path(),
                "model": model,
                "model_params": {},
                "model_injections": {},
                "batch": "fixture:batch",
                "batch_binding": {"input":model},
                "data_fingerprint": "dataset-v1",
                "steps": 10,
                "training_params": {"epochs": 1},
                "runtime": {"python_executable":"python3"}
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(directory.join("events.jsonl"), "").unwrap();
        fs::write(
            directory.join("report.json"),
            serde_json::to_vec(&serde_json::json!({
                "protocol_version": 1,
                "data": {
                    "source_manifest": {"fixture.py": if model == "fixture:old" { "old-source" } else { "new-source" }},
                    "mean_loss": loss
                }
            }))
            .unwrap(),
        )
        .unwrap();
    }

    let comparison = assert_success(&invoke(
        project.path(),
        &["runs", "compare", "run-old-complete", "run-new-failed"],
    ));
    let data = &comparison["data"];
    assert_eq!(data["read_only"], true);
    assert_eq!(data["causal_attribution"], false);
    assert_eq!(data["protocol"]["status"], "same");
    assert!(
        data["dimensions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["name"] == "source_code" && item["status"] == "changed" })
    );
    assert_eq!(
        data["source_file_changes"]["changed_files"][0],
        "fixture.py"
    );
    assert!(
        data["dimensions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["name"] == "inputs" && item["status"] == "changed" })
    );
    assert!(
        data["dimensions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["name"] == "data" && item["status"] == "changed" })
    );
    assert!(
        data["dimensions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["name"] == "metrics" && item["status"] == "changed" })
    );
    assert!(
        data["dimensions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["name"] == "weights" && item["status"] == "unknown" })
    );

    let human = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "runs",
            "compare",
            "run-old-complete",
            "run-new-failed",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human.status.success());
    let human = String::from_utf8_lossy(&human.stdout);
    assert!(human.contains("运行对照（只读）"), "{human}");
    assert!(human.contains("数据与采样：有差异"), "{human}");
    assert!(human.contains("权重：信息不足"), "{human}");

    let latest = assert_success(&invoke(project.path(), &["runs", "show"]));
    assert_eq!(latest["data"]["record"]["id"], "run-new-failed");

    #[cfg(unix)]
    {
        let (mut master_fd, mut slave_fd) = (-1, -1);
        let open_result = unsafe {
            libc::openpty(
                &mut master_fd,
                &mut slave_fd,
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(open_result, 0, "openpty failed");
        let slave = unsafe { fs::File::from_raw_fd(slave_fd) };
        let stdout = slave.try_clone().unwrap();
        let stderr = slave.try_clone().unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args(["--python", &python()])
            .current_dir(project.path())
            .stdin(Stdio::from(slave))
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .unwrap();
        let mut terminal = unsafe { fs::File::from_raw_fd(master_fd) };
        terminal.write_all(b"5\n3\n10\n\n\n\nback\n0\n").unwrap();
        let mut transcript = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let mut poll_fd = libc::pollfd {
                fd: master_fd,
                events: libc::POLLIN | libc::POLLHUP,
                revents: 0,
            };
            let ready = unsafe { libc::poll(&mut poll_fd, 1, 15_000) };
            if ready <= 0 {
                let _ = child.kill();
                panic!("interactive comparison timed out");
            }
            match terminal.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => transcript.extend_from_slice(&buffer[..count]),
                Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
                Err(error) => panic!("cannot read interactive terminal: {error}"),
            }
        }
        let status = child.wait().unwrap();
        assert!(status.success(), "interactive CLI exited with {status}");
        let transcript = String::from_utf8_lossy(&transcript);
        assert!(transcript.contains("对照两次试跑"), "{transcript}");
        assert!(
            transcript.contains("运行对照 · run-new-failed ↔ run-old-complete"),
            "{transcript}"
        );
        assert!(transcript.contains("数据与采样 · 有差异"), "{transcript}");
        assert!(transcript.contains("源码修改：fixture.py"), "{transcript}");
        assert!(
            transcript.contains("按对应试跑的原始设置再试"),
            "{transcript}"
        );
    }
}

#[cfg(unix)]
#[test]
fn workbench_run_browse_preserves_filters_and_reports_unknown_progress() {
    let project = tempfile::tempdir().unwrap();
    let runs = project.path().join(".model-maven/runs");
    for (id, name, status, started_at_ms) in [
        ("run-failed", "Failed experiment", "failed", 2_000),
        ("run-lost", "Lost supervisor", "running", 1_000),
    ] {
        let directory = runs.join(id);
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("run.json"),
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "id": id,
                "name": name,
                "kept": false,
                "operation": "train",
                "status": status,
                "started_at_ms": started_at_ms,
                "finished_at_ms": null,
                "requested_steps": 12,
                "summary": {
                    "completed_steps": 4,
                    "requested_steps": 12,
                    "mean_loss": 0.125,
                    "optimizer_updates": 4
                },
                "inputs_file": "inputs.json"
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            directory.join("inputs.json"),
            serde_json::to_vec(&serde_json::json!({
                "operation": "train",
                "project_root": project.path(),
                "model": "fixture:build_model",
                "model_params": {},
                "model_injections": {},
                "batch": "fixture:batch",
                "batch_binding": {"input": "fixture:batch"},
                "steps": 12,
                "runtime": {"python_executable": "python3"}
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(directory.join("events.jsonl"), "").unwrap();
        fs::write(
            directory.join("report.json"),
            serde_json::to_vec(&serde_json::json!({
                "protocol_version": 1,
                "diagnostics": [{
                    "code": "RUN_FIXTURE_DIAGNOSTIC",
                    "message": "fixture diagnostic",
                    "hint": "inspect this run"
                }],
                "data": {}
            }))
            .unwrap(),
        )
        .unwrap();
        if id == "run-lost" {
            fs::write(
                directory.join("progress.jsonl"),
                serde_json::to_vec(&serde_json::json!({
                    "stage": "training",
                    "step": 4,
                    "total": 12,
                    "mean_loss": 0.125,
                    "at_ms": 1_000_000_000_000_u64
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }

    let (mut master_fd, mut slave_fd) = (-1, -1);
    let open_result = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(open_result, 0, "openpty failed");
    let slave = unsafe { fs::File::from_raw_fd(slave_fd) };
    let stdout = slave.try_clone().unwrap();
    let stderr = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python()])
        .current_dir(project.path())
        .stdin(Stdio::from(slave))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap();
    let mut terminal = unsafe { fs::File::from_raw_fd(master_fd) };
    terminal
        .write_all(
            b"5\n3\n5\nstatus unknown\n1\nprogress\nback\nback\nclear\nstatus failed\n1\nback\nback\nback\nquit\n",
        )
        .unwrap();
    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut poll_fd = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut poll_fd, 1, 15_000) };
        if ready <= 0 {
            let _ = child.kill();
            panic!("interactive run browser timed out");
        }
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read interactive terminal: {error}"),
        }
    }
    let status = child.wait().unwrap();
    assert!(status.success(), "interactive CLI exited with {status}");
    let transcript = String::from_utf8_lossy(&transcript);
    assert!(transcript.contains("ID：run-lost"), "{transcript}");
    assert!(transcript.contains("状态：状态未知"), "{transcript}");
    assert!(transcript.contains("后台运行 · 状态未知"), "{transcript}");
    assert!(transcript.contains("阶段：训练"), "{transcript}");
    assert!(
        transcript.contains("最近进度：4/12 步 · Loss 0.125000"),
        "{transcript}"
    );
    assert!(transcript.contains("最近更新："), "{transcript}");
    assert!(
        transcript.contains("该运行当前不是已确认活动状态"),
        "{transcript}"
    );
    assert!(transcript.contains("运行详情 · 失败待处理"), "{transcript}");
    assert!(transcript.contains("fixture diagnostic"), "{transcript}");
    assert!(transcript.contains("ID：run-failed"), "{transcript}");
    assert!(transcript.contains("状态：失败待处理"), "{transcript}");
}

#[cfg(unix)]
#[test]
fn workbench_keeps_invalid_variant_draft_until_grouped_dimension_repair_passes() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());

    let transcript = run_workbench_pty(
        project.path(),
        b"2\n14\nfixture:build_editable\nfixture:editable_batch\nadd\nmodel.encoder.proj\n\n\n16\npreview\nadd\nmodel.head\n\n16\n\npreview\nyes\n8\n9\n16\n8\n",
    );
    assert!(transcript.contains("前向失败"), "{transcript}");
    assert!(
        transcript.contains("草稿未提交，可继续添加修复"),
        "{transcript}"
    );
    assert!(
        transcript.contains("候选验证：构造 passed · 前向 passed"),
        "{transcript}"
    );
    assert!(transcript.contains("已保存："), "{transcript}");
    assert!(transcript.contains("撤销"), "{transcript}");
    assert!(transcript.contains("重做"), "{transcript}");
    assert!(
        transcript.contains("Layer 参数更新：model.encoder.proj"),
        "{transcript}"
    );
    assert!(
        transcript.contains("Layer 参数更新：model.head"),
        "{transcript}"
    );

    let probe = assert_success(&invoke(
        project.path(),
        &[
            "--variant",
            "working",
            "model",
            "probe",
            "--model",
            "fixture:build_editable",
            "--batch",
            "fixture:editable_batch",
        ],
    ));
    assert_eq!(probe["status"], "success", "{probe:#}");
    assert_eq!(probe["data"]["outputs"]["shape"], serde_json::json!([2, 4]));
}

#[cfg(unix)]
#[test]
fn workbench_layer_crud_keeps_model_and_batch_context_and_reloads_variant() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());

    let transcript = run_workbench_pty(
        project.path(),
        b"2\n1\nfixture:build_editable\nfixture:editable_batch\nmodel.encoder\nrelu\ndropout\nDropout\n\n\n2\nmodel.encoder.dropout\n\n0.1\n4\nmodel.encoder.dropout\n\n16\n8\n",
    );
    assert!(
        transcript.contains("Layer 新增：model.encoder.dropout · Dropout"),
        "{transcript}"
    );
    assert!(
        transcript.contains("Layer 参数更新：model.encoder.dropout"),
        "{transcript}"
    );
    assert!(
        transcript.contains("Layer 删除：model.encoder.dropout"),
        "{transcript}"
    );
    assert_eq!(
        transcript.matches("模型入口（如 examples.models").count(),
        1,
        "{transcript}"
    );
    assert_eq!(
        transcript
            .matches("可选：批次入口，用于保存前验证候选前向")
            .count(),
        1,
        "{transcript}"
    );

    let reloaded = assert_success(&invoke(
        project.path(),
        &[
            "--variant",
            "working",
            "layer",
            "list",
            "--model",
            "fixture:build_editable",
            "--batch",
            "fixture:editable_batch",
        ],
    ));
    let rows = reloaded["data"]["modules"].as_array().unwrap();
    assert!(
        rows.iter().any(|row| row["path"] == "model.encoder.relu"),
        "{reloaded:#}"
    );
    assert!(
        !rows
            .iter()
            .any(|row| row["path"] == "model.encoder.dropout"),
        "removed dropout must remain absent after reloading the persistent variant: {reloaded:#}"
    );
    assert!(
        project
            .path()
            .join(".model-maven/variants/working.json")
            .is_file()
    );
    assert!(project.path().join("fixture.py").is_file());
}

#[cfg(unix)]
#[test]
fn workbench_guides_explicit_shared_layer_split_and_persists_variant() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let mut source = fs::read_to_string(project.path().join("fixture.py")).unwrap();
    source.push_str(
        "\nclass AliasedLayerModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        shared = nn.Linear(4, 4)\n        self.left = shared\n        self.right = shared\n    def forward(self, values):\n        return self.right(self.left(values))\ndef build_aliases():\n    return AliasedLayerModel()\ndef alias_batch():\n    return torch.ones(2, 4)\n",
    );
    fs::write(project.path().join("fixture.py"), source).unwrap();
    assert_success(&invoke(
        project.path(),
        &[
            "init",
            "--model",
            "fixture:build_aliases",
            "--batch",
            "fixture:alias_batch",
        ],
    ));

    let transcript = run_workbench_pty(
        project.path(),
        b"2\n15\nmodel.left\nclone\nmodel.left\nyes\nback\n8\n",
    );
    assert!(transcript.contains("调整 Layer 共享关系"), "{transcript}");
    assert!(
        transcript.contains("已观察到的共享路径：model.left、model.right"),
        "{transcript}"
    );
    assert!(
        transcript.contains("共享关系拆分：model.left"),
        "{transcript}"
    );
    assert!(
        transcript.contains("确认应用以上共享关系变更到当前 variant"),
        "{transcript}"
    );

    let variant: Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/variants/working.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        variant["edits"][0]["operations"][0]["mode"],
        "clone-selected"
    );
    assert_eq!(variant["edits"][0]["operations"][0]["path"], "model.left");
}

#[cfg(unix)]
#[test]
fn workbench_managed_and_manual_step_ask_for_their_required_entries() {
    let managed_project = tempfile::tempdir().unwrap();
    let managed = run_workbench_pty(
        managed_project.path(),
        b"3\n3\n1\nfixture:build_shared\n\n11\n8\n",
    );
    assert!(managed.contains("managed-step"), "{managed}");
    assert!(
        managed.contains("模型入口（如 examples.models"),
        "{managed}"
    );
    assert!(
        managed.contains("批次入口（如 examples.models"),
        "{managed}"
    );
    assert!(!managed.contains("完整循环入口"), "{managed}");

    let manual_project = tempfile::tempdir().unwrap();
    let manual = run_workbench_pty(
        manual_project.path(),
        b"3\n3\n2\nfixture:build_shared\nfixture:shared_batch\n\n11\n8\n",
    );
    assert!(manual.contains("manual-step"), "{manual}");
    assert!(manual.contains("模型入口（如 examples.models"), "{manual}");
    assert!(manual.contains("批次入口（如 examples.models"), "{manual}");
    assert!(
        manual.contains("训练步骤入口（model, batch, context -> dict）"),
        "{manual}"
    );
    assert!(!manual.contains("完整循环入口"), "{manual}");
}

#[cfg(unix)]
#[test]
fn workbench_managed_step_plan_runs_with_optimizer_defaults() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());

    let transcript = run_workbench_pty(
        project.path(),
        b"3\n3\n1\nfixture:build_classifier\nfixture:classification_batch\nfixture:classification_training_step\n2\n1\n\n\n\n\nn\nstart\n11\n8\n",
    );
    assert!(transcript.contains("managed-step"), "{transcript}");
    assert!(transcript.contains("AdamW 学习率"), "{transcript}");
    assert!(transcript.contains("AdamW weight decay"), "{transcript}");
    assert!(transcript.contains("正式训练完成：2/2 步"), "{transcript}");
    assert!(transcript.contains("优化器更新 2 次"), "{transcript}");

    let runs = project.path().join(".model-maven/runs");
    let run_dir = fs::read_dir(runs)
        .unwrap()
        .map(Result::unwrap)
        .map(|entry| entry.path())
        .find(|path| path.join("run.json").is_file())
        .expect("confirmed managed-step should create a run record");
    let record: Value =
        serde_json::from_slice(&fs::read(run_dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(record["operation"], "train");
    assert_eq!(record["status"], "completed");
    assert_eq!(record["training_mode"], "managed-step");
    assert_eq!(record["requested_steps"], 2);
    assert!(run_dir.join("checkpoints/latest.json").is_file());
}

#[cfg(unix)]
#[test]
fn workbench_manual_step_plan_runs_a_real_complex_model() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    copy_complex_fixture(project.path());

    let transcript = run_workbench_pty(
        project.path(),
        b"3\n3\n2\ncomplex_acceptance.model:build_model\ncomplex_acceptance.data:changing_batch\ncomplex_acceptance.training:manual_multitask_step\n2\n1\n\n\n\n\nn\nstart\n11\n8\n",
    );
    assert!(transcript.contains("manual-step"), "{transcript}");
    assert!(
        transcript.contains("complex_acceptance.model:build_model"),
        "{transcript}"
    );
    assert!(transcript.contains("正式训练完成：2/2 步"), "{transcript}");
    assert!(transcript.contains("优化器更新 4 次"), "{transcript}");

    let runs = project.path().join(".model-maven/runs");
    let run_dir = fs::read_dir(runs)
        .unwrap()
        .map(Result::unwrap)
        .map(|entry| entry.path())
        .find(|path| path.join("run.json").is_file())
        .expect("confirmed manual-step should create a run record");
    let record: Value =
        serde_json::from_slice(&fs::read(run_dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(record["operation"], "train");
    assert_eq!(record["status"], "completed");
    assert_eq!(record["training_mode"], "manual-step");
    assert_eq!(record["requested_steps"], 2);
    assert!(run_dir.join("checkpoints/latest.json").is_file());
}

#[cfg(unix)]
#[test]
fn workbench_resume_lists_and_requires_explicit_failed_run_and_reset_choice() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let completed = assert_success(&invoke(
        project.path(),
        &[
            "train",
            "--model",
            "fixture:build_classifier",
            "--batch",
            "fixture:classification_batch",
            "--step",
            "fixture:classification_training_step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
            "--name",
            "older-success",
        ],
    ));
    let completed_id = completed["data"]["run_id"].as_str().unwrap();
    let failed = report(&invoke(
        project.path(),
        &[
            "train",
            "--model",
            "fixture:build_classifier",
            "--batch",
            "fixture:classification_batch",
            "--step",
            "fixture:failing_training_step",
            "--steps",
            "4",
            "--checkpoint-every",
            "1",
            "--name",
            "failed-newer",
        ],
    ));
    assert_eq!(failed["status"], "error", "{failed:#}");
    assert_eq!(failed["diagnostics"][0]["code"], "STEP_FAILED");
    let run_id = failed["data"]["run_id"].as_str().unwrap();
    assert_ne!(run_id, completed_id);
    let shown = assert_success(&invoke(project.path(), &["runs", "show", run_id]));
    assert_eq!(shown["data"]["record"]["status"], "failed");
    assert_eq!(shown["data"]["checkpoint"]["completed_steps"], 1);

    let transcript = run_workbench_pty(project.path(), b"3\n4\n1\n2\n1\ny\n\nn\nno\n11\n8\n");
    assert!(transcript.contains("失败待处理"), "{transcript}");
    assert!(transcript.contains("failed-newer"), "{transcript}");
    assert!(transcript.contains("older-success"), "{transcript}");
    assert!(transcript.contains("checkpoint 第 1 步"), "{transcript}");
    assert!(
        transcript.contains("输入可续训记录的序号或完整 ID"),
        "{transcript}"
    );
    assert!(transcript.contains(run_id), "{transcript}");
    assert!(transcript.contains("本次续训运行的父记录"), "{transcript}");
    assert!(transcript.contains("增加 2 步"), "{transcript}");
    assert!(transcript.contains("接受重置"), "{transcript}");
    assert!(
        transcript.contains("已取消；没有启动训练或写入运行记录"),
        "{transcript}"
    );

    let run_count = fs::read_dir(project.path().join(".model-maven/runs"))
        .unwrap()
        .map(Result::unwrap)
        .filter(|entry| entry.path().join("run.json").is_file())
        .count();
    assert_eq!(
        run_count, 2,
        "cancelling a resume plan must not create a child run"
    );
}

#[cfg(unix)]
#[test]
fn workbench_repeat_displays_the_selected_run_snapshot_before_start() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let source = assert_success(&invoke(
        project.path(),
        &[
            "try",
            "--model",
            "fixture:build_classifier",
            "--batch",
            "fixture:classification_batch",
            "--step",
            "fixture:classification_training_step",
            "--steps",
            "1",
        ],
    ));
    let source_id = source["data"]["run_id"].as_str().unwrap();

    let transcript = run_workbench_pty(
        project.path(),
        b"3\n5\n1\nrepeat\ny\nno\nback\nback\n11\n8\n",
    );
    assert!(transcript.contains(source_id), "{transcript}");
    assert!(
        transcript.contains("读取该运行保存的原始输入快照"),
        "{transcript}"
    );
    assert!(
        transcript.contains("本次显式覆盖：无；其余输入从来源记录继承"),
        "{transcript}"
    );
    assert!(
        transcript.contains("已取消；没有启动训练或写入运行记录"),
        "{transcript}"
    );

    let run_count = fs::read_dir(project.path().join(".model-maven/runs"))
        .unwrap()
        .map(Result::unwrap)
        .filter(|entry| entry.path().join("run.json").is_file())
        .count();
    assert_eq!(
        run_count, 1,
        "cancelling repeat must not create another run"
    );
}

#[cfg(unix)]
#[test]
fn workbench_external_loop_plan_does_not_require_model_batch_or_loss() {
    let project = tempfile::tempdir().unwrap();
    let (mut master_fd, mut slave_fd) = (-1, -1);
    let open_result = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(open_result, 0, "openpty failed");
    let slave = unsafe { fs::File::from_raw_fd(slave_fd) };
    let stdout = slave.try_clone().unwrap();
    let stderr = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python()])
        .current_dir(project.path())
        .stdin(Stdio::from(slave))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap();
    let mut terminal = unsafe { fs::File::from_raw_fd(master_fd) };
    terminal
        .write_all(b"5\n3\n3\n3\nfixture:run_loop\n\n\n\n\nn\nno\n11\n8\n")
        .unwrap();
    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut poll_fd = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut poll_fd, 1, 15_000) };
        if ready <= 0 {
            let _ = child.kill();
            panic!("external-loop plan interaction timed out");
        }
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read interactive terminal: {error}"),
        }
    }
    let status = child.wait().unwrap();
    assert!(status.success(), "interactive CLI exited with {status}");
    let transcript = String::from_utf8_lossy(&transcript);
    assert!(transcript.contains("选择模式"), "{transcript}");
    assert!(transcript.contains("完整循环入口"), "{transcript}");
    assert!(transcript.contains("完整循环自行管理数据"), "{transcript}");
    assert!(
        transcript.contains("不要求独立 batch 或 loss"),
        "{transcript}"
    );
    assert!(
        transcript.contains("完整循环：fixture:run_loop"),
        "{transcript}"
    );
    assert!(
        !transcript.contains("模型入口（如 examples.models"),
        "{transcript}"
    );
    assert!(
        !transcript.contains("批次入口（如 examples.models"),
        "{transcript}"
    );
    assert!(
        transcript.contains("已取消；没有启动训练或写入运行记录"),
        "{transcript}"
    );
}

#[cfg(unix)]
#[test]
fn workbench_confirms_and_runs_a_real_external_loop_without_model_or_batch() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("external_loop.py"),
        r#"
import torch

def run_loop(model, context):
    weight = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=0.1)
    while context.remaining_steps:
        optimizer.zero_grad(set_to_none=True)
        loss = weight.square()
        loss.backward()
        optimizer.step()
        if not context.report_step(loss.detach(), {"weight": weight.detach()}):
            break
    return {"finished": True}

run_loop.MODEL_MAVEN_STATELESS = True
"#,
    )
    .unwrap();

    let transcript = run_workbench_pty(
        project.path(),
        b"3\n3\n3\nexternal_loop:run_loop\n2\n1\n\n\nn\nstart\n11\n8\n",
    );
    assert!(transcript.contains("external-loop"), "{transcript}");
    assert!(
        transcript.contains("完整自有循环完成：2/2 步"),
        "{transcript}"
    );
    assert!(!transcript.contains("已取消；没有启动训练"), "{transcript}");

    let runs = project.path().join(".model-maven/runs");
    let run_dir = fs::read_dir(runs)
        .unwrap()
        .map(Result::unwrap)
        .map(|entry| entry.path())
        .find(|path| path.join("run.json").is_file())
        .expect("confirmed external-loop should create a run record");
    let record: Value =
        serde_json::from_slice(&fs::read(run_dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(record["operation"], "train");
    assert_eq!(record["status"], "completed");
    assert_eq!(record["training_mode"], "external-loop");
    assert_eq!(record["requested_steps"], 2);
    assert!(run_dir.join("checkpoints/latest.json").is_file());
}

#[test]
fn layer_cli_help_explains_container_editing_and_atomic_plans() {
    let project = tempfile::tempdir().unwrap();
    let output = invoke(project.path(), &["layer", "--help"]);
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    for expected in [
        "列出模型中的 Layer",
        "查看目标 Layer 的参数",
        "向 nn.Sequential、nn.ModuleList、nn.ModuleDict 或其子类添加 Layer",
        "把多个 Layer 操作作为一个候选计划",
    ] {
        assert!(help.contains(expected), "missing {expected:?} in:\n{help}");
    }

    let apply = invoke(project.path(), &["layer", "apply", "--help"]);
    assert!(apply.status.success());
    let apply_help = String::from_utf8_lossy(&apply.stdout);
    assert!(apply_help.contains("可重复；JSON 操作对象"));
    assert!(apply_help.contains("\"action\":\"set\""));

    let add = invoke(project.path(), &["layer", "add", "--help"]);
    assert!(add.status.success());
    let add_help = String::from_utf8_lossy(&add.stdout);
    assert!(add_help.contains("ModuleDict 分支名"));
    assert!(add_help.contains("ModuleList 用数字索引"));

    let remove = invoke(project.path(), &["layer", "remove", "--help"]);
    assert!(remove.status.success());
    let remove_help = String::from_utf8_lossy(&remove.stdout);
    assert!(remove_help.contains("possible values: remove, bypass"));

    let list = invoke(project.path(), &["layer", "list", "--help"]);
    assert!(list.status.success());
    let list_help = String::from_utf8_lossy(&list.stdout);
    assert!(list_help.contains("--search <KEYWORD>"));
    assert!(list_help.contains("共享别名"));
}

#[test]
fn layer_list_search_matches_paths_types_and_shared_aliases() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let mut source = fs::read_to_string(project.path().join("fixture.py")).unwrap();
    source.push_str("\nclass AliasedLayerModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        shared = nn.Linear(4, 4)\n        self.left = shared\n        self.right = shared\n\n    def forward(self, values):\n        return self.right(values)\n\ndef build_aliases():\n    return AliasedLayerModel()\n");
    fs::write(project.path().join("fixture.py"), source).unwrap();

    let paths = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "list",
            "--model",
            "fixture:build_editable",
            "--search",
            "PROJ",
        ],
    ));
    let path_rows = paths["data"]["modules"].as_array().unwrap();
    assert_eq!(path_rows.len(), 1);
    assert_eq!(path_rows[0]["path"], "model.encoder.proj");

    let combined = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "list",
            "model.encoder",
            "--model",
            "fixture:build_editable",
            "--type",
            "Linear",
            "--search",
            "PROJ",
        ],
    ));
    let combined_rows = combined["data"]["modules"].as_array().unwrap();
    assert_eq!(combined_rows.len(), 1);
    assert_eq!(combined_rows[0]["path"], "model.encoder.proj");

    let types = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "list",
            "--model",
            "fixture:build_editable",
            "--search",
            "linear",
        ],
    ));
    let type_rows = types["data"]["modules"].as_array().unwrap();
    assert_eq!(type_rows.len(), 2);
    assert!(type_rows.iter().all(|row| row["type"] == "Linear"));

    let aliases = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "list",
            "--model",
            "fixture:build_aliases",
            "--search",
            "RIGHT",
        ],
    ));
    let alias_rows = aliases["data"]["modules"].as_array().unwrap();
    assert_eq!(alias_rows.len(), 1);
    assert!(
        alias_rows[0]["aliases"]
            .as_array()
            .unwrap()
            .iter()
            .any(|alias| alias == "model.right")
    );
}

#[test]
fn project_init_saves_defaults_for_probe_without_overwriting_or_side_effects_on_preview() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let initialized = assert_success(&invoke(
        project.path(),
        &[
            "init",
            "--model",
            "fixture:build_shared",
            "--batch",
            "fixture:keyword_batch",
            "--batch-binding",
            "kwargs",
        ],
    ));
    assert_eq!(initialized["data"]["committed"], true);
    let config_revision = initialized["data"]["config_revision"].clone();
    assert!(config_revision.as_str().is_some());
    let config_path = project.path().join(".model-maven/project.json");
    let saved = fs::read(&config_path).unwrap();

    let inspected = assert_success(&invoke(project.path(), &["model", "inspect"]));
    assert_eq!(inspected["data"]["modules"].as_array().unwrap().len(), 2);
    assert_eq!(inspected["data"]["config_revision"], config_revision);
    assert_eq!(
        inspected["data"]["input_sources"]["model"]["kind"],
        "project_config"
    );
    assert_eq!(
        inspected["data"]["input_sources"]["model"]["revision"],
        config_revision
    );
    let probed = assert_success(&invoke(project.path(), &["model", "probe"]));
    assert_eq!(probed["data"]["calls_observed"], 3);
    assert_eq!(
        probed["data"]["inputs"]["items"]["kwargs"]["items"]["left"]["shape"],
        serde_json::json!([2, 4])
    );
    assert_eq!(
        probed["data"]["inputs"]["items"]["args"]["items"],
        serde_json::json!([])
    );
    assert_eq!(
        probed["data"]["outputs"]["items"]["left"]["shape"],
        serde_json::json!([2, 4])
    );

    let overridden = assert_success(&invoke(
        project.path(),
        &[
            "model",
            "probe",
            "--model",
            "fixture:build_classifier",
            "--batch",
            "fixture:classification_batch",
            "--batch-binding",
            "envelope",
        ],
    ));
    assert_eq!(
        overridden["data"]["outputs"]["shape"],
        serde_json::json!([2, 3])
    );
    assert_eq!(overridden["data"]["config_revision"], config_revision);
    assert_eq!(
        overridden["data"]["input_sources"]["model"]["kind"],
        "command_line"
    );
    assert_eq!(
        overridden["data"]["input_sources"]["batch"]["kind"],
        "command_line"
    );

    let again = report(&invoke(
        project.path(),
        &["init", "--model", "fixture:build_classifier"],
    ));
    assert_eq!(
        again["diagnostics"][0]["code"],
        "PROJECT_ALREADY_INITIALIZED"
    );
    assert_eq!(fs::read(&config_path).unwrap(), saved);

    let preview = tempfile::tempdir().unwrap();
    let previewed = assert_success(&invoke(
        preview.path(),
        &["init", "--model", "fixture:build_shared", "--dry-run"],
    ));
    assert_eq!(previewed["data"]["committed"], false);
    assert!(!preview.path().join(".model-maven").exists());
}

#[test]
fn declared_model_injection_seams_are_saved_verified_and_replayed_in_variants() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let initialized = assert_success(&invoke(
        project.path(),
        &[
            "init",
            "--model",
            "fixture:build_assembly_model",
            "--batch",
            "fixture:assembly_batch",
            "--inject",
            "encoder=fixture:build_assembly_encoder",
            "--inject",
            "head=fixture:build_assembly_head",
        ],
    ));
    assert_eq!(
        initialized["data"]["config"]["model_injections"]["encoder"],
        "fixture:build_assembly_encoder"
    );

    let inspected = assert_success(&invoke(project.path(), &["model", "inspect"]));
    assert!(
        inspected["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|module| module["path"] == "model.encoder.0" && module["type"] == "Linear")
    );
    let evidence = inspected["data"]["origin"]["injection_evidence"]
        .as_array()
        .unwrap();
    assert_eq!(evidence.len(), 2);
    assert!(
        evidence
            .iter()
            .any(|item| item["slot"] == "encoder" && item["status"] == "registered")
    );

    let probed = assert_success(&invoke(project.path(), &["model", "probe"]));
    assert_eq!(
        probed["data"]["outputs"]["shape"],
        serde_json::json!([2, 2])
    );
    assert!(
        probed["data"]["origin"]["injection_evidence"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["slot"] == "encoder" && item["sample_forward"] == "observed")
    );

    let direct = assert_success(&invoke(
        project.path(),
        &[
            "model",
            "probe",
            "--model",
            "fixture:build_assembly_model",
            "--batch",
            "fixture:assembly_batch",
            "--inject",
            "encoder=fixture:build_assembly_encoder",
            "--inject",
            "head=fixture:build_assembly_head",
        ],
    ));
    assert_eq!(
        direct["data"]["outputs"]["shape"],
        serde_json::json!([2, 2])
    );

    let duplicate = report(&invoke(
        project.path(),
        &[
            "model",
            "inspect",
            "--model",
            "fixture:build_assembly_model",
            "--inject",
            "encoder=fixture:build_assembly_encoder",
            "--inject",
            "encoder=fixture:build_assembly_encoder",
        ],
    ));
    assert_eq!(
        duplicate["diagnostics"][0]["code"],
        "MODEL_INJECTION_INVALID"
    );

    let tried = assert_success(&invoke(
        project.path(),
        &[
            "try",
            "--step",
            "fixture:assembly_training_step",
            "--steps",
            "2",
        ],
    ));
    assert_eq!(
        tried["data"]["origin"]["model_injections"]["head"],
        "fixture:build_assembly_head"
    );
    assert_eq!(
        tried["data"]["input_sources"]["model"]["kind"],
        "project_config"
    );
    assert_eq!(
        tried["data"]["input_sources"]["batch"]["kind"],
        "project_config"
    );
    let repeated = assert_success(&invoke(project.path(), &["try", "--repeat", "latest"]));
    assert_eq!(
        repeated["data"]["origin"]["model_injections"]["encoder"],
        "fixture:build_assembly_encoder"
    );
    assert_eq!(
        repeated["data"]["input_sources"]["model"]["kind"],
        "saved_run"
    );
    assert_eq!(
        repeated["data"]["input_sources"]["batch"]["kind"],
        "saved_run"
    );

    let trained = assert_success(&invoke(
        project.path(),
        &[
            "train",
            "--step",
            "fixture:assembly_training_step",
            "--steps",
            "2",
            "--checkpoint-every",
            "1",
        ],
    ));
    assert_eq!(
        trained["data"]["origin"]["model_injections"]["encoder"],
        "fixture:build_assembly_encoder"
    );
    let resumed = assert_success(&invoke(
        project.path(),
        &["train", "--resume", "latest", "--additional-steps", "1"],
    ));
    assert_eq!(resumed["data"]["resumed_from_steps"], 2);

    let replaced = assert_success(&invoke(
        project.path(),
        &[
            "--variant",
            "assembled",
            "layer",
            "replace",
            "model.encoder",
            "--model",
            "fixture:build_assembly_model",
            "--batch",
            "fixture:assembly_batch",
            "--type",
            "Identity",
        ],
    ));
    assert_eq!(
        replaced["data"]["variant"]["origin"]["model_injections"]["encoder"],
        "fixture:build_assembly_encoder"
    );

    let reloaded = assert_success(&invoke(
        project.path(),
        &[
            "--variant",
            "assembled",
            "model",
            "probe",
            "--batch",
            "fixture:assembly_batch",
        ],
    ));
    assert_eq!(
        reloaded["data"]["outputs"]["shape"],
        serde_json::json!([2, 2])
    );
    assert!(
        reloaded["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|module| module["path"] == "model.encoder" && module["type"] == "Identity")
    );

    let undeclared = report(&invoke(
        project.path(),
        &[
            "model",
            "inspect",
            "--model",
            "fixture:build_shared",
            "--inject",
            "encoder=fixture:build_assembly_encoder",
        ],
    ));
    assert_eq!(
        undeclared["diagnostics"][0]["code"],
        "MODEL_INJECTION_SEAM_UNDECLARED"
    );

    let ignored = report(&invoke(
        project.path(),
        &[
            "model",
            "inspect",
            "--model",
            "fixture:build_assembly_unused_model",
            "--inject",
            "encoder=fixture:build_assembly_encoder",
        ],
    ));
    assert_eq!(ignored["diagnostics"][0]["code"], "MODEL_INJECTION_UNUSED");
}

#[test]
fn registered_layer_components_can_be_parameterized_and_locked_as_model_injections() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let schema = r#"{"type":"object","properties":{"negative_slope":{"type":"number","default":0.2}},"required":[],"additionalProperties":false}"#;
    assert_success(&invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/leaky",
            "--kind",
            "layer",
            "--entrypoint",
            "fixture:build_param_activation",
            "--version",
            "1.0.0",
            "--params-schema",
            schema,
        ],
    ));

    let invalid_parameter = report(&invoke(
        project.path(),
        &[
            "model",
            "inspect",
            "--model",
            "fixture:build_assembly_model",
            "--inject",
            "encoder=component:local/leaky@1.0.0",
            "--inject-param",
            "encoder.not_a_parameter=1",
            "--inject",
            "head=fixture:build_assembly_head",
        ],
    ));
    assert_eq!(
        invalid_parameter["diagnostics"][0]["code"],
        "MODEL_INJECTION_INVALID"
    );
    assert!(!project.path().join(".model-maven/project.json").exists());

    let initialized = assert_success(&invoke(
        project.path(),
        &[
            "init",
            "--model",
            "fixture:build_assembly_model",
            "--batch",
            "fixture:assembly_batch",
            "--inject",
            "encoder=component:local/leaky@1.0.0",
            "--inject-param",
            "encoder.negative_slope=0.3",
            "--inject",
            "head=fixture:build_assembly_head",
        ],
    ));
    let encoder = &initialized["data"]["config"]["model_injections"]["encoder"];
    assert_eq!(encoder["factory"], "fixture:build_param_activation");
    assert_eq!(encoder["params"]["negative_slope"], 0.3);
    assert_eq!(encoder["component_ref"]["id"], "local/leaky");
    assert_eq!(encoder["component_ref"]["version"], "1.0.0");

    let uninstall_preview = assert_success(&invoke(
        project.path(),
        &["component", "uninstall", "local/leaky@1.0.0"],
    ));
    assert_eq!(uninstall_preview["data"]["requires_force"], true);
    assert!(
        uninstall_preview["data"]["project_config_references"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reference| reference["path"] == ".model-maven/project.json")
    );
    let blocked_uninstall = report(&invoke(
        project.path(),
        &["component", "uninstall", "local/leaky@1.0.0", "--apply"],
    ));
    assert_eq!(
        blocked_uninstall["diagnostics"][0]["code"],
        "COMPONENT_IN_USE"
    );

    let inspected = assert_success(&invoke(project.path(), &["model", "inspect"]));
    assert!(
        inspected["data"]["origin"]["injection_evidence"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| {
                item["slot"] == "encoder"
                    && item["factory"] == "fixture:build_param_activation"
                    && item["params"]["negative_slope"] == 0.3
                    && item["component_ref"]["id"] == "local/leaky"
            })
    );
    let probed = assert_success(&invoke(project.path(), &["model", "probe"]));
    assert_eq!(
        probed["data"]["outputs"]["shape"],
        serde_json::json!([2, 2])
    );

    let tried = assert_success(&invoke(
        project.path(),
        &[
            "try",
            "--step",
            "fixture:assembly_training_step",
            "--steps",
            "1",
        ],
    ));
    assert_eq!(
        tried["data"]["origin"]["model_injections"]["encoder"]["component_ref"]["id"],
        "local/leaky"
    );
    let repeated = assert_success(&invoke(project.path(), &["try", "--repeat", "latest"]));
    assert_eq!(
        repeated["data"]["origin"]["model_injections"]["encoder"]["params"]["negative_slope"],
        0.3
    );

    let edited = assert_success(&invoke(
        project.path(),
        &[
            "--variant",
            "registered",
            "layer",
            "replace",
            "model.encoder",
            "--model",
            "fixture:build_assembly_model",
            "--batch",
            "fixture:assembly_batch",
            "--type",
            "Identity",
        ],
    ));
    assert_eq!(
        edited["data"]["variant"]["origin"]["model_injections"]["encoder"]["component_ref"]["id"],
        "local/leaky"
    );
    let reloaded = assert_success(&invoke(
        project.path(),
        &[
            "--variant",
            "registered",
            "model",
            "probe",
            "--batch",
            "fixture:assembly_batch",
        ],
    ));
    assert_eq!(
        reloaded["data"]["outputs"]["shape"],
        serde_json::json!([2, 2])
    );

    let component = assert_success(&invoke(
        project.path(),
        &["component", "show", "local/leaky@1.0.0"],
    ));
    let manifest_path = project.path().join(
        component["data"]["component"]["manifest_path"]
            .as_str()
            .unwrap(),
    );
    fs::OpenOptions::new()
        .append(true)
        .open(&manifest_path)
        .unwrap()
        .write_all(b"\n")
        .unwrap();
    let stale = report(&invoke(project.path(), &["model", "inspect"]));
    assert_eq!(stale["diagnostics"][0]["code"], "COMPONENT_VERSION_CHANGED");
}

#[test]
fn component_uninstall_apply_removes_registration_after_preview() {
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());

    let added = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/removable",
            "--kind",
            "layer",
            "--entrypoint",
            "fixture:build_assembly_encoder",
            "--version",
            "1.0.0",
        ],
    ));
    let manifest_path = project.path().join(
        added["data"]["component"]["manifest_path"]
            .as_str()
            .unwrap(),
    );
    assert!(manifest_path.is_file());

    let preview = assert_success(&invoke(
        project.path(),
        &["component", "uninstall", "local/removable@1.0.0"],
    ));
    assert_eq!(preview["data"]["requires_force"], false);
    assert_eq!(preview["data"]["applied"], false);

    let removed = assert_success(&invoke(
        project.path(),
        &["component", "uninstall", "local/removable@1.0.0", "--apply"],
    ));
    assert_eq!(removed["data"]["applied"], true);
    assert!(!manifest_path.exists());
}

#[test]
fn component_install_builds_and_registers_scaffolded_wheel() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());

    assert_success(&invoke(
        project.path(),
        &[
            "component",
            "scaffold",
            "local/installed",
            "--kind",
            "layer",
            "--output",
            "extensions/installed",
        ],
    ));
    let installed = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "install",
            "extensions/installed",
            "--no-build-isolation",
        ],
    ));
    assert_eq!(
        installed["data"]["installation"]["already_installed"],
        false
    );
    assert_eq!(
        installed["data"]["installation"]["environment_modified"],
        false
    );
    assert_eq!(
        installed["data"]["installation"]["components"][0]["component_id"],
        "local/installed"
    );
    let installed_root = project.path().join(
        installed["data"]["installation"]["site_packages"]
            .as_str()
            .unwrap(),
    );
    assert!(installed_root.is_dir());
    assert!(
        installed["data"]["installation"]["wheel_sha256"]
            .as_str()
            .unwrap()
            .len()
            == 64
    );

    let initialized = assert_success(&invoke(
        project.path(),
        &[
            "init",
            "--model",
            "fixture:build_assembly_model",
            "--batch",
            "fixture:assembly_batch",
            "--inject",
            "encoder=component:local/installed@0.1.0",
            "--inject",
            "head=fixture:build_assembly_head",
        ],
    ));
    assert_eq!(
        initialized["data"]["config"]["model_injections"]["encoder"]["component_ref"]["id"],
        "local/installed"
    );
    let trained = assert_success(&invoke(
        project.path(),
        &[
            "train",
            "--batch",
            "fixture:assembly_batch",
            "--step",
            "fixture:assembly_training_step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    ));
    let outside = tempfile::tempdir().unwrap();
    let destination = outside.path().join("extension-export");
    let exported = assert_success(&invoke(
        project.path(),
        &["export", destination.to_str().unwrap()],
    ));
    assert_eq!(exported["operation"], "export");
    let manifest: Value =
        serde_json::from_slice(&fs::read(destination.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["inputs"]["model_injections"]["encoder"]["component_ref"]["id"],
        "local/installed"
    );
    assert_eq!(manifest["extension_packages"].as_array().unwrap().len(), 1);
    assert_eq!(
        manifest["extension_packages"][0]["distribution"],
        "model-maven-local-installed"
    );
    assert!(destination.join("vendor/packages").is_dir());
    assert!(trained["data"]["run_id"].as_str().is_some());
    let restored = Command::new(python())
        .arg(destination.join("restore_model.py"))
        .current_dir(&destination)
        .output()
        .unwrap();
    assert!(
        restored.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&restored.stdout),
        String::from_utf8_lossy(&restored.stderr)
    );
    assert!(String::from_utf8_lossy(&restored.stdout).contains("模型已恢复"));
}

#[test]
fn doctor_reports_real_runtime_capabilities_as_json() {
    let project = tempfile::tempdir().unwrap();
    let output = invoke(project.path(), &["doctor"]);
    let value = assert_success(&output);
    assert_eq!(value["operation"], "doctor");
    assert_eq!(value["data"]["torch_available"], has_torch());
    assert_eq!(value["data"]["compatibility"]["worker_api_version"], 1);
    assert_eq!(value["data"]["compatibility"]["worker_protocol_version"], 1);
    let capabilities = value["data"]["capabilities"].as_array().unwrap();
    let state = |id: &str| {
        capabilities
            .iter()
            .find(|capability| capability["id"] == id)
            .unwrap_or_else(|| panic!("doctor omitted capability {id}"))["state"]
            .as_str()
            .unwrap()
    };
    assert_eq!(
        state("model.forward_probe"),
        if has_torch() {
            "supported"
        } else {
            "unsupported"
        }
    );
    assert_eq!(
        state("training.detached"),
        if has_torch() && cfg!(target_os = "linux") {
            "supported"
        } else {
            "unsupported"
        }
    );
    let implemented = value["data"]["implemented"].as_array().unwrap();
    assert!(
        implemented
            .iter()
            .any(|item| item == "atomic Layer edit plans")
    );
    assert!(
        implemented
            .iter()
            .any(|item| item == "local Layer Edit Provider API v1")
    );
    assert!(
        implemented
            .iter()
            .any(|item| item == "declared model factory component injection seams")
    );
    assert!(implemented.iter().any(|item| {
        item == "schema-validated registered Layer injection with manifest and transitive dependency locks"
    }));
    assert!(implemented.iter().any(|item| {
        item == "registered Layer component runtime smoke test (factory construction, finite state, declared forward probe)"
    }));
    assert!(
        implemented
            .iter()
            .any(|item| item == "bounded managed-step AdamW probe")
    );
    assert!(
        implemented
            .iter()
            .any(|item| item == "bounded manual-step with named disjoint AdamW groups")
    );
    assert!(implemented.iter().any(|item| item
        == "foreground training trial records, source/runtime checked repeat, and runs list/show"));
    assert!(implemented.iter().any(|item| {
        item == "formal training timeout requests a cooperative step-boundary checkpoint before force termination"
    }));
    assert!(implemented.iter().any(|item| {
        item == "multi-file source set plans with recoverable journal, fresh-worker revalidation, automatic rollback, conflict-checked undo/redo"
    }));
    assert!(
        implemented
            .iter()
            .any(|item| item == "Sequential/ModuleList/ModuleDict variant CRUD")
    );
    assert!(
        implemented
            .iter()
            .any(|item| item == "ModuleDict named-key CRUD with insertion-order preservation")
    );
    assert!(implemented.iter().any(|item| {
        item == "source CRUD for directly assigned nn.Sequential, nn.ModuleList, and nn.ModuleDict with built-in or importable Python factory Layers"
    }));
    assert!(implemented.iter().any(|item| {
        item == "grouped source Sequential/ModuleList/ModuleDict add/replace/remove transactions across project files with recovery, revalidation, and conflict-checked undo/redo, including registered local component factories"
    }));
    assert!(implemented.iter().any(|item| {
        item == "fixed-plan cleanup of ended-run logs and progress files with status, keep, and content-hash checks"
    }));
    assert!(implemented.iter().any(|item| {
        item == "persistent variant CRUD for Sequential/ModuleList/ModuleDict subclasses with forward validation"
    }));
    let not_implemented = value["data"]["not_implemented"].as_array().unwrap();
    assert!(
        !not_implemented
            .iter()
            .any(|item| item == "non-Layer component runtime conformance suites")
    );
    assert!(not_implemented.iter().any(|item| {
        item == "preemption of a blocked third-party Python/CUDA call before the CLI watchdog terminates its worker"
    }));
    assert!(!not_implemented.iter().any(|item| {
        item.as_str().is_some_and(|name| {
            name.contains("grouped source add")
                || name.contains("custom component factory source edits")
        })
    }));
}

#[test]
fn doctor_renders_runtime_capabilities_as_a_human_summary() {
    let project = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python(), "doctor"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Model-Maven 环境与能力诊断"));
    assert!(stdout.contains("当前可用能力："));
    assert!(stdout.contains("模型结构检查"));
    assert!(!stdout.trim_start().starts_with('{'));
}

#[test]
fn augment_command_persists_ordered_train_and_eval_pipelines() {
    let project = tempfile::tempdir().unwrap();
    let root = project.path();
    for (args, label) in [
        (&["init", "--model", "fixture:build_shared"][..], "init"),
        (
            &[
                "augment", "add", "Resize", "--id", "resize", "--target", "args.0", "--param",
                "size=224",
            ][..],
            "add resize",
        ),
        (
            &[
                "augment", "add", "ToTensor", "--id", "tensor", "--target", "args.0",
            ][..],
            "add tensor",
        ),
        (
            &["augment", "move", "tensor", "--before", "resize"][..],
            "move",
        ),
        (
            &["augment", "set", "resize", "--param", "size=128"][..],
            "set",
        ),
        (
            &[
                "augment",
                "add",
                "Normalize",
                "--split",
                "eval",
                "--id",
                "norm",
                "--target",
                "args.0",
                "--param",
                "mean=[0.5,0.5,0.5]",
                "--param",
                "std=[0.5,0.5,0.5]",
            ][..],
            "add eval",
        ),
    ] {
        let output = invoke(root, args);
        assert!(
            output.status.success(),
            "{label}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let config: Value =
        serde_json::from_slice(&fs::read(root.join(".model-maven/project.json")).unwrap()).unwrap();
    let train = config["augment"]["train"].as_array().unwrap();
    assert_eq!(train[0]["id"], "tensor");
    assert_eq!(train[1]["id"], "resize");
    assert_eq!(train[1]["parameters"]["size"], 128);
    assert_eq!(config["augment"]["eval"][0]["id"], "norm");

    let output = invoke(root, &["augment", "remove", "tensor"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    for args in [
        &[
            "augment",
            "add",
            "RandomResizedCrop",
            "--id",
            "crop",
            "--target",
            "args.0",
            "--param",
            "size=224",
            "--scale",
            "0.6,1.0",
        ][..],
        &["augment", "set", "crop", "--scale", "0.7,0.9"][..],
    ] {
        let output = invoke(root, args);
        assert!(
            output.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let config: Value =
        serde_json::from_slice(&fs::read(root.join(".model-maven/project.json")).unwrap()).unwrap();
    assert_eq!(config["augment"]["train"].as_array().unwrap().len(), 2);
    assert_eq!(config["augment"]["train"][1]["type"], "RandomResizedCrop");
    assert_eq!(
        config["augment"]["train"][1]["parameters"]["scale"],
        serde_json::json!([0.7, 0.9])
    );
}

#[test]
fn assembly_show_renders_operation_readiness_and_configuration_sources() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let init = invoke(
        project.path(),
        &["init", "--model", "fixture:build_classifier"],
    );
    assert_success(&init);

    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python(), "assembly", "show"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("操作就绪状态：可执行 · 模型结构检查"),
        "{stdout}"
    );
    assert!(stdout.contains("配置版本："), "{stdout}");
    assert!(
        stdout.contains("model · 必需 · 已就绪 · 来源：项目配置"),
        "{stdout}"
    );
    assert!(stdout.contains("batch · 可选 · 待确认"), "{stdout}");
    assert!(
        stdout.contains("下一步 · 选择示例批次以查看前向 · 可用"),
        "{stdout}"
    );

    let inspect = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python(), "model", "inspect"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(inspect.status.success());
    let inspect_stdout = String::from_utf8_lossy(&inspect.stdout);
    assert!(
        inspect_stdout.contains("本次输入来源："),
        "{inspect_stdout}"
    );
    assert!(
        inspect_stdout.contains("模型入口 · 项目配置"),
        "{inspect_stdout}"
    );
    assert!(
        inspect_stdout.contains("项目配置版本 ·"),
        "{inspect_stdout}"
    );
}

#[test]
fn project_config_history_previews_and_restores_selected_fields() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    assert_success(&invoke(
        project.path(),
        &[
            "init",
            "--model",
            "fixture:build_assembly_model",
            "--inject",
            "encoder=fixture:build_assembly_encoder",
            "--inject",
            "head=fixture:build_assembly_head",
        ],
    ));
    let original_config: Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/project.json")).unwrap(),
    )
    .unwrap();

    let edited = assert_success(&invoke(
        project.path(),
        &[
            "assembly",
            "set",
            "encoder",
            "--factory",
            "fixture:build_assembly_head",
        ],
    ));
    assert_eq!(edited["data"]["committed"], true);

    let history = assert_success(&invoke(project.path(), &["config", "history"]));
    assert_eq!(history["data"]["history"].as_array().unwrap().len(), 1);
    let old_revision = history["data"]["history"][0]["revision"].as_str().unwrap();
    assert_eq!(history["data"]["history"][0]["config"], original_config);

    let human_history = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python(), "config", "history"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human_history.status.success());
    let human_history_text = String::from_utf8_lossy(&human_history.stdout);
    assert!(human_history_text.contains("当前项目配置 · revision"));
    assert!(human_history_text.contains("模型 fixture:build_assembly_model"));

    let human_preview = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "config",
            "restore",
            old_revision,
            "--field",
            "model_injections",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human_preview.status.success());
    let human_preview_text = String::from_utf8_lossy(&human_preview.stdout);
    assert!(human_preview_text.contains("配置字段恢复预览"));
    assert!(human_preview_text.contains("--apply"));
    assert!(human_preview_text.contains("--if-current"));

    let preview = assert_success(&invoke(
        project.path(),
        &[
            "config",
            "restore",
            old_revision,
            "--field",
            "model_injections",
        ],
    ));
    assert_eq!(preview["data"]["committed"], false);
    assert_eq!(preview["data"]["fields"][0]["field"], "model_injections");
    assert_eq!(preview["data"]["fields"][0]["changed"], true);
    let current_config: Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/project.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        current_config["model_injections"]["encoder"],
        "fixture:build_assembly_head"
    );

    let preview_revision = preview["data"]["from_revision"].as_str().unwrap();
    assert_success(&invoke(
        project.path(),
        &[
            "assembly",
            "set",
            "head",
            "--factory",
            "fixture:build_assembly_encoder",
        ],
    ));
    let stale_restore = report(&invoke(
        project.path(),
        &[
            "config",
            "restore",
            old_revision,
            "--field",
            "model_injections",
            "--apply",
            "--if-current",
            preview_revision,
        ],
    ));
    assert_eq!(stale_restore["status"], "error");
    assert_eq!(
        stale_restore["diagnostics"][0]["code"],
        "PROJECT_CONFIG_CONFLICT"
    );
    let changed_config: Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/project.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        changed_config["model_injections"]["head"],
        "fixture:build_assembly_encoder"
    );

    let fresh_preview = assert_success(&invoke(
        project.path(),
        &[
            "config",
            "restore",
            old_revision,
            "--field",
            "model_injections",
        ],
    ));
    let fresh_revision = fresh_preview["data"]["from_revision"].as_str().unwrap();

    let restored = assert_success(&invoke(
        project.path(),
        &[
            "config",
            "restore",
            old_revision,
            "--field",
            "model_injections",
            "--apply",
            "--if-current",
            fresh_revision,
        ],
    ));
    assert_eq!(restored["data"]["committed"], true);
    let restored_config: Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/project.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        restored_config["model_injections"],
        original_config["model_injections"]
    );
    let no_op = assert_success(&invoke(
        project.path(),
        &[
            "config",
            "restore",
            old_revision,
            "--field",
            "model_injections",
            "--apply",
            "--if-current",
            restored["data"]["config_revision"].as_str().unwrap(),
        ],
    ));
    assert_eq!(no_op["data"]["unchanged"], true);
    assert_eq!(no_op["data"]["committed"], false);
}

#[test]
fn missing_operation_inputs_return_actionable_readiness_without_worker_runs() {
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    assert_success(&invoke(
        project.path(),
        &["init", "--model", "fixture:build_classifier"],
    ));

    let probe = report(&invoke(project.path(), &["model", "probe"]));
    assert_eq!(probe["status"], "error");
    assert_eq!(probe["diagnostics"][0]["code"], "OPERATION_NEEDS_INPUT");
    assert_eq!(
        probe["data"]["readiness"]["readiness"]["state"],
        "needs_input"
    );
    assert_eq!(probe["data"]["readiness"]["operation"], "forward_probe");
    assert_eq!(probe["data"]["readiness"]["inputs"][1]["key"], "batch");
    assert_eq!(probe["data"]["readiness"]["inputs"][1]["state"], "missing");
    assert!(
        probe["data"]["readiness"]["readiness"]["actions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|action| action["id"] == "configure_batch")
    );
    assert!(
        probe["data"]["readiness"]["inputs"][0]
            .get("value")
            .is_none()
    );
    let human_probe = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python(), "model", "probe"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!human_probe.status.success());
    let human_probe_text = String::from_utf8_lossy(&human_probe.stdout);
    assert!(human_probe_text.contains("操作就绪状态：需要输入 · 前向试跑"));
    assert!(human_probe_text.contains("batch · 必需 · 缺少"));
    assert!(human_probe_text.contains("提供示例批次工厂"));

    for args in [&["try", "--steps", "1"][..], &["train", "--steps", "1"][..]] {
        let report = report(&invoke(project.path(), args));
        assert_eq!(report["status"], "error", "{report:#}");
        assert_eq!(report["diagnostics"][0]["code"], "OPERATION_NEEDS_INPUT");
        assert!(
            report["data"]["readiness"]["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .any(|input| input["key"] == "training_step" && input["state"] == "missing")
        );
    }
    assert!(!project.path().join(".model-maven/runs").exists());

    let empty_project = tempfile::tempdir().unwrap();
    let edit = report(&invoke(
        empty_project.path(),
        &[
            "layer",
            "add",
            "--parent",
            "model",
            "--name",
            "activation",
            "--type",
            "ReLU",
        ],
    ));
    assert_eq!(edit["status"], "error", "{edit:#}");
    assert_eq!(edit["data"]["readiness"]["operation"], "apply_layer_edit");
    assert!(
        edit["data"]["readiness"]["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|input| input["key"] == "model" && input["state"] == "missing")
    );
    assert!(!empty_project.path().join(".model-maven/runs").exists());
}

#[test]
fn foreground_timeout_kills_a_slow_worker_and_returns_json() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let marker = project.path().join("slow-batch.started");
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--timeout",
            "8",
            "--python",
            &python(),
            "model",
            "probe",
            "--model",
            "fixture:build_shared",
            "--batch",
            "fixture:slow_batch",
        ])
        .env("MODEL_MAVEN_TIMEOUT_MARKER", &marker)
        .current_dir(project.path())
        .output()
        .unwrap();
    let value = report(&output);
    assert_eq!(value["diagnostics"][0]["code"], "WORKER_TIMEOUT");
    assert_eq!(output.status.code(), Some(6));
    assert_eq!(fs::read_to_string(marker).unwrap(), "started");
}

#[cfg(target_os = "linux")]
#[test]
fn foreground_timeout_kills_worker_descendants_in_its_process_group() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    fs::write(
        project.path().join("timeout_fixture.py"),
        r#"
import os
import subprocess
import sys
import time

def slow_batch_with_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    with open(os.environ["MODEL_MAVEN_TIMEOUT_CHILD_PID"], "w", encoding="utf-8") as marker:
        marker.write(str(child.pid))
    time.sleep(20)
"#,
    )
    .unwrap();
    let marker = project.path().join("timeout-child.pid");
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--timeout",
            "8",
            "--python",
            &python(),
            "model",
            "probe",
            "--model",
            "fixture:build_shared",
            "--batch",
            "timeout_fixture:slow_batch_with_child",
        ])
        .env("MODEL_MAVEN_TIMEOUT_CHILD_PID", &marker)
        .current_dir(project.path())
        .output()
        .unwrap();
    let value = report(&output);
    assert_eq!(value["diagnostics"][0]["code"], "WORKER_TIMEOUT");
    assert_eq!(output.status.code(), Some(6));

    let pid = fs::read_to_string(&marker)
        .unwrap_or_else(|error| {
            panic!(
                "worker did not write child PID: {error}; report={value:#}; stdout={}; stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .parse::<i32>()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat"));
        match stat {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Ok(stat) => {
                let state = stat
                    .rsplit_once(')')
                    .and_then(|(_, rest)| rest.split_whitespace().next());
                if state == Some("Z") {
                    break;
                }
            }
            Err(error) => panic!("could not inspect worker descendant {pid}: {error}"),
        }
        assert!(
            std::time::Instant::now() < deadline,
            "worker descendant {pid} remained alive after WORKER_TIMEOUT"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[test]
fn formal_training_timeout_saves_a_checkpoint_at_the_next_step_boundary() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("graceful_timeout_fixture.py"),
        r#"
import os
import time
import torch
from torch import nn

def build_model():
    return nn.Linear(2, 2)

def slow_loop(model, context):
    values = torch.ones(2, 2)
    marker = os.environ.get("MODEL_MAVEN_TIMEOUT_STEP_MARKER")
    if marker:
        with open(marker, "w", encoding="utf-8") as output:
            output.write("started")
    while context.remaining_steps:
        time.sleep(10)
        loss = model(values).square().mean()
        if not context.report_step(loss):
            break
"#,
    )
    .unwrap();
    let marker = project.path().join("training-step.started");
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--timeout",
            "8",
            "--python",
            &python(),
            "train",
            "--model",
            "graceful_timeout_fixture:build_model",
            "--training-mode",
            "external-loop",
            "--loop",
            "graceful_timeout_fixture:slow_loop",
            "--steps",
            "3",
            "--checkpoint-every",
            "3",
        ])
        .env("MODEL_MAVEN_TIMEOUT_STEP_MARKER", &marker)
        .current_dir(project.path())
        .output()
        .unwrap();
    let report = report(&output);
    assert_eq!(report["status"], "error", "{report:#}");
    assert_eq!(report["diagnostics"][0]["code"], "WORKER_TIMEOUT");
    assert_eq!(fs::read_to_string(marker).unwrap(), "started");

    let run_id = report["data"]["run_id"].as_str().expect("formal run ID");
    let latest_path = project
        .path()
        .join(".model-maven/runs")
        .join(run_id)
        .join("checkpoints/latest.json");
    let latest: Value = serde_json::from_slice(&fs::read(latest_path).unwrap()).unwrap();
    assert_eq!(latest["completed_steps"], 1);
}

#[cfg(unix)]
#[test]
fn sigint_cancels_the_worker_and_returns_exit_code_130() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "model",
            "probe",
            "--model",
            "fixture:build_shared",
            "--batch",
            "fixture:slow_batch",
        ])
        .current_dir(project.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let signal_result = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) };
    assert_eq!(signal_result, 0);
    let status = child.wait().unwrap();
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    let value: Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(status.code(), Some(130));
    assert_eq!(value["diagnostics"][0]["code"], "USER_INTERRUPTED");
}

#[test]
fn probe_reports_each_invocation_of_a_shared_module_without_polluting_json() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let output = invoke(
        project.path(),
        &[
            "model",
            "probe",
            "--model",
            "fixture:build_shared",
            "--batch",
            "fixture:shared_batch",
        ],
    );
    let value = assert_success(&output);
    let calls = value["data"]["calls"].as_array().unwrap();
    let shared: Vec<_> = calls
        .iter()
        .filter(|call| call["paths"][0] == "model.shared")
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
    assert!(String::from_utf8_lossy(&output.stdout).starts_with('{'));
    assert_eq!(String::from_utf8_lossy(&output.stdout).lines().count(), 1);
}

#[test]
fn probe_includes_dataset_identity_and_observed_sampling_coverage() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "from torch import nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.projection = nn.Linear(4, 2)\n    def forward(self, values):\n        return self.projection(values)\n\ndef build_model():\n    return Model()\n",
    )
    .unwrap();
    fs::write(
        project.path().join("data.py"),
        "import torch\n\nclass Samples:\n    def __len__(self): return 5\n    def __getitem__(self, index): return torch.full((4,), float(index))\n    def sample_id(self, index): return f'sample-{index}'\n\ndef build_dataset(): return Samples()\n",
    )
    .unwrap();
    let output = invoke(
        project.path(),
        &[
            "model",
            "probe",
            "--model",
            "model:build_model",
            "--batch",
            "data:build_dataset",
        ],
    );
    let value = assert_success(&output);
    let observation = &value["data"]["data_observation"];
    assert_eq!(observation["source"], "data:build_dataset");
    assert_eq!(observation["sampling"]["algorithm"], "sha256-index-rank-v1");
    assert_eq!(observation["sampling"]["actual_samples"], 1);
    assert_eq!(observation["coverage"]["kind"], "indexed-subset");
    assert_eq!(observation["coverage"]["population_size"], 5);
    assert_eq!(observation["capabilities"]["resume_cursor"], false);
}

#[test]
fn failed_probe_locates_the_last_module_and_user_source_frame() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let output = invoke(
        project.path(),
        &[
            "model",
            "probe",
            "--model",
            "fixture:build_contract_break",
            "--batch",
            "fixture:incompatible_batch",
        ],
    );
    let value = report(&output);
    assert_eq!(value["diagnostics"][0]["code"], "FORWARD_FAILED");
    let failure = &value["data"]["failure_location"];
    let last_call = &failure["last_observed_module_call"];
    assert_eq!(last_call["paths"][0], "model.fusion_projection");
    assert_eq!(last_call["module_type"], "Linear");
    assert_eq!(
        last_call["inputs"]["items"]["args"]["items"][0]["shape"],
        serde_json::json!([2, 48])
    );
    assert!(
        failure["project_stack"]
            .as_array()
            .unwrap()
            .iter()
            .any(|frame| { frame["file"] == "fixture.py" && frame["function"] == "forward" })
    );

    let human = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "model",
            "probe",
            "--model",
            "fixture:build_contract_break",
            "--batch",
            "fixture:incompatible_batch",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!human.status.success());
    let human_error = String::from_utf8_lossy(&human.stderr);
    assert!(human_error.contains("最后观测到的模块"), "{human_error}");
    assert!(
        human_error.contains("model.fusion_projection"),
        "{human_error}"
    );
    assert!(human_error.contains("fixture.py"), "{human_error}");

    let truncated = invoke(
        project.path(),
        &[
            "model",
            "probe",
            "--model",
            "fixture:build_truncated_contract_break",
            "--batch",
            "fixture:truncated_incompatible_batch",
        ],
    );
    let truncated = report(&truncated);
    assert_eq!(truncated["data"]["trace_truncated"], true);
    assert_eq!(truncated["data"]["calls"].as_array().unwrap().len(), 256);
    assert!(truncated["data"]["calls_observed"].as_u64().unwrap() > 256);
    assert_eq!(
        truncated["data"]["failure_location"]["last_observed_module_call"]["paths"][0],
        "model.fusion_projection"
    );
    assert!(
        truncated["data"]["failure_location"]["last_observed_module_call"]["call_id"]
            .as_u64()
            .unwrap()
            > 256
    );
}

#[test]
fn custom_loss_backward_reports_finite_gradients_and_rejects_nonfinite_loss() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let verified = invoke(
        project.path(),
        &[
            "model",
            "verify",
            "--model",
            "fixture:build_classifier",
            "--batch",
            "fixture:classification_batch",
            "--loss",
            "fixture:classification_loss",
        ],
    );
    let verified = assert_success(&verified);
    assert!(verified["data"]["loss"].as_f64().unwrap().is_finite());
    assert_eq!(verified["data"]["gradient_count"], 2);
    for gradient in verified["data"]["gradients"].as_array().unwrap() {
        assert_eq!(gradient["finite"], true);
        assert!(gradient["norm"].as_f64().unwrap().is_finite());
    }

    let rejected = invoke(
        project.path(),
        &[
            "model",
            "verify",
            "--model",
            "fixture:build_classifier",
            "--batch",
            "fixture:classification_batch",
            "--loss",
            "fixture:nonfinite_loss",
        ],
    );
    let rejected = report(&rejected);
    assert_eq!(rejected["status"], "error");
    assert_eq!(rejected["diagnostics"][0]["code"], "LOSS_NONFINITE");
}

#[test]
fn managed_training_probe_persists_run_evidence_but_keeps_weights_ephemeral() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let source_before = fs::read(project.path().join("fixture.py")).unwrap();
    let output = invoke(
        project.path(),
        &[
            "try",
            "--model",
            "fixture:build_classifier",
            "--batch",
            "fixture:classification_batch",
            "--step",
            "fixture:classification_training_step",
            "--steps",
            "4",
            "--lr",
            "0.01",
            "--step-param",
            "label_smoothing=0.1",
            "--step-param",
            "api_token=\"example-private-token\"",
        ],
    );
    let result = assert_success(&output);
    assert_eq!(result["operation"], "training_probe");
    assert_eq!(result["data"]["completed_steps"], 4);
    assert_eq!(result["data"]["optimizer_updates"], 4);
    assert_eq!(result["data"]["checkpoint_saved"], false);
    assert!(result["data"]["runtime"]["pytorch_version"].is_string());
    assert_eq!(result["data"]["source_changed_during_run"], false);
    let run_id = result["data"]["run_id"].as_str().unwrap();
    let source_key = project
        .path()
        .join("fixture.py")
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert!(result["data"]["source_manifest"][&source_key].is_string());
    assert!(result["data"]["mean_loss"].as_f64().unwrap().is_finite());
    assert!(result["data"]["mean_metrics"]["accuracy"].is_number());
    assert_eq!(
        result["data"]["last_outputs"]["shape"],
        serde_json::json!([2, 3])
    );
    assert!(!project.path().join(".model-maven/variants").exists());
    let run_dir = project.path().join(".model-maven/runs").join(run_id);
    assert!(run_dir.join("run.json").is_file());
    assert!(run_dir.join("inputs.json").is_file());
    assert!(run_dir.join("events.jsonl").is_file());
    assert!(run_dir.join("report.json").is_file());
    assert!(!run_dir.join("checkpoint.pt").exists());
    assert_eq!(
        fs::read(project.path().join("fixture.py")).unwrap(),
        source_before
    );

    let listed = invoke(project.path(), &["runs", "list"]);
    let listed = assert_success(&listed);
    assert_eq!(listed["data"]["runs"].as_array().unwrap().len(), 1);
    assert_eq!(listed["data"]["runs"][0]["id"], run_id);
    assert_eq!(listed["data"]["runs"][0]["status"], "completed");

    let shown = invoke(project.path(), &["runs", "show", run_id]);
    let shown = assert_success(&shown);
    assert_eq!(
        shown["data"]["inputs"]["training_step"],
        "fixture:classification_training_step"
    );
    assert_eq!(
        shown["data"]["inputs"]["training_params"]["api_token"],
        "[REDACTED]"
    );
    assert!(
        shown["data"]["inputs"]["redacted_fields"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "$.training_params.api_token")
    );
    assert_eq!(shown["data"]["inputs"]["runtime"]["timeout_seconds"], 30);
    assert_eq!(shown["data"]["events"].as_array().unwrap().len(), 3);
    assert_eq!(shown["data"]["events"][1]["event"], "supervisor_started");
    assert_eq!(shown["data"]["report"]["data"]["completed_steps"], 4);
    assert_eq!(
        shown["data"]["report"]["data"]["source_manifest"][&source_key],
        result["data"]["source_manifest"][&source_key]
    );
    let recent = invoke(project.path(), &["runs", "show"]);
    let recent = assert_success(&recent);
    assert_eq!(recent["data"]["record"]["id"], run_id);
    let human = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["runs", "show", run_id])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human.status.success());
    let human_output = String::from_utf8_lossy(&human.stdout);
    assert!(human_output.contains("运行记录 · 已完成"), "{human_output}");
    assert!(
        human_output.contains("训练：managed-step · 4/4 步"),
        "{human_output}"
    );
    assert!(
        human_output.contains("完整运行证据：m-model --json runs show"),
        "{human_output}"
    );

    let repeat_without_secret = invoke(project.path(), &["try", "--repeat"]);
    let repeat_without_secret = report(&repeat_without_secret);
    assert_eq!(repeat_without_secret["status"], "error");
    assert_eq!(
        repeat_without_secret["diagnostics"][0]["code"],
        "REPEAT_SECRET_REQUIRED"
    );
    let repeated = invoke(
        project.path(),
        &[
            "try",
            "--repeat",
            run_id,
            "--steps",
            "2",
            "--step-param",
            "api_token=\"replacement-private-token\"",
        ],
    );
    let repeated = assert_success(&repeated);
    assert_eq!(repeated["data"]["completed_steps"], 2);
    assert_eq!(repeated["data"]["repeat_of"], run_id);
    assert!(
        repeated["data"]["repeat_overrides"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item == "steps")
    );
    let repeated_id = repeated["data"]["run_id"].as_str().unwrap();
    assert_ne!(repeated_id, run_id);
    let repeated_record = invoke(project.path(), &["runs", "show", repeated_id]);
    let repeated_record = assert_success(&repeated_record);
    assert_eq!(repeated_record["data"]["record"]["parent_run_id"], run_id);
    assert_eq!(repeated_record["data"]["inputs"]["repeat_of"], run_id);
    assert_eq!(repeated_record["data"]["inputs"]["steps"], 2);
    assert_eq!(
        repeated_record["data"]["inputs"]["training_step"],
        "fixture:classification_training_step"
    );
    assert_eq!(
        repeated_record["data"]["inputs"]["training_params"]["api_token"],
        "[REDACTED]"
    );
    assert!(
        repeated_record["data"]["inputs"]["repeat_overrides"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item == "steps")
    );
    let human_repeat = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["runs", "show", repeated_id])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human_repeat.status.success());
    let human_repeat = String::from_utf8_lossy(&human_repeat.stdout);
    assert!(
        human_repeat.contains(&format!("重跑自：{run_id}")),
        "{human_repeat}"
    );
    assert!(human_repeat.contains("与原运行不同："), "{human_repeat}");
    assert!(human_repeat.contains("steps"), "{human_repeat}");
    assert!(human_repeat.contains("输入来源："), "{human_repeat}");
    assert!(human_repeat.contains("已保存运行"), "{human_repeat}");
    assert!(
        human_repeat.contains("← 本次命令 · --model"),
        "{human_repeat}"
    );

    let failed = invoke(
        project.path(),
        &[
            "try",
            "--model",
            "fixture:build_classifier",
            "--batch",
            "fixture:classification_batch",
            "--step",
            "fixture:failing_training_step",
            "--steps",
            "4",
        ],
    );
    let failed = report(&failed);
    assert_eq!(failed["status"], "error");
    assert_eq!(failed["diagnostics"][0]["code"], "STEP_FAILED");
    let failed_run_id = failed["data"]["run_id"].as_str().unwrap();
    assert_eq!(failed["data"]["completed_steps"], 1);
    assert_eq!(failed["data"]["failed_step"], 2);
    assert_eq!(failed["data"]["cause_code"], "USER_STEP_EXCEPTION");
    let after_failure = invoke(project.path(), &["runs", "list"]);
    let after_failure = assert_success(&after_failure);
    assert_eq!(after_failure["data"]["runs"].as_array().unwrap().len(), 3);
    assert_eq!(after_failure["data"]["runs"][0]["status"], "failed");
    let failed_run = invoke(project.path(), &["runs", "show", failed_run_id]);
    let failed_run = assert_success(&failed_run);
    assert_eq!(failed_run["data"]["report"]["status"], "error");
    let human_failure = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["runs", "show", failed_run_id])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human_failure.status.success());
    let human_output = String::from_utf8_lossy(&human_failure.stdout);
    assert!(human_output.contains("运行记录 · 失败"), "{human_output}");
    assert!(human_output.contains("STEP_FAILED"), "{human_output}");
    let recent = invoke(project.path(), &["runs", "show"]);
    let recent = assert_success(&recent);
    assert_eq!(recent["data"]["record"]["id"], failed_run_id);
    assert!(!project.path().join(".model-maven/variants").exists());
    assert_eq!(
        fs::read(project.path().join("fixture.py")).unwrap(),
        source_before
    );
    fs::OpenOptions::new()
        .append(true)
        .open(project.path().join("fixture.py"))
        .unwrap()
        .write_all(b"\n# changed after the recorded trial\n")
        .unwrap();
    let source_changed_repeat = invoke(
        project.path(),
        &[
            "try",
            "--repeat",
            repeated_id,
            "--step-param",
            "api_token=\"replacement-private-token\"",
        ],
    );
    let source_changed_repeat = report(&source_changed_repeat);
    assert_eq!(source_changed_repeat["status"], "error");
    assert_eq!(
        source_changed_repeat["diagnostics"][0]["code"],
        "REPEAT_SOURCE_CHANGED"
    );
    assert!(
        !source_changed_repeat["data"]["changed_files"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(source_changed_repeat["data"]["completed_steps"].is_null());
    let human_source_changed = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "try",
            "--repeat",
            repeated_id,
            "--step-param",
            "api_token=\"replacement-private-token\"",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!human_source_changed.status.success());
    let human_error = String::from_utf8_lossy(&human_source_changed.stderr);
    assert!(human_error.contains("修改文件："), "{human_error}");
    assert!(human_error.contains("fixture.py"), "{human_error}");
}

#[test]
fn runs_clean_previews_and_hash_checks_logs_and_progress_without_removing_run_history() {
    let project = tempfile::tempdir().unwrap();
    let runs_root = project.path().join(".model-maven/runs");
    let completed_id = "run-100-1-1";
    let completed = runs_root.join(completed_id);
    fs::create_dir_all(completed.join("checkpoints")).unwrap();
    fs::write(
        completed.join("run.json"),
        serde_json::json!({"schema_version":1,"id":completed_id,"status":"completed","kept":false})
            .to_string(),
    )
    .unwrap();
    fs::write(completed.join("events.jsonl"), "{\"event\":\"finished\"}\n").unwrap();
    fs::write(completed.join("worker.log"), "old worker output\n").unwrap();
    fs::write(completed.join("progress.jsonl"), "old progress\n").unwrap();

    let kept_id = "run-101-1-1";
    let kept = runs_root.join(kept_id);
    fs::create_dir_all(kept.join("checkpoints")).unwrap();
    fs::write(
        kept.join("run.json"),
        serde_json::json!({"schema_version":1,"id":kept_id,"status":"completed","kept":true})
            .to_string(),
    )
    .unwrap();
    fs::write(kept.join("worker.log"), "keep this log\n").unwrap();
    fs::write(kept.join("progress.jsonl"), "keep this progress\n").unwrap();

    let active_id = "run-102-1-1";
    let active = runs_root.join(active_id);
    fs::create_dir_all(active.join("checkpoints")).unwrap();
    fs::write(
        active.join("run.json"),
        serde_json::json!({"schema_version":1,"id":active_id,"status":"running","kept":false})
            .to_string(),
    )
    .unwrap();
    fs::write(active.join("worker.log"), "active log\n").unwrap();

    let preview = assert_success(&invoke(
        project.path(),
        &[
            "runs",
            "clean",
            "--include-logs",
            "--include-progress",
            "--dry-run",
        ],
    ));
    let plan_id = preview["data"]["id"].as_str().unwrap();
    let items = preview["data"]["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert!(items.iter().any(|item| item["kind"] == "run_log"));
    assert!(items.iter().any(|item| item["kind"] == "run_progress"));
    assert!(
        preview["data"]["skipped"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["run_id"] == kept_id && item["reason"] == "运行已标记保留" })
    );
    assert!(
        preview["data"]["skipped"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| {
                item["run_id"] == active_id && item["reason"].as_str().unwrap().contains("running")
            })
    );

    fs::write(completed.join("progress.jsonl"), "changed after preview\n").unwrap();
    let applied = assert_success(&invoke(
        project.path(),
        &["runs", "clean", "--apply", plan_id],
    ));
    assert!(
        applied["data"]["applied_items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["kind"] == "run_log")
    );
    assert_eq!(
        applied["data"]["apply_skipped"][0]["reason"],
        "文件内容与预览时不同"
    );
    assert!(!completed.join("worker.log").exists());
    assert_eq!(
        fs::read_to_string(completed.join("progress.jsonl")).unwrap(),
        "changed after preview\n"
    );
    assert!(completed.join("run.json").is_file());
    assert!(completed.join("events.jsonl").is_file());
    assert!(kept.join("worker.log").is_file());
    assert!(kept.join("progress.jsonl").is_file());
    assert!(active.join("worker.log").is_file());
}

#[test]
fn external_loop_observes_budget_and_saves_checkpoint() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    fs::write(
        project.path().join("external_loop.py"),
        r#"
import torch

def run_loop(model, context):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    while context.remaining_steps:
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(2, 4)).square().mean()
        loss.backward()
        optimizer.step()
        if not context.report_step(loss.detach(), {"updates": 1}):
            break
    return {"finished": True}

run_loop.MODEL_MAVEN_STATELESS = True
"#,
    )
    .unwrap();

    let output = invoke(
        project.path(),
        &[
            "train",
            "--model",
            "fixture:build_classifier",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop:run_loop",
            "--steps",
            "2",
            "--checkpoint-every",
            "1",
        ],
    );
    let result = assert_success(&output);
    assert_eq!(result["data"]["training_mode"], "external-loop");
    assert_eq!(result["data"]["completed_steps"], 2);
    assert_eq!(result["data"]["checkpoint_saved"], true);
    assert_eq!(result["data"]["loop_result"]["finished"], true);
    let run_id = result["data"]["run_id"].as_str().unwrap();
    let run_dir = project.path().join(".model-maven/runs").join(run_id);
    assert!(run_dir.join("checkpoints/latest.json").is_file());
}

#[cfg(unix)]
#[test]
fn workbench_probe_recovery_changes_batch_and_retries_the_forward_pass() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());

    let transcript = run_workbench_pty(
        project.path(),
        b"1\n2\nfixture:build_editable\nfixture:small_batch\n\n\n\n3\nfixture:editable_batch\nback\nback\n8\n",
    );
    assert!(
        transcript.contains("前向失败 · 没有通过验证"),
        "{transcript}"
    );
    assert!(transcript.contains("3 更换批次"), "{transcript}");
    assert!(
        transcript.contains("新批次入口 module:callable"),
        "{transcript}"
    );
    assert!(
        transcript.contains("批次：fixture:editable_batch"),
        "{transcript}"
    );
    assert!(
        transcript.matches("前向失败 · 没有通过验证").count() == 1,
        "expected the new batch to make the retry succeed:\n{transcript}"
    );
}

#[test]
fn custom_multimodal_model_handles_dynamic_batches_shared_layers_and_named_losses() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/complex_acceptance");
    let destination = project.path().join("complex_acceptance");
    fs::create_dir(&destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            fs::copy(entry.path(), destination.join(entry.file_name())).unwrap();
        }
    }

    let probe = invoke(
        project.path(),
        &[
            "model",
            "probe",
            "--model",
            "complex_acceptance.model:build_model",
            "--batch",
            "complex_acceptance.data:changing_batch",
        ],
    );
    let probe = assert_success(&probe);
    let shared_calls: Vec<_> = probe["data"]["calls"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|call| call["paths"][0] == "model.fusion.shared_projection")
        .collect();
    assert_eq!(shared_calls.len(), 2);
    assert_eq!(shared_calls[0]["invocation_index"], 1);
    assert_eq!(shared_calls[1]["invocation_index"], 2);

    let training = invoke(
        project.path(),
        &[
            "try",
            "--model",
            "complex_acceptance.model:build_model",
            "--batch",
            "complex_acceptance.data:changing_batch",
            "--step",
            "complex_acceptance.training:multitask_step",
            "--steps",
            "4",
        ],
    );
    let training = assert_success(&training);
    assert_eq!(training["data"]["completed_steps"], 4);
    assert!(training["data"]["mean_losses"]["classification"].is_number());
    assert!(training["data"]["mean_losses"]["alignment"].is_number());
    assert!(training["data"]["mean_losses"]["temporal"].is_number());
    assert_eq!(training["data"]["mean_metrics"]["video_frames"], 3.0);
    assert_eq!(training["data"]["mean_metrics"]["text_tokens"], 4.0);
    assert_eq!(
        training["data"]["last_outputs"]["items"]["frame_logits"]["shape"],
        serde_json::json!([2, 4, 4])
    );

    let manual = invoke(
        project.path(),
        &[
            "try",
            "--training-mode",
            "manual-step",
            "--model",
            "complex_acceptance.model:build_model",
            "--batch",
            "complex_acceptance.data:changing_batch",
            "--step",
            "complex_acceptance.training:manual_multitask_step",
            "--steps",
            "4",
        ],
    );
    let manual = assert_success(&manual);
    assert_eq!(manual["data"]["training_mode"], "manual-step");
    assert_eq!(
        manual["data"]["optimizer_groups"],
        serde_json::json!(["encoders", "fusion_heads"])
    );
    assert_eq!(manual["data"]["optimizer_updates_by_group"]["encoders"], 4);
    assert_eq!(
        manual["data"]["optimizer_updates_by_group"]["fusion_heads"],
        4
    );
    assert_eq!(manual["data"]["mean_metrics"]["video_trainable"], 0.5);
    assert!(manual["data"]["mean_losses"]["classification"].is_number());
    assert!(manual["data"]["mean_losses"]["alignment"].is_number());
    assert!(manual["data"]["mean_losses"]["temporal"].is_number());

    let overlap = invoke(
        project.path(),
        &[
            "try",
            "--training-mode",
            "manual-step",
            "--model",
            "complex_acceptance.model:build_model",
            "--batch",
            "complex_acceptance.data:changing_batch",
            "--step",
            "complex_acceptance.training:overlapping_optimizer_step",
            "--steps",
            "1",
        ],
    );
    let overlap = report(&overlap);
    assert_eq!(overlap["status"], "error");
    assert_eq!(overlap["diagnostics"][0]["code"], "STEP_FAILED");
    assert_eq!(overlap["data"]["cause_code"], "OPTIMIZER_PARAMETER_OVERLAP");
}

#[test]
fn complex_multimodal_contract_failure_can_be_repaired_and_reloaded_as_a_variant() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    copy_complex_fixture(project.path());
    let model = "complex_acceptance.contract_repair:build_model";
    let batch = "complex_acceptance.data:changing_batch";
    let contract_source = project.path().join("complex_acceptance/contract_repair.py");
    let original_source = fs::read(&contract_source).unwrap();

    let failed = report(&invoke(
        project.path(),
        &["model", "probe", "--model", model, "--batch", batch],
    ));
    assert_eq!(failed["diagnostics"][0]["code"], "FORWARD_FAILED");
    let failure = &failed["data"]["failure_location"];
    let failing_call = &failure["last_observed_module_call"];
    assert_eq!(failing_call["paths"][0], "model.fusion.video_projection");
    assert_eq!(
        failing_call["inputs"]["items"]["args"]["items"][0]["shape"],
        serde_json::json!([2, 48])
    );
    assert!(
        failure["project_stack"]
            .as_array()
            .unwrap()
            .iter()
            .any(|frame| {
                frame["file"] == "complex_acceptance/contract_repair.py"
                    && frame["function"] == "forward"
            })
    );

    let repaired = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "set",
            "model.fusion.video_projection",
            "--model",
            model,
            "--batch",
            batch,
            "--param",
            "in_features=48",
        ],
    ));
    assert_eq!(repaired["data"]["validation"]["forward"], "passed");
    assert_eq!(fs::read(&contract_source).unwrap(), original_source);

    let reloaded = assert_success(&invoke(
        project.path(),
        &[
            "--variant",
            "working",
            "model",
            "probe",
            "--model",
            model,
            "--batch",
            batch,
        ],
    ));
    assert_eq!(
        reloaded["data"]["outputs"]["items"]["logits"]["shape"],
        serde_json::json!([2, 4])
    );
    assert_eq!(fs::read(&contract_source).unwrap(), original_source);
}

#[test]
fn component_discovery_registers_static_package_metadata_without_importing_it() {
    let project = tempfile::tempdir().unwrap();
    let site = tempfile::tempdir().unwrap();
    let package = site.path().join("demo_ext/__model_maven__");
    let metadata = site.path().join("demo_model_maven-1.2.3.dist-info");
    fs::create_dir_all(&package).unwrap();
    fs::create_dir_all(&metadata).unwrap();
    let manifest_path = "demo_ext/__model_maven__/silu.json";
    let entrypoint = "demo_ext.layers:make_silu";
    let source_path = "demo_ext/layers.py";
    let manifest = serde_json::json!({
        "schema_version":1,
        "namespace":"example",
        "component_id":"example/silu",
        "component_version":"1.0.0",
        "extension_api_version":1,
        "kind":"layer",
        "entrypoint":entrypoint,
        "description":"Example extension layer",
        "params_schema":{"type":"object","properties":{},"required":[],"additionalProperties":false},
        "capabilities":{"construct":{"state":"supported"}}
    });
    fs::write(
        site.path().join(manifest_path),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    fs::write(
        site.path().join(source_path),
        "from pathlib import Path\nPath(__file__).resolve().parents[1].joinpath('imported.marker').write_text('imported')\nfrom torch import nn\ndef make_silu():\n    return nn.SiLU()\n",
    )
    .unwrap();
    fs::write(
        metadata.join("METADATA"),
        "Metadata-Version: 2.1\nName: demo-model-maven\nVersion: 1.2.3\n",
    )
    .unwrap();
    fs::write(
        metadata.join("entry_points.txt"),
        format!("[model_maven.components]\nsilu = {entrypoint}\n"),
    )
    .unwrap();
    fs::write(
        metadata.join("RECORD"),
        format!(
            "{manifest_path},,\n{source_path},,\ndemo_model_maven-1.2.3.dist-info/METADATA,,\ndemo_model_maven-1.2.3.dist-info/entry_points.txt,,\ndemo_model_maven-1.2.3.dist-info/RECORD,,\n"
        ),
    )
    .unwrap();

    let site_path = site.path().to_string_lossy().into_owned();
    let first = invoke(
        project.path(),
        &["component", "discover", "--path", &site_path],
    );
    let first = assert_success(&first);
    assert_eq!(first["data"]["found"], 1);
    assert_eq!(first["data"]["registered"].as_array().unwrap().len(), 1);
    assert!(!site.path().join("imported.marker").exists());

    let shown = assert_success(&invoke(
        project.path(),
        &["component", "show", "example/silu@1.0.0"],
    ));
    assert_eq!(
        shown["data"]["component"]["manifest"]["package_source"]["distribution"],
        "demo-model-maven"
    );
    assert_eq!(
        shown["data"]["component"]["manifest"]["package_source"]["module_sources"]["entrypoint"]["file"],
        source_path
    );

    let second = assert_success(&invoke(
        project.path(),
        &["component", "discover", "--path", &site_path],
    ));
    assert_eq!(second["data"]["found"], 1);
    assert_eq!(second["data"]["registered"].as_array().unwrap().len(), 0);
    assert_eq!(
        second["data"]["already_registered"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(!site.path().join("imported.marker").exists());

    let torch_available = has_torch();
    if torch_available {
        let runtime_test = assert_success(&invoke(
            project.path(),
            &["component", "test", "example/silu@1.0.0"],
        ));
        assert_eq!(
            runtime_test["data"]["module"]["type"],
            "torch.nn.modules.activation.SiLU"
        );
        assert!(site.path().join("imported.marker").exists());
        fs::remove_file(site.path().join("imported.marker")).unwrap();
    }

    fs::OpenOptions::new()
        .append(true)
        .open(site.path().join(source_path))
        .unwrap()
        .write_all(b"\n# changed source\n")
        .unwrap();
    let conflict = assert_success(&invoke(
        project.path(),
        &["component", "discover", "--path", &site_path],
    ));
    assert_eq!(conflict["data"]["registered"].as_array().unwrap().len(), 0);
    assert_eq!(conflict["data"]["issues"].as_array().unwrap().len(), 1);
    assert!(
        conflict["data"]["issues"][0]["message"]
            .as_str()
            .unwrap()
            .contains("未覆盖")
    );
    assert!(!site.path().join("imported.marker").exists());

    if torch_available {
        let stale_source = report(&invoke(
            project.path(),
            &["component", "test", "example/silu@1.0.0"],
        ));
        assert_eq!(stale_source["status"], "error");
        assert_eq!(
            stale_source["diagnostics"][0]["code"],
            "COMPONENT_PACKAGE_SOURCE_MISMATCH"
        );
        assert!(!site.path().join("imported.marker").exists());
    }

    let mut invalid_manifest = manifest.clone();
    invalid_manifest["capabilities"]["construct"]["state"] = serde_json::json!("unsupported");
    fs::write(
        site.path().join(manifest_path),
        serde_json::to_vec_pretty(&invalid_manifest).unwrap(),
    )
    .unwrap();
    let invalid = assert_success(&invoke(
        project.path(),
        &["component", "discover", "--path", &site_path],
    ));
    assert_eq!(invalid["data"]["found"], 0);
    assert!(
        invalid["data"]["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| {
                issue["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("capabilities.construct"))
            })
    );
    assert!(!site.path().join("imported.marker").exists());
}

#[test]
fn component_dependencies_resolve_semver_ranges_transitively_and_lock_variants() {
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());

    let register = |id: &str, version: &str, dependencies: Option<String>| {
        let mut arguments = vec![
            "component".to_owned(),
            "add".to_owned(),
            id.to_owned(),
            "--kind".to_owned(),
            "layer".to_owned(),
            "--entrypoint".to_owned(),
            "fixture:build_custom_activation".to_owned(),
            "--version".to_owned(),
            version.to_owned(),
        ];
        if let Some(dependencies) = dependencies {
            arguments.push("--dependencies".to_owned());
            arguments.push(dependencies);
        }
        let args = arguments.iter().map(String::as_str).collect::<Vec<_>>();
        assert_success(&invoke(project.path(), &args))
    };

    assert_success(&invoke(
        project.path(),
        &[
            "component",
            "add",
            "demo/root",
            "--kind",
            "layer",
            "--entrypoint",
            "fixture:build_custom_activation",
            "--version",
            "1.0.0",
            "--dependency",
            "demo/encoder@^1.0.0",
            "--dependency",
            "demo/z-head@^1.0.0",
            "--optional-dependency",
            "demo/optional-kernel@^3.0.0",
        ],
    ));
    register(
        "demo/encoder",
        "1.0.0",
        Some(
            serde_json::json!([
                {"component_id":"demo/a-kernel","version":"^1.0.0"}
            ])
            .to_string(),
        ),
    );
    register(
        "demo/z-head",
        "1.0.0",
        Some(
            serde_json::json!([
                {"component_id":"demo/a-kernel","version":"<1.5.0"}
            ])
            .to_string(),
        ),
    );
    register("demo/a-kernel", "1.6.0", None);
    register("demo/a-kernel", "1.4.0", None);

    let checked = assert_success(&invoke(
        project.path(),
        &["component", "check", "demo/root"],
    ));
    assert_eq!(checked["data"]["dependency_resolution"], "passed");
    let components = checked["data"]["dependencies"]["components"]
        .as_array()
        .unwrap();
    let resolved_version = |id: &str| {
        components
            .iter()
            .find(|component| component["id"] == id)
            .unwrap_or_else(|| panic!("dependency {id} was not resolved: {components:#?}"))
            ["version"]
            .as_str()
            .unwrap()
    };
    assert_eq!(resolved_version("demo/a-kernel"), "1.4.0");
    assert_eq!(resolved_version("demo/encoder"), "1.0.0");
    assert_eq!(resolved_version("demo/z-head"), "1.0.0");
    assert_eq!(
        checked["data"]["dependencies"]["optional_skipped"][0]["status"],
        "not_requested"
    );

    register(
        "demo/old-client",
        "1.0.0",
        Some(
            serde_json::json!([
                {"component_id":"demo/a-kernel","version":"<1.5.0"}
            ])
            .to_string(),
        ),
    );
    register(
        "demo/new-client",
        "1.0.0",
        Some(
            serde_json::json!([
                {"component_id":"demo/a-kernel","version":"^2.0.0"}
            ])
            .to_string(),
        ),
    );
    register("demo/a-kernel", "2.0.0", None);
    register(
        "demo/conflict",
        "1.0.0",
        Some(
            serde_json::json!([
                {"component_id":"demo/old-client","version":"^1.0.0"},
                {"component_id":"demo/new-client","version":"^1.0.0"}
            ])
            .to_string(),
        ),
    );
    let conflict = report(&invoke(
        project.path(),
        &["component", "check", "demo/conflict"],
    ));
    assert_eq!(
        conflict["diagnostics"][0]["code"],
        "COMPONENT_DEPENDENCY_RESOLUTION_FAILED"
    );
    let conflict_message = conflict["diagnostics"][0]["message"].as_str().unwrap();
    assert!(conflict_message.contains("demo/a-kernel"));
    assert!(conflict_message.contains("约束"));

    let broken = register(
        "demo/broken",
        "1.0.0",
        Some(
            serde_json::json!([
                {"component_id":"missing/base","version":"^1.0.0"}
            ])
            .to_string(),
        ),
    );
    assert_eq!(
        broken["data"]["component"]["manifest"]["component_id"],
        "demo/broken"
    );
    let missing = report(&invoke(
        project.path(),
        &["component", "check", "demo/broken"],
    ));
    assert_eq!(
        missing["diagnostics"][0]["code"],
        "COMPONENT_DEPENDENCY_RESOLUTION_FAILED"
    );
    assert!(
        missing["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains("missing/base")
    );

    register(
        "cycle/a",
        "1.0.0",
        Some(
            serde_json::json!([
                {"component_id":"cycle/b","version":"^1.0.0"}
            ])
            .to_string(),
        ),
    );
    register(
        "cycle/b",
        "1.0.0",
        Some(
            serde_json::json!([
                {"component_id":"cycle/a","version":"^1.0.0"}
            ])
            .to_string(),
        ),
    );
    let cycle = report(&invoke(project.path(), &["component", "check", "cycle/a"]));
    assert_eq!(
        cycle["diagnostics"][0]["code"],
        "COMPONENT_DEPENDENCY_RESOLUTION_FAILED"
    );
    assert!(
        cycle["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains("cycle/a -> cycle/b -> cycle/a")
    );

    if has_torch() {
        let applied = assert_success(&invoke(
            project.path(),
            &[
                "layer",
                "add",
                "--model",
                "fixture:build_editable",
                "--parent",
                "model.encoder",
                "--name",
                "resolved",
                "--component",
                "demo/root",
            ],
        ));
        assert_eq!(applied["data"]["committed"], true);
        let variant_path = project.path().join(".model-maven/variants/working.json");
        let variant: Value = serde_json::from_slice(&fs::read(&variant_path).unwrap()).unwrap();
        let locked = variant["edits"][0]["component_ref"]["dependencies"]
            .as_array()
            .unwrap();
        assert_eq!(locked.len(), 3);
        assert!(
            locked
                .iter()
                .any(|item| item["id"] == "demo/a-kernel" && item["version"] == "1.4.0")
        );
        assert!(locked.iter().any(|item| item["id"] == "demo/encoder"));
        assert!(locked.iter().any(|item| item["id"] == "demo/z-head"));

        let dependency_manifest = project
            .path()
            .join(".model-maven/components/demo/a-kernel/1.4.0.json");
        let mut changed: Value =
            serde_json::from_slice(&fs::read(&dependency_manifest).unwrap()).unwrap();
        changed["description"] = serde_json::json!("changed after variant creation");
        fs::write(
            &dependency_manifest,
            serde_json::to_vec_pretty(&changed).unwrap(),
        )
        .unwrap();
        let replay = report(&invoke(
            project.path(),
            &[
                "layer",
                "list",
                "--model",
                "fixture:build_editable",
                "--variant",
                "working",
            ],
        ));
        assert_eq!(
            replay["diagnostics"][0]["code"],
            "COMPONENT_VERSION_CHANGED"
        );
    }
}

#[test]
fn versioned_local_provider_can_edit_a_custom_container_and_replay_after_reload() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let operation = serde_json::json!({
        "action": "custom",
        "provider": "fixture:add_custom_activation",
        "payload": {"name": "silu"}
    })
    .to_string();
    let applied = invoke(
        project.path(),
        &[
            "layer",
            "apply",
            "--model",
            "fixture:build_flexible",
            "--batch",
            "fixture:editable_batch",
            "--operation",
            &operation,
        ],
    );
    let applied = assert_success(&applied);
    assert_eq!(applied["data"]["validation"]["forward"], "passed");

    let shown = invoke(
        project.path(),
        &["layer", "show", "model.encoder.layers.silu"],
    );
    let shown = assert_success(&shown);
    assert_eq!(shown["data"]["modules"][0]["type"], "SiLU");

    let undo = invoke(project.path(), &["layer", "undo"]);
    assert_success(&undo);
    let missing = invoke(
        project.path(),
        &["layer", "show", "model.encoder.layers.silu"],
    );
    let missing = report(&missing);
    assert_eq!(missing["diagnostics"][0]["code"], "LAYER_NOT_FOUND");
}

#[test]
fn registered_layer_edit_provider_requires_matching_runtime_verification() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    fs::OpenOptions::new()
        .append(true)
        .open(project.path().join("fixture.py"))
        .unwrap()
        .write_all(
            b"\ndef provider_fixture():\n    return {\n        'model': build_flexible(),\n        'expected_result': {'added': 'model.encoder.layers.silu', 'type': 'SiLU'},\n        'expected_modules': {'encoder.layers.silu': 'SiLU'},\n    }\n",
        )
        .unwrap();

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}"#;
    let registered = invoke(
        project.path(),
        &[
            "component",
            "add",
            "guarded/add-silu",
            "--kind",
            "layer-edit-provider",
            "--entrypoint",
            "fixture:add_custom_activation",
            "--version",
            "1.0.0",
            "--params-schema",
            schema,
            "--capabilities",
            r#"{"layer_edit":{"state":"supported"}}"#,
            "--conformance-fixture",
            "fixture:provider_fixture",
        ],
    );
    assert_success(&registered);

    let component_before_test = assert_success(&invoke(
        project.path(),
        &["component", "show", "guarded/add-silu"],
    ));
    assert_eq!(
        component_before_test["data"]["edit_verification"]["status"],
        "unverified"
    );

    let custom = |name: &str| {
        invoke(
            project.path(),
            &[
                "layer",
                "custom",
                "guarded/add-silu",
                "--payload",
                &format!("{{\"name\":\"{name}\"}}"),
                "--model",
                "fixture:build_flexible",
                "--batch",
                "fixture:editable_batch",
            ],
        )
    };

    let unverified = report(&custom("silu"));
    assert_eq!(unverified["status"], "error");
    assert_eq!(
        unverified["diagnostics"][0]["code"],
        "COMPONENT_PROVIDER_UNVERIFIED"
    );

    let tested = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "test",
            "guarded/add-silu",
            "--param",
            "name=\"silu\"",
        ],
    ));
    assert_eq!(
        tested["data"]["activation"]["status"],
        "enabled_for_tested_payloads"
    );
    let component_after_test = assert_success(&invoke(
        project.path(),
        &["component", "show", "guarded/add-silu"],
    ));
    assert_eq!(
        component_after_test["data"]["edit_verification"]["status"],
        "records_present"
    );
    assert_eq!(
        component_after_test["data"]["edit_verification"]["payload_count"],
        1
    );
    assert!(
        project
            .path()
            .join(".model-maven/component-verifications/guarded/add-silu/1.0.0.json")
            .is_file()
    );

    let preview = report(&invoke(
        project.path(),
        &[
            "layer",
            "custom",
            "guarded/add-silu",
            "--payload",
            r#"{"name":"silu"}"#,
            "--model",
            "fixture:build_flexible",
            "--batch",
            "fixture:editable_batch",
            "--dry-run",
        ],
    ));
    assert_eq!(preview["status"], "success", "{preview:#}");
    assert_eq!(
        preview["data"]["readiness"]["readiness"]["facts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|fact| fact["key"] == "layer_provider_verification")
            .unwrap()["state"],
        "conditional"
    );

    let wrong_payload = report(&custom("relu"));
    assert_eq!(wrong_payload["status"], "error");
    assert_eq!(
        wrong_payload["diagnostics"][0]["code"],
        "COMPONENT_PROVIDER_UNVERIFIED"
    );
    assert_eq!(
        wrong_payload["data"]["readiness"]["readiness"]["facts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|fact| fact["key"] == "layer_provider_verification")
            .unwrap()["state"],
        "missing"
    );
    fs::OpenOptions::new()
        .append(true)
        .open(project.path().join("fixture.py"))
        .unwrap()
        .write_all(b"\n# fixture source changed after verification\n")
        .unwrap();
    let stale_source = report(&custom("silu"));
    assert_eq!(stale_source["status"], "error");
    assert_eq!(
        stale_source["diagnostics"][0]["code"],
        "COMPONENT_PROVIDER_UNVERIFIED"
    );
    assert_eq!(
        stale_source["data"]["provider_verification"]["state"],
        "stale"
    );
    assert!(
        stale_source["data"]["provider_verification"]["mismatch_fields"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("source_fingerprints"))
    );

    let refreshed = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "test",
            "guarded/add-silu",
            "--param",
            "name=\"silu\"",
        ],
    ));
    assert_eq!(
        refreshed["data"]["activation"]["status"],
        "enabled_for_tested_payloads"
    );
    let applied = assert_success(&custom("silu"));
    assert_eq!(applied["data"]["validation"]["forward"], "passed");

    let shown = assert_success(&invoke(
        project.path(),
        &["layer", "show", "model.encoder.layers.silu"],
    ));
    assert_eq!(shown["data"]["modules"][0]["type"], "SiLU");
    assert_success(&invoke(project.path(), &["layer", "undo"]));
    let undone = report(&invoke(
        project.path(),
        &["layer", "show", "model.encoder.layers.silu"],
    ));
    assert_eq!(undone["diagnostics"][0]["code"], "LAYER_NOT_FOUND");
    assert_success(&invoke(project.path(), &["layer", "redo"]));
    let redone = assert_success(&invoke(
        project.path(),
        &["layer", "show", "model.encoder.layers.silu"],
    ));
    assert_eq!(redone["data"]["modules"][0]["type"], "SiLU");
}

#[test]
fn registered_provider_condition_states_gate_edit_execution() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    fs::OpenOptions::new()
        .append(true)
        .open(project.path().join("fixture.py"))
        .unwrap()
        .write_all(
            b"\ndef provider_fixture():\n    return {\n        'model': build_flexible(),\n        'expected_result': {'added': 'model.encoder.layers.silu', 'type': 'SiLU'},\n        'expected_modules': {'encoder.layers.silu': 'SiLU'},\n    }\n",
        )
        .unwrap();
    let schema = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}"#;
    let conditional = invoke(
        project.path(),
        &[
            "component",
            "add",
            "guarded/conditional",
            "--kind",
            "layer-edit-provider",
            "--entrypoint",
            "fixture:add_custom_activation",
            "--params-schema",
            schema,
            "--capabilities",
            r#"{"layer_edit":{"state":"conditional","reason":"Requires a model-specific adapter."}}"#,
            "--conformance-fixture",
            "fixture:provider_fixture",
        ],
    );
    assert_success(&conditional);

    let conditional_test = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "test",
            "guarded/conditional",
            "--param",
            "name=\"silu\"",
        ],
    ));
    assert_eq!(
        conditional_test["data"]["checks"]["layer_edit"]["status"],
        "passed"
    );
    assert_eq!(
        conditional_test["data"]["activation"]["status"],
        "not_enabled"
    );
    let conditional_edit = report(&invoke(
        project.path(),
        &[
            "layer",
            "custom",
            "guarded/conditional",
            "--payload",
            r#"{"name":"silu"}"#,
        ],
    ));
    assert_eq!(conditional_edit["status"], "error");
    assert_eq!(
        conditional_edit["diagnostics"][0]["code"],
        "COMPONENT_PROVIDER_CONDITION_UNVERIFIED"
    );

    let mismatched = invoke(
        project.path(),
        &[
            "component",
            "add",
            "guarded/mismatch",
            "--kind",
            "layer-edit-provider",
            "--entrypoint",
            "fixture:add_custom_activation",
            "--params-schema",
            schema,
            "--capabilities",
            r#"{"layer_edit":{"state":"supported"}}"#,
            "--conformance-fixture",
            "fixture:provider_fixture",
        ],
    );
    assert_success(&mismatched);
    let mismatch_test = report(&invoke(
        project.path(),
        &[
            "component",
            "test",
            "guarded/mismatch",
            "--param",
            "name=\"relu\"",
        ],
    ));
    assert_eq!(mismatch_test["status"], "error");
    assert_eq!(
        mismatch_test["diagnostics"][0]["code"],
        "COMPONENT_FIXTURE_MISMATCH"
    );
    let mismatch_edit = report(&invoke(
        project.path(),
        &[
            "layer",
            "custom",
            "guarded/mismatch",
            "--payload",
            r#"{"name":"relu"}"#,
        ],
    ));
    assert_eq!(mismatch_edit["status"], "error");
    assert_eq!(
        mismatch_edit["diagnostics"][0]["code"],
        "COMPONENT_PROVIDER_UNVERIFIED"
    );

    let unsupported = invoke(
        project.path(),
        &[
            "component",
            "add",
            "guarded/unsupported",
            "--kind",
            "layer-edit-provider",
            "--entrypoint",
            "fixture:provider_not_available",
            "--params-schema",
            schema,
            "--capabilities",
            r#"{"layer_edit":{"state":"unsupported","reason":"Not an edit provider."}}"#,
        ],
    );
    assert_success(&unsupported);
    let unsupported_test = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "test",
            "guarded/unsupported",
            "--param",
            "name=\"silu\"",
        ],
    ));
    assert_eq!(
        unsupported_test["data"]["checks"]["layer_edit"]["status"],
        "skipped"
    );
    let unsupported_edit = report(&invoke(
        project.path(),
        &[
            "layer",
            "custom",
            "guarded/unsupported",
            "--payload",
            r#"{"name":"silu"}"#,
        ],
    ));
    assert_eq!(unsupported_edit["status"], "error");
    assert_eq!(
        unsupported_edit["diagnostics"][0]["code"],
        "COMPONENT_PROVIDER_UNSUPPORTED"
    );
}

#[test]
fn persisted_variant_crud_preserves_custom_torch_container_subclasses() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let source_path = project.path().join("fixture.py");
    let source_before = fs::read(&source_path).unwrap();

    let inspected = assert_success(&invoke(
        project.path(),
        &[
            "model",
            "inspect",
            "--model",
            "fixture:build_custom_sequential",
        ],
    ));
    assert!(
        inspected["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|module| module["path"] == "model.encoder"
                && module["type"] == "CustomSequential"
                && module["can_insert"] == true)
    );

    let added = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "add",
            "--model",
            "fixture:build_custom_sequential",
            "--batch",
            "fixture:custom_sequential_batch",
            "--target",
            "variant",
            "--parent",
            "model.encoder",
            "--after",
            "relu",
            "--name",
            "dropout",
            "--type",
            "Dropout",
            "--param",
            "p=0.2",
        ],
    ));
    let added_modules = added["data"]["modules"].as_array().unwrap();
    assert!(added_modules.iter().any(|module| {
        module["path"] == "model.encoder" && module["type"] == "CustomSequential"
    }));
    assert!(
        added_modules
            .iter()
            .any(|module| module["path"] == "model.encoder.dropout")
    );

    let removed = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "remove",
            "model.encoder.dropout",
            "--model",
            "fixture:build_custom_sequential",
            "--batch",
            "fixture:custom_sequential_batch",
            "--target",
            "variant",
        ],
    ));
    let removed_modules = removed["data"]["modules"].as_array().unwrap();
    assert!(removed_modules.iter().any(|module| {
        module["path"] == "model.encoder" && module["type"] == "CustomSequential"
    }));
    assert!(
        !removed_modules
            .iter()
            .any(|module| module["path"] == "model.encoder.dropout")
    );

    for (variant, model, batch, container_type, anchor, name, inserted_path) in [
        (
            "custom-module-list",
            "fixture:build_custom_module_list",
            "fixture:custom_module_list_batch",
            "CustomModuleList",
            "0",
            None,
            "model.encoder.1",
        ),
        (
            "custom-module-dict",
            "fixture:build_custom_module_dict",
            "fixture:custom_module_dict_batch",
            "CustomModuleDict",
            "relu",
            Some("dropout"),
            "model.encoder.dropout",
        ),
    ] {
        let inspected = assert_success(&invoke(
            project.path(),
            &["model", "inspect", "--model", model],
        ));
        assert!(inspected["data"]["modules"].as_array().unwrap().iter().any(
            |module| module["path"] == "model.encoder"
                && module["type"] == container_type
                && module["can_insert"] == true
        ));

        let mut add_args = vec![
            "--variant",
            variant,
            "layer",
            "add",
            "--model",
            model,
            "--batch",
            batch,
            "--target",
            "variant",
            "--parent",
            "model.encoder",
            "--after",
            anchor,
        ];
        if let Some(name) = name {
            add_args.extend(["--name", name]);
        }
        add_args.extend(["--type", "Dropout", "--param", "p=0.2"]);
        let added = assert_success(&invoke(project.path(), &add_args));
        let added_modules = added["data"]["modules"].as_array().unwrap();
        assert!(added_modules.iter().any(|module| {
            module["path"] == "model.encoder" && module["type"] == container_type
        }));
        assert!(
            added_modules
                .iter()
                .any(|module| module["path"] == inserted_path)
        );

        let removed = assert_success(&invoke(
            project.path(),
            &[
                "--variant",
                variant,
                "layer",
                "remove",
                inserted_path,
                "--model",
                model,
                "--batch",
                batch,
                "--target",
                "variant",
            ],
        ));
        let removed_modules = removed["data"]["modules"].as_array().unwrap();
        assert!(removed_modules.iter().any(|module| {
            module["path"] == "model.encoder" && module["type"] == container_type
        }));
        assert!(
            !removed_modules
                .iter()
                .any(|module| { module["path"] == inserted_path && module["type"] == "Dropout" })
        );
    }

    assert_eq!(fs::read(source_path).unwrap(), source_before);
}

#[test]
fn layer_variant_crud_is_persistent_undoable_and_transactional() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let source_path = project.path().join("fixture.py");
    let source_before = fs::read(&source_path).unwrap();

    let dry_run = invoke(
        project.path(),
        &[
            "layer",
            "add",
            "--model",
            "fixture:build_editable",
            "--batch",
            "fixture:editable_batch",
            "--parent",
            "model.encoder",
            "--after",
            "relu",
            "--name",
            "dropout",
            "--type",
            "Dropout",
            "--param",
            "p=0.2",
            "--dry-run",
        ],
    );
    let dry_run = assert_success(&dry_run);
    assert_eq!(dry_run["data"]["committed"], false);
    assert!(!project.path().join(".model-maven").exists());

    let added = invoke(
        project.path(),
        &[
            "layer",
            "add",
            "--model",
            "fixture:build_editable",
            "--batch",
            "fixture:editable_batch",
            "--parent",
            "model.encoder",
            "--after",
            "relu",
            "--name",
            "dropout",
            "--type",
            "Dropout",
            "--param",
            "p=0.2",
        ],
    );
    assert_success(&added);

    let variant_path = project.path().join(".model-maven/variants/working.json");
    let first = serde_json::from_slice::<Value>(&fs::read(&variant_path).unwrap()).unwrap();
    assert_eq!(first["cursor"], 1);
    assert!(
        first["edits"][0]["initialization_seed"].as_u64().is_some(),
        "new structural edits must persist the seed used to initialize their weights"
    );

    let set = invoke(
        project.path(),
        &[
            "layer",
            "set",
            "model.encoder.dropout",
            "--batch",
            "fixture:editable_batch",
            "--param",
            "p=0.1",
        ],
    );
    assert_success(&set);
    let before_bad_edit = fs::read(&variant_path).unwrap();
    let revision = serde_json::from_slice::<Value>(&before_bad_edit).unwrap()["revision"].clone();

    let rejected = invoke(
        project.path(),
        &[
            "layer",
            "set",
            "model.encoder.proj",
            "--batch",
            "fixture:editable_batch",
            "--param",
            "out_features=16",
        ],
    );
    let rejected_report = report(&rejected);
    assert!(!rejected.status.success());
    assert_eq!(
        rejected_report["diagnostics"][0]["code"],
        "EDIT_VALIDATION_FAILED"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(&variant_path).unwrap()).unwrap()["revision"],
        revision,
        "a failed forward must leave the saved variant untouched"
    );

    let projection = serde_json::json!({
        "action": "set",
        "path": "model.encoder.proj",
        "params": {"out_features": 16}
    })
    .to_string();
    let wrong_head = serde_json::json!({
        "action": "set",
        "path": "model.head",
        "params": {"in_features": 32}
    })
    .to_string();
    let failed_transaction = invoke(
        project.path(),
        &[
            "layer",
            "apply",
            "--operation",
            &projection,
            "--operation",
            &wrong_head,
            "--batch",
            "fixture:editable_batch",
        ],
    );
    let failed_transaction = report(&failed_transaction);
    assert_eq!(
        failed_transaction["diagnostics"][0]["code"],
        "EDIT_VALIDATION_FAILED"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(&variant_path).unwrap()).unwrap()["revision"],
        revision,
        "a failed multi-layer candidate must not partially commit"
    );

    let head = serde_json::json!({
        "action": "set",
        "path": "model.head",
        "params": {"in_features": 16}
    })
    .to_string();
    let transaction = invoke(
        project.path(),
        &[
            "layer",
            "apply",
            "--operation",
            &projection,
            "--operation",
            &head,
            "--batch",
            "fixture:editable_batch",
        ],
    );
    let transaction = assert_success(&transaction);
    assert_eq!(transaction["data"]["validation"]["forward"], "passed");
    assert_eq!(
        transaction["data"]["validation"]["outputs"]["shape"],
        serde_json::json!([2, 4])
    );

    let undo_transaction = invoke(project.path(), &["layer", "undo"]);
    assert_success(&undo_transaction);
    let restored_projection = invoke(project.path(), &["layer", "show", "model.encoder.proj"]);
    let restored_projection = assert_success(&restored_projection);
    assert_eq!(
        restored_projection["data"]["modules"][0]["editable_params"]["out_features"],
        8
    );
    let restored_head = invoke(project.path(), &["layer", "show", "model.head"]);
    let restored_head = assert_success(&restored_head);
    assert_eq!(
        restored_head["data"]["modules"][0]["editable_params"]["in_features"],
        8
    );

    let undo = invoke(project.path(), &["layer", "undo"]);
    assert_success(&undo);
    let inspected = invoke(project.path(), &["layer", "list", "model.encoder"]);
    let inspected = assert_success(&inspected);
    let dropout = inspected["data"]["modules"]
        .as_array()
        .unwrap()
        .iter()
        .find(|module| module["path"] == "model.encoder.dropout")
        .unwrap();
    assert_eq!(dropout["editable_params"]["p"], 0.2);

    let redo = invoke(project.path(), &["layer", "redo"]);
    assert_success(&redo);
    let redo_transaction = invoke(project.path(), &["layer", "redo"]);
    assert_success(&redo_transaction);
    let replaced = invoke(
        project.path(),
        &[
            "layer",
            "replace",
            "model.encoder.dropout",
            "--component",
            "fixture:build_custom_activation",
            "--batch",
            "fixture:editable_batch",
        ],
    );
    assert_success(&replaced);
    let after_replace = invoke(project.path(), &["layer", "show", "model.encoder.dropout"]);
    let after_replace = assert_success(&after_replace);
    assert_eq!(after_replace["data"]["modules"][0]["type"], "SiLU");

    let built_in_replace = invoke(
        project.path(),
        &[
            "layer",
            "replace",
            "model.encoder.dropout",
            "--type",
            "GELU",
            "--batch",
            "fixture:editable_batch",
        ],
    );
    assert_success(&built_in_replace);
    let after_built_in_replace =
        invoke(project.path(), &["layer", "show", "model.encoder.dropout"]);
    let after_built_in_replace = assert_success(&after_built_in_replace);
    assert_eq!(after_built_in_replace["data"]["modules"][0]["type"], "GELU");

    let removed = invoke(
        project.path(),
        &[
            "layer",
            "remove",
            "model.encoder.dropout",
            "--batch",
            "fixture:editable_batch",
        ],
    );
    assert_success(&removed);
    let final_list = invoke(project.path(), &["layer", "list", "model.encoder"]);
    let final_list = assert_success(&final_list);
    assert!(
        !final_list["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|module| module["path"] == "model.encoder.dropout")
    );

    let mut changed_source = source_before.clone();
    changed_source.extend_from_slice(b"\n# intentional source change for the guard check\n");
    fs::write(&source_path, changed_source).unwrap();
    let stale = invoke(project.path(), &["layer", "list"]);
    let stale = report(&stale);
    assert_eq!(stale["diagnostics"][0]["code"], "SOURCE_CHANGED");
    fs::write(&source_path, &source_before).unwrap();
    assert_eq!(fs::read(source_path).unwrap(), source_before);
}

#[test]
fn explicit_run_seed_is_independent_from_a_persistent_variant_seed() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let added = invoke(
        project.path(),
        &[
            "layer",
            "add",
            "--model",
            "fixture:build_editable",
            "--batch",
            "fixture:editable_batch",
            "--parent",
            "model.encoder",
            "--after",
            "relu",
            "--name",
            "dropout",
            "--type",
            "Dropout",
            "--param",
            "p=0.2",
        ],
    );
    assert_success(&added);
    let variant_path = project.path().join(".model-maven/variants/working.json");
    let original_variant: Value =
        serde_json::from_slice(&fs::read(&variant_path).unwrap()).unwrap();
    assert_eq!(original_variant["seed"], 42);

    let run = |seed: &str| {
        invoke(
            project.path(),
            &[
                "--variant",
                "working",
                "try",
                "--seed",
                seed,
                "--model",
                "fixture:build_editable",
                "--batch",
                "fixture:editable_batch",
                "--step",
                "fixture:editable_training_step",
                "--steps",
                "2",
            ],
        )
    };
    let first = assert_success(&run("1234"));
    let repeated = assert_success(&run("1234"));
    assert_eq!(first["data"]["seed"], 1234);
    assert_eq!(
        first["data"]["mean_loss"], repeated["data"]["mean_loss"],
        "同一变体和运行种子应复现试跑随机性"
    );
    let different = assert_success(&run("5678"));
    assert_ne!(first["data"]["mean_loss"], different["data"]["mean_loss"]);
    let after_runs: Value = serde_json::from_slice(&fs::read(&variant_path).unwrap()).unwrap();
    assert_eq!(after_runs, original_variant, "试跑不能改写变体及其构造种子");

    let mismatched_edit = report(&invoke(
        project.path(),
        &[
            "--variant",
            "working",
            "--seed",
            "1234",
            "layer",
            "set",
            "model.encoder.dropout",
            "--batch",
            "fixture:editable_batch",
            "--param",
            "p=0.3",
        ],
    ));
    assert_eq!(mismatched_edit["status"], "error");
    assert_eq!(
        mismatched_edit["diagnostics"][0]["code"],
        "VARIANT_SEED_CONFLICT"
    );
    assert!(
        mismatched_edit["diagnostics"][0]["hint"]
            .as_str()
            .is_some_and(|hint| hint.contains("train/try") && hint.contains("--variant experiment"))
    );
}

#[test]
fn human_layer_edit_summary_explains_target_and_weight_initialization() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "layer",
            "add",
            "--model",
            "fixture:build_editable",
            "--batch",
            "fixture:editable_batch",
            "--parent",
            "model.encoder",
            "--after",
            "relu",
            "--name",
            "projection",
            "--type",
            "Linear",
            "--param",
            "in_features=8",
            "--param",
            "out_features=8",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Layer 新增：model.encoder.projection · Linear"),
        "{stdout}"
    );
    assert!(
        stdout.contains("新初始化参数：weight, bias；旧权重不会自动迁移"),
        "{stdout}"
    );
    assert!(!stdout.contains("\"action\":\"add\""), "{stdout}");
}

#[test]
fn source_set_transaction_updates_multiple_model_files_and_is_undoable() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("encoder.py"),
        "from torch import nn\n\nclass Encoder(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.proj = nn.Linear(8, 8)\n\n    def forward(self, values):\n        return self.proj(values)\n",
    )
    .unwrap();
    fs::write(
        project.path().join("head.py"),
        "from torch import nn\n\nclass Head(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.proj = nn.Linear(8, 4)\n\n    def forward(self, values):\n        return self.proj(values)\n",
    )
    .unwrap();
    fs::write(
        project.path().join("model.py"),
        "import torch\nfrom torch import nn\nfrom encoder import Encoder\nfrom head import Head\n\nclass Network(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.encoder = Encoder()\n        self.head = Head()\n\n    def forward(self, values):\n        return self.head(self.encoder(values))\n\ndef build_model():\n    return Network()\n\ndef example_batch():\n    return torch.randn(2, 8)\n",
    )
    .unwrap();
    let encoder_path = project.path().join("encoder.py");
    let head_path = project.path().join("head.py");
    let encoder_before = fs::read(&encoder_path).unwrap();
    let head_before = fs::read(&head_path).unwrap();
    let operations = [
        serde_json::json!({
            "action":"set",
            "path":"model.encoder.proj",
            "params":{"out_features":12}
        })
        .to_string(),
        serde_json::json!({
            "action":"set",
            "path":"model.head.proj",
            "params":{"in_features":12}
        })
        .to_string(),
    ];
    let mut args = vec![
        "layer",
        "apply",
        "--model",
        "model:build_model",
        "--batch",
        "model:example_batch",
        "--target",
        "source",
        "--require-forward",
        "--dry-run",
    ];
    for operation in &operations {
        args.push("--operation");
        args.push(operation);
    }
    let preview = assert_success(&invoke(project.path(), &args));
    assert_eq!(preview["data"]["committed"], false);
    assert_eq!(preview["data"]["source_paths"].as_array().unwrap().len(), 2);
    let diff = preview["data"]["source_patch"]["unified_diff"]
        .as_str()
        .unwrap();
    assert!(diff.contains("encoder.py"));
    assert!(diff.contains("head.py"));
    assert_eq!(fs::read(&encoder_path).unwrap(), encoder_before);
    assert_eq!(fs::read(&head_path).unwrap(), head_before);

    args.retain(|argument| *argument != "--dry-run");
    let source_plan = preview["data"]["source_plan"]["digest"]
        .as_str()
        .unwrap()
        .to_owned();
    args.extend(["--if-source-plan", &source_plan]);
    let mut concurrently_changed = head_before.clone();
    concurrently_changed.extend_from_slice(b"\n# concurrent change\n");
    fs::write(&head_path, &concurrently_changed).unwrap();
    let conflict = report(&invoke(project.path(), &args));
    assert_eq!(
        conflict["diagnostics"][0]["code"], "SOURCE_PLAN_CONFLICT",
        "{conflict:#}"
    );
    assert_eq!(fs::read(&encoder_path).unwrap(), encoder_before);
    assert_eq!(fs::read(&head_path).unwrap(), concurrently_changed);
    fs::write(&head_path, &head_before).unwrap();
    let committed = assert_success(&invoke(project.path(), &args));
    assert_eq!(committed["data"]["committed"], true);
    assert_eq!(
        committed["data"]["source_edit"]["files"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(
        String::from_utf8_lossy(&fs::read(&encoder_path).unwrap()).contains("nn.Linear(8, 12)")
    );
    assert!(String::from_utf8_lossy(&fs::read(&head_path).unwrap()).contains("nn.Linear(12, 4)"));

    assert_success(&invoke(
        project.path(),
        &["layer", "undo", "--target", "source"],
    ));
    assert_eq!(fs::read(&encoder_path).unwrap(), encoder_before);
    assert_eq!(fs::read(&head_path).unwrap(), head_before);
    assert_success(&invoke(
        project.path(),
        &["layer", "redo", "--target", "source"],
    ));
    assert!(
        String::from_utf8_lossy(&fs::read(&encoder_path).unwrap()).contains("nn.Linear(8, 12)")
    );
    assert!(String::from_utf8_lossy(&fs::read(&head_path).unwrap()).contains("nn.Linear(12, 4)"));
}

#[test]
fn source_undo_refuses_to_overwrite_external_changes() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    let source = project.path().join("fixture.py");
    let edited = invoke(
        project.path(),
        &[
            "layer",
            "set",
            "model.encoder.relu",
            "--model",
            "fixture:build_editable",
            "--target",
            "source",
            "--param",
            "inplace=true",
        ],
    );
    assert_success(&edited);
    let mut user_change = fs::read(&source).unwrap();
    user_change.extend_from_slice(b"\n# preserved user edit\n");
    fs::write(&source, &user_change).unwrap();

    let undo = report(&invoke(
        project.path(),
        &["layer", "undo", "--target", "source"],
    ));
    assert_eq!(undo["status"], "error", "{undo:#}");
    assert_eq!(undo["diagnostics"][0]["code"], "SOURCE_EDIT_HISTORY_FAILED");
    assert!(
        undo["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains("外部修改")
    );
    assert_eq!(fs::read(&source).unwrap(), user_change);
}

#[test]
fn source_set_supports_project_layer_constructor_parameters() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "import torch\nfrom torch import nn\n\nclass CustomScale(nn.Module):\n    def __init__(self, multiplier=2.0):\n        super().__init__()\n        self.multiplier = multiplier\n\n    def forward(self, values):\n        return values * self.multiplier\n\nclass WeightedScale(nn.Module):\n    def __init__(self, width=3):\n        super().__init__()\n        self.projection = nn.Linear(width, width)\n\n    def forward(self, values):\n        return self.projection(values)\n\nclass Network(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.custom = CustomScale(multiplier=2.0)\n        self.weighted = WeightedScale(width=3)\n\n    def forward(self, values):\n        return self.custom(values)\n\ndef build_model():\n    return Network()\n\ndef example_batch():\n    return torch.ones(2, 3)\n",
    )
    .unwrap();

    let details = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "show",
            "model.custom",
            "--model",
            "model:build_model",
        ],
    ));
    let module = &details["data"]["modules"][0];
    assert_eq!(module["source_editable_fields"][0], "multiplier");
    assert_eq!(module["can_source_set"], true);
    assert_eq!(module["can_set"], false);
    let selected_python = python();
    let human_details = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &selected_python,
            "layer",
            "show",
            "model.custom",
            "--model",
            "model:build_model",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human_details.status.success());
    let human_text = String::from_utf8_lossy(&human_details.stdout);
    assert!(
        human_text.contains("源码可改参数（--target source）"),
        "{human_text}"
    );

    let weighted = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "show",
            "model.weighted",
            "--model",
            "model:build_model",
        ],
    ));
    let weighted_module = &weighted["data"]["modules"][0];
    assert_eq!(weighted_module["can_source_set"], false);
    assert!(
        weighted_module["source_editable_fields"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        weighted_module["source_set_block_reason"]
            .as_str()
            .unwrap()
            .contains("状态迁移处理器")
    );
    let human_weighted = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &selected_python,
            "layer",
            "show",
            "model.weighted",
            "--model",
            "model:build_model",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human_weighted.status.success());
    let human_text = String::from_utf8_lossy(&human_weighted.stdout);
    assert!(
        human_text.contains("源码改参受限：该自定义 Layer 含参数或 Buffer"),
        "{human_text}"
    );

    let edited = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "set",
            "model.custom",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
            "--target",
            "source",
            "--param",
            "multiplier=3.0",
        ],
    ));
    assert_eq!(edited["data"]["validation"]["forward"], "passed");
    let source = fs::read_to_string(project.path().join("model.py")).unwrap();
    assert!(source.contains("CustomScale(multiplier=3.0)"), "{source}");
}

#[test]
fn source_structural_transaction_groups_crud_across_sequential_files() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("encoder.py"),
        "from collections import OrderedDict\nfrom torch import nn\n\nclass Encoder(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.Sequential(OrderedDict([\n            (\"proj\", nn.Linear(8, 8)),\n            (\"relu\", nn.ReLU()),\n        ]))\n        self.post = nn.Sequential(OrderedDict([\n            (\"act\", nn.ReLU()),\n        ]))\n\n    def forward(self, values):\n        return self.post(self.layers(values))\n",
    )
    .unwrap();
    fs::write(
        project.path().join("head.py"),
        "from collections import OrderedDict\nfrom torch import nn\n\nclass Head(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.Sequential(OrderedDict([\n            (\"proj\", nn.Linear(8, 4)),\n            (\"act\", nn.ReLU()),\n        ]))\n\n    def forward(self, values):\n        return self.layers(values)\n",
    )
    .unwrap();
    fs::write(
        project.path().join("model.py"),
        "import torch\nfrom torch import nn\nfrom encoder import Encoder\nfrom head import Head\n\nclass Network(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.encoder = Encoder()\n        self.head = Head()\n\n    def forward(self, values):\n        return self.head(self.encoder(values))\n\ndef build_model():\n    return Network()\n\ndef example_batch():\n    return torch.randn(2, 8)\n",
    )
    .unwrap();
    let encoder_path = project.path().join("encoder.py");
    let head_path = project.path().join("head.py");
    let encoder_before = fs::read(&encoder_path).unwrap();
    let head_before = fs::read(&head_path).unwrap();
    let operations = [
        serde_json::json!({
            "action":"add",
            "parent":"model.encoder.layers",
            "after":"relu",
            "name":"dropout",
            "layer_type":"Dropout",
            "params":{"p":0.15}
        })
        .to_string(),
        serde_json::json!({
            "action":"remove",
            "path":"model.encoder.layers.relu",
            "strategy":"remove"
        })
        .to_string(),
        serde_json::json!({
            "action":"replace",
            "path":"model.encoder.post.act",
            "layer_type":"GELU",
            "params":{}
        })
        .to_string(),
        serde_json::json!({
            "action":"replace",
            "path":"model.head.layers.act",
            "layer_type":"GELU",
            "params":{}
        })
        .to_string(),
    ];
    let mut args = vec![
        "layer",
        "apply",
        "--model",
        "model:build_model",
        "--batch",
        "model:example_batch",
        "--target",
        "source",
        "--dry-run",
    ];
    for operation in &operations {
        args.push("--operation");
        args.push(operation);
    }

    let preview = assert_success(&invoke(project.path(), &args));
    assert_eq!(preview["data"]["committed"], false);
    assert_eq!(preview["data"]["validation"]["forward"], "passed");
    assert_eq!(preview["data"]["source_paths"].as_array().unwrap().len(), 2);
    assert_eq!(preview["data"]["edit_effects"].as_array().unwrap().len(), 4);
    let diff = preview["data"]["source_patch"]["unified_diff"]
        .as_str()
        .unwrap();
    assert!(diff.contains("encoder.py"));
    assert!(diff.contains("head.py"));
    assert!(diff.contains("Dropout"));
    assert!(diff.contains("GELU"));
    assert_eq!(fs::read(&encoder_path).unwrap(), encoder_before);
    assert_eq!(fs::read(&head_path).unwrap(), head_before);

    args.retain(|argument| *argument != "--dry-run");
    let committed = assert_success(&invoke(project.path(), &args));
    assert_eq!(committed["data"]["committed"], true);
    assert_eq!(
        committed["data"]["source_edit"]["files"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let encoder_source = String::from_utf8_lossy(&fs::read(&encoder_path).unwrap()).into_owned();
    let head_source = String::from_utf8_lossy(&fs::read(&head_path).unwrap()).into_owned();
    assert!(encoder_source.contains("nn.Dropout(p=0.15)"));
    assert!(head_source.contains("nn.GELU()"));
    let modules = committed["data"]["modules"].as_array().unwrap();
    assert!(modules.iter().any(|module| {
        module["path"] == "model.encoder.layers.dropout" && module["type"] == "Dropout"
    }));
    assert!(
        !modules
            .iter()
            .any(|module| module["path"] == "model.encoder.layers.relu")
    );
    assert!(
        modules.iter().any(|module| {
            module["path"] == "model.encoder.post.act" && module["type"] == "GELU"
        })
    );
    assert!(
        modules.iter().any(|module| {
            module["path"] == "model.head.layers.act" && module["type"] == "GELU"
        })
    );

    assert_success(&invoke(
        project.path(),
        &["layer", "undo", "--target", "source"],
    ));
    assert_eq!(fs::read(&encoder_path).unwrap(), encoder_before);
    assert_eq!(fs::read(&head_path).unwrap(), head_before);
    assert_success(&invoke(
        project.path(),
        &["layer", "redo", "--target", "source"],
    ));
    let encoder_after_first_redo = fs::read(&encoder_path).unwrap();

    let followup_operations = [
        serde_json::json!({
            "action":"replace",
            "path":"model.encoder.layers.dropout",
            "layer_type":"ReLU",
            "params":{}
        })
        .to_string(),
        serde_json::json!({
            "action":"add",
            "parent":"model.encoder.layers",
            "after":"dropout",
            "name":"dropout2",
            "layer_type":"Dropout",
            "params":{"p":0.1}
        })
        .to_string(),
    ];
    let mut followup_args = vec![
        "layer",
        "apply",
        "--model",
        "model:build_model",
        "--batch",
        "model:example_batch",
        "--target",
        "source",
    ];
    for operation in &followup_operations {
        followup_args.push("--operation");
        followup_args.push(operation);
    }
    let followup = assert_success(&invoke(project.path(), &followup_args));
    assert_eq!(followup["data"]["source_validation"]["status"], "passed");
    let followup_modules = followup["data"]["modules"].as_array().unwrap();
    assert!(followup_modules.iter().any(|module| {
        module["path"] == "model.encoder.layers.dropout" && module["type"] == "ReLU"
    }));
    assert!(followup_modules.iter().any(|module| {
        module["path"] == "model.encoder.layers.dropout2" && module["type"] == "Dropout"
    }));
    let encoder_after_followup = fs::read(&encoder_path).unwrap();
    assert_eq!(
        String::from_utf8_lossy(&encoder_after_followup)
            .matches("# Rebuild this Sequential")
            .count(),
        3
    );
    assert_success(&invoke(
        project.path(),
        &["layer", "undo", "--target", "source"],
    ));
    assert_eq!(fs::read(&encoder_path).unwrap(), encoder_after_first_redo);
    assert_success(&invoke(
        project.path(),
        &["layer", "redo", "--target", "source"],
    ));
    assert_eq!(fs::read(&encoder_path).unwrap(), encoder_after_followup);

    let encoder_after_redo = fs::read(&encoder_path).unwrap();
    let head_after_redo = fs::read(&head_path).unwrap();
    let mixed_operations = [
        serde_json::json!({
            "action":"set",
            "path":"model.encoder.layers.proj",
            "params":{"out_features":8}
        })
        .to_string(),
        serde_json::json!({
            "action":"remove",
            "path":"model.head.layers.act",
            "strategy":"remove"
        })
        .to_string(),
    ];
    let mut mixed_args = vec![
        "layer",
        "apply",
        "--model",
        "model:build_model",
        "--target",
        "source",
    ];
    for operation in &mixed_operations {
        mixed_args.push("--operation");
        mixed_args.push(operation);
    }
    let rejected = report(&invoke(project.path(), &mixed_args));
    assert_eq!(
        rejected["diagnostics"][0]["code"],
        "SOURCE_EDIT_UNSUPPORTED"
    );
    assert_eq!(fs::read(&encoder_path).unwrap(), encoder_after_redo);
    assert_eq!(fs::read(&head_path).unwrap(), head_after_redo);
}

#[test]
fn source_modulelist_crud_uses_indexes_and_keeps_readable_python() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "import torch\nfrom torch import nn\n\nclass Network(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blocks = nn.ModuleList([\n            nn.Linear(4, 4),\n            nn.ReLU(),\n            nn.Identity(),\n            nn.Linear(4, 4),\n        ])\n\n    def forward(self, values):\n        for block in self.blocks:\n            values = block(values)\n        return values\n\ndef build_model():\n    return Network()\n\ndef example_batch():\n    return torch.randn(2, 4)\n",
    )
    .unwrap();
    let model_path = project.path().join("model.py");
    let original = fs::read(&model_path).unwrap();

    let listed = assert_success(&invoke(
        project.path(),
        &["layer", "list", "--model", "model:build_model"],
    ));
    assert!(
        listed["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| { row["path"] == "model.blocks" && row["can_source_insert"] == true })
    );

    let add = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "add",
            "--parent",
            "model.blocks",
            "--before",
            "1",
            "--type",
            "Dropout",
            "--param",
            "p=0.1",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
            "--target",
            "source",
            "--dry-run",
        ],
    ));
    assert_eq!(add["data"]["committed"], false);
    assert_eq!(add["data"]["validation"]["forward"], "passed");
    assert_eq!(fs::read(&model_path).unwrap(), original);

    let add = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "add",
            "--parent",
            "model.blocks",
            "--before",
            "1",
            "--type",
            "Dropout",
            "--param",
            "p=0.1",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
            "--target",
            "source",
        ],
    ));
    assert_eq!(add["data"]["committed"], true);
    let after_add = fs::read(&model_path).unwrap();
    let modules = add["data"]["modules"].as_array().unwrap();
    assert!(
        modules
            .iter()
            .any(|row| { row["path"] == "model.blocks.1" && row["type"] == "Dropout" })
    );

    let operations = [
        serde_json::json!({
            "action":"add",
            "parent":"model.blocks",
            "layer_type":"Dropout",
            "params":{"p":0.05}
        })
        .to_string(),
        serde_json::json!({
            "action":"replace",
            "path":"model.blocks.2",
            "layer_type":"GELU",
            "params":{}
        })
        .to_string(),
        serde_json::json!({
            "action":"remove",
            "path":"model.blocks.3",
            "strategy":"remove"
        })
        .to_string(),
    ];
    let mut args = vec![
        "layer",
        "apply",
        "--model",
        "model:build_model",
        "--batch",
        "model:example_batch",
        "--target",
        "source",
        "--dry-run",
    ];
    for operation in &operations {
        args.push("--operation");
        args.push(operation);
    }
    let preview = assert_success(&invoke(project.path(), &args));
    assert_eq!(preview["data"]["validation"]["forward"], "passed");
    assert_eq!(fs::read(&model_path).unwrap(), after_add);
    args.retain(|argument| *argument != "--dry-run");
    let committed = assert_success(&invoke(project.path(), &args));
    assert_eq!(committed["data"]["committed"], true);
    assert_eq!(committed["data"]["validation"]["forward"], "passed");
    let source = fs::read_to_string(&model_path).unwrap();
    assert!(source.contains("# Rebuild this ModuleList"), "{source}");
    assert!(source.contains("nn.Dropout(p=0.1)"), "{source}");
    let modules = committed["data"]["modules"].as_array().unwrap();
    assert!(
        modules
            .iter()
            .any(|row| { row["path"] == "model.blocks.1" && row["type"] == "Dropout" })
    );
    assert!(
        modules
            .iter()
            .any(|row| { row["path"] == "model.blocks.2" && row["type"] == "GELU" })
    );
    assert!(
        modules
            .iter()
            .any(|row| { row["path"] == "model.blocks.3" && row["type"] == "Linear" })
    );
    assert!(
        modules
            .iter()
            .any(|row| { row["path"] == "model.blocks.4" && row["type"] == "Dropout" })
    );
    let after_structural_edit = fs::read(&model_path).unwrap();

    assert_success(&invoke(
        project.path(),
        &["layer", "undo", "--target", "source"],
    ));
    assert_eq!(fs::read(&model_path).unwrap(), after_add);
    assert_success(&invoke(
        project.path(),
        &["layer", "undo", "--target", "source"],
    ));
    assert_eq!(fs::read(&model_path).unwrap(), original);
    assert_success(&invoke(
        project.path(),
        &["layer", "redo", "--target", "source"],
    ));
    assert_success(&invoke(
        project.path(),
        &["layer", "redo", "--target", "source"],
    ));
    assert_eq!(fs::read(&model_path).unwrap(), after_structural_edit);

    let variant_project = tempfile::tempdir().unwrap();
    fs::write(
        variant_project.path().join("model.py"),
        "import torch\nfrom torch import nn\n\nclass Network(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.ReLU(), nn.Identity(), nn.Linear(4, 4)])\n\n    def forward(self, values):\n        for block in self.blocks:\n            values = block(values)\n        return values\n\ndef build_model():\n    return Network()\n\ndef example_batch():\n    return torch.randn(2, 4)\n",
    )
    .unwrap();
    let variant_add = assert_success(&invoke(
        variant_project.path(),
        &[
            "layer",
            "add",
            "--parent",
            "model.blocks",
            "--after",
            "0",
            "--type",
            "Dropout",
            "--param",
            "p=0.2",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
        ],
    ));
    assert!(
        variant_add["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| { row["path"] == "model.blocks.1" && row["type"] == "Dropout" })
    );
    let variant_remove = assert_success(&invoke(
        variant_project.path(),
        &[
            "layer",
            "remove",
            "model.blocks.1",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
        ],
    ));
    assert!(
        variant_remove["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| { row["path"] == "model.blocks.1" && row["type"] == "ReLU" })
    );
    assert_success(&invoke(variant_project.path(), &["layer", "undo"]));
    let variant_after_undo: Value = serde_json::from_slice(
        &fs::read(
            variant_project
                .path()
                .join(".model-maven/variants/working.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(variant_after_undo["cursor"], 1);
    assert_success(&invoke(variant_project.path(), &["layer", "undo"]));
    let variant: Value = serde_json::from_slice(
        &fs::read(
            variant_project
                .path()
                .join(".model-maven/variants/working.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(variant["cursor"], 0);
}

#[test]
fn module_dict_crud_preserves_named_branch_order_in_variant_and_source() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let source = "import torch\nfrom torch import nn\n\nclass Network(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.branches = nn.ModuleDict({\n            \"left\": nn.Linear(4, 4),\n            \"right\": nn.Linear(4, 4),\n            \"extra\": nn.Identity(),\n            \"head\": nn.Linear(8, 2),\n        })\n\n    def forward(self, values):\n        left = self.branches[\"left\"](values)\n        right = self.branches[\"right\"](values)\n        return self.branches[\"head\"](torch.cat([left, right], dim=-1))\n\ndef build_model():\n    return Network()\n\ndef example_batch():\n    return torch.randn(2, 4)\n";

    let project = tempfile::tempdir().unwrap();
    fs::write(project.path().join("model.py"), source).unwrap();
    let model_path = project.path().join("model.py");
    let original = fs::read(&model_path).unwrap();
    let operations = [
        serde_json::json!({
            "action":"add",
            "parent":"model.branches",
            "before":"head",
            "name":"projection",
            "layer_type":"Identity",
            "params":{}
        })
        .to_string(),
        serde_json::json!({
            "action":"replace",
            "path":"model.branches.left",
            "layer_type":"GELU",
            "params":{}
        })
        .to_string(),
        serde_json::json!({
            "action":"remove",
            "path":"model.branches.extra",
            "strategy":"remove"
        })
        .to_string(),
    ];
    let mut args = vec![
        "layer",
        "apply",
        "--model",
        "model:build_model",
        "--batch",
        "model:example_batch",
        "--target",
        "source",
        "--dry-run",
    ];
    for operation in &operations {
        args.push("--operation");
        args.push(operation);
    }
    let preview = assert_success(&invoke(project.path(), &args));
    assert_eq!(preview["data"]["validation"]["forward"], "passed");
    assert_eq!(fs::read(&model_path).unwrap(), original);
    args.retain(|argument| *argument != "--dry-run");
    let committed = assert_success(&invoke(project.path(), &args));
    assert_eq!(committed["data"]["validation"]["forward"], "passed");
    let modules = committed["data"]["modules"].as_array().unwrap();
    assert!(
        modules
            .iter()
            .any(|row| { row["path"] == "model.branches.projection" && row["type"] == "Identity" })
    );
    assert!(
        modules
            .iter()
            .any(|row| { row["path"] == "model.branches.left" && row["type"] == "GELU" })
    );
    assert!(
        !modules
            .iter()
            .any(|row| row["path"] == "model.branches.extra")
    );
    let committed_source = fs::read_to_string(&model_path).unwrap();
    let rebuild = committed_source
        .split("# Rebuild this ModuleDict")
        .nth(1)
        .expect("ModuleDict rebuild snippet is included");
    let left = rebuild.find("'left':").unwrap();
    let right = rebuild.find("'right':").unwrap();
    let projection = rebuild.find("'projection':").unwrap();
    let head = rebuild.find("'head':").unwrap();
    assert!(left < right && right < projection && projection < head);
    assert!(rebuild.contains("self.branches['right']"));

    assert_success(&invoke(
        project.path(),
        &["layer", "undo", "--target", "source"],
    ));
    assert_eq!(fs::read(&model_path).unwrap(), original);
    assert_success(&invoke(
        project.path(),
        &["layer", "redo", "--target", "source"],
    ));
    assert_eq!(fs::read(&model_path).unwrap(), committed_source.as_bytes());

    let variant_project = tempfile::tempdir().unwrap();
    fs::write(variant_project.path().join("model.py"), source).unwrap();
    let variant_add = assert_success(&invoke(
        variant_project.path(),
        &[
            "layer",
            "add",
            "--parent",
            "model.branches",
            "--before",
            "head",
            "--name",
            "projection",
            "--type",
            "Identity",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
        ],
    ));
    assert!(
        variant_add["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| { row["path"] == "model.branches.projection" && row["type"] == "Identity" })
    );
    let variant_remove = assert_success(&invoke(
        variant_project.path(),
        &[
            "layer",
            "remove",
            "model.branches.projection",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
            "--target",
            "variant",
        ],
    ));
    assert!(
        !variant_remove["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["path"] == "model.branches.projection")
    );
}

#[test]
fn source_sequential_crud_rebuilds_registered_python_factories() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("custom_layers.py"),
        "from torch import nn\n\nclass ScaledIdentity(nn.Module):\n    def __init__(self, scale):\n        super().__init__()\n        self.scale = scale\n\n    def forward(self, values):\n        return values * self.scale\n\ndef make_scaled(scale):\n    return ScaledIdentity(scale)\n",
    )
    .unwrap();
    fs::write(
        project.path().join("model.py"),
        "import torch\nfrom collections import OrderedDict\nfrom torch import nn\n\nclass Network(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.encoder = nn.Sequential(OrderedDict([('proj', nn.Linear(8, 8)), ('act', nn.ReLU())]))\n\n    def forward(self, values):\n        return self.encoder(values)\n\ndef build_model():\n    return Network()\n\ndef example_batch():\n    return torch.randn(2, 8)\n",
    )
    .unwrap();

    let schema = r#"{"type":"object","properties":{"scale":{"type":"number","minimum":0.0,"maximum":4.0}},"required":["scale"],"additionalProperties":false}"#;
    let registered = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/scaled-identity",
            "--kind",
            "layer",
            "--entrypoint",
            "custom_layers:make_scaled",
            "--version",
            "1.2.0",
            "--params-schema",
            schema,
            "--capabilities",
            r#"{"construct":{"state":"supported"},"forward_probe":{"state":"supported","requires":["example_batch"]}}"#,
        ],
    );
    assert_success(&registered);

    let tested = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "test",
            "local/scaled-identity",
            "--param",
            "scale=0.5",
            "--batch",
            "model:example_batch",
        ],
    ));
    assert_eq!(tested["data"]["coverage"], "complete_for_declared_checks");
    assert_eq!(tested["data"]["checks"]["factory_import"], "passed");
    assert_eq!(tested["data"]["checks"]["constructs_nn_module"], "passed");
    assert_eq!(
        tested["data"]["checks"]["forward_probe"]["status"],
        "passed"
    );
    assert_eq!(
        tested["data"]["checks"]["forward_probe"]["outputs"]["shape"],
        serde_json::json!([2, 8])
    );

    let invalid_params = report(&invoke(
        project.path(),
        &[
            "component",
            "test",
            "local/scaled-identity",
            "--param",
            "scale=8.0",
        ],
    ));
    assert_eq!(invalid_params["status"], "error");
    assert_eq!(
        invalid_params["diagnostics"][0]["code"],
        "COMPONENT_RESOLUTION_FAILED"
    );

    let partial_test = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "test",
            "local/scaled-identity",
            "--param",
            "scale=1.0",
        ],
    ));
    assert_eq!(partial_test["data"]["coverage"], "partial");
    assert_eq!(
        partial_test["data"]["checks"]["forward_probe"]["status"],
        "not_run"
    );

    let construct_only = invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/construct-only",
            "--kind",
            "layer",
            "--entrypoint",
            "custom_layers:make_scaled",
            "--version",
            "1.0.0",
            "--params-schema",
            schema,
            "--capabilities",
            r#"{"construct":{"state":"supported"},"forward_probe":{"state":"unsupported","reason":"需要模型上下文"}}"#,
        ],
    );
    assert_success(&construct_only);
    let construct_only_test = assert_success(&invoke(
        project.path(),
        &[
            "component",
            "test",
            "local/construct-only",
            "--param",
            "scale=1.0",
            "--batch",
            "model:example_batch",
        ],
    ));
    assert_eq!(
        construct_only_test["data"]["checks"]["forward_probe"]["status"],
        "skipped"
    );
    assert_eq!(
        construct_only_test["data"]["checks"]["forward_probe"]["reason"],
        "需要模型上下文"
    );

    fs::OpenOptions::new()
        .append(true)
        .open(project.path().join("custom_layers.py"))
        .unwrap()
        .write_all(b"\ndef make_invalid():\n    return 'not an nn.Module'\n")
        .unwrap();
    assert_success(&invoke(
        project.path(),
        &[
            "component",
            "add",
            "local/invalid-layer",
            "--kind",
            "layer",
            "--entrypoint",
            "custom_layers:make_invalid",
            "--version",
            "1.0.0",
        ],
    ));
    let invalid_test = report(&invoke(
        project.path(),
        &["component", "test", "local/invalid-layer"],
    ));
    assert_eq!(invalid_test["status"], "error");
    assert_eq!(invalid_test["diagnostics"][0]["code"], "LAYER_TYPE_INVALID");

    let model_path = project.path().join("model.py");
    let original = fs::read(&model_path).unwrap();
    let operations = [
        serde_json::json!({
            "action":"replace",
            "path":"model.encoder.act",
            "component":"local/scaled-identity",
            "params":{"scale":0.5}
        })
        .to_string(),
        serde_json::json!({
            "action":"add",
            "parent":"model.encoder",
            "after":"act",
            "name":"custom_scale",
            "component":"local/scaled-identity",
            "params":{"scale":1.5}
        })
        .to_string(),
    ];
    let mut args = vec![
        "layer",
        "apply",
        "--model",
        "model:build_model",
        "--batch",
        "model:example_batch",
        "--target",
        "source",
        "--dry-run",
    ];
    for operation in &operations {
        args.push("--operation");
        args.push(operation);
    }
    let preview = assert_success(&invoke(project.path(), &args));
    assert_eq!(preview["data"]["committed"], false);
    assert_eq!(preview["data"]["validation"]["forward"], "passed");
    let diff = preview["data"]["source_patch"]["unified_diff"]
        .as_str()
        .unwrap();
    assert!(diff.contains("custom_layers as _m_model_component_0"));
    assert!(diff.contains("local/scaled-identity@1.2.0"));
    assert_eq!(fs::read(&model_path).unwrap(), original);

    args.retain(|argument| *argument != "--dry-run");
    let committed = assert_success(&invoke(project.path(), &args));
    assert_eq!(committed["data"]["committed"], true);
    let modules = committed["data"]["modules"].as_array().unwrap();
    assert!(modules.iter().any(|module| {
        module["path"] == "model.encoder.act"
            && module["type"]
                .as_str()
                .unwrap_or("")
                .contains("ScaledIdentity")
    }));
    assert!(modules.iter().any(|module| {
        module["path"] == "model.encoder.custom_scale"
            && module["type"]
                .as_str()
                .unwrap_or("")
                .contains("ScaledIdentity")
    }));
    let source = String::from_utf8_lossy(&fs::read(&model_path).unwrap()).into_owned();
    assert!(source.contains("_m_model_json_0.loads('"));
    assert!(source.contains("manifest sha256"));

    assert_success(&invoke(
        project.path(),
        &["layer", "undo", "--target", "source"],
    ));
    assert_eq!(fs::read(&model_path).unwrap(), original);
    assert_success(&invoke(
        project.path(),
        &["layer", "redo", "--target", "source"],
    ));

    let single = invoke(
        project.path(),
        &[
            "layer",
            "replace",
            "model.encoder.act",
            "--component",
            "custom_layers:make_scaled",
            "--param",
            "scale=2.0",
            "--target",
            "source",
            "--model",
            "model:build_model",
            "--batch",
            "model:example_batch",
        ],
    );
    assert_success(&single);
    let reloaded = assert_success(&invoke(
        project.path(),
        &[
            "layer",
            "show",
            "model.encoder.act",
            "--model",
            "model:build_model",
        ],
    ));
    assert!(
        reloaded["data"]["modules"][0]["type"]
            .as_str()
            .unwrap_or("")
            .contains("ScaledIdentity")
    );
    assert_success(&invoke(
        project.path(),
        &["layer", "undo", "--target", "source"],
    ));
}

#[test]
fn formal_manual_training_resumes_optimizer_phase_rng_and_batch_cursor() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let split = tempfile::tempdir().unwrap();
    copy_complex_fixture(split.path());
    let environment_fixture = split.path().join("model_maven_env_probe-1.0.0.dist-info");
    fs::create_dir(&environment_fixture).unwrap();
    let environment_metadata = environment_fixture.join("METADATA");
    fs::write(
        &environment_metadata,
        "Metadata-Version: 2.1\nName: model-maven-env-probe\nVersion: 1.0.0\n\n",
    )
    .unwrap();
    let initial = invoke(
        split.path(),
        &[
            "train",
            "--seed",
            "1234",
            "--training-mode",
            "manual-step",
            "--model",
            "complex_acceptance.model:build_model",
            "--batch",
            "complex_acceptance.data:changing_batch",
            "--step",
            "complex_acceptance.training:manual_multitask_step",
            "--steps",
            "3",
            "--checkpoint-every",
            "1",
        ],
    );
    let initial = assert_success(&initial);
    assert_eq!(initial["operation"], "train");
    assert_eq!(initial["data"]["completed_steps"], 3);
    assert_eq!(initial["data"]["optimizer_updates"], 6);
    assert_eq!(initial["data"]["checkpoint_saved"], true);
    assert_eq!(initial["data"]["checkpoint_completed_steps"], 3);
    let environment_fingerprint = initial["data"]["environment_snapshot"]["fingerprint_sha256"]
        .as_str()
        .expect("formal training should record its environment fingerprint");
    assert_eq!(
        initial["data"]["runtime"]["environment_fingerprint_sha256"],
        environment_fingerprint
    );
    assert_eq!(environment_fingerprint.len(), 64);
    assert!(
        initial["data"]["environment_snapshot"]["distributions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|distribution| {
                distribution["name"] == "model-maven-env-probe"
                    && distribution["version"] == "1.0.0"
            })
    );
    let parent_id = initial["data"]["run_id"].as_str().unwrap();
    let parent_dir = split.path().join(".model-maven/runs").join(parent_id);
    let checkpoint_manifest: Value =
        serde_json::from_slice(&fs::read(parent_dir.join("checkpoints/latest.json")).unwrap())
            .unwrap();
    assert_eq!(checkpoint_manifest["completed_steps"], 3);
    assert!(
        parent_dir
            .join("checkpoints")
            .join(checkpoint_manifest["file"].as_str().unwrap())
            .is_file()
    );

    let resumed = invoke(
        split.path(),
        &[
            "train",
            "--resume",
            parent_id,
            "--additional-steps",
            "2",
            "--checkpoint-every",
            "1",
        ],
    );
    let resumed = assert_success(&resumed);
    assert_eq!(resumed["data"]["completed_steps"], 5);
    assert_eq!(
        resumed["data"]["runtime"]["environment_fingerprint_sha256"],
        environment_fingerprint
    );
    assert_eq!(resumed["data"]["resumed_from_steps"], 3);
    assert_eq!(resumed["data"]["optimizer_updates"], 10);
    assert_eq!(resumed["data"]["optimizer_updates_by_group"]["encoders"], 5);
    assert_eq!(
        resumed["data"]["optimizer_updates_by_group"]["fusion_heads"],
        5
    );
    assert_eq!(resumed["data"]["resume_state"]["batch"], "restored");
    assert_eq!(
        resumed["data"]["resume_state"]["training_step"],
        "stateless"
    );
    assert_eq!(resumed["data"]["parent_run_id"], parent_id);
    let child_id = resumed["data"]["run_id"].as_str().unwrap();
    let child = invoke(split.path(), &["runs", "show", child_id]);
    let child = assert_success(&child);
    assert_eq!(child["data"]["record"]["parent_run_id"], parent_id);
    assert_eq!(child["data"]["checkpoint"]["completed_steps"], 5);
    let preserved_parent = invoke(split.path(), &["runs", "show", parent_id]);
    let preserved_parent = assert_success(&preserved_parent);
    assert_eq!(
        preserved_parent["data"]["report"]["data"]["completed_steps"],
        3
    );

    fs::write(
        &environment_metadata,
        "Metadata-Version: 2.1\nName: model-maven-env-probe\nVersion: 1.0.1\n\n",
    )
    .unwrap();
    let drifted_resume = report(&invoke(
        split.path(),
        &[
            "train",
            "--resume",
            parent_id,
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    ));
    assert_eq!(drifted_resume["status"], "error");
    assert_eq!(
        drifted_resume["diagnostics"][0]["code"],
        "RESUME_RUNTIME_CHANGED"
    );
    assert!(
        drifted_resume["data"]["runtime_changes"]
            .get("environment_fingerprint_sha256")
            .is_some()
    );

    fs::write(
        &environment_metadata,
        "Metadata-Version: 2.1\nName: model-maven-env-probe\nVersion: 1.0.0\n\n",
    )
    .unwrap();
    let checkpoint_path = parent_dir
        .join("checkpoints")
        .join(checkpoint_manifest["file"].as_str().unwrap());
    let latest_path = parent_dir.join("checkpoints/latest.json");
    let tamper_checkpoint = Command::new(python())
        .args([
            "-c",
            "import hashlib,json,sys,torch; from pathlib import Path; p=Path(sys.argv[1]); payload=torch.load(p,map_location='cpu',weights_only=True); payload['runtime']['environment_fingerprint_sha256']='f'*64; torch.save(payload,p); m=Path(sys.argv[2]); latest=json.loads(m.read_text()); latest['sha256']=hashlib.sha256(p.read_bytes()).hexdigest(); m.write_text(json.dumps(latest))",
        ])
        .arg(&checkpoint_path)
        .arg(&latest_path)
        .output()
        .unwrap();
    assert!(
        tamper_checkpoint.status.success(),
        "{}",
        String::from_utf8_lossy(&tamper_checkpoint.stderr)
    );
    let checkpoint_drift = report(&invoke(
        split.path(),
        &[
            "train",
            "--resume",
            parent_id,
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    ));
    assert_eq!(checkpoint_drift["status"], "error");
    assert_eq!(
        checkpoint_drift["diagnostics"][0]["code"],
        "RESUME_ENVIRONMENT_CHANGED"
    );

    let uninterrupted = tempfile::tempdir().unwrap();
    copy_complex_fixture(uninterrupted.path());
    let full = invoke(
        uninterrupted.path(),
        &[
            "train",
            "--seed",
            "1234",
            "--training-mode",
            "manual-step",
            "--model",
            "complex_acceptance.model:build_model",
            "--batch",
            "complex_acceptance.data:changing_batch",
            "--step",
            "complex_acceptance.training:manual_multitask_step",
            "--steps",
            "5",
            "--checkpoint-every",
            "1",
        ],
    );
    let full = assert_success(&full);
    assert_eq!(
        resumed["data"]["model_state_sha256"], full["data"]["model_state_sha256"],
        "分段恢复应与同种子的一次性五步训练产生相同模型权重"
    );
}

#[test]
fn failed_training_run_keeps_latest_checkpoint_resumable() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("recovery_fixture.py"),
        r#"
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
    def forward(self, values):
        return self.layer(values)

def build_model():
    return Model()

def batch():
    return torch.ones(2, 2), torch.zeros(2, 1)

_RECOVERED = False
_CALLS = 0

def step_state_dict():
    return {"calls": _CALLS}

def load_step_state_dict(state):
    global _RECOVERED, _CALLS
    _RECOVERED = True
    _CALLS = state["calls"]

def step(model, batch, context):
    global _CALLS
    if context.step == 3 and not _RECOVERED:
        raise RuntimeError("simulated transient failure")
    _CALLS += 1
    values, targets = batch
    return {"loss": (model(values) - targets).square().mean()}

step.state_dict = step_state_dict
step.load_state_dict = load_step_state_dict
batch.MODEL_MAVEN_STATELESS = True
"#,
    )
    .unwrap();

    let failed_run = invoke(
        project.path(),
        &[
            "train",
            "--model",
            "recovery_fixture:build_model",
            "--batch",
            "recovery_fixture:batch",
            "--step",
            "recovery_fixture:step",
            "--steps",
            "5",
            "--checkpoint-every",
            "1",
        ],
    );
    let failed_report = report(&failed_run);
    assert!(!failed_run.status.success(), "{failed_report:#}");
    assert_eq!(failed_report["diagnostics"][0]["code"], "STEP_FAILED");
    let run_id = failed_report["data"]["run_id"].as_str().unwrap();
    let failed_record = invoke(project.path(), &["runs", "show", run_id]);
    let failed_record = assert_success(&failed_record);
    assert_eq!(failed_record["data"]["record"]["status"], "failed");
    assert_eq!(failed_record["data"]["checkpoint"]["completed_steps"], 2);

    let resumed = invoke(
        project.path(),
        &[
            "train",
            "--resume",
            run_id,
            "--additional-steps",
            "2",
            "--checkpoint-every",
            "1",
        ],
    );
    let resumed = assert_success(&resumed);
    assert_eq!(resumed["data"]["resumed_from_steps"], 2);
    assert_eq!(resumed["data"]["completed_steps"], 4);
    assert_eq!(resumed["data"]["resume_state"]["training_step"], "restored");
    assert_eq!(resumed["data"]["parent_run_id"], run_id);
}

#[test]
fn export_training_run_copies_verified_sources_and_restores_weights() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    assert_success(&invoke(
        project.path(),
        &[
            "init",
            "--model",
            "fixture:build_editable",
            "--batch",
            "fixture:editable_batch",
        ],
    ));
    let edited = invoke(
        project.path(),
        &[
            "layer",
            "add",
            "--model",
            "fixture:build_editable",
            "--batch",
            "fixture:editable_batch",
            "--parent",
            "model.encoder",
            "--after",
            "relu",
            "--name",
            "dropout",
            "--type",
            "Dropout",
            "--param",
            "p=0.1",
        ],
    );
    assert_success(&edited);
    let trained = invoke(
        project.path(),
        &[
            "--variant",
            "working",
            "train",
            "--batch",
            "fixture:editable_batch",
            "--step",
            "fixture:editable_training_step",
            "--steps",
            "2",
            "--checkpoint-every",
            "1",
        ],
    );
    let trained = assert_success(&trained);
    let parent_id = trained["data"]["run_id"].as_str().unwrap();
    let resumed = invoke(
        project.path(),
        &[
            "train",
            "--resume",
            parent_id,
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    );
    let resumed = assert_success(&resumed);
    assert_eq!(resumed["data"]["resume_state"]["batch"], "stateless");
    assert_eq!(
        resumed["data"]["resume_state"]["training_step"],
        "stateless"
    );
    let run_id = resumed["data"]["run_id"].as_str().unwrap();
    let expected_state_hash = resumed["data"]["model_state_sha256"].as_str().unwrap();
    let recorded_run = invoke(project.path(), &["runs", "show", run_id]);
    let recorded_run = assert_success(&recorded_run);
    let training_environment = &recorded_run["data"]["report"]["data"]["environment_snapshot"];
    assert_eq!(training_environment["capture_timing"], "training_start");
    assert!(
        training_environment["captured_at_utc"]
            .as_str()
            .is_some_and(|timestamp| timestamp.ends_with('Z'))
    );

    let outside = tempfile::tempdir().unwrap();
    let destination = outside.path().join("model-export");
    let destination_arg = destination.to_str().unwrap();
    let exported = invoke(project.path(), &["export", destination_arg]);
    let exported = assert_success(&exported);
    assert_eq!(exported["operation"], "export");
    assert_eq!(exported["data"]["source_file_count"], 1);
    assert_eq!(
        exported["data"]["checkpoint_sha256"],
        serde_json::from_slice::<Value>(&fs::read(destination.join("manifest.json")).unwrap())
            .unwrap()["checkpoint"]["sha256"]
    );
    assert!(destination.join("source/fixture.py").is_file());
    assert!(
        destination
            .join("artifacts/model-maven-checkpoint.pt")
            .is_file()
    );
    assert!(destination.join("model_maven_worker.py").is_file());
    let export_manifest: Value =
        serde_json::from_slice(&fs::read(destination.join("manifest.json")).unwrap()).unwrap();
    assert!(export_manifest["bundle_files"]["environment.json"].is_string());
    assert!(export_manifest["bundle_files"]["requirements.snapshot.txt"].is_string());
    assert!(export_manifest["bundle_files"]["variant.json"].is_string());
    assert!(export_manifest["bundle_files"]["model_maven_worker/__init__.py"].is_string());
    assert_eq!(export_manifest["environment"]["status"], "captured");
    let environment: Value =
        serde_json::from_slice(&fs::read(destination.join("environment.json")).unwrap()).unwrap();
    assert_eq!(environment["capture_status"], "captured");
    assert_eq!(environment["capture_timing"], "training_start");
    assert_eq!(
        environment["captured_at_utc"],
        training_environment["captured_at_utc"]
    );
    assert_eq!(
        environment["fingerprint_sha256"],
        recorded_run["data"]["report"]["data"]["runtime"]["environment_fingerprint_sha256"]
    );
    assert!(
        environment["captured_at_utc"]
            .as_str()
            .is_some_and(|timestamp| timestamp.ends_with('Z'))
    );
    assert_eq!(
        environment["runtime_comparison"]["python_version"]["status"],
        "matches"
    );
    assert_eq!(
        environment["runtime_comparison"]["pytorch_version"]["status"],
        "matches"
    );
    assert!(
        environment["distributions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| {
                item["name"]
                    .as_str()
                    .is_some_and(|name| name.eq_ignore_ascii_case("torch"))
                    && item["version"] == export_manifest["runtime"]["pytorch_version"]
            })
    );

    let workbench_destination = outside.path().join("workbench-export");
    let workbench_destination_text = workbench_destination.to_string_lossy();
    let workbench_success = run_workbench_pty(
        project.path(),
        format!("6\n1\n{workbench_destination_text}\nback\n8\n").as_bytes(),
    );
    assert!(
        workbench_success.contains("导出完成"),
        "{workbench_success}"
    );
    assert!(workbench_destination.join("manifest.json").is_file());
    fs::write(
        workbench_destination.join("user-note.txt"),
        "keep this file\n",
    )
    .unwrap();
    let workbench_conflict = run_workbench_pty(
        project.path(),
        format!("6\n1\n{workbench_destination_text}\nno\nback\n8\n").as_bytes(),
    );
    assert!(
        workbench_conflict.contains("导出未完成"),
        "{workbench_conflict}"
    );
    assert!(
        workbench_conflict.contains("导出目标已存在"),
        "{workbench_conflict}"
    );
    assert!(
        workbench_conflict.contains("workbench-export"),
        "{workbench_conflict}"
    );
    assert!(
        workbench_conflict.contains("已取消覆盖；原导出目录保持不变"),
        "{workbench_conflict}"
    );
    assert_eq!(
        fs::read_to_string(workbench_destination.join("user-note.txt")).unwrap(),
        "keep this file\n"
    );
    let workbench_replaced = run_workbench_pty(
        project.path(),
        format!("6\n1\n{workbench_destination_text}\nOVERWRITE\nback\n8\n").as_bytes(),
    );
    assert!(
        workbench_replaced.contains("导出完成"),
        "{workbench_replaced}"
    );
    let backup_path = workbench_replaced
        .lines()
        .find_map(|line| line.strip_prefix("原目标已备份到："))
        .expect("overwrite confirmation reports its backup directory")
        .trim();
    assert!(Path::new(backup_path).join("user-note.txt").is_file());
    assert!(!workbench_destination.join("user-note.txt").exists());

    let requirements = fs::read_to_string(destination.join("requirements.snapshot.txt")).unwrap();
    assert!(requirements.contains(&format!(
        "torch=={}",
        export_manifest["runtime"]["pytorch_version"].as_str().unwrap()
    )));
    assert!(environment.get("requirements_snapshot").is_none());

    let restored = Command::new(python())
        .arg(destination.join("restore_model.py"))
        .current_dir(&destination)
        .output()
        .unwrap();
    assert!(
        restored.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&restored.stdout),
        String::from_utf8_lossy(&restored.stderr)
    );
    assert!(
        String::from_utf8_lossy(&restored.stdout).contains(expected_state_hash),
        "{}",
        String::from_utf8_lossy(&restored.stdout)
    );

    let project_target = project.path().join("unsafe-export");
    let project_target_arg = project_target.to_str().unwrap();
    let rejected_project_target = invoke(
        project.path(),
        &["export", project_target_arg, "--run", run_id],
    );
    assert_eq!(report(&rejected_project_target)["status"], "error");
    assert!(!project_target.exists());

    let source_path = project.path().join("fixture.py");
    let source = fs::read(&source_path).unwrap();
    fs::write(
        &source_path,
        [source.as_slice(), b"\n# changed\n".as_slice()].concat(),
    )
    .unwrap();
    let drifted = outside.path().join("drifted-export");
    let drifted_arg = drifted.to_str().unwrap();
    let rejected_drift = invoke(project.path(), &["export", drifted_arg, "--run", run_id]);
    assert_eq!(report(&rejected_drift)["status"], "error");
    assert!(!drifted.exists());
    fs::write(&source_path, source).unwrap();

    fs::write(destination.join("user-note.txt"), "keep this backup\n").unwrap();
    let rejected = invoke(
        project.path(),
        &["export", destination_arg, "--run", run_id],
    );
    let rejected = report(&rejected);
    assert_eq!(rejected["status"], "error");
    assert!(destination.join("user-note.txt").is_file());

    let replaced = invoke(
        project.path(),
        &["export", destination_arg, "--run", run_id, "--overwrite"],
    );
    let replaced = assert_success(&replaced);
    let backup = replaced["data"]["backup_path"].as_str().unwrap();
    assert!(Path::new(backup).join("user-note.txt").is_file());
    assert!(!destination.join("user-note.txt").exists());

    let original_project = project.path().to_path_buf();
    drop(project);
    assert!(!original_project.exists());
    let isolated_restore = Command::new(python())
        .args([
            "-c",
            "import runpy,sys; ns=runpy.run_path(sys.argv[1]); state=ns['load_training_state'](); assert state['completed_steps']==3; assert state['training_mode']=='managed-step'; assert state['engine_state']['optimizer_state']['state']; print('training-state-api-ok')",
        ])
        .arg(destination.join("restore_model.py"))
        .current_dir(&destination)
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .output()
        .unwrap();
    assert!(
        isolated_restore.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&isolated_restore.stdout),
        String::from_utf8_lossy(&isolated_restore.stderr)
    );
    assert!(String::from_utf8_lossy(&isolated_restore.stdout).contains(expected_state_hash));
    assert!(String::from_utf8_lossy(&isolated_restore.stdout).contains("training-state-api-ok"));

    let checkpoint_path = destination.join("artifacts/model-maven-checkpoint.pt");
    let checkpoint_bytes = fs::read(&checkpoint_path).unwrap();
    let mut tampered_checkpoint = checkpoint_bytes.clone();
    tampered_checkpoint[0] ^= 0x01;
    fs::write(&checkpoint_path, tampered_checkpoint).unwrap();
    let rejected_checkpoint = Command::new(python())
        .arg(destination.join("restore_model.py"))
        .current_dir(&destination)
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .output()
        .unwrap();
    assert!(!rejected_checkpoint.status.success());
    assert!(
        String::from_utf8_lossy(&rejected_checkpoint.stderr)
            .contains("checkpoint SHA-256 校验失败")
    );
    fs::write(checkpoint_path, checkpoint_bytes).unwrap();

    let variant_path = destination.join("variant.json");
    let variant_bytes = fs::read(&variant_path).unwrap();
    fs::write(
        &variant_path,
        [variant_bytes.as_slice(), b"\n".as_slice()].concat(),
    )
    .unwrap();
    let rejected_variant = Command::new(python())
        .arg(destination.join("restore_model.py"))
        .current_dir(&destination)
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .output()
        .unwrap();
    assert!(!rejected_variant.status.success());
    assert!(
        String::from_utf8_lossy(&rejected_variant.stderr)
            .contains("导出支持文件 SHA-256 校验失败：variant.json")
    );
    fs::write(variant_path, variant_bytes).unwrap();

    let environment_path = destination.join("environment.json");
    let environment_bytes = fs::read(&environment_path).unwrap();
    fs::remove_file(&environment_path).unwrap();
    let rejected_missing_bundle_file = Command::new(python())
        .arg(destination.join("restore_model.py"))
        .current_dir(&destination)
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .output()
        .unwrap();
    assert!(!rejected_missing_bundle_file.status.success());
    assert!(
        String::from_utf8_lossy(&rejected_missing_bundle_file.stderr)
            .contains("导出文件缺失：environment.json")
    );
    fs::write(environment_path, environment_bytes).unwrap();
}

#[test]
fn evaluation_command_uses_checkpoint_custom_metrics_and_persists_provenance() {
    if !has_torch() {
        eprintln!(
            "skipped evaluation integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("evaluation_cli_fixture.py"),
        r#"
import torch
from torch import nn

class MultiOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2)
    def forward(self, values):
        output = self.shared(values)
        return {"classification": output, "embedding": output.mean(dim=0)}

def build_model(): return MultiOutput()
def batch(): return torch.ones(3, 2), torch.zeros(3, 2)
def step(model, batch, context):
    values, target = batch
    output = model(values)
    return {"loss": (output["classification"] - target).square().mean()}
def validation_data(): return {"values": torch.ones(3, 2), "labels": ["a", "b", "a"]}
def evaluate(model, data, context):
    output = model(data["values"])
    assert set(output) == {"classification", "embedding"}
    assert not model.training
    context.report_samples(3)
    if context.max_preview_samples:
        context.report_prediction("sample-0", output["classification"][0].tolist(), data["labels"][0])
    return {"metrics": {"custom/class_macro_f1": 0.75, "retrieval/recall_at_5": 0.5}}
def invalid_evaluate(model, data, context):
    return {"metrics": {"custom/macro_f1": float("nan")}}
"#,
    )
    .unwrap();

    let trained = invoke(
        project.path(),
        &[
            "train",
            "--model",
            "evaluation_cli_fixture:build_model",
            "--batch",
            "evaluation_cli_fixture:batch",
            "--step",
            "evaluation_cli_fixture:step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
            "--request-id",
            "evaluation-source-request-123",
        ],
    );
    let trained = assert_success(&trained);
    let source_run = trained["data"]["run_id"].as_str().unwrap();
    let evaluated = invoke(
        project.path(),
        &[
            "evaluate",
            "--run",
            source_run,
            "--evaluator",
            "evaluation_cli_fixture:evaluate",
            "--data",
            "evaluation_cli_fixture:validation_data",
            "--split",
            "validation",
            "--samples",
            "4",
            "--preview-samples",
            "2",
            "--data-fingerprint",
            "validation-set-v3",
        ],
    );
    let evaluated = assert_success(&evaluated);
    assert_eq!(evaluated["operation"], "evaluate");
    assert_eq!(evaluated["data"]["source_run_id"], source_run);
    assert_eq!(evaluated["data"]["checkpoint"]["completed_steps"], 1);
    assert_eq!(
        evaluated["data"]["evaluation"]["metrics"]["custom/class_macro_f1"],
        0.75
    );
    assert_eq!(evaluated["data"]["evaluation"]["samples_evaluated"], 3);
    assert_eq!(
        evaluated["data"]["evaluation"]["sample_previews"]["state"],
        "reported"
    );
    assert_eq!(
        evaluated["data"]["evaluation"]["sample_previews"]["reported"],
        1
    );
    assert_eq!(
        evaluated["data"]["evaluation"]["sample_previews"]["samples"][0]["sample_id"],
        "sample-0"
    );
    assert_eq!(
        evaluated["data"]["coverage"]["classification_assumption"],
        false
    );
    assert_eq!(
        evaluated["data"]["coverage"]["data_identity"],
        "declared_fingerprint_and_inputs"
    );
    let human_evaluation = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "evaluate",
            "--run",
            source_run,
            "--evaluator",
            "evaluation_cli_fixture:evaluate",
            "--data",
            "evaluation_cli_fixture:validation_data",
            "--split",
            "validation",
            "--samples",
            "4",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(human_evaluation.status.success());
    let human_text = String::from_utf8_lossy(&human_evaluation.stdout);
    assert!(human_text.contains("评估结果"), "{human_text}");
    assert!(human_text.contains("实际评估样本：3"), "{human_text}");
    assert!(human_text.contains("来源训练运行："), "{human_text}");

    let evaluation_run = evaluated["data"]["run_id"].as_str().unwrap();
    let shown = assert_success(&invoke(project.path(), &["runs", "show", evaluation_run]));
    assert_eq!(shown["data"]["record"]["operation"], "evaluate");
    assert!(shown["data"]["record"]["request_id"].is_null());
    let lookup = assert_success(&invoke(
        project.path(),
        &["runs", "find-request", "evaluation-source-request-123"],
    ));
    assert_eq!(lookup["data"]["matches"].as_array().unwrap().len(), 1);
    assert_eq!(lookup["data"]["matches"][0]["id"], source_run);
    assert_eq!(
        shown["data"]["report"]["data"]["evaluation"]["sample_previews"]["reported"],
        1
    );
    assert_eq!(
        shown["data"]["report"]["data"]["provenance"]["data_identity_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );

    let invalid = report(&invoke(
        project.path(),
        &[
            "evaluate",
            "--run",
            source_run,
            "--evaluator",
            "evaluation_cli_fixture:invalid_evaluate",
            "--data",
            "evaluation_cli_fixture:validation_data",
        ],
    ));
    assert_eq!(invalid["status"], "error");
    assert_eq!(
        invalid["diagnostics"][0]["code"],
        "EVALUATION_METRIC_INVALID"
    );
    let human_failure = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "evaluate",
            "--run",
            source_run,
            "--evaluator",
            "evaluation_cli_fixture:invalid_evaluate",
            "--data",
            "evaluation_cli_fixture:validation_data",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!human_failure.status.success());
    let human_error = String::from_utf8_lossy(&human_failure.stderr);
    assert!(human_error.contains("评估未完成"), "{human_error}");
    assert!(
        human_error.contains("EVALUATION_METRIC_INVALID"),
        "{human_error}"
    );

    let fixture_path = project.path().join("evaluation_cli_fixture.py");
    let original = fs::read(&fixture_path).unwrap();
    fs::write(
        &fixture_path,
        [original.as_slice(), b"\n# drift\n"].concat(),
    )
    .unwrap();
    let drifted = report(&invoke(
        project.path(),
        &[
            "evaluate",
            "--run",
            source_run,
            "--evaluator",
            "evaluation_cli_fixture:evaluate",
            "--data",
            "evaluation_cli_fixture:validation_data",
        ],
    ));
    assert_eq!(drifted["status"], "error");
    assert_eq!(drifted["diagnostics"][0]["code"], "SOURCE_CHANGED");
}
