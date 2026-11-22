use std::{
    fs,
    io::{Read, Write},
    os::fd::FromRawFd,
    path::Path,
    process::{Command, Stdio},
};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

fn fixture(root: &Path) {
    fs::write(
        root.join("assembly_fixture.py"),
        r#"
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
    def forward(self, values):
        return self.head(self.encoder(values))

def build_model(encoder=None): return Model(encoder)
def build_encoder(): return nn.Sequential(nn.Linear(4, 4), nn.ReLU())
def build_other_encoder(): return nn.Sequential(nn.Linear(4, 4), nn.GELU())
def batch(): return torch.ones(2, 4)
"#,
    )
    .unwrap();
}

fn init(root: &Path) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            "assembly_fixture:build_model",
        ])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
fn workbench(root: &Path, input: &[u8]) -> String {
    let (mut master_fd, mut slave_fd) = (-1, -1);
    let opened = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(opened, 0);
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
    terminal.write_all(input).unwrap();
    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut descriptor = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut descriptor, 1, 60_000) };
        assert!(
            ready > 0,
            "workbench timeout:\n{}",
            String::from_utf8_lossy(&transcript)
        );
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read PTY: {error}"),
        }
    }
    let status = child.wait().unwrap();
    assert!(status.success(), "workbench exited with {status}");
    String::from_utf8_lossy(&transcript).into_owned()
}

fn assembly_report(root: &Path) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", &python(), "assembly", "show"])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[cfg(unix)]
fn workbench_with_external_config_change_before_confirmation(root: &Path) -> String {
    let (mut master_fd, mut slave_fd) = (-1, -1);
    let opened = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(opened, 0);
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
    terminal
        .write_all(
            b"2\n13\nadd\nencoder\nassembly_fixture:batch\nsingle\n\n\nassembly_fixture:build_encoder\n",
        )
        .unwrap();
    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut descriptor = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut descriptor, 1, 60_000) };
        assert!(
            ready > 0,
            "workbench did not reach assembly confirmation:\n{}",
            String::from_utf8_lossy(&transcript)
        );
        match terminal.read(&mut buffer) {
            Ok(0) => panic!("workbench exited before assembly confirmation"),
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => {
                panic!("workbench PTY closed before assembly confirmation")
            }
            Err(error) => panic!("cannot read PTY: {error}"),
        }
        if String::from_utf8_lossy(&transcript).contains("写入项目默认装配？") {
            break;
        }
    }

    let config_path = root.join(".model-maven/project.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
    config["external_revision_note"] = serde_json::json!("written while confirmation was open");
    fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    terminal.write_all(b"yes\n16\n8\n").unwrap();

    loop {
        let mut descriptor = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut descriptor, 1, 60_000) };
        assert!(
            ready > 0,
            "workbench timed out after external config change:\n{}",
            String::from_utf8_lossy(&transcript)
        );
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read PTY: {error}"),
        }
    }
    let status = child.wait().unwrap();
    assert!(status.success(), "workbench exited with {status}");
    String::from_utf8_lossy(&transcript).into_owned()
}

#[cfg(unix)]
#[test]
fn assembly_workbench_dry_run_does_not_persist_and_explains_target_and_validation() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    init(project.path());

    let transcript = workbench(
        project.path(),
        b"2\n13\npreview\nadd\nencoder\nassembly_fixture:batch\nsingle\n\n\nassembly_fixture:build_encoder\n16\n8\n",
    );
    assert!(
        transcript.contains("根工厂装配 · 项目默认配置"),
        "{transcript}"
    );
    assert!(
        transcript.contains("不是 model.encoder 这样的模块路径"),
        "{transcript}"
    );
    assert!(transcript.contains("尚未运行代表批次前向"), "{transcript}");
    assert!(transcript.contains("预览，未更改配置"), "{transcript}");
    assert_eq!(
        assembly_report(project.path())["data"]["slots"],
        serde_json::json!([])
    );
}

#[cfg(unix)]
#[test]
fn assembly_workbench_add_set_remove_round_trip_persists_real_default_configuration() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    init(project.path());

    let transcript = workbench(
        project.path(),
        b"2\n13\nadd\nencoder\nassembly_fixture:batch\nsingle\n\n\nassembly_fixture:missing_factory\n13\nadd\nencoder\nassembly_fixture:batch\nsingle\n\n\nassembly_fixture:build_encoder\nyes\n13\nset\nencoder\nassembly_fixture:batch\nsingle\n\n\nassembly_fixture:build_other_encoder\nno\n13\nset\nencoder\nassembly_fixture:batch\nsingle\n\n\nassembly_fixture:build_other_encoder\nyes\n13\nremove\nencoder\nassembly_fixture:batch\nsingle\n\n\nyes\n16\n8\n",
    );
    assert!(transcript.contains("项目默认配置"), "{transcript}");
    assert!(transcript.contains("前向检查：\"passed\""), "{transcript}");
    assert!(transcript.contains("已保存"), "{transcript}");
    assert!(
        transcript.contains("ENTRYPOINT_LOAD_FAILED"),
        "{transcript}"
    );
    assert!(transcript.contains("本次交互已取消"), "{transcript}");
    assert!(
        transcript.contains("接缝 encoder ← assembly_fixture:build_encoder"),
        "{transcript}"
    );
    let report = assembly_report(project.path());
    assert_eq!(report["data"]["slots"], serde_json::json!([]));
    assert_eq!(report["data"]["validation"]["forward"], "not_run");
}

#[cfg(unix)]
#[test]
fn assembly_confirmation_rejects_a_config_changed_while_the_prompt_is_open() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    init(project.path());

    let transcript = workbench_with_external_config_change_before_confirmation(project.path());
    assert!(
        transcript.contains("PROJECT_CONFIG_CONFLICT"),
        "{transcript}"
    );
    assert!(
        transcript.contains("确认期间项目配置已变化"),
        "{transcript}"
    );
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/project.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        config["external_revision_note"],
        "written while confirmation was open"
    );
    assert_eq!(config["model_injections"], serde_json::json!({}));
}
