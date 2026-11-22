use std::{
    fs,
    io::{Read, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
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

struct WorkbenchPty {
    child: Child,
    master: fs::File,
    transcript: Vec<u8>,
}

impl WorkbenchPty {
    fn spawn(project: &Path) -> Self {
        let (mut master_fd, mut slave_fd) = (-1, -1);
        let size = libc::winsize {
            ws_row: 30,
            ws_col: 100,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // SAFETY: openpty initializes both descriptors; size is valid for this call.
        let result = unsafe {
            libc::openpty(
                &mut master_fd,
                &mut slave_fd,
                std::ptr::null_mut(),
                std::ptr::null(),
                &size,
            )
        };
        assert_eq!(
            result,
            0,
            "openpty failed: {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: descriptors were returned by openpty; ownership moves to File.
        let master = unsafe { fs::File::from_raw_fd(master_fd) };
        let slave = unsafe { fs::File::from_raw_fd(slave_fd) };
        let stdin = slave.try_clone().unwrap();
        let stdout = slave.try_clone().unwrap();
        let stderr = slave.try_clone().unwrap();
        let python = python();
        let child = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args(["--python", &python])
            .current_dir(project)
            .env("TERM", "xterm-256color")
            .env("COLUMNS", "100")
            .stdin(Stdio::from(stdin))
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .unwrap();
        drop(slave);
        Self {
            child,
            master,
            transcript: Vec::new(),
        }
    }

    fn send(&mut self, input: &[u8]) {
        self.master.write_all(input).unwrap();
        self.master.flush().unwrap();
    }

    fn read_until(&mut self, needles: &[&str], timeout: Duration) -> String {
        let deadline = Instant::now() + timeout;
        let mut buffer = [0_u8; 4096];
        loop {
            let current = String::from_utf8_lossy(&self.transcript);
            if needles.iter().all(|needle| current.contains(needle)) {
                return current.into_owned();
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {needles:?}; exit={:?}; transcript:\n{current}",
                self.child
                    .try_wait()
                    .ok()
                    .flatten()
                    .and_then(|status| status.code())
            );
            let mut descriptor = libc::pollfd {
                fd: self.master.as_raw_fd(),
                events: libc::POLLIN | libc::POLLHUP,
                revents: 0,
            };
            // SAFETY: descriptor points to one initialized pollfd for the call.
            let ready = unsafe { libc::poll(&mut descriptor, 1, 250) };
            assert!(
                ready >= 0,
                "PTY poll failed: {}",
                std::io::Error::last_os_error()
            );
            if ready == 0 {
                continue;
            }
            match self.master.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => self.transcript.extend_from_slice(&buffer[..count]),
                Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => panic!("PTY read failed: {error}"),
            }
        }
        String::from_utf8_lossy(&self.transcript).into_owned()
    }

    fn finish(mut self) -> (Option<i32>, String) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut buffer = [0_u8; 4096];
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return (
                    status.code(),
                    String::from_utf8_lossy(&self.transcript).into_owned(),
                );
            }
            assert!(Instant::now() < deadline, "workbench did not exit");
            let mut descriptor = libc::pollfd {
                fd: self.master.as_raw_fd(),
                events: libc::POLLIN | libc::POLLHUP,
                revents: 0,
            };
            // SAFETY: descriptor points to one initialized pollfd for the call.
            let ready = unsafe { libc::poll(&mut descriptor, 1, 100) };
            assert!(
                ready >= 0,
                "PTY poll failed: {}",
                std::io::Error::last_os_error()
            );
            if ready > 0 {
                match self.master.read(&mut buffer) {
                    Ok(count) => self.transcript.extend_from_slice(&buffer[..count]),
                    Err(error) if error.raw_os_error() == Some(libc::EIO) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(error) => panic!("PTY read failed: {error}"),
                }
            }
        }
    }
}

impl Drop for WorkbenchPty {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(unix)]
#[test]
fn newer_cli_preview_supersedes_waiting_workbench_request_without_trial_offer() {
    if !has_torch() {
        eprintln!("skipped CLI preview generation race: PyTorch is required");
        return;
    }

    let workspace = tempfile::tempdir().unwrap();
    let project = workspace.path().join("preview-race");
    fs::create_dir(&project).unwrap();
    fs::write(
        project.join("race_data.py"),
        r#"
import time
import torch
from torch import nn

def build_model():
    return nn.Linear(4, 2)

def slow_batch():
    time.sleep(15)
    return torch.ones(1, 4)

def fast_batch():
    return torch.full((1, 4), 2.0)
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
            "race_data:build_model",
            "--batch",
            "race_data:slow_batch",
        ])
        .current_dir(&project)
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "project init failed: {}",
        String::from_utf8_lossy(&init.stdout)
    );

    let mut workbench = WorkbenchPty::spawn(&project);
    // Home -> outputs -> launch slow batch preview. Blank fields retain the
    // saved batch entrypoint and defaults; select the same preview page again
    // to enter its live background-request view.
    workbench.send(b"6\n4\n\n\n\n\n\n\n\n\n\nstart\n4\n");
    let waiting = workbench.read_until(
        &["后台预览仍在运行", "wait 等待结束"],
        Duration::from_secs(15),
    );
    assert!(waiting.contains("后台预览仍在运行"), "{waiting}");

    // This independent CLI process publishes the next real generation while
    // the workbench-owned first worker is still inside slow_batch().
    let newer = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "data",
            "preview",
            "--batch",
            "race_data:fast_batch",
            "--samples",
            "1",
            "--renderer",
            "summary",
            "--fingerprint",
            "second-cli-generation",
        ])
        .current_dir(&project)
        .output()
        .unwrap();
    let newer_stdout = String::from_utf8_lossy(&newer.stdout);
    assert!(
        newer.status.success(),
        "newer CLI preview failed: {newer_stdout}; {}",
        String::from_utf8_lossy(&newer.stderr)
    );
    let newer_report: serde_json::Value = serde_json::from_str(newer_stdout.trim()).unwrap();
    assert_eq!(newer_report["status"], "success", "{newer_report:#}");
    assert_eq!(newer_report["data"]["generation"], 2, "{newer_report:#}");
    assert_eq!(newer_report["data"]["published"], true, "{newer_report:#}");

    workbench.send(b"wait\n");
    let result = workbench.read_until(
        &["本次请求已被较新的预览取代", "选择操作"],
        Duration::from_secs(45),
    );
    assert!(result.contains("不属于当前结果"), "{result}");
    assert!(result.contains("当前请求："), "{result}");
    assert!(
        !result.contains("当前预览 ·"),
        "superseded generation was labeled current: {result}"
    );
    assert!(
        !result.contains("可将当前模型 `"),
        "superseded generation must not offer a managed-step trial: {result}"
    );

    let current: serde_json::Value = serde_json::from_slice(
        &fs::read(project.join(".model-maven/previews/current.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(current["generation"], 2, "{current:#}");
    assert_eq!(current["current"]["generation"], 2, "{current:#}");
    assert_eq!(current["current"]["status"], "succeeded", "{current:#}");
    assert_eq!(
        current["current"]["request_id"],
        current["last_success"]["request_id"]
    );

    workbench.send(b"back\nq\n");
    let (exit_code, transcript) = workbench.finish();
    assert_eq!(exit_code, Some(0), "{transcript}");
}
