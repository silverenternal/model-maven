#[path = "../src/presentation/widgets/layout.rs"]
mod layout;
#[path = "../src/presentation/widgets/text.rs"]
mod text;

use std::{
    fs,
    process::{Command, Stdio},
};

#[test]
fn human_text_layout_is_readable_at_40_80_and_120_columns() {
    let source = "模型目录：/workspace/项目/very-long-model-name/attention_encoder.py";
    for width in [40, 80, 120] {
        let content_width = layout::content_width(width);
        let lines = text::wrap(source, content_width);
        assert!(
            lines
                .iter()
                .all(|line| text::display_width(line) <= content_width)
        );
        assert_eq!(lines.concat(), source);
    }
}

#[test]
fn unsafe_controls_and_color_sequences_do_not_leak_into_plain_text() {
    let value = "\x1b[31m错误\x1b[0m\x1b]0;title\x07\x07";
    assert_eq!(text::plain_text(value), "错误�");
    assert!(!layout::color_policy(false, false, false));
    assert!(!layout::color_policy(true, true, false));
    assert!(!layout::color_policy(true, false, true));
    assert!(!layout::color_enabled(false));
    let width = layout::terminal_width();
    assert!((layout::MIN_WIDTH..=layout::MAX_WIDTH).contains(&width));
    const { assert!(layout::DEFAULT_WIDTH >= layout::MIN_WIDTH) };
    let _stdout_policy = layout::stdout_color_enabled();
}

#[test]
fn long_paths_are_available_as_wrapped_details_and_pages_are_stable() {
    let path = "/project/models/encoder/stages/stage_12/attention/custom_attention_layer.py";
    let lines = text::wrap(path, 40);
    assert_eq!(lines.concat(), path);
    assert!(lines.iter().all(|line| text::display_width(line) <= 40));

    let pages = layout::paginate(&lines, 2);
    assert_eq!(pages.concat().concat(), path);
}

#[cfg(unix)]
#[test]
fn terminal_width_tracks_pty_resizes_without_using_a_cached_value() {
    use std::os::fd::RawFd;
    use std::sync::Mutex;

    static ENV_AND_STDERR_LOCK: Mutex<()> = Mutex::new(());
    let _lock = ENV_AND_STDERR_LOCK.lock().unwrap();

    struct RestoreStderr(RawFd);
    impl Drop for RestoreStderr {
        fn drop(&mut self) {
            // SAFETY: the saved descriptor remains open until this guard drops.
            unsafe {
                libc::dup2(self.0, libc::STDERR_FILENO);
                libc::close(self.0);
            }
        }
    }

    struct RestoreColumns(Option<std::ffi::OsString>);
    impl Drop for RestoreColumns {
        fn drop(&mut self) {
            // SAFETY: environment access is serialized by ENV_AND_STDERR_LOCK.
            unsafe {
                match &self.0 {
                    Some(value) => std::env::set_var("COLUMNS", value),
                    None => std::env::remove_var("COLUMNS"),
                }
            }
        }
    }

    let old_columns = std::env::var_os("COLUMNS");
    // SAFETY: this test serializes its environment mutation and restores it.
    unsafe { std::env::remove_var("COLUMNS") };
    let _restore_columns = RestoreColumns(old_columns);

    let saved_stderr = unsafe { libc::dup(libc::STDERR_FILENO) };
    assert!(saved_stderr >= 0, "could not duplicate stderr");
    let _restore_stderr = RestoreStderr(saved_stderr);

    let mut master = -1;
    let mut slave = -1;
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    size.ws_row = 24;
    size.ws_col = 40;
    // SAFETY: openpty initializes both descriptors and receives a valid winsize.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null(),
                &size,
            )
        },
        0
    );
    struct CloseFd(RawFd);
    impl Drop for CloseFd {
        fn drop(&mut self) {
            // SAFETY: this guard owns the descriptor returned by openpty.
            unsafe { libc::close(self.0) };
        }
    }
    let _master = CloseFd(master);
    let _slave = CloseFd(slave);

    // SAFETY: slave is an open PTY descriptor and stderr is restored by its guard.
    assert_eq!(
        unsafe { libc::dup2(slave, libc::STDERR_FILENO) },
        libc::STDERR_FILENO
    );
    assert_eq!(layout::terminal_width(), 40);

    size.ws_col = 120;
    // SAFETY: TIOCSWINSZ updates the valid PTY slave descriptor.
    assert_eq!(unsafe { libc::ioctl(slave, libc::TIOCSWINSZ, &size) }, 0);
    assert_eq!(layout::terminal_width(), 120);
}

#[test]
fn environment_color_flags_disable_color_for_terminal_streams() {
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());
    let _lock = ENV_LOCK.lock().unwrap();

    struct RestoreEnv {
        no_color: Option<std::ffi::OsString>,
        term: Option<std::ffi::OsString>,
    }
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            // SAFETY: this test serializes its environment mutations and restores them.
            unsafe {
                match &self.no_color {
                    Some(value) => std::env::set_var("NO_COLOR", value),
                    None => std::env::remove_var("NO_COLOR"),
                }
                match &self.term {
                    Some(value) => std::env::set_var("TERM", value),
                    None => std::env::remove_var("TERM"),
                }
            }
        }
    }

    let restore = RestoreEnv {
        no_color: std::env::var_os("NO_COLOR"),
        term: std::env::var_os("TERM"),
    };
    // SAFETY: this test serializes its environment mutations and restores them.
    unsafe {
        std::env::remove_var("NO_COLOR");
        std::env::set_var("TERM", "xterm-256color");
    }
    assert!(layout::color_enabled(true));

    unsafe { std::env::set_var("NO_COLOR", "1") };
    assert!(!layout::color_enabled(true));

    unsafe {
        std::env::remove_var("NO_COLOR");
        std::env::set_var("TERM", "dumb");
    }
    assert!(!layout::color_enabled(true));
    drop(restore);
}

#[test]
fn redirected_cli_help_and_argument_errors_keep_stdout_and_stderr_separate() {
    let binary = env!("CARGO_BIN_EXE_m-model");
    let help = Command::new(binary)
        .arg("--help")
        .env_remove("NO_COLOR")
        .env_remove("CLICOLOR_FORCE")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(help.stdout.windows(5).any(|window| window == b"Usage"));
    assert!(
        help.stderr.is_empty(),
        "help leaked to stderr: {:?}",
        help.stderr
    );
    assert!(!help.stdout.contains(&0x1b));
    assert!(!help.stderr.contains(&0x1b));

    let plain_help = Command::new(binary)
        .arg("--help")
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .output()
        .unwrap();
    assert!(plain_help.status.success());
    assert!(plain_help.stderr.is_empty());
    assert!(!plain_help.stdout.contains(&0x1b));

    let invalid = Command::new(binary)
        .arg("--definitely-not-a-real-option")
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(
        invalid.stdout.is_empty(),
        "error leaked to stdout: {:?}",
        invalid.stdout
    );
    assert!(!invalid.stderr.is_empty());
    assert!(!invalid.stdout.contains(&0x1b));
    assert!(!invalid.stderr.contains(&0x1b));
}

#[cfg(unix)]
#[test]
fn workbench_component_and_run_lists_paginate_at_40_columns() {
    use std::io::{Read, Write};
    use std::os::fd::FromRawFd;

    let project = tempfile::tempdir().unwrap();
    let initialized = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["init", "--model", "fixture:build_model"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stderr)
    );

    let components = project.path().join(".model-maven/components/example");
    for index in 0..12 {
        let id = format!("example/component-{index:02}");
        let directory = components.join(format!("component-{index:02}"));
        std::fs::create_dir_all(&directory).unwrap();
        let manifest = serde_json::json!({
            "schema_version": 1,
            "namespace": "example",
            "component_id": id,
            "component_version": "1.0.0",
            "extension_api_version": 1,
            "kind": "layer",
            "entrypoint": "not_imported:factory",
            "description": format!("catalog entry {index:02}"),
            "params_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            },
            "capabilities": {"construct": {"state": "supported"}}
        });
        std::fs::write(
            directory.join("1.0.0.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    let runs = project.path().join(".model-maven/runs");
    std::fs::create_dir_all(&runs).unwrap();
    for index in 0..12 {
        let id = format!("run-layout-{index:02}");
        let directory = runs.join(&id);
        std::fs::create_dir_all(&directory).unwrap();
        let metadata = serde_json::json!({
            "schema_version": 1,
            "id": id,
            "name": format!("run-page-{index:02}"),
            "kept": false,
            "operation": "training.managed_step",
            "training_mode": "managed_step",
            "requested_steps": 1,
            "status": "completed",
            "started_at_ms": index + 1,
            "finished_at_ms": index + 2,
            "supervisor": null,
            "parent_run_id": null,
            "checkpoint_released": true,
            "summary": {}
        });
        std::fs::write(
            directory.join("run.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
    }

    let redirected_runs = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["runs", "list"])
        .current_dir(project.path())
        .env_remove("CLICOLOR_FORCE")
        .output()
        .unwrap();
    assert!(
        redirected_runs.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&redirected_runs.stdout),
        String::from_utf8_lossy(&redirected_runs.stderr)
    );
    let redirected_stdout = String::from_utf8_lossy(&redirected_runs.stdout);
    assert!(redirected_stdout.contains("run-page-11"));
    assert!(redirected_runs.stderr.is_empty());
    assert!(!redirected_runs.stdout.contains(&0x1b));

    let (mut master_fd, mut slave_fd) = (-1, -1);
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    size.ws_row = 24;
    size.ws_col = 40;
    // SAFETY: openpty initializes both descriptors and receives a valid winsize.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master_fd,
                &mut slave_fd,
                std::ptr::null_mut(),
                std::ptr::null(),
                &size,
            )
        },
        0
    );
    // SAFETY: openpty returned ownership of the slave descriptor.
    let slave = unsafe { std::fs::File::from_raw_fd(slave_fd) };
    let stdout = slave.try_clone().unwrap();
    let stderr = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(project.path())
        .stdin(Stdio::from(slave))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .env("COLUMNS", "40")
        .spawn()
        .unwrap();
    // SAFETY: openpty returned ownership of the master descriptor.
    let mut terminal = unsafe { std::fs::File::from_raw_fd(master_fd) };
    terminal
        .write_all(b"4\n1\n\nn\np\n\nback\n3\n5\nn\np\nback\nback\nq\n")
        .unwrap();

    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut poll_fd = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        // SAFETY: poll receives one valid descriptor and a pointer to its pollfd.
        let ready = unsafe { libc::poll(&mut poll_fd, 1, 30_000) };
        assert!(
            ready > 0,
            "workbench timed out: {}",
            String::from_utf8_lossy(&transcript)
        );
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read workbench PTY: {error}"),
        }
    }
    let status = child.wait().unwrap();
    let transcript = String::from_utf8_lossy(&transcript);
    assert!(
        status.success(),
        "workbench exited with {status}: {transcript}"
    );
    assert!(
        transcript.contains("运行记录 · 12 条匹配 · 第 1/2 页"),
        "{transcript}"
    );
    assert!(
        transcript.contains("运行记录 · 12 条匹配 · 第 2/2 页"),
        "{transcript}"
    );
    assert!(transcript.contains("run-page-11"), "{transcript}");
    assert!(transcript.contains("run-page-01"), "{transcript}");
    assert!(transcript.contains("n/p 翻页"), "{transcript}");
    assert!(
        transcript.contains("组件目录 · 12 个组件 · 第 1/2 页"),
        "{transcript}"
    );
    assert!(
        transcript.contains("组件目录 · 12 个组件 · 第 2/2 页"),
        "{transcript}"
    );
    assert!(transcript.contains("component-11"), "{transcript}");
    assert!(transcript.contains("component-01"), "{transcript}");
}

#[test]
fn workbench_model_tree_and_probe_calls_paginate_at_40_columns() {
    use std::io::{Read, Write};
    use std::os::fd::FromRawFd;

    let project = tempfile::tempdir().unwrap();
    std::fs::write(
        project.path().join("pager_model.py"),
        "import torch\nfrom torch import nn\n\ndef build_model():\n    return nn.Sequential(*[nn.Linear(2, 2) for _ in range(45)])\n\ndef example_batch():\n    return torch.ones(1, 2)\n",
    )
    .unwrap();
    let initialized = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "init",
            "--model",
            "pager_model:build_model",
            "--batch",
            "pager_model:example_batch",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stderr)
    );

    let (mut master_fd, mut slave_fd) = (-1, -1);
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    size.ws_row = 24;
    size.ws_col = 40;
    // SAFETY: openpty initializes both descriptors and receives a valid winsize.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master_fd,
                &mut slave_fd,
                std::ptr::null_mut(),
                std::ptr::null(),
                &size,
            )
        },
        0
    );
    // SAFETY: openpty returned ownership of the slave descriptor.
    let slave = unsafe { std::fs::File::from_raw_fd(slave_fd) };
    let stdout = slave.try_clone().unwrap();
    let stderr = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(project.path())
        .stdin(Stdio::from(slave))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .env("COLUMNS", "40")
        .spawn()
        .unwrap();
    // SAFETY: openpty returned ownership of the master descriptor.
    let mut terminal = unsafe { std::fs::File::from_raw_fd(master_fd) };
    terminal
        .write_all(b"1\n3\nn\np\nback\n2\n\n\n\nn\np\nback\nback\n8\n")
        .unwrap();

    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut poll_fd = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        // SAFETY: poll receives one valid descriptor and a pointer to its pollfd.
        let ready = unsafe { libc::poll(&mut poll_fd, 1, 30_000) };
        assert!(
            ready > 0,
            "workbench timed out: {}",
            String::from_utf8_lossy(&transcript)
        );
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read workbench PTY: {error}"),
        }
    }
    let status = child.wait().unwrap();
    let transcript = String::from_utf8_lossy(&transcript);
    assert!(
        status.success(),
        "workbench exited with {status}: {transcript}"
    );
    assert!(
        transcript.contains("模型树 · 46 个匹配节点"),
        "{transcript}"
    );
    assert!(transcript.contains("第 1/2 页"), "{transcript}");
    assert!(transcript.contains("第 2/2 页"), "{transcript}");
    assert!(transcript.contains("model.44"), "{transcript}");
    assert!(
        transcript.contains("逐次调用 · 共 46 次 · 第 1/2 页"),
        "{transcript}"
    );
    assert!(
        transcript.contains("逐次调用 · 共 46 次 · 第 2/2 页"),
        "{transcript}"
    );
    assert!(transcript.contains("model.44"), "{transcript}");
}

#[cfg(unix)]
#[test]
fn evaluation_sample_history_pages_and_displays_records_at_40_columns() {
    use std::io::{Read, Write};
    use std::os::fd::FromRawFd;

    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    let runtime = Command::new(&python).args(["-c", "import torch"]).status();
    if !runtime.is_ok_and(|status| status.success()) {
        eprintln!("skipped: evaluation pager PTY requires PyTorch");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("layout_eval.py"),
        r#"
import torch
from torch import nn

def build_model(): return nn.Linear(2, 2)
def batch(): return torch.ones(2, 2), torch.zeros(2, 2)
def step(model, batch, context):
    values, target = batch
    return {"loss": (model(values) - target).square().mean()}
def validation_data(): return {"values": torch.ones(8, 2)}
def evaluate(model, data, context):
    output = model(data["values"])
    context.report_samples(8)
    for index, row in enumerate(output):
        context.report_prediction(
            f"sample-{index:02}-long-visible-id",
            {"scores": row.tolist(), "detail": "x" * 96},
            {"target": "class-" + str(index)},
        )
    return {"metrics": {"fixture/score": 0.875}}
"#,
    )
    .unwrap();
    let trained = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            python.as_str(),
            "train",
            "--model",
            "layout_eval:build_model",
            "--batch",
            "layout_eval:batch",
            "--step",
            "layout_eval:step",
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
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&trained.stdout),
        String::from_utf8_lossy(&trained.stderr)
    );
    let trained: serde_json::Value = serde_json::from_slice(&trained.stdout).unwrap();
    let run_id = trained["data"]["run_id"].as_str().unwrap();
    let evaluated = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            python.as_str(),
            "evaluate",
            "--run",
            run_id,
            "--evaluator",
            "layout_eval:evaluate",
            "--data",
            "layout_eval:validation_data",
            "--samples",
            "8",
            "--preview-samples",
            "8",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        evaluated.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&evaluated.stdout),
        String::from_utf8_lossy(&evaluated.stderr)
    );

    let initialized = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["init", "--model", "layout_eval:build_model"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stderr)
    );

    let (mut master_fd, mut slave_fd) = (-1, -1);
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    size.ws_row = 24;
    size.ws_col = 40;
    // SAFETY: openpty initializes both descriptors and receives a valid winsize.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master_fd,
                &mut slave_fd,
                std::ptr::null_mut(),
                std::ptr::null(),
                &size,
            )
        },
        0
    );
    // SAFETY: openpty returned ownership of the slave descriptor.
    let slave = unsafe { std::fs::File::from_raw_fd(slave_fd) };
    let stdout = slave.try_clone().unwrap();
    let stderr = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(project.path())
        .stdin(Stdio::from(slave))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .env("COLUMNS", "40")
        .spawn()
        .unwrap();
    // SAFETY: openpty returned ownership of the master descriptor.
    let mut terminal = unsafe { std::fs::File::from_raw_fd(master_fd) };
    terminal
        .write_all(b"3\n5\n1\nsamples\nn\np\n5\nback\nback\nback\nq\n")
        .unwrap();

    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut poll_fd = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        // SAFETY: poll receives one valid descriptor and a pointer to its pollfd.
        let ready = unsafe { libc::poll(&mut poll_fd, 1, 30_000) };
        assert!(
            ready > 0,
            "workbench timed out: {}",
            String::from_utf8_lossy(&transcript)
        );
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read workbench PTY: {error}"),
        }
    }
    let status = child.wait().unwrap();
    let transcript = String::from_utf8_lossy(&transcript);
    assert!(
        status.success(),
        "workbench exited with {status}: {transcript}"
    );
    assert!(transcript.contains("运行详情 · 已完成"), "{transcript}");
    assert!(transcript.contains("fixture/score：0.875"), "{transcript}");
    assert!(transcript.contains("结构化样本预览：8 条"), "{transcript}");
    assert!(transcript.contains("样本预览 1–4 / 8"), "{transcript}");
    assert!(transcript.contains("样本预览 5–8 / 8"), "{transcript}");
    assert!(
        transcript.contains("sample-00-long-visible-id"),
        "{transcript}"
    );
    assert!(
        transcript.contains("sample-07-long-visible-id"),
        "{transcript}"
    );
    assert!(
        transcript.contains("\"detail\": \"xxxxxxxx"),
        "{transcript}"
    );
}
