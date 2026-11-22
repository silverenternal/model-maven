#[cfg(target_os = "linux")]
mod linux {
    use serde_json::Value;
    use std::{
        fs,
        io::{Read, Write},
        os::fd::FromRawFd,
        os::unix::process::{CommandExt, ExitStatusExt},
        path::Path,
        process::{Child, Command, Output, Stdio},
        thread,
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

    fn invoke(root: &Path, args: &[&str]) -> (Output, Value) {
        let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args(["--json", "--python", &python()])
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

    fn assert_success(output: &Output, report: &Value) {
        assert!(
            output.status.success(),
            "unexpected failure: {report:#}\nstderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(report["status"], "success", "{report:#}");
    }

    fn wait_for_terminal(root: &Path, run_id: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let (_, report) = invoke(root, &["runs", "show", run_id]);
            if !matches!(
                report["data"]["record"]["status"].as_str(),
                Some("running" | "stopping")
            ) {
                return report;
            }
            assert!(Instant::now() < deadline, "run did not stop: {report:#}");
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn stop_child_on_timeout(mut child: Child) -> Output {
        let deadline = Instant::now() + Duration::from_secs(12);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                let _ = status;
                return child.wait_with_output().unwrap();
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                panic!("attach viewer did not exit on Ctrl+C");
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    struct ProcessGroupGuard(libc::pid_t);

    impl Drop for ProcessGroupGuard {
        fn drop(&mut self) {
            if self.0 > 0 {
                unsafe {
                    libc::kill(-self.0, libc::SIGKILL);
                }
            }
        }
    }

    fn interactive_request_stop(root: &Path) -> String {
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
        terminal
            .write_all(
                b"5\n3\n5\n1\nprogress\nback\nback\nback\n5\n1\nprogress\nstop\ny\nback\nback\nback\nquit\n",
            )
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut transcript = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let mut poll_fd = libc::pollfd {
                fd: master_fd,
                events: libc::POLLIN | libc::POLLHUP,
                revents: 0,
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            let timeout = remaining.as_millis().min(i32::MAX as u128) as i32;
            let ready = unsafe { libc::poll(&mut poll_fd, 1, timeout) };
            if ready <= 0 {
                let _ = child.kill();
                panic!("interactive run stop timed out");
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

    #[test]
    fn detached_run_heartbeats_attach_is_read_only_and_stop_is_cooperative() {
        if !has_torch() {
            eprintln!(
                "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
            );
            return;
        }
        let project = tempfile::tempdir().unwrap();
        fs::write(
            project.path().join("loop.py"),
            r#"
import time

def run_loop(model, context):
    for _ in range(context.remaining_steps):
        time.sleep(0.15)
        if not context.report_step(0.5, {"updates": 1}):
            break
    return {"finished": True}

run_loop.MODEL_MAVEN_STATELESS = True
"#,
        )
        .unwrap();

        let (output, started) = invoke(
            project.path(),
            &[
                "train",
                "--training-mode",
                "external-loop",
                "--loop",
                "loop:run_loop",
                "--steps",
                "100",
                "--detach",
                "--timeout",
                "120",
            ],
        );
        assert_success(&output, &started);
        let run_id = started["data"]["run_id"].as_str().unwrap();

        let (show_output, observed) = invoke(project.path(), &["runs", "show", run_id]);
        assert_success(&show_output, &observed);
        assert_eq!(observed["data"]["record"]["supervisor"]["kind"], "detached");
        assert_eq!(observed["data"]["record"]["heartbeat_status"], "fresh");
        assert!(
            observed["data"]["record"]["heartbeat_age_ms"]
                .as_u64()
                .is_some()
        );

        let viewer = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args(["runs", "attach", run_id, "--follow=true"])
            .current_dir(project.path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        thread::sleep(Duration::from_millis(600));
        assert_eq!(
            unsafe { libc::kill(viewer.id() as libc::pid_t, libc::SIGINT) },
            0
        );
        let viewer_output = stop_child_on_timeout(viewer);
        assert!(viewer_output.status.success());
        assert!(String::from_utf8_lossy(&viewer_output.stderr).contains("后台运行继续"));

        let (_, still_running) = invoke(project.path(), &["runs", "show", run_id]);
        assert_eq!(still_running["data"]["record"]["status"], "running");
        let transcript = interactive_request_stop(project.path());
        assert!(transcript.contains("停止后台训练"), "{transcript}");
        assert!(
            transcript.contains("向此运行请求在安全检查点停止？"),
            "{transcript}"
        );
        assert!(
            transcript.matches("后台运行 · 运行中").count() >= 2,
            "leaving the first progress view must leave the run active: {transcript}"
        );
        assert!(transcript.contains(run_id), "{transcript}");
        let finished = wait_for_terminal(project.path(), run_id);
        assert_eq!(finished["data"]["record"]["status"], "interrupted");
        assert_eq!(finished["data"]["record"]["supervisor"]["kind"], "detached");

        let (output, forced_start) = invoke(
            project.path(),
            &[
                "train",
                "--training-mode",
                "external-loop",
                "--loop",
                "loop:run_loop",
                "--steps",
                "100",
                "--detach",
                "--timeout",
                "120",
            ],
        );
        assert_success(&output, &forced_start);
        let forced_id = forced_start["data"]["run_id"].as_str().unwrap();
        let (stop_output, forced_stop) =
            invoke(project.path(), &["runs", "stop", forced_id, "--force"]);
        assert_success(&stop_output, &forced_stop);
        let forced_finished = wait_for_terminal(project.path(), forced_id);
        assert_eq!(forced_finished["data"]["record"]["status"], "interrupted");
        assert_eq!(
            forced_finished["data"]["report"]["diagnostics"][0]["code"],
            "USER_INTERRUPTED"
        );
    }

    #[test]
    fn externally_killed_supervisor_is_unknown_and_keeps_last_checkpoint() {
        if !has_torch() {
            eprintln!(
                "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
            );
            return;
        }
        let project = tempfile::tempdir().unwrap();
        fs::write(
            project.path().join("loop.py"),
            r#"
import time

def run_loop(model, context):
    for _ in range(context.remaining_steps):
        if not context.report_step(0.5, {"updates": 1}):
            break
        time.sleep(0.2)
    return {"finished": True}

run_loop.MODEL_MAVEN_STATELESS = True
"#,
        )
        .unwrap();

        let (output, started) = invoke(
            project.path(),
            &[
                "train",
                "--training-mode",
                "external-loop",
                "--loop",
                "loop:run_loop",
                "--steps",
                "1000",
                "--checkpoint-every",
                "1",
                "--detach",
                "--timeout",
                "120",
            ],
        );
        assert_success(&output, &started);
        let run_id = started["data"]["run_id"].as_str().unwrap();
        let run_dir = project.path().join(".model-maven/runs").join(run_id);
        let checkpoint_dir = run_dir.join("checkpoints");
        let deadline = Instant::now() + Duration::from_secs(20);
        let latest_manifest = loop {
            let path = checkpoint_dir.join("latest.json");
            if let Ok(contents) = fs::read(&path) {
                let parsed: Value = serde_json::from_slice(&contents).unwrap();
                if parsed["file"].as_str().is_some() {
                    break contents;
                }
            }
            assert!(
                Instant::now() < deadline,
                "checkpoint was not saved before fault injection"
            );
            thread::sleep(Duration::from_millis(50));
        };
        let manifest: Value = serde_json::from_slice(&latest_manifest).unwrap();
        let checkpoint_path = checkpoint_dir.join(manifest["file"].as_str().unwrap());
        let checkpoint_before = fs::read(&checkpoint_path).unwrap();
        assert!(!checkpoint_before.is_empty());

        let (show_output, active) = invoke(project.path(), &["runs", "show", run_id]);
        assert_success(&show_output, &active);
        let pid = active["data"]["record"]["supervisor"]["pid"]
            .as_i64()
            .unwrap() as libc::pid_t;
        let _process_group_guard = ProcessGroupGuard(pid);
        assert_eq!(
            unsafe { libc::kill(pid, libc::SIGKILL) },
            0,
            "external supervisor SIGKILL failed"
        );

        let deadline = Instant::now() + Duration::from_secs(10);
        let unknown = loop {
            let (show_output, report) = invoke(project.path(), &["runs", "show", run_id]);
            assert_success(&show_output, &report);
            if report["data"]["record"]["status"] == "unknown" {
                break report;
            }
            assert!(
                Instant::now() < deadline,
                "killed supervisor was not reconciled: {report:#}"
            );
            thread::sleep(Duration::from_millis(50));
        };
        assert_eq!(unknown["data"]["record"]["last_observed_status"], "running");
        assert!(
            unknown["data"]["record"]["status_note"]
                .as_str()
                .unwrap_or_default()
                .contains("监督进程已退出")
        );
        assert_eq!(
            fs::read(checkpoint_dir.join("latest.json")).unwrap(),
            latest_manifest
        );
        assert_eq!(fs::read(&checkpoint_path).unwrap(), checkpoint_before);

        // The supervisor was deliberately killed without running its cleanup.
        // Kill the remaining worker process group after validating the recovery point.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
        thread::sleep(Duration::from_millis(100));
        let (show_output, after_cleanup) = invoke(project.path(), &["runs", "show", run_id]);
        assert_success(&show_output, &after_cleanup);
        assert_eq!(after_cleanup["data"]["record"]["status"], "unknown");
        assert_eq!(
            fs::read(checkpoint_dir.join("latest.json")).unwrap(),
            latest_manifest
        );
        assert_eq!(fs::read(&checkpoint_path).unwrap(), checkpoint_before);

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
        assert_eq!(resumed["data"]["parent_run_id"], run_id);
        assert_eq!(
            resumed["data"]["resume_state_status"], "stateless",
            "{resumed:#}"
        );
        let resumed_run_id = resumed["data"]["run_id"].as_str().unwrap();
        assert_ne!(resumed_run_id, run_id);
        let (resumed_show_output, resumed_record) =
            invoke(project.path(), &["runs", "show", resumed_run_id]);
        assert_success(&resumed_show_output, &resumed_record);
        assert_eq!(resumed_record["data"]["record"]["status"], "completed");
        assert_eq!(
            fs::read(checkpoint_dir.join("latest.json")).unwrap(),
            latest_manifest,
            "resuming must not mutate the parent's recovery pointer"
        );
        assert_eq!(
            fs::read(&checkpoint_path).unwrap(),
            checkpoint_before,
            "resuming must preserve the parent's recovery bytes"
        );
    }

    #[test]
    fn controlling_terminal_hangup_detaches_viewer_without_stopping_run_or_checkpoint() {
        if !has_torch() {
            eprintln!(
                "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
            );
            return;
        }
        let project = tempfile::tempdir().unwrap();
        fs::write(
            project.path().join("loop.py"),
            r#"
import time

def run_loop(model, context):
    context.report_step(0.5, {"updates": 1})
    while not context.stop_requested:
        time.sleep(0.05)
    context.report_step(0.5, {"updates": 1})
    return {"finished": True}

run_loop.MODEL_MAVEN_STATELESS = True
"#,
        )
        .unwrap();

        let (start_output, started) = invoke(
            project.path(),
            &[
                "train",
                "--training-mode",
                "external-loop",
                "--loop",
                "loop:run_loop",
                "--steps",
                "1000",
                "--checkpoint-every",
                "1",
                "--detach",
                "--timeout",
                "120",
            ],
        );
        assert_success(&start_output, &started);
        let run_id = started["data"]["run_id"].as_str().unwrap();
        let run_dir = project.path().join(".model-maven/runs").join(run_id);
        let checkpoint_dir = run_dir.join("checkpoints");
        let deadline = Instant::now() + Duration::from_secs(20);
        let latest_manifest = loop {
            let path = checkpoint_dir.join("latest.json");
            if let Ok(contents) = fs::read(&path) {
                let parsed: Value = serde_json::from_slice(&contents).unwrap();
                if parsed["file"].as_str().is_some() {
                    break contents;
                }
            }
            assert!(
                Instant::now() < deadline,
                "run did not commit its first checkpoint"
            );
            thread::sleep(Duration::from_millis(50));
        };
        let checkpoint_manifest: Value = serde_json::from_slice(&latest_manifest).unwrap();
        let checkpoint_path = checkpoint_dir.join(checkpoint_manifest["file"].as_str().unwrap());
        let checkpoint_before = fs::read(&checkpoint_path).unwrap();
        assert!(!checkpoint_before.is_empty());

        let (master_fd, slave_fd) = {
            let (mut master, mut slave) = (-1, -1);
            let result = unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(result, 0, "openpty failed");
            (master, slave)
        };
        let descriptor_flags = unsafe { libc::fcntl(master_fd, libc::F_GETFD) };
        assert!(
            descriptor_flags >= 0,
            "cannot inspect PTY master descriptor"
        );
        assert_eq!(
            unsafe {
                libc::fcntl(
                    master_fd,
                    libc::F_SETFD,
                    descriptor_flags | libc::FD_CLOEXEC,
                )
            },
            0,
            "cannot prevent the attach child from inheriting the PTY master"
        );
        let (show_output, running) = invoke(project.path(), &["runs", "show", run_id]);
        assert_success(&show_output, &running);
        let supervisor_pid = running["data"]["record"]["supervisor"]["pid"]
            .as_i64()
            .unwrap() as libc::pid_t;
        let _run_process_group_guard = ProcessGroupGuard(supervisor_pid);
        let slave = unsafe { fs::File::from_raw_fd(slave_fd) };
        let stdout = slave.try_clone().unwrap();
        let stderr = slave.try_clone().unwrap();
        let mut viewer_command = Command::new(env!("CARGO_BIN_EXE_m-model"));
        viewer_command
            .args(["runs", "attach", run_id, "--follow=true"])
            .current_dir(project.path())
            .stdin(Stdio::from(slave))
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        unsafe {
            viewer_command.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::ioctl(libc::STDIN_FILENO, libc::TIOCSCTTY, 0) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut viewer = viewer_command.spawn().unwrap();
        let _viewer_process_group_guard = ProcessGroupGuard(viewer.id() as libc::pid_t);
        let mut terminal = unsafe { fs::File::from_raw_fd(master_fd) };
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut transcript = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let mut poll_fd = libc::pollfd {
                fd: master_fd,
                events: libc::POLLIN | libc::POLLHUP,
                revents: 0,
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            let timeout = remaining.as_millis().min(i32::MAX as u128) as i32;
            assert!(
                timeout > 0,
                "attach viewer did not render into its terminal"
            );
            let ready = unsafe { libc::poll(&mut poll_fd, 1, timeout) };
            assert!(ready > 0, "attach viewer did not render into its terminal");
            match terminal.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => {
                    transcript.extend_from_slice(&buffer[..count]);
                    let rendered = String::from_utf8_lossy(&transcript);
                    if rendered.contains(&format!("已附加到 {run_id}")) {
                        break;
                    }
                }
                Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
                Err(error) => panic!("cannot read attach terminal: {error}"),
            }
        }
        let rendered = String::from_utf8_lossy(&transcript);
        assert!(
            rendered.contains(&format!("已附加到 {run_id}")),
            "{rendered}"
        );

        // Closing the PTY master hangs up the viewer's controlling terminal,
        // as happens when its terminal session disappears. The detached run
        // has its own session and must continue independently.
        drop(terminal);
        let exit_deadline = Instant::now() + Duration::from_secs(5);
        let viewer_status = loop {
            if let Some(status) = viewer.try_wait().unwrap() {
                break status;
            }
            assert!(
                Instant::now() < exit_deadline,
                "viewer survived terminal hangup"
            );
            thread::sleep(Duration::from_millis(25));
        };
        assert_eq!(viewer_status.signal(), Some(libc::SIGHUP));

        let (show_output, after_hangup) = invoke(project.path(), &["runs", "show", run_id]);
        assert_success(&show_output, &after_hangup);
        assert_eq!(after_hangup["data"]["record"]["status"], "running");
        assert_eq!(
            after_hangup["data"]["record"]["supervisor"]["kind"],
            "detached"
        );
        assert_eq!(after_hangup["data"]["record"]["heartbeat_status"], "fresh");
        assert_eq!(
            fs::read(checkpoint_dir.join("latest.json")).unwrap(),
            latest_manifest
        );
        assert_eq!(fs::read(&checkpoint_path).unwrap(), checkpoint_before);

        let viewer = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args(["runs", "attach", run_id, "--follow=true"])
            .current_dir(project.path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        thread::sleep(Duration::from_millis(600));
        assert_eq!(
            unsafe { libc::kill(viewer.id() as libc::pid_t, libc::SIGINT) },
            0,
            "reconnected attach viewer did not accept SIGINT"
        );
        let viewer_output = stop_child_on_timeout(viewer);
        assert!(viewer_output.status.success());
        assert!(String::from_utf8_lossy(&viewer_output.stderr).contains("后台运行继续"));
        let (show_output, after_reconnect) = invoke(project.path(), &["runs", "show", run_id]);
        assert_success(&show_output, &after_reconnect);
        assert_eq!(after_reconnect["data"]["record"]["status"], "running");
        assert_eq!(
            after_reconnect["data"]["record"]["heartbeat_status"],
            "fresh"
        );
        assert_eq!(
            fs::read(checkpoint_dir.join("latest.json")).unwrap(),
            latest_manifest
        );
        assert_eq!(fs::read(&checkpoint_path).unwrap(), checkpoint_before);

        let (stop_output, stopped) = invoke(project.path(), &["runs", "stop", run_id]);
        assert_success(&stop_output, &stopped);
        let terminal = wait_for_terminal(project.path(), run_id);
        assert_eq!(terminal["data"]["record"]["status"], "interrupted");
        assert_eq!(terminal["data"]["record"]["supervisor"]["kind"], "detached");
    }
}
