use crate::{
    adapters::worker_assets::stage_worker_package,
    domain::{
        PROTOCOL_VERSION, Report,
        capability::{CapabilityCatalog, annotate_worker_request},
    },
    ports::{Backend, PythonRuntimeControl},
};
use serde_json::Value;
#[cfg(unix)]
use std::os::unix::{fs::OpenOptionsExt, process::CommandExt};
use std::{
    io::{Read, Write},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
#[cfg(unix)]
static PREINSTALLED_INTERRUPT: AtomicBool = AtomicBool::new(false);
pub struct PythonBackend {
    pub executable: String,
    pub timeout: Duration,
    pub verbose: bool,
}
#[cfg(unix)]
extern "C" fn interrupt(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}
#[cfg(unix)]
struct InterruptGuard(libc::sighandler_t);
#[cfg(unix)]
impl InterruptGuard {
    fn install() -> Self {
        if !PREINSTALLED_INTERRUPT.swap(false, Ordering::Relaxed)
            && !INTERRUPTED.load(Ordering::Relaxed)
        {
            INTERRUPTED.store(false, Ordering::Relaxed);
        }
        Self(unsafe { libc::signal(libc::SIGINT, interrupt as *const () as libc::sighandler_t) })
    }
}
#[cfg(unix)]
impl Drop for InterruptGuard {
    fn drop(&mut self) {
        unsafe {
            libc::signal(libc::SIGINT, self.0);
        }
    }
}

#[cfg(unix)]
pub struct EarlyInterruptGuard(libc::sighandler_t);

#[cfg(unix)]
impl Drop for EarlyInterruptGuard {
    fn drop(&mut self) {
        PREINSTALLED_INTERRUPT.store(false, Ordering::Relaxed);
        unsafe {
            libc::signal(libc::SIGINT, self.0);
        }
    }
}

impl PythonBackend {
    /// Consumes a pending Ctrl+C signal for read-only UI loops that do not run a worker.
    pub fn take_interrupted() -> bool {
        INTERRUPTED.swap(false, Ordering::Relaxed)
    }

    #[cfg(unix)]
    pub fn prepare_interrupt_handling() -> EarlyInterruptGuard {
        INTERRUPTED.store(false, Ordering::Relaxed);
        PREINSTALLED_INTERRUPT.store(true, Ordering::Relaxed);
        EarlyInterruptGuard(unsafe {
            libc::signal(libc::SIGINT, interrupt as *const () as libc::sighandler_t)
        })
    }
}

fn emit_progress_line(line: &[u8]) {
    let line = String::from_utf8_lossy(line);
    let Some(payload) = line.trim().strip_prefix("MODEL_MAVEN_PROGRESS ") else {
        return;
    };
    let Ok(value) = serde_json::from_str::<Value>(payload) else {
        return;
    };
    if let Some(message) = value["message"].as_str() {
        eprintln!("{message}");
        return;
    }
    let checkpoint = if value["checkpoint"] == true {
        " · 已保存 checkpoint"
    } else {
        ""
    };
    let loss = value["mean_loss"]
        .as_f64()
        .map(|loss| format!(" · 平均 Loss {loss:.5}"))
        .unwrap_or_default();
    eprintln!(
        "训练进度：{}/{} 步{loss}{checkpoint}",
        value["step"], value["total"]
    );
}

#[cfg(unix)]
fn open_progress_file(path: &std::path::Path) -> Option<std::fs::File> {
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return None;
    }
    let mut options = std::fs::OpenOptions::new();
    options
        .create(true)
        .append(true)
        .custom_flags(libc::O_NOFOLLOW);
    options.open(path).ok()
}

#[cfg(not(unix))]
fn open_progress_file(path: &std::path::Path) -> Option<std::fs::File> {
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return None;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()
}

#[cfg(unix)]
fn open_log_file(path: &std::path::Path) -> Option<std::fs::File> {
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return None;
    }
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    options.open(path).ok()
}

#[cfg(not(unix))]
fn open_log_file(path: &std::path::Path) -> Option<std::fs::File> {
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return None;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()
}

fn record_progress_line(file: &mut Option<std::fs::File>, line: &[u8]) {
    let line = String::from_utf8_lossy(line);
    let Some(payload) = line.trim().strip_prefix("MODEL_MAVEN_PROGRESS ") else {
        return;
    };
    let Ok(value) = serde_json::from_str::<Value>(payload) else {
        return;
    };
    if let Some(file) = file {
        if serde_json::to_writer(&mut *file, &value).is_ok() {
            let _ = file.write_all(b"\n");
        }
    }
}

fn drain(
    mut input: impl Read,
    limit: usize,
    forward_progress: bool,
    progress_path: Option<std::path::PathBuf>,
    log_path: Option<std::path::PathBuf>,
) -> Vec<u8> {
    let mut kept = Vec::new();
    let mut buffer = [0; 8192];
    let mut line_buffer = Vec::new();
    let mut progress_file = progress_path.as_deref().and_then(open_progress_file);
    let capture_progress = forward_progress || progress_file.is_some();
    let mut log_file = log_path.as_deref().and_then(open_log_file);
    while let Ok(count) = input.read(&mut buffer) {
        if count == 0 {
            break;
        }
        let retain = count.min(limit.saturating_sub(kept.len()));
        kept.extend_from_slice(&buffer[..retain]);
        if let Some(file) = &mut log_file {
            let _ = file.write_all(&buffer[..count]);
        }
        if capture_progress {
            line_buffer.extend_from_slice(&buffer[..count]);
            while let Some(end) = line_buffer.iter().position(|byte| *byte == b'\n') {
                let line = line_buffer.drain(..=end).collect::<Vec<_>>();
                record_progress_line(&mut progress_file, &line);
                if forward_progress {
                    emit_progress_line(&line);
                }
            }
        }
    }
    if capture_progress && !line_buffer.is_empty() {
        record_progress_line(&mut progress_file, &line_buffer);
        if forward_progress {
            emit_progress_line(&line_buffer);
        }
    }
    kept
}

fn request_training_timeout_stop(request: &Value) -> bool {
    if request["operation"] != "train" {
        return false;
    }
    let (Some(checkpoint), Some(stop_file)) = (
        request["checkpoint_dir"].as_str(),
        request["stop_file"].as_str(),
    ) else {
        return false;
    };
    let checkpoint = std::path::Path::new(checkpoint);
    let stop_file = std::path::Path::new(stop_file);
    if checkpoint
        .file_name()
        .is_none_or(|name| name != "checkpoints")
        || stop_file
            .file_name()
            .is_none_or(|name| name != "stop.request")
        || checkpoint.parent() != stop_file.parent()
    {
        return false;
    }
    let Some(run_directory) = checkpoint.parent() else {
        return false;
    };
    let Ok(run_metadata) = std::fs::symlink_metadata(run_directory) else {
        return false;
    };
    if run_metadata.file_type().is_symlink() || !run_metadata.is_dir() {
        return false;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    match options.open(stop_file) {
        Ok(mut file) => file.write_all(b"timeout\n").is_ok(),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            std::fs::symlink_metadata(stop_file)
                .is_ok_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
        }
        Err(_) => false,
    }
}

impl Backend for PythonBackend {
    fn execute(&self, request: &Value) -> Report {
        let operation = request["operation"].as_str().unwrap_or("unknown");
        let executable = request["runtime"]["python_executable"]
            .as_str()
            .unwrap_or(&self.executable);
        let timeout = Duration::from_secs(
            request["runtime"]["timeout_seconds"]
                .as_u64()
                .unwrap_or(self.timeout.as_secs())
                .max(1),
        );
        let mut request_envelope = request.clone();
        annotate_worker_request(&mut request_envelope);
        let encoded = serde_json::to_vec(&request_envelope).expect("JSON value serializes");
        if encoded.len() > 1024 * 1024 {
            return Report::error(
                operation,
                "REQUEST_TOO_LARGE",
                "描述超过 1 MiB 上限。",
                "批次通过 Python 函数提供，不将 Tensor 塞进配置。",
            );
        }
        let worker_directory = match tempfile::Builder::new()
            .prefix("model-maven-worker-")
            .tempdir()
        {
            Ok(directory) => directory,
            Err(error) => {
                return Report::error(
                    operation,
                    "WORKER_STAGING_FAILED",
                    error.to_string(),
                    "检查系统临时目录是否可写。",
                );
            }
        };
        let staging = stage_worker_package(worker_directory.path(), false);
        if let Err(error) = staging {
            return Report::error(
                operation,
                "WORKER_STAGING_FAILED",
                error.to_string(),
                "检查系统临时目录空间和权限。",
            );
        }
        let worker_path = worker_directory.path().join("worker.py");
        #[cfg(unix)]
        let _interrupt = InterruptGuard::install();
        let mut command = Command::new(executable);
        command
            .arg("-u")
            .arg(&worker_path)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        command.process_group(0);
        #[cfg(target_os = "linux")]
        {
            let owner_pid = std::process::id() as libc::pid_t;
            unsafe {
                command.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::getppid() != owner_pid {
                        return Err(std::io::Error::other(
                            "Model-Maven supervisor exited before the worker started",
                        ));
                    }
                    Ok(())
                });
            }
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                return Report::error(
                    operation,
                    "PYTHON_UNAVAILABLE",
                    error.to_string(),
                    "用 --python 指定可用的解释器。",
                );
            }
        };
        let started = Instant::now();
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let mut stdin = child.stdin.take().expect("piped stdin");
        let stream_progress = request["interactive_progress"].as_bool().unwrap_or(false);
        let progress_path = request["progress_path"].as_str().map(Into::into);
        let log_path = request["log_path"].as_str().map(Into::into);
        let output_limit = if operation == "edit" && request["edit"]["target"] == "source" {
            64 * 1024 * 1024
        } else {
            8 * 1024 * 1024
        };
        let (out_tx, out_rx) = mpsc::channel();
        let (err_tx, err_rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = out_tx.send(drain(stdout, output_limit, false, None, None));
        });
        thread::spawn(move || {
            let _ = err_tx.send(drain(
                stderr,
                1024 * 1024,
                stream_progress,
                progress_path,
                log_path,
            ));
        });
        thread::spawn(move || {
            let _ = stdin.write_all(&encoded);
        });
        let mut timed_out = false;
        let mut interrupted = false;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if started.elapsed() < timeout && !INTERRUPTED.load(Ordering::Relaxed) => {
                    thread::sleep(Duration::from_millis(20))
                }
                _ => {
                    interrupted = INTERRUPTED.load(Ordering::Relaxed);
                    timed_out = !interrupted;
                    #[cfg(unix)]
                    if interrupted {
                        // Give the worker a chance to roll back owned temporary
                        // state (for example, a partially prepared venv). Its
                        // subprocesses share this process group and receive the
                        // interrupt too. Force-kill only if cleanup stalls.
                        eprintln!("收到中断，等待 worker 清理临时状态…");
                        unsafe {
                            libc::kill(-(child.id() as libc::pid_t), libc::SIGINT);
                        }
                        let cleanup_deadline = Instant::now() + Duration::from_secs(10);
                        let graceful_status = loop {
                            match child.try_wait() {
                                Ok(Some(status)) => break Some(status),
                                Ok(None) if Instant::now() < cleanup_deadline => {
                                    thread::sleep(Duration::from_millis(20));
                                }
                                _ => {
                                    unsafe {
                                        libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
                                    }
                                    let _ = child.kill();
                                    break child.wait().ok();
                                }
                            }
                        };
                        break graceful_status;
                    } else {
                        if request_training_timeout_stop(request) {
                            eprintln!("训练达到超时，等待步骤边界保存恢复点（最多 10 秒）…");
                            let cleanup_deadline = Instant::now() + Duration::from_secs(10);
                            let graceful_status = loop {
                                match child.try_wait() {
                                    Ok(Some(status)) => break Some(status),
                                    Ok(None) if Instant::now() < cleanup_deadline => {
                                        thread::sleep(Duration::from_millis(20));
                                    }
                                    _ => {
                                        #[cfg(unix)]
                                        unsafe {
                                            libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
                                        }
                                        let _ = child.kill();
                                        break child.wait().ok();
                                    }
                                }
                            };
                            if graceful_status.is_some() {
                                // A trainer may have spawned children before honoring
                                // the stop request. Remove any remaining members of
                                // this worker-owned process group before returning.
                                #[cfg(unix)]
                                unsafe {
                                    libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
                                }
                            }
                            break graceful_status;
                        }
                        #[cfg(unix)]
                        unsafe {
                            libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
                        }
                        let _ = child.kill();
                        break child.wait().ok();
                    }
                    #[cfg(not(unix))]
                    {
                        let _ = child.kill();
                        break child.wait().ok();
                    }
                }
            }
        };
        let output = out_rx.recv_timeout(Duration::from_secs(1));
        let errors = err_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap_or_default();
        if self.verbose && !errors.is_empty() {
            let details = String::from_utf8_lossy(&errors)
                .lines()
                .filter(|line| !line.starts_with("MODEL_MAVEN_PROGRESS "))
                .collect::<Vec<_>>()
                .join("\n");
            if !details.is_empty() {
                eprintln!("{details}");
            }
        }
        if interrupted {
            return Report::error(
                operation,
                "USER_INTERRUPTED",
                "操作已取消，worker 进程组已终止。",
                "正式训练可从最近已提交 checkpoint 恢复；自行脱离进程组的用户子进程不在清理保证内。",
            );
        }
        if timed_out {
            return Report::error(
                operation,
                "WORKER_TIMEOUT",
                format!("操作超过 {} 秒，worker 进程组已终止。", timeout.as_secs()),
                "检查耗时入口或增加 --timeout。正式训练会先请求步骤边界保存恢复点；超时报告后可用 runs show 检查是否已有 checkpoint，再按训练帮助恢复。",
            );
        }
        if let Ok(output) = output
            && let Ok(report) = serde_json::from_slice::<Report>(&output)
            && report.protocol_version == PROTOCOL_VERSION
            && report.operation == operation
            && status.is_some_and(|s| s.success())
        {
            if operation == "doctor"
                && let Err(reason) = CapabilityCatalog::from_doctor_data(&report.data)
            {
                return Report::error(
                    operation,
                    "CAPABILITY_CATALOG_INVALID",
                    format!("worker 返回的能力目录不可用：{reason}"),
                    "升级或修复 Python worker，使 doctor 能力目录与兼容性契约一致。",
                );
            }
            return report;
        }
        let detail = if errors.is_empty() {
            format!("worker 未返回完整协议（{status:?}），可能有子进程占用输出管道。")
        } else {
            String::from_utf8_lossy(&errors)
                .chars()
                .take(2000)
                .collect()
        };
        Report::error(
            operation,
            "WORKER_FAILED",
            detail,
            "使用 --verbose 查看日志，不将额外进程留在同步 Probe 中。",
        )
    }
}

#[cfg(all(test, unix))]
mod interrupt_tests {
    use super::PythonBackend;

    #[test]
    fn a_ui_loop_can_consume_sigint_without_cancelling_a_worker() {
        let _guard = PythonBackend::prepare_interrupt_handling();
        let result = unsafe { libc::raise(libc::SIGINT) };
        assert_eq!(result, 0);
        assert!(PythonBackend::take_interrupted());
        assert!(!PythonBackend::take_interrupted());
    }
}

impl PythonRuntimeControl for PythonBackend {
    fn select_python(&mut self, executable: String) {
        self.executable = executable;
    }
}
