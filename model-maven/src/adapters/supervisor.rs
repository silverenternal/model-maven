use crate::ports::RunSupervisor;
use serde_json::Value;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

pub struct NativeRunSupervisor;

#[cfg(target_os = "linux")]
fn process_identity(pid: u32) -> Option<(String, bool)> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let (_, fields) = stat.rsplit_once(')')?;
    let fields = fields.split_whitespace().collect::<Vec<_>>();
    let state = *fields.first()?;
    let start_ticks = *fields.get(19)?;
    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
    Some((
        format!("{}:{}", boot_id.trim(), start_ticks),
        !matches!(state, "Z" | "X"),
    ))
}

impl RunSupervisor for NativeRunSupervisor {
    fn take_user_interrupt(&self) -> bool {
        crate::adapters::python::PythonBackend::take_interrupted()
    }

    fn identify(&self, pid: u32) -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            process_identity(pid).and_then(|(token, alive)| alive.then_some(token))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = pid;
            None
        }
    }

    fn launch(
        &self,
        executable: &Path,
        project_root: &Path,
        run_id: &str,
        request: &Value,
        log_path: &Path,
        register_identity: &mut dyn FnMut(u32, &str) -> Result<(), String>,
    ) -> Result<(), String> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (
                executable,
                project_root,
                run_id,
                request,
                log_path,
                register_identity,
            );
            return Err("当前构建只在 Linux 提供可验证的脱离终端运行监督。".into());
        }

        #[cfg(target_os = "linux")]
        {
            let encoded = serde_json::to_vec(request).map_err(|error| error.to_string())?;
            if encoded.len() > 1024 * 1024 {
                return Err("后台运行请求超过 1 MiB 限制。".into());
            }
            let mut log_options = OpenOptions::new();
            log_options.create_new(true).write(true);
            #[cfg(unix)]
            log_options.mode(0o600);
            let log = log_options
                .open(log_path)
                .map_err(|error| error.to_string())?;
            let stdout = log.try_clone().map_err(|error| error.to_string())?;
            let mut command = Command::new(executable);
            command
                .args([
                    "internal-run",
                    "--project-root",
                    project_root.to_str().ok_or("项目路径不是有效 UTF-8。")?,
                    "--run-id",
                    run_id,
                ])
                .current_dir(project_root)
                .stdin(Stdio::piped())
                .stdout(Stdio::from(stdout))
                .stderr(Stdio::from(log));
            #[cfg(unix)]
            unsafe {
                command.pre_exec(|| {
                    if libc::setsid() == -1 {
                        Err(std::io::Error::last_os_error())
                    } else {
                        Ok(())
                    }
                });
            }
            let mut child = command.spawn().map_err(|error| error.to_string())?;
            let pid = child.id();
            let Some((token, alive)) = process_identity(pid) else {
                let _ = child.kill();
                let _ = child.wait();
                return Err("无法读取后台运行进程身份，未启动训练。".into());
            };
            if !alive {
                let _ = child.wait();
                return Err("后台运行进程在登记前已退出。".into());
            }
            if let Err(error) = register_identity(pid, &token) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
            let Some(mut input) = child.stdin.take() else {
                let _ = child.kill();
                let _ = child.wait();
                return Err("后台运行进程未提供请求输入管道。".into());
            };
            if let Err(error) = input.write_all(&encoded) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("无法传递后台运行请求：{error}"));
            }
            drop(input);
            let ready_path = project_root
                .join(".model-maven")
                .join("runs")
                .join(run_id)
                .join("supervisor.ready");
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                match fs::symlink_metadata(&ready_path) {
                    Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() => {
                        let ready = match fs::read_to_string(&ready_path) {
                            Ok(ready) => ready,
                            Err(error) => {
                                let _ = child.kill();
                                let _ = child.wait();
                                let _ = fs::remove_file(&ready_path);
                                return Err(error.to_string());
                            }
                        };
                        if ready.trim() != format!("{pid}:{token}") {
                            let _ = child.kill();
                            let _ = child.wait();
                            let _ = fs::remove_file(&ready_path);
                            return Err("后台监督进程就绪身份不匹配。".into());
                        }
                        let _ = fs::remove_file(&ready_path);
                        return Ok(());
                    }
                    Ok(_) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err("后台监督就绪标记不是普通文件。".into());
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(error.to_string());
                    }
                }
                match child.try_wait() {
                    Ok(Some(status)) => return Err(format!("后台监督进程在就绪前退出：{status}")),
                    Ok(None) => {}
                    Err(error) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(error.to_string());
                    }
                }
                if Instant::now() >= deadline || !self.is_alive(pid, &token) {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("后台监督进程未能在 10 秒内完成身份验证和启动握手。".into());
                }
                thread::sleep(Duration::from_millis(25));
            }
        }
    }

    fn is_alive(&self, pid: u32, token: &str) -> bool {
        #[cfg(target_os = "linux")]
        {
            process_identity(pid).is_some_and(|(actual, alive)| alive && actual == token)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (pid, token);
            false
        }
    }

    fn interrupt(&self, pid: u32, token: &str) -> Result<(), String> {
        if !self.is_alive(pid, token) {
            return Err("运行进程已退出或 PID 身份已变化；未发送信号。".into());
        }
        #[cfg(unix)]
        {
            let result = unsafe { libc::kill(pid as libc::pid_t, libc::SIGINT) };
            if result == -1 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = (pid, token);
            Err("当前平台不支持受监督运行的信号停止。".into())
        }
    }
}
