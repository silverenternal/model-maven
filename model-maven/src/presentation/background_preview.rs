use crate::{
    cli::{Cli, Command, DataCommand, PreviewDelivery},
    domain::Report,
};
use std::{
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Child, Command as ProcessCommand, ExitStatus, Stdio},
    time::{Duration, Instant},
};

/// Owns a preview CLI child while the human workbench remains available.
/// The child's report is kept in private temporary files so long previews do
/// not block on full stdout/stderr pipes and result data is not printed early.
pub struct BackgroundPreview {
    child: Child,
    command: Command,
    project_root: PathBuf,
    stdout: tempfile::NamedTempFile,
    stderr: tempfile::NamedTempFile,
    started: Instant,
    status: Option<ExitStatus>,
}

impl BackgroundPreview {
    pub fn launch(command: &Command, cli: &Cli, project_root: &Path) -> Result<Self, String> {
        let args = command_args(command, cli)?;
        let executable = std::env::current_exe()
            .map_err(|error| format!("无法定位当前 m-model 可执行文件：{error}"))?;
        let stdout = tempfile::Builder::new()
            .prefix("model-maven-preview-report-")
            .tempfile()
            .map_err(|error| format!("无法创建私有预览报告暂存：{error}"))?;
        let stderr = tempfile::Builder::new()
            .prefix("model-maven-preview-log-")
            .tempfile()
            .map_err(|error| format!("无法创建私有预览日志暂存：{error}"))?;
        let child = ProcessCommand::new(executable)
            .args(args)
            .current_dir(project_root)
            .stdin(Stdio::null())
            .stdout(Stdio::from(
                stdout
                    .reopen()
                    .map_err(|error| format!("无法打开预览报告暂存：{error}"))?,
            ))
            .stderr(Stdio::from(
                stderr
                    .reopen()
                    .map_err(|error| format!("无法打开预览日志暂存：{error}"))?,
            ))
            .spawn()
            .map_err(|error| format!("无法在后台启动预览：{error}"))?;
        Ok(Self {
            child,
            command: command.clone(),
            project_root: project_root.to_path_buf(),
            stdout,
            stderr,
            started: Instant::now(),
            status: None,
        })
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn command(&self) -> &Command {
        &self.command
    }

    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>, String> {
        if self.status.is_none() {
            self.status = self
                .child
                .try_wait()
                .map_err(|error| format!("读取预览子进程状态失败：{error}"))?;
        }
        Ok(self.status)
    }

    pub fn wait(&mut self) -> Result<ExitStatus, String> {
        if let Some(status) = self.status {
            return Ok(status);
        }
        let status = self
            .child
            .wait()
            .map_err(|error| format!("等待预览子进程失败：{error}"))?;
        self.status = Some(status);
        Ok(status)
    }

    pub fn cancel(&mut self) -> Result<ExitStatus, String> {
        if let Some(status) = self.try_wait()? {
            return Ok(status);
        }
        #[cfg(unix)]
        {
            let result = unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGINT) };
            if result == -1 {
                return Err(format!(
                    "无法请求预览协作取消：{}",
                    std::io::Error::last_os_error()
                ));
            }
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(status) = self.try_wait()? {
                    return Ok(status);
                }
                if Instant::now() >= deadline {
                    let _ = self.child.kill();
                    return self.wait();
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }
        #[cfg(not(unix))]
        {
            Err("当前平台暂不支持预览的协作取消；预览进程继续运行。".into())
        }
    }

    pub fn read_report(&mut self) -> Result<Report, String> {
        let mut output = String::new();
        self.stdout
            .as_file_mut()
            .seek(SeekFrom::Start(0))
            .and_then(|_| {
                self.stdout
                    .as_file_mut()
                    .take(8 * 1024 * 1024)
                    .read_to_string(&mut output)
            })
            .map_err(|error| format!("无法读取预览报告：{error}"))?;
        serde_json::from_str(&output).map_err(|error| {
            format!(
                "后台进程没有返回有效 JSON 报告：{error}；stderr：{}",
                self.read_stderr().unwrap_or_else(|_| "不可读取".into())
            )
        })
    }

    pub fn read_stderr(&mut self) -> Result<String, String> {
        self.stderr
            .as_file_mut()
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("无法读取预览日志：{error}"))?;
        let mut output = String::new();
        self.stderr
            .as_file_mut()
            .take(256 * 1024)
            .read_to_string(&mut output)
            .map_err(|error| format!("无法读取预览日志：{error}"))?;
        Ok(output)
    }
}

fn command_args(command: &Command, cli: &Cli) -> Result<Vec<String>, String> {
    let Command::Data {
        command:
            DataCommand::Preview {
                dataset,
                batch,
                parameters,
                split,
                samples,
                scan_limit,
                fingerprint,
                transforms,
                renderer,
                renderer_entrypoint,
                renderer_version,
                delivery,
                ssh_host,
                ssh_user,
                ssh_port,
                destination,
            },
    } = command
    else {
        return Err("后台预览只能启动 data preview 命令。".into());
    };
    let mut args = vec!["--json".into(), "--no-interactive".into()];
    if let Some(python) = &cli.python {
        args.extend(["--python".into(), python.clone()]);
    }
    if let Some(timeout) = cli.timeout {
        args.extend(["--timeout".into(), timeout.to_string()]);
    }
    if let Some(variant) = &cli.variant {
        args.extend(["--variant".into(), variant.clone()]);
    }
    if let Some(seed) = cli.seed {
        args.extend(["--seed".into(), seed.to_string()]);
    }
    if cli.verbose {
        args.push("--verbose".into());
    }
    args.extend(["data".into(), "preview".into()]);
    match (dataset, batch) {
        (Some(entrypoint), None) => args.extend(["--dataset".into(), entrypoint.clone()]),
        (None, Some(entrypoint)) => args.extend(["--batch".into(), entrypoint.clone()]),
        _ => return Err("预览必须且只能选择一个 Dataset 或批次工厂。".into()),
    }
    for parameter in parameters {
        args.extend(["--param".into(), parameter.clone()]);
    }
    args.extend([
        "--split".into(),
        split.clone(),
        "--samples".into(),
        samples.to_string(),
    ]);
    if let Some(scan_limit) = scan_limit {
        args.extend(["--scan-limit".into(), scan_limit.to_string()]);
    }
    if let Some(fingerprint) = fingerprint {
        args.extend(["--fingerprint".into(), fingerprint.clone()]);
    }
    if let Some(transforms) = transforms {
        args.extend([
            "--transforms".into(),
            transforms.to_string_lossy().into_owned(),
        ]);
    }
    args.extend(["--renderer".into(), renderer.clone()]);
    if let Some(entrypoint) = renderer_entrypoint {
        args.extend(["--renderer-entrypoint".into(), entrypoint.clone()]);
    }
    if let Some(version) = renderer_version {
        args.extend(["--renderer-version".into(), version.clone()]);
    }
    match delivery {
        PreviewDelivery::Local => args.extend(["--delivery".into(), "local".into()]),
        PreviewDelivery::Ssh => {
            args.extend(["--delivery".into(), "ssh".into()]);
            if let Some(host) = ssh_host {
                args.extend(["--ssh-host".into(), host.clone()]);
            }
            if let Some(user) = ssh_user {
                args.extend(["--ssh-user".into(), user.clone()]);
            }
            if let Some(port) = ssh_port {
                args.extend(["--ssh-port".into(), port.to_string()]);
            }
        }
    }
    args.extend([
        "--destination".into(),
        destination.to_string_lossy().into_owned(),
    ]);
    Ok(args)
}
