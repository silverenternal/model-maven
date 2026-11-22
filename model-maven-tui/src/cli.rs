#[cfg(windows)]
#[path = "cli_windows.rs"]
mod windows;

use serde::Deserialize;
use serde_json::Value;
use std::{
    io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
};

const PROTOCOL_VERSION: u32 = 1;

pub(crate) struct Cli {
    executable: PathBuf,
    project: PathBuf,
    legacy_transport: Arc<Mutex<Option<(ExecutableIdentity, bool)>>>,
}

#[derive(PartialEq, Eq)]
struct ExecutableIdentity {
    path: PathBuf,
    len: u64,
    modified: std::time::SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

impl ExecutableIdentity {
    fn read(path: PathBuf) -> io::Result<Self> {
        let metadata = std::fs::metadata(&path)?;
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            path,
            len: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CapabilityCheck {
    Available,
    Missing(Vec<String>),
    Legacy,
}

impl Cli {
    pub(crate) fn new(executable: PathBuf, project: PathBuf) -> Self {
        Self {
            executable,
            project,
            legacy_transport: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn project(&self) -> &Path {
        &self.project
    }

    /// A point-in-time self-reported version, not an identity or executable pin.
    pub(crate) fn reported_version(&self) -> Option<String> {
        let mut command = Command::new(&self.executable);
        command
            .arg("--version")
            .current_dir(&self.project)
            .stdin(Stdio::null());
        #[cfg(unix)]
        let output = bounded_output(&mut command, Some(std::time::Duration::from_secs(2))).ok()?;
        #[cfg(windows)]
        let output =
            windows::bounded_output(&mut command, Some(std::time::Duration::from_secs(2))).ok()?;
        #[cfg(not(any(unix, windows)))]
        let output = command.output().ok()?;
        if !output.status.success() || output.stdout.len() > 256 {
            return None;
        }
        let version = std::str::from_utf8(&output.stdout).ok()?.trim();
        if version.is_empty() || version.chars().any(char::is_control) {
            return None;
        }
        Some(version.to_owned())
    }

    pub(crate) fn with_project(&self, project: PathBuf) -> Self {
        Self {
            executable: self.executable.clone(),
            project,
            legacy_transport: self.legacy_transport.clone(),
        }
    }

    /// Recognize only the historical Clap interface that had JSON but no
    /// non-interactive or interface-catalog flags. This is a read-only probe;
    /// other malformed or unrecognized CLI responses are not treated as legacy.
    pub(crate) fn uses_legacy_transport(&self) -> bool {
        use std::io::Read;
        let path = if self.executable.components().count() == 1 && self.executable.is_relative() {
            std::env::var_os("PATH")
                .into_iter()
                .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
                .map(|directory| {
                    if directory.is_absolute() {
                        directory
                    } else {
                        self.project.join(directory)
                    }
                })
                .map(|directory| directory.join(&self.executable))
                .find(|candidate| {
                    let Ok(metadata) = std::fs::metadata(candidate) else {
                        return false;
                    };
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
                    }
                    #[cfg(not(unix))]
                    {
                        metadata.is_file()
                    }
                })
        } else {
            Some(if self.executable.is_absolute() {
                self.executable.clone()
            } else {
                self.project.join(&self.executable)
            })
        };
        let Some(path) = path else {
            return false;
        };
        let identity = ExecutableIdentity::read(path.clone()).ok();
        if let Some(identity) = &identity {
            if let Ok(cache) = self.legacy_transport.lock() {
                if let Some((cached_identity, legacy)) = cache.as_ref() {
                    if cached_identity == identity {
                        return *legacy;
                    }
                }
            }
        }
        let mut signature = [0u8; 4];
        let native = std::fs::File::open(&path)
            .and_then(|mut file| file.read_exact(&mut signature))
            .is_ok()
            && (signature == *b"\x7fELF"
                || signature.starts_with(b"MZ")
                || matches!(
                    signature,
                    [0xcf, 0xfa, 0xed, 0xfe] | [0xfe, 0xed, 0xfa, 0xcf] | [0xca, 0xfe, 0xba, 0xbe]
                ));
        if !native {
            return false;
        }
        let mut command = Command::new(&self.executable);
        command
            .arg("--help")
            .current_dir(&self.project)
            .stdin(Stdio::null());
        #[cfg(unix)]
        let output = bounded_output(&mut command, Some(std::time::Duration::from_secs(2)));
        #[cfg(windows)]
        let output = windows::bounded_output(&mut command, Some(std::time::Duration::from_secs(2)));
        #[cfg(not(any(unix, windows)))]
        let output = command.output();
        let Ok(output) = output else {
            return false;
        };
        let help = String::from_utf8_lossy(&output.stdout);
        if !output.status.success()
            || !help.contains("Usage: ")
            || !help.contains("[OPTIONS]")
            || !help.contains("[COMMAND]")
            || !help.contains("--json")
        {
            return false;
        }
        let legacy = !help.contains("--no-interactive") && !help.contains("--describe-interface");
        // Do not retain a classification made while the executable was replaced.
        if let Some(identity) = identity {
            if ExecutableIdentity::read(path).ok().as_ref() == Some(&identity) {
                if let Ok(mut cache) = self.legacy_transport.lock() {
                    *cache = Some((identity, legacy));
                }
            }
        }
        legacy
    }

    /// Discover optional features without executing model code. Missing entries
    /// mean not advertised; malformed catalogs and unrelated errors stay errors.
    pub(crate) fn check_capabilities(&self, required: &[&str]) -> io::Result<CapabilityCheck> {
        if self.uses_legacy_transport() {
            return Ok(CapabilityCheck::Legacy);
        }
        let report = self.execute(&["--describe-interface"])?;
        if report.status != "success" {
            if report.operation == "cli"
                && report
                    .diagnostics
                    .iter()
                    .any(|item| item.code == "CLI_ARGUMENT_INVALID")
            {
                return Ok(CapabilityCheck::Legacy);
            }
            let detail = report
                .diagnostics
                .first()
                .map(|item| format!("{}: {}", item.code, item.message))
                .unwrap_or_else(|| format!("{} ({})", report.operation, report.status));
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("CLI capability discovery failed: {detail}"),
            ));
        }
        if report.operation != "interface"
            || report.data["interface_version"].as_u64() != Some(1)
            || report.data["report_protocol_version"].as_u64() != Some(PROTOCOL_VERSION.into())
            || !report.data["cli_version"]
                .as_str()
                .is_some_and(|value| !value.trim().is_empty())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid or unsupported CLI interface catalog",
            ));
        }
        let entries = report.data["capabilities"].as_object().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "CLI capability catalog is not an object",
            )
        })?;
        let mut missing = Vec::new();
        for capability in required {
            match entries.get(*capability) {
                Some(Value::Bool(true)) => (),
                Some(Value::Bool(false)) | None => missing.push((*capability).to_owned()),
                Some(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid boolean for capability {capability}"),
                    ));
                }
            }
        }
        Ok(if missing.is_empty() {
            CapabilityCheck::Available
        } else {
            CapabilityCheck::Missing(missing)
        })
    }

    pub(crate) fn supports_input_snapshot(&self) -> io::Result<bool> {
        if self.uses_legacy_transport() {
            return Ok(false);
        }
        let report = self.execute(&["--describe-interface"])?;
        if report.status == "success" {
            if report.operation != "interface"
                || report.data["interface_version"].as_u64() != Some(1)
                || report.data["report_protocol_version"].as_u64() != Some(PROTOCOL_VERSION.into())
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid or unsupported CLI interface catalog",
                ));
            }
            return report.data["capabilities"]["input_snapshot.capture_replay"]
                .as_bool()
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "CLI interface catalog lacks input snapshot capability",
                    )
                });
        }
        // Older releases return a structured argument error for the unknown flag.
        // Other failures must remain visible rather than being mistaken for age.
        if report.operation != "cli"
            || !report
                .diagnostics
                .iter()
                .any(|d| d.code == "CLI_ARGUMENT_INVALID")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "CLI interface discovery failed",
            ));
        }
        let mut command = Command::new(&self.executable);
        command
            .arg("--help")
            .current_dir(&self.project)
            .stdin(Stdio::null());
        #[cfg(unix)]
        let output = bounded_output(&mut command, read_timeout())?;
        #[cfg(windows)]
        let output = windows::bounded_output(&mut command, read_timeout())?;
        #[cfg(not(any(unix, windows)))]
        let output = command.output()?;
        let help = String::from_utf8_lossy(&output.stdout);
        Ok(output.status.success()
            && help.contains("--capture-input-to")
            && help.contains("--replay-input-from"))
    }

    pub(crate) fn execute(&self, args: &[&str]) -> io::Result<Report> {
        let legacy = self.uses_legacy_transport();
        if legacy && !legacy_inspection(args) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Legacy CLI: this action is unavailable because its safety contract cannot be verified; upgrade m-model. No command was run.",
            ));
        }
        let mut command = Command::new(&self.executable);
        command
            .arg("--json")
            .args(if legacy {
                &[][..]
            } else {
                &["--no-interactive"][..]
            })
            .args(args)
            .current_dir(&self.project)
            .stdin(Stdio::null());
        #[cfg(unix)]
        let result = if timeout_safe(args) {
            bounded_output(&mut command, read_timeout())
        } else {
            protected_output(&mut command)
        };
        #[cfg(windows)]
        let result = if timeout_safe(args) {
            windows::bounded_output(&mut command, read_timeout())
        } else {
            windows::protected_output(&mut command)
        };
        #[cfg(not(any(unix, windows)))]
        let result = command.output();
        let output = result.map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("{}: {error}", self.executable.display()),
            )
        })?;
        Self::decode_report(output.status, &output.stdout)
    }

    fn decode_report(status: std::process::ExitStatus, stdout: &[u8]) -> io::Result<Report> {
        let mut report: Report = serde_json::from_slice(stdout).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("m-model JSON ({status}, {error})"),
            )
        })?;
        if report.protocol_version != PROTOCOL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "m-model protocol {} (expected {PROTOCOL_VERSION})",
                    report.protocol_version
                ),
            ));
        }
        if report.operation.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "m-model protocol error: empty operation",
            ));
        }
        if !matches!(report.status.as_str(), "success" | "error" | "inconclusive") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("m-model protocol error: unknown status {:?}", report.status),
            ));
        }
        if report.status == "success" && report.data.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "m-model protocol error: successful report lacks data",
            ));
        }
        if status.success() != (report.status == "success") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "m-model exit status disagrees with its JSON report",
            ));
        }
        report.raw_json = String::from_utf8_lossy(stdout).trim().to_owned();
        Ok(report)
    }

    pub(crate) fn start_preview(&self, args: &[String]) -> io::Result<BackgroundTask> {
        let mut command = vec!["data".into(), "preview".into()];
        command.extend_from_slice(args);
        self.start_background(&command)
    }

    pub(crate) fn start_background(&self, args: &[String]) -> io::Result<BackgroundTask> {
        let legacy = self.uses_legacy_transport();
        if legacy && !legacy_inspection(&args.iter().map(String::as_str).collect::<Vec<_>>()) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Legacy CLI: this action is unavailable because its safety contract cannot be verified; upgrade m-model. No command was run.",
            ));
        }
        let stdout = tempfile::tempfile()?;
        let stderr = tempfile::tempfile()?;
        let mut command = Command::new(&self.executable);
        command
            .arg("--json")
            .args(if legacy {
                &[][..]
            } else {
                &["--no-interactive"][..]
            })
            .args(args)
            .current_dir(&self.project)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout.try_clone()?))
            .stderr(Stdio::from(stderr.try_clone()?));
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        Ok(BackgroundTask {
            child: spawn_cli(&mut command)?,
            stdout,
            _stderr: stderr,
            project: self.project.clone(),
            cancellation_requested: false,
            result: None,
        })
    }
}

pub(crate) struct BackgroundTask {
    child: std::process::Child,
    stdout: std::fs::File,
    _stderr: std::fs::File,
    pub(crate) project: PathBuf,
    pub(crate) cancellation_requested: bool,
    result: Option<io::Result<Report>>,
}

impl BackgroundTask {
    /// Only for read-only queries started by this TUI, never for writes or trials.
    #[cfg(unix)]
    pub(crate) fn terminate_read_only(&mut self) -> io::Result<()> {
        if self.result.is_some() || self.child.try_wait()?.is_some() {
            return Ok(());
        }
        let result = unsafe { libc::kill(-(self.child.id() as libc::pid_t), libc::SIGKILL) };
        if result == -1 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        self.child.wait()?;
        Ok(())
    }

    pub(crate) fn poll(&mut self) -> io::Result<bool> {
        use std::io::{Read, Seek, SeekFrom};
        if self.result.is_some() {
            return Ok(true);
        }
        let Some(status) = self.child.try_wait()? else {
            return Ok(false);
        };
        self.stdout.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        self.stdout
            .by_ref()
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        self.result = Some(if bytes.len() > 8 * 1024 * 1024 {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "background report exceeds 8 MiB",
            ))
        } else {
            Cli::decode_report(status, &bytes)
        });
        Ok(true)
    }

    pub(crate) fn result(&self) -> Option<&io::Result<Report>> {
        self.result.as_ref()
    }

    pub(crate) fn cancel(&mut self) -> io::Result<()> {
        if self.poll()? {
            return Ok(());
        }
        #[cfg(unix)]
        {
            let result = unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGINT) };
            if result == -1 {
                return Err(io::Error::last_os_error());
            }
            self.cancellation_requested = true;
            Ok(())
        }
        #[cfg(not(unix))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "preview cancellation is unavailable on this platform",
            ))
        }
    }
}

impl Drop for BackgroundTask {
    fn drop(&mut self) {
        if self.result.is_some() || !matches!(self.child.try_wait(), Ok(None)) {
            return;
        }
        #[cfg(unix)]
        {
            // The CLI owns a separately grouped Python worker. Let the CLI
            // interrupt and reap that worker before falling back to killing
            // the CLI group; killing the CLI first can orphan the worker.
            unsafe {
                libc::kill(self.child.id() as libc::pid_t, libc::SIGINT);
            }
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(11);
            loop {
                match self.child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                    _ => break,
                }
            }
            unsafe {
                libc::kill(-(self.child.id() as libc::pid_t), libc::SIGKILL);
            }
        }
        #[cfg(not(unix))]
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(unix)]
fn protected_output(command: &mut Command) -> io::Result<std::process::Output> {
    use std::os::unix::process::CommandExt;
    let _interrupt = crate::terminal::interrupt::Guard::install();
    command.process_group(0);
    collect_protected_output(command)
}

// Confirmed writes must finish even if their report exceeds our memory budget.
// Drain surplus output before waiting so a full pipe cannot block the writer.
#[cfg(any(unix, windows))]
fn collect_protected_output(command: &mut Command) -> io::Result<std::process::Output> {
    use std::io::Read;
    command.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = spawn_cli(command)?;
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut data = Vec::new();
    let read_result = (&mut stdout)
        .take(MAX_READ_REPORT_BYTES as u64 + 1)
        .read_to_end(&mut data)
        .and_then(|_| {
            if data.len() > MAX_READ_REPORT_BYTES {
                std::io::copy(&mut stdout, &mut std::io::sink())?;
            }
            Ok(())
        });
    drop(stdout);
    let status = child.wait()?;
    read_result?;
    if data.len() > MAX_READ_REPORT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CLI report exceeds 32 MiB; the CLI exited but the outcome is unknown; inspect project state before retrying",
        ));
    }
    Ok(std::process::Output {
        status,
        stdout: data,
        stderr: Vec::new(),
    })
}

// ETXTBSY means exec never started: a concurrent writer may briefly hold the
// executable open. Retrying this spawn cannot repeat the requested operation.
fn spawn_cli(command: &mut Command) -> io::Result<std::process::Child> {
    #[cfg(unix)]
    for attempt in 0..20 {
        match command.spawn() {
            Err(error) if error.raw_os_error() == Some(libc::ETXTBSY) && attempt < 19 => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            result => return result,
        }
    }
    command.spawn()
}

#[cfg(any(unix, windows))]
fn legacy_inspection(args: &[&str]) -> bool {
    if args.iter().any(|arg| {
        matches!(
            arg.split('=').next().unwrap_or(arg),
            "--capture-input-to"
                | "--replay-input-from"
                | "--if-input-content"
                | "--if-source-manifest"
                | "--if-source-plan"
                | "--if-variant-base"
        )
    }) {
        return false;
    }
    matches!(
        args,
        ["doctor", ..]
            | ["model", "inspect" | "probe", ..]
            | ["layer", "list", ..]
            | ["component", "list" | "show" | "check", ..]
            | ["variant", "list" | "show", ..]
            | ["runs", "list" | "show", ..]
    )
}

fn timeout_safe(args: &[&str]) -> bool {
    matches!(
        args,
        ["--describe-interface"]
            | ["assembly", "show", "--config-only"]
            | ["runs", "list"]
            | ["runs", "find-request", _]
            | ["runs", "show", _]
            | ["runs", "attach", _, "--follow=false"]
            | ["model", "inspect" | "probe", ..]
            | ["doctor", ..]
            | ["config", "history", ..]
            | ["variant", "list" | "show", ..]
            | ["init", "--dry-run", ..]
            | ["layer", "set", _, "--param", _, "--dry-run", ..]
    )
}

#[cfg(any(unix, windows))]
const MAX_READ_REPORT_BYTES: usize = 32 * 1024 * 1024;

#[cfg(any(unix, windows))]
fn append_read_output(data: &mut Vec<u8>, chunk: &[u8]) -> io::Result<()> {
    if chunk.len() > MAX_READ_REPORT_BYTES.saturating_sub(data.len()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CLI read-only output exceeds 32 MiB; narrow the query or use a compatible CLI",
        ));
    }
    data.extend_from_slice(chunk);
    Ok(())
}

#[cfg(any(unix, windows))]
fn read_timeout() -> Option<std::time::Duration> {
    const DEFAULT_MILLIS: u64 = 120_000;
    let millis = std::env::var("MODEL_MAVEN_TUI_CLI_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MILLIS);
    (millis != 0).then(|| std::time::Duration::from_millis(millis))
}

#[cfg(unix)]
fn bounded_output(
    command: &mut Command,
    timeout: Option<std::time::Duration>,
) -> io::Result<std::process::Output> {
    use std::{
        io::Read,
        os::{fd::AsRawFd, unix::process::CommandExt},
        time::Instant,
    };

    struct ChildGuard {
        child: std::process::Child,
        finished: bool,
    }
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if !self.finished {
                let pid = self.child.id() as libc::pid_t;
                // SAFETY: this child is the leader of a process group created for this invocation.
                unsafe { libc::kill(-pid, libc::SIGINT) };
                let deadline = Instant::now() + std::time::Duration::from_secs(11);
                loop {
                    // WNOWAIT keeps the child PID reserved until its process group has been cleaned up.
                    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
                    // SAFETY: info is writable and pid belongs to this guard's child.
                    let result = unsafe {
                        libc::waitid(
                            libc::P_PID,
                            pid as libc::id_t,
                            &mut info,
                            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                        )
                    };
                    if result != 0 || info.si_signo != 0 || Instant::now() >= deadline {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                // SAFETY: the unreaped child's PID cannot be reused as an unrelated process group.
                unsafe { libc::kill(-pid, libc::SIGKILL) };
                let _ = self.child.wait();
            }
        }
    }
    let _interrupt = crate::terminal::interrupt::Guard::install();
    command
        .process_group(0)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let child = spawn_cli(command)?;
    let mut guard = ChildGuard {
        child,
        finished: false,
    };
    let mut stdout = guard
        .child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("CLI stdout unavailable"))?;
    let fd = stdout.as_raw_fd();
    // SAFETY: fd is an open pipe descriptor owned by stdout.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    let start = Instant::now();
    let mut data = Vec::new();
    let mut eof = false;
    let mut buffer = [0_u8; 8192];
    loop {
        if crate::terminal::interrupt::take() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "CLI operation cancelled by user",
            ));
        }
        for _ in 0..32 {
            match stdout.read(&mut buffer) {
                Ok(0) => {
                    eof = true;
                    break;
                }
                Ok(count) => append_read_output(&mut data, &buffer[..count])?,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        if let Some(status) = guard.child.try_wait()? {
            guard.finished = true;
            loop {
                match stdout.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(count) => append_read_output(&mut data, &buffer[..count])?,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                }
            }
            return Ok(std::process::Output {
                status,
                stdout: data,
                stderr: Vec::new(),
            });
        }
        if timeout.is_some_and(|timeout| start.elapsed() >= timeout) {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("CLI timed out after {} ms", timeout.unwrap().as_millis()),
            ));
        }
        let wait = timeout.map_or(std::time::Duration::from_millis(20), |timeout| {
            timeout
                .saturating_sub(start.elapsed())
                .min(std::time::Duration::from_millis(20))
        });
        if eof {
            std::thread::sleep(wait);
        } else {
            let mut poll = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: poll points to one initialized descriptor; the wait is bounded.
            let result =
                unsafe { libc::poll(&mut poll, 1, wait.as_millis().max(1) as libc::c_int) };
            if result < 0 {
                let error = io::Error::last_os_error();
                if error.kind() != io::ErrorKind::Interrupted {
                    return Err(error);
                }
            }
        }
    }
}

/// A reused report describes a previous forward, even when the CLI invocation succeeded.
pub(crate) fn reused_evidence(data: &Value) -> bool {
    data["evidence_cache"]["status"].as_str() == Some("reused")
}

/// The probe reports aliases of an instance, not which registration was used.
pub(crate) fn ambiguous_registration(call: &Value) -> bool {
    let Some(aliases) = call["alias_paths"].as_array() else {
        return false;
    };
    let mut first = None;
    for path in aliases
        .iter()
        .filter_map(Value::as_str)
        .filter(|path| !path.trim().is_empty())
    {
        if first.is_some_and(|previous| previous != path) {
            return true;
        }
        first = Some(path);
    }
    false
}

/// Only enable an edit handoff when the observed registration is unambiguous.
pub(crate) fn unique_call_registration(call: &Value, path: &str) -> bool {
    let Some(aliases) = call["alias_paths"].as_array() else {
        return false;
    };
    !path.trim().is_empty()
        && !aliases.is_empty()
        && aliases.iter().all(|alias| alias.as_str() == Some(path))
}

#[derive(Clone, Deserialize)]
pub(crate) struct Report {
    #[serde(skip)]
    pub(crate) raw_json: String,
    protocol_version: u32,
    pub(crate) operation: String,
    pub(crate) status: String,
    #[serde(default)]
    pub(crate) data: Value,
    #[serde(default)]
    pub(crate) diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Deserialize)]
pub(crate) struct Diagnostic {
    pub(crate) code: String,
    pub(crate) message: String,
    pub(crate) hint: Option<String>,
}

#[cfg(all(test, unix))]
mod tests {
    use super::{CapabilityCheck, Cli, ExecutableIdentity, legacy_inspection, timeout_safe};

    fn write_script(path: &std::path::Path, body: &str) {
        use std::os::unix::fs::PermissionsExt;
        let next = path.with_extension("next");
        std::fs::write(&next, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&next, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::rename(next, path).unwrap();
    }

    #[test]
    fn interface_discovery_uses_catalog_and_falls_back_only_for_old_cli() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("cli with spaces");
        let write = |body: &str| write_script(&script, body);
        let cli = Cli::new(script.clone(), dir.path().to_path_buf());
        write(
            r#"printf '%s\n' '{"protocol_version":1,"operation":"interface","status":"success","data":{"interface_version":1,"report_protocol_version":1,"capabilities":{"input_snapshot.capture_replay":true}}}'"#,
        );
        assert!(cli.supports_input_snapshot().unwrap());
        write(
            r#"if [ "$1" = "--help" ]; then echo '--capture-input-to --replay-input-from'; else printf '%s\n' '{"protocol_version":1,"operation":"cli","status":"error","diagnostics":[{"code":"CLI_ARGUMENT_INVALID","message":"unknown argument"}]}'; exit 2; fi"#,
        );
        assert!(cli.supports_input_snapshot().unwrap());
        write(
            r#"printf '%s\n' '{"protocol_version":1,"operation":"interface","status":"success","data":{"interface_version":1,"report_protocol_version":1,"capabilities":{}}}'"#,
        );
        assert_eq!(
            cli.supports_input_snapshot().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn optional_capability_discovery_distinguishes_missing_legacy_and_bad_catalogs() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("cli with spaces");
        let write = |body: &str| write_script(&script, body);
        let cli = Cli::new(script.clone(), dir.path().to_path_buf());
        write(
            r#"printf '%s\n' '{"protocol_version":1,"operation":"interface","status":"success","data":{"cli_version":"1.0","interface_version":1,"report_protocol_version":1,"capabilities":{"one":true,"two":false}}}'"#,
        );
        assert_eq!(
            cli.check_capabilities(&["one"]).unwrap(),
            CapabilityCheck::Available
        );
        assert_eq!(
            cli.check_capabilities(&["one", "two", "missing"]).unwrap(),
            CapabilityCheck::Missing(vec!["two".into(), "missing".into()])
        );
        write(
            r#"printf '%s\n' '{"protocol_version":1,"operation":"interface","status":"success","data":{"cli_version":"1.0","interface_version":1,"report_protocol_version":1,"capabilities":{"one":"true"}}}'"#,
        );
        assert_eq!(
            cli.check_capabilities(&["one"]).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        write(
            r#"printf '%s\n' '{"protocol_version":1,"operation":"interface","status":"success","data":{"cli_version":"1.0","interface_version":2,"report_protocol_version":1,"capabilities":{}}}'"#,
        );
        assert_eq!(
            cli.check_capabilities(&["one"]).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
        write(
            r#"printf '%s\n' '{"protocol_version":1,"operation":"cli","status":"error","diagnostics":[{"code":"CLI_ARGUMENT_INVALID","message":"unknown flag"}]}'; exit 2"#,
        );
        assert_eq!(
            cli.check_capabilities(&["one"]).unwrap(),
            CapabilityCheck::Legacy
        );
        write(
            r#"printf '%s\n' '{"protocol_version":1,"operation":"interface","status":"error","diagnostics":[{"code":"READ_FAILED","message":"cannot read catalog"}]}'; exit 2"#,
        );
        let error = cli.check_capabilities(&["one"]).unwrap_err();
        assert!(
            error.to_string().contains("READ_FAILED")
                && error.to_string().contains("cannot read catalog")
        );
    }

    #[test]
    fn report_transport_distinguishes_protocol_errors_and_business_failures() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("cli with spaces");
        let cli = Cli::new(script.clone(), dir.path().to_path_buf());
        assert_eq!(
            cli.execute(&["doctor"]).err().unwrap().kind(),
            std::io::ErrorKind::NotFound
        );
        let write = |body: &str| write_script(&script, body);
        write(
            r#"echo 'ordinary log' >&2; printf '%s\n' '{"protocol_version":1,"operation":"doctor","status":"success","data":{},"future_field":3}'"#,
        );
        assert_eq!(cli.execute(&["doctor"]).unwrap().status, "success");
        write(
            r#"printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","diagnostics":[{"code":"FORWARD_FAILED","message":"bad input"}]}'; exit 4"#,
        );
        let report = cli.execute(&["doctor"]).unwrap();
        assert_eq!(report.diagnostics[0].code, "FORWARD_FAILED");
        for body in [
            "printf 'not json'",
            r#"printf '%s\n' '{"protocol_version":1,"operation":"doctor","status":"success"}'"#,
            r#"printf '%s\n' '{"protocol_version":1,"operation":"doctor","status":"mystery","data":{}}'; exit 1"#,
            r#"printf '%s\n' '{"protocol_version":1,"operation":"","status":"success","data":{}}'"#,
            r#"printf '%s\n' '{"protocol_version":1,"operation":"doctor","status":"success","data":{}}'; exit 1"#,
            r#"printf '%s\n' '{"protocol_version":1,"operation":"doctor","status":"success","data":{}}' 'extra stdout'"#,
        ] {
            write(body);
            assert_eq!(
                cli.execute(&["doctor"]).err().unwrap().kind(),
                std::io::ErrorKind::InvalidData,
                "{body}"
            );
        }
        write(
            r#"printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"inconclusive","data":{}}'; exit 5"#,
        );
        assert_eq!(cli.execute(&["doctor"]).unwrap().status, "inconclusive");
    }

    #[test]
    fn read_only_cli_rejects_oversized_output() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("large cli");
        std::fs::write(&script, "#!/bin/sh\nhead -c 40000000 /dev/zero\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let cli = Cli::new(script, dir.path().to_path_buf());
        let error = cli.execute(&["doctor"]).err().unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("exceeds 32 MiB"), "{error}");
    }

    #[test]
    fn oversized_write_report_is_drained_before_returning_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("cli");
        write_script(
            &script,
            "head -c 34000000 /dev/zero; head -c 34000000 /dev/zero >&2; touch finished",
        );
        let mut command = std::process::Command::new(script);
        command.current_dir(dir.path());
        let error = super::protected_output(&mut command).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("inspect project state"));
        assert!(dir.path().join("finished").exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn retries_transient_executable_busy_before_command_runs() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("cli");
        write_script(&script, "printf 'm-model 1.2.3\\n'");
        let writer = std::fs::OpenOptions::new()
            .write(true)
            .open(&script)
            .unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(30));
            drop(writer);
        });
        let cli = Cli::new(script, dir.path().to_path_buf());
        assert_eq!(cli.reported_version().as_deref(), Some("m-model 1.2.3"));
        release.join().unwrap();
    }

    #[test]
    fn version_query_accepts_only_a_short_single_line() {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("cli");
        let write = |body: &str| write_script(&script, body);
        let cli = Cli::new(script.clone(), dir.path().to_path_buf());
        write("printf 'm-model 1.2.3\\n'");
        assert_eq!(cli.reported_version().as_deref(), Some("m-model 1.2.3"));
        write("printf 'first\\nsecond\\n'");
        assert_eq!(cli.reported_version(), None);
        write("exit 1");
        assert_eq!(cli.reported_version(), None);
    }

    #[test]
    fn only_read_only_cli_operations_use_the_timeout() {
        for args in [
            &["model", "inspect"][..],
            &["model", "probe", "--model", "m:f"],
            &["doctor"],
            &["config", "history"],
            &["variant", "list"],
            &["variant", "show", "branch"],
            &["runs", "find-request", "tui-1234567890123456"],
            &["init", "--dry-run", "--model", "m:f"],
            &[
                "layer",
                "set",
                "model.head",
                "--param",
                "in_features=48",
                "--dry-run",
            ],
        ] {
            assert!(timeout_safe(args), "{args:?}");
        }
        for args in [
            &["init", "--model", "m:f"][..],
            &["layer", "set", "model.head", "--param", "in_features=48"],
            &["train"],
            &["config", "restore", "rev", "--apply"],
            &["init", "--model", "--dry-run"],
        ] {
            assert!(!timeout_safe(args), "{args:?}");
        }
    }

    #[test]
    fn legacy_mode_never_forwards_mutations_or_snapshot_options() {
        assert!(legacy_inspection(&["model", "probe", "--model", "m:f"]));
        assert!(legacy_inspection(&["doctor"]));
        for args in [
            &["train"][..],
            &["layer", "set", "head"],
            &["model", "probe", "--capture-input-to", "snapshot"],
            &["model", "probe", "--capture-input-to=snapshot"],
            &["model", "probe", "--if-input-content=abc"],
        ] {
            assert!(!legacy_inspection(args), "{args:?}");
        }
    }

    #[test]
    fn legacy_classification_is_rechecked_after_binary_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let executable = dir.path().join("m-model");
        let replacement = dir.path().join("new-m-model");
        let current = std::env::current_exe().unwrap();
        std::fs::copy(&current, &executable).unwrap();
        std::fs::copy(&current, &replacement).unwrap();
        let cli = Cli::new(executable.clone(), dir.path().to_path_buf());
        *cli.legacy_transport.lock().unwrap() =
            Some((ExecutableIdentity::read(executable.clone()).unwrap(), true));
        assert!(cli.uses_legacy_transport());
        std::fs::rename(replacement, executable).unwrap();
        // This test binary's help is not the recognized legacy CLI help.
        assert!(!cli.uses_legacy_transport());
    }
}
