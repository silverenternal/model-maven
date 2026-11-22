//! Small Linux PTY driver shared by end-to-end terminal workflow tests.
//!
//! Feed user input as a transcript, then assert on labels and outcomes rather than
//! relying on the workbench's complete menu numbering.

use std::{
    fs::File,
    io::{Read, Write},
    os::fd::{AsRawFd, FromRawFd},
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct PtyRun {
    pub transcript: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
}

#[allow(dead_code)]
pub fn run_tty(
    binary: &Path,
    working_directory: &Path,
    args: &[&str],
    input: &[u8],
    timeout: Duration,
) -> Result<PtyRun, String> {
    run_tty_with_env(binary, working_directory, args, &[], input, timeout)
}

pub fn run_tty_with_env(
    binary: &Path,
    working_directory: &Path,
    args: &[&str],
    environment: &[(&str, &str)],
    input: &[u8],
    timeout: Duration,
) -> Result<PtyRun, String> {
    let mut master_fd = -1;
    let mut slave_fd = -1;
    let size = libc::winsize {
        ws_row: 30,
        ws_col: 100,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: openpty initializes both descriptors; the winsize pointer is valid
    // for the duration of this call and no name/termios output is requested.
    let opened = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null(),
            &size,
        )
    };
    if opened != 0 {
        return Err(format!(
            "openpty failed: {}",
            std::io::Error::last_os_error()
        ));
    }

    // SAFETY: both descriptors were just returned by openpty and ownership is
    // transferred to File. The child receives clones of the slave handles.
    let mut master = unsafe { File::from_raw_fd(master_fd) };
    let slave = unsafe { File::from_raw_fd(slave_fd) };
    let stdin = slave
        .try_clone()
        .map_err(|error| format!("cannot clone PTY stdin: {error}"))?;
    let stdout = slave
        .try_clone()
        .map_err(|error| format!("cannot clone PTY stdout: {error}"))?;
    let stderr = slave
        .try_clone()
        .map_err(|error| format!("cannot clone PTY stderr: {error}"))?;
    let mut child = Command::new(binary)
        .args(args)
        .current_dir(working_directory)
        .envs(environment.iter().copied())
        .env("TERM", "xterm-256color")
        .env("COLUMNS", "100")
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", binary.display()))?;
    drop(slave);
    master
        .write_all(input)
        .map_err(|error| format!("cannot send PTY input: {error}"))?;

    let deadline = Instant::now() + timeout;
    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let now = Instant::now();
        if now >= deadline {
            let _ = child.kill();
            let status = child.wait().ok();
            return Ok(PtyRun {
                transcript: String::from_utf8_lossy(&transcript).into_owned(),
                exit_code: status.and_then(|status| status.code()),
                timed_out: true,
            });
        }
        let remaining = deadline.saturating_duration_since(now);
        let millis = remaining.as_millis().clamp(1, 1000) as i32;
        let mut descriptor = libc::pollfd {
            fd: master.as_raw_fd(),
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        // SAFETY: descriptor points to one initialized pollfd for the call.
        let ready = unsafe { libc::poll(&mut descriptor, 1, millis) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("PTY poll failed: {error}"));
        }
        if ready == 0 {
            if let Some(status) = child
                .try_wait()
                .map_err(|error| format!("cannot query child process: {error}"))?
            {
                return Ok(PtyRun {
                    transcript: String::from_utf8_lossy(&transcript).into_owned(),
                    exit_code: status.code(),
                    timed_out: false,
                });
            }
            continue;
        }
        match master.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("PTY read failed: {error}"));
            }
        }
    }
    let status = child
        .wait()
        .map_err(|error| format!("cannot wait for child process: {error}"))?;
    Ok(PtyRun {
        transcript: String::from_utf8_lossy(&transcript).into_owned(),
        exit_code: status.code(),
        timed_out: false,
    })
}
