use std::{
    io::{self, Read},
    os::windows::{io::AsRawHandle, process::CommandExt},
    process::{Child, Command, Output, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
use windows_sys::Win32::{
    Foundation::{CloseHandle, ERROR_BROKEN_PIPE, ERROR_NO_DATA, HANDLE, INVALID_HANDLE_VALUE},
    System::{
        Console::{
            CTRL_BREAK_EVENT, CTRL_C_EVENT, GenerateConsoleCtrlEvent, SetConsoleCtrlHandler,
        },
        Diagnostics::ToolHelp::{
            CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
        },
        JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        },
        Pipes::PeekNamedPipe,
        Threading::{
            CREATE_NEW_PROCESS_GROUP, CREATE_SUSPENDED, OpenThread, ResumeThread,
            THREAD_SUSPEND_RESUME,
        },
    },
};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
unsafe extern "system" fn interrupt(event: u32) -> i32 {
    if event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT {
        INTERRUPTED.store(true, Ordering::Relaxed);
        1
    } else {
        0
    }
}
struct InterruptGuard;
impl InterruptGuard {
    fn install() -> io::Result<Self> {
        INTERRUPTED.store(false, Ordering::Relaxed);
        // SAFETY: the handler only sets an atomic flag and remains installed for this guard.
        if unsafe { SetConsoleCtrlHandler(Some(interrupt), 1) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self)
    }
}
impl Drop for InterruptGuard {
    fn drop(&mut self) {
        // SAFETY: this removes the handler installed by this guard.
        unsafe {
            SetConsoleCtrlHandler(Some(interrupt), 0);
        }
    }
}
struct Handle(HANDLE);
impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: this object owns a valid Windows handle.
        unsafe {
            CloseHandle(self.0);
        }
    }
}
struct ChildGuard {
    child: Child,
    done: bool,
}
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

pub(super) fn protected_output(command: &mut Command) -> io::Result<Output> {
    let _interrupt = InterruptGuard::install()?;
    // Ctrl+C targets the TUI's group, while a confirmed write runs in its own group.
    command.creation_flags(CREATE_NEW_PROCESS_GROUP);
    super::collect_protected_output(command)
}

fn suspended_thread(pid: u32) -> io::Result<Handle> {
    // SAFETY: the snapshot and initialized entry are used according to the Toolhelp API.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        let snapshot = Handle(snapshot);
        let mut entry: THREADENTRY32 = std::mem::zeroed();
        entry.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut found = Thread32First(snapshot.0, &mut entry);
        while found != 0 {
            if entry.th32OwnerProcessID == pid {
                let thread = OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID);
                return if thread.is_null() {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(Handle(thread))
                };
            }
            found = Thread32Next(snapshot.0, &mut entry);
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "suspended CLI thread not found",
        ))
    }
}

pub(super) fn bounded_output(
    command: &mut Command,
    timeout: Option<Duration>,
) -> io::Result<Output> {
    let _interrupt = InterruptGuard::install()?;
    // A kill-on-close job owns the whole CLI tree. The child starts suspended so it
    // cannot create descendants before assignment to the job.
    let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if job.is_null() {
        return Err(io::Error::last_os_error());
    }
    let job = Handle(job);
    let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if unsafe {
        SetInformationJobObject(
            job.0,
            JobObjectExtendedLimitInformation,
            &limits as *const _ as _,
            std::mem::size_of_val(&limits) as u32,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    command
        .creation_flags(CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = ChildGuard {
        child: super::spawn_cli(command)?,
        done: false,
    };
    let thread = suspended_thread(child.child.id())?;
    if unsafe { AssignProcessToJobObject(job.0, child.child.as_raw_handle() as HANDLE) } == 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { ResumeThread(thread.0) } == u32::MAX {
        return Err(io::Error::last_os_error());
    }
    drop(thread);
    let mut stdout = child
        .child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("CLI stdout unavailable"))?;
    let mut bytes = Vec::new();
    let start = Instant::now();
    loop {
        if INTERRUPTED.swap(false, Ordering::Relaxed) {
            cancel(&mut child);
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "CLI operation cancelled by user",
            ));
        }
        // Peek avoids blocking when a descendant holds the inherited pipe open.
        for _ in 0..32 {
            if !read_available(&mut stdout, &mut bytes)? {
                break;
            }
        }
        if let Some(status) = child.child.try_wait()? {
            // Drain bytes already written without waiting on inherited handles.
            while read_available(&mut stdout, &mut bytes)? {}
            child.done = true;
            return Ok(Output {
                status,
                stdout: bytes,
                stderr: Vec::new(),
            });
        }
        if timeout.is_some_and(|limit| start.elapsed() >= limit) {
            cancel(&mut child);
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("CLI timed out after {} ms", timeout.unwrap().as_millis()),
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn read_available(stdout: &mut std::process::ChildStdout, bytes: &mut Vec<u8>) -> io::Result<bool> {
    let mut available = 0;
    // SAFETY: stdout owns an open pipe; available points to writable storage.
    let ok = unsafe {
        PeekNamedPipe(
            stdout.as_raw_handle() as HANDLE,
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            &mut available,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        let error = io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(code) if code == ERROR_BROKEN_PIPE as i32 || code == ERROR_NO_DATA as i32)
        {
            return Ok(false);
        }
        return Err(error);
    }
    if available == 0 {
        return Ok(false);
    }
    let mut buffer = [0u8; 8192];
    let limit = buffer.len().min(available as usize);
    let count = stdout.read(&mut buffer[..limit])?;
    super::append_read_output(bytes, &buffer[..count])?;
    Ok(count > 0)
}

fn cancel(child: &mut ChildGuard) {
    // Best-effort request for console-aware CLI cleanup, followed by job cleanup.
    if unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, child.child.id()) } == 0 {
        return;
    }
    let deadline = Instant::now() + Duration::from_secs(11);
    while Instant::now() < deadline {
        match child.child.try_wait() {
            Ok(Some(_)) | Err(_) => break,
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn helper() {
        match std::env::var("MAVEN_WINDOWS_TEST_HELPER").as_deref() {
            Ok("output") => {
                let mut out = std::io::stdout().lock();
                out.write_all(&vec![b'X'; 512 * 1024]).unwrap();
            }
            Ok("save") => {
                std::thread::sleep(Duration::from_millis(500));
                std::fs::write(
                    std::env::var_os("MAVEN_WINDOWS_TEST_MARKER").unwrap(),
                    "saved",
                )
                .unwrap();
            }
            Ok("parent") => {
                Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "cli::windows::tests::helper", "--nocapture"])
                    .env("MAVEN_WINDOWS_TEST_HELPER", "descendant")
                    .spawn()
                    .unwrap();
                std::thread::sleep(Duration::from_secs(30));
            }
            Ok("descendant") => {
                std::fs::write(
                    std::env::var_os("MAVEN_WINDOWS_TEST_READY").unwrap(),
                    "ready",
                )
                .unwrap();
                std::thread::sleep(Duration::from_secs(4));
                std::fs::write(
                    std::env::var_os("MAVEN_WINDOWS_TEST_MARKER").unwrap(),
                    "alive",
                )
                .unwrap();
            }
            _ => {}
        }
    }

    #[test]
    fn bounded_process_drains_large_output_and_cleans_descendants_on_timeout() {
        let exe = std::env::current_exe().unwrap();
        let mut output = Command::new(&exe);
        output
            .args(["--exact", "cli::windows::tests::helper", "--nocapture"])
            .env("MAVEN_WINDOWS_TEST_HELPER", "output");
        let result = bounded_output(&mut output, Some(Duration::from_secs(20))).unwrap();
        assert!(result.status.success());
        assert_eq!(
            result.stdout.iter().filter(|&&byte| byte == b'X').count(),
            512 * 1024
        );

        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("descendant-survived");
        let ready = directory.path().join("descendant-ready");
        let mut parent = Command::new(&exe);
        parent
            .args(["--exact", "cli::windows::tests::helper", "--nocapture"])
            .env("MAVEN_WINDOWS_TEST_HELPER", "parent")
            .env("MAVEN_WINDOWS_TEST_MARKER", &marker)
            .env("MAVEN_WINDOWS_TEST_READY", &ready);
        let error = bounded_output(&mut parent, Some(Duration::from_millis(1500))).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(ready.exists(), "descendant did not start before timeout");
        std::thread::sleep(Duration::from_secs(5));
        assert!(!marker.exists(), "descendant survived timeout");

        std::fs::remove_file(&ready).unwrap();
        let mut parent = Command::new(&exe);
        parent
            .args(["--exact", "cli::windows::tests::helper", "--nocapture"])
            .env("MAVEN_WINDOWS_TEST_HELPER", "parent")
            .env("MAVEN_WINDOWS_TEST_MARKER", &marker)
            .env("MAVEN_WINDOWS_TEST_READY", &ready);
        let ready_for_interrupt = ready.clone();
        let trigger = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !ready_for_interrupt.exists() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(ready_for_interrupt.exists());
            unsafe {
                interrupt(CTRL_C_EVENT);
            }
        });
        let error = bounded_output(&mut parent, Some(Duration::from_secs(15))).unwrap_err();
        trigger.join().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        std::thread::sleep(Duration::from_secs(5));
        assert!(!marker.exists(), "descendant survived cancellation");

        let mut save = Command::new(exe);
        save.args(["--exact", "cli::windows::tests::helper", "--nocapture"])
            .env("MAVEN_WINDOWS_TEST_HELPER", "save")
            .env("MAVEN_WINDOWS_TEST_MARKER", &marker);
        let trigger = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(200));
            unsafe {
                interrupt(CTRL_C_EVENT);
            }
        });
        assert!(protected_output(&mut save).unwrap().status.success());
        trigger.join().unwrap();
        assert_eq!(std::fs::read_to_string(marker).unwrap(), "saved");
    }
}
