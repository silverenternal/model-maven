use std::sync::atomic::{AtomicBool, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn interrupt(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

pub(crate) fn take() -> bool {
    INTERRUPTED.swap(false, Ordering::Relaxed)
}

#[cfg(windows)]
pub(crate) fn request() {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

#[cfg(unix)]
pub(crate) struct Guard(libc::sighandler_t);

#[cfg(windows)]
pub(crate) struct Guard;

impl Guard {
    pub(crate) fn install() -> Self {
        INTERRUPTED.store(false, Ordering::Relaxed);
        #[cfg(unix)]
        // SAFETY: the handler only sets an atomic flag read by the welcome loop.
        { Self(unsafe { libc::signal(libc::SIGINT, interrupt as *const () as libc::sighandler_t) }) }
        #[cfg(windows)]
        { Self }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        // SAFETY: the saved disposition comes from signal on this process.
        #[cfg(unix)]
        unsafe { libc::signal(libc::SIGINT, self.0) };
    }
}
