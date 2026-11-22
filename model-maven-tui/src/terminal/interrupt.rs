use std::sync::atomic::{AtomicBool, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn interrupt(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

pub(crate) fn take() -> bool {
    INTERRUPTED.swap(false, Ordering::Relaxed)
}

pub(crate) struct Guard(libc::sighandler_t);

impl Guard {
    pub(crate) fn install() -> Self {
        INTERRUPTED.store(false, Ordering::Relaxed);
        // SAFETY: the handler only sets an atomic flag read by the welcome loop.
        Self(unsafe { libc::signal(libc::SIGINT, interrupt as *const () as libc::sighandler_t) })
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        // SAFETY: the saved disposition comes from signal on this process.
        unsafe { libc::signal(libc::SIGINT, self.0) };
    }
}
