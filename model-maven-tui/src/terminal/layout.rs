//! Terminal width and page-layout policy shared by human-facing CLI views.

#[cfg(unix)]
use std::io;

/// Preferred width when the terminal size is unknown or output is redirected.
pub const DEFAULT_WIDTH: usize = 80;
pub const MIN_WIDTH: usize = 20;
pub const MAX_WIDTH: usize = 500;

/// Return the current terminal width, preferring the conventional `COLUMNS`
/// override and falling back to the terminal attached to stderr.
pub fn terminal_width() -> usize {
    columns_override()
        .or_else(system_terminal_width)
        .map(normalize_width)
        .unwrap_or(DEFAULT_WIDTH)
}

/// Use the visible terminal grid for an interactive viewport, with `COLUMNS`
/// as the width hint for a redirected transcript.
pub(crate) fn viewport_width() -> usize {
    system_terminal_width()
        .or_else(columns_override)
        .map(normalize_width)
        .unwrap_or(DEFAULT_WIDTH)
}

fn columns_override() -> Option<usize> {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|width| *width > 0)
}

/// Normalize a terminal width so layout code always has a useful budget.
pub fn normalize_width(width: usize) -> usize {
    width.clamp(MIN_WIDTH, MAX_WIDTH)
}

/// Content width after reserving a small margin for terminal borders and
/// avoiding accidental wrapping at the right edge.
pub fn content_width(terminal_width: usize) -> usize {
    normalize_width(terminal_width).saturating_sub(2).max(1)
}

/// Whether ANSI color should be emitted for a particular output stream.
/// `NO_COLOR` disables color whenever it is present; `TERM=dumb` and redirected
/// output also stay plain.
pub fn color_enabled(stream_is_terminal: bool) -> bool {
    color_policy(
        stream_is_terminal,
        std::env::var_os("NO_COLOR").is_some(),
        std::env::var("TERM").is_ok_and(|term| term.eq_ignore_ascii_case("dumb")),
    )
}

/// Deterministic form of the color policy, useful to callers with their own
/// terminal abstraction and for testing redirected/`NO_COLOR` environments.
pub fn color_policy(stream_is_terminal: bool, no_color: bool, term_is_dumb: bool) -> bool {
    stream_is_terminal && !no_color && !term_is_dumb
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Background {
    #[default]
    Dark,
    Light,
}

impl Background {
    pub fn override_from_env() -> Option<Self> {
        match std::env::var("MODEL_MAVEN_BACKGROUND").as_deref() {
            Ok("dark") => Some(Self::Dark),
            Ok("light") => Some(Self::Light),
            _ => None,
        }
    }

    pub fn configured() -> Option<Self> {
        if let Some(background) = Self::override_from_env() {
            return Some(background);
        }
        let color = std::env::var("COLORFGBG").ok()?;
        let (_, bg) = color.rsplit_once(';')?;
        let index = bg.trim().parse::<u8>().ok()?;
        let light = match index {
            7 | 11 | 14 | 15 => true,
            0..=15 => false,
            16..=231 => {
                let values = [0, 95, 135, 175, 215, 255];
                let index = usize::from(index - 16);
                return Some(Self::from_rgb([
                    values[index / 36],
                    values[index / 6 % 6],
                    values[index % 6],
                ]));
            }
            _ => return Some(Self::from_rgb([8 + (index - 232) * 10; 3])),
        };
        Some(if light { Self::Light } else { Self::Dark })
    }

    fn from_rgb([r, g, b]: [u8; 3]) -> Self {
        if u32::from(r) * 2126 + u32::from(g) * 7152 + u32::from(b) * 722 >= 1_400_000 {
            Self::Light
        } else {
            Self::Dark
        }
    }

    #[cfg(unix)]
    pub fn parse_osc(bytes: &[u8]) -> Option<Self> {
        let value = std::str::from_utf8(bytes).ok()?.strip_prefix("11;rgb:")?;
        let mut rgb = [0; 3];
        let mut components = value.split('/');
        for channel in &mut rgb {
            let hex = components.next()?;
            if !(1..=4).contains(&hex.len()) || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return None;
            }
            let value = u32::from_str_radix(hex, 16).ok()?;
            let max = (1 << (hex.len() * 4)) - 1;
            *channel = ((value * 255 + max / 2) / max) as u8;
        }
        components.next().is_none().then(|| Self::from_rgb(rgb))
    }
}

#[cfg(unix)]
pub fn motion_enabled() -> bool {
    std::env::var_os("MODEL_MAVEN_REDUCE_MOTION").is_none_or(|value| value != "1")
}

#[cfg(unix)]
fn system_terminal_width() -> Option<usize> {
    use std::os::fd::AsRawFd;

    // SAFETY: `winsize` is a plain C data structure initialized here, and the
    // ioctl writes to it only when stderr is an open terminal descriptor.
    unsafe {
        let mut size: libc::winsize = std::mem::zeroed();
        if libc::ioctl(io::stderr().as_raw_fd(), libc::TIOCGWINSZ, &mut size) == 0
            && size.ws_col > 0
        {
            Some(size.ws_col as usize)
        } else {
            None
        }
    }
}

#[cfg(windows)]
fn system_terminal_width() -> Option<usize> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::Console::{
        CONSOLE_SCREEN_BUFFER_INFO, GetConsoleScreenBufferInfo,
    };

    let output = std::io::stderr();
    let mut info = std::mem::MaybeUninit::<CONSOLE_SCREEN_BUFFER_INFO>::uninit();
    // SAFETY: stderr owns the live handle and the call initializes info on success.
    if unsafe { GetConsoleScreenBufferInfo(output.as_raw_handle(), info.as_mut_ptr()) } == 0 {
        return None;
    }
    // SAFETY: GetConsoleScreenBufferInfo succeeded and initialized info.
    let window = unsafe { info.assume_init() }.srWindow;
    let width = i32::from(window.Right) - i32::from(window.Left) + 1;
    usize::try_from(width).ok().filter(|width| *width > 0)
}

#[cfg(not(any(unix, windows)))]
fn system_terminal_width() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_policy_handles_common_terminal_sizes() {
        for width in [40, 80, 120] {
            assert!(content_width(width) < width);
            assert!(content_width(width) >= MIN_WIDTH - 2);
        }
        assert_eq!(normalize_width(1), MIN_WIDTH);
        assert_eq!(normalize_width(10_000), MAX_WIDTH);
    }

    #[test]
    fn color_policy_is_plain_for_redirected_no_color_and_dumb_terminal() {
        assert!(color_policy(true, false, false));
        assert!(!color_policy(false, false, false));
        assert!(!color_policy(true, true, false));
        assert!(!color_policy(true, false, true));
    }

    #[cfg(unix)]
    #[test]
    fn parses_both_background_reply_precisions_and_rejects_incomplete_values() {
        assert_eq!(
            Background::parse_osc(b"11;rgb:ffff/fafa/e8e8"),
            Some(Background::Light)
        );
        assert_eq!(
            Background::parse_osc(b"11;rgb:FA/F9/F8"),
            Some(Background::Light)
        );
        assert_eq!(
            Background::parse_osc(b"11;rgb:23/27/31"),
            Some(Background::Dark)
        );
        assert_eq!(Background::parse_osc(b"10;rgb:ff/ff/ff"), None);
        assert_eq!(Background::parse_osc(b"11;rgb:ff/ff"), None);
        assert_eq!(Background::parse_osc(b"11;rgb:ff/ff/ff/q"), None);
        assert_eq!(Background::parse_osc(b"11;rgb:ff/ff/+f"), None);
    }
}
