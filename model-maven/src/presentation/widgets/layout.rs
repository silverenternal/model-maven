//! Terminal width and page-layout policy shared by human-facing CLI views.

use std::io::{self, IsTerminal};

/// Preferred width when the terminal size is unknown or output is redirected.
pub const DEFAULT_WIDTH: usize = 80;
pub const MIN_WIDTH: usize = 20;
pub const MAX_WIDTH: usize = 500;

/// Return the current terminal width, preferring the conventional `COLUMNS`
/// override and falling back to the terminal attached to stderr.
pub fn terminal_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|width| *width > 0)
        .or_else(system_terminal_width)
        .map(normalize_width)
        .unwrap_or(DEFAULT_WIDTH)
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

/// Decide color policy using stdout, which is where normal human-readable
/// summaries are written. Prompts/progress may separately inspect stderr.
pub fn stdout_color_enabled() -> bool {
    color_enabled(io::stdout().is_terminal())
}

/// Split an already-rendered page into fixed-size chunks for simple paging.
/// Callers can display one chunk at a time or print them sequentially when
/// interaction is unavailable. An empty page yields one empty chunk.
pub fn paginate(lines: &[String], page_size: usize) -> Vec<Vec<String>> {
    let page_size = page_size.max(1);
    if lines.is_empty() {
        return vec![Vec::new()];
    }
    lines.chunks(page_size).map(<[String]>::to_vec).collect()
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

#[cfg(not(unix))]
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
    fn pagination_is_deterministic_and_handles_empty_pages() {
        let lines = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(
            paginate(&lines, 2),
            vec![vec!["a".to_owned(), "b".to_owned()], vec!["c".to_owned()]]
        );
        assert_eq!(paginate(&[], 0), vec![Vec::<String>::new()]);
    }

    #[test]
    fn color_policy_is_plain_for_redirected_no_color_and_dumb_terminal() {
        assert!(color_policy(true, false, false));
        assert!(!color_policy(false, false, false));
        assert!(!color_policy(true, true, false));
        assert!(!color_policy(true, false, true));
    }
}
