use super::penrose;
use crate::terminal::{layout, locale::Locale, text};
use std::{
    io::{self, Write},
    path::Path,
};

const TITLE: &str = "Model-Maven";
const ENTER: &str = "[Enter]";

#[derive(Clone, Copy)]
struct WelcomeCopy {
    tagline: &'static str,
    directory: &'static str,
    resume: &'static str,
    connect: &'static str,
    review: &'static str,
    unavailable: &'static str,
    enter: &'static str,
    failed: &'static str,
}

fn copy_for_locale(locale: Locale) -> WelcomeCopy {
    match locale {
        Locale::Traditional => WelcomeCopy {
            tagline: "PyTorch 模型除錯與實驗工作臺",
            directory: "目前目錄",
            resume: "繼續專案",
            connect: "接入模型",
            review: "專案設定待檢查",
            unavailable: "CLI 狀態未知",
            enter: "進入工作臺",
            failed: "無法讀取歡迎畫面的輸入",
        },
        Locale::Simplified => WelcomeCopy {
            tagline: "PyTorch 模型调试与实验工作台",
            directory: "当前目录",
            resume: "继续项目",
            connect: "接入模型",
            review: "项目配置待检查",
            unavailable: "CLI 状态未知",
            enter: "进入工作台",
            failed: "无法读取欢迎界面的输入",
        },
        Locale::English => WelcomeCopy {
            tagline: "Debug and experiment with PyTorch",
            directory: "Current directory",
            resume: "Continue project",
            connect: "Connect a model",
            review: "Review project configuration",
            unavailable: "CLI status unknown",
            enter: "Open workbench",
            failed: "Could not read input on the welcome screen",
        },
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ProjectState {
    Configured,
    Unconfigured,
    NeedsReview,
    Unavailable,
}

struct WelcomeContent {
    copy: WelcomeCopy,
    name: String,
    status: Option<&'static str>,
}

impl WelcomeContent {
    fn new(copy: WelcomeCopy, root: &Path, state: Option<ProjectState>) -> Self {
        let name = root.file_name().map_or_else(
            || copy.directory.to_owned(),
            |name| {
                text::plain_text(&name.to_string_lossy())
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            },
        );
        let status = state.map(|state| match state {
            ProjectState::Configured => copy.resume,
            ProjectState::Unconfigured => copy.connect,
            ProjectState::NeedsReview => copy.review,
            ProjectState::Unavailable => copy.unavailable,
        });
        Self { copy, name, status }
    }

    fn project_line(&self, width: usize) -> String {
        let Some(status) = self.status else {
            return text::truncate(&self.name, width);
        };
        let reserved = text::display_width(status) + 3;
        let name = text::truncate(&self.name, width.saturating_sub(reserved));
        if name.is_empty() {
            text::truncate(status, width)
        } else {
            format!("{name} · {status}")
        }
    }
}

#[derive(Default)]
pub(crate) struct Handoff {
    #[cfg(unix)]
    screen: Option<crate::terminal::menu::Screen>,
    #[cfg(unix)]
    motion: Option<bool>,
}

impl Handoff {
    #[cfg(unix)]
    pub(crate) fn motion_enabled(&self) -> bool {
        self.motion.unwrap_or_else(layout::motion_enabled)
    }

    #[cfg(unix)]
    pub(crate) fn toggle_motion(&mut self) -> bool {
        let enabled = !self.motion_enabled();
        self.motion = Some(enabled);
        enabled
    }

    #[cfg(unix)]
    pub(crate) fn enter_screen(&mut self) -> io::Result<()> {
        if self.screen.is_none() {
            self.screen = Some(crate::terminal::menu::Screen::open()?);
        }
        Ok(())
    }

    pub(crate) fn present(self, page: &str) -> io::Result<()> {
        #[cfg(unix)]
        if let Some(screen) = self.screen {
            return unix::present(screen, page);
        }
        let mut output = io::stdout().lock();
        output.write_all(page.as_bytes())?;
        output.flush()
    }

    pub(crate) fn welcome(
        self,
        root: &Path,
        state: ProjectState,
        locale: Locale,
    ) -> io::Result<Option<Self>> {
        let copy = copy_for_locale(locale);
        run(&WelcomeContent::new(copy, root, Some(state)), self)
            .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", copy.failed)))
    }
}

pub(crate) fn show(
    root: &Path,
    state: ProjectState,
    locale: Locale,
) -> io::Result<Option<Handoff>> {
    Handoff::default().welcome(root, state, locale)
}

pub(crate) fn enter_workspace(root: &Path, locale: Locale) -> io::Result<bool> {
    #[cfg(unix)]
    let _interrupt = crate::terminal::interrupt::Guard::install();
    let copy = copy_for_locale(locale);
    // Dropping the handoff restores the terminal before Neovim starts.
    run(&WelcomeContent::new(copy, root, None), Handoff::default())
        .map(|handoff| handoff.is_some())
        .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", copy.failed)))
}

pub(crate) fn show_line(
    root: &Path,
    state: ProjectState,
    locale: Locale,
) -> io::Result<Option<Handoff>> {
    let copy = copy_for_locale(locale);
    run_plain(&WelcomeContent::new(copy, root, Some(state)), None)
        .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", copy.failed)))
}

fn plain(
    mut output: impl Write,
    dot_aspect: Option<f64>,
    content: &WelcomeContent,
) -> io::Result<()> {
    let width = layout::content_width(layout::viewport_width());
    writeln!(output, "{TITLE}")?;
    if text::display_width(content.copy.tagline) <= width {
        writeln!(output, "{}", content.copy.tagline)?;
    }
    if let Some(aspect) = dot_aspect {
        write!(output, "{}", penrose::still(aspect))?;
    }
    writeln!(output, "{}", content.project_line(width))?;
    writeln!(
        output,
        "{ENTER}  {}",
        text::truncate(content.copy.enter, width.saturating_sub(9))
    )?;
    output.flush()
}

fn run_plain(content: &WelcomeContent, dot_aspect: Option<f64>) -> io::Result<Option<Handoff>> {
    plain(io::stderr().lock(), dot_aspect, content)?;
    let mut line = String::new();
    loop {
        line.clear();
        io::stdin().read_line(&mut line)?;
        if !line.ends_with('\n') || line.trim().eq_ignore_ascii_case("q") {
            return Ok(None);
        }
        if line.trim().is_empty() {
            return Ok(Some(Handoff::default()));
        }
    }
}

#[cfg(not(unix))]
fn run(content: &WelcomeContent, _: Handoff) -> io::Result<Option<Handoff>> {
    run_plain(content, Some(1.0))
}

#[cfg(unix)]
use unix::run;

#[cfg(unix)]
mod unix {
    use super::*;
    use crate::terminal::{interrupt, menu};
    use std::{
        io::IsTerminal,
        os::fd::{AsRawFd, RawFd},
        time::{Duration, Instant},
    };

    const FRAME_INTERVAL: Duration = Duration::from_millis(25);
    const BEGIN_UPDATE: &str = "\x1b[?2026h";
    const END_UPDATE: &str = "\x1b[?2026l";

    #[derive(Clone, Copy)]
    struct Palette {
        art: penrose::ArtPalette,
        key: &'static str,
    }

    fn palette_for(extended: bool, background: Option<layout::Background>) -> Palette {
        use layout::Background;
        match (extended, background) {
            (true, Some(Background::Dark)) => Palette {
                art: penrose::ArtPalette {
                    faces: ["\x1b[38;5;79m", "\x1b[38;5;36m", "\x1b[38;5;29m"],
                    rim: ["\x1b[38;5;121m", "\x1b[38;5;79m", "\x1b[38;5;36m"],
                    shine: ["\x1b[38;5;158m", "\x1b[38;5;122m", "\x1b[38;5;115m"],
                },
                key: "\x1b[1;38;5;121m",
            },
            (true, Some(Background::Light)) => Palette {
                art: penrose::ArtPalette {
                    faces: ["\x1b[38;5;23m", "\x1b[38;5;29m", "\x1b[38;5;30m"],
                    rim: ["\x1b[38;5;22m", "\x1b[38;5;23m", "\x1b[38;5;29m"],
                    shine: ["\x1b[38;5;22m", "\x1b[38;5;22m", "\x1b[38;5;23m"],
                },
                key: "\x1b[1;38;5;23m",
            },
            _ => Palette {
                art: penrose::ArtPalette {
                    faces: ["\x1b[0m", "\x1b[0;2m", "\x1b[0;2m"],
                    rim: ["\x1b[0;1m", "\x1b[0m", "\x1b[0m"],
                    shine: ["\x1b[0;1m", "\x1b[0;1m", "\x1b[0m"],
                },
                key: "\x1b[1m",
            },
        }
    }

    pub(super) fn run(
        content: &WelcomeContent,
        mut handoff: Handoff,
    ) -> io::Result<Option<Handoff>> {
        let input = io::stdin();
        let output = io::stderr();
        let term = std::env::var("TERM").unwrap_or_default();
        if !output.is_terminal() || term.is_empty() || term.eq_ignore_ascii_case("dumb") {
            drop(handoff);
            return run_plain(content, None);
        }
        let Ok(keys) = WelcomeKeys::open(input.as_raw_fd()) else {
            drop(handoff);
            return run_plain(content, None);
        };
        let fd = output.as_raw_fd();

        let palette = layout::color_enabled(true).then(|| {
            let colorterm = std::env::var("COLORTERM").unwrap_or_default();
            let extended = term.contains("256color")
                || term.ends_with("-direct")
                || matches!(term.as_str(), "xterm-ghostty" | "xterm-kitty")
                || matches!(
                    colorterm.to_ascii_lowercase().as_str(),
                    "24bit" | "truecolor"
                );
            palette_for(extended, layout::Background::configured())
        });
        handoff.enter_screen()?;
        let started = Instant::now();
        let motion = handoff.motion_enabled();
        let mut next_frame = started;
        let mut last_size = None;
        let mut art = None;
        loop {
            let now = Instant::now();
            if now >= next_frame {
                let mut output = output.lock();
                let size = window_size(fd).unwrap_or(TerminalSize::FALLBACK);
                let resized = last_size != Some(size);
                if resized {
                    output.write_all(BEGIN_UPDATE.as_bytes())?;
                    art = lay_out(&mut output, size, palette, content)?;
                    last_size = Some(size);
                }
                if let Some((animation, top, left)) = art.as_mut().filter(|_| motion || resized) {
                    let seconds = if motion {
                        now.duration_since(started).as_secs_f64()
                    } else {
                        0.0
                    };
                    let patch = animation.frame(seconds, *top, *left);
                    output.write_all(patch.as_bytes())?;
                }
                if resized {
                    output.write_all(END_UPDATE.as_bytes())?;
                }
                output.flush()?;
                next_frame = now
                    + if motion {
                        FRAME_INTERVAL
                    } else {
                        Duration::from_millis(100)
                    };
            }
            if let Some(enter) = keys.wait(next_frame.saturating_duration_since(Instant::now()))? {
                return Ok(enter.then_some(handoff));
            }
        }
    }

    fn lay_out(
        output: &mut impl Write,
        TerminalSize {
            cols,
            rows,
            dot_aspect,
        }: TerminalSize,
        palette: Option<Palette>,
        content: &WelcomeContent,
    ) -> io::Result<Option<(penrose::Animation, usize, usize)>> {
        output.write_all(b"\x1b[2J")?;
        let gap = usize::from(cols >= 70 && rows >= 27);
        let previous_height_limit = if cols < 55 { 10 } else { 15 };
        let show_tagline = rows >= 15 && text::display_width(content.copy.tagline) < cols;
        let labels = 3 + usize::from(show_tagline);
        let height_by_width = (cols.saturating_sub(4) as f64 * dot_aspect / 2.0) as usize;
        let available_rows = rows.saturating_sub(labels + 2 * gap);
        let previous_height = available_rows
            .saturating_sub(2)
            .min(height_by_width)
            .min(previous_height_limit);
        let (wanted_rows, baseline_radius) = penrose::enlarged_size(previous_height, dot_aspect);
        let height = available_rows.min(height_by_width).min(wanted_rows);
        if height < 5 {
            draw_hint(output, cols, rows / 2, palette, content.copy)?;
            return Ok(None);
        }
        let width = ((2.0 * height as f64 / dot_aspect).ceil() as usize + 2).min(cols - 2);
        let top = (rows - height - labels - 2 * gap) / 2;
        let art_top = top + 1 + usize::from(show_tagline) + gap;
        draw_label(output, cols, top, TITLE)?;
        if show_tagline {
            draw_label(output, cols, top + 1, content.copy.tagline)?;
        }
        let status_row = art_top + height + gap;
        draw_label(
            output,
            cols,
            status_row,
            &content.project_line(cols.saturating_sub(1)),
        )?;
        draw_hint(output, cols, status_row + 1, palette, content.copy)?;
        Ok(Some((
            penrose::Animation::new(
                width,
                height,
                dot_aspect,
                baseline_radius,
                palette.map(|theme| theme.art),
            ),
            art_top,
            (cols - width) / 2,
        )))
    }

    fn center_at(output: &mut impl Write, cols: usize, row: usize, width: usize) -> io::Result<()> {
        write!(output, "\x1b[{};{}H", row + 1, (cols - width) / 2 + 1)
    }

    fn draw_label(output: &mut impl Write, cols: usize, row: usize, label: &str) -> io::Result<()> {
        let label = text::truncate(label, cols.saturating_sub(1));
        center_at(output, cols, row, text::display_width(&label))?;
        write!(output, "{label}")
    }

    fn draw_hint(
        output: &mut impl Write,
        cols: usize,
        row: usize,
        palette: Option<Palette>,
        copy: WelcomeCopy,
    ) -> io::Result<()> {
        let available = cols.saturating_sub(1);
        let key = text::truncate(ENTER, available);
        let key_width = text::display_width(&key);
        let action = text::truncate(copy.enter, available.saturating_sub(key_width + 2));
        let width = key_width
            + if action.is_empty() {
                0
            } else {
                2 + text::display_width(&action)
            };
        center_at(output, cols, row, width)?;
        if let Some(theme) = palette {
            write!(output, "{}{key}\x1b[0m", theme.key)?;
        } else {
            write!(output, "{key}")?;
        }
        if !action.is_empty() {
            write!(output, "  {action}")?;
        }
        Ok(())
    }

    pub(super) fn present(mut screen: menu::Screen, page: &str) -> io::Result<()> {
        let stdout = io::stdout();
        let stderr = io::stderr();
        let mut output = stdout.lock();
        if stdout.is_terminal() && same_device(stdout.as_raw_fd(), stderr.as_raw_fd()) {
            // Keep the last welcome frame visible until the menu is complete.
            let restore = menu::RESTORE_SCREEN;
            let batch = format!("{BEGIN_UPDATE}{restore}{page}{END_UPDATE}");
            output.write_all(batch.as_bytes())?;
            output.flush()?;
            screen.disarm();
        } else {
            drop(screen);
            output.write_all(page.as_bytes())?;
            output.flush()?;
        }
        Ok(())
    }

    fn same_device(left: RawFd, right: RawFd) -> bool {
        let device = |fd| {
            let mut info = std::mem::MaybeUninit::<libc::stat>::uninit();
            // SAFETY: fstat initializes the stat struct on success.
            (unsafe { libc::fstat(fd, info.as_mut_ptr()) } == 0)
                .then(|| unsafe { info.assume_init() }.st_rdev)
        };
        matches!((device(left), device(right)), (Some(left), Some(right)) if left == right)
    }

    struct WelcomeKeys {
        fd: RawFd,
        original: libc::termios,
    }

    impl WelcomeKeys {
        fn open(fd: RawFd) -> io::Result<Self> {
            let mut original = std::mem::MaybeUninit::uninit();
            // SAFETY: tcgetattr initializes the supplied termios on success.
            if unsafe { libc::tcgetattr(fd, original.as_mut_ptr()) } != 0 {
                return Err(io::Error::last_os_error());
            }
            let original = unsafe { original.assume_init() };
            let mut mode = original;
            // Preserve signal keys and input already queued for the next page.
            mode.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ECHONL);
            mode.c_cc[libc::VMIN] = 1;
            mode.c_cc[libc::VTIME] = 0;
            // SAFETY: mode is an initialized termios and fd is the input terminal.
            if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &mode) } != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self { fd, original })
        }

        fn wait(&self, timeout: Duration) -> io::Result<Option<bool>> {
            if interrupt::take() {
                return Ok(Some(false));
            }
            let mut descriptor = libc::pollfd {
                fd: self.fd,
                events: libc::POLLIN,
                revents: 0,
            };
            let millis = timeout.as_micros().div_ceil(1000) as libc::c_int;
            // SAFETY: descriptor points to one initialized pollfd.
            let ready = unsafe { libc::poll(&mut descriptor, 1, millis) };
            if ready < 0 {
                return retry_on_interrupt(io::Error::last_os_error());
            }
            if ready == 0 {
                return Ok(None);
            }
            if descriptor.revents & libc::POLLIN == 0 {
                return Ok(Some(false));
            }
            let mut key = 0_u8;
            // SAFETY: key provides one writable byte. Reading one byte leaves
            // commands queued after Enter for the normal workbench input.
            let read = unsafe { libc::read(self.fd, (&mut key as *mut u8).cast(), 1) };
            match (read, key) {
                (-1, _) => retry_on_interrupt(io::Error::last_os_error()),
                (0, _) | (_, 0x04 | b'q' | b'Q') => Ok(Some(false)),
                (_, b'\r' | b'\n') => Ok(Some(true)),
                _ => Ok(None),
            }
        }
    }

    impl Drop for WelcomeKeys {
        fn drop(&mut self) {
            // SAFETY: restore the saved attributes on the same input terminal.
            while unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.original) } != 0 {
                if io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                    break;
                }
            }
        }
    }

    fn retry_on_interrupt(error: io::Error) -> io::Result<Option<bool>> {
        if error.kind() == io::ErrorKind::Interrupted {
            Ok(interrupt::take().then_some(false))
        } else {
            Err(error)
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    struct TerminalSize {
        cols: usize,
        rows: usize,
        dot_aspect: f64,
    }

    impl TerminalSize {
        const FALLBACK: Self = Self {
            cols: 80,
            rows: 24,
            dot_aspect: 1.0,
        };
    }

    fn window_size(fd: RawFd) -> Option<TerminalSize> {
        let mut size = libc::winsize {
            ws_row: 0,
            ws_col: 0,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // SAFETY: the kernel fills the initialized winsize through the pointer.
        let result = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut size) };
        if result != 0 || size.ws_col == 0 || size.ws_row == 0 {
            return None;
        }
        // Some PTYs leave the pixel fields at zero. When present, divide
        // them by grid dimensions to estimate the pitch of the 2×4 dots.
        let dot_aspect = 2.0 * (f64::from(size.ws_xpixel) / f64::from(size.ws_col))
            / (f64::from(size.ws_ypixel) / f64::from(size.ws_row));
        Some(TerminalSize {
            cols: usize::from(size.ws_col),
            rows: usize::from(size.ws_row),
            dot_aspect: if (0.25..=4.0).contains(&dot_aspect) {
                dot_aspect
            } else {
                1.0
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Locale, ProjectState, WelcomeContent, copy_for_locale};
    use std::path::Path;

    #[test]
    fn resolves_welcome_copy_from_posix_and_language_tags() {
        for locale in ["zh", "zh_CN.UTF-8", "zh-Hans-TW"] {
            assert_eq!(
                copy_for_locale(Locale::from_tag(locale)).enter,
                "进入工作台"
            );
        }
        for locale in ["zh_TW.UTF-8", "zh-Hant-CN", "zh_HK"] {
            assert_eq!(
                copy_for_locale(Locale::from_tag(locale)).enter,
                "進入工作臺"
            );
        }
        for locale in ["", "en_US.UTF-8", "C.UTF-8", "fr_FR"] {
            assert_eq!(
                copy_for_locale(Locale::from_tag(locale)).enter,
                "Open workbench"
            );
        }
    }

    #[test]
    fn keeps_project_status_visible_and_sanitizes_the_directory_label() {
        let copy = copy_for_locale(Locale::Simplified);
        let root = Path::new("/tmp/demo-model-with-a-long-name\n\x1b[31m");
        let welcome = WelcomeContent::new(copy, root, Some(ProjectState::Configured));
        let label = welcome.project_line(24);
        assert!(label.ends_with(" · 继续项目"));
        assert!(!label.contains('\x1b') && !label.contains('\n'));
        assert_eq!(welcome.project_line(8), "继续项目");
    }
}
