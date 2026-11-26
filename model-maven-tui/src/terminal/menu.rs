//! Keyboard navigation for the workbench menu on Unix terminals.
use crate::terminal::interrupt;
use std::{
    io::{self, IsTerminal, Write},
};
#[cfg(unix)]
use std::os::fd::AsRawFd;

#[cfg(unix)]
pub(crate) struct Terminal {
    fd: libc::c_int,
    original: libc::termios,
}

#[cfg(unix)]
impl Terminal {
    pub(crate) fn open() -> io::Result<Self> {
        let fd = io::stdin().as_raw_fd();
        let mut original = std::mem::MaybeUninit::uninit();
        // SAFETY: tcgetattr initializes the provided termios on success.
        if unsafe { libc::tcgetattr(fd, original.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let original = unsafe { original.assume_init() };
        let mut mode = original;
        mode.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ECHONL);
        mode.c_cc[libc::VMIN] = 1;
        mode.c_cc[libc::VTIME] = 0;
        // SAFETY: mode contains a copy of the terminal's initialized settings.
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &mode) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { fd, original })
    }

    pub(crate) fn read(&self, timeout_ms: libc::c_int) -> io::Result<Option<u8>> {
        let mut descriptor = libc::pollfd {
            fd: self.fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: descriptor points to one initialized pollfd.
        let ready = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
        if ready < 0 {
            let error = io::Error::last_os_error();
            return if error.kind() == io::ErrorKind::Interrupted {
                Ok(None)
            } else {
                Err(error)
            };
        }
        if ready == 0 {
            return Ok(None);
        }
        if descriptor.revents & libc::POLLIN == 0 {
            return Ok(Some(4));
        }
        let mut byte = 0_u8;
        // SAFETY: byte is writable and fd refers to the input terminal.
        match unsafe { libc::read(self.fd, (&mut byte as *mut u8).cast(), 1) } {
            1 => Ok(Some(byte)),
            0 => Ok(Some(4)),
            _ => Err(io::Error::last_os_error()),
        }
    }
}

#[cfg(unix)]
impl Drop for Terminal {
    fn drop(&mut self) {
        // SAFETY: original was obtained from this terminal descriptor.
        unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.original) };
    }
}

// Preserve the existing graph event loop: translate native Windows console
// events into the same VT byte stream that the Unix input reader consumes.
#[cfg(windows)]
pub(crate) struct Terminal {
    pending: std::cell::RefCell<std::collections::VecDeque<u8>>,
}

#[cfg(windows)]
impl Terminal {
    pub(crate) fn open() -> io::Result<Self> {
        use crossterm::event::{EnableFocusChange, EnableMouseCapture};
        crate::debug_trace("raw mode start");
        crossterm::terminal::enable_raw_mode()?;
        crate::debug_trace("raw mode ready");
        if let Err(error) = crossterm::execute!(io::stdout(), EnableMouseCapture, EnableFocusChange) {
            let _ = crossterm::terminal::disable_raw_mode();
            return Err(error);
        }
        crate::debug_trace("capture ready");
        Ok(Self { pending: std::cell::RefCell::new(std::collections::VecDeque::new()) })
    }

    pub(crate) fn read(&self, timeout_ms: i32) -> io::Result<Option<u8>> {
        use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind};
        if let Some(byte) = self.pending.borrow_mut().pop_front() { return Ok(Some(byte)); }
        if !event::poll(std::time::Duration::from_millis(timeout_ms.max(0) as u64))? { return Ok(None); }
        let bytes = match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => match key.code {
                KeyCode::Char('c' | 'C') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    interrupt::request(); Vec::new()
                },
                KeyCode::Char(c) if key.modifiers.contains(KeyModifiers::CONTROL) && c.is_ascii_alphabetic() =>
                    vec![(c.to_ascii_lowercase() as u8) - b'a' + 1],
                KeyCode::Char(c) => { let mut raw = [0; 4]; c.encode_utf8(&mut raw).as_bytes().to_vec() },
                KeyCode::Enter => vec![b'\r'], KeyCode::Tab => vec![b'\t'],
                KeyCode::BackTab => b"\x1b[Z".to_vec(), KeyCode::Esc => vec![27],
                KeyCode::Backspace => vec![127], KeyCode::Up => b"\x1b[A".to_vec(),
                KeyCode::Down => b"\x1b[B".to_vec(), KeyCode::Right => b"\x1b[C".to_vec(),
                KeyCode::Left => b"\x1b[D".to_vec(), KeyCode::PageUp => b"\x1b[5~".to_vec(),
                KeyCode::PageDown => b"\x1b[6~".to_vec(), _ => Vec::new(),
            },
            Event::Mouse(mouse) => {
                let (flag, end) = match mouse.kind {
                    MouseEventKind::Down(MouseButton::Left) => (0, 'M'),
                    MouseEventKind::Up(MouseButton::Left) => (0, 'm'),
                    MouseEventKind::Drag(MouseButton::Left) => (32, 'M'),
                    MouseEventKind::Moved => (35, 'M'), MouseEventKind::ScrollUp => (64, 'M'),
                    MouseEventKind::ScrollDown => (65, 'M'), MouseEventKind::ScrollLeft => (66, 'M'),
                    MouseEventKind::ScrollRight => (67, 'M'), _ => return Ok(None),
                };
                format!("\x1b[<{flag};{};{}{end}", u32::from(mouse.column) + 1, u32::from(mouse.row) + 1).into_bytes()
            }
            Event::FocusGained => b"\x1b[I".to_vec(), Event::FocusLost => b"\x1b[O".to_vec(),
            _ => Vec::new(),
        };
        self.pending.borrow_mut().extend(bytes);
        Ok(self.pending.borrow_mut().pop_front())
    }
}

#[cfg(windows)]
impl Drop for Terminal {
    fn drop(&mut self) {
        use crossterm::event::{DisableFocusChange, DisableMouseCapture};
        let _ = crossterm::execute!(io::stdout(), DisableFocusChange, DisableMouseCapture);
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

pub(crate) const RESTORE_SCREEN: &str = "\x1b[?1049l\x1b[?25h";

pub(crate) struct Screen {
    active: bool,
}

impl Screen {
    pub(crate) fn open() -> io::Result<Self> {
        let screen = Self { active: true };
        let mut output = io::stdout().lock();
        output.write_all(b"\x1b[?1049h\x1b[?25l")?;
        output.flush()?;
        Ok(screen)
    }

    pub(crate) fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        if self.active {
            let mut output = io::stdout().lock();
            let _ = write!(output, "{RESTORE_SCREEN}\x1b[?2026l");
            let _ = output.flush();
        }
    }
}

/// Return the selected index. None means that the user left the workbench.
pub(crate) fn select<F>(keys: &[&str], render: F) -> io::Result<Option<usize>>
where
    F: FnMut(usize) -> String,
{
    select_from(keys, 0, render)
}

pub(crate) fn select_from<F>(keys: &[&str], initial: usize, render: F) -> io::Result<Option<usize>>
where
    F: FnMut(usize) -> String,
{
    select_from_with_shortcut(keys, initial, None, render)
}

/// An additional shortcut can open an overlay without becoming a focusable
/// menu row; its result index is `keys.len()`.
pub(crate) fn select_from_with_shortcut<F>(
    keys: &[&str],
    initial: usize,
    shortcut: Option<u8>,
    mut render: F,
) -> io::Result<Option<usize>>
where
    F: FnMut(usize) -> String,
{
    let terminal = Terminal::open()?;
    let _interrupt = interrupt::Guard::install();
    let _screen = Screen::open()?;
    let mut output = io::stdout().lock();
    let mut selected = initial.min(keys.len().saturating_sub(1));
    let mut last_frame = String::new();
    loop {
        if interrupt::take() {
            return Ok(None);
        }
        let frame = render(selected);
        if frame != last_frame {
            write!(output, "\x1b[H\x1b[2J{frame}")?;
            output.flush()?;
            last_frame = frame;
        }
        match terminal.read(100)? {
            None => continue,
            Some(4) => return Ok(None),
            Some(b'\r' | b'\n') => return Ok(Some(selected)),
            Some(b'\t') if !keys.contains(&"\t") => {
                selected = (selected + 1) % keys.len();
            }
            Some(27) => {
                let second = terminal.read(40)?;
                let third = if second == Some(b'[') || second == Some(b'O') {
                    terminal.read(40)?
                } else {
                    None
                };
                match third {
                    Some(b'A') => selected = selected.checked_sub(1).unwrap_or(keys.len() - 1),
                    Some(b'B') => selected = (selected + 1) % keys.len(),
                    Some(b'Z') if second == Some(b'[') => {
                        if let Some(index) = keys.iter().position(|key| *key == "\t") {
                            return Ok(Some(index));
                        }
                        selected = selected.checked_sub(1).unwrap_or(keys.len() - 1);
                    }
                    _ if second.is_none() => {
                        if let Some(index) = keys.iter().position(|key| *key == "b") {
                            return Ok(Some(index));
                        }
                    }
                    _ => {}
                }
            }
            Some(byte) if Some(byte) == shortcut => return Ok(Some(keys.len())),
            Some(byte) if byte.is_ascii() => {
                let key = char::from(byte).to_ascii_lowercase().to_string();
                if let Some(index) = keys
                    .iter()
                    .position(|candidate| candidate.eq_ignore_ascii_case(&key))
                {
                    return Ok(Some(index));
                }
                if key == "q" {
                    if let Some(index) = keys.iter().position(|candidate| *candidate == "b") {
                        return Ok(Some(index));
                    }
                }
            }
            _ => {}
        }
    }
}

pub(crate) enum Input {
    Submit(String),
    Back,
    Quit,
}

pub(crate) fn input<F>(initial: &str, mut render: F) -> io::Result<Input>
where
    F: FnMut(&str, usize) -> String,
{
    let terminal = Terminal::open()?;
    let _interrupt = interrupt::Guard::install();
    let _screen = Screen::open()?;
    let mut output = io::stdout().lock();
    let mut value = initial.to_owned();
    let mut cursor = value.len();
    let mut pending = Vec::new();
    let mut last_frame = String::new();
    let mut paste = false;
    struct PasteMode;
    impl Drop for PasteMode {
        fn drop(&mut self) {
            let _ = io::stdout().write_all(b"\x1b[?2004l");
            let _ = io::stdout().flush();
        }
    }
    output.write_all(b"\x1b[?2004h")?;
    output.flush()?;
    let _paste_mode = PasteMode;
    loop {
        if interrupt::take() {
            return Ok(Input::Quit);
        }
        let frame = render(&value, cursor);
        if frame != last_frame {
            write!(output, "\x1b[H\x1b[2J{frame}")?;
            output.flush()?;
            last_frame = frame;
        }
        match terminal.read(100)? {
            None => continue,
            Some(4) if !paste => return Ok(Input::Quit),
            Some(b'\r' | b'\n') if paste => {
                value.insert(cursor, ' ');
                cursor += 1;
            }
            Some(b'\r' | b'\n') => return Ok(Input::Submit(value)),
            Some(27) => {
                let second = terminal.read(40)?;
                if second == Some(b'[') || second == Some(b'O') {
                    let mut sequence = Vec::new();
                    while sequence.len() < 16 {
                        let Some(byte) = terminal.read(40)? else {
                            break;
                        };
                        sequence.push(byte);
                        if (0x40..=0x7e).contains(&byte) {
                            break;
                        }
                    }
                    match sequence.as_slice() {
                        b"200~" => paste = true,
                        b"201~" => paste = false,
                        b"D" if !paste => {
                            if let Some((index, _)) = value[..cursor].char_indices().next_back() {
                                cursor = index;
                            }
                        }
                        b"C" if !paste => {
                            if let Some(ch) = value[cursor..].chars().next() {
                                cursor += ch.len_utf8();
                            }
                        }
                        b"H" | b"1~" | b"7~" if !paste => cursor = 0,
                        b"F" | b"4~" | b"8~" if !paste => cursor = value.len(),
                        b"3~" if !paste => {
                            if let Some(ch) = value[cursor..].chars().next() {
                                value.drain(cursor..cursor + ch.len_utf8());
                            }
                        }
                        _ => {}
                    }
                } else if second.is_none() {
                    if !paste {
                        return Ok(Input::Back);
                    }
                }
            }
            Some(1) if !paste => cursor = 0,
            Some(5) if !paste => cursor = value.len(),
            Some(21) if !paste => {
                value.clear();
                cursor = 0;
                pending.clear();
            }
            Some(8 | 127) if !paste => {
                pending.clear();
                if let Some((index, _)) = value[..cursor].char_indices().next_back() {
                    value.drain(index..cursor);
                    cursor = index;
                }
            }
            Some(byte) if byte >= 32 => {
                pending.push(byte);
                match std::str::from_utf8(&pending) {
                    Ok(text) => {
                        value.insert_str(cursor, text);
                        cursor += text.len();
                        pending.clear();
                    }
                    Err(error) if error.error_len().is_some() => pending.clear(),
                    Err(_) => {}
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn supported() -> bool {
    io::stdin().is_terminal()
        && io::stdout().is_terminal()
        && std::env::var("TERM")
            .is_ok_and(|term| !term.is_empty() && !term.eq_ignore_ascii_case("dumb"))
}

pub(crate) fn rows() -> usize {
    #[cfg(unix)]
    {
    let mut size = libc::winsize {
        ws_row: 0,
        ws_col: 0,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: size is writable and stderr refers to the terminal.
    if unsafe { libc::ioctl(io::stdout().as_raw_fd(), libc::TIOCGWINSZ, &mut size) } == 0
        && size.ws_row > 0
    {
        usize::from(size.ws_row)
    } else {
        24
    }
    }
    #[cfg(windows)]
    { crossterm::terminal::size().map(|(_, rows)| usize::from(rows)).unwrap_or(24) }
}
