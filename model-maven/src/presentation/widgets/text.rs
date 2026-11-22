//! Safe, width-aware text helpers for terminal pages.
//!
//! Keep terminal output useful when it is redirected, when color is disabled,
//! and when labels contain paths or text supplied by extensions.

/// Remove ANSI control sequences and replace terminal control characters with
/// visible placeholders. Newlines and tabs are retained for layout.
pub fn plain_text(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            match chars.next() {
                // CSI: ESC [ ... final byte
                Some('[') => {
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
                // OSC: ESC ] ... BEL or ST
                Some(']') => {
                    let mut previous_escape = false;
                    for next in chars.by_ref() {
                        if next == '\x07' || (previous_escape && next == '\\') {
                            break;
                        }
                        previous_escape = next == '\x1b';
                    }
                }
                // Other two-byte escape sequences are discarded.
                Some(_) => {}
                None => {}
            }
            continue;
        }
        match ch {
            '\n' => output.push(ch),
            '\t' => output.push_str("    "),
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                output.push('\n');
            }
            control if control.is_control() => output.push('�'),
            printable => output.push(printable),
        }
    }
    output
}

/// Approximate terminal columns occupied by a string, ignoring ANSI sequences.
/// CJK/full-width characters and emoji count as two columns; combining marks
/// and variation selectors count as zero.
pub fn display_width(input: &str) -> usize {
    plain_text(input).chars().map(char_width).sum()
}

/// Truncate to a maximum display width, adding an ellipsis when content is cut.
pub fn truncate(input: &str, max_width: usize) -> String {
    let plain = plain_text(input);
    if display_width(&plain) <= max_width {
        return plain;
    }
    if max_width == 0 {
        return String::new();
    }
    if max_width == 1 {
        return "…".to_owned();
    }
    let limit = max_width - 1;
    let mut result = String::new();
    let mut width = 0;
    for ch in plain.chars() {
        let next = char_width(ch);
        if width + next > limit {
            break;
        }
        result.push(ch);
        width += next;
    }
    result.push('…');
    result
}

/// Wrap text to a terminal width while preserving explicit line breaks.
/// Long tokens (for example, paths) are split at character boundaries so they
/// remain readable in narrow terminals instead of overflowing the page.
pub fn wrap(input: &str, max_width: usize) -> Vec<String> {
    let width = max_width.max(1);
    let plain = plain_text(input);
    let mut lines = Vec::new();
    for paragraph in plain.split('\n') {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut line = String::new();
        let mut line_width = 0;
        for ch in paragraph.chars() {
            let ch_width = char_width(ch);
            if line_width + ch_width > width && !line.is_empty() {
                lines.push(std::mem::take(&mut line));
                line_width = 0;
            }
            // A double-width character still needs to be emitted when width=1.
            if ch == ' ' && line.is_empty() {
                continue;
            }
            line.push(ch);
            line_width += ch_width;
        }
        if !line.is_empty() {
            lines.push(line);
        }
    }
    lines
}

fn char_width(ch: char) -> usize {
    let code = ch as u32;
    if ch.is_control() || is_combining(code) || is_variation_selector(code) || is_joiner(code) {
        0
    } else if is_wide(code) {
        2
    } else {
        1
    }
}

fn is_variation_selector(code: u32) -> bool {
    (0xfe00..=0xfe0f).contains(&code) || (0xe0100..=0xe01ef).contains(&code)
}

fn is_joiner(code: u32) -> bool {
    code == 0x200d || code == 0x200c
}

fn is_combining(code: u32) -> bool {
    matches!(
        code,
        0x0300..=0x036f
            | 0x0483..=0x0489
            | 0x0591..=0x05bd
            | 0x05bf
            | 0x05c1..=0x05c2
            | 0x0610..=0x061a
            | 0x064b..=0x065f
            | 0x0670
            | 0x06d6..=0x06ed
            | 0x1ab0..=0x1aff
            | 0x1dc0..=0x1dff
            | 0x20d0..=0x20ff
            | 0xfe20..=0xfe2f
    )
}

fn is_wide(code: u32) -> bool {
    matches!(
        code,
        0x1100..=0x115f
            | 0x231a..=0x231b
            | 0x2329..=0x232a
            | 0x23e9..=0x23ec
            | 0x23f0
            | 0x23f3
            | 0x25fd..=0x25fe
            | 0x2614..=0x2615
            | 0x2648..=0x2653
            | 0x267f
            | 0x2693
            | 0x26a1
            | 0x26aa..=0x26ab
            | 0x26bd..=0x26be
            | 0x26c4..=0x26c5
            | 0x26ce
            | 0x26d4
            | 0x26ea
            | 0x26f2..=0x26f3
            | 0x26f5
            | 0x26fa
            | 0x26fd
            | 0x2705
            | 0x270a..=0x270b
            | 0x2728
            | 0x274c
            | 0x274e
            | 0x2753..=0x2755
            | 0x2757
            | 0x2795..=0x2797
            | 0x27b0
            | 0x27bf
            | 0x2b1b..=0x2b1c
            | 0x2b50
            | 0x2b55
            | 0x2e80..=0xa4cf
            | 0xac00..=0xd7a3
            | 0xf900..=0xfaff
            | 0xfe10..=0xfe19
            | 0xfe30..=0xfe6f
            | 0xff01..=0xff60
            | 0xffe0..=0xffe6
            | 0x1f000..=0x1faff
            | 0x20000..=0x3fffd
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi_and_replaces_terminal_controls() {
        assert_eq!(plain_text("\x1b[31m错误\x1b[0m\x07"), "错误�");
        assert_eq!(plain_text("\x1b]0;title\x07ok"), "ok");
        assert_eq!(plain_text("a\tb\r\nc"), "a    b\nc");
    }

    #[test]
    fn measures_cjk_combining_marks_and_ansi() {
        assert_eq!(display_width("中文"), 4);
        assert_eq!(display_width("e\u{301}"), 1);
        assert_eq!(display_width("\x1b[31m错误\x1b[0m"), 4);
    }

    #[test]
    fn truncates_and_wraps_long_paths_without_losing_unicode() {
        assert_eq!(truncate("模型/很长的网络.py", 8), "模型/很…");
        let wrapped = wrap("模型/很长的网络.py", 6);
        assert!(wrapped.iter().all(|line| display_width(line) <= 6));
        assert_eq!(wrapped.concat(), "模型/很长的网络.py");
        assert_eq!(wrap("first\n第二行", 40), ["first", "第二行"]);
    }
}
