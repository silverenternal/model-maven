use crate::presentation::contracts::{ActionDescriptor, Availability};
use crate::presentation::widgets::{layout, text};

pub fn render_actions(actions: &[ActionDescriptor]) -> String {
    render_actions_at_width(actions, layout::content_width(layout::terminal_width()))
}

fn render_actions_at_width(actions: &[ActionDescriptor], width: usize) -> String {
    actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            let availability = match &action.availability {
                Availability::Available => String::new(),
                Availability::Conditional(reason) => format!("（需要：{reason}）"),
                Availability::Unsupported(reason) => format!("（暂不支持：{reason}）"),
                Availability::Unknown(reason) => format!("（状态未知：{reason}）"),
            };
            let value = format!("{}. {}{}", index + 1, action.label, availability);
            text::wrap(&value, width).join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::render_actions_at_width;
    use crate::presentation::{contracts::ActionDescriptor, widgets::text::display_width};

    #[test]
    fn action_lists_wrap_long_unicode_labels_and_remove_terminal_controls() {
        let actions = [ActionDescriptor::available(
            "open",
            "打开一个很长的模型项目目录 /data/models/with/a/long/path\u{1b}[31m",
        )];
        let rendered = render_actions_at_width(&actions, 20);
        assert!(rendered.lines().all(|line| display_width(line) <= 20));
        assert!(!rendered.contains('\u{1b}'));
        assert!(
            rendered
                .replace('\n', "")
                .contains("/data/models/with/a/long/path")
        );
    }
}
