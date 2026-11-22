//! A local editor for the built-in graph. Only confirmation writes to the document.
use super::{
    Edit, EditError, Inlet, ModelView, OperatorId, Outlet, Port, Segment, Shape, Wire, push,
    shape_label, wire,
};
use crate::terminal::locale::Locale;

pub(super) const ROWS: usize = 5;
pub(super) const MIN_HEIGHT: usize = 12 + ROWS + 1;

#[derive(Clone, Copy)]
enum Producer {
    Blocks,
    Attention,
    Missing,
}

impl Producer {
    fn new_wire(self, to: Inlet) -> Wire {
        let (operator, slot) = match self {
            Self::Blocks => (2, 0),
            Self::Attention => (1, 0),
            Self::Missing => (3, 9),
        };
        Wire {
            from: Outlet {
                operator: OperatorId(operator),
                slot,
            },
            to,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Blocks => "encoder.blocks.out[0]",
            Self::Attention => "encoder.attention.out[0]",
            Self::Missing => "encoder.out[9]",
        }
    }
}

enum Draft {
    Shape { digits: String, replace: bool },
    Wire(Producer),
}

struct Change {
    revision: u64,
    draft: Draft,
}

enum Stage {
    Choose,
    Draft(Change),
    Confirm(Change),
}

enum Notice {
    InvalidDimension,
    InvalidConnection,
    Unchanged,
    Rejected(EditError),
    Committed(String),
}

pub(super) struct Editor {
    pub focused: bool,
    target: Inlet,
    stage: Stage,
    notice: Option<Notice>,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            focused: true,
            target: wire(2, 4, 1).to,
            stage: Stage::Choose,
            notice: None,
        }
    }
}

impl Editor {
    pub(super) fn confirmed_target(&self) -> Option<Inlet> {
        matches!(self.notice, Some(Notice::Committed(_))).then_some(self.target)
    }

    pub fn enter(&mut self, view: &ModelView<'_>) {
        self.focused = true;
        if let (Stage::Choose, Some(wire)) = (&self.stage, view.selected_wire) {
            self.target = wire.to;
        }
    }

    fn target_label(&self, view: &ModelView<'_>) -> String {
        let path = view
            .document
            .operator(self.target.operator)
            .map_or("?", |operator| operator.path.as_str());
        format!("{path}.in[{}]", self.target.slot)
    }

    pub fn cancel(&mut self) -> bool {
        self.notice = None;
        self.stage = match std::mem::replace(&mut self.stage, Stage::Choose) {
            Stage::Choose => return false,
            Stage::Draft(_) => Stage::Choose,
            Stage::Confirm(change) => Stage::Draft(change),
        };
        true
    }

    pub fn input(&mut self, key: u8, view: &mut ModelView<'_>) {
        let enter = matches!(key, b'\r' | b'\n');
        let confirm = matches!(key, b'y' | b'Y');
        let target = self.target;
        let revision = view.document.revision();
        match &mut self.stage {
            Stage::Choose => {
                let draft = match key {
                    b'1' => Draft::Shape {
                        digits: view
                            .document
                            .shape(Port::Input(target))
                            .and_then(Shape::last)
                            .map_or_else(String::new, |size| size.to_string()),
                        replace: true,
                    },
                    b'2' => Draft::Wire(
                        if view.document.source(target)
                            == Some(Producer::Blocks.new_wire(target).from)
                        {
                            Producer::Attention
                        } else {
                            Producer::Blocks
                        },
                    ),
                    _ => return,
                };
                self.stage = Stage::Draft(Change { revision, draft });
                self.notice = None;
            }
            Stage::Draft(change) if !enter => match (&mut change.draft, key) {
                (Draft::Shape { digits, replace }, b'0'..=b'9') => {
                    if std::mem::take(replace) {
                        digits.clear();
                    }
                    if digits.len() < 20 {
                        digits.push(char::from(key));
                    }
                    self.notice = None;
                }
                (Draft::Shape { digits, replace }, 8 | 127) => {
                    *replace = false;
                    digits.pop();
                    self.notice = None;
                }
                (Draft::Wire(producer), b'1'..=b'3') => {
                    *producer = [Producer::Blocks, Producer::Attention, Producer::Missing]
                        [usize::from(key - b'1')];
                    self.notice = None;
                }
                _ => {}
            },
            Stage::Draft(change) => {
                self.notice = match &change.draft {
                    Draft::Shape { digits, .. } => match digits.parse::<u64>() {
                        Ok(size)
                            if view
                                .document
                                .shape(Port::Input(target))
                                .and_then(Shape::last)
                                == Some(size) =>
                        {
                            Some(Notice::Unchanged)
                        }
                        Ok(1..) => None,
                        _ => Some(Notice::InvalidDimension),
                    },
                    Draft::Wire(producer)
                        if view.document.source(target) == Some(producer.new_wire(target).from) =>
                    {
                        Some(Notice::Unchanged)
                    }
                    Draft::Wire(producer)
                        if producer.new_wire(target).from.operator == target.operator =>
                    {
                        Some(Notice::InvalidConnection)
                    }
                    Draft::Wire(_) => None,
                };
                if self.notice.is_none() {
                    let Stage::Draft(change) = std::mem::replace(&mut self.stage, Stage::Choose)
                    else {
                        unreachable!()
                    };
                    self.stage = Stage::Confirm(change);
                }
            }
            Stage::Confirm(_) if !confirm => {}
            Stage::Confirm(_) => {
                let Stage::Confirm(change) = std::mem::replace(&mut self.stage, Stage::Choose)
                else {
                    unreachable!()
                };
                let target_label = self.target_label(view);
                let (edits, summary) = match &change.draft {
                    Draft::Shape { digits, .. } => {
                        let size = digits
                            .parse::<u64>()
                            .expect("review contains a validated dimension");
                        let port = Port::Input(target);
                        let shape = replace_last_axis(view.document.shape(port), size);
                        let summary = format!("{target_label} {}", shape.label());
                        (vec![Edit::SetShape(port, Some(shape))], summary)
                    }
                    Draft::Wire(producer) => {
                        let proposed = producer.new_wire(target);
                        let mut edits = Vec::new();
                        if let Some(from) = view.document.source(target) {
                            edits.push(Edit::Disconnect(Wire { from, to: target }));
                        }
                        edits.push(Edit::Connect(proposed));
                        (edits, format!("{} → {target_label}", producer.label()))
                    }
                };
                match view.apply_segment(Segment {
                    base_revision: change.revision,
                    edits,
                }) {
                    Ok(_) => {
                        view.hover_handoff = view
                            .document
                            .source(target)
                            .map(|from| Wire { from, to: target });
                        self.notice = Some(Notice::Committed(summary));
                    }
                    Err(error) => {
                        self.stage = Stage::Draft(change);
                        self.notice = Some(Notice::Rejected(error));
                    }
                }
            }
        }
    }

    pub fn render(&self, view: &ModelView<'_>, locale: Locale, width: usize) -> String {
        let mut out = String::new();
        let title = locale.text("DEMO EDITOR", "演示编辑", "展示編輯");
        let focus = if self.focused {
            locale.text("editor", "编辑焦点", "編輯焦點")
        } else {
            locale.text("graph", "图焦点", "圖焦點")
        };
        push(&mut out, &"─".repeat(width.min(100)), width);
        push(
            &mut out,
            &format!("{title} · {focus} · v{}", view.document.revision()),
            width,
        );
        let target = self.target;
        let target_label = self.target_label(view);
        let current = shape_label(view.document.shape(Port::Input(target)));
        let source = view
            .document
            .source(target)
            .and_then(|port| {
                view.document
                    .operator(port.operator)
                    .map(|operator| format!("{}.out[{}]", operator.path, port.slot))
            })
            .unwrap_or_else(|| "—".into());
        let (field, context, keys) = match &self.stage {
            Stage::Choose => (
                locale
                    .text(
                        "1 Port shape   2 Rewire",
                        "1 改端口形状   2 改接连线",
                        "1 改端口形狀   2 改接連線",
                    )
                    .into(),
                format!("{source} → {target_label} · {current}"),
                locale.text(
                    "Tab/Shift+Tab Graph/editor · Esc close",
                    "Tab/Shift+Tab 切换图与编辑 · Esc 关闭",
                    "Tab/Shift+Tab 切換圖與編輯 · Esc 關閉",
                ),
            ),
            Stage::Draft(change) | Stage::Confirm(change) => {
                let confirm = matches!(self.stage, Stage::Confirm(_));
                let field = match &change.draft {
                    Draft::Shape { digits, .. } => {
                        let mut candidate = digits.parse().ok().map_or_else(
                            || "?".into(),
                            |size| {
                                replace_last_axis(view.document.shape(Port::Input(target)), size)
                                    .label()
                            },
                        );
                        if !confirm && self.focused {
                            candidate.push('▏');
                        }
                        format!("{target_label} · {current} → {candidate}")
                    }
                    Draft::Wire(producer) => format!("{target_label} ← {}", producer.label()),
                };
                let context = if confirm {
                    format!(
                        "{} v{}",
                        locale.text("Confirm segment from", "确认本段，基于", "確認本段，基於"),
                        change.revision
                    )
                } else {
                    match change.draft {
                        Draft::Shape { .. } => locale
                            .text("Type the last dimension", "输入末维大小", "輸入末維大小")
                            .into(),
                        Draft::Wire(_) => locale
                            .text(
                                "1 Blocks · 2 Attention · 3 missing outlet",
                                "1 Blocks · 2 Attention · 3 缺失端口",
                                "1 Blocks · 2 Attention · 3 缺失端口",
                            )
                            .into(),
                    }
                };
                let keys = if confirm {
                    locale.text(
                        "Y Confirm · Esc revise · Tab switch",
                        "Y 确认 · Esc 修改 · Tab 切换",
                        "Y 確認 · Esc 修改 · Tab 切換",
                    )
                } else {
                    locale.text(
                        "Enter Review · Esc cancel · Tab switch",
                        "Enter 预览 · Esc 取消 · Tab 切换",
                        "Enter 預覽 · Esc 取消 · Tab 切換",
                    )
                };
                (field, context, keys)
            }
        };
        let feedback = match &self.notice {
            Some(Notice::InvalidDimension) => locale
                .text(
                    "Enter a positive integer",
                    "末维必须是正整数",
                    "末維必須是正整數",
                )
                .into(),
            Some(Notice::InvalidConnection) => locale
                .text(
                    "Choose a different producer",
                    "请选择其他来源算子",
                    "請選擇其他來源算子",
                )
                .into(),
            Some(Notice::Unchanged) => locale
                .text(
                    "Choose a different value",
                    "请选择不同的值",
                    "請選擇不同的值",
                )
                .into(),
            Some(Notice::Rejected(error)) => {
                let reason = match error {
                    EditError::MissingPort(_) => {
                        locale.text("missing port", "端口不存在", "端口不存在")
                    }
                    EditError::Stale { .. } => {
                        locale.text("stale draft", "草稿版本过期", "草稿版本過期")
                    }
                    _ => locale.text("invalid segment", "编辑段无效", "編輯段無效"),
                };
                format!(
                    "{} · {reason} · v{}",
                    locale.text("Rejected", "提交失败", "提交失敗"),
                    view.document.revision()
                )
            }
            Some(Notice::Committed(summary)) => format!(
                "{} v{} · {summary}",
                locale.text("Committed", "已提交", "已提交"),
                view.document.revision()
            ),
            None => context,
        };
        push(&mut out, &field, width);
        push(&mut out, &feedback, width);
        push(&mut out, keys, width);
        out
    }
}

fn replace_last_axis(shape: Option<&Shape>, size: u64) -> Shape {
    match shape {
        Some(Shape::Dimensions(axes)) if !axes.is_empty() => {
            let mut axes = axes.clone();
            *axes.last_mut().unwrap() = Some(size);
            Shape::Dimensions(axes)
        }
        _ => Shape::Features(size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_targets_the_selected_input_and_does_not_retarget_a_draft() {
        let mut view = ModelView::new(None);
        let first = wire(0, 1, 0);
        let second = wire(0, 2, 0);
        let mut editor = Editor::default();
        view.selected_wire = Some(first);
        editor.enter(&view);
        for key in b"148\n" {
            editor.input(*key, &mut view);
        }
        view.selected_wire = Some(second);
        editor.enter(&view);
        editor.input(b'y', &mut view);
        assert_eq!(
            view.document
                .shape(Port::Input(first.to))
                .and_then(Shape::last),
            Some(48)
        );
        assert_eq!(
            view.document
                .shape(Port::Input(second.to))
                .and_then(Shape::last),
            Some(256)
        );

        editor.enter(&view);
        let revision = view.document.revision();
        for key in b"22\ny" {
            editor.input(*key, &mut view);
        }
        assert_eq!(view.document.revision(), revision + 1);
        assert_eq!(view.document.source(second.to), Some(wire(1, 2, 0).from));
        assert_eq!(view.document.source(first.to), Some(first.from));
        for key in b"23\ny" {
            editor.input(*key, &mut view);
        }
        assert_eq!(view.document.revision(), revision + 1);
        assert_eq!(view.document.source(second.to), Some(wire(1, 2, 0).from));
    }
}
