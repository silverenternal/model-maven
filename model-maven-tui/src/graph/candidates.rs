//! Stable, explicit selection of overlapping displayed tensor routes.
use super::{ModelView, Scene, Wire, push};
use crate::terminal::locale::Locale;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Choice {
    wire: Wire,
    displayed: Wire,
}

pub(super) struct Candidates {
    choices: Vec<Choice>,
    selected: usize,
    pointed: bool,
    scope_active: bool,
    local_empty: bool,
    scope_unavailable: bool,
    saved_local: Option<(Vec<Choice>, usize)>,
    saved_scope: Option<(Vec<Choice>, usize)>,
}

impl Candidates {
    pub fn new(view: &ModelView<'_>, scene: &Scene, cell: Option<(usize, usize)>) -> Option<Self> {
        let under_pointer = cell
            .map(|(x, y)| scene.handoffs_at(x, y))
            .unwrap_or_default();
        let pointed = !under_pointer.is_empty();
        let wires: Vec<_> = if !pointed {
            scene.handoffs.iter().map(|route| route.wire).collect()
        } else {
            under_pointer
        };
        let mut unique = BTreeMap::new();
        for wire in wires {
            let displayed = scene.displayed_wire(wire, view.selected, cell);
            unique.entry(wire).or_insert(Choice { wire, displayed });
        }
        let choices: Vec<_> = unique.into_values().collect();
        if choices.is_empty() {
            let choices = Self::loaded_scope(view, scene)?;
            let selected = view
                .selected_wire
                .and_then(|wire| choices.iter().position(|choice| choice.wire == wire))
                .unwrap_or(0);
            return Some(Self {
                choices,
                selected,
                pointed: false,
                scope_active: true,
                local_empty: true,
                scope_unavailable: false,
                saved_local: None,
                saved_scope: None,
            });
        }
        let selected = view
            .selected_wire
            .and_then(|wire| choices.iter().position(|choice| choice.wire == wire))
            .unwrap_or(0);
        Some(Self {
            choices,
            selected,
            pointed,
            scope_active: false,
            local_empty: false,
            scope_unavailable: false,
            saved_local: None,
            saved_scope: None,
        })
    }

    pub fn toggle_scope(&mut self, view: &ModelView<'_>, scene: &Scene) {
        self.scope_unavailable = false;
        if self.scope_active {
            let Some((local, selected)) = self.saved_local.take() else {
                self.local_empty = true;
                return;
            };
            let previous = std::mem::replace(&mut self.choices, local);
            let previous_selected = std::mem::replace(&mut self.selected, selected);
            self.saved_scope = Some((previous, previous_selected));
            self.scope_active = false;
            self.local_empty = false;
            return;
        }
        let current = self.choices[self.selected];
        let scope = self.saved_scope.take().or_else(|| {
            let choices = Self::loaded_scope(view, scene)?;
            let selected = choices
                .iter()
                .position(|&choice| choice == current)
                .or_else(|| {
                    choices
                        .iter()
                        .position(|choice| choice.wire == current.wire)
                })
                .unwrap_or(0);
            Some((choices, selected))
        });
        let Some((scope, selected)) = scope else {
            self.scope_unavailable = true;
            return;
        };
        let local = std::mem::replace(&mut self.choices, scope);
        let local_selected = std::mem::replace(&mut self.selected, selected);
        self.saved_local = Some((local, local_selected));
        self.scope_active = true;
        self.local_empty = false;
    }

    fn loaded_scope(view: &ModelView<'_>, scene: &Scene) -> Option<Vec<Choice>> {
        let choices: Vec<_> = super::matrix::Matrix::new(view, scene, view.selected_wire)?
            .scope_wires()
            .into_iter()
            .map(|(wire, displayed)| Choice { wire, displayed })
            .collect();
        (!choices.is_empty()).then_some(choices)
    }

    pub fn movement(&mut self, forward: bool) {
        self.selected = if forward {
            (self.selected + 1).min(self.choices.len() - 1)
        } else {
            self.selected.saturating_sub(1)
        };
    }

    pub fn page(&mut self, forward: bool, height: usize) {
        let (_, count) = self.window(height);
        self.selected = if forward {
            self.selected
                .saturating_add(count)
                .min(self.choices.len() - 1)
        } else {
            self.selected.saturating_sub(count)
        };
    }

    pub fn selected(&self) -> (Wire, Wire) {
        let choice = self.choices[self.selected];
        (choice.wire, choice.displayed)
    }

    fn window(&self, height: usize) -> (usize, usize) {
        let count = height.saturating_sub(7).checked_div(2).unwrap_or(0).max(1);
        (self.selected.saturating_sub(count - 1), count)
    }

    pub fn index_at(&self, row: usize, height: usize) -> Option<usize> {
        let row = row.checked_sub(2)?;
        let (start, count) = self.window(height);
        let index = start + row / 2;
        if row / 2 >= count || index >= self.choices.len() {
            return None;
        }
        Some(index)
    }

    pub fn select(&mut self, index: usize) {
        if index < self.choices.len() {
            self.selected = index;
        }
    }

    pub fn render(
        &self,
        view: &ModelView<'_>,
        width: usize,
        height: usize,
        locale: Locale,
    ) -> String {
        let mut out = String::new();
        push(
            &mut out,
            &format!(
                "{} · {}/{}",
                locale.text("Wire candidates", "连线候选", "連線候選"),
                self.selected + 1,
                self.choices.len()
            ),
            width,
        );
        push(
            &mut out,
            if self.scope_unavailable {
                locale.text(
                    "Scope unavailable · s retry",
                    "范围不可用 · s 重试",
                    "範圍無法取得 · s 重試",
                )
            } else if self.local_empty {
                locale.text(
                    "No local routes · showing scope",
                    "当前页无连线 · 显示范围",
                    "目前頁無連線 · 顯示範圍",
                )
            } else if self.scope_active {
                locale.text(
                    "Loaded scope · s local",
                    "已加载范围 · s 局部",
                    "已載入範圍 · s 局部",
                )
            } else if self.pointed {
                locale.text(
                    "At pointer · s scope",
                    "指针附近 · s 范围",
                    "指標附近 · s 範圍",
                )
            } else {
                locale.text(
                    "Visible page · s scope",
                    "当前页 · s 范围",
                    "目前頁 · s 範圍",
                )
            },
            width,
        );
        let (start, count) = self.window(height);
        let name = |id| {
            view.document
                .operator(id)
                .map_or("?", |operator| operator.path.as_str())
        };
        for (index, choice) in self.choices.iter().enumerate().skip(start).take(count) {
            let port = choice.displayed;
            let mark = if index == self.selected { ">" } else { " " };
            let from = super::failure::port_label(
                name(port.from.operator),
                "out",
                port.from.slot,
                width.saturating_sub(3),
            );
            let to = super::failure::port_label(
                name(port.to.operator),
                "in",
                port.to.slot,
                width.saturating_sub(5),
            );
            push(&mut out, &format!("{mark} {from}"), width);
            push(&mut out, &format!("  → {to}"), width);
        }
        let selected = self.choices[self.selected];
        if selected.displayed != selected.wire {
            let wire = selected.wire;
            push(
                &mut out,
                locale.text("Underlying wire:", "底层连线：", "底層連線："),
                width,
            );
            push(
                &mut out,
                &super::failure::port_label(name(wire.from.operator), "out", wire.from.slot, width),
                width,
            );
            push(
                &mut out,
                &format!(
                    "→ {}",
                    super::failure::port_label(
                        name(wire.to.operator),
                        "in",
                        wire.to.slot,
                        width.saturating_sub(2)
                    )
                ),
                width,
            );
        }
        if width < 58 {
            push(
                &mut out,
                locale.text(
                    "↑↓/wheel · PgUp/PgDn",
                    "↑↓/滚轮 · PgUp/PgDn",
                    "↑↓/滾輪 · PgUp/PgDn",
                ),
                width,
            );
            push(
                &mut out,
                locale.text(
                    "Enter/click confirm · Esc cancel",
                    "Enter/单击确认 · Esc 取消",
                    "Enter/單擊確認 · Esc 取消",
                ),
                width,
            );
        } else {
            push(
                &mut out,
                locale.text(
                    "↑↓/wheel · PgUp/PgDn · Enter/click confirm · Esc cancel",
                    "↑↓/滚轮 · PgUp/PgDn · Enter/单击确认 · Esc 取消",
                    "↑↓/滾輪 · PgUp/PgDn · Enter/單擊確認 · Esc 取消",
                ),
                width,
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_maven_tui::diagnosis::Case;

    #[test]
    fn empty_local_page_opens_loaded_scope_without_an_empty_choice_list() {
        let offline = super::super::offline::Offline::new(Case::TwinMerge);
        let mut view = offline.view();
        let mut scene = view.scene(80, 30, false, None);
        assert!(
            !super::super::matrix::Matrix::new(&view, &scene, None)
                .unwrap()
                .scope_wires()
                .is_empty()
        );
        scene.handoffs.clear();

        let mut candidates = Candidates::new(&view, &scene, None).unwrap();
        assert!(candidates.scope_active);
        assert!(candidates.local_empty);
        assert!(!candidates.choices.is_empty());
        assert!(
            candidates
                .render(&view, 40, 24, Locale::English)
                .contains("No local routes")
        );

        candidates.toggle_scope(&view, &scene);
        assert!(candidates.scope_active);
        let selected = candidates.selected();
        let revision = view.document.revision();
        super::super::interaction::select_matrix_wire(&mut view, selected.0, selected.1);
        assert_eq!(view.selected_wire, Some(selected.0));
        assert_eq!(view.document.revision(), revision);
    }

    #[test]
    fn loaded_scope_includes_off_page_fanout_and_preserves_both_list_positions() {
        use super::super::{
            Edit, GraphDocument, Inlet, LayoutCache, Operator, OperatorId, Outlet, Port, Segment,
        };
        use std::borrow::Cow;
        let root = OperatorId(1);
        let source = OperatorId(2);
        let sink = OperatorId(50);
        let branches: Vec<_> = (3..35).map(OperatorId).collect();
        let mut edits = Vec::new();
        for id in std::iter::once(root)
            .chain(std::iter::once(source))
            .chain(branches.iter().copied())
            .chain(std::iter::once(sink))
        {
            edits.push(Edit::AddOperator(
                id,
                Operator {
                    path: format!("model.branch_{}", id.0),
                    kind: "Linear".into(),
                    parent: (id != root).then_some(root),
                    attributes: Cow::Owned(serde_json::json!({"stack": 1})),
                },
            ));
            if id != root {
                edits.push(Edit::AddPort(
                    Port::Output(Outlet {
                        operator: id,
                        slot: 0,
                    }),
                    None,
                ));
                let count = if id == sink { branches.len() } else { 1 };
                for slot in 0..count {
                    edits.push(Edit::AddPort(
                        Port::Input(Inlet {
                            operator: id,
                            slot: slot as u32,
                        }),
                        None,
                    ));
                }
            }
        }
        for (slot, &branch) in branches.iter().enumerate() {
            edits.push(Edit::Connect(Wire {
                from: Outlet {
                    operator: source,
                    slot: 0,
                },
                to: Inlet {
                    operator: branch,
                    slot: 0,
                },
            }));
            edits.push(Edit::Connect(Wire {
                from: Outlet {
                    operator: branch,
                    slot: 0,
                },
                to: Inlet {
                    operator: sink,
                    slot: slot as u32,
                },
            }));
        }
        let mut document = GraphDocument::default();
        document
            .apply(Segment {
                base_revision: 0,
                edits,
            })
            .unwrap();
        let mut view = ModelView::from_document(document, false, LayoutCache::default());
        view.focus = Some(view.indices[&root]);
        view.selected = view.indices[&branches[0]];
        view.explosion = 2;
        for flat in [false, true] {
            if view.flat_flow != flat {
                view.toggle_flat_flow();
            }
            for pixels in [false, true] {
                view.selected = view.indices[&branches[0]];
                view.selected_wire = None;
                let scene = view.scene(40, 24, pixels, None);
                let mut candidates = Candidates::new(&view, &scene, None).unwrap();
                let local_count = candidates.choices.len();
                candidates.movement(true);
                let local = candidates.selected();
                candidates.toggle_scope(&view, &scene);
                assert!(candidates.scope_active);
                assert!(candidates.choices.len() > local_count);
                assert!(candidates.choices.windows(2).all(|pair| (
                    pair[0].wire,
                    pair[0].displayed
                ) < (
                    pair[1].wire,
                    pair[1].displayed
                )));
                let remote = candidates
                    .choices
                    .iter()
                    .position(|choice| {
                        !scene
                            .handoffs
                            .iter()
                            .any(|visible| visible.wire == choice.wire)
                    })
                    .unwrap();
                candidates.select(remote);
                let chosen = candidates.selected();
                assert!(
                    candidates
                        .render(&view, 40, 24, Locale::English)
                        .contains("Loaded scope")
                );
                candidates.toggle_scope(&view, &scene);
                assert_eq!(candidates.selected(), local);
                candidates.toggle_scope(&view, &scene);
                assert_eq!(candidates.selected(), chosen);
                let revision = view.document.revision();
                super::super::interaction::select_matrix_wire(&mut view, chosen.0, chosen.1);
                assert_eq!(view.selected_wire, Some(chosen.0));
                assert_eq!(view.document.revision(), revision);
                let after = view.scene(40, 24, pixels, None);
                assert!(after.handoffs.iter().any(|route| route.wire == chosen.0));
            }
        }
    }

    #[test]
    fn overlapping_candidates_are_stably_ordered_and_keep_distinct_ports() {
        let offline = super::super::offline::Offline::new(Case::CrossedPorts);
        let view = offline.view();
        for pixels in [false, true] {
            let scene = view.scene(80, 30, pixels, None);
            let top = super::super::SCENE_HEADER + scene.tooltip_rows;
            let (x, y) = (top..top + scene.labels.len() - 1)
                .flat_map(|y| (0..scene.width).map(move |x| (x, y)))
                .find(|&(x, y)| scene.handoffs_at(x, y).len() > 1)
                .unwrap();
            let mut candidates = Candidates::new(&view, &scene, Some((x, y))).unwrap();
            assert!(
                candidates
                    .choices
                    .windows(2)
                    .all(|pair| pair[0].wire < pair[1].wire)
            );
            let first = candidates.selected();
            candidates.movement(true);
            let second = candidates.selected();
            assert_ne!(first, second);
            let text = candidates.render(&view, 80, 30, Locale::English);
            assert!(text.contains(".out["));
            assert!(text.contains(".in["));
            let before = view.selected_wire;
            let _cancelled = candidates;
            assert_eq!(view.selected_wire, before);
        }
    }

    #[test]
    fn long_candidate_lists_page_and_keep_both_underlying_endpoints_visible() {
        let offline = super::super::offline::Offline::new(Case::TwinMerge);
        let view = offline.view();
        let scene = view.scene(40, 24, false, None);
        let wire = scene.handoffs.first().unwrap().wire;
        let choices = (0..30)
            .map(|slot| {
                let mut underlying = wire;
                underlying.to.slot = slot;
                Choice {
                    wire: underlying,
                    displayed: wire,
                }
            })
            .collect();
        let mut candidates = Candidates {
            choices,
            selected: 0,
            pointed: false,
            scope_active: false,
            local_empty: false,
            scope_unavailable: false,
            saved_local: None,
            saved_scope: None,
        };
        candidates.page(true, 10);
        assert_eq!(candidates.selected, 1);
        candidates.select(29);
        assert_eq!(candidates.index_at(2, 10), Some(29));
        assert_eq!(candidates.index_at(4, 10), None);
        let text = candidates.render(&view, 40, 10, Locale::English);
        assert!(text.contains("30/30"), "{text}");
        assert!(text.contains("Underlying wire:"), "{text}");
        assert!(text.contains(".out["), "{text}");
        assert!(text.contains(".in[29]"), "{text}");
        assert!(text.contains("Esc cancel"), "{text}");
        assert!(text.lines().count() <= 10, "{text}");
        assert!(
            text.lines()
                .all(|line| crate::terminal::text::display_width(line) <= 40),
            "{text}"
        );
        candidates.page(false, 10);
        assert_eq!(candidates.selected, 28);
    }
}
