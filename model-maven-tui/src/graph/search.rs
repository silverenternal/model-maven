//! Search over the currently materialized graph. Browsing never mutates the view.
use super::{ModelView, OperatorId, Port, Wire, push};
use crate::terminal::locale::Locale;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Target {
    Action(Action),
    Module(OperatorId),
    Port(Port, Option<Wire>),
    Source(OperatorId, Vec<OperatorId>),
    SourcePort(Port, Option<Wire>, Vec<OperatorId>),
    EvidenceInput(usize),
    Diagnostic(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Action {
    Flow,
    Explosion,
    Matrix,
    FailurePath,
    RepairDiff,
    Reset,
}

impl Action {
    const ALL: [Self; 6] = [
        Self::Flow,
        Self::Explosion,
        Self::Matrix,
        Self::FailurePath,
        Self::RepairDiff,
        Self::Reset,
    ];

    fn names(self) -> [&'static str; 3] {
        match self {
            Self::Flow => [
                "Switch 2D / 3D flow",
                "切换二维 / 三维流图",
                "切換二維 / 三維流圖",
            ],
            Self::Explosion => [
                "Change slice spacing",
                "调整爆炸图切片间距",
                "調整爆炸圖切片間距",
            ],
            Self::Matrix => ["Open port matrix", "打开端口矩阵", "開啟連接埠矩陣"],
            Self::FailurePath => [
                "Open failure / call path",
                "打开故障 / 调用路径",
                "開啟故障 / 呼叫路徑",
            ],
            Self::RepairDiff => ["Open repair difference", "打开修复差异", "開啟修復差異"],
            Self::Reset => ["Reset view and layout", "重置视角与布局", "重設視角與版面"],
        }
    }

    fn next_names(self, view: &ModelView<'_>) -> [&'static str; 3] {
        match self {
            Self::Flow if view.flat_flow => [
                "Switch to 3D spatial view",
                "切换到三维空间图",
                "切換到三維空間圖",
            ],
            Self::Flow => ["Switch to 2D flow view", "切换到二维流图", "切換到二維流圖"],
            Self::Explosion if view.flat_flow => {
                ["Spread to exploded x1", "展开爆炸图 x1", "展開爆炸圖 x1"]
            }
            Self::Explosion if view.explosion == 0 => [
                "Spread slices · level 1/2",
                "展开切片 · 第 1/2 档",
                "展開切片 · 第 1/2 檔",
            ],
            Self::Explosion if view.explosion == 1 => [
                "Spread slices · level 2/2",
                "展开切片 · 第 2/2 档",
                "展開切片 · 第 2/2 檔",
            ],
            Self::Explosion => ["Reset slice spacing", "恢复切片间距", "還原切片間距"],
            _ => self.names(),
        }
    }

    fn compact_next_names(self, view: &ModelView<'_>) -> [&'static str; 3] {
        match self {
            Self::Flow if view.flat_flow => ["3D spatial view", "三维空间图", "三維空間圖"],
            Self::Flow => ["2D flow view", "二维流图", "二維流圖"],
            Self::Explosion if view.flat_flow => ["Exploded x1", "爆炸 x1", "爆炸 x1"],
            Self::Explosion if view.explosion == 0 => ["Exploded x1", "爆炸 x1", "爆炸 x1"],
            Self::Explosion if view.explosion == 1 => ["Exploded x2", "爆炸 x2", "爆炸 x2"],
            Self::Explosion => ["Reset slices", "收起切片", "收起切片"],
            Self::Matrix => ["Port matrix", "端口矩阵", "連接埠矩陣"],
            Self::FailurePath => ["Failure path", "故障路径", "故障路徑"],
            Self::RepairDiff => ["Repair diff", "修复差异", "修復差異"],
            Self::Reset => ["Reset view/layout", "重置视图/布局", "重設視圖/版面"],
        }
    }

    fn key_hint(self, view: &ModelView<'_>) -> &'static str {
        match self {
            Self::Flow if view.flat_flow => "1/z",
            Self::Flow
                if view.flow_layout().is_some()
                    || (!view.hierarchy && view.unconnected_flow().is_some()) =>
            {
                "2/z"
            }
            Self::Flow => "—",
            Self::Explosion => "3/4",
            Self::Matrix => "M",
            Self::FailurePath => "F",
            Self::RepairDiff => "D",
            Self::Reset => "r",
        }
    }

    pub fn unavailable(
        self,
        view: &ModelView<'_>,
        graph_available: bool,
        matrix_available: bool,
        failure_available: bool,
        diff_available: bool,
    ) -> Option<&'static str> {
        if !graph_available {
            return Some("graph");
        }
        match self {
            Self::Flow
                if view.flow_layout().is_none()
                    && !view.flat_flow
                    && (view.hierarchy || view.unconnected_flow().is_none()) =>
            {
                Some("flow")
            }
            Self::Matrix if !matrix_available => Some("matrix"),
            Self::FailurePath if !failure_available => Some("call"),
            Self::RepairDiff if !diff_available => Some("preview"),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Search {
    query: Vec<u8>,
    selected: usize,
    query_selected: bool,
    matrix_available: bool,
    graph_available: bool,
    failure_available: bool,
    diff_available: bool,
}

impl Search {
    pub fn new() -> Self {
        Self {
            query: Vec::new(),
            selected: 0,
            query_selected: false,
            matrix_available: true,
            graph_available: true,
            failure_available: false,
            diff_available: false,
        }
    }

    pub fn with_evidence(mut self, failure_available: bool, diff_available: bool) -> Self {
        self.failure_available = failure_available;
        self.diff_available = diff_available;
        self
    }

    pub fn with_capabilities(mut self, graph_available: bool, matrix_available: bool) -> Self {
        self.graph_available = graph_available;
        self.matrix_available = matrix_available;
        self
    }

    pub fn refresh_selection(&mut self, view: &ModelView<'_>) {
        let count = self.results(view).len();
        self.selected = self.selected.min(count.saturating_sub(1));
    }

    pub fn resume(&mut self) {
        self.query_selected = !self.query.is_empty();
    }

    pub fn available(&self, view: &ModelView<'_>, action: Action) -> bool {
        action
            .unavailable(
                view,
                self.graph_available,
                self.matrix_available,
                self.failure_available,
                self.diff_available,
            )
            .is_none()
    }

    pub fn input(&mut self, byte: u8) {
        match byte {
            21 => self.query.clear(),
            8 | 127 => {
                if self.query_selected {
                    self.query.clear();
                } else if !self.query.is_empty() {
                    let mut start = self.query.len() - 1;
                    while start > 0 && self.query[start] & 0xc0 == 0x80 {
                        start -= 1;
                    }
                    self.query.truncate(start);
                }
            }
            32..=255 => {
                if self.query_selected {
                    self.query.clear();
                }
                self.query.push(byte);
            }
            _ => return,
        }
        self.query_selected = false;
        self.selected = 0;
    }

    fn results(&self, view: &ModelView<'_>) -> Vec<(Target, String)> {
        let query = String::from_utf8_lossy(&self.query).to_lowercase();
        let mut results = Vec::new();
        for action in Action::ALL {
            let names = action.names();
            let next_names = action.next_names(view);
            if names
                .iter()
                .chain(next_names.iter())
                .any(|name| name.to_lowercase().contains(&query))
            {
                results.push((Target::Action(action), String::new()));
            }
        }
        if query.is_empty() {
            return results;
        }
        if let Some(evidence) = &view.live_evidence {
            for (code, message) in &evidence.diagnostics {
                let label = format!("{code} · {message}");
                if label.to_lowercase().contains(&query) {
                    results.push((Target::Diagnostic(code.clone()), label));
                }
            }
            for (index, input) in evidence.inputs.iter().enumerate() {
                if let Some(path) = &input.label {
                    let shape = input
                        .actual
                        .as_ref()
                        .map_or_else(|| "?".into(), |(shape, _)| shape.label());
                    let label = format!(
                        "{} #{} · {path} · {} · {shape}",
                        evidence.path,
                        evidence
                            .invocation
                            .map_or("?".into(), |value| value.to_string()),
                        input.metadata.as_deref().unwrap_or("?")
                    );
                    if label.to_lowercase().contains(&query) {
                        results.push((Target::EvidenceInput(index), label));
                    }
                }
            }
        }
        for layer in &view.layers {
            let label = format!("{} · {}", layer.path, layer.kind);
            if label.to_lowercase().contains(&query) {
                results.push((Target::Module(layer.id), label));
            }
            for (port, _) in view.document.ports_for(layer.id) {
                let (direction, slot) = match port {
                    Port::Input(p) => ("in", p.slot),
                    Port::Output(p) => ("out", p.slot),
                };
                let label = format!("{}.{direction}[{slot}]", layer.path);
                if label.to_lowercase().contains(&query) {
                    let wires: Vec<_> = match port {
                        Port::Input(to) => view
                            .document
                            .source(to)
                            .map(|from| Wire { from, to })
                            .into_iter()
                            .collect(),
                        Port::Output(from) => view
                            .document
                            .consumers(from)
                            .map(|to| Wire { from, to })
                            .collect(),
                    };
                    if wires.is_empty() {
                        results.push((Target::Port(port, None), label));
                    } else {
                        for wire in wires {
                            let (other, direction, slot, arrow) = match port {
                                Port::Input(_) => (wire.from.operator, "out", wire.from.slot, "←"),
                                Port::Output(_) => (wire.to.operator, "in", wire.to.slot, "→"),
                            };
                            let path = view
                                .document
                                .operator(other)
                                .map_or("?", |operator| operator.path.as_str());
                            results.push((
                                Target::Port(port, Some(wire)),
                                format!("{label} {arrow} {path}.{direction}[{slot}]"),
                            ));
                        }
                    }
                }
            }
        }
        if let Some(project) = &view.source_project {
            for item in project.search(view, &query) {
                let entry = if let Some((port, wire, label)) = item.port {
                    (Target::SourcePort(port, wire, item.parents), label)
                } else {
                    (
                        Target::Source(item.id, item.parents),
                        format!(
                            "{} · {} · models/generator.py:{}",
                            item.path, item.kind, item.line
                        ),
                    )
                };
                results.push(entry);
            }
        }
        results
    }

    pub fn movement(&mut self, view: &ModelView<'_>, down: bool) {
        let count = self.results(view).len();
        if count > 0 {
            self.selected = if down {
                (self.selected + 1).min(count - 1)
            } else {
                self.selected.saturating_sub(1)
            };
        }
    }

    pub fn selected(&self, view: &ModelView<'_>) -> Option<Target> {
        self.results(view)
            .get(self.selected)
            .map(|entry| entry.0.clone())
    }

    pub fn render(
        &self,
        view: &ModelView<'_>,
        width: usize,
        height: usize,
        locale: Locale,
    ) -> String {
        let mut output = String::new();
        push(
            &mut output,
            if width < 58 {
                locale.text(
                    "Search · graph/source/evidence/commands",
                    "搜索 · 图/源码/证据/命令",
                    "搜尋 · 圖/原始碼/證據/命令",
                )
            } else {
                locale.text(
                    "Search · loaded graph / source / evidence / commands",
                    "搜索 · 已加载图 / 源码 / 证据 / 命令",
                    "搜尋 · 已載入圖 / 原始碼 / 證據 / 命令",
                )
            },
            width,
        );
        push(
            &mut output,
            &format!(
                "/ {}{}{}",
                if self.query_selected { "[" } else { "" },
                crate::workbench::path_tail(
                    &String::from_utf8_lossy(&self.query),
                    width.saturating_sub(2 + usize::from(self.query_selected) * 2)
                ),
                if self.query_selected { "]" } else { "" },
            ),
            width,
        );
        let results = self.results(view);
        let count = height.saturating_sub(5).max(1);
        let start = self.selected.saturating_sub(count - 1);
        for (index, (target, label)) in results.iter().enumerate().skip(start).take(count) {
            if let Target::Action(action) = target {
                let narrow = width < 58;
                let [en, zh, traditional] = if narrow {
                    action.compact_next_names(view)
                } else {
                    action.next_names(view)
                };
                let name = locale.text(en, zh, traditional);
                let reason = match action.unavailable(
                    view,
                    self.graph_available,
                    self.matrix_available,
                    self.failure_available,
                    self.diff_available,
                ) {
                    Some("graph") => locale.text(
                        if narrow {
                            " · close view first"
                        } else {
                            " · close editor or detail view first"
                        },
                        if narrow {
                            " · 请先关闭详情"
                        } else {
                            " · 请先关闭编辑器或详情视图"
                        },
                        if narrow {
                            " · 請先關閉詳情"
                        } else {
                            " · 請先關閉編輯器或詳情視圖"
                        },
                    ),
                    Some("flow") => locale.text(
                        if narrow {
                            " · needs flow"
                        } else {
                            " · requires a flow layout"
                        },
                        if narrow {
                            " · 需要流图"
                        } else {
                            " · 需要流图布局"
                        },
                        if narrow {
                            " · 需要流程圖"
                        } else {
                            " · 需要流程圖版面"
                        },
                    ),
                    Some("spatial") => locale.text(
                        if narrow {
                            " · needs 3D"
                        } else {
                            " · switch to 3D first"
                        },
                        if narrow {
                            " · 需三维"
                        } else {
                            " · 请先切换到三维"
                        },
                        if narrow {
                            " · 需三維"
                        } else {
                            " · 請先切換到三維"
                        },
                    ),
                    Some("matrix") if narrow => {
                        locale.text(" · needs ports", " · 需要端口", " · 需要連接埠")
                    }
                    Some("call") => locale.text(" · need call", " · 需要调用", " · 需要呼叫"),
                    Some("preview") => locale.text(" · need preview", " · 需要预览", " · 需要預覽"),
                    Some(_) => locale.text(
                        " · requires loaded ports and graph view",
                        " · 需要已加载端口及图视图",
                        " · 需要已載入連接埠及圖視圖",
                    ),
                    None => "",
                };
                push(
                    &mut output,
                    &format!(
                        "{}[{}] {name}{reason}",
                        if index == self.selected { ">" } else { " " },
                        action.key_hint(view)
                    ),
                    width,
                );
                continue;
            }
            if matches!(target, Target::SourcePort(_, _, _)) {
                let prefix = format!(
                    "{} [{}] ",
                    if index == self.selected { ">" } else { " " },
                    locale.text("source", "源码", "原始碼")
                );
                let budget = width.saturating_sub(crate::terminal::text::display_width(&prefix));
                let label = [" → ", " ← "]
                    .into_iter()
                    .find_map(|arrow| {
                        label
                            .split_once(arrow)
                            .map(|(left, right)| (left, arrow, right))
                    })
                    .map_or_else(
                        || crate::workbench::path_tail(label, budget),
                        |(left, arrow, right)| {
                            let right_width =
                                crate::terminal::text::display_width(right).min(budget / 2);
                            let left_width = budget.saturating_sub(
                                crate::terminal::text::display_width(arrow) + right_width,
                            );
                            format!(
                                "{}{arrow}{}",
                                crate::workbench::path_tail(left, left_width),
                                crate::workbench::path_tail(right, right_width)
                            )
                        },
                    );
                push(&mut output, &format!("{prefix}{label}"), width);
                continue;
            }
            if matches!(target, Target::Source(_, _)) {
                push(
                    &mut output,
                    &format!(
                        "{} [{}] {label}",
                        if index == self.selected { ">" } else { " " },
                        locale.text("source", "源码", "原始碼")
                    ),
                    width,
                );
                continue;
            }
            let kind = match target {
                Target::Action(_) => unreachable!(),
                Target::EvidenceInput(_) => locale
                    .text("observed input", "观测输入", "觀測輸入")
                    .to_owned(),
                Target::Diagnostic(_) => locale.text("diagnostic", "诊断", "診斷").to_owned(),
                Target::Module(_) => locale.text("module", "模块", "模組").to_owned(),
                Target::Source(_, _) => locale
                    .text("source · not loaded", "源码 · 未加载", "原始碼 · 未載入")
                    .to_owned(),
                Target::SourcePort(_, _, _) => locale
                    .text(
                        "source port · not loaded",
                        "源码端口 · 未加载",
                        "原始碼連接埠 · 未載入",
                    )
                    .to_owned(),
                Target::Port(_, None) => locale
                    .text(
                        "port · no recorded wire",
                        "端口 · 无已记录连线",
                        "連接埠 · 無已記錄連線",
                    )
                    .to_owned(),
                Target::Port(_, Some(_)) => locale
                    .text("port · wire", "端口 · 连线", "連接埠 · 連線")
                    .to_owned(),
            };
            push(
                &mut output,
                &format!(
                    "{} {label} · {kind}",
                    if index == self.selected { ">" } else { " " }
                ),
                width,
            );
        }
        if results.is_empty() {
            push(
                &mut output,
                locale.text(
                    "No loaded matches",
                    "已加载范围内无匹配",
                    "已載入範圍內無相符項目",
                ),
                width,
            );
        }
        push(
            &mut output,
            &format!(
                "{} / {}",
                if results.is_empty() {
                    0
                } else {
                    self.selected + 1
                },
                results.len()
            ),
            width,
        );
        push(
            &mut output,
            if width >= 72 {
                locale.text(
                    "↑↓ select · Enter open · Ctrl+U clear · Esc cancel",
                    "↑↓ 选择 · Enter 打开 · Ctrl+U 清除 · Esc 取消",
                    "↑↓ 選擇 · Enter 開啟 · Ctrl+U 清除 · Esc 取消",
                )
            } else {
                locale.text(
                    "↑↓ select · Enter open · Esc cancel",
                    "↑↓ 选择 · Enter 打开 · Esc 取消",
                    "↑↓ 選擇 · Enter 開啟 · Esc 取消",
                )
            },
            width,
        );
        output
    }
}

pub(super) fn open(view: &mut ModelView<'_>, target: Target) {
    let original = view.bookmark();
    let (id, wire, port) = match target {
        Target::Action(_) => return,
        Target::EvidenceInput(_) | Target::Diagnostic(_) => return,
        Target::Module(id) => (id, None, None),
        Target::Port(port, wire) => (port.operator(), wire, wire.is_none().then_some(port)),
        Target::Source(id, parents) => {
            if let Err(error) = super::source::Project::reveal(view, &parents) {
                view.source_error = Some(error.to_string());
                return;
            }
            (id, None, None)
        }
        Target::SourcePort(port, wire, parents) => {
            if let Err(error) = super::source::Project::reveal(view, &parents) {
                view.source_error = Some(error.to_string());
                return;
            }
            let exists = view
                .document
                .ports_for(port.operator())
                .any(|(item, _)| item == port);
            let connected =
                wire.is_none_or(|wire| view.document.source(wire.to) == Some(wire.from));
            if !exists || !connected {
                view.source_error = Some("Source port changed while loading; search again.".into());
                return;
            }
            (port.operator(), wire, wire.is_none().then_some(port))
        }
    };
    let Some(&index) = view.indices.get(&id) else {
        return;
    };
    view.trail.push(original);
    view.focus = view.parents[index]
        .filter(|&parent| view.source_project.is_none() || view.roots.as_slice() != [parent]);
    view.selected = index;
    view.selected_wire = wire;
    view.selected_port = port;
    view.hover = None;
    view.hover_handoff = None;
    view.viewport = Default::default();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::offline::Offline;
    use model_maven_tui::diagnosis::Case;

    fn type_query(search: &mut Search, query: &str) {
        for byte in query.bytes() {
            search.input(byte);
        }
    }

    #[test]
    fn compact_search_header_keeps_every_search_scope_visible() {
        let offline = Offline::new(Case::CrossedPorts);
        let view = offline.view();
        let search = Search::new();
        let english = search.render(&view, 40, 24, Locale::English);
        assert!(
            english.contains("Search · graph/source/evidence/commands"),
            "{english}"
        );
        let simplified = search.render(&view, 40, 24, Locale::Simplified);
        assert!(
            simplified.contains("搜索 · 图/源码/证据/命令"),
            "{simplified}"
        );
        let traditional = search.render(&view, 40, 24, Locale::Traditional);
        assert!(
            traditional.contains("搜尋 · 圖/原始碼/證據/命令"),
            "{traditional}"
        );
    }

    #[test]
    fn graph_action_search_shows_direct_keyboard_shortcuts() {
        let offline = Offline::new(Case::CrossedPorts);
        let view = offline.view();
        let frame = Search::new().render(&view, 40, 24, Locale::English);
        for hint in ["[2/z]", "[3/4]", "[M]", "[F]", "[D]", "[r]"] {
            assert!(frame.contains(hint), "missing {hint} in:\n{frame}");
        }
        let simplified = Search::new().render(&view, 40, 24, Locale::Simplified);
        assert!(simplified.contains("[2/z] 二维流图"), "{simplified}");
        assert!(simplified.contains("[M] 端口矩阵"), "{simplified}");
        let traditional = Search::new().render(&view, 40, 24, Locale::Traditional);
        assert!(traditional.contains("[2/z] 二維流圖"), "{traditional}");
        assert!(traditional.contains("[M] 連接埠矩陣"), "{traditional}");

        let mut flat = offline.view();
        flat.toggle_flat_flow();
        let flat_frame = Search::new().render(&flat, 40, 24, Locale::English);
        assert!(flat_frame.contains("[1/z]"), "{flat_frame}");

        let modules = vec![
            serde_json::json!({"path":"", "type":"ModuleList"}),
            serde_json::json!({"path":"only", "type":"Linear"}),
        ];
        let mut hierarchy = ModelView::new(Some(&modules));
        hierarchy.hierarchy = true;
        let hierarchy_frame = Search::new().render(&hierarchy, 40, 24, Locale::English);
        assert!(
            hierarchy_frame.contains("[—] 2D flow view"),
            "{hierarchy_frame}"
        );
        assert!(!hierarchy_frame.contains("[2/z]"), "{hierarchy_frame}");
    }

    #[test]
    fn resumed_query_is_selected_for_replacement_and_can_be_cleared() {
        let mut search = Search::new();
        type_query(&mut search, "old query");
        search.resume();
        assert!(search.query_selected);
        search.input(b'n');
        assert_eq!(search.query, b"n");
        search.resume();
        search.input(127);
        assert!(search.query.is_empty());
        search.input(b'v');
        search.input(21);
        assert!(search.query.is_empty());
    }

    #[test]
    fn search_finds_a_port_and_opens_its_observed_wire_with_return_history() {
        let offline = Offline::new(Case::CrossedPorts);
        let mut view = offline.view();
        let before = view.bookmark();
        let mut search = Search::new();
        type_query(&mut search, "merge.in[1]");
        let target = search.selected(&view).expect("port match");
        assert!(matches!(target, Target::Port(Port::Input(_), Some(_))));
        open(&mut view, target);
        let wire = view.selected_wire.expect("connected port");
        assert_eq!(wire.to.slot, 1);
        assert!(view.zoom_out());
        assert_eq!(view.bookmark().selected, before.selected);
        assert_eq!(view.bookmark().focus, before.focus);
        assert_eq!(view.bookmark().wire, before.wire);
    }

    #[test]
    fn unconnected_port_has_its_own_focus_and_is_restored_by_history() {
        let offline = Offline::new(Case::CrossedPorts);
        let mut view = offline.view();
        let original = view.bookmark();
        let mut search = Search::new();
        type_query(&mut search, "adapter.out[0]");
        let target = search.selected(&view).expect("port match");
        let Target::Port(port, None) = target else {
            panic!("expected an unconnected port")
        };
        open(&mut view, Target::Port(port, None));
        assert_eq!(view.selected_port, Some(port));
        assert!(view.selected_wire.is_none());
        let scene = view.scene(80, 24, false, None);
        let frame = view.render(&scene, Locale::English, false, false);
        assert!(frame.contains("net.adapter.out[0]"), "{frame}");
        assert!(
            frame.contains("No recorded wire · status unknown"),
            "{frame}"
        );
        let port_bookmark = view.bookmark();
        let mut another = Search::new();
        type_query(&mut another, "merge.in[1]");
        let target = another.selected(&view).unwrap();
        open(&mut view, target);
        assert!(view.selected_port.is_none());
        assert!(view.zoom_out());
        assert!(view.bookmark() == port_bookmark);
        assert!(view.zoom_out());
        assert!(view.bookmark() == original);
    }

    #[test]
    fn utf8_query_backspace_preserves_previous_character() {
        let mut search = Search::new();
        type_query(&mut search, "模型x");
        search.input(127);
        assert_eq!(search.query, "模型".as_bytes());
        search.input(127);
        assert_eq!(search.query, "模".as_bytes());
    }

    #[test]
    fn search_matches_unicode_module_paths() {
        let modules = vec![serde_json::json!({"path": "模型.编码器", "type": "Linear"})];
        let view = ModelView::new(Some(&modules));
        let mut search = Search::new();
        type_query(&mut search, "编码器");
        assert!(matches!(search.selected(&view), Some(Target::Module(_))));
    }

    #[test]
    fn live_search_finds_nested_tensor_and_error_code_in_the_same_call() {
        let mut view = ModelView::new(None);
        let raw_report = format!(
            "{{\"data\":[{}],\"diagnostics\":[{{\"code\":\"FORWARD_FAILED\"}}]}}",
            (0..40)
                .map(|n| format!("{n:08}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        view.live_evidence = Some(super::super::LiveEvidence {
            path: "model.head".into(),
            ambiguous_registration: false,
            last_observed: true,
            from_calls: false,
            reused: false,
            selected_input: None,
            show_input: false,
            #[cfg(unix)]
            graph_target_unresolved: false,
            #[cfg(unix)]
            projection: None,
            input: None,
            invocation: Some(2),
            invocation_id: Some("forward:call-2".into()),
            expected: None,
            raw_report,
            #[cfg(unix)]
            route_cache: std::cell::OnceCell::new(),
            inputs: super::super::failure::observed_inputs(
                &serde_json::json!({"kind":"dict","items":{"kwargs":{"kind":"dict","items":{"mask":{"kind":"tensor","shape":[2,7],"dtype":"torch.bool"}}}}}),
                None,
            ),
            diagnostics: vec![("FORWARD_FAILED".into(), "shape mismatch".into())],
        });
        let mut tensor = Search::new();
        type_query(&mut tensor, "kwargs.mask");
        let Some(Target::EvidenceInput(index)) = tensor.selected(&view) else {
            panic!("missing observed tensor");
        };
        let mut path = super::super::failure::Path::live(&view).unwrap();
        path.select_input(index);
        let frame = path.render(&view, 80, 24, Locale::English, None);
        assert!(frame.contains("inputs.kwargs.mask"));
        assert!(frame.contains("2×7 (observed)"));
        assert!(frame.contains("forward:call-2"));
        let mut diagnostic = Search::new();
        type_query(&mut diagnostic, "forward_failed");
        let Some(Target::Diagnostic(code)) = diagnostic.selected(&view) else {
            panic!("missing diagnostic");
        };
        path.locate_report(&view, &code, 40, 6, Locale::English);
        let report = path.render(&view, 40, 6, Locale::English, None);
        assert!(report.contains("FORWARD_FAILED"), "{report}");
    }

    #[test]
    fn command_search_is_multilingual_and_displays_unavailable_reason() {
        let offline = Offline::new(Case::CrossedPorts);
        let mut view = offline.view();
        let mut search = Search::new();
        type_query(&mut search, "切片");
        assert_eq!(
            search.selected(&view),
            Some(Target::Action(Action::Explosion))
        );
        assert!(search.available(&view, Action::Explosion));
        view.toggle_flat_flow();
        assert!(search.available(&view, Action::Explosion));
        assert!(
            search
                .render(&view, 80, 24, Locale::English)
                .contains("Spread to exploded x1")
        );
        let disabled = Search::new().with_capabilities(false, false);
        assert!(!disabled.available(&view, Action::Reset));
        assert!(
            disabled
                .render(&view, 80, 24, Locale::English)
                .contains("close editor or detail view first")
        );
    }

    #[test]
    fn command_search_names_the_next_graph_preset_at_narrow_and_wide_sizes() {
        let offline = Offline::new(Case::CrossedPorts);
        let mut view = offline.view();
        let mut flow = Search::new();
        type_query(&mut flow, "flow");
        for width in [40, 100] {
            let frame = flow.render(&view, width, 24, Locale::English);
            let expected = if width < 58 {
                "2D flow view"
            } else {
                "Switch to 2D flow view"
            };
            assert!(frame.contains(expected), "{frame}");
        }
        view.toggle_flat_flow();
        assert!(
            flow.render(&view, 40, 24, Locale::English)
                .contains("3D spatial view")
        );
        assert!(
            flow.render(&view, 40, 24, Locale::Simplified)
                .contains("三维空间图")
        );
        let narrow_palette = Search::new().render(&view, 40, 24, Locale::English);
        assert!(
            narrow_palette.contains("[3/4] Exploded x1"),
            "{narrow_palette}"
        );
        assert!(!narrow_palette.contains("needs 3D"), "{narrow_palette}");
        assert!(Search::new().available(&view, Action::Explosion));
        assert!(
            Search::new()
                .render(&view, 40, 24, Locale::Simplified)
                .contains("[3/4] 爆炸 x1")
        );

        view.toggle_flat_flow();
        let mut explosion = Search::new();
        type_query(&mut explosion, "spacing");
        for (level, expected) in [(0, "Exploded x1"), (1, "Exploded x2"), (2, "Reset slices")] {
            view.explosion = level;
            assert!(
                explosion
                    .render(&view, 40, 24, Locale::English)
                    .contains(expected)
            );
        }
        view.explosion = 2;
        let mut reset = Search::new();
        type_query(&mut reset, "reset slice spacing");
        assert_eq!(
            reset.selected(&view),
            Some(Target::Action(Action::Explosion))
        );
        view.explosion = 1;
        assert!(
            explosion
                .render(&view, 40, 24, Locale::Traditional)
                .contains("爆炸 x2")
        );
    }

    #[test]
    fn failure_and_difference_commands_explain_missing_evidence_at_40_columns() {
        let view = ModelView::new(None);
        for (query, action, reason) in [
            ("failure", Action::FailurePath, "need call"),
            ("repair", Action::RepairDiff, "need preview"),
        ] {
            let mut search = Search::new();
            type_query(&mut search, query);
            assert_eq!(search.selected(&view), Some(Target::Action(action)));
            assert!(!search.available(&view, action));
            let screen = search.render(&view, 40, 24, Locale::English);
            assert!(screen.contains(reason), "{screen}");
            let search = search.with_evidence(true, true);
            assert!(search.available(&view, action));
        }
        let mut chinese = Search::new().with_evidence(true, true);
        type_query(&mut chinese, "修复差异");
        assert_eq!(
            chinese.selected(&view),
            Some(Target::Action(Action::RepairDiff))
        );
    }

    #[test]
    fn fanout_search_lists_and_selects_each_recorded_consumer() {
        let offline = Offline::new(Case::CrossedPorts);
        let mut view = offline.view();
        let first = view.document.wires().next().unwrap();
        let to = super::super::Inlet {
            operator: first.to.operator,
            slot: 99,
        };
        let second = Wire {
            from: first.from,
            to,
        };
        view.document
            .apply(super::super::Segment {
                base_revision: view.document.revision(),
                edits: vec![
                    super::super::Edit::AddPort(Port::Input(to), None),
                    super::super::Edit::Connect(second),
                ],
            })
            .unwrap();
        let path = view
            .document
            .operator(first.from.operator)
            .unwrap()
            .path
            .clone();
        let mut search = Search::new();
        type_query(&mut search, &format!("{path}.out[{}]", first.from.slot));
        let results = search.results(&view);
        assert!(results.iter().any(|(_, label)| label.contains("in[99]")));
        assert!(
            results
                .iter()
                .any(|(target, _)| *target == Target::Port(Port::Output(first.from), Some(first)))
        );
        let index = results
            .iter()
            .position(|(target, _)| *target == Target::Port(Port::Output(first.from), Some(second)))
            .unwrap();
        search.selected = index;
        let target = search.selected(&view).unwrap();
        open(&mut view, target);
        assert_eq!(view.selected_wire, Some(second));
    }
}
