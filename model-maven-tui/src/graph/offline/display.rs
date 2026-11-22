//! Detail panes for the offline workspace. The controller owns the draft and graph.
use super::super::push;
use super::{
    Action, Comparison, DetailMode, DraftError, EVIDENCE_TRACE_KEY, Offline, PreferenceNotice,
    Stage,
};
#[cfg(unix)]
use super::{DETAIL_EXPAND_KEY, FullDetail};
use crate::terminal::layout::Background;
use crate::terminal::locale::Locale;
use crate::terminal::text;
use crate::workbench::{Preset, Style};
use model_maven_tui::diagnosis::{
    Adjustment, Coverage, Evidence, Origin, Outcome, RecheckSample, WeightImpact,
};
use model_maven_tui::graph_document::Shape;

#[derive(Clone, Copy)]
enum DetailsOutput {
    Screen { focused: bool, axes_visible: bool },
    Transcript,
}

impl DetailsOutput {
    fn back_key(self) -> &'static str {
        match self {
            Self::Transcript => "b",
            Self::Screen { .. } => "Esc",
        }
    }

    fn push(self, out: &mut String, line: &str, width: usize) {
        if matches!(self, Self::Screen { .. }) || text::display_width(line) <= width {
            push(out, line, width);
            return;
        }
        let mut occupied = 0;
        for word in text::plain_text(line).split_whitespace() {
            let length = text::display_width(word);
            if occupied > 0 && occupied + 1 + length > width {
                out.push('\n');
                occupied = 0;
            }
            if length > width {
                let parts = text::wrap(word, width);
                let (last, earlier) = parts.split_last().expect("wide token produces lines");
                for part in earlier {
                    push(out, part, width);
                }
                out.push_str(last);
                occupied = text::display_width(last);
            } else {
                if occupied > 0 {
                    out.push(' ');
                    occupied += 1;
                }
                out.push_str(word);
                occupied += length;
            }
        }
        out.push('\n');
    }
}

impl Offline {
    #[cfg(unix)]
    pub(in crate::graph) fn failure_details(
        &self,
        input: usize,
        locale: Locale,
        width: usize,
    ) -> String {
        self.full_detail_text(
            FullDetail::Call {
                input,
                trace_unavailable: false,
            },
            locale,
            width,
        )
    }
    fn weight_impact(&self, locale: Locale) -> String {
        match self
            .repair()
            .preview
            .weights
            .as_ref()
            .map(|impact| &impact.value)
        {
            Some(WeightImpact::Reset(name)) => {
                format!("{} {name}", locale.text("reset", "重置", "重設"))
            }
            Some(WeightImpact::Reuse(name)) => {
                format!("{} {name}", locale.text("reuse", "复用", "重用"))
            }
            None => locale
                .text("weight impact unknown", "权重影响未知", "權重影響未知")
                .into(),
        }
    }

    #[cfg(unix)]
    pub(in crate::graph) fn scroll_full_detail(
        &mut self,
        forward: bool,
        by_page: bool,
        locale: Locale,
        width: usize,
        height: usize,
    ) -> bool {
        let DetailMode::Full { detail, offset } = self.mode else {
            return false;
        };
        let text = self.full_detail_text(detail, locale, width);
        let rows = detail_page_rows(height);
        let end = text.lines().count().saturating_sub(rows);
        let current = offset.min(end);
        let step = if by_page { rows } else { 1 };
        let offset = if forward {
            (current + step).min(end)
        } else {
            current.saturating_sub(step)
        };
        self.mode = DetailMode::Full { detail, offset };
        true
    }

    #[cfg(unix)]
    pub(in crate::graph) fn render_full_detail(
        &self,
        locale: Locale,
        width: usize,
        height: usize,
    ) -> Option<String> {
        let DetailMode::Full { detail, offset } = self.mode else {
            return None;
        };
        let text = self.full_detail_text(detail, locale, width);
        let rows = detail_page_rows(height);
        let count = text.lines().count();
        let start = offset.min(count.saturating_sub(rows));
        let end = (start + rows).min(count);
        let mut out = String::new();
        let title = match detail {
            FullDetail::Call { .. } => locale.text("Full evidence", "证据全文", "證據全文"),
            FullDetail::Preview => locale.text("Edit preview", "修改预览", "修改預覽"),
            FullDetail::Comparison { .. } => {
                locale.text("Recheck evidence", "重验证据", "重驗證據")
            }
        };
        push(
            &mut out,
            &format!("{title} · {}–{end}/{count}", start + 1),
            width,
        );
        if height <= 1 {
            return Some(out);
        }
        for row in text.lines().skip(start).take(rows) {
            push(&mut out, row, width);
        }
        if height > 2 {
            let padding = height.saturating_sub(out.lines().count() + 1);
            out.extend(std::iter::repeat_n('\n', padding));
            push(
                &mut out,
                locale.text(
                    if width < 40 {
                        "↑↓ Scroll · Esc Back"
                    } else {
                        "↑↓ Scroll · PgUp/PgDn · Esc Back"
                    },
                    if width < 40 {
                        "↑↓ 滚动 · Esc 返回"
                    } else {
                        "↑↓ 滚动 · PgUp/PgDn · Esc 返回"
                    },
                    if width < 40 {
                        "↑↓ 捲動 · Esc 返回"
                    } else {
                        "↑↓ 捲動 · PgUp/PgDn · Esc 返回"
                    },
                ),
                width,
            );
        }
        Some(out)
    }

    #[cfg(unix)]
    fn full_detail_text(&self, detail: FullDetail, locale: Locale, width: usize) -> String {
        let lines = match detail {
            FullDetail::Call {
                input,
                trace_unavailable,
            } => {
                let (report, _) = self.evidence_call();
                let mut lines = self
                    .call_evidence_lines(locale, false, input, false, false)
                    .to_vec();
                lines[0] = format!("{} {}", locale.text("Report", "报告", "報告"), report.id);
                if trace_unavailable {
                    lines.push(unavailable_edge(locale).into());
                }
                lines
            }
            FullDetail::Preview => self.full_preview_lines(locale),
            FullDetail::Comparison { side, .. } => {
                let receipt = &self.repair().receipt;
                let target = self.module_path(receipt.target.operator);
                let mut lines = vec![
                    format!(
                        "{} · v{}",
                        self.fixture.report.batch, receipt.graph_revision
                    ),
                    format!(
                        "{} {target}.in[{}]",
                        locale.text("Target", "目标", "目標"),
                        receipt.target.slot
                    ),
                ];
                for phase in [Comparison::Before, Comparison::After] {
                    lines.extend(
                        self.comparison_fields(phase, side, locale, false)
                            .into_iter()
                            .flatten(),
                    );
                }
                lines
            }
        };
        let mut out = String::new();
        for line in lines {
            DetailsOutput::Transcript.push(&mut out, &line, width);
        }
        out
    }

    #[cfg(unix)]
    fn full_preview_lines(&self, locale: Locale) -> Vec<String> {
        let Stage::Preview(draft) = self.stage else {
            return Vec::new();
        };
        let repair = self.repair();
        let preview = &repair.preview;
        let [_, call_location, observed, ..] =
            self.call_evidence_lines(locale, false, 0, false, false);
        let target = self.module_path(preview.target.operator);
        let change = match &preview.change {
            Adjustment::Shape { before, after } => {
                format!("{} → {}", before.label(), after.value.label())
            }
            Adjustment::Source { after, unavailable } => format!(
                "{} → {}",
                self.output_label(draft.existing.from),
                self.output_label(if draft.unavailable {
                    *unavailable
                } else {
                    after.value
                })
            ),
        };
        let mut lines = vec![
            format!("v{} · {call_location}", draft.revision),
            format!(
                "{} {}",
                locale.text("Report", "报告", "報告"),
                preview.report
            ),
            format!(
                "{} {target}.in[{}]",
                locale.text("Target", "目标", "目標"),
                preview.target.slot
            ),
            change,
            self.wire_label(draft.existing, false),
            self.weight_impact(locale),
            observed,
        ];
        if let Some(error) = draft.error {
            lines.push(draft_rejection(error, draft.revision, locale));
        }
        lines
    }

    #[cfg(unix)]
    pub(in crate::graph) fn graph_hint(
        &self,
        view: &super::super::ModelView<'_>,
        locale: Locale,
    ) -> String {
        use super::Shortcut;
        let pane = if self.preferences.preset == Preset::Shortcuts {
            Shortcut::SwitchPane.key().to_string()
        } else {
            "Tab".into()
        };
        let hint = locale.text("Details · ? Controls", "详情 · ? 操作", "詳情 · ? 操作");
        let modes = format!(
            "L {} · T {}",
            if view.layout_edit {
                locale.text("Layout on", "布局开", "布局開")
            } else {
                locale.text("Layout off", "布局关", "布局關")
            },
            if view.mouse_capture {
                locale.text("Mouse on", "鼠标开", "滑鼠開")
            } else {
                locale.text("Mouse off", "鼠标关", "滑鼠關")
            }
        );
        let diff = if matches!(
            self.stage,
            Stage::Preview(_) | Stage::Committed | Stage::Rechecked { .. }
        ) {
            format!(
                " · D {}",
                locale.text("Repair diff", "修复差异", "修復差異")
            )
        } else {
            String::new()
        };
        format!(
            "MODEL-MAVEN · {pane} {hint} · {modes} · F {}{diff} · {}",
            locale.text("Failure path", "故障路径", "故障路徑"),
            self.welcome_hint(locale)
        )
    }

    #[cfg(unix)]
    pub(in crate::graph) fn graph_wide_hint(
        &self,
        view: &super::super::ModelView<'_>,
        locale: Locale,
    ) -> String {
        use super::Shortcut;
        let pane = if self.preferences.preset == Preset::Shortcuts {
            Shortcut::SwitchPane.key().to_string()
        } else {
            "Tab".into()
        };
        format!(
            "{pane} {} · ? {} · M {} · T {}",
            locale.text("Details", "详情", "詳情"),
            locale.text("Controls", "操作", "操作"),
            locale.text("Port matrix", "端口矩阵", "連接埠矩陣"),
            if view.mouse_capture {
                locale.text("Mouse on", "鼠标开", "滑鼠開")
            } else {
                locale.text("Mouse off", "鼠标关", "滑鼠關")
            }
        )
    }

    #[cfg(unix)]
    fn graph_controls(&self, locale: Locale) -> [String; 9] {
        use super::Shortcut;
        let mut select = "↑↓".to_owned();
        let mut pane = "Tab".to_owned();
        let mut expand = "Enter".to_owned();
        if self.preferences.preset == Preset::Shortcuts {
            select.push_str(&format!(
                " / {}/{}",
                Shortcut::Next.key(),
                Shortcut::Previous.key()
            ));
            pane.push_str(&format!(" / {}", Shortcut::SwitchPane.key()));
            expand.push_str(&format!(" / {}", Shortcut::Expand.key()));
        }
        let details = locale.text("Details", "详情", "詳情");
        let choose = locale.text("Select", "选择", "選擇");
        let unfold = locale.text("Expand", "展开", "展開");
        let back = locale.text("Back", "返回", "返回");
        let trace = locale.text(
            "w Cycle · W Wire list · u Source · d Destination",
            "w 切换 · W 连线列表 · u 源端口 · d 目标端口",
            "w 切換 · W 連線清單 · u 來源埠 · d 目標埠",
        );
        let mut pages = locale
            .text("PgUp/PgDn Pages", "PgUp/PgDn 翻页", "PgUp/PgDn 翻頁")
            .to_owned();
        if self.available(Action::Settings) {
            let settings = Action::Settings;
            pages.push_str(&format!(
                " · {} {}",
                settings.key(self.preferences.preset),
                settings.label(locale)
            ));
        }
        [
            format!("{pane} {details} · {select} {choose} · Shift+Tab"),
            format!(
                "{expand} {unfold} · Esc {back} · {}",
                self.welcome_hint(locale)
            ),
            trace.into(),
            pages,
            locale
                .text("/ Search and commands", "/ 搜索与命令", "/ 搜尋與命令")
                .into(),
            locale
                .text(
                    "M Port matrix · F Failure path",
                    "M 端口矩阵 · F 故障路径",
                    "M 連接埠矩陣 · F 故障路徑",
                )
                .into(),
            locale
                .text(
                    "D Repair diff (during repair)",
                    "D 修复差异（修复时）",
                    "D 修復差異（修復時）",
                )
                .into(),
            locale
                .text(
                    "L Layout edit · T Mouse capture",
                    "L 布局编辑 · T 鼠标捕获",
                    "L 版面編輯 · T 滑鼠擷取",
                )
                .into(),
            locale
                .text(
                    "Edit mode: drag module · Drag canvas to rotate · r Reset",
                    "编辑模式拖模块 · 拖空白旋转 · r 重置",
                    "編輯模式拖模組 · 拖空白旋轉 · r 重置",
                )
                .into(),
        ]
    }

    fn welcome_hint(&self, locale: Locale) -> String {
        let action = Action::Welcome;
        format!(
            "{} {}",
            action.key(self.preferences.preset),
            action.label(locale)
        )
    }

    pub fn render(
        &self,
        locale: Locale,
        width: usize,
        focused: bool,
        axes_visible: bool,
    ) -> String {
        self.render_details(
            locale,
            width,
            DetailsOutput::Screen {
                focused,
                axes_visible,
            },
        )
    }

    pub(super) fn render_transcript(
        &self,
        view: &super::ModelView<'_>,
        locale: Locale,
        width: usize,
    ) -> String {
        let mut out = String::new();
        let transcript = DetailsOutput::Transcript;
        if self.mode != DetailMode::Settings {
            let heading = if self.fixture.snapshot.id != self.fixture.report.snapshot {
                locale.text(
                    "Offline · matching snapshot unavailable",
                    "离线 · 缺少匹配的结构快照",
                    "離線 · 缺少相符的結構快照",
                )
            } else {
                locale.text(
                    "Offline · structure snapshot",
                    "离线 · 结构快照",
                    "離線 · 結構快照",
                )
            };
            transcript.push(
                &mut out,
                &format!("{heading} v{}", view.document.revision()),
                width,
            );
            if let Some(wire) = view.selected_wire {
                transcript.push(&mut out, &self.wire_label(wire, false), width);
            }
        }
        out.push_str(&self.render_details(locale, width, transcript));
        transcript.push(
            &mut out,
            locale.text(
                "Line mode · Enter submits a command · q quits",
                "行式模式 · 回车执行指令 · q 退出",
                "行式模式 · Enter 執行指令 · q 離開",
            ),
            width,
        );
        out
    }

    fn render_details(&self, locale: Locale, width: usize, presentation: DetailsOutput) -> String {
        let (focused, axes_visible) = match presentation {
            DetailsOutput::Screen {
                focused,
                axes_visible,
            } => (focused, axes_visible),
            DetailsOutput::Transcript => (true, false),
        };
        let transcript = matches!(presentation, DetailsOutput::Transcript);
        let compact = !transcript && width < 55;
        let back_key = presentation.back_key();
        let back = locale.text("Back", "返回", "返回");
        let push = |out: &mut String, line: &str, width| presentation.push(out, line, width);
        let mut out = String::new();
        let appearance = super::super::appearance::Appearance {
            style: self.preferences.style,
            background: Background::default(),
        };
        let divider = appearance.divider().to_string();
        match self.mode {
            DetailMode::CallEvidence {
                input,
                trace_unavailable,
            } => {
                return self.render_call_evidence(
                    locale,
                    width,
                    presentation,
                    input,
                    false,
                    trace_unavailable,
                );
            }
            DetailMode::Tracing { input } => {
                return self.render_call_evidence(locale, width, presentation, input, true, false);
            }
            _ => {}
        }
        #[cfg(unix)]
        if self.mode == DetailMode::GraphHelp {
            push(&mut out, &divider.repeat(width.min(100)), width);
            push(
                &mut out,
                locale.text("Graph controls", "图操作", "圖操作"),
                width,
            );
            for control in self.graph_controls(locale) {
                push(&mut out, &control, width);
            }
            return out;
        }
        if self.mode == DetailMode::Receipts {
            push(&mut out, &divider.repeat(width.min(100)), width);
            push(
                &mut out,
                locale.text("Offline receipts", "离线回执", "離線回執"),
                width,
            );
            for (repair, status) in self.fixture.repairs.iter().zip(&self.receipts) {
                let outcome = match status {
                    Some(sample) => {
                        let report = repair.recheck(*sample);
                        match (&report.coverage.value, report.outcome, compact) {
                            (Coverage::Truncated, _, true) => {
                                locale.text("? Partial trace", "? 覆盖不足", "? 覆蓋不足")
                            }
                            (_, Outcome::Passed, true) => {
                                locale.text("✓ Passed", "✓ 通过", "✓ 通過")
                            }
                            (Coverage::Captured, _, true) => {
                                locale.text("× Failed", "× 失败", "× 失敗")
                            }
                            (Coverage::Truncated, _, false) => {
                                locale.text("? Coverage incomplete", "? 覆盖不足", "? 覆蓋不足")
                            }
                            (_, Outcome::Passed, false) => {
                                locale.text("✓ Sample passed", "✓ 样例通过", "✓ 樣例通過")
                            }
                            _ => locale.text("× Sample failed", "× 样例失败", "× 樣例失敗"),
                        }
                    }
                    None => locale.text("Awaiting recheck ◌", "待重验 ◌", "待重驗 ◌"),
                };
                let receipt = &repair.receipt;
                let port = format!(
                    "{}.in[{}]",
                    self.module_path(receipt.target.operator),
                    receipt.target.slot
                );
                let revision = receipt.graph_revision;
                let identity = if compact {
                    format!("v{revision}")
                } else {
                    format!("v{revision} · {}", receipt.snapshot.variant)
                };
                push(&mut out, &format!("{identity} · {port} · {outcome}"), width);
            }
            for _ in self.receipts.len()..4 {
                push(&mut out, "", width);
            }
            push(&mut out, &format!("l / {back_key} {back}"), width);
            return out;
        }
        if self.mode == DetailMode::Samples {
            push(&mut out, &divider.repeat(width.min(100)), width);
            push(&mut out, Action::Samples.label(locale), width);
            for sample in Action::RECHECK {
                let action = Action::Recheck(sample);
                push(
                    &mut out,
                    &format!(
                        "{}  {}",
                        action.key(self.preferences.preset),
                        action.label(locale)
                    ),
                    width,
                );
            }
            let keys = if transcript { "b" } else { "Esc / b" };
            push(&mut out, &format!("{keys} {back}"), width);
            return out;
        }
        if self.mode == DetailMode::Help {
            let entries: Vec<_> = Action::ALL
                .into_iter()
                .filter(|action| {
                    !matches!(action, Action::Help | Action::Recheck(_)) && self.available(*action)
                })
                .map(|action| {
                    let mut key = action.key(self.preferences.preset).to_string();
                    if action == Action::Preview {
                        key.push_str(" / Enter");
                    }
                    format!("{key}  {}", action.label(locale))
                })
                .collect();
            push(&mut out, &divider.repeat(width.min(100)), width);
            push(
                &mut out,
                locale.text("Actions in this pane", "当前面板操作", "目前面板操作"),
                width,
            );
            for group in entries.chunks(2) {
                push(&mut out, &group.join(" · "), width);
            }
            for _ in entries.len().div_ceil(2)..4 {
                push(&mut out, "", width);
            }
            push(&mut out, &format!("? / {back_key} {back}"), width);
            return out;
        }
        if self.mode == DetailMode::Settings {
            let style = match self.preferences.style {
                Style::Default => locale.text("Default", "默认", "預設"),
                Style::Compact => locale.text("Compact", "紧凑", "緊湊"),
                Style::Story => locale.text("Story", "叙事", "敘事"),
            };
            let preset = match self.preferences.preset {
                Preset::Default => locale.text("Default", "默认", "預設"),
                Preset::Shortcuts => locale.text("Shortcuts", "快捷键", "快速鍵"),
            };
            for line in [
                divider.repeat(width.min(100)),
                locale.text("Settings", "设置", "設定").into(),
                format!(
                    "1  {} · {style}",
                    locale.text("Style", "视觉风格", "視覺風格")
                ),
                format!(
                    "2  {} · {preset}",
                    locale.text("Controls", "操作预设", "操作預設")
                ),
                locale
                    .text("r  Restore defaults", "r  恢复默认", "r  恢復預設")
                    .into(),
                self.preference_notice(locale).into(),
                if transcript {
                    format!("b {back}")
                } else {
                    format!("Esc / b  {back}")
                },
            ] {
                push(&mut out, &line, width);
            }
            return out;
        }
        let report = &self.fixture.report;
        let call = &report.calls[self.selected];
        let snapshot_matches = self.fixture.snapshot.id == report.snapshot;
        let at_focus = report.focus().is_some_and(|focus| focus.id == call.id);
        let status = match report.outcome {
            Outcome::FailedAt(_) if at_focus => locale.text("Failed", "前向失败", "前向失敗"),
            Outcome::LastObserved(_) if at_focus => {
                locale.text("Last observed", "最后观测", "最後觀測")
            }
            Outcome::Passed => locale.text("Passed", "通过", "通過"),
            _ => locale.text("Observed call", "观测调用", "觀測呼叫"),
        };
        let label = |shape: Option<&Evidence<Shape>>| {
            shape.map_or_else(|| "?".into(), |shape| shape.value.label())
        };
        let origin = |origin| match origin {
            Origin::Observation => locale.text("observed", "观测", "觀測"),
            Origin::Declaration => locale.text("declared", "声明", "宣告"),
            Origin::Preview => locale.text("preview", "预览", "預覽"),
        };
        let missing = locale.text("unknown", "未知", "未知");
        let source = report.source.as_ref().filter(|_| at_focus).map_or_else(
            || missing.into(),
            |location| format!("{}:{}", location.value.0, location.value.1),
        );
        let focus = if focused {
            locale.text("Details", "详情焦点", "詳情焦點")
        } else {
            locale.text("Graph", "图焦点", "圖焦點")
        };
        let expected = call.expected.as_ref();
        if let DetailMode::Compare { side, .. } = self.mode {
            return self.render_comparison(side, locale, width, presentation);
        }
        if matches!(self.mode, DetailMode::Axes { .. }) {
            let (evidence, candidate) = self.axis_evidence();
            let title = if candidate.is_some() {
                locale.text(
                    "Proposed last-axis alignment",
                    "候选末轴对齐",
                    "候選末軸對齊",
                )
            } else {
                locale.text("Last-axis alignment", "末轴对齐", "末軸對齊")
            };
            let source = |shape: Option<&Evidence<Shape>>| {
                shape.map_or_else(
                    || "?".into(),
                    |shape| format!("{} ({})", shape.value.label(), origin(shape.origin)),
                )
            };
            let [left, right] = evidence.map(source);
            let [context, location] = if axes_visible {
                let evidence_report = match self.stage {
                    Stage::Rechecked { sample } => self.repair().recheck(sample),
                    _ => report,
                };
                let location = evidence_report.source.as_ref().filter(|_| {
                    evidence_report
                        .focus()
                        .is_some_and(|focus| focus.id == call.id)
                });
                [
                    format!("{} · {}", evidence_report.batch, evidence_report.id),
                    location.map_or_else(
                        || {
                            locale
                                .text("Source unavailable", "来源未知", "來源未知")
                                .into()
                        },
                        |entry| format!("{}:{}", entry.value.0, entry.value.1),
                    ),
                ]
            } else {
                let extents = evidence.map(|shape| shape.and_then(|shape| shape.value.last()));
                let bars = axis_bars(extents, width.saturating_sub(12).clamp(4, 24));
                std::array::from_fn(|index| {
                    format!(
                        "{}  {}",
                        axis_label(index, candidate, locale, false),
                        bars[index]
                    )
                })
            };
            for line in [
                divider.repeat(width.min(100)),
                title.into(),
                format!("{left} ↔ {right}"),
                format!("{} #{}", call.id.alias, call.id.invocation),
                context,
                location,
                format!("{back_key} / a  {back}"),
            ] {
                push(&mut out, &line, width);
            }
            return out;
        }
        let mut observed = if let Some(rejection) = &self.recheck_mismatch {
            format!("{} ≠ {}", rejection.expected, rejection.received)
        } else {
            format!(
                "{} {} · {} {}",
                locale.text("Observed", "实际", "實際"),
                label(call.input.as_ref()),
                locale.text("Expected", "预期", "預期"),
                label(expected),
            )
        };
        let (feedback, actions) = match self.stage {
            Stage::Evidence => {
                let actions = if transcript {
                    if self.available(Action::Preview) {
                        locale.text(
                            "j/k Calls · {preview} Preview",
                            "j/k 调用 · {preview} 预览",
                            "j/k 呼叫 · {preview} 預覽",
                        )
                    } else {
                        locale.text("j/k Calls", "j/k 调用", "j/k 呼叫")
                    }
                } else if self.available(Action::Preview) {
                    if width < 55 {
                        locale.text(
                            "↑↓ calls · {preview} Preview · Tab",
                            "↑↓ 调用 · {preview} 预览 · Tab",
                            "↑↓ 呼叫 · {preview} 預覽 · Tab",
                        )
                    } else {
                        locale.text(
                            "↑↓ calls · {preview} Preview · Tab graph · q quit",
                            "↑↓ 调用 · {preview} 预览 · Tab 图 · q 退出",
                            "↑↓ 呼叫 · {preview} 預覽 · Tab 圖 · q 離開",
                        )
                    }
                } else if snapshot_matches {
                    locale.text(
                        "↑↓ calls · Tab graph/details · q quit",
                        "↑↓ 切换调用 · Tab 图和详情 · q 退出",
                        "↑↓ 切換呼叫 · Tab 圖和詳情 · q 離開",
                    )
                } else {
                    locale.text("↑↓ calls", "↑↓ 切换调用", "↑↓ 切換呼叫")
                };
                let hint = if self.available(Action::Preview) {
                    call.incoming
                        .as_ref()
                        .map_or(missing, |edge| origin(edge.origin))
                } else if !snapshot_matches {
                    locale.text(
                        "Matching snapshot unavailable",
                        "缺少匹配的结构快照",
                        "缺少相符的結構快照",
                    )
                } else {
                    locale.text(
                        "Verified edit target unavailable",
                        "缺少可确认的编辑目标",
                        "缺少可確認的編輯目標",
                    )
                };
                (hint.into(), actions)
            }
            Stage::Preview(draft) => {
                let preview = &self.repair().preview;
                let weights = self.weight_impact(locale);
                let slot = preview.target.slot;
                let revision = draft.revision;
                if compact {
                    observed = match &preview.change {
                        Adjustment::Shape { before, after } => {
                            format!(
                                "v{revision} · in[{slot}] {} → {}",
                                before.label(),
                                after.value.label()
                            )
                        }
                        Adjustment::Source { .. } => {
                            format!(
                                "v{revision} · {}.in[{slot}]",
                                self.module_path(preview.target.operator)
                            )
                        }
                    };
                }
                let feedback = if let Some(error) = draft.error {
                    draft_rejection(error, revision, locale)
                } else {
                    match &preview.change {
                        _ if compact => weights,
                        Adjustment::Shape { before, after } => format!(
                            "v{revision} · in[{slot}] {} → {} · {weights}",
                            before.label(),
                            after.value.label()
                        ),
                        Adjustment::Source { .. } => {
                            format!("v{revision} · in[{slot}] · {weights}")
                        }
                    }
                };
                let actions = if matches!(preview.change, Adjustment::Source { .. }) {
                    locale.text(
                        "y Confirm · x Source · {back} cancel",
                        "y 确认本段 · x 来源 · {back} 取消",
                        "y 確認本段 · x 來源 · {back} 取消",
                    )
                } else {
                    locale.text(
                        "y Confirm · {back} cancel",
                        "y 确认本段 · {back} 取消",
                        "y 確認本段 · {back} 取消",
                    )
                };
                (feedback, actions)
            }
            Stage::Committed => {
                let receipt = &self.repair().receipt;
                (
                    format!(
                        "v{} · {} · {}",
                        receipt.graph_revision,
                        receipt.snapshot.variant,
                        locale.text("Awaiting recheck ◌", "待重验 ◌", "待重驗 ◌")
                    ),
                    locale.text(
                        "r Recheck samples · {pass} Passing example · {fail} Failing example",
                        "r 重验样例 · {pass} 通过样例 · {fail} 失败样例",
                        "r 重驗樣例 · {pass} 通過樣例 · {fail} 失敗樣例",
                    ),
                )
            }
            Stage::Rechecked { sample } => {
                let recheck = self.repair().recheck(sample);
                let target = self.rechecked_call(recheck);
                let result = if recheck.coverage.value == Coverage::Truncated {
                    locale.text("? Coverage incomplete", "? 重验覆盖不足", "? 重驗覆蓋不足")
                } else if recheck.outcome == Outcome::Passed {
                    locale.text("✓ Passed", "✓ 重验通过", "✓ 重驗通過")
                } else {
                    locale.text("× Still failing", "× 重验仍失败", "× 重驗仍失敗")
                };
                (
                    format!(
                        "{result} · {} → {} · {}",
                        label(target.and_then(|call| call.input.as_ref())),
                        label(target.and_then(|call| call.expected.as_ref())),
                        recheck.id
                    ),
                    locale.text(
                        "c Compare · r Recheck samples · {pass} Pass · {fail} Fail",
                        "c 对照 · r 重验样例 · {pass} 通过 · {fail} 失败",
                        "c 對照 · r 重驗樣例 · {pass} 通過 · {fail} 失敗",
                    ),
                )
            }
        };
        let key = |action: Action| action.key(self.preferences.preset).to_string();
        let actions = actions
            .replace("{back}", back_key)
            .replace("{preview}", &key(Action::Preview))
            .replace("{pass}", &key(Action::Recheck(RecheckSample::Passing)))
            .replace("{fail}", &key(Action::Recheck(RecheckSample::Failing)));
        #[cfg(unix)]
        let actions = if !transcript
            && self.preferences.preset == Preset::Shortcuts
            && matches!(self.stage, Stage::Evidence)
        {
            use super::Shortcut;
            actions
                .replace(
                    "↑↓",
                    &format!("{}/{}", Shortcut::Next.key(), Shortcut::Previous.key()),
                )
                .replace("Tab", &Shortcut::SwitchPane.key().to_string())
        } else {
            actions
        };
        let actions = if self.available(Action::Next) {
            let next = Action::Next.label(locale);
            format!("{} {next} · {actions}", key(Action::Next))
        } else {
            actions
        };
        let actions = if self.available(Action::Source) && !matches!(self.stage, Stage::Preview(_))
        {
            let label = locale.text("Source", "改接来源", "改接來源");
            format!("{} {label} · {actions}", key(Action::Source))
        } else {
            actions
        };
        let actions = if (!compact || transcript || !snapshot_matches)
            && self.available(Action::CallEvidence)
        {
            let evidence = Action::CallEvidence;
            format!("{actions} · {} {}", key(evidence), evidence.label(locale))
        } else {
            actions
        };
        let actions = if self.available(Action::Settings) {
            let settings = Action::Settings.label(locale);
            format!("{actions} · {} {settings}", key(Action::Settings))
        } else {
            actions
        };
        #[cfg(unix)]
        let actions = if !transcript && matches!(self.stage, Stage::Preview(_)) {
            let full = format!(
                "{} {}",
                char::from(DETAIL_EXPAND_KEY),
                locale.text("Full", "全文", "全文")
            );
            if compact {
                format!("{full} · {actions}")
            } else {
                format!("{actions} · {full}")
            }
        } else {
            actions
        };
        let feedback = if focused && matches!(self.stage, Stage::Preview(_)) {
            format!("{} {feedback}", appearance.focus())
        } else {
            feedback
        };
        let notice = self.preference_notice(locale);
        for line in [
            divider.repeat(width.min(100)),
            format!(
                "{} · {status} · {focus}",
                locale.text("Offline sample", "离线样例", "離線樣例")
            ),
            format!(
                "{} #{} · {source} · a {}",
                call.id.alias,
                call.id.invocation,
                locale.text("Axes", "对齐", "對齊")
            ),
            observed,
            if let Some(rejection) = &self.recheck_mismatch {
                rejection.mismatch.label(locale).into()
            } else if let Stage::Preview(draft) = self.stage {
                let preview = &self.repair().preview;
                let wire = draft.existing;
                let target = self.module_path(wire.to.operator);
                let outlet = |port: super::super::Outlet| {
                    if compact {
                        format!("{}[{}]", self.module_path(port.operator), port.slot)
                    } else {
                        self.output_label(port)
                    }
                };
                match preview.change {
                    Adjustment::Shape { .. } => self.wire_label(wire, compact),
                    Adjustment::Source {
                        ref after,
                        unavailable,
                    } => {
                        let source = if draft.unavailable {
                            unavailable
                        } else {
                            after.value
                        };
                        if compact {
                            format!("{} → {}", outlet(wire.from), outlet(source))
                        } else {
                            format!(
                                "{target}.in[{}] · {} → {}",
                                wire.to.slot,
                                outlet(wire.from),
                                outlet(source)
                            )
                        }
                    }
                }
            } else if matches!(self.stage, Stage::Committed) {
                let preview = &self.repair().preview;
                let target = self.module_path(preview.target.operator);
                let slot = preview.target.slot;
                let change = match &preview.change {
                    Adjustment::Shape { before, after } => {
                        format!("{} → {}", before.label(), after.value.label())
                    }
                    Adjustment::Source { after, .. } => format!(
                        "{} → {}",
                        self.output_label(preview.affected.from),
                        self.output_label(after.value)
                    ),
                };
                format!("{target}.in[{slot}] · {change}")
            } else {
                format!(
                    "{} · {}",
                    report.batch,
                    if report.coverage.value == Coverage::Truncated {
                        locale.text("truncated", "记录截断", "記錄截斷")
                    } else {
                        locale.text("trace captured", "已记录调用", "已記錄呼叫")
                    },
                )
            },
            if notice.is_empty() {
                feedback
            } else {
                format!("{feedback} · {notice}")
            },
            format!("? · {actions} · {}", self.welcome_hint(locale)),
        ] {
            push(&mut out, &line, width);
        }
        out
    }

    fn preference_notice(&self, locale: Locale) -> &'static str {
        match self.preference_error {
            Some(PreferenceNotice::Defaults) => locale.text(
                "Could not read settings. Using defaults.",
                "读取设置失败，使用默认值",
                "讀取設定失敗，使用預設值",
            ),
            Some(PreferenceNotice::Partial) => locale.text(
                "Invalid setting replaced by its default.",
                "无效选项已回退到默认值",
                "無效選項已回退到預設值",
            ),
            Some(PreferenceNotice::SaveFailed) => {
                locale.text("Could not save settings.", "保存设置失败", "儲存設定失敗")
            }
            None => "",
        }
    }

    fn render_call_evidence(
        &self,
        locale: Locale,
        width: usize,
        presentation: DetailsOutput,
        input: usize,
        tracing: bool,
        trace_unavailable: bool,
    ) -> String {
        let transcript = matches!(presentation, DetailsOutput::Transcript);
        let compact = !transcript && width < 55;
        let (_, call) = self.evidence_call();
        let mut actions = Vec::new();
        if !tracing && !call.other_inputs.is_empty() {
            let keys = if transcript || self.preferences.preset == Preset::Shortcuts {
                "j/k"
            } else {
                "↑↓"
            };
            actions.push(format!("{keys} {}", locale.text("Inputs", "输入", "輸入")));
        }
        if !tracing && self.evidence_input(input).wire.is_some() && !trace_unavailable {
            actions.push(format!(
                "{} {}",
                char::from(EVIDENCE_TRACE_KEY),
                locale.text(
                    if compact { "Source" } else { "Producer" },
                    "生产者",
                    "生產者"
                )
            ));
        }
        #[cfg(unix)]
        if !transcript && !tracing {
            let full = format!(
                "{} {}",
                char::from(DETAIL_EXPAND_KEY),
                locale.text("Full", "全文", "全文")
            );
            if compact {
                actions.insert(0, full);
            } else {
                actions.push(full);
            }
        }
        actions.push(format!(
            "{} {}",
            presentation.back_key(),
            locale.text("Back", "返回", "返回")
        ));
        let mut out = String::new();
        for row in self.call_evidence_lines(locale, compact, input, tracing, trace_unavailable) {
            presentation.push(&mut out, &row, width);
        }
        presentation.push(&mut out, &actions.join(" · "), width);
        out
    }

    fn call_evidence_lines(
        &self,
        locale: Locale,
        compact: bool,
        input: usize,
        tracing: bool,
        trace_unavailable: bool,
    ) -> [String; 6] {
        let (report, call) = self.evidence_call();
        let evidence = self.evidence_input(input);
        let shape = |evidence: Option<(&Shape, Origin)>| {
            evidence.map_or_else(
                || "?".into(),
                |(shape, origin)| {
                    let origin = match origin {
                        Origin::Observation => {
                            locale.text(if compact { "obs" } else { "observed" }, "观测", "觀測")
                        }
                        Origin::Declaration => {
                            locale.text(if compact { "decl" } else { "declared" }, "声明", "宣告")
                        }
                        Origin::Preview => locale.text("preview", "预览", "預覽"),
                    };
                    format!("{} ({origin})", shape.label())
                },
            )
        };
        let title = if tracing {
            locale.text("Source trace", "来源追踪", "來源追蹤")
        } else {
            Action::CallEvidence.label(locale)
        };
        let source = report
            .source
            .as_ref()
            .filter(|_| report.focus().is_some_and(|focus| focus.id == call.id))
            .map(|location| format!("{}:{}", location.value.0, location.value.1));
        let call_label = format!(
            "{} #{} · {}",
            call.id.alias,
            call.id.invocation,
            source.as_deref().unwrap_or("?")
        );
        let coverage = if report.coverage.value == Coverage::Truncated {
            locale.text("trace truncated", "记录截断", "記錄截斷")
        } else {
            locale.text("trace captured", "已记录调用", "已記錄呼叫")
        };
        let link = if trace_unavailable {
            unavailable_edge(locale).into()
        } else {
            evidence.wire.map_or_else(
                || locale.text("Source unknown", "来源未知", "來源未知").into(),
                |wire| self.wire_label(wire, compact),
            )
        };
        let (observed, expected, output, recorded) = if compact {
            (
                locale.text("IN", "输入", "輸入"),
                locale.text("Exp", "预期", "預期"),
                locale.text("OUT", "输出", "輸出"),
                locale.text("Inputs", "输入记录", "輸入記錄"),
            )
        } else {
            (
                locale.text("Input", "输入", "輸入"),
                locale.text("Expected", "预期", "預期"),
                locale.text("Output", "输出", "輸出"),
                locale.text("Input evidence", "输入记录", "輸入記錄"),
            )
        };
        let output_shape = shape(call.output.as_ref().map(|item| (&item.value, item.origin)));
        let input_count = format!("{recorded} {}/{}", input + 1, call.other_inputs.len() + 1);
        let output_details = if compact {
            format!("{input_count} · {output} {output_shape}")
        } else {
            format!("{output} {output_shape} · {input_count}")
        };
        [
            format!("{title} · {}", report.id),
            call_label,
            format!(
                "{observed} {} · {expected} {}",
                shape(evidence.actual),
                shape(evidence.expected)
            ),
            output_details,
            link,
            format!("{} · {coverage}", report.batch),
        ]
    }

    fn render_comparison(
        &self,
        side: Comparison,
        locale: Locale,
        width: usize,
        presentation: DetailsOutput,
    ) -> String {
        let push = |out: &mut String, line: &str, width| presentation.push(out, line, width);
        let mut out = String::new();
        let receipt = &self.repair().receipt;
        let screen = matches!(presentation, DetailsOutput::Screen { .. });
        let compact = screen && width < 55;
        let side_by_side = screen && width >= 96;
        let mut heading = format!(
            "{} · v{}",
            self.fixture.report.batch, receipt.graph_revision
        );
        if compact {
            heading.push_str(" · ");
            heading.push_str(receipt.snapshot.model.name);
        }
        push(&mut out, &heading, width);
        if side_by_side {
            let [Some(before), Some(after)] = [Comparison::Before, Comparison::After]
                .map(|phase| self.comparison_fields(phase, side, locale, false))
            else {
                return out;
            };
            let left = (width - 3) / 2;
            let right = width - 3 - left;
            for (before, after) in before.into_iter().zip(after) {
                let before = text::truncate(&before, left);
                let after = text::truncate(&after, right);
                let padding = left.saturating_sub(text::display_width(&before));
                push(
                    &mut out,
                    &format!("{before}{:padding$} │ {after}", ""),
                    width,
                );
            }
        } else if let Some(fields) = self.comparison_fields(side, side, locale, compact) {
            for field in fields {
                push(&mut out, &field, width);
            }
        }
        let controls = if side_by_side {
            locale.text(
                "Switch graph evidence · v Full · Esc receipt",
                "切换图中证据 · v 全文 · Esc 回执",
                "切換圖中證據 · v 全文 · Esc 回執",
            )
        } else if screen {
            locale.text(
                "Compare · v Full · Esc Back",
                "对照 · v 全文 · Esc 回执",
                "對照 · v 全文 · Esc 回執",
            )
        } else {
            locale.text(
                "Switch before/after · b receipt",
                "切换修复前后 · b 回执",
                "切換修復前後 · b 回執",
            )
        };
        let keys = if self.preferences.preset == Preset::Shortcuts {
            "j/k/c"
        } else if screen {
            "↑↓/c"
        } else {
            "c"
        };
        push(&mut out, &format!("{keys} {controls}"), width);
        out
    }

    fn comparison_fields(
        &self,
        side: Comparison,
        active: Comparison,
        locale: Locale,
        compact: bool,
    ) -> Option<[String; 5]> {
        let (report, call) = self.comparison_evidence(side)?;
        let marker = if active == side { "› " } else { "  " };
        let phase = match side {
            Comparison::Before => locale.text("Before", "修复前", "修復前"),
            Comparison::After => locale.text("After", "修复后", "修復後"),
        };
        let (outcome, coverage) = if report.coverage.value == Coverage::Truncated {
            (
                locale.text("? Coverage incomplete", "? 覆盖不足", "? 覆蓋不足"),
                if compact {
                    locale.text(
                        "Target captured · rest unknown",
                        "已记录目标 · 其余调用未知",
                        "已記錄目標 · 其餘呼叫未知",
                    )
                } else {
                    locale.text(
                        "Target captured · remaining trace unknown",
                        "已记录目标 · 其余调用未知",
                        "已記錄目標 · 其餘呼叫未知",
                    )
                },
            )
        } else {
            (
                match report.outcome {
                    Outcome::Passed => locale.text("✓ Sample passed", "✓ 样例通过", "✓ 樣例通過"),
                    Outcome::FailedAt(id) if id == call.id => {
                        locale.text("× Target failed", "× 目标失败", "× 目標失敗")
                    }
                    _ => locale.text("• Target observed", "• 观测到目标", "• 觀測到目標"),
                },
                locale.text("Target captured", "已记录目标", "已記錄目標"),
            )
        };
        let shape = |evidence: Option<&Evidence<Shape>>| {
            evidence.map_or_else(|| "?".into(), |value| value.value.label())
        };
        let link = call
            .incoming
            .as_ref()
            .filter(|edge| edge.origin == Origin::Observation)
            .map_or_else(
                || {
                    locale
                        .text("Source unavailable", "来源未知", "來源未知")
                        .into()
                },
                |edge| self.wire_label(edge.value, compact),
            );
        let source = report
            .source
            .as_ref()
            .filter(|_| report.focus().is_some_and(|focus| focus.id == call.id));
        let detail = if let Some(source) = source {
            format!("{coverage} · {}:{}", source.value.0, source.value.1)
        } else if report.coverage.value != Coverage::Truncated {
            format!(
                "{coverage} · {} {}",
                report.calls.len(),
                locale.text("recorded calls", "次调用", "次呼叫")
            )
        } else {
            coverage.to_string()
        };
        Some([
            format!("{marker}{phase} · {outcome}"),
            if compact {
                format!("{} · {}", report.snapshot.variant, report.id)
            } else {
                format!(
                    "{} / {} · {}",
                    report.snapshot.model.name, report.snapshot.variant, report.id
                )
            },
            format!(
                "{} #{} · {} → {}",
                call.id.alias,
                call.id.invocation,
                shape(call.input.as_ref()),
                shape(call.expected.as_ref())
            ),
            link,
            detail,
        ])
    }

    fn wire_label(&self, wire: super::Wire, compact: bool) -> String {
        let sink = self.module_path(wire.to.operator);
        let input = wire.to.slot;
        if compact {
            let source = self.module_path(wire.from.operator);
            let output = wire.from.slot;
            format!("{source}[{output}] → {sink}.in[{input}]")
        } else {
            let source = self.output_label(wire.from);
            format!("{source} → {sink}.in[{input}]")
        }
    }
}

#[cfg(unix)]
fn detail_page_rows(height: usize) -> usize {
    height.saturating_sub(2).max(1)
}

fn draft_rejection(error: DraftError, revision: u64, locale: Locale) -> String {
    match error {
        DraftError::SourceUnavailable => {
            let message = locale.text(
                "Rejected · source port unavailable",
                "提交被拒绝 · 来源端口不存在",
                "提交遭拒絕 · 來源埠不存在",
            );
            format!("{message} · v{revision}")
        }
        DraftError::AlreadyConnected => locale
            .text(
                "Source already connected",
                "当前来源已连接",
                "目前來源已連接",
            )
            .into(),
        DraftError::Stale => locale
            .text(
                "Rejected · prior graph retained",
                "提交被拒绝 · 保留上一版",
                "提交遭拒絕 · 保留上一版",
            )
            .into(),
    }
}

fn unavailable_edge(locale: Locale) -> &'static str {
    locale.text(
        "Edge unavailable in current graph",
        "当前图中没有这条观测连线",
        "目前圖中沒有這條觀測連線",
    )
}

pub(in crate::graph) fn axis_label(
    index: usize,
    candidate: Option<usize>,
    locale: Locale,
    short: bool,
) -> &'static str {
    if candidate == Some(index) {
        locale.text(if short { "New" } else { "Candidate" }, "候选", "候選")
    } else if index == 0 {
        locale.text(if short { "Obs" } else { "Observed" }, "实际", "實際")
    } else {
        locale.text(if short { "Exp" } else { "Expected" }, "预期", "預期")
    }
}

pub(in crate::graph) fn axis_bars(axes: [Option<u64>; 2], width: usize) -> [String; 2] {
    let largest = axes
        .iter()
        .flatten()
        .copied()
        .max()
        .unwrap_or_default()
        .max(1);
    axes.map(|axis| match axis {
        Some(size) => {
            let cells = ((u128::from(size) * width as u128 + u128::from(largest) / 2)
                / u128::from(largest)) as usize;
            format!("├{}┤", "━".repeat(cells))
        }
        None => format!("┆{}?", "┄".repeat(width)),
    })
}
