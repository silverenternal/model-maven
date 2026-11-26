//! Local, call-specific evidence. A module-level edge is never treated as a call link.
use super::evidence::ObservedConsumer;
pub(super) use super::evidence::{Input, attach_observed_matches, observed_inputs};
use super::{ModelView, Shape, Wire, push, shape_label};
use crate::terminal::locale::Locale;
use model_maven_tui::diagnosis::Origin;

pub(super) fn port_label(path: &str, direction: &str, slot: u32, width: usize) -> String {
    let suffix = format!(".{direction}[{slot}]");
    let budget = width.saturating_sub(crate::terminal::text::display_width(&suffix));
    if crate::terminal::text::display_width(path) <= budget {
        return format!("{path}{suffix}");
    }
    let mut start = path.len();
    for (index, _) in path.char_indices().rev() {
        if crate::terminal::text::display_width(&path[index..]) + 1 > budget {
            break;
        }
        start = index;
    }
    format!("…{}{suffix}", &path[start..])
}

fn metadata_label(metadata: &str, locale: Locale) -> String {
    if let Some(dtype) = metadata.strip_prefix("tensor · ") {
        return format!("{} · {dtype}", locale.text("tensor", "张量", "張量"));
    }
    if let Some(count) = metadata
        .strip_prefix("truncated · ")
        .and_then(|part| part.strip_suffix(" omitted"))
    {
        return format!(
            "{} · {count} {}",
            locale.text("truncated", "已截断", "已截斷"),
            locale.text("omitted", "项省略", "項省略")
        );
    }
    match metadata {
        "truncated · nesting limit" => locale.text(
            "truncated · nesting limit",
            "已截断 · 层级上限",
            "已截斷 · 層級上限",
        ),
        "truncated · display limit" => locale.text(
            "truncated · display limit",
            "已截断 · 显示上限",
            "已截斷 · 顯示上限",
        ),
        "empty dict" => locale.text("empty dict", "空字典", "空字典"),
        "empty tuple" => locale.text("empty tuple", "空元组", "空元組"),
        "empty list" => locale.text("empty list", "空列表", "空清單"),
        "number" => locale.text("number", "数值", "數值"),
        "string" => locale.text("string", "字符串", "字串"),
        "bool" => locale.text("bool", "布尔值", "布林值"),
        "null" => locale.text("null", "空值", "空值"),
        "truncated" => locale.text("truncated", "已截断", "已截斷"),
        "opaque" => locale.text("opaque", "不透明对象", "不透明物件"),
        "unrecognized" => locale.text("unrecognized", "无法识别", "無法辨識"),
        other => other,
    }
    .to_owned()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Stop {
    Failure,
    LastObserved,
    Passed,
    Call,
}

#[derive(Clone)]
struct Consumers {
    entries: Vec<ObservedConsumer>,
    selected: usize,
    limited: bool,
    diagram: bool,
}

// A saved model parameter is a declaration, not a tensor provenance edge.
// Associate declarations only where the report uniquely maps a seam to this
// registration path; a model's self-report is not proof of runtime flow.
fn declared_routes(
    report: &serde_json::Value,
    alias: &str,
    ambiguous: bool,
) -> Vec<(String, String, String)> {
    if ambiguous {
        return Vec::new();
    }
    let origin = &report["data"]["origin"];
    let mut result = Vec::new();
    let self_report = &origin["declared_routes"];
    if self_report["scope"] == "model_self_reported_at_construction"
        && self_report["status"] == "available"
    {
        for binding in self_report["routes"]
            .as_array()
            .into_iter()
            .flatten()
            .take(32)
        {
            let Some(paths) = binding["registered_paths"].as_array() else {
                continue;
            };
            if paths.len() != 1 || paths[0].as_str() != Some(alias) {
                continue;
            }
            let (Some(slot), Some(input)) = (binding["slot"].as_str(), binding["input"].as_str())
            else {
                continue;
            };
            if slot.len() <= 128 && input.len() <= 128 {
                result.push((slot.to_owned(), input.to_owned(), "model".into()));
            }
        }
    }
    if let (Some(routes), Some(bindings)) = (
        origin["params"]["input_routes"].as_object(),
        origin["injection_evidence"].as_array(),
    ) {
        result.extend(
            bindings
                .iter()
                .filter_map(|binding| {
                    let paths = binding["registered_paths"].as_array()?;
                    if paths.len() != 1 || paths[0].as_str() != Some(alias) {
                        return None;
                    }
                    let slot = binding["slot"].as_str()?;
                    let input = routes.get(slot)?.as_str()?;
                    if slot.len() > 128 || input.len() > 128 {
                        return None;
                    }
                    Some((slot.to_owned(), input.to_owned(), "config".into()))
                })
                .take(32),
        );
    }
    result.truncate(32);
    result
}

pub(super) fn matching_input_indices(inputs: &[Input], query: &str) -> (Vec<usize>, bool) {
    fn normalize(value: &str) -> String {
        value
            .to_lowercase()
            .replace('×', "x")
            .replace(',', "x")
            .replace(['[', ']', ' '], "")
    }
    let query = normalize(query.trim());
    let incomplete = inputs.iter().any(|input| {
        input
            .metadata
            .as_deref()
            .is_some_and(|metadata| metadata.starts_with("truncated"))
    });
    if query.is_empty() {
        return (Vec::new(), incomplete);
    }
    let matches = inputs
        .iter()
        .enumerate()
        .filter(|(_, input)| {
            if input
                .metadata
                .as_deref()
                .is_some_and(|metadata| metadata.starts_with("truncated"))
            {
                return false;
            }
            input
                .label
                .as_deref()
                .is_some_and(|value| normalize(value).contains(&query))
                || input
                    .metadata
                    .as_deref()
                    .is_some_and(|value| normalize(value).contains(&query))
                || input
                    .actual
                    .as_ref()
                    .is_some_and(|(shape, _)| normalize(&shape.label()).contains(&query))
        })
        .map(|(index, _)| index)
        .collect();
    (matches, incomplete)
}

pub(super) fn unique_input_match(inputs: &[Input], query: &str) -> Option<usize> {
    let (matches, incomplete) = matching_input_indices(inputs, query);
    (!incomplete && matches.len() == 1).then(|| matches[0])
}

#[derive(Clone)]
pub(super) struct Path {
    pub alias: String,
    pub invocation: Option<u64>,
    pub invocation_id: Option<String>,
    pub report: Option<String>,
    pub stop: Stop,
    reused: bool,
    ambiguous_registration: bool,
    declared_routes: Vec<(String, String, String)>,
    pub inputs: Vec<Input>,
    selected: usize,
    selected_matches: Vec<usize>,
    candidate_hop: bool,
    navigation_limit: bool,
    consumers: Option<Consumers>,
    report_open: bool,
    report_offset: usize,
    report_match: Option<String>,
}

impl Path {
    pub fn new(
        alias: String,
        invocation: Option<u64>,
        report: Option<String>,
        stop: Stop,
        inputs: Vec<Input>,
    ) -> Self {
        let selected_matches = vec![0; inputs.len()];
        Self {
            alias,
            invocation,
            invocation_id: None,
            report,
            stop,
            reused: false,
            ambiguous_registration: false,
            declared_routes: Vec::new(),
            inputs,
            selected: 0,
            selected_matches,
            candidate_hop: false,
            navigation_limit: false,
            consumers: None,
            report_open: false,
            report_offset: 0,
            report_match: None,
        }
    }

    pub fn live(view: &ModelView<'_>) -> Option<Self> {
        let evidence = view.live_evidence.as_ref()?;
        let inputs = evidence.inputs.clone();
        let mut path = Self::new(
            evidence.path.clone(),
            evidence.invocation,
            None,
            if evidence.last_observed {
                Stop::LastObserved
            } else {
                Stop::Call
            },
            if inputs.is_empty() {
                vec![Input {
                    wire: None,
                    actual: evidence
                        .input
                        .clone()
                        .map(|shape| (shape, Origin::Observation)),
                    expected: evidence
                        .expected
                        .clone()
                        .map(|shape| (shape, Origin::Declaration)),
                    label: None,
                    metadata: None,
                    matches: Vec::new(),
                }]
            } else {
                inputs
            },
        );
        path.invocation_id = evidence.invocation_id.clone();
        path.reused = evidence.reused;
        path.ambiguous_registration = evidence.ambiguous_registration;
        if !evidence.ambiguous_registration {
            path.declared_routes = evidence
                .route_cache
                .get_or_init(|| {
                    if !evidence.raw_report.contains("\"input_routes\"")
                        && !evidence.raw_report.contains("\"declared_routes\"")
                    {
                        return Vec::new();
                    }
                    serde_json::from_str(&evidence.raw_report)
                        .ok()
                        .map(|report| {
                            declared_routes(
                                &report,
                                &evidence.path,
                                evidence.ambiguous_registration,
                            )
                        })
                        .unwrap_or_default()
                })
                .clone();
        }
        if let Some(index) = evidence.selected_input {
            path.select_input(index);
        }
        Some(path)
    }

    pub fn movement(&mut self, forward: bool) {
        if !self.inputs.is_empty() {
            self.selected = if forward {
                (self.selected + 1).min(self.inputs.len() - 1)
            } else {
                self.selected.saturating_sub(1)
            };
        }
    }

    pub fn wire(&self) -> Option<Wire> {
        self.inputs.get(self.selected).and_then(|input| input.wire)
    }

    pub fn move_match(&mut self, forward: bool) {
        let Some(input) = self.inputs.get(self.selected) else {
            return;
        };
        let Some(selected) = self.selected_matches.get_mut(self.selected) else {
            return;
        };
        if input.wire.is_none() && !input.matches.is_empty() {
            *selected = if forward {
                (*selected + 1).min(input.matches.len() - 1)
            } else {
                selected.saturating_sub(1)
            };
        }
    }

    /// Open an explicitly matched earlier call without treating it as a producer.
    pub fn follow_match(&self, view: &ModelView<'_>) -> Option<Self> {
        let input = self.inputs.get(self.selected)?;
        if input.wire.is_some() {
            return None;
        }
        let matched = input
            .matches
            .get(*self.selected_matches.get(self.selected)?)?;
        Self::open_recorded_call(
            view,
            &matched.invocation_id,
            &matched.alias,
            matched.ambiguous_alias,
            matched.invocation,
            None,
        )
    }

    fn open_recorded_call(
        view: &ModelView<'_>,
        id: &str,
        alias: &str,
        ambiguous: bool,
        invocation: Option<u64>,
        input_index: Option<usize>,
    ) -> Option<Self> {
        let evidence = view.live_evidence.as_ref()?;
        let report: serde_json::Value = serde_json::from_str(&evidence.raw_report).ok()?;
        let data = &report["data"];
        let forward = data["forward_id"].as_str()?;
        let mut calls = data["calls"].as_array()?.iter().filter(|call| {
            call["invocation_id"].as_str() == Some(id)
                && call["forward_id"].as_str() == Some(forward)
        });
        let first = calls.next();
        if calls.next().is_some() {
            return None;
        }
        let last = &data["failure_location"]["last_observed_module_call"];
        let call = first.or_else(|| {
            (last["invocation_id"].as_str() == Some(id)
                && last["forward_id"].as_str() == Some(forward))
            .then_some(last)
        })?;
        let expected = (!ambiguous)
            .then(|| {
                view.layers
                    .iter()
                    .find(|layer| layer.path == alias && layer.kind == "Linear")
            })
            .flatten()
            .and_then(|layer| layer.data["editable_params"]["in_features"].as_u64())
            .map(Shape::Features);
        let mut inputs = observed_inputs(&call["inputs"], expected.as_ref());
        attach_observed_matches(&mut inputs, call, &report);
        let mut path = Self::new(alias.to_owned(), invocation, None, Stop::Call, inputs);
        path.invocation_id = Some(id.to_owned());
        path.ambiguous_registration = ambiguous;
        path.declared_routes = declared_routes(&report, alias, ambiguous);
        path.reused = evidence.reused;
        path.candidate_hop = true;
        if let Some(index) = input_index {
            path.select_input(index);
        }
        Some(path)
    }

    pub fn consumers_open(&self) -> bool {
        self.consumers.is_some()
    }

    pub fn graph_target(&self) -> Option<&str> {
        if let Some(consumers) = &self.consumers
            && let Some(selected) = consumers.entries.get(consumers.selected)
        {
            return (!selected.ambiguous_alias).then_some(selected.alias.as_str());
        }
        (!self.ambiguous_registration).then_some(self.alias.as_str())
    }

    pub fn graph_projection(&self, view: &ModelView<'_>) -> Option<super::ObservedProjection> {
        let consumers = self.consumers.as_ref()?;
        let mut counts = std::collections::HashMap::new();
        let mut unmapped = 0;
        for consumer in &consumers.entries {
            let mut layers = view
                .layers
                .iter()
                .filter(|layer| layer.path == consumer.alias);
            let layer = layers.next();
            if consumer.ambiguous_alias || layer.is_none() || layers.next().is_some() {
                unmapped += 1;
            } else if let Some(layer) = layer {
                *counts.entry(layer.id).or_insert(0) += 1;
            }
        }
        Some(super::ObservedProjection {
            source: (!self.ambiguous_registration).then(|| self.alias.clone()),
            counts,
            hidden: false,
            limited: consumers.limited,
            unmapped,
        })
    }

    pub fn close_consumers(&mut self) {
        self.consumers = None;
    }

    pub fn show_consumers(&mut self, view: &ModelView<'_>) {
        let (entries, limited) = self
            .invocation_id
            .as_deref()
            .zip(view.live_evidence.as_ref())
            .and_then(|(id, evidence)| {
                serde_json::from_str::<serde_json::Value>(&evidence.raw_report)
                    .ok()
                    .map(|report| super::evidence::observed_consumers(id, &report))
            })
            .unwrap_or_default();
        self.consumers = Some(Consumers {
            entries,
            selected: 0,
            limited,
            diagram: true,
        });
    }

    pub fn move_consumer(&mut self, forward: bool, page: bool, height: usize) {
        if let Some(consumers) = &mut self.consumers {
            let step = if page {
                height.saturating_sub(6).max(1)
            } else {
                1
            };
            consumers.selected = if forward {
                consumers
                    .selected
                    .saturating_add(step)
                    .min(consumers.entries.len().saturating_sub(1))
            } else {
                consumers.selected.saturating_sub(step)
            };
        }
    }

    pub fn toggle_consumer_layout(&mut self) {
        if let Some(consumers) = &mut self.consumers {
            consumers.diagram = !consumers.diagram;
        }
    }

    pub fn consumer_at(
        &self,
        column: usize,
        row: usize,
        width: usize,
        height: usize,
    ) -> Option<usize> {
        let consumers = self.consumers.as_ref()?;
        if column >= width || row >= height {
            return None;
        }
        let rows = height.saturating_sub(6).max(1);
        let offset = row.checked_sub(3)?;
        if offset >= rows {
            return None;
        }
        let index = (consumers.selected / rows) * rows + offset;
        (index < consumers.entries.len()).then_some(index)
    }

    pub fn select_consumer(&mut self, index: usize) {
        if let Some(consumers) = &mut self.consumers
            && index < consumers.entries.len()
        {
            consumers.selected = index;
        }
    }

    pub fn open_consumer(&self, view: &ModelView<'_>) -> Option<Self> {
        let consumers = self.consumers.as_ref()?;
        let target = consumers.entries.get(consumers.selected)?;
        Self::open_recorded_call(
            view,
            &target.invocation_id,
            &target.alias,
            target.ambiguous_alias,
            target.invocation,
            Some(target.input_index),
        )
    }

    pub fn mark_navigation_limit(&mut self) {
        self.navigation_limit = true;
    }

    pub fn select_input(&mut self, index: usize) {
        self.selected = index.min(self.inputs.len().saturating_sub(1));
    }

    pub fn selected_input(&self) -> usize {
        self.selected
    }

    pub fn locate_report(
        &mut self,
        view: &ModelView<'_>,
        needle: &str,
        width: usize,
        height: usize,
        locale: Locale,
    ) {
        if view
            .live_evidence
            .as_ref()
            .is_none_or(|evidence| evidence.raw_report.is_empty())
        {
            return;
        }
        self.report_open = true;
        self.report_match = Some(needle.to_owned());
        let lines = self.report_lines(view, None, width, locale);
        let joined = lines.concat();
        let row = joined
            .find(needle)
            .map(|position| {
                let mut length = 0;
                lines
                    .iter()
                    .position(|line| {
                        length += line.len();
                        length > position
                    })
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        self.report_offset = row
            .saturating_sub(height.saturating_sub(2) / 3)
            .min(lines.len().saturating_sub(height.saturating_sub(2).max(1)));
    }

    pub fn report_open(&self) -> bool {
        self.report_open
    }

    pub fn toggle_report(
        &mut self,
        view: &ModelView<'_>,
        offline: Option<&super::offline::Offline>,
    ) {
        if self.report_open
            || offline.is_some()
            || view
                .live_evidence
                .as_ref()
                .is_some_and(|evidence| !evidence.raw_report.is_empty())
        {
            self.report_open = !self.report_open;
            self.report_offset = 0;
            self.report_match = None;
        }
    }

    fn report_lines(
        &self,
        view: &ModelView<'_>,
        offline: Option<&super::offline::Offline>,
        width: usize,
        locale: Locale,
    ) -> Vec<String> {
        let text = if let Some(offline) = offline {
            offline.failure_details(self.selected, locale, width)
        } else {
            view.live_evidence
                .as_ref()
                .map_or("", |evidence| evidence.raw_report.as_str())
                .to_owned()
        };
        crate::terminal::text::wrap(&text, width)
    }

    pub fn scroll_report(
        &mut self,
        view: &ModelView<'_>,
        offline: Option<&super::offline::Offline>,
        width: usize,
        height: usize,
        locale: Locale,
        forward: bool,
        page: bool,
    ) {
        let rows = height.saturating_sub(2).max(1);
        let max = self
            .report_lines(view, offline, width, locale)
            .len()
            .saturating_sub(rows);
        let step = if page { rows } else { 1 };
        self.report_offset = if forward {
            self.report_offset.saturating_add(step).min(max)
        } else {
            self.report_offset.saturating_sub(step)
        };
    }

    pub fn render(
        &self,
        view: &ModelView<'_>,
        width: usize,
        height: usize,
        locale: Locale,
        offline: Option<&super::offline::Offline>,
    ) -> String {
        let mut out = String::new();
        if self.report_open {
            let lines = self.report_lines(view, offline, width, locale);
            let rows = height.saturating_sub(2).max(1);
            let start = self.report_offset.min(lines.len().saturating_sub(rows));
            let end = (start + rows).min(lines.len());
            let title = if offline.is_some() {
                locale.text(
                    "Recorded call evidence",
                    "已记录的调用证据",
                    "已記錄的呼叫證據",
                )
            } else {
                locale.text("Original CLI report", "CLI 报告原文", "CLI 報告原文")
            };
            let matched = self
                .report_match
                .as_ref()
                .map_or_else(String::new, |code| format!(" · {code}"));
            push(
                &mut out,
                &format!(
                    "{title}{matched} · {}–{end}/{}",
                    if lines.is_empty() { 0 } else { start + 1 },
                    lines.len()
                ),
                width,
            );
            for line in lines.iter().skip(start).take(rows) {
                push(&mut out, line, width);
            }
            push(
                &mut out,
                locale.text(
                    "↑↓/Pg scroll · v/Esc back",
                    "↑↓/Pg 滚动 · v/Esc 返回",
                    "↑↓/Pg 捲動 · v/Esc 返回",
                ),
                width,
            );
            return out;
        }
        if let Some(consumers) = &self.consumers {
            push(
                &mut out,
                locale.text(
                    "Later input matches · partial",
                    "较晚输入匹配 · 部分观测",
                    "較晚輸入匹配 · 部分觀測",
                ),
                width,
            );
            push(
                &mut out,
                &format!(
                    "{}{} · {}",
                    if consumers.diagram { "◉ " } else { "" },
                    self.alias,
                    self.invocation_id.as_deref().unwrap_or("?")
                ),
                width,
            );
            push(
                &mut out,
                &format!(
                    "{} {}{}",
                    consumers.entries.len(),
                    locale.text("recorded matches", "条已记录匹配", "筆已記錄匹配"),
                    if consumers.limited {
                        locale.text(" · limit", " · 达上限", " · 達上限")
                    } else {
                        ""
                    }
                ),
                width,
            );
            let rows = height.saturating_sub(6).max(1);
            let start = (consumers.selected / rows) * rows;
            if consumers.entries.is_empty() {
                push(
                    &mut out,
                    locale.text(
                        "No verified matches · partial",
                        "没有已核对的匹配 · 部分观测",
                        "沒有已核對的匹配 · 部分觀測",
                    ),
                    width,
                );
            }
            for (index, entry) in consumers.entries.iter().enumerate().skip(start).take(rows) {
                push(
                    &mut out,
                    &format!(
                        "{}{} {}/{} {}{}{}",
                        if index == consumers.selected {
                            ">"
                        } else {
                            " "
                        },
                        if consumers.diagram {
                            if index + 1 == consumers.entries.len() {
                                "└╌"
                            } else {
                                "├╌"
                            }
                        } else {
                            "  "
                        },
                        index + 1,
                        consumers.entries.len(),
                        if entry.ambiguous_alias { "alias? " } else { "" },
                        entry.alias,
                        entry.invocation.map_or(String::new(), |n| format!(" #{n}"))
                    ),
                    width,
                );
            }
            if let Some(entry) = consumers.entries.get(consumers.selected) {
                push(
                    &mut out,
                    &format!("{} · {}", entry.output_path, entry.input_path),
                    width,
                );
                push(
                    &mut out,
                    locale.text(
                        "Same object/version · not proven flow",
                        "同对象/版本 · 不证明数据流",
                        "同物件/版本 · 不證明資料流",
                    ),
                    width,
                );
            }
            push(
                &mut out,
                locale.text(
                    "↑↓/Pg · g layout · ↵ open · b/c",
                    "↑↓/Pg · g 布局 · ↵ 打开 · b/c",
                    "↑↓/Pg · g 佈局 · ↵ 開啟 · b/c",
                ),
                width,
            );
            return out;
        }
        let stop = match self.stop {
            Stop::Failure => locale.text("Failure", "失败位置", "失敗位置"),
            Stop::LastObserved => locale.text(
                "Last observed · failure location unknown",
                "最后观测 · 失败位置未知",
                "最後觀測 · 失敗位置未知",
            ),
            Stop::Passed => locale.text("Recheck passed", "重验通过", "重驗通過"),
            Stop::Call => locale.text("Observed call", "观测调用", "觀測呼叫"),
        };
        push(
            &mut out,
            if self.candidate_hop {
                locale.text(
                    "Candidate call · partial match",
                    "候选调用 · 部分匹配",
                    "候選呼叫 · 部分匹配",
                )
            } else if self.stop == Stop::Call {
                locale.text(
                    "Call path · call evidence",
                    "调用路径 · 调用证据",
                    "呼叫路徑 · 呼叫證據",
                )
            } else {
                locale.text(
                    "Failure path · call evidence",
                    "故障路径 · 调用证据",
                    "故障路徑 · 呼叫證據",
                )
            },
            width,
        );
        if self.reused {
            push(
                &mut out,
                locale.text(
                    "Reused · no new forward",
                    "复用 · 本次未前向",
                    "重用 · 本次未前向",
                ),
                width,
            );
        }
        if self.ambiguous_registration {
            push(
                &mut out,
                locale.text(
                    "Shared call · registration unknown",
                    "共享调用 · 注册路径未知",
                    "共享呼叫 · 註冊路徑未知",
                ),
                width,
            );
        }
        push(
            &mut out,
            &format!(
                "{stop} · {}{}",
                self.alias,
                self.invocation.map_or(String::new(), |n| format!(" #{n}"))
            ),
            width,
        );
        let show_route = height >= 20 && !self.declared_routes.is_empty();
        if let Some(report) = self.report.as_ref().filter(|_| height >= 14 && !show_route) {
            push(
                &mut out,
                &format!("{} {report}", locale.text("Report", "报告", "報告")),
                width,
            );
        }
        if let Some(id) = self
            .invocation_id
            .as_ref()
            .filter(|_| height >= 16 && !show_route)
        {
            push(
                &mut out,
                &format!("{} {id}", locale.text("Call ID", "调用 ID", "呼叫 ID")),
                width,
            );
        }
        if show_route {
            let (slot, input, source) = &self.declared_routes[0];
            push(
                &mut out,
                if source == "model" {
                    locale.text(
                        "Model-declared route · not observed flow",
                        "模型自述接线 · 非观测数据流",
                        "模型自述接線 · 非觀測資料流",
                    )
                } else {
                    locale.text(
                        "Config-declared route · not observed flow",
                        "配置声明接线 · 非观测数据流",
                        "設定宣告接線 · 非觀測資料流",
                    )
                },
                width,
            );
            push(
                &mut out,
                &format!(
                    "{} → {}{}",
                    crate::terminal::text::plain_text(input),
                    crate::terminal::text::plain_text(slot),
                    if self.declared_routes.len() > 1 {
                        " …"
                    } else {
                        ""
                    }
                ),
                width,
            );
        }
        if let Some(input) = self.inputs.get(self.selected) {
            push(
                &mut out,
                &format!(
                    "{} {}/{}",
                    locale.text("Input", "输入", "輸入"),
                    self.selected + 1,
                    self.inputs.len()
                ),
                width,
            );
            if let Some(label) = &input.label {
                push(
                    &mut out,
                    &format!(
                        "{label} · {}",
                        metadata_label(input.metadata.as_deref().unwrap_or("?"), locale)
                    ),
                    width,
                );
            }
            if let Some(wire) = input.wire {
                let from = view
                    .document
                    .operator(wire.from.operator)
                    .map_or("?", |operator| operator.path.as_str());
                let to = view
                    .document
                    .operator(wire.to.operator)
                    .map_or("?", |operator| operator.path.as_str());
                push(
                    &mut out,
                    &format!(
                        "  {}",
                        port_label(from, "out", wire.from.slot, width.saturating_sub(2))
                    ),
                    width,
                );
                if height >= 12 {
                    push(&mut out, "       │", width);
                }
                push(
                    &mut out,
                    &format!(
                        "       ↓ {}",
                        port_label(to, "in", wire.to.slot, width.saturating_sub(9))
                    ),
                    width,
                );
                if view.document.source(wire.to) != Some(wire.from) {
                    push(
                        &mut out,
                        locale.text(
                            "  Call-level source; overview differs or has no edge",
                            "  调用级来源；总览连线不同或缺失",
                            "  呼叫級來源；總覽連線不同或缺失",
                        ),
                        width,
                    );
                } else {
                    push(
                        &mut out,
                        locale.text(
                            "  Observed source · earlier call unknown",
                            "  已观测来源 · 上游调用未知",
                            "  已觀測來源 · 上游呼叫未知",
                        ),
                        width,
                    );
                }
            } else if let Some(matched) = input.matches.get(self.selected_matches[self.selected]) {
                push(
                    &mut out,
                    &format!(
                        "  {} {}",
                        input.matches.len(),
                        locale.text(
                            "object matches · partial",
                            "个对象匹配 · 部分观测",
                            "個物件匹配 · 部分觀測",
                        )
                    ),
                    width,
                );
                push(
                    &mut out,
                    &format!(
                        "  {}/{} ←→ {}{}{} · {}",
                        self.selected_matches[self.selected] + 1,
                        input.matches.len(),
                        if matched.ambiguous_alias {
                            locale.text("alias? ", "别名? ", "別名? ")
                        } else {
                            ""
                        },
                        matched.alias,
                        matched
                            .invocation
                            .map_or(String::new(), |index| format!(" #{index}")),
                        matched.output_path,
                    ),
                    width,
                );
                if height >= 16 {
                    push(
                        &mut out,
                        locale.text(
                            "  Partial · not proof of provenance",
                            "  部分观测 · 未证明完整来源",
                            "  部分觀測 · 未證明完整來源",
                        ),
                        width,
                    );
                }
            } else {
                push(
                    &mut out,
                    locale.text(
                        "  ? Source not recorded · path stops here",
                        "  ? 未记录来源 · 路径止于此处",
                        "  ? 未記錄來源 · 路徑止於此處",
                    ),
                    width,
                );
            }
            let origin = |origin| match origin {
                Origin::Observation => locale.text("observed", "观测", "觀測"),
                Origin::Declaration => locale.text("declared", "声明", "宣告"),
                Origin::Preview => locale.text("preview", "预览", "預覽"),
            };
            let shape = |value: &Option<(Shape, Origin)>| {
                value.as_ref().map_or_else(
                    || "?".into(),
                    |(s, o)| format!("{} ({})", shape_label(Some(s)), origin(*o)),
                )
            };
            push(
                &mut out,
                &format!(
                    "1 {} {}",
                    locale.text("Actual", "实际", "實際"),
                    shape(&input.actual)
                ),
                width,
            );
            push(
                &mut out,
                &format!(
                    "2 {} {}",
                    locale.text("Expected", "预期", "預期"),
                    shape(&input.expected)
                ),
                width,
            );
            if height >= 16 {
                let extents = [&input.actual, &input.expected]
                    .map(|shape| shape.as_ref().and_then(|(shape, _)| shape.last()));
                let bars = super::offline::axis_bars(extents, width.saturating_sub(8).clamp(4, 24));
                push(
                    &mut out,
                    locale.text(
                        "Last axis · shared scale",
                        "末轴 · 共同尺度",
                        "末軸 · 共同尺度",
                    ),
                    width,
                );
                push(&mut out, &format!("  1 {}", bars[0]), width);
                push(&mut out, &format!("  2 {}", bars[1]), width);
            }
        } else {
            push(
                &mut out,
                locale.text(
                    "No input evidence recorded",
                    "没有已记录的输入证据",
                    "沒有已記錄的輸入證據",
                ),
                width,
            );
        }
        push(
            &mut out,
            if self.navigation_limit {
                locale.text(
                    "128-hop limit · b back · v evidence",
                    "已达 128 跳上限 · b 返回 · v 证据",
                    "已達 128 跳上限 · b 返回 · v 證據",
                )
            } else if self.candidate_hop
                || self
                    .inputs
                    .get(self.selected)
                    .is_some_and(|input| !input.matches.is_empty() && input.wire.is_none())
            {
                locale.text(
                    "↑↓ input ←→ match ↵ open c v b/F",
                    "↑↓ 输入 ←→ 候选 ↵ 打开 c v b/F",
                    "↑↓ 輸入 ←→ 候選 ↵ 開啟 c v b/F",
                )
            } else if view
                .live_evidence
                .as_ref()
                .is_some_and(|evidence| evidence.from_calls)
            {
                locale.text(
                    "↑↓ input · c later · v · l calls · F",
                    "↑↓ 输入 · c 后续 · v · l 调用 · F",
                    "↑↓ 輸入 · c 後續 · v · l 呼叫 · F",
                )
            } else if view.live_evidence.is_some() && self.invocation_id.is_some() {
                locale.text(
                    "↑↓ input · c later · v · ↵/F graph",
                    "↑↓ 输入 · c 后续 · v · ↵/F 图",
                    "↑↓ 輸入 · c 後續 · v · ↵/F 圖",
                )
            } else if width < 48 {
                locale.text(
                    "↑↓ input · v evidence · ↵ graph · Esc",
                    "↑↓ 输入 · v 证据 · ↵ 图 · Esc 返回",
                    "↑↓ 輸入 · v 證據 · ↵ 圖 · Esc 返回",
                )
            } else {
                locale.text(
                    "↑↓ input · v evidence · Enter graph · F/Esc back",
                    "↑↓ 输入 · v 证据 · Enter 图中选择 · F/Esc 返回",
                    "↑↓ 輸入 · v 證據 · Enter 圖中選取 · F/Esc 返回",
                )
            },
            width,
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(unix, windows))]
    fn declared_route_requires_unique_reported_seam_binding_and_is_not_a_flow_edge() {
        let report = serde_json::json!({"data":{"origin":{
            "params":{"input_routes":{"b_encoder":"a", "other":"b"}},
            "injection_evidence":[
                {"slot":"b_encoder","registered_paths":["model.encoders.enc_b"]},
                {"slot":"other","registered_paths":["model.shared", "model.other"]}
            ]
        }}});
        assert_eq!(
            declared_routes(&report, "model.encoders.enc_b", false),
            vec![("b_encoder".into(), "a".into(), "config".into())]
        );
        assert!(declared_routes(&report, "model.encoders.enc_b", true).is_empty());
        assert!(declared_routes(&report, "model.shared", false).is_empty());
        assert!(declared_routes(&report, "model.unknown", false).is_empty());
        let mut view = ModelView::new(None);
        view.live_evidence = Some(super::super::LiveEvidence {
            path: "model.encoders.enc_b".into(),
            ambiguous_registration: false,
            last_observed: true,
            from_calls: false,
            reused: false,
            input: None,
            invocation: Some(1),
            invocation_id: Some("f:1".into()),
            expected: None,
            raw_report: report.to_string(),
            #[cfg(any(unix, windows))]
            route_cache: std::cell::OnceCell::new(),
            inputs: Vec::new(),
            selected_input: None,
            show_input: true,
            graph_target_unresolved: false,
            projection: None,
            diagnostics: Vec::new(),
        });
        let path = Path::live(&view).unwrap();
        let rendered = path.render(&view, 80, 24, Locale::English, None);
        assert!(
            rendered.contains("Config-declared route · not observed flow")
                && rendered.contains("a → b_encoder"),
            "{rendered}"
        );
        assert!(rendered.lines().count() <= 24, "{rendered}");
        assert!(path.wire().is_none());
        assert!(
            path.render(&view, 40, 10, Locale::English, None)
                .lines()
                .count()
                <= 10
        );
    }

    #[test]
    fn self_reported_route_requires_available_complete_unique_registration() {
        let mut report = serde_json::json!({"data":{"origin":{"declared_routes":{
            "scope":"model_self_reported_at_construction", "status":"available",
            "routes":[{"slot":"b_encoder","input":"b","registered_paths":["model.encoders.enc_b"]}]
        }}}});
        assert_eq!(
            declared_routes(&report, "model.encoders.enc_b", false),
            vec![("b_encoder".into(), "b".into(), "model".into())]
        );
        report["data"]["origin"]["declared_routes"]["status"] = "unavailable".into();
        assert!(declared_routes(&report, "model.encoders.enc_b", false).is_empty());
        report["data"]["origin"]["declared_routes"]["status"] = "available".into();
        report["data"]["origin"]["declared_routes"]["routes"][0]["registered_paths"] =
            serde_json::json!(["model.encoders.enc_b", "model.other"]);
        assert!(declared_routes(&report, "model.encoders.enc_b", false).is_empty());
    }

    #[test]
    fn same_object_matches_require_a_unique_earlier_call_in_the_same_forward() {
        let tensor = serde_json::json!({"kind":"tensor","dtype":"torch.float32","shape":[2,4]});
        let call = serde_json::json!({
            "call_id":4,"forward_id":"f1","invocation_id":"f1:4","alias_paths":["model.head"],
            "inputs":{"kind":"dict","items":{"args":{"kind":"tuple","items":[tensor.clone()]},"kwargs":{"kind":"dict","items":{}}}},
            "input_tensor_matches":[{"input_path":"inputs.args[0]","prior_outputs":[
                {"invocation_id":"f1:2","output_path":"outputs"},
                {"invocation_id":"f1:3","output_path":"outputs.hidden[0]"}
            ]}]
        });
        let mut report = serde_json::json!({"data":{
            "forward_id":"f1","tensor_identity_trace":{"kind":"same_tensor_object_and_version","scope":"partial"},
            "calls":[
                {"call_id":2,"forward_id":"f1","invocation_id":"f1:2","invocation_index":1,"alias_paths":["model.src","model.shared"],"outputs":tensor.clone()},
                {"call_id":3,"forward_id":"f1","invocation_id":"f1:3","invocation_index":1,"alias_paths":["model.pass"],"outputs":{"kind":"dict","items":{"hidden":{"kind":"list","items":[tensor]}}},
                 "inputs":call["inputs"].clone(), "input_tensor_matches":[{"input_path":"inputs.args[0]","prior_outputs":[{"invocation_id":"f1:2","output_path":"outputs"}]}]},
                call.clone()
            ]
        }});
        let make_inputs = || observed_inputs(&call["inputs"], None);
        let mut inputs = make_inputs();
        attach_observed_matches(&mut inputs, &call, &report);
        assert_eq!(inputs[0].matches.len(), 2);
        assert!(inputs[0].matches[0].ambiguous_alias);
        let mut view = super::super::ModelView::new(None);
        view.live_evidence = Some(super::super::LiveEvidence {
            path: "model.head".into(),
            ambiguous_registration: false,
            last_observed: true,
            from_calls: false,
            reused: false,
            input: None,
            invocation: Some(1),
            invocation_id: Some("f1:4".into()),
            expected: None,
            raw_report: report.to_string(),
            #[cfg(any(unix, windows))]
            route_cache: std::cell::OnceCell::new(),
            inputs: Vec::new(),
            selected_input: None,
            show_input: true,
            #[cfg(any(unix, windows))]
            graph_target_unresolved: false,
            #[cfg(any(unix, windows))]
            projection: None,
            diagnostics: Vec::new(),
        });
        let mut path = Path::new(
            "model.head".into(),
            Some(1),
            None,
            Stop::LastObserved,
            inputs,
        );
        let frame = path.render(&view, 40, 24, Locale::English, None);
        assert!(frame.contains("2 object matches · partial"), "{frame}");
        assert!(frame.contains("alias? model.src"), "{frame}");
        assert!(frame.contains("not proof of provenance"), "{frame}");
        path.move_match(true);
        let frame = path.render(&view, 40, 24, Locale::English, None);
        assert!(frame.contains("2/2 ←→ model.pass"), "{frame}");
        let mut source = path.follow_match(&view).expect("validated candidate call");
        assert_eq!(source.invocation_id.as_deref(), Some("f1:3"));
        assert_eq!(source.inputs[0].matches[0].invocation_id, "f1:2");
        assert!(
            source
                .render(&view, 40, 24, Locale::English, None)
                .contains("Candidate call · partial match")
        );
        source.show_consumers(&view);
        assert_eq!(source.consumers.as_ref().unwrap().entries.len(), 1);
        let projection_modules = vec![serde_json::json!({"path":"model.head", "type":"Linear"})];
        let projection_view = super::super::ModelView::new(Some(&projection_modules));
        let projection = source.graph_projection(&projection_view).unwrap();
        assert_eq!(
            projection.counts.get(&projection_view.layers[0].id),
            Some(&1)
        );
        assert_eq!(projection.unmapped, 0);
        source.consumers.as_mut().unwrap().entries[0].ambiguous_alias = true;
        let ambiguous = source.graph_projection(&projection_view).unwrap();
        assert!(ambiguous.counts.is_empty());
        assert_eq!(ambiguous.unmapped, 1);
        source.consumers.as_mut().unwrap().entries[0].ambiguous_alias = false;
        assert_eq!(source.consumer_at(0, 3, 40, 24), Some(0));
        assert_eq!(source.consumer_at(0, 4, 40, 24), None);
        assert!(
            source
                .render(&view, 40, 24, Locale::English, None)
                .contains(">└╌ 1/1 model.head")
        );
        source.toggle_consumer_layout();
        assert!(
            source
                .render(&view, 40, 24, Locale::English, None)
                .contains(">   1/1 model.head")
        );
        source.toggle_consumer_layout();
        let consumer = source.open_consumer(&view).expect("validated later call");
        assert_eq!(consumer.invocation_id.as_deref(), Some("f1:4"));
        assert_eq!(consumer.selected_input(), 0);
        assert!(
            source
                .render(&view, 40, 24, Locale::English, None)
                .contains("not proven flow")
        );
        let mut other_forward = report.clone();
        other_forward["data"]["calls"][2]["forward_id"] = "other".into();
        view.live_evidence.as_mut().unwrap().raw_report = other_forward.to_string();
        source.show_consumers(&view);
        assert!(source.consumers.as_ref().unwrap().entries.is_empty());
        let mut many = report.clone();
        let calls = many["data"]["calls"].as_array_mut().unwrap();
        calls.truncate(2);
        for index in 0..130 {
            let mut next = call.clone();
            next["invocation_id"] = format!("f1:{}", index + 4).into();
            next["call_id"] = (index + 4).into();
            calls.push(next);
        }
        view.live_evidence.as_mut().unwrap().raw_report = many.to_string();
        source.show_consumers(&view);
        assert_eq!(source.consumers.as_ref().unwrap().entries.len(), 128);
        assert!(source.consumers.as_ref().unwrap().limited);
        source.move_consumer(true, true, 24);
        assert_eq!(source.consumer_at(4, 17, 40, 24), Some(32));
        source.select_consumer(32);
        assert_eq!(source.consumers.as_ref().unwrap().selected, 32);
        assert_eq!(source.consumer_at(4, 3, 40, 10), Some(32));
        assert_eq!(source.consumer_at(40, 3, 40, 10), None);
        assert_eq!(source.consumer_at(4, 7, 40, 10), None);
        view.live_evidence.as_mut().unwrap().raw_report = report.to_string();
        path.move_match(true);
        assert!(
            path.render(&view, 40, 24, Locale::English, None)
                .contains("2/2 ←→ model.pass")
        );
        path.move_match(false);
        assert!(path.follow_match(&view).unwrap().graph_target().is_none());
        assert!(
            path.render(&view, 40, 24, Locale::English, None)
                .contains("1/2 ←→ alias? model.src")
        );
        let mut second_input = path.inputs[0].clone();
        second_input.label = Some("inputs.args[1]".into());
        let mut multi = Path::new(
            "model.head".into(),
            Some(1),
            None,
            Stop::LastObserved,
            vec![path.inputs[0].clone(), second_input],
        );
        multi.move_match(true);
        multi.movement(true);
        assert!(
            multi
                .render(&view, 40, 24, Locale::English, None)
                .contains("1/2 ←→ alias? model.src")
        );
        multi.movement(false);
        assert!(
            multi
                .render(&view, 40, 24, Locale::English, None)
                .contains("2/2 ←→ model.pass")
        );
        report["data"]["calls"][1]["forward_id"] = "other".into();
        let mut inputs = make_inputs();
        attach_observed_matches(&mut inputs, &call, &report);
        assert_eq!(inputs[0].matches.len(), 1);
        let duplicate = report["data"]["calls"][0].clone();
        report["data"]["calls"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        let mut inputs = make_inputs();
        attach_observed_matches(&mut inputs, &call, &report);
        assert!(inputs[0].matches.is_empty());
        report["data"]["tensor_identity_trace"]["kind"] = "unknown".into();
        let mut inputs = make_inputs();
        attach_observed_matches(&mut inputs, &call, &report);
        assert!(inputs[0].matches.is_empty());
    }

    #[test]
    fn input_search_jumps_only_to_a_unique_complete_observation() {
        let tree = serde_json::json!({"kind":"dict","items":{
            "args":{"kind":"tuple","items":[{"kind":"tensor","dtype":"torch.float32","shape":[2,48]}]},
            "kwargs":{"kind":"dict","items":{"mask":{"kind":"tensor","dtype":"torch.bool","shape":[2,7]}}}
        }});
        let mut inputs = observed_inputs(&tree, None);
        assert_eq!(unique_input_match(&inputs, "inputs.kwargs.mask"), Some(1));
        assert_eq!(unique_input_match(&inputs, "2x7"), Some(1));
        assert_eq!(unique_input_match(&inputs, "[2,48]"), Some(0));
        assert_eq!(unique_input_match(&inputs, "tensor"), None);
        assert_eq!(
            matching_input_indices(&inputs, "tensor"),
            (vec![0, 1], false)
        );
        assert_eq!(unique_input_match(&inputs, "outputs"), None);
        assert_eq!(unique_input_match(&inputs, ""), None);
        inputs.push(Input {
            wire: None,
            actual: None,
            expected: None,
            label: None,
            metadata: Some("truncated · 2 omitted".into()),
            matches: Vec::new(),
        });
        assert_eq!(unique_input_match(&inputs, "inputs.kwargs.mask"), None);
        assert_eq!(
            matching_input_indices(&inputs, "inputs.kwargs.mask"),
            (vec![1], true)
        );
    }

    #[test]
    fn live_failure_location_does_not_invent_a_producer() {
        let mut view = ModelView::new(None);
        view.live_evidence = Some(super::super::LiveEvidence {
            path: "model.head".into(),
            ambiguous_registration: false,
            last_observed: true,
            from_calls: false,
            reused: false,
            selected_input: None,
            show_input: false,
            #[cfg(any(unix, windows))]
            graph_target_unresolved: false,
            #[cfg(any(unix, windows))]
            projection: None,
            input: Some(Shape::Dimensions(vec![Some(2), Some(48)])),
            invocation: Some(2),
            invocation_id: None,
            expected: Some(Shape::Features(32)),
            raw_report: "{\"diagnostics\":[{\"code\":\"FORWARD_FAILED\"}]}".into(),
            #[cfg(any(unix, windows))]
            route_cache: std::cell::OnceCell::new(),
            inputs: Vec::new(),
            diagnostics: Vec::new(),
        });
        view.live_evidence.as_mut().unwrap().last_observed = false;
        assert_eq!(Path::live(&view).unwrap().stop, Stop::Call);
        view.live_evidence.as_mut().unwrap().last_observed = true;
        assert_eq!(Path::live(&view).unwrap().stop, Stop::LastObserved);
        let path = Path::live(&view).unwrap();
        let text = path.render(&view, 80, 24, Locale::English, None);
        assert!(text.contains("Last observed · failure location unknown"));
        assert!(text.contains("Source not recorded"));
        assert!(text.contains("2×48 (observed)"));
        assert!(path.wire().is_none());
        view.live_evidence.as_mut().unwrap().ambiguous_registration = true;
        let shared = Path::live(&view)
            .unwrap()
            .render(&view, 40, 24, Locale::English, None);
        assert!(
            shared.contains("Shared call · registration unknown"),
            "{shared}"
        );
        let shared_compact =
            Path::live(&view)
                .unwrap()
                .render(&view, 30, 10, Locale::English, None);
        assert!(shared_compact.lines().count() <= 10, "{shared_compact}");
        view.live_evidence.as_mut().unwrap().ambiguous_registration = false;
        let compact = path.render(&view, 30, 10, Locale::English, None);
        assert!(compact.lines().count() <= 10);
        assert!(compact.contains("2×48 (observed)"));
        assert!(compact.contains("…×32 (declared)"));
        view.live_evidence.as_mut().unwrap().reused = true;
        let reused = Path::live(&view)
            .unwrap()
            .render(&view, 80, 24, Locale::English, None);
        assert!(reused.contains("Reused · no new forward"), "{reused}");
        let small_reused = Path::live(&view)
            .unwrap()
            .render(&view, 30, 10, Locale::English, None);
        assert!(small_reused.lines().count() <= 10, "{small_reused}");
        assert!(
            small_reused.contains("Reused · no new forward"),
            "{small_reused}"
        );
        view.live_evidence.as_mut().unwrap().ambiguous_registration = true;
        let both = Path::live(&view)
            .unwrap()
            .render(&view, 40, 10, Locale::English, None);
        assert!(both.lines().count() <= 10, "{both}");
        assert!(
            both.contains("Shared call · registration unknown"),
            "{both}"
        );
        assert!(both.contains("Reused · no new forward"), "{both}");
    }

    #[test]
    fn narrow_port_label_preserves_slot_and_path_suffix() {
        let label = port_label("模型.编码器.注意力", "out", 12, 18);
        assert!(label.ends_with("注意力.out[12]"));
        assert!(crate::terminal::text::display_width(&label) <= 18);
    }

    #[test]
    fn original_report_scrolls_and_returns_to_the_same_input() {
        let mut view = ModelView::new(None);
        let raw_report = (0..20)
            .map(|index| format!("field_{index}: value_{index}"))
            .collect::<Vec<_>>()
            .join("\n");
        view.live_evidence = Some(super::super::LiveEvidence {
            path: "model.head".into(),
            ambiguous_registration: false,
            last_observed: true,
            from_calls: false,
            reused: false,
            selected_input: None,
            show_input: false,
            #[cfg(any(unix, windows))]
            graph_target_unresolved: false,
            #[cfg(any(unix, windows))]
            projection: None,
            input: None,
            invocation: Some(2),
            invocation_id: None,
            expected: None,
            raw_report,
            #[cfg(any(unix, windows))]
            route_cache: std::cell::OnceCell::new(),
            inputs: Vec::new(),
            diagnostics: Vec::new(),
        });
        let mut path = Path::live(&view).unwrap();
        path.toggle_report(&view, None);
        let first = path.render(&view, 50, 6, Locale::English, None);
        assert!(first.contains("Original CLI report"));
        assert!(first.contains("field_0"));
        path.scroll_report(&view, None, 50, 6, Locale::English, true, true);
        let next = path.render(&view, 50, 6, Locale::English, None);
        assert!(next.contains("field_4"));
        assert!(!next.contains("field_0"));
        path.toggle_report(&view, None);
        assert!(
            path.render(&view, 50, 12, Locale::English, None)
                .contains("Input 1/1")
        );
    }

    #[test]
    fn nested_live_inputs_keep_paths_and_only_assign_expected_to_linear_input() {
        use serde_json::json;
        let tree = json!({"kind":"dict","items":{
            "args":{"kind":"tuple","items":[
                {"kind":"tensor","shape":[2,48],"dtype":"torch.float32"},
                {"kind":"dict","items":{"features":{"kind":"tensor","shape":[2,7],"dtype":"torch.int64"},"<truncated>":3}}
            ],"truncated":1},
            "kwargs":{"kind":"dict","items":{"mask":{"kind":"list","items":[{"kind":"tensor","shape":[2,1],"dtype":"torch.bool"}],"truncated":0}}}
        }});
        let inputs = observed_inputs(&tree, Some(&Shape::Features(32)));
        let named = |path: &str| {
            inputs
                .iter()
                .find(|input| input.label.as_deref() == Some(path))
                .unwrap()
        };
        assert_eq!(
            named("inputs.args[0]").actual.as_ref().unwrap().0.last(),
            Some(48)
        );
        assert!(named("inputs.args[0]").expected.is_some());
        assert!(named("inputs.args[1].features").expected.is_none());
        assert_eq!(
            named("inputs.kwargs.mask[0]")
                .actual
                .as_ref()
                .unwrap()
                .0
                .last(),
            Some(1)
        );
        assert!(
            inputs
                .iter()
                .any(|input| input.metadata.as_deref() == Some("truncated · 3 omitted"))
        );
        assert!(
            inputs
                .iter()
                .any(|input| input.metadata.as_deref() == Some("truncated · 1 omitted"))
        );
        let keyword = observed_inputs(
            &json!({"kind":"dict","items":{"kwargs":{"kind":"dict","items":{"input":{"kind":"tensor","shape":[1,5]}}}}}),
            Some(&Shape::Features(4)),
        );
        assert!(keyword[0].expected.is_some());
    }
}
