//! A shared-layout review of recorded, tentative and committed repair evidence.
use super::push;
use crate::terminal::{locale::Locale, text};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Project only explicitly declared built-in Sequential order. Missing edges
/// are unknown; they do not establish an absent runtime connection.
fn declared_relations(records: Option<&[serde_json::Value]>) -> Option<BTreeSet<(String, String)>> {
    let records = records.filter(|records| records.len() <= 4096)?;
    let segment = super::report_segment(
        records.iter().map(std::borrow::Cow::Borrowed).collect(),
        false,
    );
    let mut operators = HashMap::new();
    let mut wires = Vec::new();
    for edit in segment.edits {
        match edit {
            super::Edit::AddOperator(id, operator) => {
                operators.insert(id, operator);
            }
            super::Edit::Connect(wire) => wires.push(wire),
            _ => {}
        }
    }
    let mut relations = BTreeSet::new();
    for wire in wires {
        let (Some(from), Some(to)) = (
            operators.get(&wire.from.operator),
            operators.get(&wire.to.operator),
        ) else {
            continue;
        };
        let Some(parent_id) = from.parent.filter(|parent| Some(*parent) == to.parent) else {
            continue;
        };
        let Some(parent) = operators.get(&parent_id) else {
            continue;
        };
        if !parent.attributes["sequential_children"].is_array() {
            continue;
        }
        relations.insert((from.path.clone(), to.path.clone()));
    }
    Some(relations)
}

#[derive(Clone)]
struct DeclaredDiagram {
    before: Vec<String>,
    after: Vec<String>,
    offset: usize,
    horizontal: usize,
    max_width: usize,
}

#[derive(Clone)]
pub(crate) struct Diff {
    pub title: String,
    pub status: String,
    before: Vec<String>,
    after: Vec<String>,
    pub details: Vec<String>,
    side: usize,
    detail_open: bool,
    offset: usize,
    content_offset: usize,
    horizontal_offset: usize,
    max_content_width: usize,
    diagram: Option<DeclaredDiagram>,
    diagram_open: bool,
    receipt_commit: Option<String>,
    receipt_status: Option<String>,
}

impl Diff {
    pub fn live_preview(
        report: &crate::cli::Report,
        before: Option<&[serde_json::Value]>,
        target: &str,
        locale: Locale,
        requested_training: bool,
    ) -> Self {
        let find = |modules: Option<&[serde_json::Value]>| -> Option<(String, String)> {
            let mut matches = modules?
                .iter()
                .filter(|module| module["path"].as_str() == Some(target));
            let module = matches.next()?;
            if matches.next().is_some() {
                return None;
            }
            let value = |key| {
                module["editable_params"][key]
                    .as_u64()
                    .map_or("?".into(), |value| value.to_string())
            };
            Some((value("in_features"), value("out_features")))
        };
        let previous = find(before).unwrap_or(("?".into(), "?".into()));
        let candidate = find(report.data["modules"].as_array().map(Vec::as_slice))
            .unwrap_or(("?".into(), "?".into()));
        let mode_conflict =
            crate::workbench::mode_mismatch(&report.data["validation"]["mode"], requested_training);
        let mode_unknown =
            crate::workbench::reported_training_mode(&report.data["validation"]["mode"]).is_none();
        let status = if report.status != "success" {
            locale.text("Preview rejected", "预览已拒绝", "預覽已拒絕")
        } else if report.data["committed"] == false {
            locale.text(
                "Preview · not committed",
                "预览 · 尚未提交",
                "預覽 · 尚未提交",
            )
        } else if report.data["committed"] == true {
            locale.text("CLI reports committed", "CLI 报告已提交", "CLI 報告已提交")
        } else {
            locale.text("Commit state unknown", "提交状态未知", "提交狀態未知")
        };
        let status = if mode_conflict {
            format!(
                "{} · {status}",
                locale.text("Mode mismatch", "模式不一致", "模式不一致")
            )
        } else if mode_unknown && report.status == "success" {
            format!(
                "{} · {status}",
                locale.text("Mode unverified", "模式未核对", "模式未核對")
            )
        } else {
            status.to_owned()
        };
        let mut effects = Vec::new();
        let mut alias_effects = Vec::new();
        if let Some(items) = report.data["edit_effects"].as_array() {
            for item in items
                .iter()
                .filter(|item| item["path"].as_str() == Some(target))
            {
                alias_effects.extend(crate::workbench::reported_alias_effects(Some(item), locale));
                for (field, name) in [
                    (
                        "initialized",
                        locale.text("Initialized", "重新初始化", "重新初始化"),
                    ),
                    ("preserved", locale.text("Preserved", "保留", "保留")),
                ] {
                    if let Some(values) = item[field].as_array() {
                        for value in values.iter().filter_map(serde_json::Value::as_str) {
                            effects.push(format!("{name} {target}.{value}"));
                        }
                    }
                }
            }
        }
        if alias_effects.is_empty() {
            alias_effects.extend(crate::workbench::reported_alias_effects(None, locale));
        }
        if effects.is_empty() {
            effects.push(
                locale
                    .text("Weight effect unknown", "权重影响未知", "權重影響未知")
                    .into(),
            );
        }
        let fields = |heading: &str, features: &(String, String)| {
            vec![
                heading.into(),
                format!("┌─ {target}"),
                format!("│  in_features  {}", features.0),
                format!("└─ out_features {}", features.1),
                locale
                    .text(
                        "Runtime connections not compared",
                        "未对照运行时连线",
                        "未對照執行時連線",
                    )
                    .into(),
            ]
        };
        let before_rows = fields(
            locale.text("Before · inspected", "修改前 · 已检查", "修改前 · 已檢查"),
            &previous,
        );
        let mut after_rows = fields(
            locale.text(
                "Candidate · CLI preview",
                "候选 · CLI 预览",
                "候選 · CLI 預覽",
            ),
            &candidate,
        );
        after_rows.extend(effects.iter().cloned());
        after_rows.extend(alias_effects.iter().cloned());
        if let Some(forward) = report.data["validation"]["forward"].as_str() {
            after_rows.push(format!(
                "{} {forward}",
                locale.text("Candidate forward", "候选前向", "候選前向")
            ));
        }
        let mode = report.data["validation"]["mode"]
            .as_str()
            .filter(|mode| !mode.trim().is_empty())
            .map(text::plain_text)
            .unwrap_or_else(|| locale.text("unknown", "未知", "未知").to_owned());
        after_rows.push(format!(
            "{} {mode}",
            locale.text("Candidate mode · CLI", "候选模式 · CLI", "候選模式 · CLI")
        ));
        if mode_conflict {
            after_rows.push(
                locale
                    .text(
                        "Mode mismatch · requested mode not validated",
                        "模式不一致 · 未验证请求模式",
                        "模式不一致 · 未驗證請求模式",
                    )
                    .to_owned(),
            );
        } else if mode_unknown {
            after_rows.push(
                locale
                    .text(
                        "Candidate mode unverified",
                        "候选模式无法核对",
                        "候選模式無法核對",
                    )
                    .to_owned(),
            );
        }
        let mut before_rows = before_rows;
        let common_height = before_rows.len().max(after_rows.len());
        before_rows.resize(common_height, String::new());
        after_rows.resize(common_height, String::new());
        let candidate_modules = report.data["modules"].as_array().map(Vec::as_slice);
        let old_relations = declared_relations(before);
        let new_relations = declared_relations(candidate_modules);
        let heading = locale.text(
            "Sequential order · declaration",
            "Sequential 顺序 · 声明",
            "Sequential 順序 · 宣告",
        );
        let caveat = locale.text(
            "Order only · runtime flow unknown",
            "仅为顺序 · 运行时数据流未知",
            "僅為順序 · 執行時資料流未知",
        );
        before_rows.extend([heading.to_owned(), caveat.to_owned()]);
        after_rows.extend([heading.to_owned(), caveat.to_owned()]);
        let relations: BTreeSet<_> = old_relations
            .iter()
            .flatten()
            .chain(new_relations.iter().flatten())
            .cloned()
            .collect();
        let mut diagram_before = Vec::new();
        let mut diagram_after = Vec::new();
        if !relations.is_empty() {
            diagram_before.push(
                locale
                    .text("Before · declared", "修改前 · 声明", "修改前 · 宣告")
                    .to_owned(),
            );
            diagram_after.push(
                locale
                    .text("Candidate · declared", "候选 · 声明", "候選 · 宣告")
                    .to_owned(),
            );
            let caveat = locale
                .text(
                    "Order only · runtime flow unknown",
                    "仅为顺序 · 运行时数据流未知",
                    "僅為順序 · 執行時資料流未知",
                )
                .to_owned();
            diagram_before.push(caveat.clone());
            diagram_after.push(caveat);
        }
        if relations.is_empty() {
            let empty = |available: bool| {
                if available {
                    locale.text(
                        "No declared links recorded",
                        "未记录声明连线",
                        "未記錄宣告連線",
                    )
                } else {
                    locale.text(
                        "Report unavailable or too large",
                        "报告缺失或过大",
                        "報告缺失或過大",
                    )
                }
            };
            before_rows.push(empty(old_relations.is_some()).to_owned());
            after_rows.push(empty(new_relations.is_some()).to_owned());
        }
        for (from, to) in relations {
            let in_old = old_relations
                .as_ref()
                .is_some_and(|links| links.contains(&(from.clone(), to.clone())));
            let in_new = new_relations
                .as_ref()
                .is_some_and(|links| links.contains(&(from.clone(), to.clone())));
            let mark = if in_old && in_new { "=" } else { "?" };
            let link = format!(
                "{mark} {} → {}",
                text::plain_text(&from),
                text::plain_text(&to)
            );
            let missing = |available: bool| {
                if available {
                    locale.text("? Not declared in report", "? 报告未声明", "? 報告未宣告")
                } else {
                    locale.text(
                        "? Report unavailable or too large",
                        "? 报告缺失或过大",
                        "? 報告缺失或過大",
                    )
                }
                .to_owned()
            };
            let diagram = |present: bool, available: bool| {
                if present {
                    let target_mark = locale.text(" [target]", " [目标]", " [目標]");
                    [
                        format!(
                            "{mark} ╭─ {}{}",
                            text::plain_text(&from),
                            if from == target { target_mark } else { "" }
                        ),
                        format!(
                            "  ╰─▶ {}{}",
                            text::plain_text(&to),
                            if to == target { target_mark } else { "" }
                        ),
                        String::new(),
                    ]
                } else {
                    [
                        missing(available),
                        locale
                            .text("  ? relation unknown", "  ? 关系未知", "  ? 關係未知")
                            .to_owned(),
                        String::new(),
                    ]
                }
            };
            diagram_before.extend(diagram(in_old, old_relations.is_some()));
            diagram_after.extend(diagram(in_new, new_relations.is_some()));
            before_rows.push(if in_old {
                link.clone()
            } else {
                missing(old_relations.is_some())
            });
            after_rows.push(if in_new {
                link
            } else {
                missing(new_relations.is_some())
            });
        }
        before_rows.push(
            locale
                .text(
                    "= both reports · ? unknown on one side",
                    "= 两份报告均有 · ? 一侧未知",
                    "= 兩份報告均有 · ? 一側未知",
                )
                .to_owned(),
        );
        after_rows.push(
            locale
                .text(
                    "= both reports · ? unknown on one side",
                    "= 两份报告均有 · ? 一侧未知",
                    "= 兩份報告均有 · ? 一側未知",
                )
                .to_owned(),
        );
        before_rows.push(
            locale
                .text("Reported module tree", "报告的模块树", "報告的模組樹")
                .into(),
        );
        after_rows.push(
            locale
                .text("Reported module tree", "报告的模块树", "報告的模組樹")
                .into(),
        );
        let mut modules: BTreeMap<String, (Vec<&serde_json::Value>, Vec<&serde_json::Value>)> =
            BTreeMap::new();
        for (side, records) in [(0, before), (1, candidate_modules)] {
            for module in records.into_iter().flatten() {
                if let Some(path) = module["path"].as_str().filter(|path| !path.is_empty()) {
                    let entry = modules.entry(path.to_owned()).or_default();
                    if side == 0 {
                        entry.0.push(module);
                    } else {
                        entry.1.push(module);
                    }
                }
            }
        }
        for (path, (old, new)) in modules {
            let ambiguous = old.len() > 1 || new.len() > 1;
            let mark = if ambiguous {
                "?"
            } else if old.is_empty() {
                "+"
            } else if new.is_empty() {
                "-"
            } else if old[0] != new[0] {
                "~"
            } else {
                " "
            };
            let row = |records: &[&serde_json::Value], available: bool| {
                if records.is_empty() {
                    return if available {
                        "—".to_owned()
                    } else {
                        locale.text("Not reported", "未报告", "未報告").to_owned()
                    };
                }
                if records.len() != 1 {
                    return format!(
                        "? {} ({})",
                        text::plain_text(&path),
                        locale.text("duplicate path", "路径重复", "路徑重複")
                    );
                }
                format!(
                    "{mark} {} · {}",
                    text::plain_text(&path),
                    text::plain_text(records[0]["type"].as_str().unwrap_or("?"))
                )
            };
            before_rows.push(row(&old, before.is_some()));
            after_rows.push(row(&new, candidate_modules.is_some()));
        }
        before_rows.push(
            locale
                .text(
                    "+/- only in one report · ~ fields differ · ? ambiguous",
                    "+/- 仅见于单份报告 · ~ 字段不同 · ? 歧义",
                    "+/- 僅見於單份報告 · ~ 欄位不同 · ? 歧義",
                )
                .into(),
        );
        after_rows.push(
            locale
                .text(
                    "+/- only in one report · ~ fields differ · ? ambiguous",
                    "+/- 仅见于单份报告 · ~ 字段不同 · ? 歧义",
                    "+/- 僅見於單份報告 · ~ 欄位不同 · ? 歧義",
                )
                .into(),
        );
        let mut details = vec![
            format!("{} {target}", locale.text("Target", "目标", "目標")),
            format!("in_features: {} → {}", previous.0, candidate.0),
            format!("out_features: {} → {}", previous.1, candidate.1),
            status.clone(),
            locale
                .text(
                    "The inspected snapshot has not been matched to the preview base revision",
                    "检查快照尚未与预览基准版本核对",
                    "檢查快照尚未與預覽基準版本核對",
                )
                .into(),
        ];
        details.extend(effects);
        details.extend(alias_effects);
        if let Some(variant) = report.data["variant_name"].as_str() {
            details.push(format!(
                "{} {variant}",
                locale.text("Candidate variant", "候选变体", "候選變體")
            ));
        }
        let forward = report.data["validation"]["forward"]
            .as_str()
            .unwrap_or_else(|| locale.text("unknown", "未知", "未知"));
        details.push(format!(
            "{} {forward}",
            locale.text("Candidate forward", "候选前向", "候選前向")
        ));
        details.push(format!(
            "{} {mode}",
            locale.text("Candidate mode · CLI", "候选模式 · CLI", "候選模式 · CLI")
        ));
        if mode_conflict {
            details.push(
                locale
                    .text(
                        "Mode mismatch · requested mode not validated",
                        "模式不一致 · 未验证请求模式",
                        "模式不一致 · 未驗證請求模式",
                    )
                    .to_owned(),
            );
        } else if mode_unknown {
            details.push(
                locale
                    .text(
                        "Candidate mode unverified",
                        "候选模式无法核对",
                        "候選模式無法核對",
                    )
                    .to_owned(),
            );
        }
        details.push(
            locale
                .text(
                    "The same sample is not established for candidate validation",
                    "候选验证不能确认样本相同",
                    "候選驗證不能確認樣本相同",
                )
                .into(),
        );
        for diagnostic in &report.diagnostics {
            details.push(format!("[{}] {}", diagnostic.code, diagnostic.message));
        }
        let mut result = Self::new(target.to_owned(), status, before_rows, after_rows, details);
        if !diagram_before.is_empty() {
            let max_width = diagram_before
                .iter()
                .chain(&diagram_after)
                .map(|row| text::display_width(row))
                .max()
                .unwrap_or(0);
            result.diagram = Some(DeclaredDiagram {
                before: diagram_before,
                after: diagram_after,
                offset: 0,
                horizontal: 0,
                max_width,
            });
        }
        result
    }
    pub fn new(
        title: String,
        status: String,
        before: Vec<String>,
        after: Vec<String>,
        details: Vec<String>,
    ) -> Self {
        let max_content_width = before
            .iter()
            .chain(&after)
            .map(|row| text::display_width(row))
            .max()
            .unwrap_or(0);
        Self {
            title,
            status,
            before,
            after,
            details,
            side: 0,
            detail_open: false,
            offset: 0,
            content_offset: 0,
            horizontal_offset: 0,
            max_content_width,
            diagram: None,
            diagram_open: false,
            receipt_commit: None,
            receipt_status: None,
        }
    }
    pub fn preserve_side(&mut self, previous: &Self) {
        self.side = previous.side;
        self.content_offset = previous.content_offset;
        self.horizontal_offset = previous.horizontal_offset;
        if let (Some(current), Some(old)) = (&mut self.diagram, &previous.diagram) {
            current.offset = old.offset;
            current.horizontal = old.horizontal;
            self.diagram_open = previous.diagram_open;
        }
    }
    pub fn change_side(&mut self) {
        self.side = 1 - self.side;
    }
    pub fn toggle_diagram(&mut self) {
        if !self.detail_open && self.diagram.is_some() {
            self.diagram_open = !self.diagram_open;
        }
    }
    pub fn attach_receipt(
        &mut self,
        commit: String,
        status: String,
        outcome: &str,
        locale: Locale,
    ) {
        self.status = format!(
            "{} · {}",
            locale.text("Historical preview", "历史预览", "歷史預覽"),
            self.status
        );
        self.receipt_commit = Some(commit.clone());
        self.receipt_status = Some(status.clone());
        self.details.insert(0, status);
        self.details.insert(0, commit);
        self.details.push(
            locale
                .text(
                    "Historical commit and recheck output",
                    "历史提交与重验输出",
                    "歷史提交與重驗輸出",
                )
                .to_owned(),
        );
        self.details.extend(outcome.lines().map(str::to_owned));
    }

    fn body_rows(&self, height: usize) -> usize {
        height
            .saturating_sub(if self.receipt_status.is_some() { 7 } else { 5 })
            .max(1)
    }
    pub fn detail_open(&self) -> bool {
        self.detail_open
    }
    pub fn toggle_details(&mut self) {
        self.detail_open = !self.detail_open;
        self.offset = 0;
    }

    fn column_width(width: usize) -> usize {
        if width >= 72 {
            width.saturating_sub(3) / 2
        } else {
            width
        }
    }

    fn max_horizontal(&self, width: usize) -> usize {
        let column = Self::column_width(width);
        let content_width = if self.diagram_open {
            self.diagram
                .as_ref()
                .map_or(self.max_content_width, |diagram| diagram.max_width)
        } else {
            self.max_content_width
        };
        content_width.saturating_sub(column.saturating_sub(2).max(1))
    }

    pub fn scroll_horizontal(&mut self, width: usize, forward: bool) {
        if self.detail_open {
            return;
        }
        let max = self.max_horizontal(width);
        let offset = if self.diagram_open {
            &mut self.diagram.as_mut().unwrap().horizontal
        } else {
            &mut self.horizontal_offset
        };
        let current = (*offset).min(max);
        let step = (Self::column_width(width) / 3).clamp(1, 8);
        *offset = if forward {
            current.saturating_add(step).min(max)
        } else {
            current.saturating_sub(step)
        };
    }
    pub fn scroll(&mut self, width: usize, height: usize, forward: bool, page: bool) {
        let body_rows = self.body_rows(height);
        let (count, rows, offset) = if self.detail_open {
            let count = self
                .details
                .iter()
                .map(|line| text::wrap(line, width).len())
                .sum();
            (count, height.saturating_sub(2).max(1), &mut self.offset)
        } else if self.diagram_open {
            let diagram = self.diagram.as_mut().unwrap();
            (
                diagram.before.len().max(diagram.after.len()),
                body_rows,
                &mut diagram.offset,
            )
        } else {
            (
                self.before.len().max(self.after.len()),
                body_rows,
                &mut self.content_offset,
            )
        };
        let last = count.saturating_sub(rows);
        let step = if page { rows } else { 1 };
        *offset = if forward {
            offset.saturating_add(step).min(last)
        } else {
            offset.saturating_sub(step)
        };
    }
    pub fn render(&self, width: usize, height: usize, locale: Locale) -> String {
        let mut out = String::new();
        if self.detail_open {
            let lines: Vec<_> = self
                .details
                .iter()
                .flat_map(|line| text::wrap(line, width))
                .collect();
            let rows = height.saturating_sub(2).max(1);
            let start = self.offset.min(lines.len().saturating_sub(rows));
            let end = (start + rows).min(lines.len());
            push(
                &mut out,
                &format!(
                    "{} · {}–{end}/{}",
                    locale.text("Repair evidence", "修复证据", "修復證據"),
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
        push(
            &mut out,
            if self.diagram_open {
                locale.text(
                    "Declared order diagram · partial",
                    "声明顺序图 · 部分证据",
                    "宣告順序圖 · 部分證據",
                )
            } else {
                locale.text(
                    "Repair difference · same target",
                    "修复差异 · 同一目标",
                    "修復差異 · 同一目標",
                )
            },
            width,
        );
        push(&mut out, &self.title, width);
        push(&mut out, &self.status, width);
        if let Some(commit) = &self.receipt_commit {
            push(&mut out, commit, width);
        }
        if let Some(status) = &self.receipt_status {
            push(&mut out, status, width);
        }
        let rows = self.body_rows(height);
        let (before, after, offset, horizontal) = if self.diagram_open {
            let diagram = self.diagram.as_ref().unwrap();
            (
                &diagram.before,
                &diagram.after,
                diagram.offset,
                diagram.horizontal,
            )
        } else {
            (
                &self.before,
                &self.after,
                self.content_offset,
                self.horizontal_offset,
            )
        };
        let horizontal = horizontal.min(self.max_horizontal(width));
        if width >= 72 {
            let column = (width.saturating_sub(3)) / 2;
            let start = offset.min(before.len().max(after.len()).saturating_sub(rows));
            for index in start..(start + rows).min(before.len().max(after.len())) {
                let left = text::column_window(
                    before.get(index).map_or("", String::as_str),
                    horizontal,
                    column,
                );
                let spaces = column.saturating_sub(text::display_width(&left));
                let right = text::column_window(
                    after.get(index).map_or("", String::as_str),
                    horizontal,
                    column,
                );
                push(
                    &mut out,
                    &format!("{left}{} │ {right}", " ".repeat(spaces)),
                    width,
                );
            }
        } else {
            let selected = if self.side == 0 { before } else { after };
            let start = offset.min(before.len().max(after.len()).saturating_sub(rows));
            for line in selected.iter().skip(start).take(rows) {
                push(
                    &mut out,
                    &text::column_window(line, horizontal, width),
                    width,
                );
            }
        }
        push(
            &mut out,
            if self.diagram.is_some() {
                if self.diagram_open {
                    locale.text(
                        "g summary · n/p · [/] · Tab · v · Esc",
                        "g 摘要 · n/p · [/] · Tab · v · Esc",
                        "g 摘要 · n/p · [/] · Tab · v · Esc",
                    )
                } else {
                    locale.text(
                        "g diagram · n/p · [/] · Tab · v · Esc",
                        "g 图 · n/p · [/] · Tab · v · Esc",
                        "g 圖 · n/p · [/] · Tab · v · Esc",
                    )
                }
            } else {
                locale.text(
                    "n/p scroll · [/] pan · Tab · v · Esc",
                    "n/p 滚动 · [/] 平移 · Tab · v · Esc",
                    "n/p 捲動 · [/] 平移 · Tab · v · Esc",
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
    use serde_json::json;

    #[test]
    fn live_preview_aligns_reported_modules_and_scrolls_both_sides_together() {
        let before = vec![
            json!({"path":"model","type":"Net"}),
            json!({"path":"model.a","type":"Linear"}),
            json!({"path":"model.duplicate","type":"Linear"}),
            json!({"path":"model.duplicate","type":"Linear"}),
            json!({"path":"model.head","type":"Linear","editable_params":{"in_features":32}}),
            json!({"path":"model.old","type":"ReLU"}),
        ];
        let report: crate::cli::Report = serde_json::from_value(json!({
            "protocol_version":1,"operation":"edit","status":"success",
            "data":{"committed":false,"modules":[
                {"path":"model","type":"Net"},
                {"path":"model.a","type":"Linear"},
                {"path":"model.duplicate","type":"Linear"},
                {"path":"model.head","type":"Linear","editable_params":{"in_features":48}},
                {"path":"model.new","type":"SiLU"}
            ]}
        }))
        .unwrap();
        let mut diff =
            Diff::live_preview(&report, Some(&before), "model.head", Locale::English, true);
        assert_eq!(diff.before.len(), diff.after.len());
        let position = |prefix: &str| {
            diff.before
                .iter()
                .position(|row| row.contains(prefix))
                .unwrap()
        };
        let head = position("model.head · Linear");
        assert!(diff.before[head].starts_with("~ "));
        assert!(diff.after[head].contains("model.head · Linear"));
        let duplicate = position("duplicate path");
        assert!(diff.after[duplicate].starts_with("? "));
        let old = position("model.old · ReLU");
        assert_eq!(diff.after[old], "—");
        let new = diff
            .after
            .iter()
            .position(|row| row.contains("model.new · SiLU"))
            .unwrap();
        assert_eq!(diff.before[new], "—");
        assert!(diff.after[new].starts_with("+ "));
        diff.scroll(40, 10, true, true);
        diff.scroll(40, 10, true, true);
        assert!(diff.content_offset > 0);
        diff.content_offset = head.saturating_sub(1);
        let offset = diff.content_offset;
        let left = diff.render(40, 10, Locale::English);
        diff.change_side();
        let right = diff.render(40, 10, Locale::English);
        assert_eq!(diff.content_offset, offset);
        assert!(
            left.contains("model.head") && right.contains("model.head"),
            "{left}\n{right}"
        );
        assert!(left.contains("n/p scroll") && right.contains("n/p scroll"));
        diff.toggle_details();
        diff.toggle_details();
        assert_eq!(diff.content_offset, offset);
        let no_snapshot = Diff::live_preview(&report, None, "model.head", Locale::English, false);
        assert!(no_snapshot.before.iter().any(|row| row == "Not reported"));
    }

    #[test]
    fn live_preview_uses_reported_candidate_and_does_not_invent_connections() {
        let before = vec![
            json!({"path":"model.head","editable_params":{"in_features":32,"out_features":8}}),
        ];
        let report: crate::cli::Report = serde_json::from_value(json!({
            "protocol_version":1,"operation":"edit","status":"success",
            "data":{"committed":false,"variant_name":"working","validation":{"forward":"passed","mode":"train"},
                "modules":[{"path":"model.head","editable_params":{"in_features":48,"out_features":8}}],
                "edit_effects":[{"path":"model.head","initialized":["weight"],"preserved":["bias"],"affected_aliases":["model.head","model.alias"]}]}
        })).unwrap();
        let mut diff =
            Diff::live_preview(&report, Some(&before), "model.head", Locale::English, true);
        let frame = diff.render(100, 24, Locale::English);
        assert!(frame.contains("in_features  32"));
        assert!(frame.contains("in_features  48"));
        assert!(frame.contains("Runtime connections not compared"));
        assert!(frame.contains("Initialized model.head.weight"));
        assert!(frame.contains("model.alias"));
        assert!(frame.contains("Preview · not committed"));
        diff.toggle_details();
        let details = diff.render(100, 24, Locale::English);
        assert!(details.contains("Candidate forward passed"));
        assert!(details.contains("Candidate mode · CLI train"));
        assert!(details.contains("model.alias"));
        assert!(details.contains("same sample is not established"));
        assert!(details.contains("inspected snapshot has not been matched"));
        let mut mismatch =
            Diff::live_preview(&report, Some(&before), "model.head", Locale::English, false);
        let narrow_frame = mismatch.render(40, 24, Locale::English);
        assert!(narrow_frame.contains("Mode mismatch"), "{narrow_frame}");
        mismatch.toggle_details();
        let details = mismatch.render(100, 24, Locale::English);
        assert!(
            details.contains("Mode mismatch · requested mode not validated"),
            "{details}"
        );

        let rejected: crate::cli::Report = serde_json::from_value(json!({
            "protocol_version":1,"operation":"edit","status":"error",
            "diagnostics":[{"code":"EDIT_REJECTED","message":"bad candidate"}]}))
        .unwrap();
        let rejected = Diff::live_preview(
            &rejected,
            Some(&before),
            "model.head",
            Locale::English,
            false,
        );
        let frame = rejected.render(100, 24, Locale::English);
        assert!(frame.contains("Preview rejected"));
        assert!(frame.contains("in_features  ?"));
        let mut rejected = rejected;
        rejected.toggle_details();
        assert!(
            rejected
                .render(100, 24, Locale::English)
                .contains("Candidate forward unknown")
        );
    }

    #[test]
    fn long_mixed_width_paths_can_be_panned_on_both_aligned_sides() {
        let path = format!("model.{}", "很长的模块_".repeat(12));
        let mut diff = Diff::new(
            "target".into(),
            "Preview".into(),
            vec![format!("~ {path}原始端")],
            vec![format!("~ {path}候选端")],
            vec!["evidence".into()],
        );
        let initial = diff.render(40, 10, Locale::English);
        assert!(!initial.contains("原始端"));
        for _ in 0..100 {
            diff.scroll_horizontal(40, true);
        }
        let position = diff.horizontal_offset;
        assert!(position > 0);
        let before = diff.render(40, 10, Locale::English);
        assert!(before.contains("原始端"), "{before}");
        diff.change_side();
        let after = diff.render(40, 10, Locale::English);
        assert!(after.contains("候选端"), "{after}");
        assert_eq!(diff.horizontal_offset, position);
        let wide = diff.render(100, 10, Locale::English);
        assert!(wide.contains("原始端") && wide.contains("候选端"), "{wide}");
        assert_eq!(diff.horizontal_offset, position);
        diff.toggle_details();
        diff.scroll_horizontal(40, false);
        diff.toggle_details();
        assert_eq!(diff.horizontal_offset, position);
        for _ in 0..100 {
            diff.scroll_horizontal(40, false);
        }
        assert_eq!(diff.horizontal_offset, 0);
    }

    #[test]
    fn sequential_relations_are_explicit_and_missing_relations_remain_unknown() {
        let old = vec![
            json!({"path":"model", "type":"Sequential", "sequential_children":["a","b","c"]}),
            json!({"path":"model.a", "type":"Linear"}),
            json!({"path":"model.b", "type":"Linear"}),
            json!({"path":"model.c", "type":"ReLU"}),
        ];
        let mut changed = old.clone();
        changed[0]["sequential_children"] = json!(["a", "c", "b"]);
        let report: crate::cli::Report = serde_json::from_value(json!({"protocol_version":1,"operation":"edit","status":"success","data":{"committed":false,"modules":changed}})).unwrap();
        let mut diff = Diff::live_preview(&report, Some(&old), "model.b", Locale::English, false);
        let old_edge = diff
            .before
            .iter()
            .position(|row| row.contains("model.a → model.b"))
            .unwrap();
        assert!(diff.before[old_edge].starts_with('?'));
        assert_eq!(diff.after[old_edge], "? Not declared in report");
        let new_edge = diff
            .after
            .iter()
            .position(|row| row.contains("model.a → model.c"))
            .unwrap();
        assert_eq!(diff.before[new_edge], "? Not declared in report");
        assert!(diff.after[new_edge].starts_with('?'));
        diff.toggle_diagram();
        let left = diff.render(40, 24, Locale::English);
        assert!(
            left.contains("? ╭─ model.a") && left.contains("╰─▶ model.b"),
            "{left}"
        );
        assert!(left.contains("model.b [target]"), "{left}");
        diff.change_side();
        let right = diff.render(40, 24, Locale::English);
        assert!(
            right.contains("? Not declared in report") && right.contains("? relation unknown"),
            "{right}"
        );
        diff.scroll(40, 8, true, true);
        let position = diff.diagram.as_ref().unwrap().offset;
        assert!(position > 0);
        diff.toggle_diagram();
        diff.scroll(40, 8, true, true);
        diff.toggle_diagram();
        assert_eq!(diff.diagram.as_ref().unwrap().offset, position);
        diff.toggle_details();
        diff.toggle_details();
        assert!(diff.diagram_open);
        assert_eq!(diff.diagram.as_ref().unwrap().offset, position);
        let missing_before = Diff::live_preview(&report, None, "model.b", Locale::English, false);
        assert!(
            missing_before
                .before
                .iter()
                .any(|row| row == "? Report unavailable or too large")
        );
        let equal_report: crate::cli::Report = serde_json::from_value(json!({"protocol_version":1,"operation":"edit","status":"success","data":{"modules":old.clone()}})).unwrap();
        let mut equal =
            Diff::live_preview(&equal_report, Some(&old), "model.b", Locale::English, false);
        assert!(
            equal
                .before
                .iter()
                .any(|row| row.starts_with("= model.a → model.b"))
        );
        assert!(
            equal
                .after
                .iter()
                .any(|row| row.starts_with("= model.a → model.b"))
        );
        equal.toggle_diagram();
        let wide = equal.render(100, 24, Locale::English);
        assert_eq!(wide.matches("= ╭─ model.a").count(), 2, "{wide}");
        assert_eq!(wide.matches("╰─▶ model.b").count(), 2, "{wide}");
        equal.attach_receipt(
            "Historical · commit confirmed".into(),
            "Recheck · CLI reports success".into(),
            "CLI outcome\nCompleted",
            Locale::English,
        );
        let receipt = equal.render(100, 24, Locale::English);
        assert!(
            receipt.contains("Historical preview")
                && receipt.contains("Historical · commit confirmed")
                && receipt.contains("Recheck · CLI reports success"),
            "{receipt}"
        );
        assert!(receipt.contains("= ╭─ model.a"), "{receipt}");
        equal.toggle_details();
        let evidence = equal.render(100, 24, Locale::English);
        assert!(
            evidence.contains("Historical · commit confirmed")
                && evidence.contains("Recheck · CLI reports success")
                && evidence.contains("CLI outcome"),
            "{evidence}"
        );
        equal.toggle_details();
        assert!(equal.diagram_open);
        let mut legacy = old.clone();
        legacy[0]
            .as_object_mut()
            .unwrap()
            .remove("sequential_children");
        assert!(declared_relations(Some(&legacy)).unwrap().is_empty());
        let legacy_numeric = vec![
            json!({"path":"model", "type":"Sequential"}),
            json!({"path":"model.0", "type":"Linear"}),
            json!({"path":"model.1", "type":"ReLU"}),
        ];
        assert!(
            declared_relations(Some(&legacy_numeric))
                .unwrap()
                .is_empty()
        );
        legacy[0]["sequential_children"] = json!(["a", "a", "c"]);
        assert!(declared_relations(Some(&legacy)).unwrap().is_empty());
        let too_large = vec![json!({}); 4097];
        assert!(declared_relations(Some(&too_large)).is_none());
    }

    #[test]
    fn declaration_diagram_preserves_its_own_pan_and_both_long_paths() {
        let first = format!("{}FIRST_END", "长模块".repeat(15));
        let head = format!("{}TARGET_END", "很长层".repeat(15));
        let target = format!("model.{head}");
        let modules = vec![
            json!({"path":"model", "type":"Sequential", "sequential_children":[first,head]}),
            json!({"path":format!("model.{first}"), "type":"Linear"}),
            json!({"path":target, "type":"Linear"}),
        ];
        let report: crate::cli::Report = serde_json::from_value(json!({"protocol_version":1,"operation":"edit","status":"success","data":{"modules":modules}})).unwrap();
        let mut diff = Diff::live_preview(
            &report,
            report.data["modules"].as_array().map(Vec::as_slice),
            &target,
            Locale::English,
            false,
        );
        diff.toggle_diagram();
        for _ in 0..100 {
            diff.scroll_horizontal(40, true);
        }
        let horizontal = diff.diagram.as_ref().unwrap().horizontal;
        let before = diff.render(40, 24, Locale::English);
        assert!(
            before.contains("FIRST_END") && before.contains("TARGET_END"),
            "{before}"
        );
        diff.change_side();
        let candidate = diff.render(40, 24, Locale::English);
        assert!(
            candidate.contains("FIRST_END") && candidate.contains("TARGET_END"),
            "{candidate}"
        );
        diff.toggle_diagram();
        assert_eq!(diff.horizontal_offset, 0);
        diff.toggle_diagram();
        assert_eq!(diff.diagram.as_ref().unwrap().horizontal, horizontal);
        let wide = diff.render(100, 24, Locale::English);
        assert_eq!(wide.matches("TARGET_END").count(), 2, "{wide}");
    }
}
