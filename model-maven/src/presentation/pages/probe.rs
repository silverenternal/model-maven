use serde_json::Value;

#[derive(Clone, Debug)]
pub struct ProbeCallPage {
    page: usize,
    page_size: usize,
}

impl Default for ProbeCallPage {
    fn default() -> Self {
        Self {
            page: 0,
            page_size: 40,
        }
    }
}

impl ProbeCallPage {
    pub fn page(&self) -> usize {
        self.page
    }

    pub fn page_count(&self, calls: &[Value]) -> usize {
        calls.len().div_ceil(self.page_size).max(1)
    }

    pub fn visible_calls<'a>(&self, calls: &'a [Value]) -> &'a [Value] {
        let pages = self.page_count(calls);
        let page = self.page.min(pages - 1);
        let start = page * self.page_size;
        let end = start.saturating_add(self.page_size).min(calls.len());
        calls.get(start..end).unwrap_or_default()
    }

    pub fn next(&mut self, calls: &[Value]) {
        let last = self.page_count(calls).saturating_sub(1);
        self.page = (self.page + 1).min(last);
    }

    pub fn previous(&mut self) {
        self.page = self.page.saturating_sub(1);
    }

    pub fn select(&self, input: &str, calls: &[Value]) -> Option<usize> {
        let index = input.parse::<usize>().ok()?.checked_sub(1)?;
        let page = self.page.min(self.page_count(calls) - 1);
        let start = page.checked_mul(self.page_size)?;
        if index < start || index >= start + self.visible_calls(calls).len() {
            return None;
        }
        (index < calls.len()).then_some(index)
    }
}

pub fn render_probe_context(
    report: &Value,
    model: Option<&str>,
    batch: Option<&str>,
) -> Vec<String> {
    let calls = report["calls"].as_array().map_or(0, Vec::len);
    let mut lines = vec![
        "前向证据 · 仅本次执行观测，不代表所有动态分支".into(),
        format!("模型：{}", model.unwrap_or("未知入口")),
        format!("批次：{}", batch.unwrap_or("未知批次")),
        "样本范围：仅本次批次入口返回的数据；不推断整个数据集的覆盖情况。".into(),
        "未观测到的分支保持未知，不视为验证通过。".into(),
        format!(
            "模式：{} · 保留调用记录：{calls} · 观测模块调用：{}",
            report["observed_mode"].as_str().unwrap_or("未知"),
            report["calls_observed"]
                .as_u64()
                .map_or_else(|| "未知".into(), |count| count.to_string())
        ),
    ];
    if report["trace_truncated"] == true {
        lines.push("调用轨迹已达到采集上限；缺失调用不能解释为未执行。".into());
    }
    if let Some(coverage) = report["coverage"].as_str() {
        lines.push(format!("覆盖范围：{coverage}"));
    }
    lines
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeFailureAction {
    ViewSource,
    InspectLastCall,
    ChangeBatch,
    Retry,
    Back,
}

pub fn resolve_failure_action(input: &str) -> Option<ProbeFailureAction> {
    match input.trim().to_ascii_lowercase().as_str() {
        "1" | "source" | "源码" => Some(ProbeFailureAction::ViewSource),
        "2" | "last" | "调用" => Some(ProbeFailureAction::InspectLastCall),
        "3" | "batch" | "批次" => Some(ProbeFailureAction::ChangeBatch),
        "4" | "retry" | "重试" => Some(ProbeFailureAction::Retry),
        "back" | "b" | "返回" | "q" => Some(ProbeFailureAction::Back),
        _ => None,
    }
}

pub fn render_probe_failure_context(
    report: &Value,
    model: Option<&str>,
    batch: Option<&str>,
) -> Vec<String> {
    let mut lines = vec![
        "前向失败 · 没有通过验证".into(),
        format!("模型：{}", model.unwrap_or("未知入口")),
        format!("批次：{}", batch.unwrap_or("未知批次")),
        "模型和本次选择已保留；查看观测、调整批次或修复源码后可重试。".into(),
    ];
    let location = &report["failure_location"];
    if let Some(call) = location["last_observed_module_call"].as_object() {
        let paths = call["paths"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .unwrap_or_default();
        lines.push(format!(
            "最后观测调用：{} · {} · 本模块第 {} 次",
            call.get("module_type")
                .and_then(Value::as_str)
                .unwrap_or("未知模块"),
            paths,
            call.get("invocation_index")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        ));
    } else {
        lines.push("尚未观测到模块调用；失败发生在模型构造或批次准备阶段。".into());
    }
    if let Some(frames) = location["project_stack"].as_array() {
        for frame in frames {
            lines.push(format!(
                "项目源码：{}:{} · {}",
                frame["file"].as_str().unwrap_or("未知文件"),
                frame["line"],
                frame["function"].as_str().unwrap_or("未知函数")
            ));
        }
    }
    let has_last_call = location["last_observed_module_call"].is_object();
    lines.push("推荐下一步：".into());
    lines.extend(crate::presentation::widgets::recovery_actions::render(&[
        serde_json::json!({"label":"查看项目源码","availability":"available"}),
        serde_json::json!({
            "label":"查看最后调用输入",
            "availability":if has_last_call { "available" } else { "unsupported" },
            "reason":if has_last_call { "" } else { "失败发生在模块前向前，没有调用输入可查看" }
        }),
        serde_json::json!({"label":"更换批次后重新前向","availability":"available"}),
    ]));
    lines.push("操作：1 查看源码 · 2 查看调用输入 · 3 更换批次 · 4 修复后重试 · back 返回".into());
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn call_selection_is_scoped_to_the_visible_page() {
        let calls = (0..83).map(|i| json!({"call_id":i})).collect::<Vec<_>>();
        let mut page = ProbeCallPage::default();
        assert_eq!(page.page_count(&calls), 3);
        assert_eq!(page.select("40", &calls), Some(39));
        page.next(&calls);
        assert_eq!(page.select("41", &calls), Some(40));
        assert_eq!(page.select("1", &calls), None);
        assert_eq!(page.select("80", &calls), Some(79));
        page.previous();
        assert_eq!(page.select("1", &calls), Some(0));
    }

    #[test]
    fn failed_probe_preserves_recoverable_choices_without_claiming_success() {
        let report = json!({"failure_location":{
            "last_observed_module_call":{"paths":["model.encoder"],"module_type":"Linear","invocation_index":1},
            "project_stack":[{"file":"model.py","line":42,"function":"forward"}]
        }});
        let page =
            render_probe_failure_context(&report, Some("toy:build"), Some("toy:batch")).join("\n");
        assert!(page.contains("前向失败 · 没有通过验证"));
        assert!(page.contains("model.py:42"));
        assert!(page.contains("模型和本次选择已保留"));
        assert!(page.contains("1. 查看项目源码 · 可用"));
        assert!(page.contains("2. 查看最后调用输入 · 可用"));
        assert_eq!(
            resolve_failure_action("3"),
            Some(ProbeFailureAction::ChangeBatch)
        );
        assert_eq!(
            resolve_failure_action("1"),
            Some(ProbeFailureAction::ViewSource)
        );
        assert_eq!(resolve_failure_action("4"), Some(ProbeFailureAction::Retry));
    }

    #[test]
    fn probe_actions_mark_unavailable_call_inspection_when_forward_never_started() {
        let report = json!({"failure_location":{"project_stack":[]}});
        let text = render_probe_failure_context(&report, Some("toy:build"), None).join("\n");
        assert!(text.contains("2. 查看最后调用输入 · 当前不支持"));
        assert!(text.contains("没有调用输入可查看"));
    }

    #[test]
    fn probe_context_states_sample_scope_partial_coverage_and_truncation() {
        let report = json!({
            "calls": [{"call_id":1}],
            "calls_observed": 257,
            "trace_truncated": true,
            "observed_mode": "eval",
            "coverage": "仅本批次已采集的前向；未反向、未更新、不证明所有分支。"
        });
        let context =
            render_probe_context(&report, Some("toy:build"), Some("toy:batch")).join("\n");
        assert!(context.contains("仅本次批次入口返回的数据"));
        assert!(context.contains("未观测到的分支保持未知"));
        assert!(context.contains("保留调用记录：1 · 观测模块调用：257"));
        assert!(context.contains("调用轨迹已达到采集上限"));
        assert!(context.contains("不证明所有分支"));
    }
}
