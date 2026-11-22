use serde_json::Value;

const PAGE_SIZE: usize = 10;

#[derive(Clone, Debug, Default)]
pub struct RunsFilter {
    pub query: String,
    pub status: Option<String>,
    pub kept: Option<bool>,
    pub page: usize,
}

impl RunsFilter {
    pub fn filtered<'a>(&self, runs: &'a [Value]) -> Vec<&'a Value> {
        let query = self.query.to_lowercase();
        runs.iter()
            .filter(|run| {
                let status_matches = match self.status.as_deref() {
                    Some("active") => {
                        matches!(run["status"].as_str(), Some("running" | "stopping"))
                    }
                    Some(expected) => run["status"] == expected,
                    None => true,
                };
                let query_matches = query.is_empty()
                    || run["id"]
                        .as_str()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains(&query)
                    || run["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains(&query);
                let kept_matches = self.kept.is_none_or(|kept| run["kept"] == kept);
                status_matches && query_matches && kept_matches
            })
            .collect()
    }

    pub fn pages(&self, count: usize) -> usize {
        count.div_ceil(PAGE_SIZE).max(1)
    }

    pub fn next(&mut self, count: usize) {
        self.page = (self.page + 1).min(self.pages(count).saturating_sub(1));
    }

    pub fn previous(&mut self) {
        self.page = self.page.saturating_sub(1);
    }

    pub fn select<'a>(&self, input: &str, filtered: &[&'a Value]) -> Option<&'a Value> {
        if let Ok(index) = input.trim().parse::<usize>() {
            if index == 0 {
                return None;
            }
            return filtered.get(self.page * PAGE_SIZE + index - 1).copied();
        }
        filtered
            .iter()
            .copied()
            .find(|run| run["id"].as_str() == Some(input.trim()))
    }
}

pub fn render(runs: &[Value], filter: &RunsFilter) -> String {
    let filtered = filter.filtered(runs);
    let page_count = filter.pages(filtered.len());
    let start = (filter.page * PAGE_SIZE).min(filtered.len());
    let end = (start + PAGE_SIZE).min(filtered.len());
    let mut lines = vec![format!(
        "运行记录 · {} 条匹配 · 第 {}/{} 页",
        filtered.len(),
        filter.page + 1,
        page_count
    )];
    if !filter.query.is_empty() {
        lines.push(format!("名称/ID：{}", filter.query));
    }
    if let Some(status) = filter.status.as_deref() {
        lines.push(format!("状态：{}", status_label(status)));
    }
    if let Some(kept) = filter.kept {
        lines.push(format!("保留：{}", if kept { "是" } else { "否" }));
    }
    if filtered.is_empty() {
        lines.push("没有匹配的运行记录。".into());
        return lines.join("\n");
    }
    for (index, run) in filtered[start..end].iter().enumerate() {
        let name = run["name"].as_str().filter(|name| !name.is_empty());
        let operation = operation_label(run["operation"].as_str().unwrap_or(""));
        let summary = &run["summary"];
        let progress = if run["live_progress"].is_object() {
            &run["live_progress"]
        } else {
            summary
        };
        let recovery = recovery_label(run);
        lines.push(format!(
            "{}. {} · {} · {} · {} · {}/{} 步 · Loss {}",
            index + 1,
            name.unwrap_or(operation),
            age_label(run["started_at_ms"].as_u64()),
            status_label(run["status"].as_str().unwrap_or("unknown")),
            recovery,
            progress["step"]
                .as_u64()
                .or_else(|| progress["completed_steps"].as_u64())
                .map_or_else(|| "?".into(), |value| value.to_string()),
            progress["total"]
                .as_u64()
                .or_else(|| progress["requested_steps"].as_u64())
                .map_or_else(|| "?".into(), |value| value.to_string()),
            progress["mean_loss"]
                .as_f64()
                .map_or_else(|| "未记录".into(), |value| format!("{value:.5}"))
        ));
        lines.push(format!("   ID：{}", run["id"].as_str().unwrap_or("未知")));
    }
    lines.join("\n")
}

fn recovery_label(run: &Value) -> &'static str {
    if run["operation"] == "train"
        && run["checkpoint"].is_object()
        && matches!(
            run["status"].as_str(),
            Some("completed" | "failed" | "interrupted")
        )
    {
        "可检查恢复点"
    } else if run["status"] == "unknown" {
        "恢复状态待确认"
    } else {
        "无已确认恢复点"
    }
}

fn operation_label(operation: &str) -> &'static str {
    match operation {
        "training_probe" => "训练步骤试跑",
        "train" => "正式训练",
        "evaluate" => "模型评估",
        _ => "运行",
    }
}

fn status_label(status: &str) -> &'static str {
    match status {
        "active" => "活动",
        "running" => "运行中",
        "stopping" => "正在停止",
        "completed" => "已完成",
        "failed" => "失败待处理",
        "interrupted" => "已中断",
        _ => "状态未知",
    }
}

fn age_label(started_at_ms: Option<u64>) -> String {
    let Some(started_at_ms) = started_at_ms else {
        return "时间未知".into();
    };
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let minutes = now_ms.saturating_sub(started_at_ms) / 60_000;
    match minutes {
        0 => "刚刚".into(),
        1..=59 => format!("{minutes} 分钟前"),
        60..=1439 => format!("{} 小时前", minutes / 60),
        _ => format!("{} 天前", minutes / 1440),
    }
}

#[cfg(test)]
mod tests {
    use super::{RunsFilter, render};
    use serde_json::json;

    #[test]
    fn filtering_and_selection_preserve_name_status_and_id_context() {
        let runs = vec![
            json!({"id":"run-1","name":"Baseline","status":"completed","kept":true,"operation":"train","started_at_ms":1,"summary":{"completed_steps":4,"requested_steps":8,"mean_loss":0.2}}),
            json!({"id":"run-2","name":"Latest","status":"failed","kept":false,"operation":"training_probe"}),
        ];
        let filter = RunsFilter {
            query: "base".into(),
            status: Some("completed".into()),
            kept: Some(true),
            page: 0,
        };
        let filtered = filter.filtered(&runs);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filter.select("1", &filtered).unwrap()["id"], "run-1");
        assert_eq!(
            filter.select("run-1", &filtered).unwrap()["name"],
            "Baseline"
        );
        let output = render(&runs, &filter);
        assert!(output.contains("Baseline"));
        assert!(!output.contains("失败待处理"));
    }

    #[test]
    fn unknown_status_and_missing_metrics_are_not_rendered_as_success_or_zero() {
        let runs = vec![json!({"id":"run-3","status":"unknown","operation":"train"})];
        let output = render(&runs, &RunsFilter::default());
        assert!(output.contains("状态未知"));
        assert!(output.contains("Loss 未记录"));
    }
}
