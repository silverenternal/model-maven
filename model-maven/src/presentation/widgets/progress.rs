use serde_json::Value;
use std::time::Duration;

/// Render only progress facts supplied by the caller. Missing totals never produce
/// percentages or estimated completion times.
pub fn render(event: &Value, elapsed: Option<Duration>) -> Vec<String> {
    let state = event["state"].as_str().unwrap_or("unknown");
    let stage = event["stage"].as_str().unwrap_or("unknown");
    let mut lines = vec![
        state_label(state).to_owned(),
        format!("阶段：{}", stage_label(stage)),
    ];

    let completed = first_number(event, &["completed", "completed_steps", "step"]);
    let total = first_number(event, &["total", "requested_steps", "budget"])
        .or_else(|| event["budget"]["total_steps"].as_u64());
    match (completed, total.filter(|total| *total > 0)) {
        (Some(completed), Some(total)) => {
            let percentage = completed.saturating_mul(100) / total;
            if let Some(loss) = event["mean_loss"].as_f64() {
                lines.push(format!(
                    "最近进度：{completed}/{total} 步 · Loss {loss:.6} · {percentage}%"
                ));
            } else {
                lines.push(format!("最近进度：{completed}/{total} 步 · {percentage}%"));
            }
        }
        (Some(completed), _) => lines.push(format!("已处理：{completed} · 总量未上报")),
        (None, _) => lines.push("进度：暂未上报；不会将未知进度显示为 0。".into()),
    }

    if let Some(message) = event["message"]
        .as_str()
        .filter(|message| !message.is_empty())
    {
        lines.push(format!("当前反馈：{message}"));
    }
    if let Some(elapsed) = elapsed {
        lines.push(format!("已用时：{}", format_elapsed(elapsed)));
    }
    lines.push(format!(
        "停止能力：{}",
        match event["cancellation"].as_str().unwrap_or("unknown") {
            "supported" => "可请求合作停止，结果以运行状态为准",
            "unsupported" => "当前入口不支持合作停止",
            _ => "尚未确认；请勿假定已停止",
        }
    ));
    lines
}

fn first_number(event: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| event[key].as_u64())
}

fn state_label(state: &str) -> &'static str {
    match state {
        "running" => "正在执行",
        "preparing" => "正在准备",
        "completed" => "已完成",
        "failed" => "执行失败",
        "stopping" => "正在请求停止",
        "cancelled" => "已取消",
        _ => "等待执行反馈",
    }
}

fn stage_label(stage: &str) -> &str {
    match stage {
        "initializing" => "初始化",
        "loading_data" => "加载数据",
        "training" => "训练",
        "evaluating" => "评估",
        "checkpointing" => "保存检查点",
        "unknown" => "阶段尚未上报",
        other => other,
    }
}

fn format_elapsed(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 3_600 {
        format!(
            "{} 小时 {} 分 {} 秒",
            seconds / 3_600,
            seconds % 3_600 / 60,
            seconds % 60
        )
    } else if seconds >= 60 {
        format!("{} 分 {} 秒", seconds / 60, seconds % 60)
    } else {
        format!("{seconds} 秒")
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn unknown_totals_do_not_create_percentages_or_eta() {
        let text = render(
            &json!({"state":"running","stage":"training","completed_steps":4}),
            Some(Duration::from_secs(61)),
        )
        .join("\n");
        assert!(text.contains("正在执行"));
        assert!(text.contains("阶段：训练"));
        assert!(text.contains("已处理：4 · 总量未上报"));
        assert!(text.contains("已用时：1 分 1 秒"));
        assert!(!text.contains('%'));
        assert!(!text.contains("预计"));
    }

    #[test]
    fn progress_reports_real_budget_stage_and_cancellation_capability() {
        let text = render(
            &json!({
                "state":"running",
                "stage":"checkpointing",
                "completed":3,
                "total":8,
                "message":"checkpoint 写入中",
                "cancellation":"supported"
            }),
            Some(Duration::from_secs(7)),
        )
        .join("\n");
        assert!(text.contains("保存检查点"));
        assert!(text.contains("3/8 步 · 37%"));
        assert!(text.contains("checkpoint 写入中"));
        assert!(text.contains("合作停止"));
    }
}
