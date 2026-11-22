use serde_json::Value;

pub fn render(data: &Value) -> Vec<String> {
    let record = &data["record"];
    let status = record["status"].as_str().unwrap_or("unknown");
    let mut lines = vec![
        format!("后台运行 · {}", status_label(status)),
        format!("运行 ID：{}", data["run_id"].as_str().unwrap_or("未知")),
    ];
    if record["heartbeat_status"].is_string() {
        lines.push(format!(
            "监督心跳：{}",
            match record["heartbeat_status"].as_str().unwrap_or("unknown") {
                "fresh" => "正常",
                "stale" => "已过期",
                "identity_mismatch" => "身份不匹配",
                "starting_or_unavailable" => "尚未可用",
                _ => "状态未知",
            }
        ));
    }
    if let Some(progress) = data["progress"].as_array().and_then(|items| items.last()) {
        let mut progress_event = progress.clone();
        progress_event["state"] = Value::String(status.to_owned());
        progress_event["completed"] = progress["step"].clone();
        progress_event["cancellation"] = Value::String("unknown".into());
        lines.extend(crate::presentation::widgets::progress::render(
            &progress_event,
            elapsed_since(record, status),
        ));
        lines.push(format!(
            "最近更新：{}",
            progress["at_ms"]
                .as_u64()
                .map_or_else(|| "时间未上报".into(), relative_time)
        ));
    } else {
        lines.push("阶段：尚未上报".into());
        lines.push("尚未收到进度事件；不会将缺少的步数解释为 0。".into());
        lines.extend(crate::presentation::widgets::progress::render(
            &serde_json::json!({"state":status,"stage":"unknown"}),
            elapsed_since(record, status),
        ));
    }
    if let Some(note) = record["status_note"].as_str() {
        lines.push(format!("状态说明：{note}"));
    }
    if let Some(log) = data["log_tail"]
        .as_str()
        .filter(|log| !log.trim().is_empty())
    {
        lines.push("\n最近日志：".into());
        lines.extend(
            log.lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(str::to_owned),
        );
    }
    if matches!(status, "unknown" | "interrupted" | "failed") {
        lines.push(
            "下一步：检查运行详情、最近 checkpoint 和 worker.log；不要假定任务仍在运行。".into(),
        );
    }
    lines
}

fn status_label(status: &str) -> &'static str {
    match status {
        "running" => "运行中",
        "stopping" => "正在停止",
        "completed" => "已完成",
        "failed" => "失败待处理",
        "interrupted" => "已中断",
        _ => "状态未知",
    }
}

fn elapsed_since(record: &Value, status: &str) -> Option<std::time::Duration> {
    let started_at_ms = record["started_at_ms"].as_u64()?;
    if let Some(finished_at_ms) = record["finished_at_ms"].as_u64() {
        return Some(std::time::Duration::from_millis(
            finished_at_ms.saturating_sub(started_at_ms),
        ));
    }
    if !matches!(status, "running" | "stopping") {
        return None;
    }
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;
    Some(std::time::Duration::from_millis(
        now_ms.saturating_sub(started_at_ms),
    ))
}

fn relative_time(at_ms: u64) -> String {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let seconds = now_ms.saturating_sub(at_ms) / 1_000;
    match seconds {
        0 => "刚刚".into(),
        1..=59 => format!("{seconds} 秒前"),
        60..=3_599 => format!("{} 分钟前", seconds / 60),
        3_600..=86_399 => format!("{} 小时前", seconds / 3_600),
        _ => format!("{} 天前", seconds / 86_400),
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn stale_or_missing_progress_is_never_shown_as_running_zero() {
        let lines = render(&json!({
            "run_id":"run-stale",
            "record":{"status":"unknown","heartbeat_status":"stale","status_note":"心跳已过期"},
            "progress":[],"log_tail":"worker stopped"
        }));
        let text = lines.join("\n");
        assert!(text.contains("后台运行 · 状态未知"));
        assert!(text.contains("监督心跳：已过期"));
        assert!(text.contains("不会将缺少的步数解释为 0"));
        assert!(text.contains("不要假定任务仍在运行"));
    }
}
