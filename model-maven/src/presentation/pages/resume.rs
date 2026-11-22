use serde_json::Value;

pub fn render(runs: &[Value]) -> String {
    let candidates = runs
        .iter()
        .filter(|run| run["operation"] == "train")
        .collect::<Vec<_>>();
    let mut lines = vec!["续训来源 · 正式训练记录".to_owned()];
    if candidates.is_empty() {
        lines.push("没有正式训练记录。先完成一次正式训练，再从它的 checkpoint 续训。".into());
        return lines.join("\n");
    }
    for (index, run) in candidates.iter().enumerate() {
        let id = run["id"].as_str().unwrap_or("未知 ID");
        let name = run["name"]
            .as_str()
            .filter(|name| !name.is_empty())
            .unwrap_or("未命名");
        let status = status_label(run["status"].as_str().unwrap_or("unknown"));
        let completed = number(run["summary"]["completed_steps"].as_u64());
        let requested = number(
            run["summary"]["requested_steps"]
                .as_u64()
                .or_else(|| run["requested_steps"].as_u64()),
        );
        let checkpoint = if run["checkpoint"].is_object() {
            format!(
                "checkpoint 第 {} 步 · {}",
                number(run["checkpoint"]["completed_steps"].as_u64()),
                run["checkpoint"]["sha256"]
                    .as_str()
                    .map(short_hash)
                    .unwrap_or_else(|| "摘要未知".into())
            )
        } else if run["checkpoint_released"] == true {
            "checkpoint 已释放".into()
        } else {
            "没有已确认 checkpoint".into()
        };
        lines.push(format!(
            "{}. {} · {} · {status} · 训练 {completed}/{requested} 步",
            index + 1,
            name,
            age_label(run["started_at_ms"].as_u64())
        ));
        lines.push(format!("   {checkpoint} · ID：{id}"));
        if !resume_available(run) {
            lines.push(format!("   不可选择：{}", unavailable_reason(run)));
        }
    }
    lines.push("输入可续训记录的序号或完整 ID；输入 back 返回。".into());
    lines.join("\n")
}

pub fn resume_available(run: &Value) -> bool {
    run["operation"] == "train"
        && matches!(
            run["status"].as_str(),
            Some("completed" | "failed" | "interrupted")
        )
        && run["checkpoint"].is_object()
        && run["checkpoint_released"] != true
}

pub fn render_selected(record: &Value) -> Vec<String> {
    let metadata = &record["record"];
    let inputs = &record["inputs"];
    let mode = inputs["training_mode"]
        .as_str()
        .or_else(|| metadata["training_mode"].as_str())
        .unwrap_or("unknown");
    let checkpoint = &record["checkpoint"];
    let mut lines = vec![
        format!(
            "续训来源：{} · {}",
            metadata["name"].as_str().filter(|name| !name.is_empty()).unwrap_or("未命名"),
            metadata["id"].as_str().unwrap_or("未知 ID")
        ),
        format!("训练模式：{}", mode_label(mode)),
        format!(
            "本次续训运行的父记录：{}",
            metadata["id"].as_str().unwrap_or("未知 ID")
        ),
        format!(
            "恢复点：第 {} 步 · SHA-256 {}",
            number(checkpoint["completed_steps"].as_u64()),
            checkpoint["sha256"].as_str().map(short_hash).unwrap_or_else(|| "未知".into())
        ),
        "继承：模型、数据/批次、训练入口、模式、优化器、环境与 checkpoint 中的随机状态；本次只增加显式训练步数预算。".into(),
        "自定义 batch/step 或 external-loop 状态是否可恢复，要由 Python state_dict/load_state_dict 入口确认；缺失时工具会拒绝，除非你显式接受重置。".into(),
    ];
    if metadata["checkpoint_released"] == true {
        lines.push("此 checkpoint 已释放，不能续训。".into());
    }
    lines
}

fn mode_label(mode: &str) -> &'static str {
    match mode {
        "managed-step" => "CLI 管理优化步骤",
        "manual-step" => "用户管理优化步骤",
        "external-loop" => "用户管理完整循环",
        _ => "来源记录未说明",
    }
}

pub fn unavailable_reason(run: &Value) -> &'static str {
    if matches!(
        run["status"].as_str(),
        Some("running" | "stopping" | "unknown")
    ) {
        "任务仍活动或状态未知，不能安全恢复"
    } else if run["checkpoint_released"] == true {
        "checkpoint 已释放"
    } else if !run["checkpoint"].is_object() {
        "没有可验证的 checkpoint"
    } else {
        "运行状态不支持续训"
    }
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

fn number(value: Option<u64>) -> String {
    value.map_or_else(|| "未知".into(), |value| value.to_string())
}

fn short_hash(value: &str) -> String {
    value.chars().take(12).collect()
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
    use super::{render, render_selected, resume_available};
    use serde_json::json;

    #[test]
    fn only_terminal_training_runs_with_retained_checkpoints_are_selectable() {
        let runs = vec![
            json!({"id":"active","operation":"train","status":"running","checkpoint":{"completed_steps":4}}),
            json!({"id":"unknown","operation":"train","status":"unknown","checkpoint":{"completed_steps":4}}),
            json!({"id":"released","operation":"train","status":"failed","checkpoint":{"completed_steps":4},"checkpoint_released":true}),
            json!({"id":"no-checkpoint","operation":"train","status":"completed"}),
            json!({"id":"failed","name":"复杂模型实验","operation":"train","status":"failed","started_at_ms":0,"summary":{"completed_steps":4,"requested_steps":10},"checkpoint":{"completed_steps":4,"sha256":"abcdef1234567890"}}),
            json!({"id":"probe","operation":"training_probe","status":"failed","checkpoint":{"completed_steps":4}}),
        ];

        assert!(!resume_available(&runs[0]));
        assert!(!resume_available(&runs[1]));
        assert!(!resume_available(&runs[2]));
        assert!(!resume_available(&runs[3]));
        assert!(resume_available(&runs[4]));
        assert!(!resume_available(&runs[5]));

        let rendered = render(&runs).to_string();
        assert!(rendered.contains("复杂模型实验"), "{rendered}");
        assert!(rendered.contains("失败待处理"), "{rendered}");
        assert!(rendered.contains("checkpoint 第 4 步"), "{rendered}");
        assert!(
            rendered.contains("不可选择：任务仍活动或状态未知"),
            "{rendered}"
        );
        assert!(
            rendered.contains("不可选择：checkpoint 已释放"),
            "{rendered}"
        );
    }

    #[test]
    fn selected_resume_explains_parent_checkpoint_and_custom_state_limit() {
        let selected = json!({
            "record":{"id":"run-parent","name":"ResNet 实验","training_mode":"external-loop"},
            "inputs":{"training_mode":"external-loop"},
            "checkpoint":{"completed_steps":12,"sha256":"0123456789abcdef"}
        });

        let rendered = render_selected(&selected).join("\n");
        assert!(rendered.contains("父记录：run-parent"), "{rendered}");
        assert!(rendered.contains("恢复点：第 12 步"), "{rendered}");
        assert!(rendered.contains("0123456789ab"), "{rendered}");
        assert!(rendered.contains("external-loop"), "{rendered}");
        assert!(
            rendered.contains("state_dict/load_state_dict"),
            "{rendered}"
        );
        assert!(!rendered.contains("反向传播已验证"), "{rendered}");
    }
}
