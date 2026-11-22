use serde_json::Value;

pub fn render(data: &Value) -> Vec<String> {
    let record = &data["record"];
    let inputs = &data["inputs"];
    let status = match record["status"].as_str().unwrap_or("unknown") {
        "running" => "运行中",
        "stopping" => "正在停止",
        "completed" => "已完成",
        "failed" => "失败待处理",
        "interrupted" => "已中断",
        _ => "状态未知",
    };
    let mut lines = vec![
        format!("运行详情 · {status}"),
        format!(
            "名称：{}",
            record["name"]
                .as_str()
                .filter(|name| !name.is_empty())
                .unwrap_or("未命名")
        ),
        format!("ID：{}", record["id"].as_str().unwrap_or("未知")),
        format!(
            "类型：{}",
            operation_label(record["operation"].as_str().unwrap_or(""))
        ),
    ];
    if let Some(parent) = record["parent_run_id"].as_str() {
        lines.push(format!("来源运行：{parent}"));
    }
    for (label, key) in [
        ("模型", "model"),
        ("批次", "batch"),
        ("训练步骤", "training_step"),
        ("完整训练循环", "training_loop"),
    ] {
        if let Some(value) = inputs[key].as_str() {
            lines.push(format!("{label}：{value}"));
        }
    }
    let summary = &record["summary"];
    if !summary.is_null() {
        lines.push(format!(
            "进度摘要：{}/{} 步 · Loss {} · 优化器更新 {} 次",
            optional_number(summary["completed_steps"].as_u64()),
            optional_number(summary["requested_steps"].as_u64()),
            summary["mean_loss"]
                .as_f64()
                .map_or_else(|| "未记录".into(), |value| format!("{value:.6}")),
            optional_number(summary["optimizer_updates"].as_u64())
        ));
    }
    if record["checkpoint_released"] == true {
        lines.push("恢复权重：已释放；本运行不能续训或导出权重。".into());
    } else if data["checkpoint"].is_object() {
        lines.push(format!(
            "最近恢复点：第 {} 步 · SHA-256 {}",
            optional_number(data["checkpoint"]["completed_steps"].as_u64()),
            data["checkpoint"]["sha256"].as_str().unwrap_or("未知")
        ));
    }
    if let Some(note) = record["status_note"].as_str() {
        lines.push(format!("状态说明：{note}"));
    }
    if let Some(diagnostics) = data["report"]["diagnostics"].as_array() {
        for diagnostic in diagnostics {
            lines.push(format!(
                "诊断：{} · {}",
                diagnostic["code"].as_str().unwrap_or("RUN_DIAGNOSTIC"),
                diagnostic["message"].as_str().unwrap_or("未提供说明")
            ));
            if let Some(hint) = diagnostic["hint"].as_str().filter(|hint| !hint.is_empty()) {
                lines.push(format!("下一步：{hint}"));
            }
        }
    }
    if inputs["redacted_fields"]
        .as_array()
        .is_some_and(|fields| !fields.is_empty())
    {
        lines.push("凭证字段已脱敏；完整字段列表可通过 JSON 查看。".into());
    }
    lines
}

fn operation_label(operation: &str) -> &'static str {
    match operation {
        "training_probe" => "训练步骤试跑",
        "train" => "正式训练",
        "evaluate" => "模型评估",
        _ => "运行",
    }
}

fn optional_number(value: Option<u64>) -> String {
    value.map_or_else(|| "未知".into(), |value| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn details_show_failure_recovery_and_redaction_without_fabricating_metrics() {
        let lines = render(&json!({
            "record":{"id":"run-x","name":"Baseline","operation":"train","status":"unknown","status_note":"监督身份未知","summary":{"completed_steps":3,"requested_steps":10,"mean_loss":null}},
            "inputs":{"model":"model:build","batch":"data:batch","redacted_fields":["$.training_params.api_key"]},
            "report":{"diagnostics":[{"code":"RUN_PROCESS_LOST","message":"监督进程已退出","hint":"检查最近 checkpoint"}]},
            "checkpoint":{"completed_steps":3,"sha256":"abc"}
        }));
        let text = lines.join("\n");
        assert!(text.contains("状态未知"));
        assert!(text.contains("Loss 未记录"));
        assert!(text.contains("监督身份未知"));
        assert!(text.contains("检查最近 checkpoint"));
        assert!(text.contains("凭证字段已脱敏"));
    }
}
