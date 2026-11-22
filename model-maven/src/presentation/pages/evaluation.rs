use crate::domain::Report;
use serde_json::Value;

/// Render evaluator-owned metrics and provenance without assuming a task type.
/// Only records from the explicit preview contract are described as samples.
pub fn render(report: &Report) -> Vec<String> {
    let data = &report.data;
    let evaluation = data
        .get("evaluation")
        .filter(|value| value.is_object())
        .unwrap_or(data);
    let checkpoint = evaluation
        .get("checkpoint")
        .or_else(|| data.get("checkpoint"));
    let coverage = evaluation.get("coverage").or_else(|| data.get("coverage"));
    let data_identity = data
        .get("provenance")
        .and_then(|provenance| provenance.get("data_identity"));
    let mut lines = vec![if report.success() {
        "评估结果".to_owned()
    } else {
        "评估未完成".to_owned()
    }];

    push_text(&mut lines, "评估运行", data.get("evaluation_run_id"));
    push_text(&mut lines, "来源训练运行", data.get("source_run_id"));
    push_text(&mut lines, "评估入口", evaluation.get("entrypoint"));
    push_text(&mut lines, "数据切分", evaluation.get("split"));
    push_text(
        &mut lines,
        "数据入口",
        data_identity.and_then(|identity| identity.get("entrypoint")),
    );
    if checkpoint
        .and_then(|checkpoint| checkpoint.get("completed_steps"))
        .and_then(Value::as_u64)
        .is_some()
    {
        lines.push(format!(
            "恢复权重：第 {} 步 checkpoint",
            checkpoint
                .and_then(|checkpoint| checkpoint["completed_steps"].as_u64())
                .unwrap_or(0)
        ));
    }
    if let Some(status) = coverage
        .and_then(|coverage| coverage.get("checkpoint_weights"))
        .and_then(Value::as_str)
    {
        lines.push(format!(
            "权重来源校验：{}",
            match status {
                "verified" => "已验证",
                "unknown" => "未知",
                other => other,
            }
        ));
    }

    let sample_count = evaluation.get("samples_evaluated");
    let sample_limit = evaluation.get("sample_limit");
    match sample_count.and_then(Value::as_u64) {
        Some(count) => {
            let mut line = format!("实际评估样本：{count}");
            if let Some(limit) = sample_limit.and_then(Value::as_u64) {
                line.push_str(&format!("（请求上限 {limit}）"));
            }
            lines.push(line);
        }
        None => lines.push("实际评估样本：未知（评估器未报告样本数）".into()),
    }
    if let Some(state) = coverage
        .and_then(|coverage| coverage.get("samples"))
        .and_then(Value::as_str)
        && state != "observed"
    {
        lines.push("样本覆盖：未知；不能据此推断已处理数量。".into());
    }

    let metrics = evaluation.get("metrics").and_then(Value::as_object);
    match metrics {
        Some(metrics) if !metrics.is_empty() => {
            lines.push("具名指标：".into());
            for (name, value) in metrics {
                lines.push(format!("  {name}：{}", scalar(value)));
            }
        }
        _ => lines.push("具名指标：没有可显示的评估指标。".into()),
    }

    let data_identity_state = coverage
        .and_then(|coverage| coverage.get("data_identity"))
        .and_then(Value::as_str);
    if let Some(identity) = data_identity_state {
        lines.push(format!("数据身份范围：{}", identity_label(identity)));
    } else {
        lines.push("数据身份范围：未知。".into());
    }
    if let Some(fingerprint) = data_identity
        .and_then(|identity| identity.get("data_fingerprint"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        lines.push(format!("已声明数据指纹：{}", shorten(fingerprint)));
    } else {
        lines.push("未声明数据内容指纹；来源仅由数据入口、参数、切分和 seed 描述。".into());
    }

    let previews = evaluation.get("sample_previews");
    if let Some(previews) = previews.filter(|value| value.is_object()) {
        let reported = previews.get("reported").and_then(Value::as_u64);
        let state = previews
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        lines.push(match reported {
            Some(count) => format!("结构化样本预览：{count} 条（{state}）；可从运行详情浏览。"),
            None => format!("结构化样本预览数量未知（{state}）；可从运行详情检查已报告记录。"),
        });
    } else if evaluation
        .get("custom_artifacts")
        .and_then(Value::as_object)
        .is_some_and(|artifacts| !artifacts.is_empty())
    {
        lines.push("评估器声明了附加产物；未将它们解释为样本或预测结果。".into());
    } else {
        lines.push("当前评估器未声明可浏览的样本或预测记录。".into());
    }

    for diagnostic in &report.diagnostics {
        lines.push(format!("{}：{}", diagnostic.code, diagnostic.message));
        if let Some(hint) = diagnostic.hint.as_deref().filter(|hint| !hint.is_empty()) {
            lines.push(format!("下一步：{hint}"));
        }
    }
    lines
}

fn push_text(lines: &mut Vec<String>, label: &str, value: Option<&Value>) {
    if let Some(value) = value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        lines.push(format!("{label}：{value}"));
    }
}

fn scalar(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(_) | Value::Bool(_) | Value::Null => value.to_string(),
        Value::Array(_) | Value::Object(_) => "格式无效".into(),
    }
}

fn identity_label(identity: &str) -> &str {
    match identity {
        "declared_fingerprint_and_inputs" => "已声明指纹与评估输入",
        "entrypoint_parameters_split_and_seed_only" => "仅入口、参数、切分和 seed（无内容指纹）",
        _ => identity,
    }
}

fn shorten(value: &str) -> String {
    const LIMIT: usize = 48;
    if value.chars().count() <= LIMIT {
        value.to_owned()
    } else {
        format!("{}…", value.chars().take(LIMIT - 1).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::domain::Report;
    use serde_json::json;

    #[test]
    fn evaluation_summary_keeps_provenance_metrics_and_observed_coverage() {
        let report = Report::ok(
            "evaluate",
            json!({
                "evaluation_run_id":"eval-123",
                "source_run_id":"train-456",
                "checkpoint":{"completed_steps":19},
                "evaluation": {
                    "entrypoint":"metrics:evaluate",
                    "split":"validation",
                    "sample_limit":50,
                    "samples_evaluated":37,
                    "metrics":{"f1/macro":0.8125,"loss":0.34},
                    "custom_artifacts":{"chart":"metrics.json"}
                },
                "provenance":{"data_identity":{"entrypoint":"dataset:build","data_fingerprint":"dataset-sha256-abc"}},
                "coverage":{"checkpoint_weights":"verified","samples":"observed","data_identity":"declared_fingerprint_and_inputs"}
            }),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("评估运行：eval-123"));
        assert!(text.contains("来源训练运行：train-456"));
        assert!(text.contains("第 19 步 checkpoint"));
        assert!(text.contains("实际评估样本：37（请求上限 50）"));
        assert!(text.contains("f1/macro：0.8125"));
        assert!(text.contains("已声明指纹与评估输入"));
        assert!(text.contains("未将它们解释为样本或预测结果"));
        assert!(!text.contains("分类准确率"));
    }

    #[test]
    fn evaluation_summary_advertises_only_reported_preview_records() {
        let report = Report::ok(
            "evaluate",
            json!({"evaluation":{"sample_previews":{"state":"reported","reported":2,"samples":[{"sample_id":"a"},{"sample_id":"b"}]}}}),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("结构化样本预览：2 条（reported）；可从运行详情浏览。"));
    }

    #[test]
    fn missing_sample_count_and_data_fingerprint_remain_unknown() {
        let report = Report::ok(
            "evaluate",
            json!({
                "evaluation": {
                    "split":"test",
                    "sample_limit":10,
                    "samples_evaluated":null,
                    "metrics":{"retrieval/recall@5":0.5},
                    "data_fingerprint":null
                },
                "source_run_id":"train-no-count",
                "coverage":{"samples":"unknown","data_identity":"entrypoint_parameters_split_and_seed_only"},
                "provenance":{"data_identity":{"entrypoint":"data:build","data_fingerprint":null}}
            }),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("实际评估样本：未知"));
        assert!(text.contains("样本覆盖：未知"));
        assert!(text.contains("仅入口、参数、切分和 seed（无内容指纹）"));
        assert!(text.contains("未声明数据内容指纹"));
        assert!(!text.contains("样本：0"));
    }

    #[test]
    fn failure_report_keeps_actionable_diagnostic() {
        let report = Report::error(
            "evaluate",
            "EVALUATION_RUN_UNAVAILABLE",
            "训练运行没有有效 checkpoint",
            "查看 m-model runs show <id>。",
        );
        let text = render(&report).join("\n");
        assert!(text.contains("评估未完成"));
        assert!(text.contains("EVALUATION_RUN_UNAVAILABLE"));
        assert!(text.contains("查看 m-model runs show <id>"));
    }
}
