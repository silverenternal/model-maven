use serde_json::Value;

use crate::presentation::widgets::{input_diff, metric_comparison};

pub fn render(data: &Value) -> Vec<String> {
    let left = data["left_run_id"].as_str().unwrap_or("未知");
    let right = data["right_run_id"].as_str().unwrap_or("未知");
    let protocol = state(data["protocol"]["status"].as_str());
    let changed = data["summary"]["changed_dimensions"].as_u64().unwrap_or(0);
    let unknown = data["summary"]["unknown_dimensions"].as_u64().unwrap_or(0);
    let classification = if data["protocol"]["status"] != "same" {
        "协议不同或未确认"
    } else if unknown > 0 && changed > 0 {
        "条件有差异，且部分信息不足"
    } else if unknown > 0 {
        "信息不足"
    } else if changed > 0 {
        "条件有差异"
    } else {
        "记录证据相同"
    };
    let mut lines = vec![
        format!("运行对照 · {left} ↔ {right}"),
        format!("协议：{protocol}"),
        format!("结论范围：{classification}"),
        String::new(),
    ];
    for (key, title) in [
        ("inputs", "输入与设置"),
        ("data", "数据与采样"),
        ("samples", "样本身份（训练代码报告）"),
        ("budget", "训练预算"),
    ] {
        if let Some(dimension) = data["dimensions"]
            .as_array()
            .and_then(|dimensions| dimensions.iter().find(|item| item["name"] == key))
        {
            lines.extend(input_diff::render(dimension, title));
        }
    }
    for (key, label) in [
        ("source_code", "源码"),
        ("runtime_device", "设备与运行时"),
        ("weights", "权重"),
    ] {
        if let Some(dimension) = data["dimensions"]
            .as_array()
            .and_then(|dimensions| dimensions.iter().find(|item| item["name"] == key))
        {
            let label_text = state(dimension["status"].as_str());
            if dimension["status"] == "changed" {
                lines.push(format!(
                    "{label}：{label_text} · {} → {}",
                    summary_value(&dimension["left"], key == "weights"),
                    summary_value(&dimension["right"], key == "weights")
                ));
            } else {
                lines.push(format!("{label}：{label_text}"));
            }
        }
        if key == "source_code" {
            for (field, change_label) in [
                ("changed_files", "修改"),
                ("added_files", "新增"),
                ("removed_files", "移除"),
            ] {
                if let Some(files) = data["source_file_changes"][field].as_array() {
                    let paths = files.iter().filter_map(Value::as_str).collect::<Vec<_>>();
                    if !paths.is_empty() {
                        lines.push(format!("  源码{change_label}：{}", paths.join("、")));
                    }
                }
            }
        }
    }
    if let Some(metrics) = data["dimensions"]
        .as_array()
        .and_then(|dimensions| dimensions.iter().find(|item| item["name"] == "metrics"))
    {
        lines.extend(metric_comparison::render(metrics));
    }
    lines.push(String::new());
    lines.push("样本身份来自训练代码显式报告；工具不校验它与批次内容是否一致。".into());
    lines.push("记录对照只展示观察到的差异，不推断指标变化原因。".into());
    lines
}

fn state(status: Option<&str>) -> &'static str {
    match status {
        Some("same") => "相同",
        Some("changed") => "有差异",
        _ => "信息不足",
    }
}

fn summary_value(value: &Value, short_hash: bool) -> String {
    if let Some(value) = value.as_str() {
        if short_hash && value.len() > 16 {
            return format!("{}…", value.chars().take(12).collect::<String>());
        }
        return value.to_owned();
    }
    let text = value.to_string();
    const LIMIT: usize = 96;
    if text.chars().count() <= LIMIT {
        text
    } else {
        format!("{}…", text.chars().take(LIMIT - 1).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn page_renders_major_dimensions_without_claiming_causality() {
        let page = render(&json!({
            "left_run_id":"run-left",
            "right_run_id":"run-right",
            "protocol":{"status":"same"},
            "summary":{"changed_dimensions":2,"unknown_dimensions":2},
            "source_file_changes":{"status":"changed","changed_files":["model.py"],"added_files":[],"removed_files":[]},
            "dimensions":[
                {"name":"inputs","status":"same","left":{},"right":{},"left_missing_fields":[],"right_missing_fields":[],"changed_fields":[]},
                {"name":"source_code","status":"changed"},
                {"name":"data","status":"unknown","left":{},"right":{},"left_missing_fields":[],"right_missing_fields":[],"changed_fields":[]},
                {"name":"samples","status":"unknown","left":{},"right":{},"left_missing_fields":["algorithm","sample_count","sample_ids_sha256"],"right_missing_fields":["algorithm","sample_count","sample_ids_sha256"],"changed_fields":[]},
                {"name":"budget","status":"same","left":{},"right":{},"left_missing_fields":[],"right_missing_fields":[],"changed_fields":[]},
                {"name":"runtime_device","status":"changed","left":"cpu","right":"cuda:0"},
                {"name":"weights","status":"unknown"},
                {"name":"metrics","status":"changed","left":{"accuracy":0.8},"right":{"accuracy":0.9}}
            ]
        }));
        assert!(
            page.iter()
                .any(|line| line.contains("run-left ↔ run-right"))
        );
        assert!(page.iter().any(|line| {
            line.contains("结论范围：条件有差异，且部分信息不足")
        }));
        assert!(page.iter().any(|line| line.contains("源码修改：model.py")));
        assert!(
            page.iter()
                .any(|line| line.contains("设备与运行时：有差异 · cpu → cuda:0"))
        );
        assert!(
            page.iter()
                .any(|line| line.contains("数据与采样 · 信息不足"))
        );
        assert!(
            page.iter()
                .any(|line| line.contains("样本身份（训练代码报告） · 信息不足"))
        );
        assert!(
            page.iter()
                .any(|line| line.contains("工具不校验它与批次内容是否一致"))
        );
        assert!(page.iter().any(|line| line.contains("左侧未记录：algorithm、sample_count、sample_ids_sha256")));
        assert!(page.iter().any(|line| line.contains("Δ +0.100000")));
        assert!(page.iter().any(|line| line.contains("不推断指标变化原因")));
    }
}
