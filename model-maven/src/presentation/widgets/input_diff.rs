use serde_json::Value;

/// Render only input fields that differ or are missing from persisted evidence.
pub fn render(dimension: &Value, title: &str) -> Vec<String> {
    let mut lines = vec![format!("{title} · {}", state(dimension["status"].as_str()))];
    let left = dimension["left"].as_object();
    let right = dimension["right"].as_object();
    let changed = dimension["changed_fields"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();

    if !changed.is_empty() {
        for field in changed {
            let left_value = left.and_then(|values| values.get(field));
            let right_value = right.and_then(|values| values.get(field));
            lines.push(format!(
                "  {field}：{} → {}",
                concise(left_value),
                concise(right_value)
            ));
        }
    } else if dimension["status"] == "same" {
        lines.push("  已记录字段相同。".into());
    }

    for side in ["left_missing_fields", "right_missing_fields"] {
        let missing = dimension[side]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            lines.push(format!(
                "  {}未记录：{}",
                if side == "left_missing_fields" {
                    "左侧"
                } else {
                    "右侧"
                },
                missing.join("、")
            ));
        }
    }
    lines
}

fn concise(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return "未记录".into();
    };
    let text = serde_json::to_string(value).unwrap_or_else(|_| "<无法显示>".into());
    const LIMIT: usize = 120;
    if text.chars().count() <= LIMIT {
        text
    } else {
        format!("{}…", text.chars().take(LIMIT - 1).collect::<String>())
    }
}

fn state(status: Option<&str>) -> &'static str {
    match status {
        Some("same") => "相同",
        Some("changed") => "有差异",
        _ => "信息不足",
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn changed_inputs_show_compact_values_and_missing_evidence() {
        let lines = render(
            &json!({
                "status":"changed",
                "left":{"lr":0.001},
                "right":{"lr":0.0001},
                "changed_fields":["lr"],
                "left_missing_fields":["seed"],
                "right_missing_fields":[]
            }),
            "输入",
        );
        assert!(lines.iter().any(|line| line.contains("0.001 → 0.0001")));
        assert!(lines.iter().any(|line| line.contains("左侧未记录：seed")));
    }
}
