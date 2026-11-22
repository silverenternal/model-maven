use serde_json::Value;
use std::collections::BTreeSet;

pub fn render(dimension: &Value) -> Vec<String> {
    let mut lines = vec![format!(
        "指标 · {}",
        match dimension["status"].as_str() {
            Some("same") => "相同",
            Some("changed") => "有差异",
            _ => "信息不足",
        }
    )];
    let left = dimension["left"].as_object();
    let right = dimension["right"].as_object();
    let keys = left
        .into_iter()
        .flat_map(|values| values.keys())
        .chain(right.into_iter().flat_map(|values| values.keys()))
        .cloned()
        .collect::<BTreeSet<_>>();
    if keys.is_empty() {
        lines.push("  两侧均未记录可比较的指标。".into());
        return lines;
    }
    for key in keys {
        let left_value = left.and_then(|values| values.get(&key));
        let right_value = right.and_then(|values| values.get(&key));
        let delta = match (
            left_value.and_then(Value::as_f64),
            right_value.and_then(Value::as_f64),
        ) {
            (Some(left), Some(right)) => format!(" · Δ {:+.6}", right - left),
            _ => String::new(),
        };
        lines.push(format!(
            "  {key}：{} → {}{delta}",
            display_value(left_value),
            display_value(right_value)
        ));
    }
    lines
}

fn display_value(value: Option<&Value>) -> String {
    match value {
        Some(Value::Number(number)) => number.to_string(),
        Some(value) => value.to_string(),
        None => "未记录".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn metric_rows_show_numeric_change_and_missing_values() {
        let lines = render(&json!({
            "status":"changed",
            "left":{"accuracy":0.8},
            "right":{"accuracy":0.9,"f1":0.7}
        }));
        assert!(lines.iter().any(|line| line.contains("Δ +0.100000")));
        assert!(lines.iter().any(|line| line.contains("f1：未记录 → 0.7")));
    }
}
