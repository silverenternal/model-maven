use serde_json::Value;

pub fn render(report: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    let validation = &report["data"]["validation"];
    if validation.is_object() {
        lines.push(format!(
            "检查范围：构造 {} · 前向 {}",
            validation["construction"].as_str().unwrap_or("unknown"),
            validation["forward"].as_str().unwrap_or("unknown")
        ));
    }
    if let Some(effects) = report["data"]["edit_effects"].as_array() {
        for effect in effects {
            let action = effect["action"].as_str().unwrap_or("edit");
            let path = effect["path"].as_str().unwrap_or("未指定路径");
            let label = match action {
                "add" => "新增",
                "set" => "修改",
                "replace" => "替换",
                "remove" if effect["strategy"] == "bypass" => "以 Identity 替代",
                "remove" => "删除",
                _ => "编辑",
            };
            lines.push(format!("{label}：{path}"));
            if let Some(aliases) = effect["affected_aliases"]
                .as_array()
                .filter(|items| items.len() > 1)
            {
                let paths = aliases
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(format!("  共享影响：{paths}"));
            }
            for (label, field) in [("保留状态", "preserved"), ("新初始化", "initialized")] {
                if let Some(fields) = effect[field].as_array().filter(|items| !items.is_empty()) {
                    lines.push(format!(
                        "  {label}：{}",
                        fields
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }
    }
    if let Some(patches) = report["data"]["source_patches"].as_array() {
        for patch in patches {
            lines.push(format!(
                "源码文件：{} · {} → {}",
                patch["path"].as_str().unwrap_or("未知文件"),
                patch["before_sha256"].as_str().unwrap_or("未知"),
                patch["after_sha256"].as_str().unwrap_or("未知")
            ));
            if let Some(diff) = patch["unified_diff"]
                .as_str()
                .filter(|diff| !diff.is_empty())
            {
                lines.push(diff.to_owned());
            }
        }
    }
    if lines.is_empty() {
        lines.push("编辑预览未提供结构差异或状态影响详情。".into());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn diff_summary_shows_shared_effects_and_source_files() {
        let result = render(&json!({"data":{
            "validation":{"construction":"passed","forward":"not_run"},
            "edit_effects":[{"action":"set","path":"model.shared","affected_aliases":["model.shared","model.head.shared"],"preserved":["weight"]}],
            "source_patches":[{"path":"model.py","before_sha256":"before","after_sha256":"after","unified_diff":"-old\n+new\n"}]
        }})).join("\n");
        assert!(result.contains("共享影响：model.shared, model.head.shared"));
        assert!(result.contains("保留状态：weight"));
        assert!(result.contains("源码文件：model.py"));
        assert!(result.contains("-old\n+new"));
    }
}
