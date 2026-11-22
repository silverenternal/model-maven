use serde_json::Value;

pub fn render(target: &str, operations: &[Value], preview: Option<&Value>) -> Vec<String> {
    let mut lines = vec![format!(
        "Layer 计划 · {} 项 · 目标 {target}",
        operations.len()
    )];
    for (index, operation) in operations.iter().enumerate() {
        let action = match operation["action"].as_str().unwrap_or("unknown") {
            "set" => "修改参数",
            "add" => "添加 Layer",
            "replace" => "替换 Layer",
            "remove" if operation["strategy"] == "bypass" => "以 Identity 替代",
            "remove" => "删除 Layer",
            _ => "未识别操作",
        };
        let path = operation["path"]
            .as_str()
            .or_else(|| operation["parent"].as_str())
            .unwrap_or("未指定路径");
        let field_names = operation["params"]
            .as_object()
            .map(|params| params.keys().cloned().collect::<Vec<_>>().join(", "))
            .filter(|names| !names.is_empty())
            .map(|names| format!(" · 字段 {names}"))
            .unwrap_or_default();
        lines.push(format!("{}. {action} · {path}{field_names}", index + 1));
    }
    if let Some(preview) = preview {
        if let Some(validation) = preview["data"]["validation"].as_object() {
            let construction = validation["construction"].as_str().unwrap_or("unknown");
            let forward = validation["forward"].as_str().unwrap_or("unknown");
            lines.push(format!("候选验证：构造 {construction} · 前向 {forward}"));
        } else if let Some(diagnostic) = preview["diagnostics"]
            .as_array()
            .and_then(|items| items.first())
        {
            lines.push(format!(
                "候选验证未通过：{} · {}；草稿仍可修复，尚未提交。",
                diagnostic["code"].as_str().unwrap_or("EDIT_FAILED"),
                diagnostic["message"].as_str().unwrap_or("未提供原因")
            ));
        }
    }
    lines.push("参数值不在草稿摘要中重复显示；检查报告展示实际结构和状态影响。".into());
    lines
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn draft_summary_shows_order_paths_fields_and_verification_without_values() {
        let plan = render(
            "variant:working",
            &[json!({"action":"set","path":"model.encoder.proj","params":{"out_features":16,"api_token":"secret"}})],
            Some(&json!({"data":{"validation":{"construction":"passed","forward":"not_run"}}})),
        )
        .join("\n");
        assert!(plan.contains("1 项 · 目标 variant:working"));
        assert!(plan.contains("修改参数 · model.encoder.proj · 字段 api_token, out_features"));
        assert!(plan.contains("构造 passed · 前向 not_run"));
        assert!(!plan.contains("secret"));
    }

    #[test]
    fn failed_preview_keeps_draft_actionable_and_uncommitted() {
        let plan = render(
            "variant:working",
            &[json!({"action":"set","path":"model.head","params":{"in_features":16}})],
            Some(&json!({"diagnostics":[{"code":"EDIT_VALIDATION_FAILED","message":"shape mismatch"}]})),
        )
        .join("\n");
        assert!(plan.contains("草稿仍可修复，尚未提交"));
        assert!(plan.contains("shape mismatch"));
    }
}
