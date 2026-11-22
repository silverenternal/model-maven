use serde_json::Value;

pub fn render_context(
    model: Option<&str>,
    batch: Option<&str>,
    variant: Option<&str>,
) -> Vec<String> {
    vec![
        "Layer 编辑计划".into(),
        format!("模型：{}", model.unwrap_or("未选择")),
        format!(
            "验证批次：{}",
            batch.unwrap_or("未选择；只验证构造，不运行前向")
        ),
        format!(
            "目标：持久变体 {}；原始 Python 源码保持不变。",
            variant.unwrap_or("working")
        ),
        "成组参数修改会先暂存草稿；前向失败时可继续修复或撤销最后一项，只有最终检查通过后才提交。"
            .into(),
    ]
}

pub fn can_patch(row: &Value) -> bool {
    row["can_set"] == true && row["editable_params"].is_object()
}

#[cfg(test)]
mod tests {
    use super::{can_patch, render_context};
    use serde_json::json;

    #[test]
    fn edit_context_makes_target_and_verification_scope_explicit() {
        let context = render_context(Some("project:model"), None, Some("working")).join("\n");
        assert!(context.contains("持久变体 working"));
        assert!(context.contains("只验证构造，不运行前向"));
        assert!(context.contains("原始 Python 源码保持不变"));
    }

    #[test]
    fn only_rows_with_declared_parameter_fields_offer_patch() {
        assert!(can_patch(
            &json!({"can_set":true,"editable_params":{"p":0.5}})
        ));
        assert!(!can_patch(&json!({"can_set":false,"editable_params":null})));
    }
}
