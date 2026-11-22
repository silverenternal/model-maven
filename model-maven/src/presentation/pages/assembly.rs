use serde_json::Value;

pub fn render(report: &Value) -> Vec<String> {
    let model = report["data"]["model"].as_str().unwrap_or("未知模型入口");
    let path = report["data"]["path"].as_str().unwrap_or("配置路径未知");
    let validation = &report["data"]["validation"];
    let forward = match validation["forward"].as_str() {
        Some("passed") => "代表批次前向已运行且通过",
        Some("failed") => "代表批次前向失败",
        Some("not_run") => "尚未运行代表批次前向（当前只确认模型可构造）",
        _ => "前向状态未知",
    };
    let mut lines = vec![
        "根工厂装配 · 项目默认配置".into(),
        format!("模型工厂：{model}"),
        format!("配置文件：{path}"),
        "写入目标：项目默认装配；已有模型变体保留各自的固定快照。".into(),
        format!("检查状态：{forward}"),
    ];
    let slots = report["data"]["slots"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    if slots.is_empty() {
        lines.push("尚无显式接缝；根工厂仍由项目 Python 代码定义。".into());
    }
    for slot in slots {
        let name = slot["slot"].as_str().unwrap_or("未知接缝");
        let binding = slot["binding"]
            .as_str()
            .or_else(|| slot["factory"].as_str())
            .or_else(|| slot["component_ref"]["id"].as_str())
            .unwrap_or("绑定信息未知");
        lines.push(format!("\n接缝 {name} ← {binding}"));
        if let Some(version) = slot["component_ref"]["version"].as_str() {
            lines.push(format!("  组件版本：{version}"));
        }
        let paths = slot["registered_paths"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        lines.push(if paths.is_empty() {
            "  模型注册路径：未提供".into()
        } else {
            format!("  模型注册路径：{paths}")
        });
        let sample_forward = match slot["sample_forward"].as_str() {
            Some("observed") => "代表前向中已观测到此接缝",
            Some("not_observed") => "代表前向未观测到此接缝",
            _ => "尚无代表前向调用证据",
        };
        lines.push(format!("  调用覆盖：{sample_forward}"));
        if let Some(params) = slot["params"].as_object().filter(|value| !value.is_empty()) {
            lines.push(format!(
                "  组件参数：{}",
                serde_json::to_string(params).unwrap_or_else(|_| "无法显示".into())
            ));
        }
    }
    lines.push("此页面只编辑根工厂公开的接缝；内部网络路径请到 Layer 树中查看。".into());
    lines
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn assembly_page_distinguishes_project_default_from_variant_and_module_paths() {
        let lines = render(&json!({"data":{
            "model":"project.models:build_model",
            "path":".model-maven/project.json",
            "validation":{"construction":"passed","forward":"not_run"},
            "slots":[{"slot":"encoder","binding":"models:build_encoder","registered_paths":["model.encoder"],"sample_forward":null}]
        }})).join("\n");
        assert!(lines.contains("项目默认配置"));
        assert!(lines.contains("已有模型变体保留各自的固定快照"));
        assert!(lines.contains("接缝 encoder"));
        assert!(lines.contains("模型注册路径：model.encoder"));
        assert!(lines.contains("尚未运行代表批次前向"));
        assert!(lines.contains("尚无代表前向调用证据"));
    }

    #[test]
    fn assembly_page_reports_real_sample_coverage_and_parameter_binding() {
        let lines = render(&json!({"data":{
            "validation":{"forward":"passed"},
            "slots":[{"slot":"head","component_ref":{"id":"org/classifier","version":"2.1.0"},"params":{"classes":10},"registered_paths":["model.head"],"sample_forward":"observed"}]
        }})).join("\n");
        assert!(lines.contains("代表批次前向已运行且通过"));
        assert!(lines.contains("接缝 head ← org/classifier"));
        assert!(lines.contains("组件版本：2.1.0"));
        assert!(lines.contains("代表前向中已观测到此接缝"));
        assert!(lines.contains("{\"classes\":10}"));
    }
}
