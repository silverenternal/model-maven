use serde_json::Value;

pub fn render_catalog(report: &Value) -> Vec<String> {
    let components = report["data"]["components"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    render_catalog_page(report, 0, components.len().max(1))
}

pub fn render_catalog_page(report: &Value, requested_page: usize, page_size: usize) -> Vec<String> {
    let components = report["data"]["components"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let page_size = page_size.max(1);
    let page_count = components.len().div_ceil(page_size).max(1);
    let page = requested_page.min(page_count - 1);
    let start = page.saturating_mul(page_size).min(components.len());
    let end = start.saturating_add(page_size).min(components.len());
    let mut lines = vec![format!(
        "组件目录 · {} 个组件 · 第 {}/{} 页",
        components.len(),
        page + 1,
        page_count
    )];
    if components.is_empty() {
        lines.push("当前项目尚无已登记组件；可发现已安装扩展，或从组件包目录安装。".into());
        return lines;
    }
    for component in &components[start..end] {
        let manifest = &component["manifest"];
        let id = manifest["component_id"].as_str().unwrap_or("未知组件");
        let version = manifest["component_version"].as_str().unwrap_or("未知版本");
        let kind = manifest["kind"].as_str().unwrap_or("未知类型");
        lines.push(format!("\n{id}@{version} · {kind}"));
        if let Some(description) = manifest["description"].as_str() {
            lines.push(format!("  {description}"));
        }
        lines.push(format!(
            "  入口：{}",
            manifest["entrypoint"].as_str().unwrap_or("未声明")
        ));
        if let Some(source) = component["manifest_path"].as_str() {
            lines.push(format!("  描述来源：{source}"));
        }
        if let Some(dependencies) = manifest["dependencies"].as_array() {
            if dependencies.is_empty() {
                lines.push("  组件依赖：无".into());
            } else {
                for dependency in dependencies {
                    lines.push(format!(
                        "  依赖：{} {}{}",
                        dependency["component_id"].as_str().unwrap_or("未知组件"),
                        dependency["version"].as_str().unwrap_or("未声明版本约束"),
                        if dependency["optional"] == true {
                            "（可选）"
                        } else {
                            ""
                        }
                    ));
                }
            }
        }
        if let Some(capabilities) = manifest["capabilities"].as_object() {
            for (name, capability) in capabilities {
                let state = capability
                    .as_str()
                    .or_else(|| capability["state"].as_str())
                    .unwrap_or("unknown");
                lines.push(format!("  能力：{name} · {}", state_label(state)));
            }
        }
        let schema = &manifest["params_schema"];
        let required = schema["required"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        if let Some(properties) = schema["properties"].as_object() {
            for (name, property) in properties {
                lines.push(format!(
                    "  参数：{name} · {}{}",
                    property["type"].as_str().unwrap_or("any"),
                    if required.contains(&name.as_str()) {
                        " · 必填"
                    } else {
                        " · 可选"
                    }
                ));
            }
        }
        lines.push(format!(
            "  可用操作：{}",
            available_actions(manifest).join(" / ")
        ));
        if can_assemble(manifest) {
            lines.push("  可用位置：根工厂装配接缝".into());
        }
    }
    lines
}

pub fn available_actions(manifest: &Value) -> Vec<String> {
    let mut actions = vec!["show".into(), "check".into(), "uninstall".into()];
    if matches!(
        manifest["kind"].as_str(),
        Some("layer" | "layer-edit-provider")
    ) {
        actions.push("test".into());
    }
    actions
}

pub fn can_assemble(manifest: &Value) -> bool {
    manifest["kind"] == "layer" && manifest["capabilities"]["construct"]["state"] == "supported"
}

fn state_label(state: &str) -> &'static str {
    match state {
        "supported" => "支持",
        "conditional" => "有条件支持",
        "unsupported" => "不支持",
        "unknown" => "未知",
        _ => "未声明",
    }
}

#[cfg(test)]
mod tests {
    use super::{available_actions, can_assemble, render_catalog};
    use serde_json::json;

    #[test]
    fn catalog_shows_schema_dependencies_and_declared_capability_without_overstating_support() {
        let report = json!({"data":{"components":[{"manifest":{
            "component_id":"example/conditional-layer","component_version":"0.3.0","kind":"layer",
            "description":"Small test layer","entrypoint":"pkg.layers:make_layer",
            "dependencies":[{"component_id":"example/base","version":">=1.0.0,<2.0.0","optional":false}],
            "params_schema":{"type":"object","required":["width"],"properties":{"width":{"type":"integer"},"bias":{"type":"boolean"}}},
            "capabilities":{"construct":{"state":"supported"},"forward_probe":{"state":"conditional"}}
        }}]}});
        let view = render_catalog(&report).join("\n");
        assert!(view.contains("example/conditional-layer@0.3.0"));
        assert!(view.contains("example/base >=1.0.0,<2.0.0"));
        assert!(view.contains("width · integer · 必填"));
        assert!(view.contains("forward_probe · 有条件支持"));
        assert!(!view.contains("完全兼容"));
        assert!(view.contains("test"));
    }

    #[test]
    fn actions_follow_manifest_kind_and_capabilities() {
        let provider = available_actions(&json!({"kind":"layer-edit-provider"}));
        assert!(provider.contains(&"test".into()));
        assert!(!can_assemble(&json!({"kind":"layer-edit-provider"})));
        assert!(can_assemble(
            &json!({"kind":"layer","capabilities":{"construct":{"state":"supported"}}})
        ));
    }
}
