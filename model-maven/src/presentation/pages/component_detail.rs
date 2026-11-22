use super::components::available_actions;
use serde_json::Value;

pub fn render(report: &Value) -> Vec<String> {
    let component = &report["data"]["component"];
    let manifest = &component["manifest"];
    let id = manifest["component_id"].as_str().unwrap_or("未知组件");
    let version = manifest["component_version"].as_str().unwrap_or("未知版本");
    let dependencies = &report["data"]["dependency_resolution"];
    let mut lines = vec![
        format!("组件详情 · {id}@{version}"),
        format!(
            "类型：{} · 静态清单：{}",
            manifest["kind"].as_str().unwrap_or("未知"),
            report["data"]["static_validation"]
                .as_str()
                .unwrap_or("未检查")
        ),
        format!(
            "入口：{}",
            manifest["entrypoint"].as_str().unwrap_or("未声明")
        ),
        format!(
            "描述来源：{}",
            component["manifest_path"].as_str().unwrap_or("未提供")
        ),
        format!(
            "运行时导入：{}",
            report["data"]["runtime_import"]
                .as_str()
                .unwrap_or("未运行")
        ),
        format!(
            "依赖解析：{}",
            dependencies["status"].as_str().unwrap_or("未检查")
        ),
    ];
    if let Some(reason) = dependencies["reason"].as_str() {
        lines.push(format!("  原因：{reason}"));
    }
    if let Some(description) = manifest["description"].as_str() {
        lines.push(format!("说明：{description}"));
    }
    if let Some(source) = manifest["package_source"].as_object() {
        lines.push(format!(
            "扩展包来源：{}@{} · entry point {}",
            source["distribution"].as_str().unwrap_or("未知分发包"),
            source["distribution_version"]
                .as_str()
                .unwrap_or("未知版本"),
            source["entry_point"].as_str().unwrap_or("未声明")
        ));
    }
    if let Some(capabilities) = manifest["capabilities"].as_object() {
        lines.push("声明能力：".into());
        for (name, value) in capabilities {
            let state = value
                .as_str()
                .or_else(|| value["state"].as_str())
                .unwrap_or("unknown");
            lines.push(format!("  {name} · {state}"));
        }
    }
    if let Some(dependencies) = manifest["dependencies"].as_array() {
        for dependency in dependencies {
            lines.push(format!(
                "依赖：{} {}{}",
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
    let required = manifest["params_schema"]["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    if let Some(properties) = manifest["params_schema"]["properties"].as_object() {
        lines.push("参数 Schema：".into());
        for (name, property) in properties {
            lines.push(format!(
                "  {name} · {}{}{}",
                property["type"].as_str().unwrap_or("any"),
                if required.contains(&name.as_str()) {
                    " · 必填"
                } else {
                    " · 可选"
                },
                property["description"]
                    .as_str()
                    .map(|description| format!(" · {description}"))
                    .unwrap_or_default()
            ));
        }
    }
    if report["data"]["edit_verification"].is_object() {
        let verification = &report["data"]["edit_verification"];
        lines.push(format!(
            "Provider payload 验证：{} · {} 组记录",
            verification["status"].as_str().unwrap_or("unknown"),
            verification["payload_count"].as_u64().unwrap_or(0)
        ));
        if let Some(runtime) = verification["runtime_recheck"].as_str() {
            lines.push(format!("运行时重检范围：{runtime}"));
        }
    }
    lines.push(format!(
        "当前可用操作：{}",
        available_actions(manifest).join(" / ")
    ));
    lines
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn detail_keeps_static_runtime_and_dependency_evidence_separate() {
        let detail = render(&json!({"data":{
            "component":{"manifest":{"component_id":"example/layer","component_version":"1.0.0","kind":"layer","entrypoint":"pkg:make","description":"Demo","capabilities":{"construct":{"state":"supported"}}}},
            "static_validation":"passed","runtime_import":"not_run",
            "dependency_resolution":{"status":"unresolved","reason":"missing example/base"}
        }})).join("\n");
        assert!(detail.contains("静态清单：passed"));
        assert!(detail.contains("运行时导入：not_run"));
        assert!(detail.contains("依赖解析：unresolved"));
        assert!(detail.contains("missing example/base"));
        assert!(!detail.contains("运行时兼容保证"));
    }
}
