use crate::presentation::contracts::{ActionDescriptor, PageDescriptor, PageId};
use serde_json::Value;

pub fn render(
    project: &str,
    saved: Option<&Value>,
    revision: Option<&str>,
    active_model: Option<&str>,
    active_model_source: Option<&str>,
    active_batch: Option<&str>,
    active_variant: Option<&str>,
) -> PageDescriptor {
    let mut lines = vec![format!("项目：{project}")];
    match saved {
        Some(config) => {
            lines.push("已保存配置：.model-maven/project.json".into());
            lines.push(format!(
                "配置版本：{}",
                revision.unwrap_or("不可用；无法读取当前 revision")
            ));
            lines.push(format!("默认模型：{}", entrypoint(config.get("model"))));
            lines.push(format!("默认批次：{}", entrypoint(config.get("batch"))));
            if let Some(binding) = config.get("batch_binding").and_then(Value::as_str) {
                lines.push(format!("默认批次绑定：{binding}"));
            }
            if let Some(injections) = config.get("model_injections").and_then(Value::as_object) {
                let seams = injections
                    .iter()
                    .map(|(slot, injection)| {
                        let component = injection["component_ref"]["id"]
                            .as_str()
                            .zip(injection["component_ref"]["version"].as_str())
                            .map(|(id, version)| format!("{id}@{version}"));
                        format!(
                            "{slot} → {}",
                            component.unwrap_or_else(|| "Python 工厂".into())
                        )
                    })
                    .collect::<Vec<_>>();
                if !seams.is_empty() {
                    lines.push(format!("模型装配：{}", seams.join("；")));
                }
                lines.push("组件参数在本页隐藏；进入装配详情查看。".into());
            }
        }
        None => {
            lines.push("没有已保存的项目默认配置。".into());
            lines.push("临时会话入口不会写入 project.json。".into());
        }
    }
    lines.push("\n本次工作台采用：".into());
    lines.push(format!(
        "模型：{} · 来源：{}",
        active_model.unwrap_or("未选择"),
        value_source(
            active_model,
            saved.and_then(|config| config["model"].as_str()),
            active_model_source
        )
    ));
    lines.push(format!(
        "批次：{} · 来源：{}",
        active_batch.unwrap_or("未设置"),
        value_source(
            active_batch,
            saved.and_then(|config| config["batch"].as_str()),
            None
        )
    ));
    lines.push(format!(
        "模型变体：{} · 来源：{}",
        active_variant.unwrap_or("默认模型"),
        if active_variant.is_some() {
            "本次命令"
        } else {
            "默认结构"
        }
    ));
    lines.push("每次运行使用独立输入快照；本页查看配置与工作台选择，不会保存或重放运行。".into());
    PageDescriptor {
        id: PageId::new("configuration"),
        title: "配置与输入来源".into(),
        summary: lines.join("\n"),
        actions: vec![ActionDescriptor::available("back", "返回")],
    }
}

fn entrypoint(value: Option<&Value>) -> &str {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("未设置")
}

fn value_source(active: Option<&str>, saved: Option<&str>, hint: Option<&str>) -> &'static str {
    if matches!(
        hint,
        Some("本次临时入口" | "恢复的临时入口" | "本次工作台会话")
    ) {
        return "本次工作台会话";
    }
    match (active, saved) {
        (None, _) => "未设置",
        (Some(active), Some(saved)) if active == saved => "项目默认配置",
        (Some(_), Some(_)) | (Some(_), None) => "本次工作台会话",
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use serde_json::json;

    #[test]
    fn configuration_summary_distinguishes_saved_defaults_from_session_overrides() {
        let page = render(
            "flowers",
            Some(&json!({
                "schema_version":1,
                "model":"models:build",
                "batch":"data:sample",
                "model_injections":{"encoder":{"factory":"layers:encoder","params":{"token":"hidden"}}}
            })),
            Some("rev-123"),
            Some("models:build"),
            Some("项目默认配置"),
            Some("data:other_sample"),
            Some("working"),
        );
        assert!(page.summary.contains("配置版本：rev-123"));
        assert!(
            page.summary
                .contains("模型：models:build · 来源：项目默认配置")
        );
        assert!(
            page.summary
                .contains("批次：data:other_sample · 来源：本次工作台会话")
        );
        assert!(page.summary.contains("模型装配：encoder → Python 工厂"));
        assert!(!page.summary.contains("hidden"));
    }

    #[test]
    fn temporary_inputs_are_not_described_as_saved_configuration() {
        let page = render(
            "scratch",
            None,
            None,
            Some("scratch:build"),
            Some("本次临时入口"),
            None,
            None,
        );
        assert!(page.summary.contains("没有已保存的项目默认配置"));
        assert!(
            page.summary
                .contains("模型：scratch:build · 来源：本次工作台会话")
        );
        assert!(page.summary.contains("不会写入 project.json"));
    }
}
