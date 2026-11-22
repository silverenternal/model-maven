use crate::presentation::contracts::{FieldDescriptor, FieldValueSource};

pub fn render_field_help(field: &FieldDescriptor) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(description) = field.description.as_deref() {
        lines.push(format!("用途：{description}"));
    } else {
        lines.push("用途：Schema 未提供说明。".into());
    }
    lines.push(format!(
        "当前值：{} · 来源：{}",
        field
            .current_value
            .as_ref()
            .map(|value| {
                if sensitive(field) {
                    "<已隐藏>".into()
                } else {
                    value.to_string()
                }
            })
            .unwrap_or_else(|| "未设置".into()),
        match field.value_source {
            FieldValueSource::Current => "当前配置",
            FieldValueSource::UserInput => "本次表单输入",
            FieldValueSource::Draft => "上次未提交的表单草稿",
            FieldValueSource::Default => "Schema 默认值",
            FieldValueSource::Unset => "未设置",
        }
    ));
    if let Some(default) = field.default.as_ref() {
        lines.push(format!(
            "默认值：{}",
            if sensitive(field) {
                "<已隐藏>".into()
            } else {
                default.to_string()
            }
        ));
    }
    let kind = field.schema["type"].as_str().unwrap_or("未知类型");
    lines.push(format!(
        "类型：{kind}{}",
        if field.required {
            " · 必填"
        } else {
            " · 选填"
        }
    ));
    for bound in [
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    ] {
        if let Some(value) = field.schema.get(bound) {
            lines.push(format!("{bound}：{value}"));
        }
    }
    if let Some(choices) = field.schema["enum"].as_array() {
        lines.push(format!(
            "可选值：{}",
            choices
                .iter()
                .enumerate()
                .map(|(index, value)| format!("#{} {}", index + 1, value))
                .collect::<Vec<_>>()
                .join("；")
        ));
    }
    lines
}

fn sensitive(field: &FieldDescriptor) -> bool {
    if field.schema["writeOnly"] == true || field.schema["sensitive"] == true {
        return true;
    }
    let name = field.path.to_ascii_lowercase();
    [
        "password",
        "secret",
        "token",
        "credential",
        "api_key",
        "private_key",
    ]
    .iter()
    .any(|marker| name.contains(marker))
}
