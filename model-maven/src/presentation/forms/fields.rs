use crate::presentation::{
    contracts::{FieldDescriptor, FieldValueSource},
    forms::schema::validate_value,
    session_store,
    terminal::prompt_allow_empty,
    widgets::field_help::render_field_help,
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FormMode {
    /// Construct a new value and fill Schema defaults when the user accepts them.
    Full,
    /// Collect only explicit changes; defaults never become implicit edits.
    Patch,
}

#[derive(Debug, Eq, PartialEq)]
enum FieldInput {
    Value(Value),
    Skip,
    Back,
    Cancel,
}

pub fn prompt_schema_object(
    schema: &Value,
    prefix: &str,
    current: Option<&Map<String, Value>>,
    mode: FormMode,
) -> Option<Map<String, Value>> {
    let properties = schema["properties"].as_object()?;
    if properties
        .iter()
        .any(|(name, field_schema)| sensitive_field(name, field_schema))
    {
        println!("会话恢复会跳过敏感字段和未显式标记安全的自由文本字段。");
    }
    let (draft_root, draft_id, schema_revision) = form_identity(prefix, schema);
    let mut values = match (draft_root.as_deref(), draft_id.as_deref()) {
        (Some(root), Some(id)) => {
            match session_store::load_form_draft(root, id, &schema_revision) {
                Ok(Some(draft)) if draft.stale => {
                    println!("此表单的 Schema 已变化，旧草稿不能直接恢复。");
                    match prompt_allow_empty("1 丢弃旧草稿并重新填写，2 返回（保留草稿）")?.as_str()
                    {
                        "1" => {
                            if let Err(error) = session_store::clear_form_draft(root, id) {
                                eprintln!("无法丢弃旧表单草稿：{error}");
                                return None;
                            }
                            Map::new()
                        }
                        _ => return None,
                    }
                }
                Ok(Some(draft)) => {
                    println!(
                        "发现上次未提交的表单草稿（{} 个字段）。",
                        draft.values.len()
                    );
                    match prompt_allow_empty("1 恢复草稿，2 丢弃并重新填写，3 返回")?.as_str()
                    {
                        "1" => draft.values,
                        "2" => {
                            if let Err(error) = session_store::clear_form_draft(root, id) {
                                eprintln!("无法丢弃表单草稿：{error}");
                                return None;
                            }
                            Map::new()
                        }
                        _ => return None,
                    }
                }
                Ok(None) => Map::new(),
                Err(error) => {
                    eprintln!("无法恢复表单草稿：{error}；未覆盖旧草稿。");
                    return None;
                }
            }
        }
        _ => Map::new(),
    };
    let mut draft_keys = values.keys().cloned().collect::<HashSet<_>>();
    let required = schema["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<HashSet<_>>();
    let names = properties.keys().cloned().collect::<Vec<_>>();
    let mut index = 0;
    while index < names.len() {
        let name = &names[index];
        let path = format!("{prefix}.{name}");
        let field_required = required.contains(name.as_str());
        let field_schema = properties.get(name)?;
        let (current_value, value_source) = if let Some(value) = values.get(name).cloned() {
            let source = if draft_keys.contains(name) {
                FieldValueSource::Draft
            } else {
                FieldValueSource::UserInput
            };
            (Some(value), source)
        } else {
            let value = current.and_then(|values| values.get(name).cloned());
            let source = if value.is_some() {
                FieldValueSource::Current
            } else if field_schema.get("default").is_some() {
                FieldValueSource::Default
            } else {
                FieldValueSource::Unset
            };
            (value, source)
        };
        let mut descriptor = FieldDescriptor::from_schema(
            path.clone(),
            field_schema.clone(),
            field_required,
            current_value.clone(),
        );
        descriptor.value_source = value_source;
        for line in render_field_help(&descriptor) {
            println!("  {path} · {line}");
        }
        match prompt_field(&descriptor, mode)? {
            FieldInput::Value(value) => {
                values.insert(name.clone(), value);
                draft_keys.remove(name);
                persist_form_draft(
                    draft_root.as_deref(),
                    draft_id.as_deref(),
                    &schema_revision,
                    schema,
                    &values,
                );
                index += 1;
            }
            FieldInput::Skip => {
                if mode == FormMode::Full {
                    if let Some(value) = current_value.or_else(|| descriptor.default.clone()) {
                        values.insert(name.clone(), value);
                        draft_keys.remove(name);
                        persist_form_draft(
                            draft_root.as_deref(),
                            draft_id.as_deref(),
                            &schema_revision,
                            schema,
                            &values,
                        );
                    } else if field_required {
                        eprintln!("此字段必填，请输入值或 Schema 默认值。\n");
                        continue;
                    }
                }
                index += 1;
            }
            FieldInput::Back => {
                if index > 0 {
                    index -= 1;
                } else {
                    eprintln!("已经是第一个字段。\n");
                }
            }
            FieldInput::Cancel => {
                if let (Some(root), Some(id)) = (draft_root.as_deref(), draft_id.as_deref())
                    && let Err(error) = session_store::clear_form_draft(root, id)
                {
                    eprintln!("表单已取消，但未能清除已保存草稿：{error}");
                }
                return None;
            }
        }
    }
    if let (Some(root), Some(id)) = (draft_root.as_deref(), draft_id.as_deref())
        && let Err(error) = session_store::clear_form_draft(root, id)
    {
        eprintln!("表单已完成，但未能清除旧草稿：{error}");
    }
    Some(values)
}

fn form_identity(prefix: &str, schema: &Value) -> (Option<PathBuf>, Option<String>, String) {
    let schema_bytes = serde_json::to_vec(schema).unwrap_or_default();
    let mut schema_digest = Sha256::new();
    schema_digest.update(schema_bytes);
    let schema_revision = format!("{:x}", schema_digest.finalize());
    let mut id_digest = Sha256::new();
    id_digest.update(prefix.as_bytes());
    let id = format!("{:x}", id_digest.finalize());
    let root = std::env::current_dir()
        .ok()
        .and_then(|path| path.canonicalize().ok());
    (root, Some(id), schema_revision)
}

fn persist_form_draft(
    root: Option<&Path>,
    form_id: Option<&str>,
    schema_revision: &str,
    schema: &Value,
    values: &Map<String, Value>,
) {
    let (Some(root), Some(form_id)) = (root, form_id) else {
        return;
    };
    let safe = safe_values(schema, values);
    let result = if safe.is_empty() {
        session_store::clear_form_draft(root, form_id)
    } else {
        session_store::save_form_draft(root, form_id, schema_revision, &safe)
    };
    if let Err(error) = result {
        eprintln!("无法保存表单恢复点：{error}");
    }
}

fn safe_values(schema: &Value, values: &Map<String, Value>) -> Map<String, Value> {
    let Some(properties) = schema["properties"].as_object() else {
        return Map::new();
    };
    values
        .iter()
        .filter_map(|(name, value)| {
            let field_schema = properties.get(name)?;
            if sensitive_field(name, field_schema) {
                return None;
            }
            Some((name.clone(), safe_value(field_schema, value)))
        })
        .collect()
}

fn safe_value(schema: &Value, value: &Value) -> Value {
    if let Some(object) = value.as_object() {
        return Value::Object(safe_values(schema, object));
    }
    if let Some(items) = value.as_array() {
        let item_schema = &schema["items"];
        return Value::Array(
            items
                .iter()
                .map(|item| safe_value(item_schema, item))
                .collect(),
        );
    }
    value.clone()
}

fn sensitive_field(name: &str, schema: &Value) -> bool {
    let sensitive_name = schema["writeOnly"] == true
        || schema["sensitive"] == true
        || schema["format"] == "password"
        || [
            "password",
            "secret",
            "token",
            "credential",
            "authorization",
            "api_key",
            "private_key",
        ]
        .iter()
        .any(|marker| format!(".{name}.").to_ascii_lowercase().contains(marker));
    if sensitive_name {
        return true;
    }
    match schema["type"].as_str() {
        Some("string") => {
            schema["enum"].as_array().is_none()
                && schema.get("const").is_none()
                && schema["x-model-maven-draft-safe"] != true
        }
        Some("any") => true,
        Some("array") => schema
            .get("items")
            .is_none_or(|items| sensitive_field("item", items)),
        _ => false,
    }
}

pub fn prompt_schema_payload(schema: &Value) -> Option<Value> {
    prompt_schema_object(schema, "payload", None, FormMode::Full).map(Value::Object)
}

fn prompt_field(field: &FieldDescriptor, mode: FormMode) -> Option<FieldInput> {
    let schema = &field.schema;
    let kind = schema["type"].as_str()?;
    let required_label = if field.required { "必填" } else { "选填" };
    loop {
        let input = prompt_allow_empty(&format!(
            "{}（{kind} · {required_label}；Enter 保留当前值/默认，:skip 跳过，:back 返回，:cancel 取消表单）",
            field.path
        ))?;
        match input.as_str() {
            ":cancel" => return Some(FieldInput::Cancel),
            ":back" => return Some(FieldInput::Back),
            ":skip" => {
                if mode == FormMode::Full {
                    if let Some(value) = field
                        .current_value
                        .clone()
                        .or_else(|| field.default.clone())
                    {
                        match validate_value(&value, schema, &field.path, 0) {
                            Ok(()) => return Some(FieldInput::Value(value)),
                            Err(error) => {
                                eprintln!("{error} 当前值/默认值无效，请输入修正后的值。\n");
                                continue;
                            }
                        }
                    }
                    if field.required {
                        eprintln!("必填字段不能跳过。\n");
                        continue;
                    }
                }
                return Some(FieldInput::Skip);
            }
            "" => match mode {
                FormMode::Full => {
                    if let Some(value) = field
                        .current_value
                        .clone()
                        .or_else(|| field.default.clone())
                    {
                        match validate_value(&value, schema, &field.path, 0) {
                            Ok(()) => return Some(FieldInput::Value(value)),
                            Err(error) => {
                                eprintln!("{error} 当前值/默认值无效，请输入修正后的值。\n");
                                continue;
                            }
                        }
                    }
                    if field.required {
                        eprintln!("此字段必填，请输入值。\n");
                        continue;
                    }
                    return Some(FieldInput::Skip);
                }
                FormMode::Patch => return Some(FieldInput::Skip),
            },
            _ => {}
        }
        let parsed = parse_field_value(&input, field, mode);
        match parsed {
            Ok(None) => return Some(FieldInput::Cancel),
            Ok(Some(value)) => match validate_value(&value, schema, &field.path, 0) {
                Ok(()) => return Some(FieldInput::Value(value)),
                Err(error) => {
                    eprintln!("{error} 请重试，当前字段仍可修改；输入 :cancel 可取消整个表单。\n")
                }
            },
            Err(error) => {
                eprintln!("{error} 请重试，当前字段仍可修改；输入 :cancel 可取消整个表单。\n")
            }
        }
    }
}

fn parse_field_value(
    input: &str,
    field: &FieldDescriptor,
    mode: FormMode,
) -> Result<Option<Value>, String> {
    let schema = &field.schema;
    if schema["type"] == "object" && matches!(input, "yes" | "y") {
        let current = field.current_value.as_ref().and_then(Value::as_object);
        return Ok(prompt_schema_object(schema, &field.path, current, mode).map(Value::Object));
    }
    if let Some(choices) = schema["enum"].as_array()
        && let Some(index) = input
            .strip_prefix('#')
            .and_then(|value| value.parse::<usize>().ok())
    {
        return choices
            .get(index.checked_sub(1).ok_or("枚举序号从 1 开始。")?)
            .cloned()
            .map(Some)
            .ok_or_else(|| "枚举序号超出范围。".into());
    }
    if schema["type"] == "string" {
        return Ok(Some(if input == "\"\"" {
            Value::String(String::new())
        } else {
            Value::String(input.to_owned())
        }));
    }
    serde_json::from_str(input)
        .map(Some)
        .map_err(|error| format!("不是有效 JSON：{error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presentation::widgets::field_help::render_field_help;
    use serde_json::json;

    #[test]
    fn field_help_shows_schema_limits_current_source_and_hides_secrets() {
        let field = FieldDescriptor::from_schema(
            "optimizer.lr",
            json!({"type":"number","description":"学习率","minimum":0.0,"maximum":1.0,"default":0.1}),
            true,
            Some(json!(0.03)),
        );
        let help = render_field_help(&field).join("\n");
        assert!(help.contains("学习率"));
        assert!(help.contains("当前配置"));
        assert!(help.contains("0.03"));
        assert!(help.contains("minimum"));

        let secret = FieldDescriptor::from_schema(
            "api_token",
            json!({"type":"string","writeOnly":true,"default":"secret-value"}),
            false,
            Some(json!("live-secret")),
        );
        let help = render_field_help(&secret).join("\n");
        assert!(!help.contains("secret-value"));
        assert!(!help.contains("live-secret"));
        assert!(help.contains("<已隐藏>"));
    }

    #[test]
    fn persistent_form_values_omit_sensitive_fields_recursively() {
        let schema = json!({
            "type":"object",
            "properties":{
                "width":{"type":"integer"},
                "note":{"type":"string","x-model-maven-draft-safe":true},
                "api_token":{"type":"string"},
                "nested":{"type":"object","properties":{
                    "label":{"type":"string","enum":["safe"]},
                    "password":{"type":"string"}
                }}
            }
        });
        let values = json!({
            "width":32,
            "note":"safe note",
            "free_text":"do not persist",
            "api_token":"private",
            "nested":{"label":"safe","password":"private"}
        });
        let safe = safe_values(&schema, values.as_object().unwrap());
        assert_eq!(safe["width"], 32);
        assert_eq!(safe["note"], "safe note");
        assert!(safe.get("free_text").is_none());
        assert!(safe.get("api_token").is_none());
        assert_eq!(safe["nested"]["label"], "safe");
        assert!(safe["nested"].get("password").is_none());
    }
}
