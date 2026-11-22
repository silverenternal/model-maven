use serde_json::Value;

pub fn validate_value(
    value: &Value,
    schema: &Value,
    path: &str,
    depth: usize,
) -> Result<(), String> {
    if depth > 16 {
        return Err(format!("{path} 的 Schema 嵌套超过 16 层。"));
    }
    let kind = schema["type"].as_str().unwrap_or("unknown");
    let matches_type = match kind {
        "any" => true,
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        _ => false,
    };
    if !matches_type {
        return Err(format!("{path} 应为 {kind}。"));
    }
    if let Some(choices) = schema["enum"].as_array()
        && !choices.contains(value)
    {
        return Err(format!("{path} 不在允许值中。"));
    }
    if let Some(number) = value.as_f64() {
        if let Some(minimum) = schema["minimum"].as_f64()
            && number < minimum
        {
            return Err(format!("{path} 不能小于 {minimum}。"));
        }
        if let Some(maximum) = schema["maximum"].as_f64()
            && number > maximum
        {
            return Err(format!("{path} 不能大于 {maximum}。"));
        }
    }
    if let Some(text) = value.as_str() {
        if let Some(minimum) = schema["minLength"].as_u64()
            && text.chars().count() < minimum as usize
        {
            return Err(format!("{path} 长度不能小于 {minimum}。"));
        }
        if let Some(maximum) = schema["maxLength"].as_u64()
            && text.chars().count() > maximum as usize
        {
            return Err(format!("{path} 长度不能大于 {maximum}。"));
        }
    }
    if let Some(items) = value.as_array() {
        if let Some(minimum) = schema["minItems"].as_u64()
            && items.len() < minimum as usize
        {
            return Err(format!("{path} 至少需要 {minimum} 个元素。"));
        }
        if let Some(maximum) = schema["maxItems"].as_u64()
            && items.len() > maximum as usize
        {
            return Err(format!("{path} 最多允许 {maximum} 个元素。"));
        }
    }
    if kind == "object"
        && let Some(object) = value.as_object()
    {
        let properties = schema["properties"]
            .as_object()
            .ok_or_else(|| format!("{path} 的 object Schema 缺少 properties。"))?;
        if schema["additionalProperties"] == false
            && let Some(unknown) = object.keys().find(|name| !properties.contains_key(*name))
        {
            return Err(format!("{path}.{unknown} 未在 Schema 中声明。"));
        }
        if let Some(required) = schema["required"].as_array() {
            for name in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(name) {
                    return Err(format!("缺少必填字段：{path}.{name}。"));
                }
            }
        }
        for (name, child) in object {
            if let Some(child_schema) = properties.get(name) {
                validate_value(child, child_schema, &format!("{path}.{name}"), depth + 1)?;
            }
        }
    }
    if kind == "array"
        && let Some(items) = value.as_array()
    {
        if schema.get("items").is_none() {
            return Err(format!("{path} 的 array Schema 缺少 items。"));
        }
        for (index, child) in items.iter().enumerate() {
            validate_value(
                child,
                &schema["items"],
                &format!("{path}[{index}]"),
                depth + 1,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_text_and_array_bounds_with_paths() {
        assert!(
            validate_value(
                &json!("a"),
                &json!({"type":"string","minLength":2}),
                "name",
                0
            )
            .unwrap_err()
            .contains("name")
        );
        assert!(
            validate_value(
                &json!([1, 2]),
                &json!({"type":"array","items":{"type":"integer"},"maxItems":1}),
                "layers",
                0
            )
            .unwrap_err()
            .contains("layers")
        );
    }

    #[test]
    fn accepts_nested_json_values_for_untyped_extension_parameters() {
        let schema = json!({"type":"any"});
        assert!(validate_value(&json!({"width": 32}), &schema, "params.config", 0).is_ok());
        assert!(validate_value(&json!([1, 2]), &schema, "params.shape", 0).is_ok());
    }
}
