use serde_json::Value;

pub fn render_layer_detail(row: &Value, report: &Value, probe: Option<&Value>) -> Vec<String> {
    let path = row["path"].as_str().unwrap_or("<无路径>");
    let layer_type = row["type"].as_str().unwrap_or("未知类型");
    let mut lines = vec![
        format!("Layer 详情 · {path}"),
        format!("类型：{layer_type}"),
    ];
    if let Some(id) = row["instance_id"].as_str() {
        lines.push(format!("模块身份：{id}"));
    }
    if let Some(aliases) = row["aliases"].as_array() {
        let aliases = aliases.iter().filter_map(Value::as_str).collect::<Vec<_>>();
        if aliases.len() > 1 {
            lines.push(format!("共享别名：{}", aliases.join("、")));
            lines.push("共享参数只按同一个模块身份统计；修改会影响所有别名。".into());
        }
    }
    if let Some(count) = row["own_parameters"].as_u64() {
        lines.push(format!("本层直接参数：{count}"));
    }
    if let Some(count) = row["parameters"].as_u64() {
        lines.push(format!(
            "该子树参数：{count}（模型总数由根节点报告，不在别名之间求和）"
        ));
    }
    append_location(&mut lines, "构造位置", &row["construction_site"]);
    append_location(&mut lines, "类型定义", &row["source_definition"]);
    if let Some(source) = row["source"].as_str() {
        lines.push(format!("代码来源：{source}"));
    }
    append_control_source(&mut lines, path, row, report);
    append_probe_evidence(&mut lines, row, probe);
    let source_block = row["source_set_block_reason"].as_str();
    let variant_actions = row["can_set"] == true || row["can_insert"] == true;
    let source_set = row["can_source_set"] == true && source_block.is_none();
    let source_insert = row["can_source_insert"] == true;
    lines.push(format!(
        "变体操作：{}",
        if variant_actions {
            format!(
                "{}{}",
                if row["can_set"] == true {
                    "参数修改"
                } else {
                    ""
                },
                if row["can_insert"] == true {
                    " 容器插入"
                } else {
                    ""
                }
            )
        } else {
            "当前没有直接变体编辑操作".into()
        }
    ));
    lines.push(format!(
        "源码操作：{}",
        if source_set || source_insert {
            format!(
                "{}{}{}",
                if source_set { "参数修改 " } else { "" },
                if source_insert { "容器插入 " } else { "" },
                source_block
                    .map(|reason| format!("（参数修改受限：{reason}）"))
                    .unwrap_or_default()
            )
        } else if let Some(reason) = source_block {
            format!("参数修改受限：{reason}")
        } else {
            "当前没有受支持的源码编辑器；观察结果不会写回配置".to_owned()
        }
    ));
    if let Some(fields) = row["editable_params"].as_object()
        && !fields.is_empty()
    {
        lines.push(format!("变体可改字段：{}", field_names(fields)));
    }
    if let Some(fields) = row["source_editable_fields"].as_array()
        && !fields.is_empty()
    {
        let fields = fields.iter().filter_map(Value::as_str).collect::<Vec<_>>();
        lines.push(format!("源码可改字段：{}", fields.join("、")));
    }
    if row["can_insert"] == true || row["can_source_insert"] == true {
        lines.push("容器操作：可在此容器中插入子 Layer。".into());
    }
    lines.push("以上详情来自当前已取得的模型树报告；展开不会重新运行模型。".into());
    lines
}

fn append_probe_evidence(lines: &mut Vec<String>, row: &Value, probe: Option<&Value>) {
    let Some(probe) = probe else {
        lines
            .push("形状/调用：尚无与当前模型输入匹配的前向证据；显式执行前向探测后可查看。".into());
        lines.push("查看详情不会自动运行模型。".into());
        return;
    };
    let instance_id = row["instance_id"].as_str();
    let aliases = row["aliases"].as_array();
    let calls = probe["calls"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|call| {
            instance_id.is_some_and(|id| call["module_instance_id"].as_str() == Some(id))
                || aliases.is_some_and(|aliases| {
                    call["paths"].as_array().is_some_and(|paths| {
                        paths
                            .iter()
                            .filter_map(Value::as_str)
                            .any(|path| aliases.iter().any(|alias| alias.as_str() == Some(path)))
                    })
                })
        })
        .collect::<Vec<_>>();
    let trace_truncated = probe["trace_truncated"] == true;
    if let Some(mode) = probe["observed_mode"].as_str() {
        lines.push(format!("前向模式：{mode}"));
    }
    if calls.is_empty() {
        lines.push(if trace_truncated {
            "本次调用：未出现在保留记录中；由于调用轨迹已截断，不能据此判断为 0 次。".into()
        } else {
            "本次前向调用：0（当前批次未触发此 Layer）".into()
        });
        return;
    }
    lines.push(if trace_truncated {
        format!("本次已记录前向调用：至少 {} 次", calls.len())
    } else {
        format!("本次前向调用：{} 次", calls.len())
    });
    for call in calls {
        let index = call["invocation_index"].as_u64().unwrap_or(0);
        let mut inputs = Vec::new();
        collect_tensor_shapes(&call["inputs"], "输入", &mut inputs);
        let mut outputs = Vec::new();
        collect_tensor_shapes(&call["outputs"], "输出", &mut outputs);
        lines.push(format!(
            "  第 {index} 次 · {}",
            call["module_type"].as_str().unwrap_or("未知类型")
        ));
        lines.extend(inputs.into_iter().map(|shape| format!("    {shape}")));
        lines.extend(outputs.into_iter().map(|shape| format!("    {shape}")));
    }
    if trace_truncated {
        lines.push("调用记录已截断；显示次数不代表未记录调用的具体输入输出。".into());
    }
    if let Some(coverage) = probe["coverage"].as_str() {
        lines.push(format!("探测覆盖：{coverage}"));
    }
}

fn collect_tensor_shapes(value: &Value, label: &str, shapes: &mut Vec<String>) {
    if value["kind"] == "tensor" {
        shapes.push(format!(
            "{label} shape={} dtype={}",
            value["shape"], value["dtype"]
        ));
        return;
    }
    if let Some(items) = value["items"].as_object() {
        for (key, child) in items {
            collect_tensor_shapes(child, &format!("{label}.{key}"), shapes);
        }
    } else if let Some(items) = value["items"].as_array() {
        for (index, child) in items.iter().enumerate() {
            collect_tensor_shapes(child, &format!("{label}[{index}]"), shapes);
        }
    }
}

fn append_control_source(lines: &mut Vec<String>, path: &str, row: &Value, report: &Value) {
    let aliases = row["aliases"].as_array();
    let evidence = report
        .pointer("/origin/injection_evidence")
        .and_then(Value::as_array)
        .and_then(|entries| {
            entries.iter().find(|entry| {
                entry["registered_paths"].as_array().is_some_and(|paths| {
                    paths.iter().filter_map(Value::as_str).any(|registered| {
                        registered == path
                            || aliases.is_some_and(|aliases| {
                                aliases
                                    .iter()
                                    .any(|alias| alias.as_str() == Some(registered))
                            })
                    })
                })
            })
        });
    if let Some(evidence) = evidence {
        lines.push(format!(
            "控制来源：显式装配接缝 {} · {}",
            evidence["slot"].as_str().unwrap_or("未知接缝"),
            evidence["factory"].as_str().unwrap_or("工厂来源未知")
        ));
        if let Some(status) = evidence["sample_forward"].as_str() {
            lines.push(format!("装配样本前向：{status}"));
        }
    } else if report["variant_applied"] == true
        || report
            .pointer("/origin/variant_applied")
            .and_then(Value::as_bool)
            == Some(true)
    {
        lines.push("控制来源：代码/装配基础上的显式变体；当前节点的变更以变体差异为准。".into());
    } else {
        lines.push("控制来源：Python 模型实例；未声明的内部 Layer 不会反写成配置。".into());
    }
}

fn append_location(lines: &mut Vec<String>, label: &str, value: &Value) {
    if let Some(location) = value.as_object() {
        let file = location
            .get("file")
            .and_then(Value::as_str)
            .unwrap_or("未知");
        let line = location.get("line").and_then(Value::as_u64);
        lines.push(match line {
            Some(line) => format!("{label}：{file}:{line}"),
            None => format!("{label}：{file}"),
        });
    }
}

fn field_names(fields: &serde_json::Map<String, Value>) -> String {
    fields.keys().cloned().collect::<Vec<_>>().join("、")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detail_explains_source_shared_impact_and_available_edit_paths() {
        let row = json!({
            "path":"model.encoder.proj", "type":"Linear", "instance_id":"m-7",
            "aliases":["model.encoder.proj","model.decoder.proj"],
            "own_parameters":80,"parameters":80,
            "construction_site":{"file":"model.py","line":24},
            "source_definition":{"file":"layers.py","line":12},
            "can_set":true,"can_source_set":false,"editable_params":{"out_features":16},
            "source_set_block_reason":"有状态参数；需要显式迁移。"
        });
        let report = json!({"variant_applied":false});
        let detail = render_layer_detail(&row, &report, None).join("\n");
        assert!(detail.contains("model.encoder.proj"));
        assert!(detail.contains("model.decoder.proj"));
        assert!(detail.contains("共享参数只按同一个模块身份统计"));
        assert!(detail.contains("model.py:24"));
        assert!(detail.contains("out_features"));
        assert!(detail.contains("需要显式迁移"));
        assert!(!detail.contains("源码操作：存在受支持的源码编辑能力"));
        assert!(detail.contains("Python 模型实例"));
        assert!(detail.contains("尚无与当前模型输入匹配的前向证据"));
        assert!(detail.contains("展开不会重新运行模型"));
    }

    #[test]
    fn declared_injection_is_shown_as_assembly_control_source() {
        let row = json!({
            "path":"model.encoder","type":"Linear",
            "aliases":["model.encoder","model.shared_encoder"]
        });
        let report = json!({"origin":{"injection_evidence":[{
            "slot":"encoder","factory":"components:encoder","registered_paths":["model.shared_encoder"],"sample_forward":"passed"
        }]}});
        let detail = render_layer_detail(&row, &report, None).join("\n");
        assert!(detail.contains("控制来源：显式装配接缝 encoder"));
        assert!(detail.contains("装配样本前向：passed"));
    }

    #[test]
    fn matching_probe_report_shows_per_invocation_shapes_without_running_again() {
        let row = json!({
            "path":"model.shared","type":"Linear","instance_id":"module-2",
            "aliases":["model.shared","model.head"]
        });
        let report = json!({});
        let probe = json!({
            "calls":[{
                "module_instance_id":"module-2","paths":["model.shared","model.head"],
                "module_type":"Linear","invocation_index":2,
                "inputs":{"kind":"dict","items":{"args":{"kind":"tuple","items":[{"kind":"tensor","shape":[3,4],"dtype":"torch.float32"}]}}},
                "outputs":{"kind":"tensor","shape":[3,8],"dtype":"torch.float32"}
            }],
            "trace_truncated":false,"coverage":"仅当前批次"
        });
        let detail = render_layer_detail(&row, &report, Some(&probe)).join("\n");
        assert!(detail.contains("本次前向调用：1 次"));
        assert!(detail.contains("第 2 次"));
        assert!(detail.contains("输入.args[0] shape=[3,4]"));
        assert!(detail.contains("输出 shape=[3,8]"));
        assert!(detail.contains("探测覆盖：仅当前批次"));
    }

    #[test]
    fn truncated_probe_does_not_claim_missing_layer_calls_are_zero() {
        let row = json!({"path":"model.rare","type":"Linear","instance_id":"module-9"});
        let report = json!({});
        let probe = json!({"calls":[],"trace_truncated":true});
        let detail = render_layer_detail(&row, &report, Some(&probe)).join("\n");
        assert!(detail.contains("不能据此判断为 0 次"));
    }
}
