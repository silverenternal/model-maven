use crate::domain::Report;
use serde_json::Value;

pub fn format_doctor_summary(data: &Value) -> String {
    let mut lines = vec!["Model-Maven 环境与能力诊断".to_owned()];
    lines.push(format!(
        "Python：{} ({})",
        data["python"].as_str().unwrap_or("未知"),
        data["python_version"].as_str().unwrap_or("未知版本")
    ));
    let torch = if data["torch_available"] == true {
        data["torch_version"]
            .as_str()
            .map(|version| format!("可用 · {version}"))
            .unwrap_or_else(|| "可用".into())
    } else {
        "不可用".into()
    };
    lines.push(format!("PyTorch：{torch}"));
    lines.push(format!(
        "Worker：协议 v{} · API v{}",
        data["compatibility"]["worker_protocol_version"]
            .as_u64()
            .map(|version| version.to_string())
            .unwrap_or_else(|| "未知".into()),
        data["compatibility"]["worker_api_version"]
            .as_u64()
            .map(|version| version.to_string())
            .unwrap_or_else(|| "未知".into())
    ));

    let capabilities = data["capabilities"].as_array().cloned().unwrap_or_default();
    let supported = capabilities
        .iter()
        .filter(|capability| capability["state"] == "supported")
        .count();
    lines.push(format!(
        "\n当前可用能力：{supported}/{}",
        capabilities.len()
    ));
    for capability in capabilities {
        let id = capability["id"].as_str().unwrap_or("unknown");
        let label = capability_label(id);
        let state = match capability["state"].as_str() {
            Some("supported") => "可用",
            Some("conditional") => "有条件",
            Some("unsupported") => "不可用",
            _ => "未确认",
        };
        lines.push(format!("  {state} · {label}"));
        if let Some(reason) = capability["reason"].as_str() {
            if !reason.is_empty() {
                lines.push(format!("    原因：{reason}"));
            }
        }
    }
    if let Some(hint) = data["hint"].as_str() {
        lines.push(format!("\n下一步：{hint}"));
    }
    lines.join("\n")
}

/// Render an operation readiness plan without exposing resolved input values.
pub fn format_readiness_summary(plan: &Value) -> String {
    let readiness = &plan["readiness"];
    let mut lines = vec![format!(
        "操作就绪状态：{} · {}",
        readiness_state_label(readiness["state"].as_str().unwrap_or("unknown")),
        readiness_operation_label(plan["operation"].as_str().unwrap_or("未知操作"))
    )];

    if let Some(revision) = plan["base_revision"].as_str() {
        lines.push(format!("配置版本：{}", short_revision(revision)));
    }

    if let Some(inputs) = plan["inputs"].as_array() {
        for input in inputs {
            let key = input["key"].as_str().unwrap_or("未知输入");
            let required = input["required"] == true;
            let state = fact_state_label(input["state"].as_str().unwrap_or("unknown"));
            let source = &input["source"];
            let source_kind = input_source_label(source["kind"].as_str().unwrap_or("unknown"));
            let reference = source["reference"]
                .as_str()
                .map(|reference| format!(" · {reference}"))
                .unwrap_or_default();
            let source_revision = source["revision"]
                .as_str()
                .map(|revision| format!(" · revision {}", short_revision(revision)))
                .unwrap_or_default();
            lines.push(format!(
                "  {} · {} · {} · 来源：{}{}{}",
                key,
                if required { "必需" } else { "可选" },
                state,
                source_kind,
                reference,
                source_revision
            ));
        }
    }

    if let Some(facts) = readiness["facts"].as_array() {
        for fact in facts {
            let key = fact["key"].as_str().unwrap_or("未知条件");
            let state = fact_state_label(fact["state"].as_str().unwrap_or("unknown"));
            lines.push(format!(
                "  条件 · {key} · {} · {}",
                if fact["required"] == true {
                    "必需"
                } else {
                    "可选"
                },
                state
            ));
            if let Some(reason) = fact["reason"].as_str() {
                lines.push(format!("    {reason}"));
            }
        }
    }

    if let Some(capabilities) = readiness["capabilities"].as_array() {
        for capability in capabilities {
            let id = capability["id"].as_str().unwrap_or("unknown");
            lines.push(format!(
                "  能力 · {} · {}",
                capability_label(id),
                fact_state_label(capability["state"].as_str().unwrap_or("unknown"))
            ));
            if let Some(reason) = capability["reason"].as_str() {
                lines.push(format!("    {reason}"));
            }
        }
    }

    if let Some(conditions) = readiness["conditions"].as_array() {
        for condition in conditions.iter().filter_map(Value::as_str) {
            lines.push(format!("  注意 · {condition}"));
        }
    }

    if let Some(actions) = readiness["actions"].as_array() {
        for action in actions {
            let label = action["label"].as_str().unwrap_or("查看建议操作");
            let availability = action["availability"].as_str().unwrap_or("unknown");
            lines.push(format!(
                "下一步 · {} · {}",
                label,
                action_availability_label(availability)
            ));
            if let Some(reason) = action["reason"].as_str() {
                lines.push(format!("  {reason}"));
            }
        }
    }

    lines.join("\n")
}

fn readiness_state_label(state: &str) -> &str {
    match state {
        "ready" => "可执行",
        "needs_input" => "需要输入",
        "conditional" => "有条件",
        "unsupported" => "不支持",
        _ => "待确认",
    }
}

fn comparison_state(state: Option<&str>) -> &'static str {
    match state {
        Some("same") => "相同",
        Some("changed") => "有差异",
        _ => "信息不足",
    }
}

fn readiness_operation_label(operation: &str) -> &str {
    match operation {
        "inspect_model" => "模型结构检查",
        "forward_probe" => "前向试跑",
        "loss_probe" => "Loss 与梯度检查",
        "managed_train" => "管理式训练",
        "manual_train" => "用户训练步",
        "external_train" => "用户训练循环",
        "component_inspect" => "组件详情",
        "component_check" => "组件依赖检查",
        "component_test" => "组件符合性测试",
        "plan_layer_edit" => "Layer 编辑预览",
        "apply_layer_edit" => "Layer 编辑提交",
        "layer_history" => "Layer 编辑历史",
        "layer_query" => "Layer 查询",
        "edit_model" => "模型编辑",
        _ => operation,
    }
}

fn fact_state_label(state: &str) -> &str {
    match state {
        "available" => "已就绪",
        "missing" => "缺少",
        "unsupported" => "不支持",
        "conditional" => "有条件",
        _ => "待确认",
    }
}

fn input_source_label(kind: &str) -> &str {
    match kind {
        "project_config" => "项目配置",
        "command_line" => "本次命令",
        "user_session" => "工作会话",
        "saved_run" => "已保存运行",
        "model_variant" => "模型变体",
        "data_adapter" => "数据适配器",
        "user_callable" => "用户代码",
        "generated" => "生成内容",
        "external" => "外部来源",
        _ => "来源未确认",
    }
}

fn action_availability_label(availability: &str) -> &str {
    match availability {
        "available" => "可用",
        "conditional" => "有条件",
        "unsupported" => "不支持",
        _ => "待确认",
    }
}

fn short_revision(revision: &str) -> &str {
    revision.get(..12).unwrap_or(revision)
}

fn capability_label(id: &str) -> &str {
    match id {
        "model.inspect" => "模型结构检查",
        "model.forward_probe" => "前向试跑与调用取证",
        "model.loss_probe" => "Loss 与梯度检查",
        "model.layer_edit" => "Layer 结构编辑",
        "training.managed_step" => "管理式训练步",
        "training.manual_step" => "用户训练步",
        "training.external_loop" => "用户完整训练循环",
        "training.checkpoint_resume" => "checkpoint 续训",
        "training.detached" => "后台训练监督",
        "data.dataset_probe" => "批次与数据采样检查",
        "extensions.static_discovery" => "扩展组件发现",
        "extensions.layer_conformance" => "Layer 扩展符合性检查",
        "environment.setup" => "项目 Python 环境准备",
        _ => id,
    }
}
use std::io::{self, IsTerminal, Write};

pub fn is_interactive(json_mode: bool, no_interactive: bool) -> bool {
    !json_mode && !no_interactive && io::stdin().is_terminal() && io::stderr().is_terminal()
}

pub fn stderr_is_terminal() -> bool {
    io::stderr().is_terminal()
}

pub fn prompt(label: &str) -> Option<String> {
    prompt_allow_empty(label).filter(|value| !value.is_empty())
}

pub fn prompt_allow_empty(label: &str) -> Option<String> {
    eprint!("{label}: ");
    io::stderr().flush().ok()?;
    let mut value = String::new();
    if io::stdin().read_line(&mut value).ok()? == 0 {
        return None;
    }
    Some(value.trim().to_string())
}

fn describe_tree(value: &Value) -> String {
    if value["kind"] == "tensor" {
        return format!(
            "Tensor {} {}",
            value["shape"],
            value["dtype"].as_str().unwrap_or("")
        );
    }
    if let Some(items) = value["items"].as_object() {
        return items
            .iter()
            .map(|(k, v)| format!("{k}: {}", describe_tree(v)))
            .collect::<Vec<_>>()
            .join("; ");
    }
    if let Some(items) = value["items"].as_array() {
        return format!(
            "[{}]",
            items
                .iter()
                .map(describe_tree)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    value.to_string()
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut amount = bytes as f64;
    let mut unit = 0;
    while amount >= 1024.0 && unit < UNITS.len() - 1 {
        amount /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{amount:.1} {}", UNITS[unit])
    }
}

fn display_edit_effect(effect: &Value) {
    let action = effect["action"].as_str().unwrap_or("edit");
    if action == "clone-selected" {
        println!(
            "共享关系拆分：{} · 保留共享路径 {}",
            effect["selected_path"].as_str().unwrap_or("未知路径"),
            effect["retained_aliases"]
                .as_array()
                .map(|values| values
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("、"))
                .unwrap_or_else(|| "信息不足".into())
        );
        println!("  拆分副本保留原模块参数与 buffers；之后两条路径分别更新。");
        return;
    }
    if action == "share-selected" {
        println!(
            "共享关系连接：{} → {}",
            effect["source_path"].as_str().unwrap_or("未知路径"),
            effect["target_paths"]
                .as_array()
                .map(|values| values
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("、"))
                .unwrap_or_else(|| "信息不足".into())
        );
        println!("  目标模块原有参数与 buffers 将被源模块实例取代。");
        return;
    }
    if matches!(action, "undo" | "redo") {
        println!("Layer 历史：{} · 当前游标 {}", action, effect["cursor"]);
        return;
    }
    if action == "custom" {
        let provider = effect["provider"].as_str().unwrap_or("custom provider");
        let summary = effect["result"]
            .as_object()
            .map(|result| {
                result
                    .iter()
                    .map(|(key, value)| format!("{key}={value}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| effect["result"].to_string());
        println!("Layer 自定义编辑：{provider}");
        if !summary.is_empty() && summary != "null" {
            println!("  Provider 摘要：{summary}");
        }
        return;
    }

    let path = if action == "add" {
        format!(
            "{}.{}",
            effect["path"].as_str().unwrap_or(""),
            effect["name"].as_str().unwrap_or("")
        )
    } else {
        effect["path"].as_str().unwrap_or("").to_owned()
    };
    let description = if action == "remove" && effect["strategy"] == "bypass" {
        "Layer 移除（替换为 Identity）"
    } else {
        match action {
            "add" => "Layer 新增",
            "set" => "Layer 参数更新",
            "replace" => "Layer 替换",
            "remove" => "Layer 删除",
            _ => "Layer 操作",
        }
    };
    let module_type = effect["module_type"].as_str();
    if action == "add" {
        if let Some(module_type) = module_type {
            println!("{description}：{path} · {module_type}");
        } else {
            println!("{description}：{path}");
        }
    } else {
        println!("{description}：{path}");
    }

    if let Some(aliases) = effect["affected_aliases"].as_array()
        && aliases.len() > 1
    {
        println!(
            "  已同步更新共享实例的所有引用：{}",
            aliases
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if let Some(parameters) = effect["preserved"].as_array()
        && !parameters.is_empty()
    {
        println!(
            "  保留参数：{}",
            parameters
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if let Some(parameters) = effect["initialized"].as_array()
        && !parameters.is_empty()
    {
        println!(
            "  新初始化参数：{}；旧权重不会自动迁移",
            parameters
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

pub fn display(report: &Report, json: bool) {
    if json {
        println!(
            "{}",
            serde_json::to_string(report).expect("report serializes")
        );
        return;
    }
    display_input_provenance(report);
    if report.data["readiness"].is_object() {
        println!("{}", format_readiness_summary(&report.data["readiness"]));
    }
    if report.operation == "preview" {
        for line in crate::presentation::pages::preview::render(report) {
            println!("{line}");
        }
        return;
    }
    if report.operation == "component_test" {
        if let Some(seed) = report.data["seed_decimal"].as_str() {
            println!("  本次验证种子：{seed}（仅覆盖本次执行；可用 --seed 复用）");
        }
    }
    if !report.success() {
        if report.operation == "evaluate" {
            for line in crate::presentation::pages::evaluation::render(report) {
                eprintln!("{line}");
            }
            return;
        }
        let diagnostic_lines = crate::presentation::pages::diagnostic::render(report, true);
        for line in diagnostic_lines {
            if report.status == "inconclusive" {
                println!("{line}");
            } else {
                eprintln!("{line}");
            }
        }
        if report.status == "inconclusive" {
            return;
        }
        if report.operation == "training_probe" {
            for (label, key) in [
                ("修改文件", "changed_files"),
                ("缺失文件", "missing_files"),
                ("新增文件", "new_files"),
            ] {
                if let Some(paths) = report.data[key].as_array()
                    && !paths.is_empty()
                {
                    eprintln!(
                        "{label}：{}",
                        paths
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("、")
                    );
                }
            }
        }
    }
    if report.operation == "doctor" {
        if report.success() {
            println!("{}", format_doctor_summary(&report.data));
        }
        return;
    }
    if report.operation == "preset_catalog" && report.success() {
        let presets = report.data["presets"].as_object();
        let query = report.data["query"].as_str();
        println!(
            "内置模型预设{}",
            query
                .map(|value| format!(" · 搜索：{value}"))
                .unwrap_or_default()
        );
        match presets {
            Some(presets) if !presets.is_empty() => {
                for (name, preset) in presets {
                    let task = preset["task"].as_str().unwrap_or("未分类任务");
                    let input = &preset["default_input"];
                    let input_shape = if input.is_object() {
                        format!(
                            "{}×{}×{}",
                            input["channels"].as_u64().unwrap_or(0),
                            input["height"].as_u64().unwrap_or(0),
                            input["width"].as_u64().unwrap_or(0)
                        )
                    } else {
                        "由模型决定".into()
                    };
                    let weights = preset["weights"]
                        .as_array()
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join("、")
                        })
                        .unwrap_or_else(|| "未声明".into());
                    println!("\n{name} · {task}");
                    println!("  默认输入：{input_shape} · 权重选项：{weights}");
                    if preset["download_is_explicit"] == true {
                        println!(
                            "  官方权重仅在构建时显式选择后下载；运行 m-model doctor 检查当前环境。"
                        );
                    }
                    if let Some(entrypoint) = preset["model"].as_str() {
                        println!("  模型工厂：{entrypoint}");
                    }
                }
            }
            _ => println!("没有匹配项。运行 m-model data presets 查看全部预设。"),
        }
        println!("\n下一步：m-model data presets <关键词> · --json 可获取结构化目录。 ");
        return;
    }
    if report.operation == "config_history" && report.success() {
        let current = report.data["current_revision"]
            .as_str()
            .map(short_revision)
            .unwrap_or("未知");
        println!("当前项目配置 · revision {current}");
        let history = report.data["history"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if history.is_empty() {
            println!(
                "还没有历史版本。之后通过 m-model assembly add/set/remove 修改配置时会自动记录。"
            );
            return;
        }
        for (index, entry) in history.iter().enumerate() {
            let revision = entry["revision"]
                .as_str()
                .map(short_revision)
                .unwrap_or("无效 revision");
            let model = entry["config"]["model"].as_str().unwrap_or("未设置");
            let batch = entry["config"]["batch"].as_str().unwrap_or("未设置批次");
            println!(
                "{}. revision {revision} · 模型 {model} · 批次 {batch}",
                index + 1
            );
        }
        println!("\n查看详情：m-model config restore <revision> --field <字段>（仅预览）");
        return;
    }
    if report.operation == "config_set_batch" && report.success() {
        println!(
            "默认批次{}：{}",
            if report.data["committed"] == true {
                "已更新"
            } else if report.data["unchanged"] == true {
                "未变化"
            } else {
                "预览"
            },
            report.data["batch"].as_str().unwrap_or("未知")
        );
        println!(
            "批次绑定：{}",
            report.data["batch_binding"].as_str().unwrap_or("自动推断")
        );
        if report.data["committed"] != true && report.data["unchanged"] != true {
            println!(
                "当前配置修订：{}",
                report.data["from_revision"].as_str().unwrap_or("未知")
            );
            println!(
                "确认后重复本命令，并添加 --apply --if-current <当前配置修订>。此预览未运行批次或前向。"
            );
        } else {
            println!("{}", report.data["note"].as_str().unwrap_or(""));
        }
        return;
    }
    if report.operation == "config_restore" && report.success() {
        println!(
            "配置字段恢复{}：{} → {}",
            if report.data["committed"] == true {
                "已提交"
            } else {
                "预览"
            },
            report.data["from_revision"]
                .as_str()
                .map(short_revision)
                .unwrap_or("未知"),
            report.data["restored_from_revision"]
                .as_str()
                .map(short_revision)
                .unwrap_or("未知")
        );
        if let Some(fields) = report.data["fields"].as_array() {
            for field in fields {
                let name = field["field"].as_str().unwrap_or("unknown");
                println!(
                    "  {name}：{} → {}{}",
                    config_value_label(&field["current"]),
                    config_value_label(&field["restored"]),
                    if field["changed"] == true {
                        ""
                    } else {
                        " · 无变化"
                    }
                );
            }
        }
        if report.data["unchanged"] == true {
            println!("{}", report.data["note"].as_str().unwrap_or(""));
            return;
        }
        if report.data["committed"] != true {
            println!("{}", report.data["note"].as_str().unwrap_or(""));
            if let Some(revision) = report.data["restored_from_revision"].as_str()
                && let Some(fields) = report.data["fields"].as_array()
            {
                for field in fields.iter().filter_map(|field| field["field"].as_str()) {
                    println!(
                        "提交此字段：m-model config restore {revision} --field {field} --apply --if-current {}",
                        report.data["from_revision"]
                            .as_str()
                            .unwrap_or("<预览revision>")
                    );
                }
            }
        } else {
            println!("新配置 revision：{}", report.data["config_revision"]);
            println!("{}", report.data["note"].as_str().unwrap_or(""));
        }
        return;
    }
    if report.operation == "environment_lock" && report.success() {
        println!(
            "依赖锁已{}：{} 个包 · {}",
            if report.data["cached"] == true {
                "复用"
            } else {
                "生成"
            },
            report.data["package_count"].as_u64().unwrap_or(0),
            report.data["lock_file"]
                .as_str()
                .unwrap_or("requirements.lock.txt")
        );
        println!("{}", report.data["note"].as_str().unwrap_or(""));
        println!("下一步：m-model env setup");
        return;
    }
    if report.operation == "environment_setup" && report.success() {
        let interpreter = report.data["python"].as_str().unwrap_or(".venv/bin/python");
        println!(
            "项目 Python 环境已{}：{}",
            if report.data["created"] == true {
                "创建并验证"
            } else {
                "复用并验证"
            },
            report.data["venv"].as_str().unwrap_or(".venv")
        );
        println!("解释器：{interpreter}");
        println!(
            "已核对 {} 个锁定包和 pip check。",
            report.data["package_count"].as_u64().unwrap_or(0)
        );
        println!("后续 m-model 命令会自动使用项目虚拟环境；显式 --python 仍可覆盖。");
        return;
    }
    if report.operation == "environment_check" && report.success() {
        let check = &report.data["environment_check"];
        println!(
            "项目 Python 环境符合依赖锁。解释器：{}",
            check["python"]["version"].as_str().unwrap_or("未知")
        );
        println!("{}", report.data["note"].as_str().unwrap_or(""));
        return;
    }
    if report.operation == "environment_library_inventory" && report.success() {
        println!(
            "可选 Python 库 · {}",
            report.data["python_executable"]
                .as_str()
                .unwrap_or("所选解释器")
        );
        println!("发现模式：只读 distribution metadata；未导入库、未安装依赖、未绑定后端。\n");
        for library in report.data["libraries"].as_array().into_iter().flatten() {
            let name = library["distribution"].as_str().unwrap_or("unknown");
            let versions = library["installed_versions"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|item| {
                    Some(format!(
                        "{}=={}",
                        item["name"].as_str()?,
                        item["version"].as_str()?
                    ))
                })
                .collect::<Vec<_>>();
            println!(
                "{name} · {}",
                if versions.is_empty() {
                    "未发现已安装 distribution".into()
                } else {
                    versions.join(", ")
                }
            );
            println!(
                "  Model-Maven 适配：{}",
                match library["adapter_status"].as_str() {
                    Some("implemented") => "有专用执行路径（PyTorch 原生）",
                    Some("package-only") => "未提供专用训练/模型后端；可由用户代码显式导入",
                    _ => "未知包；不推断适配能力",
                }
            );
        }
        println!("\n{}", report.data["note"].as_str().unwrap_or(""));
        return;
    }
    if report.operation == "environment_library_probe" && report.success() {
        println!(
            "显式 Python 库导入探测 · {}",
            report.data["python_executable"]
                .as_str()
                .unwrap_or("所选解释器")
        );
        println!("仅对用户点名的候选库尝试导入；没有安装依赖，也没有绑定训练后端。\n");
        for probe in report.data["probes"].as_array().into_iter().flatten() {
            let name = probe["distribution"].as_str().unwrap_or("unknown");
            match probe["status"].as_str().unwrap_or("unknown") {
                "imported" => println!(
                    "{name} · 导入成功 · {}",
                    probe["import_version"]
                        .as_str()
                        .or_else(|| probe["distribution_version"].as_str())
                        .unwrap_or("版本未知")
                ),
                "not_installed" => println!("{name} · 当前解释器未安装"),
                "import_failed" => println!(
                    "{name} · 导入失败 · {}: {}",
                    probe["failure_type"].as_str().unwrap_or("ImportError"),
                    probe["failure_reason"].as_str().unwrap_or("未提供错误详情")
                ),
                _ => println!("{name} · 探测状态未知"),
            }
            println!(
                "  Model-Maven 后端：{}",
                match probe["adapter_status"].as_str() {
                    Some("implemented") => "PyTorch 原生执行路径",
                    _ => "无专用执行后端声明",
                }
            );
        }
        println!("\n{}", report.data["note"].as_str().unwrap_or(""));
        return;
    }
    if report.operation == "environment_wheelhouse_archive" && report.success() {
        println!(
            "已归档 {} 个锁定 wheel：{}",
            report.data["file_count"].as_u64().unwrap_or(0),
            report.data["path"].as_str().unwrap_or("未报告路径")
        );
        println!("该归档仅适用于锁定时的 Python ABI；OS、驱动与加速器未包含。 ");
        return;
    }
    if report.operation == "environment_wheelhouse_check" {
        println!(
            "Wheel 归档：索引 {} · 文件 {} · ABI {}",
            if report.data["index_available"] == true {
                "可读"
            } else {
                "缺失/无效"
            },
            if report.data["archive_available"] == true {
                "完整"
            } else {
                "缺失/损坏"
            },
            if report.data["compatible"] == true {
                "匹配"
            } else {
                "不匹配"
            }
        );
        for (label, key) in [("缺失", "missing"), ("损坏", "corrupt")] {
            if let Some(files) = report.data[key]
                .as_array()
                .filter(|files| !files.is_empty())
            {
                let files = files.iter().filter_map(Value::as_str).collect::<Vec<_>>();
                println!("{label}文件：{}", files.join("、"));
            }
        }
        return;
    }
    if report.operation == "environment_wheelhouse_replay" && report.success() {
        println!(
            "已从本地 wheel 归档离线安装依赖：{}",
            report.data["venv"].as_str().unwrap_or("目标环境")
        );
        println!("{}", report.data["note"].as_str().unwrap_or(""));
        return;
    }
    if matches!(
        report.operation.as_str(),
        "environment_check" | "environment_setup"
    ) && !report.success()
        && report.data.get("missing").is_some()
    {
        println!(
            "现有环境差异：{} 个缺失、{} 个版本不符、{} 个额外包。",
            report.data["missing"].as_array().map_or(0, Vec::len),
            report.data["version_mismatch"]
                .as_array()
                .map_or(0, Vec::len),
            report.data["extra"].as_array().map_or(0, Vec::len)
        );
        if report.data["python_matches"] == false {
            println!("Python 版本、ABI 或目标平台与依赖锁不同。");
        }
        if report.data["requirements_match"] == false {
            println!("requirements.txt 或递归包含的依赖输入已变化。");
        }
        if let Some(check) = report.data["pip_check"].as_str()
            && !check.is_empty()
        {
            println!("依赖关系检查：{check}");
        }
    }
    if report.operation == "init" && report.success() {
        println!(
            "{}项目默认入口：{}",
            if report.data["committed"] == true {
                "已保存 "
            } else {
                "初始化预览 · "
            },
            report.data["path"].as_str().unwrap_or("")
        );
        println!("模型：{}", report.data["config"]["model"]);
        if let Some(batch) = report.data["config"]["batch"].as_str() {
            println!("批次：{batch}");
            println!("可直接运行：m-model model probe");
        } else {
            println!("尚未设置默认批次；前向验证时可通过 --batch 提供。");
        }
        if let Some(injections) = report.data["config"]["model_injections"].as_object() {
            for (slot, factory) in injections {
                println!(
                    "模型接缝：{slot} ← {}",
                    factory.as_str().unwrap_or("无效入口")
                );
            }
        }
        println!("后续命令可省略已保存的 --model/--batch；显式参数优先。");
        println!("{}", report.data["note"].as_str().unwrap_or(""));
        return;
    }
    if report.operation.starts_with("assembly_") && report.success() {
        if report.operation == "assembly_show" {
            println!(
                "项目模型：{}",
                report.data["model"].as_str().unwrap_or("未知")
            );
            println!("配置：{}", report.data["path"].as_str().unwrap_or(""));
            let slots = report.data["slots"].as_array().cloned().unwrap_or_default();
            if slots.is_empty() {
                println!(
                    "当前没有显式装配接缝。可用 `m-model assembly add <slot>` 添加根工厂接缝。"
                );
            } else {
                for slot in slots {
                    let name = slot["slot"].as_str().unwrap_or("<unknown>");
                    let paths = slot["registered_paths"]
                        .as_array()
                        .map(|paths| {
                            paths
                                .iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();
                    let binding = if let Some(value) = slot["binding"].as_str() {
                        value.to_owned()
                    } else if slot["component_ref"].is_object() {
                        format!(
                            "{}@{}",
                            slot["component_ref"]["id"].as_str().unwrap_or("component"),
                            slot["component_ref"]["version"]
                                .as_str()
                                .unwrap_or("unknown")
                        )
                    } else {
                        slot["factory"]
                            .as_str()
                            .unwrap_or("registered component")
                            .to_owned()
                    };
                    println!(
                        "{name} ← {}{}",
                        binding,
                        if paths.is_empty() {
                            String::new()
                        } else {
                            format!(" · 注册路径 {paths}")
                        }
                    );
                    if let Some(params) = slot["params"]
                        .as_object()
                        .filter(|params| !params.is_empty())
                    {
                        println!(
                            "  参数：{}",
                            serde_json::to_string(params).unwrap_or_default()
                        );
                    }
                    if slot["sample_forward"] == "not_observed" {
                        println!("  提示：该模块已注册，但示例批次没有执行此路径。");
                    }
                }
            }
            println!(
                "验证：构造 {} · 前向 {}",
                report.data["validation"]["construction"], report.data["validation"]["forward"]
            );
            if let Some(readiness) = report.data.get("readiness") {
                println!("\n{}", format_readiness_summary(readiness));
            }
        } else {
            let action = report.data["action"].as_str().unwrap_or("edit");
            let slot = report.data["slot"].as_str().unwrap_or("");
            let state = if report.data["committed"] == true {
                "已保存"
            } else {
                "预览，未更改配置"
            };
            println!("装配接缝 {action}：{slot} · {state}");
            println!("配置：{}", report.data["path"].as_str().unwrap_or(""));
            println!(
                "验证：构造 {} · 前向 {}",
                report.data["validation"]["construction"], report.data["validation"]["forward"]
            );
        }
        println!("{}", report.data["note"].as_str().unwrap_or(""));
        return;
    }
    if report.operation == "variant_list" && report.success() {
        let variants = report.data["variants"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if variants.is_empty() {
            println!("本项目还没有模型变体。首次 Layer 编辑会创建 working 变体。");
            return;
        }
        for variant in variants {
            let name = variant["name"].as_str().unwrap_or("<invalid-name>");
            if variant["readable"] != true {
                println!("{name} · 无法读取：{}", variant["error"]);
                continue;
            }
            let revision = variant["revision"].as_str().unwrap_or("未知");
            let revision = revision.get(..12).unwrap_or(revision);
            println!(
                "{name} · {} · 编辑 {}/{} · 设备策略 {} · revision {}",
                variant["model"].as_str().unwrap_or("模型入口未知"),
                variant["cursor"].as_u64().unwrap_or(0),
                variant["edit_count"].as_u64().unwrap_or(0),
                variant["device_policy"].as_str().unwrap_or("未知"),
                revision
            );
        }
        return;
    }
    if report.operation == "variant_show" && report.success() {
        let variant = &report.data["variant"];
        println!(
            "变体：{}\n模型：{}\n编辑游标：{}/{}\nrevision：{}",
            report.data["name"].as_str().unwrap_or(""),
            variant
                .pointer("/origin/model")
                .and_then(Value::as_str)
                .unwrap_or("未知"),
            variant["cursor"],
            variant["edits"].as_array().map_or(0, Vec::len),
            variant["revision"].as_str().unwrap_or("未知")
        );
        println!("设备重建策略：{}（查看文件本身不执行重建）",
            variant.get("device_policy").map_or("legacy_cpu", |value| value.as_str().unwrap_or("未知")));
        if let Some(edits) = variant["edits"].as_array() {
            for (index, edit) in edits.iter().enumerate() {
                println!(
                    "  {}. {} · {}",
                    index + 1,
                    edit["action"].as_str().unwrap_or("未知操作"),
                    edit["path"]
                        .as_str()
                        .or_else(|| edit["parent"].as_str())
                        .or_else(|| edit["provider"].as_str())
                        .unwrap_or("自定义计划")
                );
            }
        }
        if report.data["redacted_fields"]
            .as_array()
            .is_some_and(|fields| !fields.is_empty())
        {
            println!("常见凭证字段已脱敏；JSON 报告包含字段位置。变体文件本身未修改。");
        }
        println!("{}", report.data["note"].as_str().unwrap_or(""));
        return;
    }
    if report.operation == "variant_copy" && report.success() {
        println!(
            "已创建编辑分支：{} → {}",
            report.data["source"].as_str().unwrap_or(""),
            report.data["destination"].as_str().unwrap_or("")
        );
        println!("{}", report.data["note"].as_str().unwrap_or(""));
        return;
    }
    if report.operation == "component_discover" && report.success() {
        println!(
            "扫描到 {} 个组件；新登记 {} 个，已登记 {} 个。",
            report.data["found"].as_u64().unwrap_or(0),
            report.data["registered"].as_array().map_or(0, Vec::len),
            report.data["already_registered"]
                .as_array()
                .map_or(0, Vec::len)
        );
        if let Some(components) = report.data["registered"].as_array() {
            for component in components {
                let manifest = &component["manifest"];
                println!(
                    "  已登记 {}@{} · {}",
                    manifest["component_id"].as_str().unwrap_or(""),
                    manifest["component_version"].as_str().unwrap_or(""),
                    manifest["kind"].as_str().unwrap_or("")
                );
            }
        }
        if let Some(components) = report.data["already_registered"].as_array() {
            for component in components {
                println!(
                    "  已存在 {}@{}",
                    component["id"].as_str().unwrap_or(""),
                    component["version"].as_str().unwrap_or("")
                );
            }
        }
        if let Some(issues) = report.data["issues"].as_array() {
            for issue in issues {
                let source = issue["distribution"]
                    .as_str()
                    .or_else(|| issue["id"].as_str())
                    .unwrap_or("组件");
                println!(
                    "  跳过 {source}：{}",
                    issue["message"].as_str().unwrap_or("无法验证清单")
                );
            }
        }
        println!("只读取包元数据和静态文件；不会导入扩展代码，也不会安装依赖。");
        return;
    }
    if report.operation == "component_uninstall" && report.success() {
        println!(
            "{} {}@{}",
            if report.data["applied"] == true {
                "已移除组件"
            } else {
                "组件移除预览"
            },
            report.data["component_id"].as_str().unwrap_or(""),
            report.data["component_version"].as_str().unwrap_or("")
        );
        if let Some(distribution) = report.data["distribution"].as_str() {
            println!("Python distribution：{distribution}");
        }
        if let Some(references) = report.data["project_config_references"].as_array()
            && !references.is_empty()
        {
            println!("项目默认配置引用（移除后需更新配置才能使用）：");
            for reference in references {
                println!(
                    "  {} · {}",
                    reference["path"].as_str().unwrap_or("项目配置"),
                    reference["component"].as_str().unwrap_or("")
                );
            }
        }
        if report.data["wheel_removed"] == true {
            println!("项目扩展 wheel：{}", report.data["extension_path"]);
        } else if report.data["extension_retained"] == true {
            println!(
                "共享 wheel 仍被其他已登记组件使用，保留在 {}",
                report.data["extension_path"]
            );
        }
        if let Some(references) = report.data["variant_references"].as_array()
            && !references.is_empty()
        {
            println!("已保存的运行或变体引用（移除后无法重载）：");
            for reference in references {
                println!(
                    "  {} · {}",
                    reference["path"].as_str().unwrap_or("变体"),
                    reference["component"].as_str().unwrap_or("")
                );
            }
        }
        if let Some(imports) = report.data["source_imports"].as_array()
            && !imports.is_empty()
        {
            println!("项目源码导入（移除后可能无法导入）：");
            for import in imports {
                println!(
                    "  {}:{} · {}",
                    import["path"].as_str().unwrap_or(""),
                    import["line"].as_u64().unwrap_or(0),
                    import["module"].as_str().unwrap_or("")
                );
            }
        }
        if let Some(unreadable) = report.data["unreadable_sources"].as_array()
            && !unreadable.is_empty()
        {
            println!("有源码无法安全扫描：");
            for source in unreadable {
                println!(
                    "  {} · {}",
                    source["path"].as_str().unwrap_or(""),
                    source["reason"].as_str().unwrap_or("扫描不完整")
                );
            }
        }
        if report.data["requires_force"] == true {
            println!(
                "发现配置/变体引用或源码扫描不完整；显式 CLI 操作需加 --force。配置、变体与源码文件不会被改写。"
            );
        } else if report.data["applied"] != true {
            println!("预览没有改动文件；使用 --apply 执行移除。");
        }
        if let Some(path) = report.data["cleanup_pending"].as_str() {
            println!("wheel 已从注册目录断开，但无法彻底删除暂存目录：{path}");
            if let Some(error) = report.data["cleanup_error"].as_str() {
                println!("清理错误：{error}");
            }
        }
        return;
    }
    if report.operation == "component_list" && report.success() {
        let components = report.data["components"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if components.is_empty() {
            println!(
                "本项目还没有注册组件。使用 m-model component discover 扫描已安装扩展，或 component add 登记本地入口。"
            );
        }
        for item in components {
            let manifest = &item["manifest"];
            println!(
                "{} {} · {} · {}",
                manifest["component_id"].as_str().unwrap_or(""),
                manifest["component_version"].as_str().unwrap_or(""),
                manifest["kind"].as_str().unwrap_or(""),
                manifest["entrypoint"].as_str().unwrap_or("")
            );
            if let Some(description) = manifest["description"]
                .as_str()
                .filter(|text| !text.is_empty())
            {
                println!("  {description}");
            }
        }
        return;
    }
    if matches!(
        report.operation.as_str(),
        "component_show" | "component_check" | "component_add"
    ) && report.success()
    {
        let descriptor = &report.data["component"];
        let manifest = &descriptor["manifest"];
        println!(
            "组件 {} {} · {}",
            manifest["component_id"].as_str().unwrap_or(""),
            manifest["component_version"].as_str().unwrap_or(""),
            manifest["kind"].as_str().unwrap_or("")
        );
        if report.operation == "component_check" {
            println!("清单与必需组件依赖解析通过；运行时入口未导入或执行。");
            let dependencies = report.data["dependencies"]["components"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            if dependencies.is_empty() {
                println!("必需组件依赖：无");
            } else {
                println!("解析后的组件依赖：");
                for dependency in dependencies {
                    println!(
                        "  {}@{} · {}",
                        dependency["id"].as_str().unwrap_or(""),
                        dependency["version"].as_str().unwrap_or(""),
                        dependency["kind"].as_str().unwrap_or("")
                    );
                }
            }
            if let Some(optional) = report.data["dependencies"]["optional_skipped"].as_array()
                && !optional.is_empty()
            {
                println!("未请求的可选依赖：");
                for dependency in optional {
                    println!(
                        "  {} ({})",
                        dependency["component_id"].as_str().unwrap_or(""),
                        dependency["constraint"].as_str().unwrap_or("")
                    );
                }
            }
        } else if report.operation == "component_add" {
            println!(
                "已注册到 {}",
                descriptor["manifest_path"].as_str().unwrap_or("")
            );
        }
        println!("入口：{}", manifest["entrypoint"].as_str().unwrap_or(""));
        if !manifest["description"].as_str().unwrap_or("").is_empty() {
            println!("说明：{}", manifest["description"]);
        }
        println!(
            "参数 Schema：{}",
            serde_json::to_string_pretty(&manifest["params_schema"]).unwrap_or_default()
        );
        println!(
            "能力：{}",
            serde_json::to_string_pretty(&manifest["capabilities"]).unwrap_or_default()
        );
        if report.operation == "component_show" {
            let resolution = &report.data["dependency_resolution"];
            if resolution["status"] == "resolved" {
                let dependencies = resolution["report"]["components"]
                    .as_array()
                    .map_or(0, Vec::len);
                println!("依赖图：已解析 · {dependencies} 个传递组件");
            } else {
                println!(
                    "依赖图：未解析 · {}",
                    resolution["reason"].as_str().unwrap_or("未知原因")
                );
            }
        }
        if report.operation == "component_show" && manifest["kind"] == "layer-edit-provider" {
            let verification = &report.data["edit_verification"];
            let status = verification["status"].as_str().unwrap_or("unknown");
            match status {
                "records_present" => println!(
                    "Layer 编辑验证：记录了 {} 个 payload；执行时仍会复核源码和环境。",
                    verification["payload_count"].as_u64().unwrap_or(0)
                ),
                "stale" => println!("Layer 编辑验证：已过期；需重新运行 component test。"),
                "unreadable" => println!(
                    "Layer 编辑验证记录无法读取：{}",
                    verification["reason"].as_str().unwrap_or("未知错误")
                ),
                _ => println!("Layer 编辑验证：尚未建立通过记录。"),
            }
        }
        if report.operation == "component_add" && manifest["kind"] == "layer" {
            println!(
                "在 Layer add/replace 中传入 --component {} 可按 Schema 校验参数。",
                manifest["component_id"]
            );
            println!(
                "使用 m-model component test {} 验证工厂构造；声明 forward_probe 时可加 --batch 做真实前向。",
                manifest["component_id"]
            );
        }
        if report.operation == "component_add" && manifest["kind"] == "layer-edit-provider" {
            println!(
                "先用 component test {} 验证具体 payload；通过 fixture 后，m-model layer custom 才能执行该 payload 并锁定组件版本。",
                manifest["component_id"]
            );
        }
        return;
    }
    if report.operation == "component_test" && report.success() {
        if let Some(edit) = report.data["checks"]["layer_edit"].as_object() {
            println!(
                "组件 {}@{} · Provider 编辑 {} · 覆盖范围 {}",
                report.data["component"]["id"].as_str().unwrap_or(""),
                report.data["component"]["version"].as_str().unwrap_or(""),
                edit.get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                report.data["coverage"].as_str().unwrap_or("unknown")
            );
            let activation_ready =
                report.data["activation"]["status"] == "enabled_for_tested_payloads";
            println!(
                "  本 payload：{}",
                if activation_ready {
                    "可执行；编辑时仍会复核版本、源码和运行环境"
                } else {
                    "未启用"
                }
            );
            if let Some(reason) = report.data["activation"]["reason"].as_str() {
                println!("  未启用原因：{reason}");
            }
            if let Some(reason) = edit.get("reason").and_then(Value::as_str) {
                println!("  {reason}");
            }
            if let Some(result) = edit.get("result") {
                println!(
                    "  编辑摘要：{}",
                    serde_json::to_string_pretty(result).unwrap_or_default()
                );
            }
            if let Some(modules) = edit.get("expected_modules").and_then(Value::as_object) {
                for (path, check) in modules {
                    println!(
                        "  结构预期 {path}：{}（实际 {:?}）",
                        check["status"].as_str().unwrap_or("unknown"),
                        check["actual"].as_str()
                    );
                }
            }
            for limit in report.data["limits"].as_array().into_iter().flatten() {
                if let Some(limit) = limit.as_str() {
                    println!("  范围：{limit}");
                }
            }
            return;
        }
        println!(
            "组件 {}@{} · 构造通过 · 覆盖范围 {}",
            report.data["component"]["id"].as_str().unwrap_or(""),
            report.data["component"]["version"].as_str().unwrap_or(""),
            report.data["coverage"].as_str().unwrap_or("unknown")
        );
        println!(
            "  工厂导入：{} · nn.Module 构造：{} · 状态有限性：{}",
            report.data["checks"]["factory_import"],
            report.data["checks"]["constructs_nn_module"],
            report.data["checks"]["constructed_state_finite"]
        );
        let forward = &report.data["checks"]["forward_probe"];
        println!(
            "  前向试跑：{}{}",
            forward["status"].as_str().unwrap_or("unknown"),
            forward["reason"]
                .as_str()
                .map(|reason| format!(" · {reason}"))
                .unwrap_or_default()
        );
        if let Some(capabilities) = report.data["capability_results"].as_object() {
            for (name, result) in capabilities {
                println!(
                    "  能力 {name} · 声明={} · 检查={}",
                    result["declared_state"].as_str().unwrap_or("unknown"),
                    result["status"].as_str().unwrap_or("unknown")
                );
            }
        }
        for limit in report.data["limits"].as_array().into_iter().flatten() {
            if let Some(limit) = limit.as_str() {
                println!("  范围：{limit}");
            }
        }
        return;
    }
    if report.operation == "runs_list" {
        if let Some(runs) = report.data["runs"].as_array() {
            if runs.is_empty() {
                println!(
                    "{}",
                    if report.data["filtered"] == true {
                        "没有匹配的运行记录。"
                    } else {
                        "本项目还没有运行记录。"
                    }
                );
            }
            for run in runs {
                let summary = &run["summary"];
                let status = run["status"].as_str().unwrap_or("unknown");
                let id = run["id"].as_str().unwrap_or("");
                let label = run["name"]
                    .as_str()
                    .filter(|name| !name.is_empty())
                    .map(|name| format!("{name} ({id})"))
                    .unwrap_or_else(|| id.to_owned());
                let kept = run["kept"] == true;
                let mode = run["training_mode"]
                    .as_str()
                    .or_else(|| summary["training_mode"].as_str())
                    .unwrap_or("");
                let kept_note = if kept { " · 保留" } else { "" };
                if matches!(status, "running" | "stopping") {
                    let progress = &run["live_progress"];
                    if progress["step"].is_number() {
                        println!(
                            "{}  {}  {} · {}/{} 步 · Loss {}{}",
                            label,
                            status,
                            mode,
                            progress["step"],
                            progress["total"],
                            progress["mean_loss"],
                            if progress["checkpoint"] == true {
                                " · 已保存 checkpoint"
                            } else {
                                kept_note
                            }
                        );
                    } else {
                        println!("{label}  {status}  {mode} · 等待首个步骤{kept_note}");
                    }
                } else {
                    let checkpoint = if run["checkpoint"].is_object() {
                        " · 可恢复"
                    } else {
                        ""
                    };
                    println!(
                        "{}  {}  {} · {}/{} 步 · Loss {}{}{}",
                        label,
                        status,
                        mode,
                        summary["completed_steps"],
                        summary["requested_steps"],
                        summary["mean_loss"],
                        checkpoint,
                        kept_note
                    );
                }
                if let Some(note) = run["status_note"].as_str() {
                    println!("  说明：{note}");
                }
            }
        }
        return;
    }
    if report.operation == "runs_storage" && report.success() {
        println!(
            "运行记录占用：{}",
            format_bytes(report.data["total_bytes"].as_u64().unwrap_or(0))
        );
        for run in report.data["runs"].as_array().into_iter().flatten() {
            let categories = &run["categories"];
            println!(
                "{}  {}  {} · 元数据 {} · 日志 {} · 进度 {} · checkpoints {}",
                run["id"].as_str().unwrap_or(""),
                run["status"].as_str().unwrap_or("unknown"),
                format_bytes(run["total_bytes"].as_u64().unwrap_or(0)),
                format_bytes(categories["metadata"].as_u64().unwrap_or(0)),
                format_bytes(categories["logs"].as_u64().unwrap_or(0)),
                format_bytes(categories["progress"].as_u64().unwrap_or(0)),
                format_bytes(categories["checkpoints"].as_u64().unwrap_or(0))
            );
            if let Some(paths) = run["skipped_symlinks"]
                .as_array()
                .filter(|paths| !paths.is_empty())
            {
                println!(
                    "  跳过符号链接：{}",
                    serde_json::to_string(paths).unwrap_or_default()
                );
            }
        }
        println!("仅统计 .model-maven/runs 内文件；源码和数据集不复制到运行记录。");
        return;
    }
    if report.operation == "runs_attach" && report.success() {
        let record = &report.data["record"];
        let status = record["status"].as_str().unwrap_or("unknown");
        println!(
            "运行 {} · {}",
            report.data["run_id"].as_str().unwrap_or(""),
            status
        );
        if let Some(progress) = report.data["progress"]
            .as_array()
            .and_then(|items| items.iter().rev().find(|item| item["step"].is_number()))
        {
            println!(
                "进度：{}/{} 步 · 平均 Loss {}",
                progress["step"], progress["total"], progress["mean_loss"]
            );
        }
        if report.data["progress"].as_array().is_some_and(|items|
            items.iter().any(|item| item["event"] == "progress_line_omitted")) {
            println!("过大的进度行已省略；请检查运行日志。 ");
        }
        if let Some(log) = report.data["log_tail"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
        {
            println!("最近日志：\n{}", log.trim_end());
        }
        if matches!(status, "running" | "stopping") {
            println!(
                "停止：m-model runs stop {}",
                report.data["run_id"].as_str().unwrap_or("<id>")
            );
        }
        return;
    }
    if report.operation == "runs_stop" && report.success() {
        println!("运行 {} · {}", report.data["run_id"], report.data["status"]);
        println!("{}", report.data["message"].as_str().unwrap_or(""));
        return;
    }
    if report.operation == "runs_clean_plan" && report.success() {
        let plan_id = report.data["id"].as_str().unwrap_or("");
        let items = report.data["items"].as_array().cloned().unwrap_or_default();
        println!(
            "清理预览 {plan_id} · {} 项 · 最多释放 {}",
            items.len(),
            format_bytes(report.data["total_bytes"].as_u64().unwrap_or(0))
        );
        for item in &items {
            println!(
                "{}  {}  {} · {}",
                item["item_id"].as_str().unwrap_or(""),
                item["run_id"].as_str().unwrap_or(""),
                format_bytes(item["size_bytes"].as_u64().unwrap_or(0)),
                item["path"].as_str().unwrap_or("")
            );
            println!("  {}", item["reason"].as_str().unwrap_or(""));
        }
        if let Some(skipped) = report.data["skipped"].as_array() {
            for item in skipped {
                println!("跳过 {}：{}", item["run_id"], item["reason"]);
            }
        }
        if items.iter().any(|item| item["release_checkpoint"] == true) {
            println!("该计划会移除正式恢复权重并失去续训能力；执行时还需显式确认：");
            println!("m-model runs clean --apply {plan_id} --release-checkpoints");
        } else if items.is_empty() {
            println!("没有符合所选范围的运行文件可清理。");
        } else {
            println!("执行固定清单：m-model runs clean --apply {plan_id}");
        }
        return;
    }
    if report.operation == "runs_clean_apply" && report.success() {
        let items = report.data["applied_items"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let released = items
            .iter()
            .any(|item| item["kind"] == "latest_checkpoint_release");
        let bytes = items
            .iter()
            .map(|item| item["size_bytes"].as_u64().unwrap_or(0))
            .fold(0_u64, u64::saturating_add);
        println!(
            "清理计划 {} · 删除 {} 项 · 释放 {}",
            report.data["id"],
            items.len(),
            format_bytes(bytes)
        );
        if released {
            println!("已释放指定运行的 checkpoint；运行摘要仍保留，但该运行不能续训或导出权重。");
        }
        if let Some(skipped) = report.data["apply_skipped"].as_array() {
            for item in skipped {
                println!("未删除 {}：{}", item["item_id"], item["reason"]);
            }
        }
        if let Some(warnings) = report.data["apply_warnings"].as_array() {
            for item in warnings {
                println!("提醒 {}：{}", item["item_id"], item["reason"]);
            }
        }
        return;
    }
    if matches!(
        report.operation.as_str(),
        "runs_rename" | "runs_keep" | "runs_unkeep"
    ) && report.success()
    {
        if report.operation == "runs_rename" {
            println!(
                "运行 {} 已命名为：{}",
                report.data["run_id"], report.data["name"]
            );
        } else if report.data["kept"] == true {
            println!("运行 {} 已标记保留。", report.data["run_id"]);
        } else {
            println!("运行 {} 已取消保留标记。", report.data["run_id"]);
        }
        return;
    }
    if report.operation == "runs_show" && report.success() {
        let record = &report.data["record"];
        let inputs = &report.data["inputs"];
        let summary = &record["summary"];
        let status = match record["status"].as_str().unwrap_or("unknown") {
            "completed" => "已完成",
            "failed" => "失败",
            "interrupted" => "已中断",
            "unknown" => "状态未知",
            other => other,
        };
        println!("运行记录 · {status}");
        println!("ID：{}", record["id"].as_str().unwrap_or("未知"));
        if let Some(name) = record["name"].as_str().filter(|name| !name.is_empty()) {
            println!("名称：{name}");
        }
        if record["kept"] == true {
            println!("保留：是");
        }
        if record["checkpoint_released"] == true {
            println!("恢复权重：已按清理计划释放；不能从此运行续训或导出权重");
        }
        if let Some(parent) = inputs["repeat_of"].as_str() {
            println!("重跑自：{parent}");
        } else if let Some(parent) = record["parent_run_id"].as_str() {
            println!("续自运行：{parent}");
        }
        if let Some(note) = record["status_note"].as_str() {
            println!("说明：{note}");
        }
        if let Some(overrides) = inputs["repeat_overrides"].as_array()
            && !overrides.is_empty()
        {
            let values = overrides
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            println!("与原运行不同：{}", values.join("、"));
        }
        if let Some(checkpoint) = report.data["checkpoint"].as_object() {
            println!(
                "恢复点：第 {} 步 · SHA-256 {}",
                checkpoint.get("completed_steps").unwrap_or(&Value::Null),
                checkpoint
                    .get("sha256")
                    .and_then(Value::as_str)
                    .unwrap_or("未知")
            );
        }
        if !summary.is_null() {
            if summary["training_mode"] == "generic-external-loop" {
                let source = match summary["completed_steps_source"].as_str() {
                    Some("worker_report") => "用户循环最后接受的报告",
                    Some("checkpoint_manifest_unverified") => "恢复点清单所载步数，未核验；后续进度未知",
                    _ => "来源未知",
                };
                println!("用户循环：{} / {} 步（{}）", summary["completed_steps"], summary["requested_steps"], source);
            } else if summary["training_mode"] == "external-loop" {
                println!(
                    "完整自有循环：{}/{} 步 · 平均 Loss {}",
                    summary["completed_steps"], summary["requested_steps"], summary["mean_loss"]
                );
            } else {
                println!(
                    "训练：{} · {}/{} 步 · 平均 Loss {} · 优化器更新 {} 次",
                    summary["training_mode"].as_str().unwrap_or("未知模式"),
                    summary["completed_steps"],
                    summary["requested_steps"],
                    summary["mean_loss"],
                    summary["optimizer_updates"]
                );
            }
        }
        for (label, key) in [
            ("模型", "model"),
            ("批次", "batch"),
            ("训练步骤", "training_step"),
            ("完整训练循环", "training_loop"),
        ] {
            if let Some(value) = inputs[key].as_str() {
                println!("{label}：{value}");
            }
        }
        if let Some(sources) = inputs["input_sources"].as_object()
            && !sources.is_empty()
        {
            println!("输入来源：");
            for (key, source) in sources {
                let label = match key.as_str() {
                    "model" => "模型入口",
                    "batch" => "批次入口",
                    "batch_binding" => "批次绑定",
                    "training_step" => "训练步骤",
                    "training_loop" => "训练循环",
                    "model_injections" => "模型装配",
                    other => other,
                };
                println!("  {label} · {}", format_input_source(source, 0));
            }
            if let Some(revision) = inputs["config_revision"].as_str() {
                println!("  项目配置版本 · {}", short_revision(revision));
            }
        }
        if let Some(runtime) = inputs["runtime"].as_object() {
            if let Some(python) = runtime.get("python_executable").and_then(Value::as_str) {
                println!("Python：{python}");
            }
        }
        if let Some(events) = report.data["events"].as_array() {
            let event_names = events
                .iter()
                .filter_map(|event| event["event"].as_str())
                .collect::<Vec<_>>();
            if !event_names.is_empty() {
                println!("事件：{}", event_names.join(" → "));
            }
        }
        if let Some(diagnostics) = report.data["report"]["diagnostics"].as_array() {
            for diagnostic in diagnostics {
                println!(
                    "{}：{}",
                    diagnostic["code"].as_str().unwrap_or("运行诊断"),
                    diagnostic["message"].as_str().unwrap_or("未知错误")
                );
                if let Some(hint) = diagnostic["hint"].as_str().filter(|hint| !hint.is_empty()) {
                    println!("下一步：{hint}");
                }
            }
        }
        if inputs["redacted_fields"]
            .as_array()
            .is_some_and(|fields| !fields.is_empty())
        {
            println!("凭证字段已脱敏；完整字段列表可通过 JSON 查看。");
        }
        println!(
            "完整运行证据：m-model --json runs show {}",
            record["id"].as_str().unwrap_or("<id>")
        );
        return;
    }
    if report.operation == "runs_compare" && report.success() {
        let left = report.data["left_run_id"].as_str().unwrap_or("未知");
        let right = report.data["right_run_id"].as_str().unwrap_or("未知");
        println!("运行对照（只读）· {left} ↔ {right}");
        println!(
            "协议：{}",
            comparison_state(report.data["protocol"]["status"].as_str())
        );
        if let Some(dimensions) = report.data["dimensions"].as_array() {
            for dimension in dimensions {
                let label = match dimension["name"].as_str().unwrap_or("") {
                    "inputs" => "输入与设置",
                    "source_code" => "源码",
                    "data" => "数据与采样",
                    "optimizer_factory" => "自定义优化器声明",
                    "supervision" => "监督配置声明",
                    "samples" => "样本身份（训练代码报告）",
                    "budget" => "训练预算",
                    "requested_device" => "请求的设备",
                    "runtime_device" => "观察到的设备（名称不证明物理身份）",
                    "weights" => "权重",
                    "metrics" => "指标",
                    "training_mean_metrics" => "训练平均指标",
                    other => other,
                };
                println!(
                    "{label}：{}",
                    comparison_state(dimension["status"].as_str())
                );
            }
        }
        println!("对照展示记录差异，不推断指标变化原因；使用 --json 查看字段详情。");
        return;
    }
    if report.operation == "export" && report.success() {
        println!(
            "工程包已导出：{} · run {} · checkpoint 第 {} 步",
            report.data["path"].as_str().unwrap_or(""),
            report.data["run_id"].as_str().unwrap_or(""),
            report.data["checkpoint_completed_steps"]
        );
        println!(
            "已校验 {} 个源码文件和 {} 字节；可用 `python restore_model.py` 恢复模型。",
            report.data["source_file_count"], report.data["verified_bytes"]
        );
        let extension_package_count = report.data["extension_package_count"].as_u64().unwrap_or(0);
        let dependency_package_count = report.data["dependency_package_count"]
            .as_u64()
            .unwrap_or(0);
        if extension_package_count > 0 {
            println!(
                "已随包携带 {} 个扩展和 {} 个 Python 依赖，共 {} 个文件（{} 字节）。",
                extension_package_count,
                dependency_package_count,
                report.data["extension_file_count"],
                report.data["extension_bytes"]
            );
        } else {
            println!("当前模型未使用需要随包携带的已登记扩展。");
        }
        match (
            report.data["environment_snapshot_status"].as_str(),
            report.data["environment_snapshot_timing"].as_str(),
        ) {
            (Some("captured"), Some("training_start")) => println!(
                "已附带 {} 个训练开始时安装包的版本快照；requirements.snapshot.txt 不是可直接复现的依赖锁。",
                report.data["environment_distribution_count"]
            ),
            (Some("captured"), _) => println!(
                "已附带 {} 个导出时安装包的版本快照；requirements.snapshot.txt 不是可直接复现的依赖锁。",
                report.data["environment_distribution_count"]
            ),
            (Some("runtime_mismatch"), Some("training_start")) => println!(
                "已附带训练开始时的环境清单，但 Python 或 PyTorch 版本与训练记录不一致；请检查 environment.json。"
            ),
            (Some("runtime_mismatch"), _) => println!(
                "已附带导出时的环境清单，但 Python 或 PyTorch 版本与训练记录不一致；其他包也可能已变化，请检查 environment.json。"
            ),
            _ => println!("没有捕获到完整 Python 包清单；请查看导出包中的 environment.json。"),
        }
        if let Some(path) = report.data["backup_path"].as_str() {
            println!("旧目标备份：{path}");
        }
        return;
    }
    if let Some(name) = report.data["variant_name"].as_str() {
        println!("模型变体：{name}");
    }
    if let Some(injections) = report
        .data
        .pointer("/origin/injection_evidence")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
    {
        println!("模型接缝：");
        for injection in injections {
            let paths = injection["registered_paths"]
                .as_array()
                .map(|paths| {
                    paths
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            println!(
                "  {} ← {} · {} · {}{}",
                injection["slot"].as_str().unwrap_or(""),
                injection["factory"].as_str().unwrap_or(""),
                injection["module_type"].as_str().unwrap_or("未知模块"),
                paths,
                injection["sample_forward"]
                    .as_str()
                    .map(|status| format!(" · 样本前向 {status}"))
                    .unwrap_or_default()
            );
        }
    }
    if let Some(rows) = report.data["modules"].as_array() {
        for row in rows {
            println!(
                "{}  {}  参数={}{}",
                row["path"].as_str().unwrap_or(""),
                row["type"].as_str().unwrap_or(""),
                row["parameters"],
                if row["aliases"].as_array().is_some_and(|a| a.len() > 1) {
                    "（共享）"
                } else {
                    ""
                }
            );
            if let Some(site) = row["construction_site"].as_object() {
                println!(
                    "  创建位置：{}:{}",
                    site.get("file").and_then(Value::as_str).unwrap_or("未知"),
                    site.get("line").unwrap_or(&Value::Null)
                );
            } else if let Some(definition) = row["source_definition"].as_object() {
                println!(
                    "  类型定义：{}:{}",
                    definition
                        .get("file")
                        .and_then(Value::as_str)
                        .unwrap_or("未知"),
                    definition.get("line").unwrap_or(&Value::Null)
                );
            }
            if report.operation == "layers" && !row["editable_params"].is_null() {
                println!("  变体可改参数：{}", row["editable_params"]);
            } else if report.operation == "layers" && row["can_source_set"] == true {
                println!(
                    "  源码可改参数（--target source）：{}",
                    row["source_editable_fields"]
                );
            }
            if report.operation == "layers"
                && let Some(reason) = row["source_set_block_reason"].as_str()
            {
                println!("  源码改参受限：{reason}");
            }
        }
    }
    if let Some(calls) = report.data["calls"].as_array() {
        println!("\n本批次前向调用：");
        for call in calls {
            println!(
                "{} · 第 {} 次\n  输入 {}\n  输出 {}",
                call["paths"][0].as_str().unwrap_or(""),
                call["invocation_index"],
                describe_tree(&call["inputs"]),
                describe_tree(&call["outputs"])
            );
        }
    }
    if report.operation == "verify" && report.success() {
        println!(
            "损失：{} · 模式：{} · 产生梯度的参数：{}",
            report.data["loss"], report.data["observed_mode"], report.data["gradient_count"]
        );
        if let Some(gradients) = report.data["gradients"].as_array() {
            for gradient in gradients {
                println!(
                    "  {} · 有限={} · 范数={}",
                    gradient["name"], gradient["finite"], gradient["norm"]
                );
            }
        }
        if let Some(disconnected) = report.data["disconnected_parameters"].as_array()
            && !disconnected.is_empty()
        {
            println!(
                "未参与本次损失的参数：{}",
                serde_json::to_string(disconnected).unwrap()
            );
        }
    }
    if report.operation == "training_probe" && report.success() {
        if report.data["training_mode"] == "external-loop" {
            println!(
                "完整自有循环试跑：{}/{} 步 · 平均 Loss {}",
                report.data["completed_steps"],
                report.data["requested_steps"],
                report.data["mean_loss"]
            );
            println!("forward/backward/optimizer 由用户循环控制，CLI 未独立验证这些更新。");
        } else {
            println!(
                "训练试跑（{}）：{}/{} 步 · 优化器更新 {} 次 · 平均 Loss {} · 产生梯度的参数张量 {}/{}（共 {} 个可训练参数值）",
                report.data["training_mode"]
                    .as_str()
                    .unwrap_or("managed-step"),
                report.data["completed_steps"],
                report.data["requested_steps"],
                report.data["optimizer_updates"],
                report.data["mean_loss"],
                report.data["parameters_with_gradient"],
                report.data["trainable_parameters"],
                report.data["trainable_parameter_count"]
            );
        }
        if let Some(updates) = report.data["optimizer_updates_by_group"].as_object() {
            for (name, count) in updates {
                println!("  {name}: {count} 次更新");
            }
        }
        if let Some(metrics) = report.data["mean_metrics"].as_object()
            && !metrics.is_empty()
        {
            println!("平均指标：{}", serde_json::to_string(metrics).unwrap());
        }
        if let Some(losses) = report.data["mean_losses"].as_object()
            && !losses.is_empty()
        {
            println!("具名损失：{}", serde_json::to_string(losses).unwrap());
        }
        println!("本次模型权重和优化器状态未保存；这不是可恢复训练任务。");
        if let Some(run_id) = report.data["run_id"].as_str() {
            println!("运行记录：{run_id} · 查看详情：m-model runs show {run_id}");
        }
        if let Some(parent) = report.data["repeat_of"].as_str() {
            println!("本次重跑自：{parent}");
        }
        if let Some(overrides) = report.data["repeat_overrides"].as_array() {
            let values = overrides
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            if !values.is_empty() {
                println!("覆盖设置：{}", values.join("、"));
            }
        }
    }
    if report.operation == "train" && report.success() {
        if report.data["detached"] == true {
            println!(
                "后台训练已启动：{}",
                report.data["run_id"].as_str().unwrap_or("")
            );
            println!(
                "查看进度：{}",
                report.data["attach"]
                    .as_str()
                    .unwrap_or("m-model runs attach")
            );
            println!(
                "安全停止：{}",
                report.data["stop"].as_str().unwrap_or("m-model runs stop")
            );
            return;
        }
        if report.data["training_mode"] == "external-loop" {
            println!(
                "完整自有循环完成：{}/{} 步 · 平均 Loss {}",
                report.data["completed_steps"],
                report.data["requested_steps"],
                report.data["mean_loss"]
            );
            println!("循环事件与有限性已检查；forward/backward/optimizer 由用户代码控制。");
        } else {
            println!(
                "正式训练完成：{}/{} 步 · 平均 Loss {} · 优化器更新 {} 次",
                report.data["completed_steps"],
                report.data["requested_steps"],
                report.data["mean_loss"],
                report.data["optimizer_updates"]
            );
        }
        if let Some(run_id) = report.data["run_id"].as_str() {
            println!("运行记录：{run_id} · 详情：m-model runs show {run_id}");
        }
        if let Some(checkpoint) = report.data["checkpoint"].as_object() {
            println!(
                "已保存恢复点：第 {} 步 · SHA-256 {}",
                checkpoint.get("completed_steps").unwrap_or(&Value::Null),
                checkpoint
                    .get("sha256")
                    .and_then(Value::as_str)
                    .unwrap_or("未知")
            );
        }
        if report.data["environment_snapshot"]["schema_version"] == 1 {
            let fingerprint = report.data["runtime"]["environment_fingerprint_sha256"]
                .as_str()
                .and_then(|value| value.get(..12));
            if let Some(fingerprint) = fingerprint {
                println!(
                    "已记录训练开始时的 Python 环境：{} 个 distribution；续训会核对指纹 sha256:{}；可通过 `m-model --json runs show {}` 查看。",
                    report.data["environment_snapshot"]["distribution_count"],
                    fingerprint,
                    report.data["run_id"].as_str().unwrap_or("")
                );
            } else {
                println!(
                    "已记录训练开始时的 Python 环境：{} 个 distribution；环境指纹不可用，可通过 `m-model --json runs show {}` 查看原因。",
                    report.data["environment_snapshot"]["distribution_count"],
                    report.data["run_id"].as_str().unwrap_or("")
                );
            }
        } else {
            println!(
                "训练已完成，但未能记录 Python 包清单；原因可通过 `m-model --json runs show {}` 查看。",
                report.data["run_id"].as_str().unwrap_or("")
            );
        }
        if let Some(parent) = report.data["parent_run_id"].as_str() {
            println!("续自运行：{parent}");
        }
        if let Some(state) = report.data["resume_state"].as_object()
            && state
                .values()
                .any(|value| value == "reset-on-resume" || value == "reset")
        {
            println!(
                "恢复状态有重置项：{}",
                serde_json::to_string(state).unwrap()
            );
        }
    }
    if report.operation == "evaluate" && report.success() {
        println!(
            "{}",
            crate::presentation::pages::evaluation::render(report).join("\n")
        );
    }
    if let Some(coverage) = report.data["coverage"].as_str() {
        println!("\n{coverage}");
    }
    if report.operation == "edit" && report.success() {
        if report.data["target"] == "source" || report.data["source_path"].is_string() {
            if let Some(path) = report.data["source_path"].as_str() {
                println!(
                    "\n{}源码：{}",
                    if report.data["committed"] == true {
                        "已提交并重建验证 "
                    } else {
                        "源码预览 "
                    },
                    path
                );
            } else if let Some(paths) = report.data["source_paths"]
                .as_array()
                .or_else(|| report.data["source_edit"]["paths"].as_array())
            {
                println!(
                    "\n{}源码：{} 个文件",
                    if report.data["committed"] == true {
                        "已提交并重建验证 "
                    } else {
                        "源码预览 "
                    },
                    paths.len()
                );
                for path in paths.iter().filter_map(Value::as_str) {
                    println!("  {path}");
                }
            }
            if let Some(edit) = report.data["source_edit"].as_object()
                && edit.get("id").is_some()
            {
                let files = edit.get("files").and_then(Value::as_array);
                if let Some(files) = files.filter(|files| files.len() > 1) {
                    println!(
                        "源码事务 {} · {} 个文件",
                        edit.get("id").and_then(Value::as_str).unwrap_or(""),
                        files.len()
                    );
                    for file in files {
                        println!(
                            "  {} · SHA-256 {} → {} · 备份 {}",
                            file["path"].as_str().unwrap_or(""),
                            file["before_sha256"].as_str().unwrap_or(""),
                            file["after_sha256"].as_str().unwrap_or(""),
                            file["backup"].as_str().unwrap_or("")
                        );
                    }
                } else {
                    println!(
                        "源码事务 {} · {} · SHA-256 {} → {} · 备份 {}",
                        edit.get("id").and_then(Value::as_str).unwrap_or(""),
                        edit.get("path").and_then(Value::as_str).unwrap_or(""),
                        edit.get("before_sha256")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                        edit.get("after_sha256")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                        edit.get("backup").and_then(Value::as_str).unwrap_or("")
                    );
                }
            }
            if let Some(edit) = report.data["source_edit"].as_object()
                && edit.get("action").is_some()
            {
                println!(
                    "源码历史 {} · {} · 游标 {}",
                    edit.get("action").and_then(Value::as_str).unwrap_or(""),
                    edit.get("path").and_then(Value::as_str).unwrap_or(""),
                    edit.get("cursor").unwrap_or(&Value::Null)
                );
            }
            if let Some(diff) = report.data["source_patch"]["unified_diff"].as_str()
                && !diff.is_empty()
            {
                println!("\n{diff}");
            }
            if let Some(effects) = report.data["edit_effects"].as_array() {
                for effect in effects {
                    display_edit_effect(effect);
                }
            }
            println!("验证：{}", report.data["validation"]);
            if report.data["variant"].is_object() {
                println!("变体设备策略：{}；候选验证后的注册状态：{}",
                    report.data["variant"].get("device_policy").map_or("legacy_cpu", |value| value.as_str().unwrap_or("未知")),
                    report.data["model_state_placement"]);
            }
            if report.data["source_validation"].is_object() {
                println!(
                    "提交后重载复核：{} · {} · {} 个模块",
                    report.data["source_validation"]["operation"]
                        .as_str()
                        .unwrap_or("验证"),
                    report.data["source_validation"]["status"]
                        .as_str()
                        .unwrap_or("unknown"),
                    report.data["source_validation"]["modules"]
                        .as_u64()
                        .unwrap_or(0)
                );
            }
            println!("{}", report.data["note"].as_str().unwrap_or(""));
        } else {
            println!(
                "\n{}：{}",
                if report.data["committed"] == true {
                    "已保存"
                } else {
                    "仅预览，未保存"
                },
                report.data["variant_path"].as_str().unwrap_or("")
            );
            println!("验证：{}", report.data["validation"]);
            if let Some(effects) = report.data["edit_effects"].as_array() {
                for effect in effects {
                    display_edit_effect(effect);
                }
            }
            println!("{}", report.data["note"].as_str().unwrap_or(""));
        }
    }
}

fn display_input_provenance(report: &Report) {
    let Some(sources) = report.data["input_sources"].as_object() else {
        return;
    };
    if sources.is_empty() {
        return;
    }
    println!("本次输入来源：");
    for (key, source) in sources {
        let label = match key.as_str() {
            "model" => "模型入口",
            "batch" => "批次入口",
            "batch_binding" => "批次绑定",
            "training_step" => "训练步骤",
            "training_loop" => "训练循环",
            "variant" => "模型变体",
            _ => key,
        };
        let kind = input_source_label(source["kind"].as_str().unwrap_or("unknown"));
        let reference = source["reference"]
            .as_str()
            .map(|value| format!(" · {value}"))
            .unwrap_or_default();
        let revision = source["revision"]
            .as_str()
            .map(|value| format!(" · revision {}", short_revision(value)))
            .unwrap_or_default();
        println!("  {label} · {kind}{reference}{revision}");
    }
    if let Some(revision) = report.data["config_revision"].as_str() {
        println!("  项目配置版本 · {}", short_revision(revision));
    }
}

fn format_input_source(source: &Value, depth: usize) -> String {
    let kind = input_source_label(source["kind"].as_str().unwrap_or("unknown"));
    let reference = source["reference"]
        .as_str()
        .map(|value| format!(" · {value}"))
        .unwrap_or_default();
    let revision = source["revision"]
        .as_str()
        .map(|value| format!(" · revision {}", short_revision(value)))
        .unwrap_or_default();
    let mut summary = format!("{kind}{reference}{revision}");
    if depth < 4
        && let Some(upstream) = source.get("upstream").filter(|value| value.is_object())
    {
        summary.push_str(" ← ");
        summary.push_str(&format_input_source(upstream, depth + 1));
    } else if depth == 4 && source.get("upstream").is_some() {
        summary.push_str(" ← 上游来源链已省略");
    }
    summary
}

fn config_value_label(value: &Value) -> String {
    match value {
        Value::Null => "未设置".into(),
        Value::String(value) => value.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "无法显示".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{format_doctor_summary, format_input_source, format_readiness_summary};
    use serde_json::json;

    #[test]
    fn doctor_summary_renders_registered_runtime_capabilities_for_people() {
        let summary = format_doctor_summary(&json!({
            "python":"/usr/bin/python3",
            "python_version":"3.12.1",
            "torch_available":true,
            "torch_version":"2.6.0",
            "compatibility":{"worker_protocol_version":1,"worker_api_version":1},
            "capabilities":[
                {"id":"model.inspect","state":"supported"},
                {"id":"training.detached","state":"unsupported","reason":"Linux only"},
                {"id":"future.capability","state":"unknown","reason":"No provider evidence"}
            ],
            "hint":"m-model env setup"
        }));
        assert!(summary.contains("当前可用能力：1/3"));
        assert!(summary.contains("模型结构检查"));
        assert!(summary.contains("后台训练监督"));
        assert!(summary.contains("原因：Linux only"));
        assert!(summary.contains("future.capability"));
        assert!(summary.contains("下一步：m-model env setup"));
    }

    #[test]
    fn readiness_summary_shows_missing_inputs_provenance_and_actions_without_values() {
        let summary = format_readiness_summary(&json!({
            "operation":"forward_probe",
            "base_revision":"12345678901234567890",
            "inputs":[
                {
                    "key":"model",
                    "state":"available",
                    "required":true,
                    "value":"private-model-value",
                    "source":{"kind":"project_config","reference":".model-maven/project.json#/model","revision":"abcdef1234567890"}
                },
                {
                    "key":"batch",
                    "state":"missing",
                    "required":true,
                    "value":null,
                    "source":{"kind":"unknown"}
                }
            ],
            "readiness":{
                "state":"needs_input",
                "facts":[],
                "capabilities":[],
                "conditions":[],
                "actions":[
                    {"id":"configure_batch","label":"选择示例批次以查看前向","availability":"available"}
                ]
            }
        }));

        assert!(summary.contains("需要输入 · 前向试跑"));
        assert!(summary.contains("配置版本：123456789012"));
        assert!(summary.contains("model · 必需 · 已就绪 · 来源：项目配置"));
        assert!(summary.contains("revision abcdef123456"));
        assert!(summary.contains("batch · 必需 · 缺少 · 来源：来源未确认"));
        assert!(summary.contains("下一步 · 选择示例批次以查看前向 · 可用"));
        assert!(!summary.contains("private-model-value"));
    }

    #[test]
    fn run_source_summary_preserves_the_saved_run_and_original_input_lineage() {
        let summary = format_input_source(
            &json!({
                "kind":"saved_run",
                "reference":"run-123",
                "revision":"abcdef1234567890",
                "upstream":{
                    "kind":"command_line",
                    "reference":"--model"
                }
            }),
            0,
        );
        assert!(summary.contains("已保存运行 · run-123"));
        assert!(summary.contains("revision abcdef123456"));
        assert!(summary.contains("← 本次命令 · --model"));
    }
}
