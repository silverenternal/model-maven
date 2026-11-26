use crate::{
    app::{
        Application,
        classification::{
            classification_classes, create_classification_preset_files,
            validate_classification_preset_targets,
        },
    },
    cli::*,
    domain::Report,
    ports::{Backend, PythonRuntimeControl},
    presentation::{
        background_preview::BackgroundPreview,
        contracts::{NavigationResult, PageId},
        forms::fields::{FormMode, prompt_schema_object},
        pages::{
            assembly::render as render_assembly,
            component_detail::render as render_component_detail,
            components::{
                available_actions as component_actions,
                render_catalog_page as render_component_catalog_page,
            },
            configuration::render as render_configuration,
            edit_plan::render as render_edit_plan,
            environment::render as render_environment,
            export::render as render_export,
            home::register_pages,
            layer_detail::render_layer_detail,
            layer_edit::{
                can_patch as layer_can_patch, render_context as render_layer_edit_context,
            },
            model_tree::ModelTreePage,
            navigation::{Selection, resolve_selection},
            onboarding::render as render_onboarding,
            probe::{
                ProbeCallPage, ProbeFailureAction, render_probe_context,
                render_probe_failure_context, resolve_failure_action,
            },
            project_open::{render as render_project_open, resolve_path as resolve_project_path},
            resume::{
                render as render_resume_candidates, render_selected as render_resume_selection,
                resume_available, unavailable_reason,
            },
            run_detail::render as render_run_detail,
            run_progress::render as render_run_progress,
            runs::{RunsFilter, render as render_runs},
            session_restore::render as render_session_restore,
            storage::{render_clean_plan, render_clean_result, render_storage},
            workbench_resume::{
                page_name as resumable_page_name, render as render_workbench_resume,
            },
        },
        service::{ApplicationUiServices, UiServices},
        session::SessionState,
        session_store::{self, OnboardingDraft},
        terminal::{display, is_interactive, prompt, prompt_allow_empty},
        widgets::{
            action_list::render_actions,
            call_list::render_call_list,
            edit_diff::render as render_edit_diff,
            layout,
            model_tree::{render_model_tree, toggle_path},
            run_plan::render as render_run_plan,
            tensor_tree::TensorTreePage,
            text,
        },
    },
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

struct ProbeCache {
    model: Option<String>,
    batch: Option<String>,
    variant: Option<String>,
    seed: Option<u64>,
    report: Value,
}

impl ProbeCache {
    fn matches(&self, model: &Option<String>, batch: &Option<String>, cli: &Cli) -> bool {
        self.model == *model
            && self.batch == *batch
            && self.variant == cli.variant
            && self.seed == cli.seed
    }
}

fn input_snapshot(model: &Option<String>, batch: &Option<String>) -> Inputs {
    Inputs {
        model: model.clone(),
        batch: batch.clone(),
        ..Inputs::default()
    }
}

fn ensure_model(model: &mut Option<String>, model_source: &mut Option<String>, cli: &Cli) -> bool {
    if model.is_some() || cli.variant.is_some() {
        if cli.variant.is_some() {
            *model_source = Some("命令行模型变体".into());
        }
        return true;
    }
    *model = prompt("模型入口（如 examples.models:build_shared）");
    if model.is_some() {
        *model_source = Some("本次工作台会话".into());
    }
    model.is_some()
}

fn ensure_batch(batch: &mut Option<String>, required: bool) -> bool {
    if batch.is_some() {
        return true;
    }
    *batch = prompt(if required {
        "批次入口（如 examples.models:shared_batch）"
    } else {
        "可选：批次入口，用于保存前验证候选前向（留空仅检查结构）"
    });
    !required || batch.is_some()
}

fn interactive_layer_report<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    model: &Option<String>,
    batch: &Option<String>,
) -> Option<Value> {
    let report = execute_ui(
        app,
        cli,
        &Command::Layer {
            command: LayerCommand::List {
                parent: None,
                inputs: input_snapshot(model, batch),
                layer_type: None,
                search: None,
            },
        },
    );
    if !report.success() {
        display(&report, false);
        return None;
    }
    Some(report.data)
}

/// Browse only the already collected layer report. Returns true when the caller should refresh.
fn browse_model_layers(rows: &[Value], report: &Value, probe: Option<&Value>) -> bool {
    let mut page = ModelTreePage::default();
    loop {
        println!("\n{}", render_model_tree(rows, &page));
        let Some(input) =
            prompt_allow_empty("搜索词/路径、序号详情、n/p 翻页、c <路径> 折叠、r 刷新、back 返回")
        else {
            return false;
        };
        let input = input.trim();
        match input {
            "back" | "b" | "返回" | "q" | "退出" => return false,
            "n" | "next" => page.next(rows),
            "p" | "prev" => page.previous(),
            "r" | "refresh" => return true,
            "" => page.set_query(""),
            _ if input.starts_with("c ") => match toggle_path(&mut page, rows, input[2..].trim()) {
                Ok(collapsed) => println!(
                    "{} {}。",
                    input[2..].trim(),
                    if collapsed { "已折叠" } else { "已展开" }
                ),
                Err(error) => println!("{error}"),
            },
            _ => {
                let visible = page.current_page(rows);
                let selected = input
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| index.checked_sub(1))
                    .and_then(|index| visible.get(index).copied())
                    .or_else(|| {
                        page.visible_rows(rows)
                            .into_iter()
                            .find(|row| row["path"].as_str() == Some(input))
                    });
                if let Some(row) = selected {
                    println!("\n{}", render_layer_detail(row, report, probe).join("\n"));
                    let _ = prompt_allow_empty("按 Enter 返回模型树");
                } else {
                    page.set_query(input);
                }
            }
        }
    }
}

fn browse_probe_success(report: &Value, model: Option<&str>, batch: Option<&str>) {
    let calls = report["calls"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut page = ProbeCallPage::default();
    loop {
        println!(
            "\n{}",
            render_probe_context(report, model, batch).join("\n")
        );
        println!("{}", render_call_list(calls, page.page(), 40));
        let Some(input) = prompt_allow_empty("调用序号、all、n/p 翻页或 back 返回")
        else {
            return;
        };
        match input.trim().to_ascii_lowercase().as_str() {
            "back" | "b" | "返回" | "q" => return,
            "n" | "next" => page.next(calls),
            "p" | "prev" => page.previous(),
            "all" => {
                for (index, call) in page.visible_calls(calls).iter().enumerate() {
                    render_call_snapshot(call, page.page() * 40 + index + 1);
                }
            }
            "" => {}
            _ => match page.select(input.trim(), calls) {
                Some(index) => browse_probe_call(&calls[index]),
                None => println!("请输入当前页序号，或输入 n/p 翻页。\n"),
            },
        }
    }
}

fn render_call_snapshot(call: &Value, ordinal: usize) {
    let ordinal = if ordinal == 0 {
        "调用".into()
    } else {
        format!("调用 #{ordinal}")
    };
    println!(
        "{ordinal} · {} · {} · 本模块第 {} 次",
        call["module_type"].as_str().unwrap_or("未知模块"),
        call["paths"]
            .as_array()
            .map(|paths| paths
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" | "))
            .unwrap_or_default(),
        call["invocation_index"].as_u64().unwrap_or(0)
    );
    println!(
        "输入：\n{}",
        TensorTreePage::new("输入").render(&call["inputs"], "输入")
    );
    println!(
        "输出：\n{}",
        TensorTreePage::new("输出").render(&call["outputs"], "输出")
    );
}

fn browse_probe_call(call: &Value) {
    loop {
        let paths = call["paths"]
            .as_array()
            .map(|paths| {
                paths
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .unwrap_or_default();
        println!(
            "\n模块调用 · {} · {} · 本模块第 {} 次 · call_id={} · parent={}",
            call["module_type"].as_str().unwrap_or("未知模块"),
            paths,
            call["invocation_index"].as_u64().unwrap_or(0),
            call["call_id"]
                .as_u64()
                .map_or_else(|| "未知".into(), |id| id.to_string()),
            call["parent_call_id"]
                .as_u64()
                .map_or_else(|| "无".into(), |id| id.to_string())
        );
        println!("1 查看输入端口\n2 查看输出端口\n3 查看输入与输出摘要\nback 返回调用列表");
        let Some(input) = prompt_allow_empty("选择端口") else {
            return;
        };
        match input.trim() {
            "1" | "input" | "输入" => browse_tensor_port(&call["inputs"], "输入"),
            "2" | "output" | "输出" => browse_tensor_port(&call["outputs"], "输出"),
            "3" | "all" | "全部" => render_call_snapshot(call, 0),
            "back" | "b" | "返回" | "q" => return,
            "" => {}
            _ => println!("请输入 1、2、3 或 back。"),
        }
    }
}

fn browse_tensor_port(value: &Value, label: &str) {
    let mut page = TensorTreePage::new(label);
    loop {
        println!("\n{}", page.render(value, label));
        let Some(input) = prompt_allow_empty("输入节点序号展开/折叠；back 返回端口选择")
        else {
            return;
        };
        if matches!(input.trim(), "back" | "b" | "返回" | "q") {
            return;
        }
        if input.trim().is_empty() {
            continue;
        }
        match input.trim().parse::<usize>() {
            Ok(index) => match page.toggle_visible(value, label, index) {
                Ok(_) => {}
                Err(error) => println!("{error}"),
            },
            Err(_) => println!("请输入节点序号或 back。"),
        }
    }
}

fn run_probe_interaction<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    model: &mut Option<String>,
    batch: &mut Option<String>,
    mut command: Command,
) -> Option<Value> {
    let (contract, contract_digest, batch_fingerprint, path, local, model_device) = match &command {
        Command::Model {
            command:
                ModelCommand::Probe {
                    contract,
                    if_contract_sha256,
                    batch_fingerprint,
                    path,
                    local,
                    model_device,
                    ..
                },
        } => (
            contract.clone(),
            if_contract_sha256.clone(),
            batch_fingerprint.clone(),
            path.clone(),
            *local,
            model_device.clone(),
        ),
        _ => (None, None, None, None, false, None),
    };
    loop {
        let report = execute_ui(app, cli, &command);
        display(&report, false);
        if report.success() {
            browse_probe_success(&report.data, model.as_deref(), batch.as_deref());
            return Some(report.data);
        }
        println!(
            "\n{}",
            render_probe_failure_context(&report.data, model.as_deref(), batch.as_deref())
                .join("\n")
        );
        let input = prompt_allow_empty(
            "1 查看项目源码、2 查看最后调用输入、3 更换批次、4 修复后重试、back 返回",
        )?;
        match resolve_failure_action(&input) {
            Some(ProbeFailureAction::ViewSource) => render_probe_source_context(&report.data),
            Some(ProbeFailureAction::InspectLastCall) => {
                let last = &report.data["failure_location"]["last_observed_module_call"];
                if last.is_object() {
                    browse_tensor_port(&last["inputs"], "失败调用输入");
                } else {
                    println!("尚未进入模块前向；请检查模型构造或批次工厂。\n");
                }
            }
            Some(ProbeFailureAction::ChangeBatch) => {
                let input = prompt_allow_empty(
                    "新批次入口 module:callable（Enter 保留当前批次；:cancel 返回）",
                )?;
                if input.trim() == ":cancel" {
                    return None;
                }
                if !input.trim().is_empty() {
                    *batch = Some(input.trim().to_owned());
                }
                command = Command::Model {
                    command: ModelCommand::Probe {
                        path: path.clone(),
                        inputs: input_snapshot(model, batch),
                        call: "all".into(),
                        contract: contract.clone(),
                        if_contract_sha256: contract_digest.clone(),
                        batch_fingerprint: batch_fingerprint.clone(),
                        model_device: model_device.clone(),
                        local,
                    },
                };
            }
            Some(ProbeFailureAction::Retry) => {
                command = Command::Model {
                    command: ModelCommand::Probe {
                        path: path.clone(),
                        inputs: input_snapshot(model, batch),
                        call: "all".into(),
                        contract: contract.clone(),
                        if_contract_sha256: contract_digest.clone(),
                        batch_fingerprint: batch_fingerprint.clone(),
                        model_device: model_device.clone(),
                        local,
                    },
                };
            }
            Some(ProbeFailureAction::Back) => return None,
            None => println!("请输入 1、2、3、4 或 back。"),
        }
    }
}

fn render_probe_source_context(report: &Value) {
    let frames = report["failure_location"]["project_stack"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    if frames.is_empty() {
        println!("失败报告没有项目源码位置；可检查上方错误信息和模型/批次入口。\n");
        return;
    }
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(root) => root,
        Err(error) => {
            println!("无法定位当前项目目录：{error}\n");
            return;
        }
    };
    for frame in frames {
        let Some(file) = frame["file"].as_str() else {
            continue;
        };
        let path = std::path::Path::new(file);
        let path = if path.is_absolute() {
            path.to_owned()
        } else {
            root.join(path)
        };
        let canonical = match path.canonicalize() {
            Ok(path) if path.starts_with(&root) => path,
            Ok(_) => {
                println!("跳过项目目录之外的源码路径：{file}");
                continue;
            }
            Err(error) => {
                println!("无法读取源码 {file}：{error}");
                continue;
            }
        };
        let Ok(source) = std::fs::read_to_string(&canonical) else {
            println!("无法以文本方式读取源码：{}", canonical.display());
            continue;
        };
        let Some(line) = frame["line"]
            .as_u64()
            .and_then(|line| usize::try_from(line).ok())
        else {
            println!("源码位置没有有效行号：{}", canonical.display());
            continue;
        };
        let total_lines = source.lines().count();
        let start = line.saturating_sub(3).max(1);
        let end = line.saturating_add(3).min(total_lines);
        println!(
            "\n{}:{} · {}",
            canonical
                .strip_prefix(&root)
                .unwrap_or(&canonical)
                .display(),
            line,
            frame["function"].as_str().unwrap_or("未知函数")
        );
        for (index, text) in source.lines().enumerate() {
            let number = index + 1;
            if (start..=end).contains(&number) {
                println!(
                    "{} {:>5} | {}",
                    if number == line { ">" } else { " " },
                    number,
                    text
                );
            }
        }
    }
    let _ = prompt_allow_empty("按 Enter 返回失败诊断");
}

fn prompt_layer_path(
    rows: &[Value],
    purpose: &str,
    direct_child_of: Option<&str>,
) -> Option<String> {
    let eligible: Vec<&Value> = rows
        .iter()
        .filter(|row| {
            let Some(parent) = direct_child_of else {
                return true;
            };
            row["path"].as_str().is_some_and(|path| {
                path.strip_prefix(parent)
                    .and_then(|suffix| suffix.strip_prefix('.'))
                    .is_some_and(|suffix| !suffix.is_empty() && !suffix.contains('.'))
            })
        })
        .collect();
    if eligible.is_empty() {
        println!("没有可供选择的 Layer。检查模型入口或父容器路径。");
        return None;
    }

    loop {
        let scope = direct_child_of
            .map(|path| format!("父容器 {path} 的直接子层"))
            .unwrap_or_else(|| "整个模型".to_owned());
        let query = prompt_allow_empty(&format!(
            "{purpose} · {scope}（输入路径/名称/类型搜索，留空浏览，q 返回）"
        ))?;
        if matches!(query.as_str(), "q" | "Q" | "quit" | "取消") {
            return None;
        }

        if let Some(row) = eligible
            .iter()
            .find(|row| row["path"].as_str() == Some(query.as_str()))
        {
            return row["path"].as_str().map(str::to_owned);
        }

        let needle = query.trim().to_lowercase();
        let matches: Vec<&Value> = eligible
            .iter()
            .copied()
            .filter(|row| {
                needle.is_empty()
                    || row["path"]
                        .as_str()
                        .is_some_and(|value| value.to_lowercase().contains(&needle))
                    || row["type"]
                        .as_str()
                        .is_some_and(|value| value.to_lowercase().contains(&needle))
                    || row["aliases"].as_array().is_some_and(|aliases| {
                        aliases.iter().any(|alias| {
                            alias
                                .as_str()
                                .is_some_and(|value| value.to_lowercase().contains(&needle))
                        })
                    })
            })
            .collect();
        if matches.is_empty() {
            println!("没有匹配的 Layer；试试模块名称、类型（如 Linear）或完整路径。");
            continue;
        }
        if matches.len() > 40 {
            println!(
                "匹配到 {} 个 Layer。请输入更具体的名称或类型。",
                matches.len()
            );
            continue;
        }
        println!("\n可选 Layer：");
        for (index, row) in matches.iter().enumerate() {
            let aliases = row["aliases"].as_array();
            let shared = aliases.is_some_and(|values| values.len() > 1);
            let mut fields: Vec<String> = row["editable_params"]
                .as_object()
                .into_iter()
                .flat_map(|params| params.keys().cloned())
                .chain(
                    row["source_editable_fields"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .map(str::to_owned),
                )
                .collect();
            fields.sort();
            fields.dedup();
            let editable = if fields.is_empty() {
                String::new()
            } else {
                format!(" · 可改参数={}", fields.join(","))
            };
            let limitation = row["source_set_block_reason"]
                .as_str()
                .map(|reason| format!(" · 源码受限={reason}"))
                .unwrap_or_default();
            println!(
                "  {}. {} · {} · 参数={}{}{}{}",
                index + 1,
                row["path"].as_str().unwrap_or("<无路径>"),
                row["type"].as_str().unwrap_or("未知类型"),
                row["parameters"].as_u64().unwrap_or(0),
                if shared { " · 共享实例" } else { "" },
                editable,
                limitation
            );
        }
        if matches.len() == 1 {
            return matches[0]["path"].as_str().map(str::to_owned);
        }
        let selection = prompt("输入序号选择，或继续输入关键词缩小范围")?;
        if matches!(selection.as_str(), "q" | "Q" | "quit" | "取消") {
            return None;
        }
        if let Ok(index) = selection.parse::<usize>()
            && let Some(row) = index.checked_sub(1).and_then(|index| matches.get(index))
        {
            return row["path"].as_str().map(str::to_owned);
        }
    }
}

fn prompt_addable_container_path(rows: &[Value], purpose: &str) -> Option<String> {
    let sequential: Vec<Value> = rows
        .iter()
        .filter(|row| row["can_insert"] == true)
        .cloned()
        .collect();
    if sequential.is_empty() {
        println!("模型中没有可直接编辑的 nn.Sequential、nn.ModuleList 或 nn.ModuleDict 容器。");
        return None;
    }
    prompt_layer_path(&sequential, purpose, None)
}

fn prompt_source_insertable_path(rows: &[Value], purpose: &str) -> Option<String> {
    let containers: Vec<Value> = rows
        .iter()
        .filter(|row| row["can_source_insert"] == true)
        .cloned()
        .collect();
    if containers.is_empty() {
        println!("模型中没有可直接编辑的 nn.Sequential、nn.ModuleList 或 nn.ModuleDict 容器。");
        return None;
    }
    prompt_layer_path(&containers, purpose, None)
}

fn prompt_container_add_name(rows: &[Value], parent: &str) -> Option<String> {
    if container_uses_indices(rows, parent) {
        Some(String::new())
    } else if container_uses_named_keys(rows, parent) {
        prompt("新 ModuleDict 分支名")
    } else {
        prompt("新 Sequential 子层名")
    }
}

fn container_uses_indices(rows: &[Value], parent: &str) -> bool {
    rows.iter()
        .any(|row| row["path"].as_str() == Some(parent) && row["type"] == "ModuleList")
}

fn container_uses_named_keys(rows: &[Value], parent: &str) -> bool {
    rows.iter()
        .any(|row| row["path"].as_str() == Some(parent) && row["type"] == "ModuleDict")
}

fn container_anchor_prompt(rows: &[Value], parent: &str, before: bool) -> &'static str {
    if container_uses_indices(rows, parent) {
        if before {
            "放在哪个 ModuleList 索引之前（如 0）"
        } else {
            "放在哪个 ModuleList 索引之后（如 0）"
        }
    } else if container_uses_named_keys(rows, parent) {
        if before {
            "放在哪个 ModuleDict 分支之前（输入分支名）"
        } else {
            "放在哪个 ModuleDict 分支之后（输入分支名）"
        }
    } else if before {
        "放在哪个子层之前（输入 Sequential 子层名）"
    } else {
        "放在哪个子层之后（输入 Sequential 子层名）"
    }
}

fn prompt_editable_layer_path(rows: &[Value], purpose: &str) -> Option<String> {
    let targets: Vec<Value> = rows
        .iter()
        .filter(|row| row["path"] != "model")
        .cloned()
        .collect();
    if targets.is_empty() {
        println!("模型中没有可编辑的子 Layer。");
        return None;
    }
    prompt_layer_path(&targets, purpose, None)
}

fn prompt_params(context: &str) -> Option<Vec<String>> {
    Some(parameters_for_cli(&prompt_parameter_map(context)?))
}

fn prompt_layer_patch(row: &Value) -> Option<Vec<String>> {
    let current = row["editable_params"].as_object()?;
    let schema = builtin_layer_schema(row["type"].as_str().unwrap_or(""))
        .filter(|schema| schema["properties"].is_object())
        .unwrap_or_else(|| {
            let properties = current
                .iter()
                .map(|(name, value)| {
                    let kind = match value {
                        Value::Bool(_) => "boolean",
                        Value::Number(_) => "number",
                        Value::String(_) => "string",
                        Value::Array(_) => "array",
                        Value::Object(_) => "object",
                        Value::Null => "any",
                    };
                    (
                        name.clone(),
                        serde_json::json!({
                            "type":kind,
                            "description":"当前 Layer 可编辑字段；留空保留原值"
                        }),
                    )
                })
                .collect::<serde_json::Map<_, _>>();
            serde_json::json!({"type":"object","properties":properties})
        });
    let path = row["path"].as_str().unwrap_or("unknown");
    prompt_schema_object(
        &schema,
        &format!("layer.patch.{path}"),
        Some(current),
        FormMode::Patch,
    )
    .map(|params| parameters_for_cli(&params))
}

fn prompt_external_loop_params() -> Option<Vec<String>> {
    println!(
        "完整循环自行读取数据并管理 forward/backward/optimizer；下面逐项输入参数名和值，值使用 JSON，例如 lr 对应 0.0001。"
    );
    println!(
        "需要可靠续训时，让循环入口 callable 提供成对 state_dict()/load_state_dict(state)；用户循环语义不会被 CLI 独立验证。"
    );
    prompt_params("training.external_loop")
}

fn dynamic_parameter_draft_identity(context: &str) -> (String, String) {
    let form_id = crate::domain::input::content_revision(&serde_json::json!({
        "form":"dynamic_parameter_map",
        "context":context
    }))
    .expect("dynamic form identity is serializable");
    let schema_revision = crate::domain::input::content_revision(&serde_json::json!({
        "kind":"dynamic_parameter_map",
        "version":1
    }))
    .expect("dynamic form schema identity is serializable");
    (form_id, schema_revision)
}

fn dynamic_param_key_is_sensitive(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    [
        "password",
        "secret",
        "token",
        "credential",
        "authorization",
        "apikey",
        "privatekey",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
}

fn dynamic_param_value_is_safe(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => true,
        Value::Array(values) => values.iter().all(dynamic_param_value_is_safe),
        Value::String(_) | Value::Object(_) => false,
    }
}

fn safe_dynamic_params(params: &serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
    params
        .iter()
        .filter(|(key, value)| {
            !dynamic_param_key_is_sensitive(key) && dynamic_param_value_is_safe(value)
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn persist_dynamic_params_draft(
    root: Option<&std::path::Path>,
    form_id: &str,
    schema_revision: &str,
    params: &serde_json::Map<String, Value>,
) {
    let Some(root) = root else {
        return;
    };
    let safe = safe_dynamic_params(params);
    let result = if safe.is_empty() {
        session_store::clear_form_draft(root, form_id)
    } else {
        session_store::save_form_draft(root, form_id, schema_revision, &safe)
    };
    if let Err(error) = result {
        eprintln!("无法保存参数恢复点：{error}");
    }
}

fn prompt_parameter_map(context: &str) -> Option<serde_json::Map<String, Value>> {
    let root = std::env::current_dir()
        .ok()
        .and_then(|path| path.canonicalize().ok());
    let (form_id, schema_revision) = dynamic_parameter_draft_identity(context);
    let mut params = match root.as_deref() {
        Some(root) => match session_store::load_form_draft(root, &form_id, &schema_revision) {
            Ok(Some(draft)) if draft.stale => {
                println!("动态参数定义已变化，旧草稿不能直接恢复。");
                match prompt_allow_empty("1 丢弃旧草稿并重新填写，2 返回（保留草稿）")?.as_str()
                {
                    "1" => {
                        if let Err(error) = session_store::clear_form_draft(root, &form_id) {
                            eprintln!("无法丢弃旧参数草稿：{error}");
                            return None;
                        }
                        serde_json::Map::new()
                    }
                    _ => return None,
                }
            }
            Ok(Some(draft)) => {
                println!(
                    "发现上次未完成的参数草稿（{} 个安全字段）。",
                    draft.values.len()
                );
                match prompt_allow_empty("1 恢复草稿，2 丢弃并重新填写，3 返回")?.as_str()
                {
                    "1" => safe_dynamic_params(&draft.values),
                    "2" => {
                        if let Err(error) = session_store::clear_form_draft(root, &form_id) {
                            eprintln!("无法丢弃参数草稿：{error}");
                            return None;
                        }
                        serde_json::Map::new()
                    }
                    _ => return None,
                }
            }
            Ok(None) => serde_json::Map::new(),
            Err(error) => {
                eprintln!("无法恢复参数草稿：{error}；未覆盖旧草稿。");
                return None;
            }
        },
        None => serde_json::Map::new(),
    };
    loop {
        let name = prompt_allow_empty("参数名称（留空完成，:cancel 取消整组参数）")?;
        if name.is_empty() {
            if let Some(root) = root.as_deref()
                && let Err(error) = session_store::clear_form_draft(root, &form_id)
            {
                eprintln!("参数已提交，但未能清除恢复点：{error}");
            }
            return Some(params);
        }
        let name = name.trim();
        if name == ":cancel" {
            if let Some(root) = root.as_deref()
                && let Err(error) = session_store::clear_form_draft(root, &form_id)
            {
                eprintln!("参数已取消，但未能清除恢复点：{error}");
            }
            return None;
        }
        if name.is_empty() || params.contains_key(name) {
            eprintln!("参数名不能为空或重复；请重试。\n");
            continue;
        }
        let schema = serde_json::json!({
            "type":"any",
            "description":"输入 JSON 值：数字、字符串、布尔值、数组或对象。"
        });
        let value = prompt_single_schema_value(&format!("params.{name}"), schema)?;
        params.insert(name.to_owned(), value);
        persist_dynamic_params_draft(root.as_deref(), &form_id, &schema_revision, &params);
    }
}

fn parameters_for_cli(params: &serde_json::Map<String, Value>) -> Vec<String> {
    params
        .iter()
        .map(|(name, value)| {
            format!(
                "{name}={}",
                serde_json::to_string(value).expect("parameter values are JSON")
            )
        })
        .collect()
}

fn prompt_interactive_params(context: &str) -> Option<serde_json::Map<String, Value>> {
    prompt_parameter_map(context)
}

fn source_plan_command(inputs: &Inputs, operations: &[String], dry_run: bool) -> Command {
    Command::Layer {
        command: LayerCommand::Apply {
            operations: operations.to_vec(),
            edit: EditOptions {
                variant_device_policy: None,
                layer_rng_devices: Vec::new(),
                base_rng_scope: None,
                base_rng_devices: Vec::new(),
                inputs: inputs.clone(),
                target: "source".into(),
                params: Vec::new(),
                dry_run,
                require_forward: false,
            },
        },
    }
}

fn source_operation_changes_tree(operation: &Value) -> bool {
    matches!(
        operation["action"].as_str(),
        Some("add" | "replace" | "remove")
    )
}

fn refresh_source_plan_rows<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    inputs: &Inputs,
    operations: &[String],
) -> Option<Vec<Value>> {
    let mut structural_inputs = inputs.clone();
    structural_inputs.batch = None;
    structural_inputs.batch_params.clear();
    structural_inputs.batch_binding = None;
    structural_inputs.training = false;
    let report = execute_ui(
        app,
        cli,
        &source_plan_command(&structural_inputs, operations, true),
    );
    if !report.success() {
        display(&report, false);
        return None;
    }
    Some(
        report.data["modules"]
            .as_array()
            .cloned()
            .unwrap_or_default(),
    )
}

fn preview_confirm_source_plan<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    inputs: Inputs,
    operations: Vec<String>,
    confirmation_label: &str,
) {
    let preview = execute_ui(app, cli, &source_plan_command(&inputs, &operations, true));
    display(&preview, false);
    if preview.success()
        && prompt(confirmation_label).is_some_and(|answer| matches!(answer.as_str(), "yes" | "y"))
    {
        let (Some(plan), Some(manifest)) = (
            preview.data["source_plan"]["digest"].as_str(),
            preview.data["source_manifest_evidence"]["digest"].as_str(),
        ) else {
            println!("预览缺少可核对的源码计划或清单；原文件未修改。");
            return;
        };
        let mut confirmed_cli = cli.clone();
        confirmed_cli.if_source_plan = Some(plan.to_owned());
        confirmed_cli.if_source_manifest = Some(manifest.to_owned());
        confirmed_cli.if_source_config = preview.data["config_revision"].as_str().map(str::to_owned);
        let validation = &preview.data["validation"];
        if validation["forward"] == "passed" {
            let Some(input) = validation["batch_content"]["digest"].as_str() else {
                println!("预览没有可核对的前向输入；原文件未修改。请提供稳定输入后重试。");
                return;
            };
            confirmed_cli.if_input_content = Some(input.to_owned());
        }
        display(
            &execute_ui(app, &confirmed_cli, &source_plan_command(&inputs, &operations, false)),
            false,
        );
    } else if preview.success() {
        println!("已取消；原文件未修改。");
    }
}

fn prompt_single_schema_value(label: &str, schema: Value) -> Option<Value> {
    let form_schema = serde_json::json!({
        "type":"object",
        "required":["value"],
        "properties":{"value":schema}
    });
    prompt_schema_object(&form_schema, label, None, FormMode::Full)?.remove("value")
}

fn prompt_training_steps() -> Option<u32> {
    prompt_single_schema_value(
        "try.steps",
        serde_json::json!({"type":"integer","minimum":1,"maximum":1000,"default":3,"description":"试跑执行步数"}),
    )?
    .as_u64()
    .and_then(|value| u32::try_from(value).ok())
}

fn prompt_formal_training_steps(label: &str, default: u32) -> Option<u32> {
    prompt_single_schema_value(
        label,
        serde_json::json!({"type":"integer","minimum":1,"maximum":1000000,"default":default,"description":"训练步数预算"}),
    )?
    .as_u64()
    .and_then(|value| u32::try_from(value).ok())
}

fn prompt_positive_float(label: &str, default: f64, allow_zero: bool) -> Option<f64> {
    let minimum = if allow_zero { 0.0 } else { f64::MIN_POSITIVE };
    prompt_single_schema_value(
        label,
        serde_json::json!({"type":"number","minimum":minimum,"default":default,"description":if allow_zero {"可输入零或正数"} else {"必须大于零"}}),
    )?
    .as_f64()
}

fn prompt_checkpoint_interval() -> Option<u32> {
    prompt_formal_training_steps("checkpoint 保存间隔", 50)
}

fn prompt_detach() -> bool {
    prompt("训练脱离终端继续运行？y/N")
        .is_some_and(|answer| matches!(answer.to_ascii_lowercase().as_str(), "y" | "yes"))
}

fn prompt_run_name() -> Option<String> {
    prompt("运行名称（可留空）")
}

fn prompt_training_mode() -> Option<TrainingMode> {
    println!(
        "\n{}",
        crate::presentation::pages::training::render_mode_guide()
    );
    loop {
        match prompt("选择模式 [1]") {
            None => return None,
            Some(value)
                if value.is_empty()
                    || value == "1"
                    || value.eq_ignore_ascii_case("managed-step") =>
            {
                return Some(TrainingMode::ManagedStep);
            }
            Some(value) if value == "back" || value == "b" => return None,
            Some(value) if value == "2" || value.eq_ignore_ascii_case("manual-step") => {
                return Some(TrainingMode::ManualStep);
            }
            Some(value) if value == "3" || value.eq_ignore_ascii_case("external-loop") => {
                return Some(TrainingMode::ExternalLoop);
            }
            Some(_) => eprintln!("请输入 1、2、3 或 back。"),
        }
    }
}

fn attach_follow<B: Backend>(app: &Application<B>, cli: &Cli, requested: Option<String>) -> Report {
    let one_shot = |id| Command::Runs {
        command: RunsCommand::Attach {
            id,
            follow: Some(false),
        },
    };
    let mut report = execute_ui(app, cli, &one_shot(requested));
    if !report.success() {
        return report;
    }
    let Some(run_id) = report.data["run_id"].as_str().map(str::to_owned) else {
        return Report::error(
            "runs_attach",
            "RUN_RECORD_INVALID",
            "附加结果缺少运行 ID。",
            "运行 m-model runs list 查看记录。",
        );
    };
    eprintln!("已附加到 {run_id}；Ctrl+C 退出查看，训练会继续运行。\n");
    let mut seen_step = 0_u64;
    loop {
        if app.supervisor.take_user_interrupt() {
            eprintln!("已退出进度查看；后台运行继续。\n");
            report.data["viewer_detached"] = serde_json::json!(true);
            report.data["viewer_message"] = serde_json::json!("已退出进度查看；后台运行继续。 ");
            return report;
        }
        if let Some(progress) = report.data["progress"].as_array() {
            let start = if seen_step == 0 && progress.len() > 10 {
                progress.len() - 1
            } else {
                0
            };
            for item in &progress[start..] {
                let step = item["step"].as_u64().unwrap_or(0);
                if step > seen_step {
                    println!(
                        "训练进度：{}/{} 步 · 平均 Loss {}{}",
                        step,
                        item["total"],
                        item["mean_loss"],
                        if item["checkpoint"] == true {
                            " · 已保存 checkpoint"
                        } else {
                            ""
                        }
                    );
                    seen_step = step;
                }
            }
        }
        let status = report.data["record"]["status"]
            .as_str()
            .unwrap_or("unknown");
        if !matches!(status, "running" | "stopping") {
            return report;
        }
        std::thread::sleep(Duration::from_millis(750));
        report = execute_ui(app, cli, &one_shot(Some(run_id.clone())));
        if !report.success() {
            return report;
        }
    }
}

fn preview_form_error(code: &str, message: &str, hint: &str) -> Report {
    Report::error("preview", code, message, hint)
}

fn preview_form_cancelled() -> Report {
    preview_form_error(
        "USER_INTERRUPTED",
        "已取消数据预览设置；没有启动预览。",
        "再次运行 m-model data preview。",
    )
}

fn preview_form_value(label: &str, current: &str, default: &str) -> Result<String, Report> {
    let shown = if current.is_empty() { default } else { current };
    let answer =
        prompt_allow_empty(&format!("{label} [{shown}]")).ok_or_else(preview_form_cancelled)?;
    let answer = answer.trim();
    if matches!(
        answer.to_ascii_lowercase().as_str(),
        "back" | "cancel" | ":cancel" | "取消"
    ) {
        return Err(preview_form_cancelled());
    }
    Ok(if answer.is_empty() {
        shown.to_owned()
    } else if answer == "-" {
        String::new()
    } else {
        answer.to_owned()
    })
}

fn preview_form_required(label: &str, current: &str) -> Result<String, Report> {
    loop {
        let value = preview_form_value(label, current, "")?;
        if !value.is_empty() {
            return Ok(value);
        }
        eprintln!("{label} 不能为空；请重新输入或取消预览。\n");
    }
}

fn prompt_preview_command(command: &Command) -> Result<Command, Report> {
    let Command::Data {
        command:
            DataCommand::Preview {
                dataset: current_dataset,
                batch: current_batch,
                whole_batch,
                parameters,
                split,
                no_split,
                augment_split,
                samples,
                scan_limit,
                fingerprint,
                transforms,
                renderer,
                renderer_entrypoint,
                renderer_version,
                delivery,
                ssh_host,
                ssh_user,
                ssh_port,
                destination,
                ..
            },
    } = command
    else {
        unreachable!("preview form is only used for data preview commands")
    };

    let current_kind = if current_dataset.is_some() {
        "dataset"
    } else if current_batch.is_some() {
        "batch"
    } else {
        ""
    };
    let (dataset, batch) = loop {
        let value = preview_form_value("预览来源：dataset / batch", current_kind, "batch")?;
        match value.trim().to_ascii_lowercase().as_str() {
            "dataset" => break (true, false),
            "batch" => break (false, true),
            _ => eprintln!("请输入 dataset 或 batch；输入 back 取消。\n"),
        }
    };
    let current_entrypoint = current_dataset
        .as_deref()
        .or(current_batch.as_deref())
        .unwrap_or("");
    let entrypoint = preview_form_required(
        if dataset {
            "Dataset 工厂入口（如 project.data:build_dataset）"
        } else {
            "批次工厂入口（如 project.data:example_batch）"
        },
        current_entrypoint,
    )?;
    let selected_whole_batch = if batch {
        loop {
            let current = if *whole_batch { "whole" } else { "auto" };
            let value = preview_form_value("批次返回值：auto 自动识别数据集 / whole 整体作为一项", current, "auto")?;
            match value.trim().to_ascii_lowercase().as_str() {
                "auto" => break false,
                "whole" => break true,
                _ => eprintln!("请输入 auto 或 whole；输入 back 取消。\n"),
            }
        }
    } else {
        false
    };

    let mut parameters = parameters.clone();
    loop {
        let input = prompt_allow_empty("可选工厂参数 NAME=JSON；空行完成，back 取消")
            .ok_or_else(preview_form_cancelled)?;
        let input = input.trim();
        if input.is_empty() {
            break;
        }
        if matches!(input, "back" | "cancel" | "取消") {
            return Err(preview_form_cancelled());
        }
        let Some((name, raw)) = input.split_once('=') else {
            eprintln!("格式应为 NAME=JSON，例如 size=224；请重新输入。\n");
            continue;
        };
        let name = name.trim();
        if name.is_empty() {
            eprintln!("参数名称不能为空；请重新输入。\n");
            continue;
        }
        let value = match serde_json::from_str::<Value>(raw.trim()) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("参数值不是有效 JSON：{error}；请重新输入。\n");
                continue;
            }
        };
        let normalized = format!("{name}={}", serde_json::to_string(&value).unwrap());
        if let Some(existing) = parameters.iter_mut().find(|parameter| {
            parameter
                .split_once('=')
                .is_some_and(|(existing, _)| existing.trim() == name)
        }) {
            *existing = normalized;
        } else {
            parameters.push(normalized);
        }
    }

    let split = if *no_split {
        println!("本次不指定切分，也不自动应用项目 train/eval 增强。\n");
        split.clone()
    } else {
        loop {
            let value = preview_form_value("切分名称", split, "train")?;
            if !value.is_empty() {
                break value;
            }
            eprintln!("切分名称不能为空；请重新输入或取消预览。\n");
        }
    };
    let samples = loop {
        let value = preview_form_value("最多预览样本数（1–64）", &samples.to_string(), "12")?;
        match value.parse::<usize>() {
            Ok(value @ 1..=64) => break value,
            _ => eprintln!("样本数请输入 1 到 64 的整数。\n"),
        }
    };
    let fingerprint = preview_form_value(
        "可选数据/切分指纹（- 清除）",
        fingerprint.as_deref().unwrap_or(""),
        "",
    )?;
    let fingerprint = (!fingerprint.is_empty()).then_some(fingerprint);
    let transforms = preview_form_value(
        "可选有序变换 JSON 文件（- 清除）",
        &transforms
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default(),
        "",
    )?;
    let transforms = (!transforms.is_empty()).then(|| PathBuf::from(transforms));

    let renderer = loop {
        let value = preview_form_value("Renderer：auto / summary / custom", renderer, "auto")?;
        match value.as_str() {
            "auto" | "summary" | "custom" => break value,
            _ => eprintln!("Renderer 请输入 auto、summary 或 custom。\n"),
        }
    };
    let (renderer_entrypoint, renderer_version) = if renderer == "custom" {
        let entrypoint = preview_form_required(
            "自定义 Renderer 工厂入口 module:callable",
            renderer_entrypoint.as_deref().unwrap_or(""),
        )?;
        let version = preview_form_required(
            "Renderer 版本标记（代码变化时更新）",
            renderer_version.as_deref().unwrap_or(""),
        )?;
        (Some(entrypoint), Some(version))
    } else {
        (None, None)
    };

    let delivery = loop {
        let default = match delivery {
            PreviewDelivery::Local => "local",
            PreviewDelivery::Ssh => "ssh",
        };
        match preview_form_value("交付方式：local / ssh", default, "local")?
            .to_ascii_lowercase()
            .as_str()
        {
            "local" => break PreviewDelivery::Local,
            "ssh" => break PreviewDelivery::Ssh,
            _ => eprintln!("交付方式请输入 local 或 ssh。\n"),
        }
    };
    let (ssh_host, ssh_user, ssh_port, destination) = match delivery {
        PreviewDelivery::Local => (None, None, None, destination.clone()),
        PreviewDelivery::Ssh => {
            let host = preview_form_required("SSH 主机名/IP", ssh_host.as_deref().unwrap_or(""))?;
            let user = preview_form_required("SSH 用户名", ssh_user.as_deref().unwrap_or(""))?;
            let port = loop {
                let current = ssh_port.map(|value| value.to_string()).unwrap_or_default();
                let value = preview_form_value("SSH 端口", &current, "22")?;
                match value.parse::<u16>() {
                    Ok(value) if value > 0 => break value,
                    _ => eprintln!("端口请输入 1 到 65535 的整数。\n"),
                }
            };
            let path = preview_form_value(
                "本机保存路径",
                &destination.to_string_lossy(),
                "./preview.html",
            )?;
            (Some(host), Some(user), Some(port), PathBuf::from(path))
        }
    };

    println!(
        "预览计划：{} · 切分 {split} · 最多 {samples} 个样本 · {renderer} · {}",
        if dataset { "Dataset" } else { "批次工厂" },
        match delivery {
            PreviewDelivery::Local => "本机文件".to_owned(),
            PreviewDelivery::Ssh => format!("SSH 取回指引（端口 {}）", ssh_port.unwrap_or(0)),
        }
    );
    if !parameters.is_empty() {
        println!(
            "已设置 {} 个工厂参数；具体值不会在摘要中回显。",
            parameters.len()
        );
    }
    if selected_whole_batch {
        println!("批次工厂的整个返回值将作为一项预览，不自动采样数据集；项目增强仍按当前选择应用。");
    }
    if let Some(path) = &transforms {
        println!("有序变换：{}", path.display());
    }

    Ok(Command::Data {
        command: DataCommand::Preview {
            dataset: dataset.then(|| entrypoint.trim().to_owned()),
            batch: batch.then(|| entrypoint.trim().to_owned()),
            whole_batch: selected_whole_batch,
            parameters,
            split,
            no_split: *no_split,
            augment_split: augment_split.clone(),
            samples,
            scan_limit: *scan_limit,
            fingerprint,
            transforms,
            renderer,
            renderer_entrypoint,
            renderer_version,
            delivery,
            ssh_host,
            ssh_user,
            ssh_port,
            destination,
        },
    })
}

pub fn run_command<B: Backend>(app: &Application<B>, cli: &Cli, command: &Command) -> Report {
    match crate::app::normalize_training_command(command) {
        Ok(Some(normalized)) => return run_command(app, cli, &normalized),
        Ok(None) => {},
        Err(report) => return report,
    }
    if let Command::Model {
        command:
            ModelCommand::Supervise {
                command: SuperviseCommand::Guide { inputs },
            },
    } = command
    {
        if !is_interactive(cli.json, cli.no_interactive) {
            return Report::error(
                "semantic_supervision_guide",
                "GUIDED_FLOW_REQUIRES_TTY",
                "监督引导需要交互终端；JSON 和脚本场景请使用分步命令。",
                "先运行 model probe，再使用 model mark、model supervise rule add/preview；penalty 仅在 train/try 显式传 --apply-supervision 后改变训练目标。",
            );
        }
        return run_supervision_guide(app, cli, inputs.clone());
    }

    if let Command::Data {
        command:
            DataCommand::Index {
                root,
                verify: true,
                exclude_files,
                apply_exclusions: false,
                clear_exclusions: false,
            },
    } = command
        && is_interactive(cli.json, cli.no_interactive)
    {
        return run_data_index_recovery(app, cli, root, exclude_files);
    }
    if let Command::Data {
        command: DataCommand::Index {
            clear_exclusions: true,
            ..
        },
    } = command
        && is_interactive(cli.json, cli.no_interactive)
    {
        if prompt("将复验全部图像，并仅移除项目中的排除协议；输入 CLEAR 确认，其他输入取消")
            .is_none_or(|answer| answer.trim() != "CLEAR")
        {
            return Report::error(
                "data_index",
                "USER_INTERRUPTED",
                "用户取消了排除协议清除。",
                "未修改项目协议或数据文件。",
            );
        }
        let report = execute_ui(app, cli, command);
        if report.success() {
            println!(
                "{}",
                crate::presentation::pages::data_index::render(&report.data).join("\n")
            );
        } else {
            display(&report, false);
        }
        return report;
    }

    let prompted_command = if is_interactive(cli.json, cli.no_interactive)
        && matches!(
            command,
            Command::Data {
                command: DataCommand::Preview {
                    dataset: None,
                    batch: None,
                    ..
                }
            }
        ) {
        match prompt_preview_command(command) {
            Ok(command) => Some(command),
            Err(report) => return report,
        }
    } else {
        None
    };
    if let Some(command) = prompted_command {
        return run_preview_interaction(app, cli, command);
    }
    if let Command::Runs {
        command: RunsCommand::Attach { id, follow },
    } = command
    {
        let should_follow = follow
            .unwrap_or(!cli.json && ApplicationUiServices::new(app, cli).stderr_is_terminal());
        if should_follow && cli.json {
            return Report::error(
                "runs_attach",
                "JSON_FOLLOW_UNSUPPORTED",
                "JSON 模式下持续跟随会产生多个文档。",
                "使用 --json runs attach --follow=false 获取一次结构化快照。",
            );
        }
        if should_follow {
            return attach_follow(app, cli, id.clone());
        }
    }
    execute_ui(app, cli, command)
}

fn run_data_index_recovery<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    root: &Path,
    exclude_files: &[String],
) -> Report {
    let inspect = Command::Data {
        command: DataCommand::Index {
            root: root.to_path_buf(),
            verify: true,
            exclude_files: exclude_files.to_vec(),
            apply_exclusions: false,
            clear_exclusions: false,
        },
    };
    let report = execute_ui(app, cli, &inspect);
    if report.success() {
        println!(
            "{}",
            crate::presentation::pages::data_index::render(&report.data).join("\n")
        );
        return report;
    }
    display(&report, false);
    let Some(diagnostic) = report
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "CLASS_SAMPLES_DAMAGED")
    else {
        return report;
    };
    let _ = diagnostic;
    let damaged = report.data["damaged_files"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if damaged.is_empty() {
        return report;
    }
    println!("本次验证发现损坏样本；不会自动替换或删除文件：");
    for (index, item) in damaged.iter().enumerate() {
        println!(
            "  {}. {} · {} · {}",
            index + 1,
            item["path"].as_str().unwrap_or("未知路径"),
            item["label"].as_str().unwrap_or("未知类别"),
            item["error"].as_str().unwrap_or("无法解码")
        );
    }
    loop {
        let Some(input) = prompt_allow_empty(
            "输入要排除的编号（逗号分隔）、all 全部排除，或 back 保留文件并返回",
        ) else {
            return report;
        };
        let selected = if input.trim().eq_ignore_ascii_case("all") {
            (0..damaged.len()).collect::<Vec<_>>()
        } else if matches!(
            input.trim().to_ascii_lowercase().as_str(),
            "back" | "b" | "返回" | "q"
        ) {
            return report;
        } else {
            let parsed = input
                .split(',')
                .map(str::trim)
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>();
            match parsed {
                Ok(indices)
                    if !indices.is_empty()
                        && indices
                            .iter()
                            .all(|index| (1..=damaged.len()).contains(index)) =>
                {
                    indices.into_iter().map(|index| index - 1).collect()
                }
                _ => {
                    println!("请输入损坏清单中的序号、all 或 back。\n");
                    continue;
                }
            }
        };
        let mut selected = selected;
        selected.sort_unstable();
        selected.dedup();
        let paths = selected
            .iter()
            .filter_map(|index| damaged[*index]["path"].as_str().map(str::to_owned))
            .collect::<Vec<_>>();
        println!("排除预览 · 保留原始文件");
        for path in &paths {
            println!("  {path}");
        }
        if prompt("将排除清单保存到当前 Model-Maven 项目，并改变后续分类批次的数据指纹？输入 EXCLUDE 确认").is_none_or(|answer| answer.trim() != "EXCLUDE")
        {
            println!("已取消；项目协议与数据文件未改变。\n");
            return report;
        }
        let apply = execute_ui(
            app,
            cli,
            &Command::Data {
                command: DataCommand::Index {
                    root: root.to_path_buf(),
                    verify: true,
                    exclude_files: paths,
                    apply_exclusions: true,
                    clear_exclusions: false,
                },
            },
        );
        if apply.success() {
            println!(
                "{}",
                crate::presentation::pages::data_index::render(&apply.data).join("\n")
            );
        } else {
            display(&apply, false);
        }
        return apply;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SupervisionCandidate {
    module_path: String,
    port: String,
    module_type: String,
}

fn guide_required(label: &str, current: Option<&str>) -> Result<String, Report> {
    loop {
        let label = match current.filter(|value| !value.is_empty()) {
            Some(value) => format!("{label} [{value}，回车采用]"),
            None => label.to_owned(),
        };
        let Some(value) = prompt_allow_empty(&label) else {
            return Err(Report::error(
                "semantic_supervision_guide",
                "USER_INTERRUPTED",
                "用户取消了监督引导。",
                "可重新运行 m-model model supervise guide。",
            ));
        };
        if matches!(
            value.to_ascii_lowercase().as_str(),
            "back" | "b" | "返回" | "q"
        ) {
            return Err(Report::error(
                "semantic_supervision_guide",
                "USER_INTERRUPTED",
                "用户返回并退出监督引导。",
                "已完成并保存的语义标记仍保留；可重新运行引导继续添加规则。",
            ));
        }
        let value = if value.is_empty() {
            current.unwrap_or_default().to_owned()
        } else {
            value
        };
        if !value.trim().is_empty() {
            return Ok(value.trim().to_owned());
        }
        eprintln!("此项必填。输入 back 可退出。\n");
    }
}

fn guide_choice(label: &str, choices: &[&str]) -> Result<usize, Report> {
    loop {
        let Some(value) = prompt_allow_empty(label) else {
            return Err(Report::error(
                "semantic_supervision_guide",
                "USER_INTERRUPTED",
                "用户取消了监督引导。",
                "已完成并保存的语义标记仍保留；可重新运行引导继续添加规则。",
            ));
        };
        if let Some(index) = value
            .parse::<usize>()
            .ok()
            .and_then(|value| value.checked_sub(1))
            .filter(|index| *index < choices.len())
        {
            return Ok(index);
        }
        if matches!(
            value.to_ascii_lowercase().as_str(),
            "back" | "b" | "返回" | "q"
        ) {
            return Err(Report::error(
                "semantic_supervision_guide",
                "USER_INTERRUPTED",
                "用户返回并退出监督引导。",
                "已完成并保存的语义标记仍保留；可重新运行引导继续添加规则。",
            ));
        }
        eprintln!("请输入有效序号。\n");
    }
}

fn guide_optional_number(label: &str) -> Result<Option<f64>, Report> {
    loop {
        let Some(value) = prompt_allow_empty(label) else {
            return Err(Report::error(
                "semantic_supervision_guide",
                "USER_INTERRUPTED",
                "用户取消了监督引导。",
                "已完成并保存的标记仍保留。",
            ));
        };
        if value.is_empty() {
            return Ok(None);
        }
        if let Ok(value) = value.parse::<f64>()
            && value.is_finite()
        {
            return Ok(Some(value));
        }
        eprintln!("请输入有限数字，或回车留空。\n");
    }
}

fn run_supervision_guide<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    mut inputs: Inputs,
) -> Report {
    let root = match std::env::current_dir() {
        Ok(root) => root,
        Err(error) => {
            return Report::error(
                "semantic_supervision_guide",
                "PROJECT_UNAVAILABLE",
                error.to_string(),
                "从模型项目根目录运行引导命令。",
            );
        }
    };
    let project_config = match app.project_config.load(&root) {
        Ok(config) => config,
        Err(error) => {
            return Report::error(
                "semantic_supervision_guide",
                "PROJECT_CONFIG_INVALID",
                error,
                "先修复 m-model init/config 的项目配置错误，再运行引导。",
            );
        }
    };
    if inputs.model.is_none() && cli.variant.is_none() {
        inputs.model = project_config
            .as_ref()
            .and_then(|config| config["model"].as_str())
            .map(str::to_owned);
    }
    if inputs.batch.is_none() {
        inputs.batch = project_config
            .as_ref()
            .and_then(|config| config["batch"].as_str())
            .map(str::to_owned);
    }
    if inputs.model.is_none() && cli.variant.is_none() {
        inputs.model = match guide_required("模型工厂入口（module:callable）", None) {
            Ok(value) => Some(value),
            Err(report) => return report,
        };
    }
    if inputs.batch.is_none() {
        inputs.batch = match guide_required("代表性批次工厂入口（module:callable）", None)
        {
            Ok(value) => Some(value),
            Err(report) => return report,
        };
    }

    eprintln!("\n第 1 步 · 运行真实前向，列出本次观察到的实现块和端口。\n");
    let probe = execute_ui(
        app,
        cli,
        &Command::Model {
            command: ModelCommand::Probe {
                path: None,
                inputs: inputs.clone(),
                call: "all".into(),
                contract: None,
                if_contract_sha256: None,
                batch_fingerprint: None,
                model_device: None,
                local: false,
            },
        },
    );
    if !probe.success() {
        return probe;
    }
    if probe.data["trace_truncated"] == true {
        return Report::error(
            "semantic_supervision_guide",
            "PROBE_TRACE_TRUNCATED",
            "调用轨迹被截断，当前无法安全展示完整标记候选。",
            "先运行 m-model model probe --path <更具体的模块路径> 缩小观测范围，再按输出中的路径和端口手动标记。",
        );
    }

    let mut candidates = BTreeMap::<(String, String), String>::new();
    if let Some(calls) = probe.data["calls"].as_array() {
        for call in calls {
            let module_type = call["module_type"].as_str().unwrap_or("模块").to_owned();
            let paths = call["paths"].as_array().into_iter().flatten();
            let ports = call["input_output_paths"].as_array().into_iter().flatten();
            for path in paths
                .filter_map(Value::as_str)
                .filter(|path| path.starts_with("model."))
            {
                for port in ports.clone().filter_map(Value::as_str) {
                    candidates
                        .entry((path.to_owned(), port.to_owned()))
                        .or_insert_with(|| module_type.clone());
                }
            }
        }
    }
    if candidates.is_empty() {
        return Report::error(
            "semantic_supervision_guide",
            "SUPERVISION_CANDIDATES_MISSING",
            "本次真实前向没有发现可标记的模型模块端口。",
            "检查代表批次是否经过目标模块；也可运行 m-model model probe 查看当前调用树。",
        );
    }

    eprintln!("请选择要手动赋予语义的实现块端口；工具不会猜测它的含义：\n");
    let candidates = candidates
        .into_iter()
        .map(|((module_path, port), module_type)| SupervisionCandidate {
            module_path,
            port,
            module_type,
        })
        .collect::<Vec<_>>();
    for (index, candidate) in candidates.iter().enumerate() {
        eprintln!(
            "  {}. {} ({}) · {}",
            index + 1,
            candidate.module_path,
            candidate.module_type,
            candidate.port
        );
    }
    eprintln!("\n本次候选仅来自当前批次实际执行的分支；未观察分支保持未知。\n");
    let choices = candidates.iter().map(|_| "候选").collect::<Vec<_>>();
    let selected = match guide_choice("实现块/端口序号（输入 back 返回）", &choices) {
        Ok(index) => &candidates[index],
        Err(report) => return report,
    };
    let role = match guide_required("语义角色（自由命名，例如 embedding、logits、action）", None)
    {
        Ok(value) => value,
        Err(report) => return report,
    };
    let meaning = match guide_required("这段实现的含义（由你填写）", None) {
        Ok(value) => value,
        Err(report) => return report,
    };

    eprintln!(
        "\n第 2 步 · 保存人工标记：{} · {} · {}\n含义：{}\n",
        selected.module_path, selected.port, role, meaning
    );
    let mark_report = execute_ui(
        app,
        cli,
        &Command::Model {
            command: ModelCommand::Mark {
                inputs: inputs.clone(),
                path: selected.module_path.clone(),
                port: selected.port.clone(),
                role: role.clone(),
                meaning: meaning.clone(),
                id: None,
                if_anchor: None,
                model_device: None,
            },
        },
    );
    if !mark_report.success() {
        return mark_report;
    }
    let mark_id = match mark_report.data["mark"]["id"].as_str() {
        Some(id) => id.to_owned(),
        None => {
            return Report::error(
                "semantic_supervision_guide",
                "SEMANTIC_MARK_RESULT_INVALID",
                "标记已提交，但服务结果没有返回标记 ID。",
                "运行 m-model model marks 查看保存结果。",
            );
        }
    };
    let source_file = mark_report.data["mark"]["anchor"]["source_file"]
        .as_str()
        .unwrap_or("未知源码");
    let source_digest = mark_report.data["mark"]["anchor"]["source_sha256"]
        .as_str()
        .unwrap_or("未报告");
    eprintln!(
        "标记 ID：{mark_id}；实现来源 {source_file} · SHA-256 {source_digest}；绑定到当前模型配置。\n"
    );

    eprintln!(
        "第 3 步 · 定义一条绑定到该语义标记的规则。\n  1. observe · 只记录数值摘要\n  2. require · 违规时拒绝训练步骤\n  3. penalty · 显式加权范围惩罚（仅预览，训练需 --apply-supervision）\n"
    );
    let modes = ["observe", "require", "penalty"];
    let mode_index = match guide_choice("规则模式序号", &modes) {
        Ok(index) => index,
        Err(report) => return report,
    };
    let mode = modes[mode_index];
    let kind = if mode == "penalty" {
        "range"
    } else {
        eprintln!(
            "规则类型：1 finite（必须有限） · 2 range（数值范围） · 3 batch-alignment（模型端口与人工选择的批次目标维度对齐）\n"
        );
        match guide_choice("规则类型序号", &["finite", "range", "batch-alignment"]) {
            Ok(0) => "finite",
            Ok(1) => "range",
            Ok(_) => "batch-alignment",
            Err(report) => return report,
        }
    };
    let (minimum, maximum) = if kind == "range" {
        let minimum = match guide_optional_number("范围下界（可留空）") {
            Ok(value) => value,
            Err(report) => return report,
        };
        let maximum = match guide_optional_number("范围上界（可留空）") {
            Ok(value) => value,
            Err(report) => return report,
        };
        if minimum.is_none() && maximum.is_none() {
            return Report::error(
                "semantic_supervision_guide",
                "SUPERVISION_RULE_VALUE_INVALID",
                "range 规则至少需要一个边界。",
                "重新运行引导并填写下界或上界。",
            );
        }
        if minimum
            .zip(maximum)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return Report::error(
                "semantic_supervision_guide",
                "SUPERVISION_RULE_VALUE_INVALID",
                "范围下界大于上界。",
                "重新运行引导并调整边界。",
            );
        }
        (minimum, maximum)
    } else {
        (None, None)
    };
    let (target_path, axis) = if kind == "batch-alignment" {
        let target_path = match guide_required(
            "批次目标 Tensor 路径（如 batch.args[1] 或 batch.kwargs.labels）",
            None,
        ) {
            Ok(value) => value,
            Err(report) => return report,
        };
        let axis = match guide_required("要对齐的维度序号（通常是 0，批次维）", Some("0"))
        {
            Ok(value) => match value.parse::<u32>() {
                Ok(axis) => axis,
                _ => {
                    return Report::error(
                        "semantic_supervision_guide",
                        "SUPERVISION_RULE_VALUE_INVALID",
                        "对齐维度必须是非负整数，且须在实际张量维数范围内。",
                        "重新运行引导并填写有效维度。",
                    );
                }
            },
            Err(report) => return report,
        };
        (Some(target_path), Some(axis))
    } else {
        (None, None)
    };
    let weight = if mode == "penalty" {
        match guide_optional_number("Penalty 权重（必须大于 0）") {
            Ok(Some(weight)) if weight > 0.0 => Some(weight),
            Ok(_) => {
                return Report::error(
                    "semantic_supervision_guide",
                    "SUPERVISION_RULE_VALUE_INVALID",
                    "penalty 权重必须是大于 0 的有限数字。",
                    "重新运行引导并提供有效权重。",
                );
            }
            Err(report) => return report,
        }
    } else {
        None
    };
    if mode == "penalty" {
        eprintln!(
            "确认：该规则只保存并预览；它不会在此刻训练或改动目标。正式训练只有显式传 --apply-supervision 才会把 penalty 加入 Loss。\n"
        );
    }
    let rule_report = execute_ui(
        app,
        cli,
        &Command::Model {
            command: ModelCommand::Supervise {
                command: SuperviseCommand::Rule {
                    command: SupervisionRuleCommand::Add {
                        if_state_revision: None,
                        mark: mark_id.clone(),
                        kind: match kind {
                            "finite" => SupervisionRuleKind::Finite,
                            "range" => SupervisionRuleKind::Range,
                            _ => SupervisionRuleKind::BatchAlignment,
                        },
                        mode: match mode {
                            "observe" => SupervisionRuleMode::Observe,
                            "require" => SupervisionRuleMode::Require,
                            _ => SupervisionRuleMode::Penalty,
                        },
                        min: minimum,
                        max: maximum,
                        weight,
                        target_path,
                        axis,
                        id: None,
                        plugin: None,
                        plugin_source: None,
                        plugin_params: Vec::new(),
                    },
                },
            },
        },
    );
    if !rule_report.success() {
        return rule_report;
    }

    eprintln!("\n第 4 步 · 对当前代表批次做只读预览；不会反向传播、更新参数或改变 Loss。\n");
    let preview = execute_ui(
        app,
        cli,
        &Command::Model {
            command: ModelCommand::Supervise {
                command: SuperviseCommand::Preview {
                    inputs: inputs.clone(),
                    model_device: None,
                },
            },
        },
    );
    if !preview.success() {
        return preview;
    }
    eprintln!(
        "覆盖状态：{} · {}",
        preview.data["status"].as_str().unwrap_or("unknown"),
        preview.data["coverage"]
            .as_str()
            .unwrap_or("未提供覆盖说明")
    );
    if let Some(results) = preview.data["results"].as_array() {
        for result in results {
            let rule_id = result["rule_id"].as_str().unwrap_or("unknown");
            let status = result["status"].as_str().unwrap_or("unknown");
            if let Some(elements) = result["element_count"].as_u64() {
                eprintln!(
                    "  规则 {rule_id} · {status} · 违反 {} / 覆盖 {elements}",
                    result["violation_count"].as_u64().unwrap_or(0),
                );
            } else {
                eprintln!("  规则 {rule_id} · {status} · 本次没有可测量值");
            }
            if let Some(penalty) = result["observations"][0]["weighted_penalty_preview"].as_f64() {
                eprintln!("    加权 penalty 预览值：{penalty:.6}（尚未应用）");
            }
            if result["kind"] == "batch-alignment"
                && let Some(observation) = result["observations"]
                    .as_array()
                    .and_then(|items| items.first())
            {
                eprintln!(
                    "    目标 {} · 模型形状 {} · 目标形状 {} · 对齐维度 {}",
                    result["expected"]["target_path"]
                        .as_str()
                        .unwrap_or("unknown"),
                    observation["shape"],
                    observation["target_shape"],
                    observation["axis"]
                );
            }
        }
    }
    eprintln!(
        "\n这些规则不会自动进入训练。后续试跑或训练只有显式添加 --apply-supervision 才会执行；{}\n",
        if mode == "penalty" {
            "penalty 会作为加权项加入 Loss。"
        } else {
            "require 会在 tracked update 前拒绝违规步骤。"
        }
    );

    let mut result = Report::ok(
        "semantic_supervision_guide",
        json!({
            "mark":mark_report.data["mark"],
            "rule":rule_report.data["rule"],
            "preview":preview.data,
            "training_activation_requires_explicit_flag":true,
            "penalty_changes_objective":mode == "penalty",
            "scope":"当前代表批次与本次执行分支；未观察分支和未绑定样本目标不作推断"
        }),
    );
    if matches!(
        preview.data["status"].as_str(),
        Some("stale" | "incomplete")
    ) {
        result.status = "inconclusive".into();
    }
    result
}

fn run_preview_interaction<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    mut command: Command,
) -> Report {
    loop {
        let report = execute_ui(app, cli, &command);
        if report.success()
            || report
                .diagnostics
                .first()
                .is_some_and(|diagnostic| diagnostic.code == "USER_INTERRUPTED")
        {
            return report;
        }

        let diagnostic = report.diagnostics.first();
        eprintln!(
            "预览失败：{}{}",
            diagnostic
                .map(|diagnostic| diagnostic.message.as_str())
                .unwrap_or("操作未成功"),
            diagnostic
                .and_then(|diagnostic| diagnostic.hint.as_deref())
                .filter(|hint| !hint.is_empty())
                .map_or_else(String::new, |hint| format!("\n修复提示：{hint}"))
        );
        let action = loop {
            let Some(action) =
                prompt_allow_empty("1 修改设置并重试 · 2 原样重试 · back 返回并查看诊断")
            else {
                return report;
            };
            let action = action.trim().to_ascii_lowercase();
            if matches!(
                action.as_str(),
                "1" | "edit" | "修改" | "2" | "retry" | "重试" | "back" | "b" | "返回" | "q"
            ) {
                break action;
            }
            eprintln!("请输入 1、2 或 back。\n");
        };
        match action.as_str() {
            "1" | "edit" | "修改" => match prompt_preview_command(&command) {
                Ok(updated) => command = updated,
                Err(cancelled)
                    if cancelled
                        .diagnostics
                        .first()
                        .is_some_and(|diagnostic| diagnostic.code == "USER_INTERRUPTED") =>
                {
                    return report;
                }
                Err(error) => return error,
            },
            "2" | "retry" | "重试" => {}
            "back" | "b" | "返回" | "q" => return report,
            _ => unreachable!("preview retry action was validated"),
        }
    }
}

fn prompt_layer_kind() -> Option<(Option<String>, Option<String>)> {
    println!("可输入所选 Python 环境中 torch.nn 的公开 Module 类名，例如 Linear 或 Conv2d；实际可用性由运行时验证。");
    let kind = prompt("Layer 类型，或输入 custom 使用本地 Python 工厂")?;
    if kind == "custom" {
        let component =
            prompt("组件 ID（namespace/name[@version]）或 Python 工厂入口（module:callable）")?;
        Some((None, Some(component)))
    } else {
        (!kind.is_empty()).then_some((Some(kind), None))
    }
}

fn prompt_layer_parameters<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    component: &mut Option<String>,
    layer_type: Option<&str>,
) -> Option<serde_json::Map<String, Value>> {
    if component.is_none()
        && let Some(layer_type) = layer_type
    {
        if let Some(schema) = builtin_layer_schema(layer_type) {
            return prompt_schema_object(
                &schema,
                &format!("layer.new.{layer_type}"),
                None,
                FormMode::Full,
            );
        }
        println!("此类型没有内置参数表单，请按名称逐项输入 JSON 参数；构造时由所选 Python 环境校验。");
        return prompt_parameter_map(&format!("layer.new.{layer_type}"));
    }
    prompt_registered_component_parameters(app, cli, component, "layer")
}

fn builtin_layer_schema(layer_type: &str) -> Option<Value> {
    let schema = match layer_type {
        "Linear" => serde_json::json!({
            "type":"object", "required":["in_features","out_features"],
            "properties":{
                "in_features":{"type":"integer","minimum":1,"description":"输入特征数"},
                "out_features":{"type":"integer","minimum":1,"description":"输出特征数"},
                "bias":{"type":"boolean","default":true,"description":"是否包含偏置项"}
            }
        }),
        "Dropout" => serde_json::json!({
            "type":"object", "properties":{
                "p":{"type":"number","minimum":0.0,"maximum":1.0,"default":0.5,"description":"每个元素被置零的概率"},
                "inplace":{"type":"boolean","default":false,"description":"是否原位修改输入"}
            }
        }),
        "ReLU" => serde_json::json!({
            "type":"object", "properties":{
                "inplace":{"type":"boolean","default":false,"description":"是否原位修改输入"}
            }
        }),
        "GELU" => serde_json::json!({
            "type":"object", "properties":{
                "approximate":{"type":"string","enum":["none","tanh"],"default":"none","description":"GELU 近似算法"}
            }
        }),
        "LayerNorm" => serde_json::json!({
            "type":"object", "required":["normalized_shape"],
            "properties":{
                "normalized_shape":{"type":"array","minItems":1,"items":{"type":"integer","minimum":1},"description":"需要归一化的尾部维度；单维也写成 [维度]"},
                "eps":{"type":"number","minimum":0.0,"default":0.00001,"description":"数值稳定项"},
                "elementwise_affine":{"type":"boolean","default":true,"description":"是否学习逐元素仿射参数"}
            }
        }),
        "Conv2d" => serde_json::json!({
            "type":"object", "required":["in_channels","out_channels","kernel_size"],
            "properties":{
                "in_channels":{"type":"integer","minimum":1,"description":"输入通道数"},
                "out_channels":{"type":"integer","minimum":1,"description":"输出通道数"},
                "kernel_size":{"type":"integer","minimum":1,"description":"卷积核尺寸；该交互表单此处使用正整数"},
                "stride":{"type":"integer","minimum":1,"default":1,"description":"卷积步长"},
                "padding":{"type":"integer","minimum":0,"default":0,"description":"零填充大小"},
                "dilation":{"type":"integer","minimum":1,"default":1,"description":"膨胀系数"},
                "groups":{"type":"integer","minimum":1,"default":1,"description":"分组数"},
                "bias":{"type":"boolean","default":true,"description":"是否包含偏置项"}
            }
        }),
        "Identity" => serde_json::json!({"type":"object","properties":{}}),
        _ => return None,
    };
    Some(schema)
}

fn prompt_layer_edit_provider_parameters<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    component: &mut Option<String>,
) -> Option<serde_json::Map<String, Value>> {
    prompt_registered_component_parameters(app, cli, component, "layer-edit-provider")
}

fn prompt_registered_component_parameters<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    component: &mut Option<String>,
    expected_kind: &str,
) -> Option<serde_json::Map<String, Value>> {
    let Some(requested) = component.as_deref() else {
        return prompt_parameter_map(&format!("component_parameters.{expected_kind}.unbound"));
    };
    if requested.contains(':') {
        println!("此 Python 入口没有注册参数 Schema；请输入它接受的关键字参数。");
        return prompt_parameter_map(&format!("component_parameters.{expected_kind}.{requested}"));
    }

    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(root) => root,
        Err(error) => {
            eprintln!("无法读取当前项目目录：{error}");
            return None;
        }
    };
    let descriptor = match ApplicationUiServices::new(app, cli).inspect_component(&root, requested)
    {
        Ok(descriptor) => descriptor,
        Err(error) => {
            eprintln!("无法读取已注册组件：{error}");
            return None;
        }
    };
    if descriptor["manifest"]["kind"] != expected_kind {
        eprintln!(
            "组件 {} 的类型是 {}，需要 {expected_kind} 组件。",
            descriptor["manifest"]["component_id"]
                .as_str()
                .unwrap_or(requested),
            descriptor["manifest"]["kind"].as_str().unwrap_or("未知")
        );
        return None;
    }
    let manifest = &descriptor["manifest"];
    let id = manifest["component_id"].as_str().unwrap_or(requested);
    let version = manifest["component_version"].as_str().unwrap_or("0.0.0");
    let selected = format!("{id}@{version}");
    println!(
        "{selected} · {}\n按组件 Schema 填写参数；默认值和合法范围会在这里提示。",
        manifest["description"].as_str().unwrap_or("")
    );
    let payload = Value::Object(prompt_schema_object(
        &manifest["params_schema"],
        &format!("{expected_kind}.{selected}"),
        None,
        FormMode::Full,
    )?);
    let Some(values) = payload.as_object() else {
        eprintln!("组件参数 Schema 顶层必须是 object。");
        return None;
    };
    let services = ApplicationUiServices::new(app, cli);
    let resolved_result = match expected_kind {
        "layer" => services.resolve_layer_component(&root, &selected, values),
        "layer-edit-provider" => services.resolve_layer_edit_provider(&root, &selected, values),
        _ => unreachable!("only registered component parameter forms are supported"),
    };
    let resolved = match resolved_result {
        Ok(resolved) => resolved,
        Err(error) => {
            eprintln!("组件参数校验失败：{error}");
            return None;
        }
    };
    let resolved_key = if expected_kind == "layer-edit-provider" {
        "resolved_payload"
    } else {
        "resolved_params"
    };
    let params = resolved[resolved_key].as_object()?;
    *component = Some(selected);
    Some(params.clone())
}

fn execute_ui<B: Backend>(app: &Application<B>, cli: &Cli, command: &Command) -> Report {
    ApplicationUiServices::new(app, cli).execute(command)
}

fn choose_run(runs: &[Value], default_index: usize, label: &str) -> Option<String> {
    loop {
        let input = prompt_allow_empty(label)?;
        let input = input.trim();
        if input.is_empty() {
            return runs
                .get(default_index)
                .and_then(|run| run["id"].as_str())
                .map(str::to_owned);
        }
        if matches!(input.to_ascii_lowercase().as_str(), "back" | "b" | "返回") {
            return None;
        }
        let selected = input
            .parse::<usize>()
            .ok()
            .and_then(|index| index.checked_sub(1))
            .and_then(|index| runs.get(index))
            .or_else(|| runs.iter().find(|run| run["id"].as_str() == Some(input)));
        if let Some(id) = selected.and_then(|run| run["id"].as_str()) {
            return Some(id.to_owned());
        }
        println!("请输入列表序号或完整运行 ID；输入 back 返回。");
    }
}

fn repeat_trial_command(run_id: String) -> Command {
    Command::Try {
        inputs: TrainingInputs::default(),
        repeat: Some(run_id),
        name: None,
        training_mode: None,
        step: None,
        training_loop: None,
        steps: None,
        lr: None,
        weight_decay: None,
        apply_supervision: false,
        no_supervision: false,
        params: Vec::new(),
        loop_params: Vec::new(),
    }
}

fn training_source_id(command: &Command) -> Option<&str> {
    match command {
        Command::Try { repeat, .. } => repeat.as_deref(),
        Command::Train { resume, .. } => resume.as_deref(),
        _ => None,
    }
}

fn confirm_training_plan<B: Backend>(app: &Application<B>, cli: &Cli, command: &Command) -> bool {
    let source_record = if let Some(id) = training_source_id(command) {
        let report = execute_ui(
            app,
            cli,
            &Command::Runs {
                command: RunsCommand::Show {
                    id: Some(id.to_owned()),
                },
            },
        );
        if !report.success() {
            display(&report, false);
            return false;
        }
        Some(report.data)
    } else {
        None
    };
    let readiness = match app.plan_operation_inputs(cli, command) {
        Ok(plan) => plan.and_then(|plan| serde_json::to_value(plan).ok()),
        Err(report) => {
            display(&report, false);
            return false;
        }
    };
    println!(
        "\n{}",
        render_run_plan(command, cli, readiness.as_ref(), source_record.as_ref()).join("\n")
    );
    let state = readiness
        .as_ref()
        .and_then(|plan| plan["readiness"]["state"].as_str())
        .unwrap_or("unknown");
    if matches!(state, "needs_input" | "unsupported") {
        println!("当前计划有未满足的必需输入或不支持项；没有启动任务。返回表单修正后重试。\n");
        return false;
    }
    if state != "ready" {
        println!("当前计划仍有未确认的必需能力或条件，无法安全启动。请按上方原因检查运行环境或补齐条件，再重新准备计划。\n");
        return false;
    }
    let confirmation = "确认按以上来源、预算和覆盖启动任务？输入 start 开始，其他输入取消";
    if prompt_allow_empty(confirmation).is_some_and(|answer| {
        matches!(
            answer.trim().to_ascii_lowercase().as_str(),
            "start" | "开始"
        )
    }) {
        let current = match app.plan_operation_inputs(cli, command) {
            Ok(plan) => plan.and_then(|plan| serde_json::to_value(plan).ok()),
            Err(report) => {
                display(&report, false);
                return false;
            }
        };
        if current != readiness
            || current.as_ref().and_then(|plan| plan["readiness"]["state"].as_str()) != Some("ready")
        {
            println!("确认期间计划或能力状态已变化，或无法重新核对；任务未启动。请重新准备并审核。\n");
            return false;
        }
        true
    } else {
        println!("已取消；没有启动训练或写入运行记录。\n");
        false
    }
}

fn run_training_with_plan<B: Backend>(app: &Application<B>, cli: &Cli, command: &Command) {
    if confirm_training_plan(app, cli, command) {
        let report = run_command(app, cli, command);
        display(&report, false);
        let failure_code = report.data["cause_code"]
            .as_str()
            .or_else(|| report.data["failure_code"].as_str())
            .or_else(|| {
                report
                    .diagnostics
                    .iter()
                    .find(|diagnostic| {
                        matches!(
                            diagnostic.code.as_str(),
                            "CHECKPOINT_WRITE_FAILED" | "CHECKPOINT_STORE_FAILED"
                        )
                    })
                    .map(|diagnostic| diagnostic.code.as_str())
            });
        if let Some(code @ ("CHECKPOINT_WRITE_FAILED" | "CHECKPOINT_STORE_FAILED")) = failure_code {
            recover_checkpoint_storage(app, cli, code);
        }
    }
}

fn recover_checkpoint_storage<B: Backend>(app: &Application<B>, cli: &Cli, cause_code: &str) {
    println!("Checkpoint 根因：{cause_code}。旧 checkpoint 和源码不会被此恢复流程删除。\n");
    loop {
        let Some(action) = prompt_allow_empty(
            "Checkpoint 保存失败：storage 查看运行占用 · clean 预览并清理已结束运行的日志/进度 · back 返回",
        ) else {
            return;
        };
        match action.trim().to_ascii_lowercase().as_str() {
            "back" | "b" | "返回" | "q" | "" => return,
            "storage" | "占用" | "1" => {
                let report = execute_ui(
                    app,
                    cli,
                    &Command::Runs {
                        command: RunsCommand::Storage { id: None },
                    },
                );
                if report.success() {
                    println!("{}", render_storage(&report.data).join("\n"));
                } else {
                    display(&report, false);
                }
            }
            "clean" | "清理" | "2" => {
                let plan = execute_ui(
                    app,
                    cli,
                    &Command::Runs {
                        command: RunsCommand::Clean {
                            run: None,
                            include_logs: true,
                            include_progress: true,
                            include_checkpoints: false,
                            artifact: None,
                            dry_run: true,
                            apply: None,
                            release_checkpoints: false,
                        },
                    },
                );
                if !plan.success() {
                    display(&plan, false);
                    continue;
                }
                println!("{}", render_clean_plan(&plan.data).join("\n"));
                if plan.data["items"].as_array().is_none_or(Vec::is_empty) {
                    continue;
                }
                let Some(plan_id) = plan.data["id"].as_str() else {
                    println!("清理服务未返回固定计划 ID；未改动任何文件。\n");
                    continue;
                };
                if prompt("此安全清理仅包含已结束运行的日志/进度，不含 checkpoint。输入 CLEAN 确认，其他输入取消").is_none_or(|answer| answer.trim() != "CLEAN")
                {
                    println!("已取消；文件未改变。\n");
                    continue;
                }
                let result = execute_ui(
                    app,
                    cli,
                    &Command::Runs {
                        command: RunsCommand::Clean {
                            run: None,
                            include_logs: false,
                            include_progress: false,
                            include_checkpoints: false,
                            artifact: None,
                            dry_run: false,
                            apply: Some(plan_id.to_owned()),
                            release_checkpoints: false,
                        },
                    },
                );
                if result.success() {
                    println!("{}", render_clean_result(&result.data).join("\n"));
                } else {
                    display(&result, false);
                }
                println!(
                    "清理不会自动重启训练；确认保存路径和磁盘空间后，再从训练页面重新启动。\n"
                );
            }
            _ => println!("请输入 storage、clean 或 back。\n"),
        }
    }
}

fn choose_resume_source<B: Backend>(app: &Application<B>, cli: &Cli) -> Option<(String, Value)> {
    let mut runs = fetch_run_list(app, cli)?;
    runs.retain(|run| run["operation"] == "train");
    if runs.is_empty() {
        println!("\n{}\n", render_resume_candidates(&runs));
        return None;
    }
    loop {
        println!("\n{}", render_resume_candidates(&runs));
        let input = prompt_allow_empty("选择要续训的正式运行；back 返回")?;
        let input = input.trim();
        if matches!(
            input.to_ascii_lowercase().as_str(),
            "back" | "b" | "返回" | "q"
        ) {
            return None;
        }
        let selected = input
            .parse::<usize>()
            .ok()
            .and_then(|index| index.checked_sub(1))
            .and_then(|index| runs.get(index))
            .or_else(|| runs.iter().find(|run| run["id"].as_str() == Some(input)));
        let Some(selected) = selected else {
            println!("请输入列表序号或完整运行 ID。\n");
            continue;
        };
        if !resume_available(selected) {
            println!("{}\n", unavailable_reason(selected));
            continue;
        }
        let Some(id) = selected["id"].as_str().map(str::to_owned) else {
            println!("此记录缺少运行 ID，不能续训。\n");
            continue;
        };
        let report = execute_ui(
            app,
            cli,
            &Command::Runs {
                command: RunsCommand::Show {
                    id: Some(id.clone()),
                },
            },
        );
        if !report.success() {
            display(&report, false);
            continue;
        }
        println!("\n{}", render_resume_selection(&report.data).join("\n"));
        return Some((id, report.data));
    }
}

fn run_priority(run: &Value) -> u8 {
    match run["status"].as_str().unwrap_or("unknown") {
        "running" | "stopping" => 0,
        "failed" | "interrupted" | "unknown" => 1,
        _ => 2,
    }
}

fn sort_run_records(runs: &mut [Value]) {
    runs.sort_by(|left, right| {
        run_priority(left).cmp(&run_priority(right)).then_with(|| {
            right["started_at_ms"]
                .as_u64()
                .unwrap_or(0)
                .cmp(&left["started_at_ms"].as_u64().unwrap_or(0))
        })
    });
}

fn fetch_run_list<B: Backend>(app: &Application<B>, cli: &Cli) -> Option<Vec<Value>> {
    let report = execute_ui(
        app,
        cli,
        &Command::Runs {
            command: RunsCommand::List {
                query: None,
                status: None,
                kept: None,
            },
        },
    );
    if !report.success() {
        display(&report, false);
        return None;
    }
    let mut runs = report.data["runs"].as_array().cloned().unwrap_or_default();
    sort_run_records(&mut runs);
    Some(runs)
}

fn browse_run_progress<B: Backend>(app: &Application<B>, cli: &Cli, run_id: &str) {
    loop {
        let report = execute_ui(
            app,
            cli,
            &Command::Runs {
                command: RunsCommand::Attach {
                    id: Some(run_id.to_owned()),
                    follow: Some(false),
                },
            },
        );
        if !report.success() {
            display(&report, false);
            return;
        }
        println!("\n{}", render_run_progress(&report.data).join("\n"));
        let active = matches!(
            report.data["record"]["status"].as_str(),
            Some("running" | "stopping")
        );
        let prompt_text = if active {
            "Enter 刷新 · stop 请求协作停止 · back 返回（离开不影响运行）"
        } else {
            "Enter 刷新 · back 返回（该运行当前不是已确认活动状态）"
        };
        let Some(input) = prompt_allow_empty(prompt_text) else {
            return;
        };
        match input.trim().to_ascii_lowercase().as_str() {
            "" | "r" | "refresh" | "刷新" => continue,
            "back" | "b" | "返回" | "q" => return,
            "stop" | "停止" => {
                if !active {
                    println!("该运行状态为未知或已结束；没有发送停止请求。\n");
                    continue;
                }
                if !prompt("向此运行请求在安全检查点停止？[y/N]").is_some_and(|answer| {
                    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
                }) {
                    println!("已取消停止请求。\n");
                    continue;
                }
                let stopped = execute_ui(
                    app,
                    cli,
                    &Command::Runs {
                        command: RunsCommand::Stop {
                            id: Some(run_id.to_owned()),
                            force: false,
                        },
                    },
                );
                display(&stopped, false);
            }
            _ => println!("输入 refresh、stop 或 back。"),
        }
    }
}

fn browse_run_detail<B: Backend>(app: &Application<B>, cli: &Cli, run_id: &str) {
    loop {
        let report = execute_ui(
            app,
            cli,
            &Command::Runs {
                command: RunsCommand::Show {
                    id: Some(run_id.to_owned()),
                },
            },
        );
        if !report.success() {
            display(&report, false);
            return;
        }
        println!("\n{}", render_run_detail(&report.data).join("\n"));
        let operation = report.data["record"]["operation"]
            .as_str()
            .unwrap_or_default();
        if operation == "evaluate"
            && let Some(stored) = report.data["report"].as_object()
            && let Ok(stored_report) =
                serde_json::from_value::<Report>(Value::Object(stored.clone()))
        {
            println!(
                "{}",
                crate::presentation::pages::evaluation::render(&stored_report).join("\n")
            );
        }
        let can_repeat = operation == "training_probe";
        let status = report.data["record"]["status"].as_str().unwrap_or_default();
        let can_evaluate =
            operation == "train" && status == "completed" && report.data["checkpoint"].is_object();
        let can_browse_samples = operation == "evaluate"
            && report.data["report"]["data"]["evaluation"]["sample_previews"]["samples"]
                .as_array()
                .is_some_and(|samples| !samples.is_empty());
        let mut actions = vec!["progress 查看进度", "stop 停止"];
        if can_repeat {
            actions.push("repeat 按原设置重试");
        }
        if can_evaluate {
            actions.push("evaluate 配置并运行评估");
        }
        if can_browse_samples {
            actions.push("samples 浏览已报告的样本预览");
        }
        actions.extend(["refresh 刷新", "back 返回列表"]);
        let prompt_text = actions.join(" · ");
        let Some(input) = prompt_allow_empty(&prompt_text) else {
            return;
        };
        match input.trim().to_ascii_lowercase().as_str() {
            "back" | "b" | "返回" | "q" => return,
            "" | "refresh" | "r" | "刷新" => continue,
            "progress" | "p" | "进度" => browse_run_progress(app, cli, run_id),
            "stop" | "停止" => {
                let status = report.data["record"]["status"]
                    .as_str()
                    .unwrap_or("unknown");
                if !matches!(status, "running" | "stopping") {
                    println!("当前状态为 {status}，不能请求停止。");
                    continue;
                }
                if !prompt("请求该运行在安全检查点停止？[y/N]").is_some_and(|answer| {
                    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
                }) {
                    println!("已取消停止请求。\n");
                    continue;
                }
                let stopped = execute_ui(
                    app,
                    cli,
                    &Command::Runs {
                        command: RunsCommand::Stop {
                            id: Some(run_id.to_owned()),
                            force: false,
                        },
                    },
                );
                display(&stopped, false);
            }
            "repeat" | "重试" if can_repeat => {
                if prompt("使用此运行保存的原始输入和预算重新试跑？[y/N]").is_some_and(|answer| {
                    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
                }) {
                    run_training_with_plan(app, cli, &repeat_trial_command(run_id.to_owned()));
                }
            }
            "evaluate" | "评估" if can_evaluate => {
                if let Some(command) = prompt_evaluation_command(run_id.to_owned()) {
                    let evaluated = execute_ui(app, cli, &command);
                    display(&evaluated, false);
                    if evaluated.success() {
                        browse_evaluation_previews(
                            &evaluated.data["evaluation"]["sample_previews"],
                        );
                    }
                }
            }
            "samples" | "sample" | "样本" if can_browse_samples => {
                let previews = &report.data["report"]["data"]["evaluation"]["sample_previews"];
                if previews.is_null() {
                    println!("此评估记录没有结构化样本预览。\n");
                } else {
                    browse_evaluation_previews(previews);
                }
            }
            _ => println!("没有这个详情操作。"),
        }
    }
}

fn prompt_evaluation_command(run_id: String) -> Option<Command> {
    let evaluator = prompt_allow_empty(
        "评估入口 module:evaluate（model, data, context -> metrics；back 取消）",
    )?;
    if evaluator.trim().is_empty() || evaluator.trim().eq_ignore_ascii_case("back") {
        return None;
    }
    let data = prompt_allow_empty("验证/测试数据工厂 module:build_data")?;
    if data.trim().is_empty() || data.trim().eq_ignore_ascii_case("back") {
        return None;
    }
    let checkpoint = prompt_allow_empty("Checkpoint [latest]")?;
    let split = prompt_allow_empty("数据切分 [test]")?;
    let samples = prompt_allow_empty("最多评估样本数（留空表示由评估入口决定）")?;
    let preview_samples = prompt_allow_empty("最多保存多少条样本预览（1–32；留空跳过）")?;
    let data_fingerprint = prompt_allow_empty("数据版本指纹（可选；留空跳过）")?;
    let samples = parse_optional_positive(&samples)?;
    let preview_samples = parse_optional_positive(&preview_samples)?;
    if preview_samples.is_some_and(|count| count > 32) {
        println!("预览条数必须在 1–32 之间。\n");
        return None;
    }
    if let (Some(previews), Some(limit)) = (preview_samples, samples)
        && previews > limit
    {
        println!("预览条数不能超过评估样本上限。\n");
        return None;
    }
    let params = prompt_params("evaluate")?;
    let data_params = prompt_params("evaluate.data")?;
    println!(
        "\n评估计划\n  来源运行：{run_id}\n  评估入口：{}\n  数据工厂：{}\n  数据切分：{}\n  样本上限：{}\n  预览上限：{}\n  数据指纹：{}",
        evaluator.trim(),
        data.trim(),
        nonempty_or(split.clone(), "test"),
        samples.map_or_else(|| "由评估入口决定".into(), |count| count.to_string()),
        preview_samples.map_or_else(|| "不请求".into(), |count| count.to_string()),
        if data_fingerprint.trim().is_empty() {
            "未提供"
        } else {
            "已提供"
        }
    );
    if !prompt("开始评估？[y/N]")
        .is_some_and(|answer| matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes"))
    {
        println!("已取消；没有启动评估。\n");
        return None;
    }
    Some(Command::Evaluate {
        run: run_id,
        checkpoint: nonempty_or(checkpoint, "latest"),
        if_checkpoint_sha256: None,
        split: nonempty_or(split, "test"),
        evaluator: evaluator.trim().to_owned(),
        device: None,
        data: data.trim().to_owned(),
        samples,
        preview_samples,
        data_fingerprint: nonempty_option(data_fingerprint),
        params,
        data_params,
    })
}

fn parse_optional_positive(value: &str) -> Option<Option<u64>> {
    if value.trim().is_empty() {
        Some(None)
    } else {
        match value.trim().parse::<u64>() {
            Ok(value) if value > 0 => Some(Some(value)),
            _ => {
                println!("请输入正整数，或留空跳过。\n");
                None
            }
        }
    }
}

fn nonempty_or(value: String, fallback: &str) -> String {
    if value.trim().is_empty() {
        fallback.to_owned()
    } else {
        value.trim().to_owned()
    }
}

fn nonempty_option(value: String) -> Option<String> {
    (!value.trim().is_empty()).then(|| value.trim().to_owned())
}

fn browse_evaluation_previews(previews: &Value) {
    let samples = previews["samples"].as_array().cloned().unwrap_or_default();
    if samples.is_empty() {
        println!("评估器没有报告可浏览的样本预览。\n");
        return;
    }
    let reported = previews["reported"]
        .as_u64()
        .unwrap_or(samples.len() as u64);
    println!("评估器明确报告了 {reported} 条预览记录。只展示这些记录，不推断其他产物。\n");
    let page_size = 4usize;
    let mut page = 0usize;
    loop {
        let start = page * page_size;
        let end = (start + page_size).min(samples.len());
        println!("样本预览 {}–{} / {}", start + 1, end, samples.len());
        for (index, sample) in samples[start..end].iter().enumerate() {
            let id = sample["sample_id"].as_str().unwrap_or("(无 ID)");
            println!(
                "  {}. {}",
                start + index + 1,
                id.chars()
                    .map(|c| if c.is_control() { '�' } else { c })
                    .collect::<String>()
            );
        }
        let Some(input) = prompt_allow_empty("序号查看完整 JSON · n/p 翻页 · Enter/back 返回")
        else {
            return;
        };
        let input = input.trim();
        if input.is_empty()
            || matches!(
                input.to_ascii_lowercase().as_str(),
                "back" | "b" | "返回" | "q"
            )
        {
            return;
        }
        if input.eq_ignore_ascii_case("n") {
            if end < samples.len() {
                page += 1;
            }
            continue;
        }
        if input.eq_ignore_ascii_case("p") {
            page = page.saturating_sub(1);
            continue;
        }
        if let Ok(number) = input.parse::<usize>() {
            if let Some(sample) = number.checked_sub(1).and_then(|index| samples.get(index)) {
                println!(
                    "{}\n",
                    serde_json::to_string_pretty(sample)
                        .unwrap_or_else(|_| "无法格式化样本记录。".into())
                );
            } else {
                println!("请输入当前预览中的有效序号。\n");
            }
        } else {
            println!("请输入序号、n、p 或 back。\n");
        }
    }
}

fn browse_runs_interactive<B: Backend>(app: &Application<B>, cli: &Cli) {
    let Some(mut runs) = fetch_run_list(app, cli) else {
        return;
    };
    let mut filter = RunsFilter::default();
    loop {
        println!("\n{}", render_runs(&runs, &filter));
        let Some(input) = prompt_allow_empty(
            "序号/运行 ID 查看详情 · /词搜索 · status <状态|all> · kept <yes|no|all> · n/p 翻页 · refresh · clear · back",
        ) else {
            return;
        };
        let input = input.trim();
        match input.to_ascii_lowercase().as_str() {
            "back" | "b" | "返回" | "q" => return,
            "n" | "next" => {
                let count = filter.filtered(&runs).len();
                filter.next(count);
            }
            "p" | "prev" => filter.previous(),
            "clear" | "清除筛选" => filter = RunsFilter::default(),
            "refresh" | "r" | "刷新" => {
                if let Some(refreshed) = fetch_run_list(app, cli) {
                    runs = refreshed;
                    filter.page = filter
                        .page
                        .min(filter.pages(filter.filtered(&runs).len()) - 1);
                }
            }
            "" => continue,
            _ if input.starts_with('/') => {
                filter.query = input[1..].trim().to_owned();
                filter.page = 0;
            }
            _ if input.to_ascii_lowercase().starts_with("search ") => {
                filter.query = input[7..].trim().to_owned();
                filter.page = 0;
            }
            _ if input.to_ascii_lowercase().starts_with("status ") => {
                let value = input[7..].trim().to_ascii_lowercase();
                filter.status = if matches!(value.as_str(), "" | "all" | "全部") {
                    None
                } else if matches!(value.as_str(), "active" | "running" | "stopping" | "活动") {
                    Some("active".into())
                } else if matches!(value.as_str(), "failed" | "失败") {
                    Some("failed".into())
                } else if matches!(value.as_str(), "completed" | "完成") {
                    Some("completed".into())
                } else if matches!(value.as_str(), "interrupted" | "中断") {
                    Some("interrupted".into())
                } else if matches!(value.as_str(), "unknown" | "未知") {
                    Some("unknown".into())
                } else {
                    println!("状态可选：active、failed、completed、interrupted、unknown、all。\n");
                    continue;
                };
                filter.page = 0;
            }
            _ if input.to_ascii_lowercase().starts_with("kept ") => {
                let value = input[5..].trim().to_ascii_lowercase();
                filter.kept = match value.as_str() {
                    "yes" | "true" | "是" => Some(true),
                    "no" | "false" | "否" => Some(false),
                    "all" | "全部" => None,
                    _ => {
                        println!("请输入 kept yes、kept no 或 kept all。\n");
                        continue;
                    }
                };
                filter.page = 0;
            }
            _ => {
                let filtered = filter.filtered(&runs);
                if let Some(selected) = filter.select(input, &filtered) {
                    if let Some(id) = selected["id"].as_str() {
                        browse_run_detail(app, cli, id);
                        if let Some(refreshed) = fetch_run_list(app, cli) {
                            runs = refreshed;
                            filter.page = filter
                                .page
                                .min(filter.pages(filter.filtered(&runs).len()) - 1);
                        }
                    }
                } else {
                    println!("没有匹配的运行。可输入当前页序号、完整 ID 或筛选命令。\n");
                }
            }
        }
    }
}

fn background_preview_summary(preview: &mut BackgroundPreview) -> String {
    let project = preview
        .project_root()
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "当前项目".into());
    match preview.try_wait() {
        Ok(Some(status)) => format!(
            "项目 {project} 的后台数据预览已结束（{}，用时 {} 秒）；进入“数据预览、产物与清理”查看结果。",
            if status.success() {
                "成功"
            } else {
                "未成功"
            },
            preview.elapsed().as_secs()
        ),
        Ok(None) => format!(
            "项目 {project} 的后台数据预览运行中（进程 {}，已用时 {} 秒）；可继续编辑或训练，再回“数据预览、产物与清理”查看。",
            preview.pid(),
            preview.elapsed().as_secs()
        ),
        Err(error) => {
            format!(
                "项目 {project} 的后台数据预览状态未知：{error}；进入“数据预览、产物与清理”检查。"
            )
        }
    }
}

fn display_background_preview_result(
    preview: &mut BackgroundPreview,
    status: std::process::ExitStatus,
) -> Option<Report> {
    match preview.read_report() {
        Ok(report) => {
            display(&report, false);
            Some(report)
        }
        Err(error) => {
            println!("{error}");
            if !status.success()
                && let Ok(stderr) = preview.read_stderr()
                && !stderr.trim().is_empty()
            {
                println!("后台预览日志：\n{}", stderr.trim());
            }
            None
        }
    }
}

fn finish_background_preview<B: PythonRuntimeControl>(
    app: &mut Application<B>,
    cli: &mut Cli,
    current_project_root: &Path,
    model: Option<&str>,
    preview: &mut BackgroundPreview,
    status: std::process::ExitStatus,
    active: &mut Option<BackgroundPreview>,
) {
    let request = preview.command().clone();
    let preview_project_root = preview.project_root().to_path_buf();
    match display_background_preview_result(preview, status) {
        Some(report)
            if report.success()
                && report
                    .data
                    .get("published")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true) =>
        {
            if preview.project_root() == current_project_root {
                offer_preview_trial(app, cli, model, &request);
            } else {
                println!("此预览属于另一项目目录；结果已展示，但不会与当前项目模型组合试跑。\n");
            }
        }
        Some(report) if report.success() => {
            let request_id = report.data["request_id"].as_str().unwrap_or("未知");
            let generation = report.data["generation"].as_u64();
            let current_generation = report.data["current"]["generation"].as_u64();
            println!(
                "后台预览未发布为当前结果（状态：{}；请求：{}；generation：{}；当前 generation：{}）。旧结果仅供查看，不会用于试跑。\n",
                report.data["status"].as_str().unwrap_or("unknown"),
                request_id,
                generation.map_or_else(|| "未知".to_owned(), |value| value.to_string()),
                current_generation.map_or_else(|| "未知".to_owned(), |value| value.to_string()),
            );
        }
        Some(report) => recover_background_preview(
            app,
            cli,
            current_project_root,
            &preview_project_root,
            &request,
            &report,
            active,
        ),
        None => {}
    }
}

fn recover_background_preview<B: PythonRuntimeControl>(
    app: &mut Application<B>,
    cli: &mut Cli,
    current_project_root: &Path,
    preview_project_root: &Path,
    original: &Command,
    report: &Report,
    active: &mut Option<BackgroundPreview>,
) {
    let can_prepare_environment = report.diagnostics.iter().any(|diagnostic| {
        matches!(
            diagnostic.code.as_str(),
            "IMAGE_DECODER_UNAVAILABLE"
                | "ENVIRONMENT_RUNTIME_DEPENDENCY_MISSING"
                | "COMPONENT_DEPENDENCY_MISSING"
                | "PRESET_DEPENDENCY_MISSING"
        )
    });
    loop {
        let actions = if can_prepare_environment {
            "预览未成功：edit 修改设置后重试 · retry 原设置重试 · env 准备项目环境 · back 返回（原文件和样本未修改）"
        } else {
            "预览未成功：edit 修改设置后重试 · retry 原设置重试 · back 返回（原文件和样本未修改）"
        };
        let Some(action) = prompt_allow_empty(actions) else {
            return;
        };
        let command = match action.trim().to_ascii_lowercase().as_str() {
            "edit" | "1" | "修改" => match prompt_preview_command(original) {
                Ok(command) => command,
                Err(report) => {
                    display(&report, false);
                    return;
                }
            },
            "retry" | "2" | "重试" => original.clone(),
            "env" | "environment" | "环境" if can_prepare_environment => {
                if current_project_root != preview_project_root {
                    println!(
                        "这次预览属于另一个项目。请先切回该项目，再从其预览记录处理环境；当前项目不会被改动。\n"
                    );
                    continue;
                }
                let requirements = prompt_allow_empty("依赖文件 [requirements.txt]")
                    .filter(|value| !value.trim().is_empty())
                    .unwrap_or_else(|| "requirements.txt".into());
                if !prompt("环境准备可能下载并安装依赖；输入 setup 确认，其他输入取消")
                    .is_some_and(|answer| answer.trim().eq_ignore_ascii_case("setup"))
                {
                    println!("已取消环境准备；没有安装依赖。\n");
                    continue;
                }
                let environment = execute_ui(
                    app,
                    cli,
                    &Command::Env {
                        command: EnvironmentCommand::Setup {
                            requirements: requirements.into(),
                            venv: None,
                            update_lock: false,
                        },
                    },
                );
                println!("{}", render_environment(&environment).join("\n"));
                if environment.success() && cli.python.is_none() {
                    if let Some(executable) = environment.data["python"].as_str().filter(|value| !value.is_empty()) {
                        let executable = executable.to_owned();
                        cli.python = Some(executable.clone());
                        ApplicationUiServices::select_python(app, executable);
                    } else {
                        println!("环境准备回执没有解释器路径；请检查环境并明确选择 Python。\n");
                    }
                }
                continue;
            }
            "back" | "b" | "返回" | "q" | "" => return,
            _ if can_prepare_environment => {
                println!("请输入 edit、retry、env 或 back。\n");
                continue;
            }
            _ => {
                println!("请输入 edit、retry 或 back。\n");
                continue;
            }
        };
        if !prompt("将按以上设置重新启动独立后台预览；输入 start 确认，其他输入返回")
            .is_some_and(|answer| answer.trim().eq_ignore_ascii_case("start"))
        {
            println!("已取消重试；没有启动新的预览。\n");
            return;
        }
        match BackgroundPreview::launch(&command, cli, preview_project_root) {
            Ok(preview) => {
                println!("后台预览重试已启动（进程 {}）。\n", preview.pid());
                *active = Some(preview);
                return;
            }
            Err(error) => {
                println!("重试没有启动：{error}");
                return;
            }
        }
    }
}

fn offer_preview_trial<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    model: Option<&str>,
    preview_command: &Command,
) {
    let Command::Data {
        command:
            DataCommand::Preview {
                batch: Some(batch),
                whole_batch,
                parameters,
                ..
            },
    } = preview_command
    else {
        println!("Dataset 预览没有定义训练批次契约；如需试跑，请先提供显式批次工厂。\n");
        return;
    };
    let Some(model) = model else {
        println!("预览成功；尚未选择模型，返回工作台选择模型后再启动试跑。\n");
        return;
    };
    println!(
        "可将当前模型 `{model}` 与预览使用的批次工厂 `{batch}` 作为一份新输入快照，启动 1 步 managed-step 试跑。试跑会重新调用批次工厂，不复用预览中的临时样本；开始前会显示并校验运行计划。"
    );
    if *whole_batch {
        println!("试跑记录将使用 single-raw 绑定；训练步骤仍由用户代码决定如何使用返回值。");
    }
    let Some(step) = prompt("训练步骤入口 module:step；输入 back 留在工作台") else {
        return;
    };
    if matches!(
        step.trim().to_ascii_lowercase().as_str(),
        "back" | "b" | "返回"
    ) {
        return;
    }
    if !step.contains(':') {
        println!("训练步骤需填写 module:callable；没有启动试跑。\n");
        return;
    }
    let command = Command::Try {
        inputs: TrainingInputs {
            model: Some(model.into()),
            batch: Some(batch.clone()),
            batch_params: parameters.clone(),
            batch_binding: (*whole_batch).then_some(BatchBinding::SingleRaw),
            ..TrainingInputs::default()
        },
        repeat: None,
        name: Some("preview-data-trial".into()),
        training_mode: Some(TrainingMode::ManagedStep),
        step: Some(step),
        training_loop: None,
        steps: Some(1),
        lr: None,
        weight_decay: None,
        apply_supervision: false,
        no_supervision: false,
        params: Vec::new(),
        loop_params: Vec::new(),
    };
    run_training_with_plan(app, cli, &command);
}

fn manage_background_preview<B: PythonRuntimeControl>(
    app: &mut Application<B>,
    cli: &mut Cli,
    project_root: &Path,
    model: Option<&str>,
    batch: Option<&str>,
    active: &mut Option<BackgroundPreview>,
) {
    if let Some(preview) = active.as_mut() {
        match preview.try_wait() {
            Ok(Some(status)) => {
                let mut completed = active.take().expect("background preview exists");
                finish_background_preview(
                    app,
                    cli,
                    project_root,
                    model,
                    &mut completed,
                    status,
                    active,
                );
                return;
            }
            Err(error) => {
                println!("无法读取后台预览状态：{error}\n");
                return;
            }
            Ok(None) => {}
        }
        loop {
            let Some(action) = prompt_allow_empty(
                "后台预览仍在运行：Enter/refresh 查看状态 · wait 等待结束 · cancel 取消本次预览 · back 继续工作台",
            ) else {
                return;
            };
            match action.trim().to_ascii_lowercase().as_str() {
                "" | "refresh" | "r" => {
                    if let Some(preview) = active.as_mut() {
                        println!("{}\n", background_preview_summary(preview));
                        if let Ok(Some(status)) = preview.try_wait() {
                            let mut completed = active.take().expect("background preview exists");
                            finish_background_preview(
                                app,
                                cli,
                                project_root,
                                model,
                                &mut completed,
                                status,
                                active,
                            );
                            return;
                        }
                    }
                }
                "wait" | "w" | "等待" => {
                    let preview = active.as_mut().expect("background preview exists");
                    loop {
                        match preview.try_wait() {
                            Ok(Some(_)) => break,
                            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(200)),
                            Err(error) => {
                                println!("等待预览状态失败：{error}");
                                return;
                            }
                        }
                    }
                    let mut completed = active.take().expect("background preview exists");
                    match completed.try_wait() {
                        Ok(Some(status)) => finish_background_preview(
                            app,
                            cli,
                            project_root,
                            model,
                            &mut completed,
                            status,
                            active,
                        ),
                        Ok(None) => println!("预览仍在运行；可以返回工作台继续编辑。"),
                        Err(error) => println!("无法读取预览状态：{error}"),
                    }
                    return;
                }
                "cancel" | "c" | "取消" => {
                    let preview = active.as_mut().expect("background preview exists");
                    match preview.cancel() {
                        Ok(status) => {
                            let mut completed = active.take().expect("background preview exists");
                            display_background_preview_result(&mut completed, status);
                        }
                        Err(error) => {
                            println!("取消请求未完成：{error}\n预览仍由后台进程管理。")
                        }
                    }
                    return;
                }
                "back" | "b" | "返回" | "q" => return,
                _ => println!("请输入 refresh、wait、cancel 或 back。"),
            }
        }
    }

    let command = Command::Data {
        command: DataCommand::Preview {
            dataset: None,
            batch: batch.map(str::to_owned),
            whole_batch: false,
            parameters: Vec::new(),
            split: "train".into(),
            no_split: false,
            augment_split: None,
            samples: 12,
            scan_limit: None,
            fingerprint: None,
            transforms: None,
            renderer: "auto".into(),
            renderer_entrypoint: None,
            renderer_version: None,
            delivery: PreviewDelivery::Local,
            ssh_host: None,
            ssh_user: None,
            ssh_port: None,
            destination: "./preview.html".into(),
        },
    };
    let command = match prompt_preview_command(&command) {
        Ok(command) => command,
        Err(report) => {
            display(&report, false);
            return;
        }
    };
    if !prompt(
        "后台预览将使用以上独立请求；你可以回工作台继续编辑或训练。输入 start 启动，其他输入取消",
    )
    .is_some_and(|answer| answer.trim().eq_ignore_ascii_case("start"))
    {
        println!("已取消；没有启动预览。\n");
        return;
    }
    match BackgroundPreview::launch(&command, cli, project_root) {
        Ok(preview) => {
            println!(
                "后台预览已启动（进程 {}）。可继续工作；稍后从“数据预览、产物与清理”查看、等待或取消。\n",
                preview.pid()
            );
            *active = Some(preview);
        }
        Err(error) => println!("{error}\n预览没有启动。"),
    }
}

fn current_work_summary<B: Backend>(app: &Application<B>, cli: &Cli) -> String {
    let report = execute_ui(
        app,
        cli,
        &Command::Runs {
            command: RunsCommand::List {
                query: None,
                status: None,
                kept: None,
            },
        },
    );
    if !report.success() {
        return "运行状态未知；可进入“试跑与训练”查看诊断。".into();
    }
    let active = report.data["runs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|run| matches!(run["status"].as_str(), Some("running" | "stopping")))
        .collect::<Vec<_>>();
    match active.as_slice() {
        [] => "没有活动运行。".into(),
        [run] => format!(
            "活动运行 {} · {}；进入“试跑与训练”查看进度或停止。",
            run["id"].as_str().unwrap_or("未知 ID"),
            run["status"].as_str().unwrap_or("状态未知")
        ),
        runs => format!("有 {} 个活动运行；进入“试跑与训练”逐个查看。", runs.len()),
    }
}

fn python_environment_summary(executable: &str) -> String {
    match std::process::Command::new(executable)
        .args(["-c", "import sys; print(sys.version.split()[0])"])
        .output()
    {
        Ok(output) if output.status.success() => format!(
            "Python {} 可用；仍需检查 PyTorch 与 Worker 兼容性",
            String::from_utf8_lossy(&output.stdout).trim()
        ),
        Ok(output) => format!(
            "Python 可执行文件返回错误（退出码 {}）；可运行 m-model doctor 或用 --python 指定解释器",
            output.status
        ),
        Err(_) => format!(
            "找不到 Python 可执行文件 `{executable}`；可安装 Python、运行 m-model doctor，或用 --python 指定已有解释器"
        ),
    }
}

fn configuration_revision(config: Option<&Value>) -> Option<String> {
    config.and_then(|config| crate::domain::input::content_revision(config).ok())
}

fn is_resumable_page(page_id: &str) -> bool {
    matches!(
        page_id,
        "model" | "edit" | "runs" | "components" | "environment" | "outputs" | "project"
    )
}

pub fn interactive<B: PythonRuntimeControl>(app: &mut Application<B>, cli: &mut Cli) {
    let mut session = SessionState::new(PageId::new("home"));
    let mut session_root = std::env::current_dir()
        .and_then(|path| path.canonicalize())
        .unwrap_or_else(|_| PathBuf::from("."));
    let mut defaults = match ApplicationUiServices::new(app, cli).project_defaults(&session_root) {
        Ok(defaults) => defaults,
        Err(error) => {
            eprintln!("无法读取项目默认入口：{error}");
            None
        }
    };
    let mut onboarding_draft = match session_store::load(&session_root) {
        Ok(draft) => draft,
        Err(error) => {
            eprintln!("无法恢复向导会话：{error}");
            None
        }
    };
    let mut page_draft = match session_store::load_page_draft(&session_root) {
        Ok(draft) => draft,
        Err(error) => {
            eprintln!("无法恢复上次工作台位置：{error}");
            None
        }
    };
    session.model = defaults
        .as_ref()
        .and_then(|config| config["model"].as_str().map(str::to_owned));
    session.model_source = if cli.variant.is_some() {
        Some("命令行模型变体".to_owned())
    } else {
        session.model.as_ref().map(|_| "项目默认配置".to_owned())
    };
    session.batch = defaults
        .as_ref()
        .and_then(|config| config["batch"].as_str().map(str::to_owned));
    if onboarding_draft.is_some() {
        session.current_page = PageId::new("session_restore");
    } else if page_draft.is_some() {
        session.current_page = PageId::new("workbench_resume");
    } else if session.model.is_none() {
        session.current_page = PageId::new("onboarding");
    }
    session.variant = cli.variant.clone();
    let mut project = std::env::current_dir()
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "当前目录".into());
    let mut probe_cache: Option<ProbeCache> = None;
    let mut background_preview: Option<BackgroundPreview> = None;
    loop {
        if onboarding_draft.is_none() && session.current_page.0 != "workbench_resume" {
            if is_resumable_page(&session.current_page.0) {
                if let Err(error) = session_store::save_page_draft(
                    &session_root,
                    &session.current_page.0,
                    configuration_revision(defaults.as_ref()),
                ) {
                    eprintln!("无法保存工作台位置：{error}");
                }
            } else if session.current_page.0 == "home" {
                if let Err(error) = session_store::clear_page_draft(&session_root) {
                    eprintln!("无法清除工作台位置：{error}");
                }
                page_draft = None;
            }
        }
        let current_work = background_preview.as_mut().map_or_else(
            || current_work_summary(app, cli),
            background_preview_summary,
        );
        let mut pages = register_pages(
            &project,
            session.model.as_deref(),
            session.model_source.as_deref(),
            session.batch.as_deref(),
            session.variant.as_deref(),
            &current_work,
        )
        .expect("built-in page IDs are unique");
        let current_path = std::env::current_dir()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "当前目录不可读取".into());
        pages
            .register(render_onboarding(
                &current_path,
                session.model.is_some(),
                &cli.python.as_deref().filter(|value| !value.trim().is_empty())
                    .map(python_environment_summary)
                    .unwrap_or_else(|| "尚未选择或发现 Python；请用 --python 指定解释器".into()),
            ))
            .expect("onboarding page ID is unique");
        pages
            .register(render_project_open(&current_path, session.model.is_some()))
            .expect("project-open page ID is unique");
        if let Some(draft) = &onboarding_draft {
            let source_changed = draft.config_revision != configuration_revision(defaults.as_ref())
                || session_store::source_is_stale(&session_root, draft);
            pages
                .register(render_session_restore(
                    &current_path,
                    &draft.model,
                    draft.batch.as_deref(),
                    source_changed,
                ))
                .expect("session-restore page ID is unique");
        }
        if let Some(draft) = &page_draft {
            pages
                .register(render_workbench_resume(
                    resumable_page_name(&draft.page_id),
                    draft.config_revision != configuration_revision(defaults.as_ref()),
                ))
                .expect("workbench-resume page ID is unique");
        }
        let Some(page) = pages.get(&session.current_page) else {
            eprintln!("页面未注册：{}", session.current_page.0);
            break;
        };
        let content_width = layout::content_width(layout::terminal_width());
        println!("\n{}", text::wrap(&page.title, content_width).join("\n"));
        println!("{}", text::wrap(&page.summary, content_width).join("\n"));
        println!("{}", render_actions(&page.actions));
        let Some(input) = prompt("选择操作") else {
            break;
        };
        let choice = match resolve_selection(page, &input) {
            Selection::Navigate(NavigationResult::RunAction(action)) => action,
            Selection::Navigate(navigation) => {
                let quitting = navigation == NavigationResult::Quit;
                session.navigate(navigation);
                if quitting || session.quit_requested() {
                    break;
                }
                continue;
            }
            Selection::Ambiguous(actions) => {
                println!(
                    "匹配到多个操作，请选用以下完整名称或本页序号：{}",
                    actions.join("、")
                );
                continue;
            }
            Selection::Invalid => {
                println!("没有找到该操作；输入动作名称、关键词或本页序号。\n");
                continue;
            }
        };
        if session.current_page.0 == "project" && choice == "40" {
            session.current_page = PageId::new("project_open");
            continue;
        }
        if choice == "39" {
            manage_background_preview(
                app,
                cli,
                &session_root,
                session.model.as_deref(),
                session.batch.as_deref(),
                &mut background_preview,
            );
            continue;
        }
        if session.current_page.0 == "workbench_resume" {
            match choice.as_str() {
                "resume_page" => {
                    let Some(draft) = page_draft.as_ref() else {
                        session.current_page = PageId::new("home");
                        continue;
                    };
                    if pages.get(&PageId::new(draft.page_id.clone())).is_some() {
                        session.current_page = PageId::new(draft.page_id.clone());
                    } else {
                        println!("上次页面已不再可用；已返回首页。\n");
                        session.current_page = PageId::new("home");
                    }
                    if let Err(error) = session_store::clear_page_draft(&session_root) {
                        println!("页面已打开，但恢复点未能清除：{error}\n");
                    }
                    page_draft = None;
                    continue;
                }
                "discard" => {
                    match session_store::clear_page_draft(&session_root) {
                        Ok(()) => {
                            page_draft = None;
                            session.current_page = PageId::new("home");
                            println!("已丢弃页面恢复点；没有重放上次操作。\n");
                        }
                        Err(error) => println!("无法丢弃页面恢复点：{error}\n"),
                    }
                    continue;
                }
                _ => {}
            }
        }
        if session.current_page.0 == "session_restore" {
            let source_changed = onboarding_draft.as_ref().is_some_and(|draft| {
                draft.config_revision != configuration_revision(defaults.as_ref())
                    || session_store::source_is_stale(&session_root, draft)
            });
            match choice.as_str() {
                "restore" => {
                    if source_changed {
                        println!(
                            "项目配置或模型源码需要重新确认；先显式重新检查模型入口。没有启动模型或训练。\n"
                        );
                        continue;
                    }
                    if let Some(draft) = &onboarding_draft {
                        session.model = Some(draft.model.clone());
                        session.batch = draft.batch.clone();
                        session.model_source = Some("恢复的临时入口".into());
                        session.current_page = PageId::new("home");
                    }
                    continue;
                }
                "revalidate" => {
                    let Some(draft) = onboarding_draft.clone() else {
                        session.current_page = PageId::new("onboarding");
                        continue;
                    };
                    session.model = Some(draft.model.clone());
                    session.batch = draft.batch.clone();
                    session.model_source = Some("恢复的临时入口".into());
                    let revalidated = if let Some(batch) = draft.batch.as_deref() {
                        let command = Command::Model {
                            command: ModelCommand::Probe {
                                path: None,
                                inputs: Inputs {
                                    model: Some(draft.model.clone()),
                                    batch: Some(batch.to_owned()),
                                    ..Inputs::default()
                                },
                                call: "all".into(),
                                contract: None,
                                if_contract_sha256: None,
                                batch_fingerprint: None,
                                model_device: None,
                                local: false,
                            },
                        };
                        run_probe_interaction(
                            app,
                            cli,
                            &mut session.model,
                            &mut session.batch,
                            command,
                        )
                        .is_some()
                    } else {
                        let report = execute_ui(
                            app,
                            cli,
                            &Command::Model {
                                command: ModelCommand::Inspect {
                                    path: None,
                                    inputs: Inputs {
                                        model: Some(draft.model.clone()),
                                        batch: None,
                                        ..Inputs::default()
                                    },
                                },
                            },
                        );
                        display(&report, false);
                        report.success()
                    };
                    if revalidated {
                        match session_store::clear(&session_root) {
                            Ok(()) => {
                                onboarding_draft = None;
                                session.current_page = PageId::new("home");
                            }
                            Err(error) => println!("入口检查通过，但恢复点未能清除：{error}\n"),
                        }
                    }
                    continue;
                }
                "discard" => {
                    match session_store::clear(&session_root) {
                        Ok(()) => {
                            onboarding_draft = None;
                            session.current_page = PageId::new("onboarding");
                            println!("已丢弃接入会话；没有修改项目默认入口。\n");
                        }
                        Err(error) => println!("无法丢弃接入会话：{error}\n"),
                    }
                    continue;
                }
                "return_onboarding" => {
                    session.current_page = PageId::new("onboarding");
                    continue;
                }
                _ => {}
            }
        }
        if session.current_page.0 == "onboarding" {
            match choice.as_str() {
                "connect" => {
                    let Some(model) = prompt("根模型工厂入口（如 my_project.model:build_model）")
                    else {
                        continue;
                    };
                    let model = model.trim().to_owned();
                    if model.is_empty() {
                        println!("模型入口不能为空；没有修改项目配置。\n");
                        continue;
                    }
                    let batch = prompt_allow_empty(
                        "可选：代表批次工厂入口（如 my_project.data:example_batch）",
                    )
                    .map(|value| value.trim().to_owned())
                    .filter(|value| !value.is_empty());
                    let save = prompt("入口处理方式：1 临时查看（不写项目配置），2 保存为默认入口");
                    let Some(save) = save else {
                        continue;
                    };
                    if !matches!(save.trim(), "1" | "2") {
                        println!("请输入 1 或 2；没有修改项目配置。\n");
                        continue;
                    }
                    let save_default = save.trim() == "2";
                    if save_default {
                        let report = execute_ui(
                            app,
                            cli,
                            &Command::Init {
                                model: Some(model.clone()),
                                batch: batch.clone(),
                                injections: Vec::new(),
                                injection_params: Vec::new(),
                                batch_binding: None,
                                dry_run: false,
                            },
                        );
                        display(&report, false);
                        if !report.success() {
                            println!("默认入口未保存；你可以修正后重试。\n");
                            continue;
                        }
                        defaults = match ApplicationUiServices::new(app, cli)
                            .project_defaults(&session_root)
                        {
                            Ok(defaults) => defaults,
                            Err(error) => {
                                println!("默认入口已写入，但无法重新读取：{error}\n");
                                None
                            }
                        };
                        if let Err(error) = session_store::clear(&session_root) {
                            println!("默认入口已保存，但旧向导会话未能清除：{error}\n");
                        }
                        onboarding_draft = None;
                    } else {
                        let draft = match OnboardingDraft::new(
                            &session_root,
                            &model,
                            batch.as_deref(),
                            configuration_revision(defaults.as_ref()),
                        ) {
                            Ok(draft) => draft,
                            Err(error) => {
                                println!("无法保存恢复点：{error}\n");
                                continue;
                            }
                        };
                        if let Err(error) = session_store::save(&session_root, &draft) {
                            println!("无法保存恢复点：{error}\n");
                            continue;
                        }
                        onboarding_draft = Some(draft);
                    }
                    session.model = Some(model.clone());
                    session.model_source = Some(if save_default {
                        "已保存默认入口".into()
                    } else {
                        "本次临时入口".into()
                    });
                    session.batch = batch.clone();
                    session.current_page = PageId::new("home");
                    let inspect = if batch.is_none() {
                        true
                    } else {
                        prompt("首次查看：1 检查模型结构，2 前向查看代表批次")
                            .is_none_or(|answer| answer.trim() != "2")
                    };
                    if inspect {
                        let command = Command::Model {
                            command: ModelCommand::Inspect {
                                path: None,
                                inputs: Inputs {
                                    model: Some(model),
                                    batch,
                                    ..Inputs::default()
                                },
                            },
                        };
                        let report = execute_ui(app, cli, &command);
                        display(&report, false);
                        if report.success() && !save_default {
                            match session_store::clear(&session_root) {
                                Ok(()) => onboarding_draft = None,
                                Err(error) => println!("查看成功，但恢复点未能清除：{error}\n"),
                            }
                        }
                    } else {
                        let command = Command::Model {
                            command: ModelCommand::Probe {
                                path: None,
                                inputs: Inputs {
                                    model: Some(model),
                                    batch,
                                    ..Inputs::default()
                                },
                                call: "all".into(),
                                contract: None,
                                if_contract_sha256: None,
                                batch_fingerprint: None,
                                model_device: None,
                                local: false,
                            },
                        };
                        if let Some(report) = run_probe_interaction(
                            app,
                            cli,
                            &mut session.model,
                            &mut session.batch,
                            command,
                        ) {
                            if !save_default {
                                match session_store::clear(&session_root) {
                                    Ok(()) => onboarding_draft = None,
                                    Err(error) => {
                                        println!("前向查看成功，但恢复点未能清除：{error}\n")
                                    }
                                }
                            }
                            probe_cache = Some(ProbeCache {
                                model: session.model.clone(),
                                batch: session.batch.clone(),
                                variant: cli.variant.clone(),
                                seed: cli.seed,
                                report,
                            });
                        }
                    }
                    continue;
                }
                "preset" => {
                    let Some(preset_choice) = prompt(
                        "选择起步模型：1 Tiny CNN（轻量，无额外依赖），2 ResNet18（需兼容 torchvision；随机权重，不下载）",
                    ) else {
                        continue;
                    };
                    let (preset, preset_label) = match preset_choice.trim() {
                        "1" => ("tiny-cnn", "Tiny CNN 分类预设"),
                        "2" => {
                            let preflight = execute_ui(
                                app,
                                cli,
                                &Command::Model {
                                    command: ModelCommand::Inspect {
                                        path: None,
                                        inputs: Inputs {
                                            model: Some(
                                                "model_maven_worker.data.presets:build_resnet18"
                                                    .into(),
                                            ),
                                            model_params: vec![
                                                "num_classes=2".into(),
                                                "weights=\"random\"".into(),
                                            ],
                                            ..Inputs::default()
                                        },
                                    },
                                },
                            );
                            if !preflight.success() {
                                display(&preflight, false);
                                println!(
                                    "ResNet18 环境检查未通过；没有创建或修改项目文件。可安装兼容的 torchvision 后重试，或选择 Tiny CNN。\n"
                                );
                                continue;
                            }
                            ("resnet18", "ResNet18 分类预设（随机初始化）")
                        }
                        _ => {
                            println!("请输入 1 或 2；没有创建或修改项目文件。\n");
                            continue;
                        }
                    };
                    let Some(input_path) =
                        prompt("分类数据目录（每个类别一个子目录，可填相对路径）")
                    else {
                        continue;
                    };
                    let requested_path = PathBuf::from(input_path.trim());
                    let requested_path = if requested_path.is_absolute() {
                        requested_path
                    } else {
                        session_root.join(requested_path)
                    };
                    let data_root = match requested_path.canonicalize() {
                        Ok(path) if path.is_dir() => path,
                        Ok(_) => {
                            println!("所选数据路径不是目录；没有创建文件。\n");
                            continue;
                        }
                        Err(error) => {
                            println!("无法打开所选数据目录：{error}；没有创建文件。\n");
                            continue;
                        }
                    };
                    let classes = match classification_classes(&data_root) {
                        Ok(classes) => classes,
                        Err(error) => {
                            println!("{error}\n没有创建或修改项目文件。\n");
                            continue;
                        }
                    };
                    println!(
                        "\n检测到 {} 个类别目录：{}\n数据目录：{}\n",
                        classes.len(),
                        classes.join("、"),
                        data_root.display()
                    );
                    let Some(use_classes) = prompt("使用检测到的类别数继续？1 使用，2 取消")
                    else {
                        continue;
                    };
                    if use_classes.trim() != "1" {
                        println!("已取消；没有创建或修改项目文件。\n");
                        continue;
                    }
                    if defaults.is_some() {
                        println!(
                            "当前项目已有默认配置；分类预设不会覆盖现有项目。请新建项目目录后再使用。\n"
                        );
                        continue;
                    }
                    let Some(confirmation) = prompt(
                        "将创建 classification.py，并保存模型/批次默认入口；现有数据保持原样。1 创建，2 取消",
                    ) else {
                        continue;
                    };
                    if confirmation.trim() != "1" {
                        println!("已取消；没有创建或修改项目文件。\n");
                        continue;
                    }
                    if let Err(error) = validate_classification_preset_targets(&session_root) {
                        println!("{error}\n项目文件未修改。\n");
                        continue;
                    }
                    let preview = execute_ui(
                        app,
                        cli,
                        &Command::Init {
                            model: Some("classification:model".into()),
                            batch: Some("classification:batch".into()),
                            injections: Vec::new(),
                            injection_params: Vec::new(),
                            batch_binding: Some(BatchBinding::Envelope),
                            dry_run: true,
                        },
                    );
                    display(&preview, false);
                    if !preview.success() {
                        println!("入口预览未通过；项目文件未创建。\n");
                        continue;
                    }
                    let source = match create_classification_preset_files(
                        &session_root,
                        &data_root,
                        preset,
                        classes.len(),
                        None,
                    ) {
                        Ok(source) => source,
                        Err(error) => {
                            println!("{error}\n项目配置未修改。\n");
                            continue;
                        }
                    };
                    let setup = execute_ui(
                        app,
                        cli,
                        &Command::Init {
                            model: Some("classification:model".into()),
                            batch: Some("classification:batch".into()),
                            injections: Vec::new(),
                            injection_params: Vec::new(),
                            batch_binding: Some(BatchBinding::Envelope),
                            dry_run: false,
                        },
                    );
                    display(&setup, false);
                    if !setup.success() {
                        let source_path = session_root.join("classification.py");
                        if fs::read(&source_path).is_ok_and(|current| current == source.as_bytes())
                        {
                            let _ = fs::remove_file(source_path);
                        }
                        println!("预设入口没有保存，已清理本次新建的分类文件。\n");
                        continue;
                    }
                    defaults = match ApplicationUiServices::new(app, cli)
                        .project_defaults(&session_root)
                    {
                        Ok(defaults) => defaults,
                        Err(error) => {
                            println!("预设已保存，但无法重新读取项目配置：{error}\n");
                            None
                        }
                    };
                    session.model = Some("classification:model".into());
                    session.model_source = Some(preset_label.into());
                    session.batch = Some("classification:batch".into());
                    session.current_page = PageId::new("home");
                    println!(
                        "已创建可编辑的 {preset_label} classification.py 和项目默认入口。数据目录未移动；可直接编辑模型/批次函数。\n一步训练试跑：m-model try --step classification:step --steps 1\n"
                    );
                    let command = Command::Model {
                        command: ModelCommand::Probe {
                            path: None,
                            inputs: Inputs {
                                model: session.model.clone(),
                                batch: session.batch.clone(),
                                ..Inputs::default()
                            },
                            call: "all".into(),
                            contract: None,
                            if_contract_sha256: None,
                            batch_fingerprint: None,
                            model_device: None,
                            local: false,
                        },
                    };
                    if let Some(report) = run_probe_interaction(
                        app,
                        cli,
                        &mut session.model,
                        &mut session.batch,
                        command,
                    ) {
                        probe_cache = Some(ProbeCache {
                            model: session.model.clone(),
                            batch: session.batch.clone(),
                            variant: cli.variant.clone(),
                            seed: cli.seed,
                            report,
                        });
                    }
                    continue;
                }
                "environment" => {
                    let report = execute_ui(app, cli, &Command::Doctor);
                    println!("{}", render_environment(&report).join("\n"));
                    println!(
                        "若缺少项目依赖，可查看 doctor 结果后运行 `m-model env setup`；该命令会先解析并显示依赖计划。\n"
                    );
                    continue;
                }
                "continue_workbench" => {
                    session.current_page = PageId::new("home");
                    continue;
                }
                _ => {}
            }
        } else if session.current_page.0 == "project_open" && choice == "open_path" {
            let Some(input_path) = prompt("项目目录（绝对路径或相对当前目录的路径）")
            else {
                continue;
            };
            let current = match std::env::current_dir() {
                Ok(path) => path,
                Err(error) => {
                    println!("无法读取当前目录：{error}\n");
                    continue;
                }
            };
            let target = match resolve_project_path(&input_path, &current) {
                Ok(path) => path,
                Err(error) => {
                    println!("{error}\n");
                    continue;
                }
            };
            if let Err(error) = std::env::set_current_dir(&target) {
                println!("无法切换到项目目录 {}：{error}\n", target.display());
                continue;
            }
            // Worker capabilities describe a Python environment, not the
            // workspace process. A new project may select a different one.
            *app.runtime_capabilities.borrow_mut() = None;
            *app.runtime_declarations.borrow_mut() = None;
            session_root = target.clone();
            onboarding_draft = match session_store::load(&session_root) {
                Ok(draft) => draft,
                Err(error) => {
                    println!("目标目录中的向导会话无法恢复：{error}\n");
                    None
                }
            };
            page_draft = match session_store::load_page_draft(&session_root) {
                Ok(draft) => draft,
                Err(error) => {
                    println!("目标目录中的页面恢复点无法读取：{error}\n");
                    None
                }
            };
            let loaded_defaults =
                match ApplicationUiServices::new(app, cli).project_defaults(&target) {
                    Ok(defaults) => defaults,
                    Err(error) => {
                        println!("项目配置无法读取：{error}\n");
                        if !cli.python_explicit {
                            cli.python = Some(String::new());
                            app.backend.select_python(String::new());
                        }
                        defaults = None;
                        session.model = None;
                        session.batch = None;
                        project = target.file_name().map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_else(|| target.to_string_lossy().into_owned());
                        session.current_page = PageId::new("onboarding");
                        continue;
                    }
                };
            if !cli.python_explicit {
                let selected = loaded_defaults.as_ref()
                    .and_then(|config| config["runtime_python"].as_str().map(str::to_owned));
                let environment = if selected.is_none() { crate::project_python_executable() } else { None };
                if selected.is_none() && environment.is_none()
                    && std::fs::symlink_metadata(".model-maven/environment.json").is_ok()
                {
                    println!("目标项目的 Python 环境配置无效或解释器不可用；请修复环境或显式指定 --python。\n");
                    // Do not keep using the previous project's interpreter in
                    // the newly selected directory. An empty executable fails
                    // closed until the user repairs or explicitly selects one.
                    crate::record_project_python_selection(&target, None);
                    cli.python = Some(String::new());
                    app.backend.select_python(String::new());
                    defaults = loaded_defaults;
                    session.model = None;
                    session.batch = None;
                    project = target.file_name().map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_else(|| target.to_string_lossy().into_owned());
                    session.current_page = PageId::new("onboarding");
                    continue;
                }
                crate::record_project_python_selection(&target, selected.clone());
                cli.python = selected.or(environment).or_else(crate::path_python_executable);
                app.backend.select_python(cli.python.clone().unwrap_or_default());
            }
            defaults = loaded_defaults;
            session.model = defaults
                .as_ref()
                .and_then(|config| config["model"].as_str().map(str::to_owned));
            session.model_source = session.model.as_ref().map(|_| "项目默认配置".to_owned());
            session.batch = defaults
                .as_ref()
                .and_then(|config| config["batch"].as_str().map(str::to_owned));
            project = target
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| target.to_string_lossy().into_owned());
            session.current_page = if onboarding_draft.is_some() {
                PageId::new("session_restore")
            } else if page_draft.is_some() {
                PageId::new("workbench_resume")
            } else if session.model.is_some() {
                PageId::new("home")
            } else {
                println!("该目录尚未保存 Model-Maven 默认入口；可选择接入 Python 模型。\n");
                PageId::new("onboarding")
            };
            continue;
        }
        if matches!(
            choice.as_str(),
            "2" | "4"
                | "5"
                | "6"
                | "7"
                | "8"
                | "9"
                | "12"
                | "13"
                | "14"
                | "15"
                | "19"
                | "20"
                | "23"
                | "24"
                | "25"
                | "26"
                | "27"
                | "28"
                | "29"
                | "33"
        ) {
            probe_cache = None;
        }
        if choice == "30" {
            display(
                &execute_ui(
                    app,
                    cli,
                    &Command::Component {
                        command: ComponentCommand::Discover {
                            distribution: None,
                            paths: Vec::new(),
                        },
                    },
                ),
                false,
            );
            continue;
        }
        if choice == "34" {
            let history = execute_ui(
                app,
                cli,
                &Command::Config {
                    command: ProjectConfigCommand::History,
                },
            );
            display(&history, false);
            if !history.success() || history.data["history"].as_array().is_none_or(Vec::is_empty) {
                continue;
            }
            let Some(selection) = prompt("选择历史序号或输入 revision；留空返回")
            else {
                continue;
            };
            let selection = selection.trim();
            if selection.is_empty() {
                continue;
            }
            let revision = selection
                .parse::<usize>()
                .ok()
                .and_then(|index| index.checked_sub(1))
                .and_then(|index| history.data["history"][index]["revision"].as_str())
                .unwrap_or(selection)
                .to_owned();
            let Some(field_input) =
                prompt("恢复字段（逗号分隔）：model, batch, batch_binding, model_injections")
            else {
                continue;
            };
            let fields = field_input
                .split(',')
                .map(str::trim)
                .filter(|field| !field.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let restore_command = |apply, if_current| Command::Config {
                command: ProjectConfigCommand::Restore {
                    revision: revision.clone(),
                    fields: fields.clone(),
                    apply,
                    if_current,
                },
            };
            let preview = execute_ui(app, cli, &restore_command(false, None));
            display(&preview, false);
            let has_changes = preview.data["fields"]
                .as_array()
                .is_some_and(|changes| changes.iter().any(|change| change["changed"] == true));
            if preview.success()
                && has_changes
                && prompt("写入上述字段，并将当前版本留在历史中？[y/N]").is_some_and(|answer| {
                    matches!(answer.to_ascii_lowercase().as_str(), "y" | "yes")
                })
            {
                if let Some(expected_revision) =
                    preview.data["from_revision"].as_str().map(str::to_owned)
                {
                    display(
                        &execute_ui(app, cli, &restore_command(true, Some(expected_revision))),
                        false,
                    );
                }
            }
            continue;
        }
        if choice == "37" {
            let revision = if defaults.is_some() {
                let history = execute_ui(
                    app,
                    cli,
                    &Command::Config {
                        command: ProjectConfigCommand::History,
                    },
                );
                if !history.success() {
                    display(&history, false);
                }
                history.data["current_revision"].as_str().map(str::to_owned)
            } else {
                None
            };
            let page = render_configuration(
                &std::env::current_dir()
                    .map(|path| path.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| project.clone()),
                defaults.as_ref(),
                revision.as_deref(),
                session.model.as_deref(),
                session.model_source.as_deref(),
                session.batch.as_deref(),
                session.variant.as_deref(),
            );
            println!("\n{}\n{}", page.title, page.summary);
            let _ = prompt_allow_empty("按 Enter 返回");
            continue;
        }
        if choice == "31" {
            println!("1 准备/复用项目环境  2 检查锁与当前环境  3 解析并生成依赖锁  0 返回");
            let command = match prompt("环境操作").as_deref() {
                Some("1") => {
                    let requirements = prompt("依赖文件 [requirements.txt]")
                        .filter(|value| !value.is_empty())
                        .unwrap_or_else(|| "requirements.txt".into());
                    Command::Env {
                        command: EnvironmentCommand::Setup {
                            requirements: requirements.into(),
                            venv: None,
                            update_lock: false,
                        },
                    }
                }
                Some("2") => Command::Env {
                    command: EnvironmentCommand::Check { venv: None },
                },
                Some("3") => {
                    let requirements = prompt("依赖文件 [requirements.txt]")
                        .filter(|value| !value.is_empty())
                        .unwrap_or_else(|| "requirements.txt".into());
                    Command::Env {
                        command: EnvironmentCommand::Lock {
                            requirements: requirements.into(),
                            refresh: false,
                        },
                    }
                }
                _ => continue,
            };
            let report = execute_ui(app, cli, &command);
            let setup_succeeded = matches!(
                command,
                Command::Env {
                    command: EnvironmentCommand::Setup { .. }
                }
            ) && report.success();
            println!("{}", render_environment(&report).join("\n"));
            if setup_succeeded && cli.python.is_none() {
                if let Some(executable) = report.data["python"].as_str().filter(|value| !value.is_empty()) {
                    let executable = executable.to_owned();
                    cli.python = Some(executable.clone());
                    ApplicationUiServices::select_python(app, executable);
                    println!("本次工作台会话已切换到项目虚拟环境。");
                } else {
                    println!("环境准备回执没有解释器路径；请检查环境并明确选择 Python。\n");
                }
            }
            continue;
        }
        if choice == "10" {
            let report = execute_ui(app, cli, &Command::Doctor);
            println!("{}", render_environment(&report).join("\n"));
            continue;
        }
        if choice == "11" {
            session.model = None;
            session.model_source = None;
            session.batch = None;
            println!("已清除本次会话中的模型和批次入口；已保存变体不受影响。");
            continue;
        }
        if matches!(choice.as_str(), "26" | "27") {
            let command = Command::Layer {
                command: if choice == "26" {
                    LayerCommand::Undo {
                        inputs: Inputs::default(),
                        target: "source".into(),
                        dry_run: false,
                        require_forward: false,
                    }
                } else {
                    LayerCommand::Redo {
                        inputs: Inputs::default(),
                        target: "source".into(),
                        dry_run: false,
                        require_forward: false,
                    }
                },
            };
            display(&execute_ui(app, cli, &command), false);
            continue;
        }
        if choice == "16" {
            browse_runs_interactive(app, cli);
            continue;
        }
        if choice == "17" {
            display(
                &execute_ui(
                    app,
                    cli,
                    &Command::Runs {
                        command: RunsCommand::Show { id: None },
                    },
                ),
                false,
            );
            continue;
        }
        if choice == "18" {
            let Some(directory) = prompt("导出目录（须位于当前项目目录之外）")
            else {
                continue;
            };
            let export = |overwrite| Command::Export {
                directory: directory.clone().into(),
                run: None,
                if_checkpoint_sha256: None,
                overwrite,
            };
            let report = execute_ui(app, cli, &export(false));
            let target_conflict = report.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == "EXPORT_FAILED" && diagnostic.message.contains("导出目标已存在")
            });
            println!("{}", render_export(&report).join("\n"));
            if target_conflict {
                let confirmed = prompt_allow_empty(
                    "目标已存在。输入 OVERWRITE 确认先备份原目录再替换；其他输入保留原目录",
                )
                .is_some_and(|answer| answer.trim() == "OVERWRITE");
                if confirmed {
                    let report = execute_ui(app, cli, &export(true));
                    println!("{}", render_export(&report).join("\n"));
                } else {
                    println!("已取消覆盖；原导出目录保持不变。\n");
                }
            }
            continue;
        }
        if choice == "19" {
            let id = prompt("运行 ID（留空自动选择唯一活动训练）");
            let command = Command::Runs {
                command: RunsCommand::Attach {
                    id,
                    follow: Some(true),
                },
            };
            let report = run_command(app, cli, &command);
            display(&report, false);
            continue;
        }
        if choice == "20" {
            let id = prompt("运行 ID（留空自动选择唯一活动训练）");
            let command = Command::Runs {
                command: RunsCommand::Stop { id, force: false },
            };
            let report = execute_ui(app, cli, &command);
            display(&report, false);
            continue;
        }
        if choice == "21" {
            let report = execute_ui(
                app,
                cli,
                &Command::Runs {
                    command: RunsCommand::Storage { id: None },
                },
            );
            if report.success() {
                println!("{}", render_storage(&report.data).join("\n"));
            } else {
                display(&report, false);
            }
            continue;
        }
        if choice == "35" {
            let report = execute_ui(
                app,
                cli,
                &Command::Runs {
                    command: RunsCommand::List {
                        query: None,
                        status: None,
                        kept: None,
                    },
                },
            );
            if !report.success() {
                display(&report, false);
                continue;
            }
            let mut runs = report.data["runs"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|run| run["operation"] == "training_probe")
                .cloned()
                .collect::<Vec<_>>();
            runs.sort_by(|left, right| {
                right["started_at_ms"]
                    .as_u64()
                    .unwrap_or(0)
                    .cmp(&left["started_at_ms"].as_u64().unwrap_or(0))
            });
            if runs.len() < 2 {
                match runs.len() {
                    0 => println!(
                        "还没有训练步骤试跑记录；完成首次试跑后，这里会保留它作为对照基线。"
                    ),
                    _ => println!("目前只有一次训练步骤试跑；再完成一次即可与这条记录比较。"),
                }
                continue;
            }
            println!("可比较的训练步骤试跑：");
            for (index, run) in runs.iter().enumerate() {
                println!(
                    "  {}. {} · {} · {} · {} 步 · Loss {}",
                    index + 1,
                    run["name"]
                        .as_str()
                        .filter(|name| !name.is_empty())
                        .unwrap_or_else(|| run["id"].as_str().unwrap_or("未知 ID")),
                    run["id"].as_str().unwrap_or("未知 ID"),
                    run["status"].as_str().unwrap_or("状态未知"),
                    run["summary"]["completed_steps"],
                    run["summary"]["mean_loss"]
                );
            }
            let Some(left) = choose_run(&runs, 0, "选择左侧运行序号或 ID；留空选最近一次")
            else {
                continue;
            };
            let Some(right) = choose_run(&runs, 1, "选择右侧运行序号或 ID；留空选第二近一次")
            else {
                continue;
            };
            if left == right {
                println!("请选择两条不同的运行记录。");
                continue;
            }
            let report = execute_ui(
                app,
                cli,
                &Command::Runs {
                    command: RunsCommand::Compare {
                        left: left.clone(),
                        right: right.clone(),
                    },
                },
            );
            if report.success() {
                println!(
                    "\n{}",
                    crate::presentation::pages::comparison::render(&report.data).join("\n")
                );
            } else {
                display(&report, false);
                continue;
            }
            match prompt_allow_empty("输入 left/right 按对应试跑的原始设置再试；留空返回")
                .as_deref()
                .map(str::trim)
            {
                Some("left" | "1") => {
                    display(&run_command(app, cli, &repeat_trial_command(left)), false);
                }
                Some("right" | "2") => {
                    display(&run_command(app, cli, &repeat_trial_command(right)), false);
                }
                _ => {}
            }
            continue;
        }
        if choice == "32" {
            let mut run =
                prompt("运行 ID（留空检查所有已结束运行）").filter(|value| !value.is_empty());
            let include_artifacts = prompt("同时清理 worker.log 和 progress.jsonl？[Y/n]")
                .is_none_or(|answer| {
                    !matches!(answer.trim().to_ascii_lowercase().as_str(), "n" | "no")
                });
            let release_checkpoints = prompt("要预览并释放某个已结束运行的 checkpoint？[y/N]")
                .is_some_and(|answer| {
                    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
                });
            let artifact = if release_checkpoints {
                let run_id = match run.clone() {
                    Some(id) => id,
                    None => match prompt("请输入要检查的运行 ID") {
                        Some(id) if !id.trim().is_empty() => {
                            run = Some(id.clone());
                            id
                        }
                        _ => continue,
                    },
                };
                let details = execute_ui(
                    app,
                    cli,
                    &Command::Runs {
                        command: RunsCommand::Show { id: Some(run_id) },
                    },
                );
                if !details.success() {
                    display(&details, false);
                    continue;
                }
                let candidates = details.data["artifacts"]["artifacts"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|item| {
                        item["availability"] == "present"
                            && item["complete"] == true
                            && (item["id"] == "training-checkpoint"
                                || (item["kind"] == "checkpoint"
                                    && item["references"].as_array().is_some_and(Vec::is_empty)))
                    })
                    .collect::<Vec<_>>();
                if candidates.is_empty() {
                    println!("没有可显式释放的完整 checkpoint。\n");
                    continue;
                }
                println!("可释放的 checkpoint：");
                for (index, item) in candidates.iter().enumerate() {
                    let files = item["files"].as_array().map_or(0, Vec::len);
                    let bytes = item["files"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|file| file["size_bytes"].as_u64().unwrap_or(0))
                        .fold(0_u64, u64::saturating_add);
                    println!(
                        "  {}. {} · {} 个文件 · {} bytes · {}",
                        index + 1,
                        item["id"].as_str().unwrap_or("unknown"),
                        files,
                        bytes,
                        item["owner"].as_str().unwrap_or("未知 owner")
                    );
                }
                let selection = match prompt("输入要释放项的编号（留空取消）")
                    .and_then(|value| value.trim().parse::<usize>().ok())
                {
                    Some(index) if (1..=candidates.len()).contains(&index) => index - 1,
                    _ => continue,
                };
                candidates[selection]["id"].as_str().map(str::to_owned)
            } else {
                None
            };
            if release_checkpoints && artifact.is_none() {
                continue;
            }
            let preview = execute_ui(
                app,
                cli,
                &Command::Runs {
                    command: RunsCommand::Clean {
                        run,
                        include_logs: include_artifacts,
                        include_progress: include_artifacts,
                        include_checkpoints: release_checkpoints,
                        artifact: artifact.clone(),
                        dry_run: true,
                        apply: None,
                        release_checkpoints: false,
                    },
                },
            );
            if !preview.success() {
                display(&preview, false);
                continue;
            }
            println!("{}", render_clean_plan(&preview.data).join("\n"));
            if preview.data["items"].as_array().is_none_or(Vec::is_empty) {
                continue;
            }
            let Some(plan_id) = preview.data["id"].as_str() else {
                continue;
            };
            let confirmation = if release_checkpoints {
                "RELEASE"
            } else {
                "CLEAN"
            };
            if prompt(&format!(
                "输入 {confirmation} 确认固定清单；释放 checkpoint 后仍保留运行摘要"
            ))
            .as_deref()
                != Some(confirmation)
            {
                println!("已取消；文件未改变。\n");
                continue;
            }
            let result = execute_ui(
                app,
                cli,
                &Command::Runs {
                    command: RunsCommand::Clean {
                        run: None,
                        include_logs: false,
                        include_progress: false,
                        include_checkpoints: release_checkpoints,
                        artifact: None,
                        dry_run: false,
                        apply: Some(plan_id.to_owned()),
                        release_checkpoints,
                    },
                },
            );
            if result.success() {
                println!("{}", render_clean_result(&result.data).join("\n"));
            } else {
                display(&result, false);
            }
            continue;
        }
        if choice == "33" {
            let overview = execute_ui(
                app,
                cli,
                &Command::Assembly {
                    command: AssemblyCommand::Show {
                        inputs: AssemblyInputs::default(),
                        config_only: false,
                    },
                },
            );
            if overview.success() {
                println!(
                    "\n{}",
                    render_assembly(&serde_json::to_value(&overview).unwrap_or(Value::Null))
                        .join("\n")
                );
            } else {
                display(&overview, false);
            }
            let Some(action) = prompt(
                "装配操作：add / set / remove / preview（根工厂接缝，不是 Layer 路径；留空返回）",
            ) else {
                continue;
            };
            if action == "show" {
                continue;
            }
            if !matches!(action.as_str(), "add" | "set" | "remove" | "preview") {
                println!("请输入 add、set、remove 或 preview。\n");
                continue;
            }
            let dry_run = action == "preview";
            let action = if dry_run {
                let Some(action) = prompt("预览哪项操作：add / set / remove") else {
                    continue;
                };
                if !matches!(action.as_str(), "add" | "set" | "remove") {
                    println!("请输入 add、set 或 remove。\n");
                    continue;
                }
                action
            } else {
                action
            };
            let Some(slot) = prompt(
                "根工厂接缝名（如 encoder、fusion、head；不是 model.encoder 这样的模块路径）",
            ) else {
                continue;
            };
            let batch = prompt("可选：临时代表批次 module:factory（留空使用项目默认值）")
                .filter(|value| !value.is_empty());
            let has_default_batch = defaults
                .as_ref()
                .is_some_and(|config| config["batch"].is_string());
            let has_batch = batch.is_some() || has_default_batch;
            let batch_binding = if has_batch {
                match prompt("批次绑定 envelope / single / single-raw / args / kwargs（single-raw 整体传入；留空自动推断）")
                    .as_deref()
                {
                    Some("envelope") => Some(BatchBinding::Envelope),
                    Some("single") => Some(BatchBinding::Single),
                    Some("single-raw") => Some(BatchBinding::SingleRaw),
                    Some("args") => Some(BatchBinding::Args),
                    Some("kwargs") => Some(BatchBinding::Kwargs),
                    _ => None,
                }
            } else {
                None
            };
            let batch_params = if has_batch {
                println!("以下是批次工厂参数；仅用于本次候选验证，不会保存为项目批次参数。");
                let Some(params) = prompt_parameter_map("assembly.batch") else {
                    println!("批次参数表单已取消；装配未更改。\n");
                    continue;
                };
                parameters_for_cli(&params)
            } else {
                Vec::new()
            };
            println!("以下是根模型工厂构造参数；留空表示不覆盖项目配置。");
            let Some(model_params) = prompt_parameter_map("assembly.model") else {
                println!("模型参数表单已取消；装配未更改。\n");
                continue;
            };
            let inputs = AssemblyInputs {
                if_config_revision: None,
                model_params: parameters_for_cli(&model_params),
                batch_params,
                batch,
                batch_binding,
            };
            if action != "remove" {
                let Some(binding) =
                    prompt("绑定已登记 Layer 组件 ID@VERSION，或输入 Python module:factory")
                else {
                    continue;
                };
                if binding.contains(':') {
                    let command = if action == "add" {
                        AssemblyCommand::Add {
                            slot,
                            component: None,
                            factory: Some(binding),
                            params: Vec::new(),
                            inputs,
                            dry_run,
                        }
                    } else {
                        AssemblyCommand::Set {
                            slot,
                            component: None,
                            factory: Some(binding),
                            params: Vec::new(),
                            inputs,
                            dry_run,
                        }
                    };
                    display(&execute_ui(app, cli, &Command::Assembly { command }), false);
                } else {
                    let mut component = Some(binding);
                    let Some(params) = prompt_layer_parameters(app, cli, &mut component, None)
                    else {
                        println!("组件参数输入已取消；装配未更改。\n");
                        continue;
                    };
                    let component = component.expect("registered component selection is retained");
                    let params = parameters_for_cli(&params);
                    let command = if action == "add" {
                        AssemblyCommand::Add {
                            slot,
                            component: Some(component),
                            factory: None,
                            params,
                            inputs,
                            dry_run,
                        }
                    } else {
                        AssemblyCommand::Set {
                            slot,
                            component: Some(component),
                            factory: None,
                            params,
                            inputs,
                            dry_run,
                        }
                    };
                    display(&execute_ui(app, cli, &Command::Assembly { command }), false);
                }
            } else {
                display(
                    &execute_ui(
                        app,
                        cli,
                        &Command::Assembly {
                            command: AssemblyCommand::Remove {
                                slot,
                                inputs,
                                dry_run,
                            },
                        },
                    ),
                    false,
                );
            }
            continue;
        }
        if choice == "22" {
            let query = prompt("搜索组件（留空显示全部）").filter(|value| !value.is_empty());
            let catalog = execute_ui(
                app,
                cli,
                &Command::Component {
                    command: ComponentCommand::List { query },
                },
            );
            if !catalog.success() {
                display(&catalog, false);
                continue;
            }
            let available_component_actions = catalog.data["components"]
                .as_array()
                .into_iter()
                .flatten()
                .flat_map(|component| component_actions(&component["manifest"]))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .join(" / ");
            let catalog_value = serde_json::to_value(&catalog).unwrap_or(Value::Null);
            let mut catalog_page = 0usize;
            let catalog_page_size = 10usize;
            let component_action = loop {
                let width = layout::content_width(layout::terminal_width());
                for line in
                    render_component_catalog_page(&catalog_value, catalog_page, catalog_page_size)
                {
                    for wrapped in text::wrap(&line, width) {
                        println!("{wrapped}");
                    }
                }
                let action_summary = if available_component_actions.is_empty() {
                    "当前无已登记组件"
                } else {
                    available_component_actions.as_str()
                };
                let action = prompt(&format!(
                    "n/p 翻页 · 组件动作：{action_summary}；install 安装、discover 发现、scaffold 创建包；留空返回"
                ));
                match action
                    .as_ref()
                    .map(|value| value.trim().to_ascii_lowercase())
                    .as_deref()
                {
                    Some("n" | "next") => {
                        let count = catalog.data["components"].as_array().map_or(0, Vec::len);
                        let page_count = count.div_ceil(catalog_page_size).max(1);
                        catalog_page = (catalog_page + 1).min(page_count - 1);
                    }
                    Some("p" | "prev") => catalog_page = catalog_page.saturating_sub(1),
                    _ => break action,
                }
            };
            if component_action
                .as_ref()
                .is_some_and(|answer| answer.eq_ignore_ascii_case("show"))
            {
                let Some(id) = prompt("组件 ID（namespace/name[@version]）") else {
                    continue;
                };
                let detail = execute_ui(
                    app,
                    cli,
                    &Command::Component {
                        command: ComponentCommand::Show { id },
                    },
                );
                if detail.success() {
                    println!(
                        "\n{}",
                        render_component_detail(
                            &serde_json::to_value(&detail).unwrap_or(Value::Null)
                        )
                        .join("\n")
                    );
                } else {
                    display(&detail, false);
                }
                continue;
            }
            if component_action
                .as_ref()
                .is_some_and(|answer| answer.eq_ignore_ascii_case("discover"))
            {
                display(
                    &execute_ui(
                        app,
                        cli,
                        &Command::Component {
                            command: ComponentCommand::Discover {
                                distribution: None,
                                paths: Vec::new(),
                            },
                        },
                    ),
                    false,
                );
                continue;
            }
            if component_action
                .as_ref()
                .is_some_and(|answer| answer.eq_ignore_ascii_case("install"))
            {
                let Some(path) = prompt("扩展包目录（含 pyproject.toml）") else {
                    continue;
                };
                let no_build_isolation = prompt(
                    "使用当前 Python 环境的构建依赖跳过隔离下载？这要求 [build-system].requires 已安装。[y/N]",
                )
                .is_some_and(|answer| matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes"));
                display(
                    &execute_ui(
                        app,
                        cli,
                        &Command::Component {
                            command: ComponentCommand::Install {
                                path: path.into(),
                                no_build_isolation,
                            },
                        },
                    ),
                    false,
                );
                continue;
            }
            if component_action
                .as_ref()
                .is_some_and(|answer| answer.eq_ignore_ascii_case("uninstall"))
            {
                let Some(id) = prompt("精确组件版本（namespace/name@MAJOR.MINOR.PATCH）")
                else {
                    continue;
                };
                let preview = execute_ui(
                    app,
                    cli,
                    &Command::Component {
                        command: ComponentCommand::Uninstall {
                            id: id.clone(),
                            apply: false,
                            force: false,
                            if_plan: None,
                        },
                    },
                );
                display(&preview, false);
                let active_runs = preview.data["active_runs"]
                    .as_array()
                    .or_else(|| preview.data["plan"]["active_runs"].as_array())
                    .cloned()
                    .unwrap_or_default();
                if !active_runs.is_empty() {
                    println!("卸载被以下运行阻止：");
                    for run in &active_runs {
                        println!(
                            "  {} · {}",
                            run["id"].as_str().unwrap_or("未知运行"),
                            run["status"].as_str().unwrap_or("状态未知")
                        );
                    }
                }
                if !preview.success() || !active_runs.is_empty() {
                    continue;
                }
                let Some(plan_revision) = preview.data["plan_revision"]
                    .as_str()
                    .filter(|s| {
                        s.len() == 64
                            && s.bytes()
                                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    })
                    .map(str::to_owned)
                else {
                    println!("组件移除预览缺少可核对的计划摘要；未移除。\n");
                    continue;
                };
                let requires_force = preview.data["requires_force"] == true;
                let expected = if requires_force { "FORCE" } else { id.as_str() };
                let prompt_text = if requires_force {
                    "存在变体/源码引用或源码扫描不完整；输入 FORCE 接受可能的重载失败并继续"
                } else {
                    "输入完整组件 ID@版本确认移除"
                };
                if prompt(prompt_text).as_deref() != Some(expected) {
                    println!("已取消；组件和扩展文件未改变。\n");
                    continue;
                }
                display(
                    &execute_ui(
                        app,
                        cli,
                        &Command::Component {
                            command: ComponentCommand::Uninstall {
                                id,
                                apply: true,
                                force: requires_force,
                                if_plan: Some(plan_revision),
                            },
                        },
                    ),
                    false,
                );
                continue;
            }
            if component_action
                .as_ref()
                .is_some_and(|answer| answer.eq_ignore_ascii_case("scaffold"))
            {
                let Some(id) = prompt("新组件 ID（namespace/name）") else {
                    continue;
                };
                let kind = match prompt("类型：layer 或 layer-edit-provider [layer]")
                    .filter(|value| !value.is_empty())
                    .as_deref()
                    .unwrap_or("layer")
                {
                    "layer" | "1" => "layer",
                    "layer-edit-provider" | "provider" | "2" => "layer-edit-provider",
                    _ => {
                        eprintln!("类型只能是 layer 或 layer-edit-provider。\n");
                        continue;
                    }
                };
                let version = prompt("初始版本 [0.1.0]")
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "0.1.0".into());
                let default_output = id
                    .rsplit('/')
                    .next()
                    .map(|name| format!("extensions/{name}"))
                    .unwrap_or_else(|| "extensions/new-component".into());
                let output = prompt(&format!("项目内输出目录 [{default_output}]"))
                    .filter(|value| !value.is_empty())
                    .unwrap_or(default_output);
                display(
                    &execute_ui(
                        app,
                        cli,
                        &Command::Component {
                            command: ComponentCommand::Scaffold {
                                id,
                                kind: kind.into(),
                                version,
                                output: output.into(),
                            },
                        },
                    ),
                    false,
                );
                continue;
            }
            if component_action
                .as_ref()
                .is_some_and(|answer| answer.eq_ignore_ascii_case("check"))
            {
                if let Some(id) = prompt("组件 ID（namespace/name[@version]）") {
                    display(
                        &execute_ui(
                            app,
                            cli,
                            &Command::Component {
                                command: ComponentCommand::Check { id },
                            },
                        ),
                        false,
                    );
                }
                continue;
            }
            if component_action.is_some_and(|answer| answer.eq_ignore_ascii_case("test")) {
                let Some(id) = prompt("Layer 组件 ID（namespace/name[@version]）") else {
                    continue;
                };
                if id.contains(':') {
                    println!("component test 需要已注册的 Layer 组件 ID。");
                    continue;
                }
                let kind = std::env::current_dir()
                    .ok()
                    .and_then(|path| path.canonicalize().ok())
                    .and_then(|root| {
                        ApplicationUiServices::new(app, cli)
                            .inspect_component(&root, &id)
                            .ok()
                    })
                    .and_then(|descriptor| {
                        descriptor["manifest"]["kind"].as_str().map(str::to_owned)
                    });
                let mut selected = Some(id);
                if !matches!(kind.as_deref(), Some("layer" | "layer-edit-provider")) {
                    eprintln!(
                        "{}",
                        kind.map_or_else(
                            || "无法解析已注册组件 ID。".into(),
                            |kind| format!("当前没有 {kind} 组件的运行时符合性检查。")
                        )
                    );
                    continue;
                }
                let selected_id = selected.as_deref().expect("component ID was selected");
                let can_probe = std::env::current_dir()
                    .ok()
                    .and_then(|path| path.canonicalize().ok())
                    .and_then(|root| {
                        ApplicationUiServices::new(app, cli)
                            .inspect_component(&root, selected_id)
                            .ok()
                    })
                    .is_some_and(|descriptor| {
                        descriptor["manifest"]["kind"] == "layer" && {
                            let capability =
                                &descriptor["manifest"]["capabilities"]["forward_probe"];
                            matches!(
                                capability["state"].as_str().or_else(|| capability.as_str()),
                                Some("supported" | "conditional")
                            )
                        }
                    });
                loop {
                    let params = match kind.as_deref() {
                        Some("layer") => prompt_layer_parameters(app, cli, &mut selected, None),
                        Some("layer-edit-provider") => {
                            prompt_layer_edit_provider_parameters(app, cli, &mut selected)
                        }
                        _ => unreachable!("the component kind was checked above"),
                    };
                    let Some(params) = params else {
                        println!("组件参数表单已取消；没有运行测试。");
                        break;
                    };
                    let Some(selected_id) = selected.clone() else {
                        println!("组件版本选择已取消；没有运行测试。");
                        break;
                    };
                    let batch = if can_probe {
                        prompt("可选：真实前向批次工厂 module:callable（留空只测导入和构造）")
                    } else {
                        None
                    };
                    let batch_params = if batch.is_some() {
                        let Some(params) =
                            prompt_parameter_map(&format!("component_test.batch.{selected_id}"))
                        else {
                            println!("批次参数表单已取消；组件未测试。\n");
                            break;
                        };
                        params
                    } else {
                        serde_json::Map::new()
                    };
                    let result = execute_ui(
                        app,
                        cli,
                        &Command::Component {
                            command: ComponentCommand::Test {
                                id: selected_id,
                                params: parameters_for_cli(&params),
                                batch,
                                batch_params: parameters_for_cli(&batch_params),
                                batch_binding: None,
                            },
                        },
                    );
                    let passed = result.success();
                    display(&result, false);
                    if passed
                        || !prompt(
                            "组件测试未通过。输入 retry 留在此组件上修正参数并重测；其他输入返回",
                        )
                        .is_some_and(|answer| answer.eq_ignore_ascii_case("retry"))
                    {
                        break;
                    }
                }
            }
            continue;
        }
        if choice == "24" {
            display(
                &execute_ui(
                    app,
                    cli,
                    &Command::Variant {
                        command: VariantCommand::List { query: None },
                    },
                ),
                false,
            );
            match prompt("输入 show 查看详情、copy 创建分支，或留空返回").as_deref()
            {
                Some("show") => {
                    if let Some(name) = prompt("变体名称") {
                        display(
                            &execute_ui(
                                app,
                                cli,
                                &Command::Variant {
                                    command: VariantCommand::Show { name },
                                },
                            ),
                            false,
                        );
                    }
                }
                Some("copy") => {
                    let Some(source) = prompt("源变体名称") else {
                        continue;
                    };
                    let Some(destination) = prompt("新分支名称") else {
                        continue;
                    };
                    display(
                        &execute_ui(
                            app,
                            cli,
                            &Command::Variant {
                                command: VariantCommand::Copy {
                                    source,
                                    destination,
                                },
                            },
                        ),
                        false,
                    );
                }
                _ => {}
            }
            continue;
        }
        if choice == "15" {
            let Some((resume, source)) = choose_resume_source(app, cli) else {
                continue;
            };
            let Some(additional_steps) = prompt_formal_training_steps("续训步数", 100) else {
                println!("续训表单已取消；没有启动任务。\n");
                continue;
            };
            let Some(checkpoint_every) = prompt_checkpoint_interval() else {
                println!("续训表单已取消；没有启动任务。\n");
                continue;
            };
            let mode = source["inputs"]["training_mode"]
                .as_str()
                .or_else(|| source["record"]["training_mode"].as_str())
                .unwrap_or("unknown");
            let reset_scope = if mode == "external-loop" {
                "用户循环状态"
            } else {
                "批次工厂或 training_step 自定义状态"
            };
            let allow_state_reset = prompt(&format!(
                "若 {reset_scope} 没有可恢复 hooks，是否接受只重置该自定义状态？模型、优化器与随机状态仍从 checkpoint 恢复。[y/N]"
            ))
            .is_some_and(|answer| {
                matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
            });
            let command = Command::Train {
                inputs: TrainingInputs::default(),
                resume: Some(resume),
                if_checkpoint_sha256: None,
                migrate_from: None,
                migration_plan: None,
                name: prompt_run_name(),
                request_id: None,
                training_mode: None,
                step: None,
                training_loop: None,
                steps: None,
                additional_steps: Some(additional_steps),
                checkpoint_every,
                lr: None,
                weight_decay: None,
                apply_supervision: false,
                params: Vec::new(),
                loop_params: Vec::new(),
                allow_state_reset,
                detach: prompt_detach(),
            };
            run_training_with_plan(app, cli, &command);
            continue;
        }
        if !("1"..="9").contains(&choice.as_str())
            && !("12"..="14").contains(&choice.as_str())
            && !matches!(choice.as_str(), "23" | "25" | "28" | "29" | "36" | "38")
        {
            continue;
        }
        let choosing_training_mode = matches!(choice.as_str(), "13" | "14");
        if !choosing_training_mode
            && !ensure_model(&mut session.model, &mut session.model_source, cli)
        {
            continue;
        }
        if choice == "2" && !ensure_batch(&mut session.batch, true) {
            println!("前向试跑需要一个批次入口。");
            continue;
        }
        if choice == "12" && !ensure_batch(&mut session.batch, true) {
            println!("此操作需要一个批次入口。");
            continue;
        }
        if matches!(
            choice.as_str(),
            "4" | "5" | "6" | "7" | "23" | "25" | "28" | "29" | "36" | "38"
        ) {
            let _ = ensure_batch(&mut session.batch, false);
        }
        let inputs = input_snapshot(&session.model, &session.batch);
        let mut layer_report = if matches!(
            choice.as_str(),
            "3" | "4" | "5" | "6" | "7" | "23" | "25" | "28" | "29" | "36" | "38"
        ) {
            let Some(report) = interactive_layer_report(app, cli, &session.model, &session.batch)
            else {
                continue;
            };
            Some(report)
        } else {
            None
        };
        let mut layer_rows = layer_report
            .as_ref()
            .and_then(|report| report["modules"].as_array())
            .cloned()
            .unwrap_or_default();
        if choice == "38" {
            let shared_rows = layer_rows
                .iter()
                .filter(|row| {
                    row["aliases"]
                        .as_array()
                        .is_some_and(|aliases| aliases.len() > 1)
                })
                .cloned()
                .collect::<Vec<_>>();
            if shared_rows.is_empty() {
                println!("当前前向可见的模型树没有共享 Layer；未提交变更。\n");
                continue;
            }
            let Some(selected) =
                prompt_layer_path(&shared_rows, "选择要调整共享关系的 Layer", None)
            else {
                println!("共享关系编辑已取消。\n");
                continue;
            };
            let selected_row = shared_rows
                .iter()
                .find(|row| row["path"].as_str() == Some(selected.as_str()));
            let aliases = selected_row
                .and_then(|row| row["aliases"].as_array())
                .map(|values| values.iter().filter_map(Value::as_str).collect::<Vec<_>>())
                .unwrap_or_default();
            if aliases.len() < 2 {
                println!("共享别名证据已变化；请刷新模型树后重试。\n");
                continue;
            }
            println!("已观察到的共享路径：{}", aliases.join("、"));
            println!(
                "clone 会复制参数/buffers并拆分别名；share 会用源模块替换目标模块及其原参数/buffers。共有连接语义仍由模型代码决定。\n"
            );
            let Some(mode) = prompt("输入 clone 拆分一个路径，或 share 将目标路径绑定到源路径")
            else {
                continue;
            };
            let operation = match mode.trim() {
                "clone" => {
                    let Some(path) = prompt("输入要拆分的完整共享路径") else {
                        continue;
                    };
                    if !aliases.contains(&path.as_str()) {
                        println!("该路径不在当前共享别名快照中；未提交变更。\n");
                        continue;
                    }
                    serde_json::json!({"action":"sharing","mode":"clone-selected","path":path})
                }
                "share" => {
                    let Some(source) = prompt("输入保留的源模块完整路径") else {
                        continue;
                    };
                    if !aliases.contains(&source.as_str()) {
                        println!("源路径不在当前共享别名快照中；未提交变更。\n");
                        continue;
                    }
                    let Some(targets) = prompt("输入目标完整路径，以逗号分隔") else {
                        continue;
                    };
                    let targets = targets
                        .split(',')
                        .map(str::trim)
                        .filter(|path| !path.is_empty())
                        .map(str::to_owned)
                        .collect::<Vec<_>>();
                    let observed_paths = layer_rows
                        .iter()
                        .filter_map(|row| row["path"].as_str())
                        .collect::<Vec<_>>();
                    if targets.is_empty()
                        || targets
                            .iter()
                            .any(|path| !observed_paths.contains(&path.as_str()))
                    {
                        println!("目标必须是当前模型树中可见的模块路径；未提交变更。\n");
                        continue;
                    }
                    serde_json::json!({"action":"sharing","mode":"share-selected","source":source,"targets":targets})
                }
                _ => {
                    println!("请输入 clone 或 share；未提交变更。\n");
                    continue;
                }
            };
            let operation =
                serde_json::to_string(&operation).expect("sharing operation serializes");
            let make_command = |dry_run| Command::Layer {
                command: LayerCommand::Apply {
                    operations: vec![operation.clone()],
                    edit: EditOptions {
                variant_device_policy: None,
                layer_rng_devices: Vec::new(),
                base_rng_scope: None,
                base_rng_devices: Vec::new(),
                        inputs: inputs.clone(),
                        target: "variant".into(),
                        params: Vec::new(),
                        dry_run,
                        require_forward: false,
                    },
                },
            };
            let preview = execute_ui(app, cli, &make_command(true));
            display(&preview, false);
            if preview.success()
                && prompt("确认应用以上共享关系变更到当前 variant？输入 yes 确认")
                    .is_some_and(|answer| matches!(answer.as_str(), "yes" | "y"))
            {
                display(&execute_ui(app, cli, &make_command(false)), false);
            } else if preview.success() {
                println!("已取消；variant 未修改。\n");
            }
            continue;
        }
        if choice == "36" {
            let target = cli.variant.as_deref().unwrap_or("working");
            println!(
                "\n{}",
                render_layer_edit_context(
                    session.model.as_deref(),
                    session.batch.as_deref(),
                    Some(target)
                )
                .join("\n")
            );
            let mut operations = Vec::<Value>::new();
            loop {
                println!(
                    "\n{}",
                    render_edit_plan(&format!("variant:{target}"), &operations, None).join("\n")
                );
                let Some(action) = prompt_allow_empty(
                    "计划操作：add / undo（撤销最后一项）/ preview / clear / back",
                ) else {
                    break;
                };
                match action.trim().to_ascii_lowercase().as_str() {
                    "add" => {
                        if operations.len() >= 64 {
                            println!("已达到一次原子变体计划最多 64 项操作的上限。");
                            continue;
                        }
                        let Some(path) =
                            prompt_editable_layer_path(&layer_rows, "要修改哪个 Layer")
                        else {
                            continue;
                        };
                        let Some(row) = layer_rows
                            .iter()
                            .find(|row| row["path"].as_str() == Some(path.as_str()))
                        else {
                            println!("所选 Layer 已不在当前树快照中；请刷新模型树后重试。");
                            continue;
                        };
                        if !layer_can_patch(row) {
                            println!("此 Layer 没有已声明的参数编辑能力；没有加入计划。");
                            continue;
                        }
                        let Some(params) = prompt_layer_patch(row) else {
                            println!("Layer 参数表单已取消；草稿保持不变。");
                            continue;
                        };
                        if params.is_empty() {
                            println!("没有修改字段；草稿保持不变。");
                            continue;
                        }
                        let params = params
                            .into_iter()
                            .map(|parameter| {
                                let (name, value) = parameter
                                    .split_once('=')
                                    .expect("Layer patch fields contain a name and JSON value");
                                (
                                    name.to_owned(),
                                    serde_json::from_str(value)
                                        .expect("Layer patch values were validated as JSON"),
                                )
                            })
                            .collect::<serde_json::Map<String, Value>>();
                        operations.push(serde_json::json!({
                            "action":"set",
                            "path":path,
                            "params":params
                        }));
                        if operations.len() == 64 {
                            println!("已达到一次原子变体计划最多 64 项操作的上限。");
                        }
                    }
                    "undo" => {
                        if operations.pop().is_none() {
                            println!("草稿中没有可撤销的操作。");
                        }
                    }
                    "clear" => {
                        operations.clear();
                        println!("草稿已清空；当前变体未改变。");
                    }
                    "preview" => {
                        if operations.is_empty() {
                            println!("先添加至少一项 Layer 参数修改。");
                            continue;
                        }
                        let Some(policy_choice) = prompt_allow_empty(
                            "变体设备策略：留空沿用已有策略（新变体基础模型及新增层保留工厂设备） / factory / cpu；替换层参考旧层首个状态",
                        ) else { continue; };
                        let variant_device_policy = match policy_choice.trim().to_ascii_lowercase().as_str() {
                            "" => None,
                            "factory" => Some("factory".to_owned()),
                            "cpu" => Some("legacy-cpu".to_owned()),
                            _ => { println!("请输入 factory、cpu 或留空；未开始预览。"); continue; }
                        };
                        let command = Command::Layer {
                            command: LayerCommand::Apply {
                                operations: operations
                                    .iter()
                                    .map(|operation| {
                                        serde_json::to_string(operation)
                                            .expect("Layer operations serialize")
                                    })
                                    .collect(),
                                edit: EditOptions {
                                    variant_device_policy,
                                    layer_rng_devices: Vec::new(),
                base_rng_scope: None,
                base_rng_devices: Vec::new(),
                                    inputs: inputs.clone(),
                                    target: "variant".into(),
                                    params: Vec::new(),
                                    dry_run: true,
                                    require_forward: false,
                                },
                            },
                        };
                        let preview = execute_ui(app, cli, &command);
                        let preview_value = serde_json::to_value(&preview).unwrap_or(Value::Null);
                        println!(
                            "\n{}",
                            render_edit_plan(
                                &format!("variant:{target}"),
                                &operations,
                                Some(&preview_value)
                            )
                            .join("\n")
                        );
                        println!("\n{}", render_edit_diff(&preview_value).join("\n"));
                        display(&preview, false);
                        if !preview.success() {
                            println!(
                                "前向或候选检查失败；草稿未提交，可继续添加修复、撤销最后一项或放弃。"
                            );
                            continue;
                        }
                        let Some(preview_seed) = preview.data["variant"]["seed"].as_u64() else {
                            println!("候选没有返回可重建的初始化种子；未提交，请检查 worker 版本。\n");
                            continue;
                        };
                        let base = &preview.data["variant_base"];
                        let Some(base_token) = base["token"].as_str().filter(|token| {
                            *token == "absent" || (token.len() == 64 && token.bytes().all(|byte|
                                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
                        }) else {
                            println!("候选未返回可核对的变体基准；未提交，请重新预览。\n");
                            continue;
                        };
                        if base["name"].as_str() != Some(target)
                            || base["algorithm"] != "canonical-json-sha256-v1" {
                            println!("候选变体基准与当前目标不一致；未提交。\n");
                            continue;
                        }
                        println!("候选初始化种子：{preview_seed}；提交时将沿用该种子。外部状态仍可能变化。");
                        println!("变体设备策略：{}；工厂设备策略不会自动迁移输入。",
                            preview.data["variant"]["device_policy"].as_str().unwrap_or("legacy_cpu"));
                        if !prompt(
                            "候选检查通过。提交以上完整计划为一个可撤销变体版本？输入 yes 确认",
                        )
                        .is_some_and(|answer| {
                            matches!(answer.trim().to_ascii_lowercase().as_str(), "yes" | "y")
                        }) {
                            println!("未提交；草稿仍可修改或预览。\n");
                            continue;
                        }
                        let committed = match command {
                            Command::Layer {
                                command:
                                    LayerCommand::Apply {
                                        operations,
                                        mut edit,
                                    },
                            } => {
                                edit.dry_run = false;
                                Command::Layer {
                                    command: LayerCommand::Apply { operations, edit },
                                }
                            }
                            _ => unreachable!("variant plan constructs a Layer Apply command"),
                        };
                        let mut commit_cli = cli.clone();
                        commit_cli.seed = Some(preview_seed);
                        commit_cli.if_variant_base = Some(base_token.to_owned());
                        let result = execute_ui(app, &commit_cli, &committed);
                        let committed_successfully = result.success();
                        display(&result, false);
                        if committed_successfully {
                            break;
                        }
                        println!("提交未完成；当前变体保持原样，草稿仍在此对话中。\n");
                    }
                    "back" | "b" | "返回" | "q" => {
                        println!("已退出编辑对话；变体未改变。\n");
                        break;
                    }
                    "" => {}
                    _ => println!("请输入 add、undo、preview、clear 或 back。"),
                }
            }
            continue;
        }
        if choice == "3" {
            loop {
                let cached_probe = probe_cache
                    .as_ref()
                    .filter(|cache| cache.matches(&session.model, &session.batch, cli))
                    .map(|cache| &cache.report);
                if !browse_model_layers(
                    &layer_rows,
                    layer_report
                        .as_ref()
                        .expect("report loaded for model browser"),
                    cached_probe,
                ) {
                    break;
                }
                let Some(fresh_report) =
                    interactive_layer_report(app, cli, &session.model, &session.batch)
                else {
                    break;
                };
                layer_rows = fresh_report["modules"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                layer_report = Some(fresh_report);
                probe_cache = None;
                println!("模型树快照已刷新；原选中节点已清除，避免路径重编号后误选。\n");
            }
            continue;
        }
        if choice == "28" {
            if cli.variant.is_some() {
                println!(
                    "源码编辑与 --variant 互斥；请不带 --variant 重新打开工作台。变体编辑不会改动原文件。"
                );
                continue;
            }
            let mut operations = Vec::new();
            while let Some(path) = prompt_editable_layer_path(&layer_rows, "要修改哪个 Layer")
            {
                let Some(params) = prompt_interactive_params(&format!("source.layer.{path}"))
                else {
                    println!("参数表单已取消；源码未修改。\n");
                    continue;
                };
                if params.is_empty() {
                    println!("至少填写一个参数；请重新输入 Layer 路径。源码未修改。");
                    continue;
                }
                operations.push(
                    serde_json::to_string(&serde_json::json!({
                        "action": "set",
                        "path": path,
                        "params": params,
                    }))
                    .expect("JSON operation serializes"),
                );
                if operations.len() == 64 {
                    println!("已达到一次事务最多 64 项 set 操作的上限。");
                    break;
                }
                if !prompt("继续添加另一个 Layer？输入 yes 继续，留空预览当前计划")
                    .is_some_and(|answer| matches!(answer.as_str(), "yes" | "y"))
                {
                    break;
                }
            }
            if operations.is_empty() {
                println!("没有有效操作；源码未修改。");
                continue;
            }
            preview_confirm_source_plan(
                app,
                cli,
                inputs.clone(),
                operations,
                "确认按以上完整计划写入源码？输入 yes 确认",
            );
            continue;
        }
        if choice == "29" {
            if cli.variant.is_some() {
                println!(
                    "源码编辑与 --variant 互斥；请不带 --variant 重新打开工作台。变体编辑不会改动原文件。"
                );
                continue;
            }
            let mut operations = Vec::new();
            while operations.len() < 64 {
                let Some(action) = prompt("计划操作：add / replace / remove") else {
                    break;
                };
                let operation = match action.as_str() {
                    "add" => {
                        let Some(parent) = prompt_source_insertable_path(
                            &layer_rows,
                            "选择 Sequential/ModuleList/ModuleDict 父容器",
                        ) else {
                            println!("缺少父节点；此项未加入计划。");
                            continue;
                        };
                        let position = prompt("插入位置：before / after / append（默认 append）");
                        let (before, after) = match position.as_deref() {
                            Some("before") => {
                                let Some(anchor) =
                                    prompt(container_anchor_prompt(&layer_rows, &parent, true))
                                else {
                                    println!("缺少定位子层；此项未加入计划。");
                                    continue;
                                };
                                (Some(anchor), None)
                            }
                            Some("after") => {
                                let Some(anchor) =
                                    prompt(container_anchor_prompt(&layer_rows, &parent, false))
                                else {
                                    println!("缺少定位子层；此项未加入计划。");
                                    continue;
                                };
                                (None, Some(anchor))
                            }
                            None | Some("append") => (None, None),
                            _ => {
                                println!("位置请输入 before、after 或 append；此项未加入计划。");
                                continue;
                            }
                        };
                        let Some(name) = prompt_container_add_name(&layer_rows, &parent) else {
                            println!("缺少 Layer 名称；此项未加入计划。");
                            continue;
                        };
                        let Some((layer_type, mut component)) = prompt_layer_kind() else {
                            println!("Layer 类型无效或已取消；此项未加入计划。");
                            continue;
                        };
                        let Some(params) = prompt_layer_parameters(
                            app,
                            cli,
                            &mut component,
                            layer_type.as_deref(),
                        ) else {
                            println!("参数输入已取消；此项未加入计划。");
                            continue;
                        };
                        serde_json::json!({
                            "action":"add", "parent":parent, "before":before,
                            "after":after, "name":name, "layer_type":layer_type,
                            "component":component,
                            "params":params
                        })
                    }
                    "replace" => {
                        let Some(path) =
                            prompt_editable_layer_path(&layer_rows, "要替换哪个 Layer")
                        else {
                            println!("缺少 Layer 路径；此项未加入计划。");
                            continue;
                        };
                        let Some((layer_type, mut component)) = prompt_layer_kind() else {
                            println!("Layer 类型无效或已取消；此项未加入计划。");
                            continue;
                        };
                        let Some(params) = prompt_layer_parameters(
                            app,
                            cli,
                            &mut component,
                            layer_type.as_deref(),
                        ) else {
                            println!("参数输入已取消；此项未加入计划。");
                            continue;
                        };
                        serde_json::json!({
                            "action":"replace", "path":path,
                            "layer_type":layer_type, "component":component,
                            "params":params
                        })
                    }
                    "remove" => {
                        let Some(path) =
                            prompt_editable_layer_path(&layer_rows, "要删除哪个 Layer")
                        else {
                            println!("缺少 Layer 路径；此项未加入计划。");
                            continue;
                        };
                        let strategy = prompt("策略 remove 或 bypass Identity（默认 remove）")
                            .unwrap_or_else(|| "remove".into());
                        serde_json::json!({"action":"remove", "path":path, "strategy":strategy})
                    }
                    _ => {
                        println!("操作请输入 add、replace 或 remove；源码未修改。");
                        continue;
                    }
                };
                let refresh_tree = source_operation_changes_tree(&operation);
                operations
                    .push(serde_json::to_string(&operation).expect("JSON operation serializes"));
                if refresh_tree {
                    if let Some(updated_rows) =
                        refresh_source_plan_rows(app, cli, &inputs, &operations)
                    {
                        layer_rows = updated_rows;
                        println!("已刷新计划模型树；后续操作按候选结构和索引选择。");
                    } else {
                        operations.pop();
                        println!("该操作未加入计划；当前模型树仍按已保留的计划显示。");
                        continue;
                    }
                }
                if operations.len() == 64 {
                    println!("已达到一次事务最多 64 项结构 CRUD 操作的上限。");
                    break;
                }
                if !prompt("继续添加结构操作？输入 yes 继续，留空预览当前计划")
                    .is_some_and(|answer| matches!(answer.as_str(), "yes" | "y"))
                {
                    break;
                }
            }
            if operations.is_empty() {
                println!("没有有效操作；源码未修改。");
                continue;
            }
            preview_confirm_source_plan(
                app,
                cli,
                inputs.clone(),
                operations,
                "确认按以上完整结构计划写入源码？输入 yes 确认",
            );
            continue;
        }
        if choice == "25" {
            if cli.variant.is_some() {
                println!(
                    "源码编辑与 --variant 互斥；请不带 --variant 重新打开工作台。变体编辑不会改动原文件。"
                );
                continue;
            }
            let Some(action) = prompt("源码操作：add / set / replace / remove") else {
                continue;
            };
            let (path, parent, before, after, name, layer_type, component, strategy, params) =
                match action.as_str() {
                    "set" => {
                        let Some(path) =
                            prompt_editable_layer_path(&layer_rows, "要修改哪个 Layer")
                        else {
                            continue;
                        };
                        let Some(params) = prompt_parameter_map(&format!("source.set.{path}"))
                        else {
                            println!("参数表单已取消；源码未修改。\n");
                            continue;
                        };
                        (Some(path), None, None, None, None, None, None, None, params)
                    }
                    "add" => {
                        let Some(parent) = prompt_source_insertable_path(
                            &layer_rows,
                            "选择 Sequential/ModuleList/ModuleDict 父容器",
                        ) else {
                            continue;
                        };
                        let position = prompt("插入位置：before / after / append（默认 append）");
                        let (before, after) = match position.as_deref() {
                            Some("before") => (
                                prompt(container_anchor_prompt(&layer_rows, &parent, true)),
                                None,
                            ),
                            Some("after") => (
                                None,
                                prompt(container_anchor_prompt(&layer_rows, &parent, false)),
                            ),
                            None | Some("append") => (None, None),
                            _ => {
                                println!("位置请输入 before、after 或 append。文件未修改。");
                                continue;
                            }
                        };
                        let Some(name) = prompt_container_add_name(&layer_rows, &parent) else {
                            continue;
                        };
                        let Some((layer_type, mut component)) = prompt_layer_kind() else {
                            println!("Layer 类型无效或已取消；文件未修改。");
                            continue;
                        };
                        let Some(params) = prompt_layer_parameters(
                            app,
                            cli,
                            &mut component,
                            layer_type.as_deref(),
                        ) else {
                            println!("参数输入已取消；文件未修改。");
                            continue;
                        };
                        (
                            None,
                            Some(parent),
                            before,
                            after,
                            Some(name),
                            layer_type,
                            component,
                            None,
                            params,
                        )
                    }
                    "replace" => {
                        let Some(path) =
                            prompt_editable_layer_path(&layer_rows, "要替换哪个 Layer")
                        else {
                            continue;
                        };
                        let Some((layer_type, mut component)) = prompt_layer_kind() else {
                            println!("Layer 类型无效或已取消；文件未修改。");
                            continue;
                        };
                        let Some(params) = prompt_layer_parameters(
                            app,
                            cli,
                            &mut component,
                            layer_type.as_deref(),
                        ) else {
                            println!("参数输入已取消；文件未修改。");
                            continue;
                        };
                        (
                            Some(path),
                            None,
                            None,
                            None,
                            None,
                            layer_type,
                            component,
                            None,
                            params,
                        )
                    }
                    "remove" => {
                        let Some(path) =
                            prompt_editable_layer_path(&layer_rows, "要删除哪个 Layer")
                        else {
                            continue;
                        };
                        let strategy = prompt("策略 remove 或 bypass Identity（默认 remove）")
                            .unwrap_or_else(|| "remove".into());
                        (
                            Some(path),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            Some(strategy),
                            serde_json::Map::new(),
                        )
                    }
                    _ => {
                        println!("未知源码操作；文件未修改。");
                        continue;
                    }
                };
            let make_command = |dry_run| {
                let edit = EditOptions {
                variant_device_policy: None,
                layer_rng_devices: Vec::new(),
                base_rng_scope: None,
                base_rng_devices: Vec::new(),
                    inputs: inputs.clone(),
                    target: "source".into(),
                    params: parameters_for_cli(&params),
                    dry_run,
                    require_forward: false,
                };
                Command::Layer {
                    command: match action.as_str() {
                        "set" => LayerCommand::Set {
                            path: path.clone().expect("set path was collected"),
                            edit,
                        },
                        "add" => LayerCommand::Add {
                            parent: parent.clone().expect("add parent was collected"),
                            before: before.clone(),
                            after: after.clone(),
                            name: name.clone().expect("add name was collected"),
                            layer_type: layer_type.clone(),
                            component: component.clone(),
                            edit,
                        },
                        "replace" => LayerCommand::Replace {
                            path: path.clone().expect("replace path was collected"),
                            layer_type: layer_type.clone(),
                            component: component.clone(),
                            edit,
                        },
                        "remove" => LayerCommand::Remove {
                            path: path.clone().expect("remove path was collected"),
                            strategy: strategy.clone().expect("remove strategy was collected"),
                            edit,
                        },
                        _ => unreachable!(),
                    },
                }
            };
            let preview = execute_ui(app, cli, &make_command(true));
            display(&preview, false);
            if preview.success()
                && prompt("确认按以上差异写入 Python 源码？输入 yes 确认")
                    .is_some_and(|answer| matches!(answer.as_str(), "yes" | "y"))
            {
                display(&execute_ui(app, cli, &make_command(false)), false);
            } else if preview.success() {
                println!("已取消；原文件未修改。");
            }
            continue;
        }
        let command = match choice.as_str() {
            "1" => Command::Model {
                command: ModelCommand::Inspect { path: None, inputs },
            },
            "2" => Command::Model {
                command: {
                    let Some(path) =
                        prompt_allow_empty("可选：模型端口契约 JSON 文件（Enter 跳过）")
                    else {
                        continue;
                    };
                    let Some(module_path) = prompt_allow_empty(
                        "可选：Layer 路径（局部隔离重放；Enter 只检查完整模型）",
                    ) else {
                        continue;
                    };
                    let module_path = module_path.trim();
                    let Some(fingerprint) = prompt_allow_empty(
                        "可选：数据/切分指纹（Dataset 提供稳定 sample_id 时用于验证缓存；Enter 跳过）",
                    ) else {
                        continue;
                    };
                    ModelCommand::Probe {
                        path: (!module_path.is_empty()).then(|| module_path.to_owned()),
                        inputs,
                        call: "all".into(),
                        contract: (!path.trim().is_empty()).then(|| PathBuf::from(path.trim())),
                        if_contract_sha256: None,
                        batch_fingerprint: (!fingerprint.trim().is_empty())
                            .then(|| fingerprint.trim().to_owned()),
                        model_device: None,
                        local: !module_path.is_empty(),
                    }
                },
            },
            "3" => unreachable!("Layer browser is handled before command construction"),
            "12" => {
                let Some(loss) = prompt("损失函数入口（outputs, batch -> scalar Tensor）")
                else {
                    continue;
                };
                Command::Model {
                    command: ModelCommand::Verify { loss, inputs },
                }
            }
            "13" => {
                let Some(mode) = prompt_training_mode() else {
                    println!("已取消训练试跑表单；没有启动任务。\n");
                    continue;
                };
                if mode == TrainingMode::ExternalLoop {
                    let Some(training_loop) =
                        prompt("完整循环入口（module:run_loop；model, context, **参数 -> 结果）")
                    else {
                        continue;
                    };
                    let mut loop_inputs = inputs;
                    loop_inputs.batch = None;
                    loop_inputs.batch_params.clear();
                    let Some(steps) = prompt_training_steps() else {
                        println!("试跑表单已取消；没有启动任务。\n");
                        continue;
                    };
                    let Some(loop_params) = prompt_external_loop_params() else {
                        println!("循环参数表单已取消；没有启动任务。\n");
                        continue;
                    };
                    Command::Try {
                        inputs: loop_inputs.into(),
                        repeat: None,
                        name: prompt_run_name(),
                        training_mode: Some(mode),
                        step: None,
                        training_loop: Some(training_loop),
                        steps: Some(steps),
                        lr: None,
                        weight_decay: None,
                        apply_supervision: false,
                        no_supervision: false,
                        params: Vec::new(),
                        loop_params,
                    }
                } else {
                    if !ensure_model(&mut session.model, &mut session.model_source, cli) {
                        continue;
                    }
                    if !ensure_batch(&mut session.batch, true) {
                        println!(
                            "managed/manual-step 试跑需要批次入口。external-loop 可自己管理数据。"
                        );
                        continue;
                    }
                    let Some(step) = prompt("训练步骤入口（model, batch, context -> dict）")
                    else {
                        continue;
                    };
                    let Some(steps) = prompt_training_steps() else {
                        println!("试跑表单已取消；没有启动任务。\n");
                        continue;
                    };
                    let Some(params) = prompt_params(&format!("try.step.{step}")) else {
                        println!("训练参数表单已取消；没有启动任务。\n");
                        continue;
                    };
                    Command::Try {
                        inputs: input_snapshot(&session.model, &session.batch).into(),
                        repeat: None,
                        name: prompt_run_name(),
                        training_mode: Some(mode),
                        step: Some(step),
                        training_loop: None,
                        steps: Some(steps),
                        lr: Some(1e-4),
                        weight_decay: Some(0.01),
                        apply_supervision: false,
                        no_supervision: false,
                        params,
                        loop_params: Vec::new(),
                    }
                }
            }
            "14" => {
                let Some(mode) = prompt_training_mode() else {
                    println!("已取消正式训练表单；没有启动任务。\n");
                    continue;
                };
                if mode == TrainingMode::ExternalLoop {
                    let Some(training_loop) =
                        prompt("完整循环入口（module:run_loop；model, context, **参数 -> 结果）")
                    else {
                        continue;
                    };
                    let mut loop_inputs = inputs;
                    loop_inputs.batch = None;
                    loop_inputs.batch_params.clear();
                    let Some(steps) = prompt_formal_training_steps("正式训练总步骤预算", 100)
                    else {
                        println!("训练表单已取消；没有启动任务。\n");
                        continue;
                    };
                    let Some(checkpoint_every) = prompt_checkpoint_interval() else {
                        println!("训练表单已取消；没有启动任务。\n");
                        continue;
                    };
                    let Some(loop_params) = prompt_external_loop_params() else {
                        println!("循环参数表单已取消；没有启动任务。\n");
                        continue;
                    };
                    Command::Train {
                        inputs: loop_inputs.into(),
                        resume: None,
                        if_checkpoint_sha256: None,
                        migrate_from: None,
                        migration_plan: None,
                        name: prompt_run_name(),
                        request_id: None,
                        training_mode: Some(mode),
                        step: None,
                        training_loop: Some(training_loop),
                        steps: Some(steps),
                        additional_steps: None,
                        checkpoint_every,
                        lr: None,
                        weight_decay: None,
                        apply_supervision: false,
                        params: Vec::new(),
                        loop_params,
                        allow_state_reset: false,
                        detach: prompt_detach(),
                    }
                } else {
                    if !ensure_model(&mut session.model, &mut session.model_source, cli) {
                        continue;
                    }
                    if !ensure_batch(&mut session.batch, true) {
                        println!(
                            "managed/manual-step 正式训练需要批次入口。external-loop 可自己管理数据。"
                        );
                        continue;
                    }
                    let Some(step) = prompt("训练步骤入口（model, batch, context -> dict）")
                    else {
                        continue;
                    };
                    let Some(steps) = prompt_formal_training_steps("正式训练总步数", 100)
                    else {
                        println!("训练表单已取消；没有启动任务。\n");
                        continue;
                    };
                    let Some(checkpoint_every) = prompt_checkpoint_interval() else {
                        println!("训练表单已取消；没有启动任务。\n");
                        continue;
                    };
                    let Some(lr) = prompt_positive_float("AdamW 学习率", 1e-4, false) else {
                        println!("训练表单已取消；没有启动任务。\n");
                        continue;
                    };
                    let Some(weight_decay) =
                        prompt_positive_float("AdamW weight decay", 0.01, true)
                    else {
                        println!("训练表单已取消；没有启动任务。\n");
                        continue;
                    };
                    let Some(params) = prompt_params(&format!("train.step.{step}")) else {
                        println!("训练参数表单已取消；没有启动任务。\n");
                        continue;
                    };
                    Command::Train {
                        inputs: input_snapshot(&session.model, &session.batch).into(),
                        resume: None,
                        if_checkpoint_sha256: None,
                        migrate_from: None,
                        migration_plan: None,
                        name: prompt_run_name(),
                        request_id: None,
                        training_mode: Some(mode),
                        step: Some(step),
                        training_loop: None,
                        steps: Some(steps),
                        additional_steps: None,
                        checkpoint_every,
                        lr: Some(lr),
                        weight_decay: Some(weight_decay),
                        apply_supervision: false,
                        params,
                        loop_params: Vec::new(),
                        allow_state_reset: false,
                        detach: prompt_detach(),
                    }
                }
            }
            "4" => {
                let Some(parent) = prompt_addable_container_path(
                    &layer_rows,
                    "选择 Sequential/ModuleList/ModuleDict 父容器",
                ) else {
                    continue;
                };
                let after = if container_uses_indices(&layer_rows, &parent) {
                    prompt("ModuleList 子层索引（如 0；留空表示末尾）")
                } else {
                    prompt("Sequential 子层名（如 relu；留空表示末尾）")
                };
                let Some(name) = prompt_container_add_name(&layer_rows, &parent) else {
                    continue;
                };
                let Some((layer_type, mut component)) = prompt_layer_kind() else {
                    println!("类型无效或已取消。");
                    continue;
                };
                let Some(params) =
                    prompt_layer_parameters(app, cli, &mut component, layer_type.as_deref())
                else {
                    println!("参数输入已取消。");
                    continue;
                };
                Command::Layer {
                    command: LayerCommand::Add {
                        parent,
                        before: None,
                        after,
                        name,
                        layer_type,
                        component,
                        edit: EditOptions {
                variant_device_policy: None,
                layer_rng_devices: Vec::new(),
                base_rng_scope: None,
                base_rng_devices: Vec::new(),
                            inputs,
                            target: "variant".into(),
                            params: parameters_for_cli(&params),
                            dry_run: false,
                            require_forward: false,
                        },
                    },
                }
            }
            "5" => {
                let Some(path) = prompt_editable_layer_path(&layer_rows, "要修改哪个 Layer")
                else {
                    continue;
                };
                let Some(row) = layer_rows
                    .iter()
                    .find(|row| row["path"].as_str() == Some(path.as_str()))
                else {
                    println!("所选 Layer 已不在当前树快照中；请刷新模型树后重试。");
                    continue;
                };
                let Some(params) = prompt_layer_patch(row) else {
                    println!("参数表单已取消；没有提交 Layer 修改。\n");
                    continue;
                };
                if params.is_empty() {
                    println!("没有修改任何字段；变体未改变。\n");
                    continue;
                }
                Command::Layer {
                    command: LayerCommand::Set {
                        path,
                        edit: EditOptions {
                variant_device_policy: None,
                layer_rng_devices: Vec::new(),
                base_rng_scope: None,
                base_rng_devices: Vec::new(),
                            inputs,
                            target: "variant".into(),
                            params,
                            dry_run: false,
                            require_forward: false,
                        },
                    },
                }
            }
            "6" => {
                let Some(path) = prompt_editable_layer_path(&layer_rows, "要替换哪个 Layer")
                else {
                    continue;
                };
                let Some((layer_type, mut component)) = prompt_layer_kind() else {
                    println!("类型无效或已取消。");
                    continue;
                };
                let Some(params) =
                    prompt_layer_parameters(app, cli, &mut component, layer_type.as_deref())
                else {
                    println!("参数输入已取消。");
                    continue;
                };
                Command::Layer {
                    command: LayerCommand::Replace {
                        path,
                        layer_type,
                        component,
                        edit: EditOptions {
                variant_device_policy: None,
                layer_rng_devices: Vec::new(),
                base_rng_scope: None,
                base_rng_devices: Vec::new(),
                            inputs,
                            target: "variant".into(),
                            params: parameters_for_cli(&params),
                            dry_run: false,
                            require_forward: false,
                        },
                    },
                }
            }
            "7" => {
                let Some(path) = prompt_editable_layer_path(&layer_rows, "要删除哪个 Layer")
                else {
                    continue;
                };
                let strategy = prompt("策略 remove 或 bypass Identity（默认 remove）")
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "remove".into());
                Command::Layer {
                    command: LayerCommand::Remove {
                        path,
                        strategy,
                        edit: EditOptions {
                variant_device_policy: None,
                layer_rng_devices: Vec::new(),
                base_rng_scope: None,
                base_rng_devices: Vec::new(),
                            inputs,
                            target: "variant".into(),
                            params: Vec::new(),
                            dry_run: false,
                            require_forward: false,
                        },
                    },
                }
            }
            "23" => {
                let Some(provider) = prompt(
                    "Layer Edit Provider（组件 namespace/name[@version] 或 Python module:callable）",
                ) else {
                    continue;
                };
                let mut selected_provider = provider.clone();
                let mut registered_schema = None;
                if !provider.contains(':') {
                    match std::env::current_dir().and_then(|path| path.canonicalize()) {
                        Ok(root) => match ApplicationUiServices::new(app, cli)
                            .inspect_component(&root, &provider)
                        {
                            Ok(descriptor)
                                if descriptor["manifest"]["kind"] == "layer-edit-provider" =>
                            {
                                let manifest = &descriptor["manifest"];
                                let component_id =
                                    manifest["component_id"].as_str().unwrap_or(&provider);
                                let version =
                                    manifest["component_version"].as_str().unwrap_or("0.0.0");
                                selected_provider = format!("{component_id}@{version}");
                                println!(
                                    "{}@{} · {}\n按 Schema 填写参数；可选字段留空使用默认值或跳过。",
                                    component_id,
                                    version,
                                    manifest["description"].as_str().unwrap_or("")
                                );
                                registered_schema = Some((root, manifest["params_schema"].clone()));
                            }
                            Ok(descriptor) => {
                                println!(
                                    "组件 {} 的类型是 {}，不是 layer-edit-provider。",
                                    descriptor["manifest"]["component_id"]
                                        .as_str()
                                        .unwrap_or(&provider),
                                    descriptor["manifest"]["kind"].as_str().unwrap_or("未知")
                                );
                                continue;
                            }
                            Err(error) => println!("未找到注册 Provider Schema：{error}"),
                        },
                        Err(error) => println!("无法读取当前项目目录：{error}"),
                    }
                }
                let payload = if let Some((root, schema)) = registered_schema {
                    let Some(payload) = prompt_schema_object(
                        &schema,
                        &format!("layer-edit-provider.{selected_provider}"),
                        None,
                        FormMode::Full,
                    )
                    .map(Value::Object) else {
                        println!("Provider 表单已取消；没有执行编辑。");
                        continue;
                    };
                    let Some(values) = payload.as_object() else {
                        println!("Provider Schema 顶层必须是 object；没有执行编辑。");
                        continue;
                    };
                    match ApplicationUiServices::new(app, cli).resolve_layer_edit_provider(
                        &root,
                        &selected_provider,
                        values,
                    ) {
                        Ok(resolved) => resolved["resolved_payload"].clone(),
                        Err(error) => {
                            println!("Provider 参数校验失败：{error}\n没有执行编辑。");
                            continue;
                        }
                    }
                } else {
                    let Some(payload) =
                        prompt_parameter_map(&format!("layer_edit_provider.{selected_provider}"))
                    else {
                        println!("Provider 参数表单已取消；没有执行编辑。\n");
                        continue;
                    };
                    Value::Object(payload)
                };
                if !provider.contains(':') {
                    let Some(payload_values) = payload.as_object() else {
                        println!("Provider payload 必须是对象；没有执行编辑。\n");
                        continue;
                    };
                    let test_report = execute_ui(
                        app,
                        cli,
                        &Command::Component {
                            command: ComponentCommand::Test {
                                id: selected_provider.clone(),
                                params: parameters_for_cli(payload_values),
                                batch: None,
                                batch_params: Vec::new(),
                                batch_binding: None,
                            },
                        },
                    );
                    display(&test_report, false);
                    if !test_report.success()
                        || test_report.data["activation"]["status"] != "enabled_for_tested_payloads"
                    {
                        println!("Provider 没有通过本次 payload 的编辑验证；没有修改模型。\n");
                        continue;
                    }
                }
                Command::Layer {
                    command: LayerCommand::Custom {
                        provider: selected_provider,
                        payload: serde_json::to_string(&payload)
                            .expect("Provider payload serializes"),
                        edit: EditOptions {
                variant_device_policy: None,
                layer_rng_devices: Vec::new(),
                base_rng_scope: None,
                base_rng_devices: Vec::new(),
                            inputs,
                            target: "variant".into(),
                            params: Vec::new(),
                            dry_run: false,
                            require_forward: false,
                        },
                    },
                }
            }
            "8" => Command::Layer {
                command: LayerCommand::Undo {
                    inputs,
                    target: "variant".into(),
                    dry_run: false,
                    require_forward: false,
                },
            },
            "9" => Command::Layer {
                command: LayerCommand::Redo {
                    inputs,
                    target: "variant".into(),
                    dry_run: false,
                    require_forward: false,
                },
            },
            _ => continue,
        };
        if choice == "2" {
            if let Some(report) =
                run_probe_interaction(app, cli, &mut session.model, &mut session.batch, command)
            {
                probe_cache = Some(ProbeCache {
                    model: session.model.clone(),
                    batch: session.batch.clone(),
                    variant: cli.variant.clone(),
                    seed: cli.seed,
                    report,
                });
            }
            continue;
        }
        if matches!(choice.as_str(), "13" | "14") {
            run_training_with_plan(app, cli, &command);
            continue;
        }
        let report = execute_ui(app, cli, &command);
        display(&report, false);
    }
}

#[cfg(test)]
mod form_schema_tests {
    use super::{
        ProbeCache, builtin_layer_schema, dynamic_parameter_draft_identity,
        persist_dynamic_params_draft, repeat_trial_command, safe_dynamic_params,
    };
    use crate::cli::{Cli, Command};
    use crate::presentation::forms::schema::validate_value;
    use serde_json::json;

    #[test]
    fn built_in_layer_forms_validate_required_values_defaults_and_bounds() {
        let linear = builtin_layer_schema("Linear").unwrap();
        assert!(
            validate_value(
                &json!({"in_features": 4, "out_features": 8, "bias": true}),
                &linear,
                "layer",
                0
            )
            .is_ok()
        );
        assert!(validate_value(&json!({"in_features": 4}), &linear, "layer", 0).is_err());

        let dropout = builtin_layer_schema("Dropout").unwrap();
        assert!(validate_value(&json!({"p": 0.5}), &dropout, "layer", 0).is_ok());
        assert!(validate_value(&json!({"p": 1.1}), &dropout, "layer", 0).is_err());
        assert!(builtin_layer_schema("unknown").is_none());
    }

    #[test]
    fn cached_probe_is_bound_to_model_batch_variant_and_seed() {
        let cache = ProbeCache {
            model: Some("project:model".into()),
            batch: Some("project:batch".into()),
            variant: Some("working".into()),
            seed: Some(7),
            report: json!({}),
        };
        let cli = Cli {
            describe_interface: false,
            json: false,
            no_interactive: false,
            python: None,
            python_explicit: false,
            timeout: None,
            variant: Some("working".into()),
            if_variant_base: None,
            if_input_content: None,
            if_source_manifest: None,
            if_source_plan: None,
            if_source_config: None,
            capture_input_to: None,
            replay_input_from: None,
            seed: Some(7),
            verbose: false,
            command: None,
        };
        assert!(cache.matches(
            &Some("project:model".into()),
            &Some("project:batch".into()),
            &cli
        ));
        assert!(!cache.matches(
            &Some("project:model".into()),
            &Some("other:batch".into()),
            &cli
        ));
    }

    #[test]
    fn comparison_repeat_uses_only_the_selected_saved_run() {
        let Command::Try {
            inputs,
            repeat,
            steps,
            lr,
            ..
        } = repeat_trial_command("run-selected".into())
        else {
            panic!("expected trial repeat command")
        };
        assert_eq!(repeat.as_deref(), Some("run-selected"));
        assert!(inputs.model.is_none());
        assert!(inputs.model_params.is_empty());
        assert!(inputs.injections.is_empty());
        assert!(inputs.batch_params.is_empty());
        assert!(steps.is_none());
        assert!(lr.is_none());
    }

    #[test]
    fn dynamic_parameter_drafts_keep_only_untyped_nonsecret_values() {
        let params = serde_json::from_value(json!({
            "width": 32,
            "enabled": true,
            "sizes": [1, 2.5, false, null],
            "caption": "may contain a credential",
            "options": {"token": "secret"},
            "api_key": 1234
        }))
        .unwrap();
        assert_eq!(
            safe_dynamic_params(&params),
            serde_json::from_value(json!({
                "width": 32,
                "enabled": true,
                "sizes": [1, 2.5, false, null]
            }))
            .unwrap()
        );
    }

    #[test]
    fn dynamic_parameter_drafts_are_scoped_to_the_prompt_context() {
        let assembly_model = dynamic_parameter_draft_identity("assembly.model");
        let assembly_batch = dynamic_parameter_draft_identity("assembly.batch");
        assert_ne!(assembly_model.0, assembly_batch.0);
        assert_eq!(assembly_model.1, assembly_batch.1);
    }

    #[test]
    fn dynamic_parameter_draft_persistence_omits_unsafe_values() {
        let project = tempfile::tempdir().unwrap();
        let (form_id, schema_revision) = dynamic_parameter_draft_identity("source.layer.encoder");
        let params = serde_json::from_value(json!({
            "features": 64,
            "prompt": "user supplied text",
            "secret_token": "private",
            "shape": [1, 3, 224, 224]
        }))
        .unwrap();

        persist_dynamic_params_draft(Some(project.path()), &form_id, &schema_revision, &params);

        let draft = crate::presentation::session_store::load_form_draft(
            project.path(),
            &form_id,
            &schema_revision,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            draft.values,
            serde_json::from_value(json!({
                "features": 64,
                "shape": [1, 3, 224, 224]
            }))
            .unwrap()
        );
    }
}
