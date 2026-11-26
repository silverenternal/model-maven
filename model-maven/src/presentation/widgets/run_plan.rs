use crate::{
    cli::{Cli, Command, TrainingMode},
    presentation::pages::training::{mode_description, mode_label},
};
use serde_json::Value;

pub fn render(
    command: &Command,
    cli: &Cli,
    readiness: Option<&Value>,
    source_record: Option<&Value>,
) -> Vec<String> {
    let mut lines = vec!["训练计划 · 执行前预览".to_owned()];
    let mode = command_mode(command, source_record);
    if let Some(mode) = mode {
        lines.push(format!(
            "模式：{} · {}",
            mode_label(mode),
            mode_description(mode)
        ));
    } else {
        lines.push("模式：从来源运行继承；来源记录当前不可读取。".into());
    }
    if !matches!(command, Command::Try { .. } | Command::Train { .. }) {
        lines.push("当前命令不是训练操作。".into());
        return lines;
    }
    let (requested_policy, historical) = match command {
        Command::Try { inputs, repeat, .. } => (inputs.batch_device_policy.as_deref(), repeat.is_some()),
        Command::Train { inputs, resume, .. } => (inputs.batch_device_policy.as_deref(), resume.is_some()),
        _ => (None, false),
    };
    if mode == Some(TrainingMode::ExternalLoop) {
        lines.push("批次设备：由用户循环管理。".into());
    } else if let Some(policy) = requested_policy {
        lines.push(format!("批次设备策略：{policy}（显式选择；续训仍需与原记录一致）。"));
    } else if historical {
        let policy = match source_record {
            None => "来源记录不可用，未知",
            Some(record) if !record["inputs"].is_object() => "来源输入不可用，未知",
            Some(record) => match record["inputs"].get("batch_device_policy") {
                None | Some(Value::Null) => "历史递归行为（未记录策略）",
                Some(Value::String(value)) if matches!(value.as_str(), "preserve" | "recursive-strict") => value.as_str(),
                _ => "记录无效或不受支持",
            },
        };
        lines.push(format!("批次设备策略：{policy}。"));
    } else {
        lines.push("批次设备策略：recursive-strict（新运行默认；不透明对象由用户步骤处理）。".into());
    }

    if let Some(plan) = readiness {
        for key in ["model", "batch", "training_step", "training_loop"] {
            if let Some(binding) = plan["inputs"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|binding| binding["key"] == key)
            {
                let required = binding["required"] == true;
                let value = if binding["value"].is_null() && !required {
                    match key {
                        "batch" if mode == Some(TrainingMode::ExternalLoop) => {
                            "不需要；完整循环自行管理数据".into()
                        }
                        "model" if mode == Some(TrainingMode::ExternalLoop) => {
                            "未提供；循环可自行构造或不使用模型".into()
                        }
                        _ => "未提供（可选）".into(),
                    }
                } else {
                    display_value(binding)
                };
                let source = source_label(binding["source"]["kind"].as_str().unwrap_or("unknown"));
                let reference = binding["source"]["reference"]
                    .as_str()
                    .map(|reference| format!(" · {reference}"))
                    .unwrap_or_default();
                lines.push(format!(
                    "输入 · {}：{value} · {} · 来源：{source}{reference}",
                    input_label(key),
                    if required { "必需" } else { "可选" }
                ));
            }
        }
    }

    match command {
        Command::Try {
            repeat,
            steps,
            lr,
            weight_decay,
            no_supervision,
            params,
            loop_params,
            ..
        } => {
            lines.push(format!(
                "预算：{} 步（training_probe，不创建正式训练 checkpoint）",
                steps.map_or_else(|| "继承或待确认".into(), |steps| steps.to_string())
            ));
            if let Some(run_id) = repeat {
                lines.push(format!(
                    "来源运行：{run_id}；读取该运行保存的原始输入快照。"
                ));
                if *no_supervision {
                    lines.push("监督：本次明确关闭，不继承历史监督规则。".into());
                }
                lines.push(format!(
                    "本次显式覆盖：{}",
                    if params.is_empty()
                        && loop_params.is_empty()
                        && lr.is_none()
                        && weight_decay.is_none()
                    {
                        "无；其余输入从来源记录继承".into()
                    } else {
                        overrides(params, loop_params, *lr, *weight_decay)
                    }
                ));
            } else {
                lines.push(format!("步骤参数：{}", parameter_values(params)));
                lines.push(format!("循环参数：{}", parameter_values(loop_params)));
                if let Some(lr) = lr {
                    lines.push(format!("AdamW 学习率：{lr}"));
                }
                if let Some(weight_decay) = weight_decay {
                    lines.push(format!("AdamW weight decay：{weight_decay}"));
                }
            }
            lines.push("运行产物：试跑报告、指标与输入/源码证据；不保存正式训练权重。".into());
        }
        Command::Train {
            resume,
            steps,
            additional_steps,
            checkpoint_every,
            lr,
            weight_decay,
            params,
            loop_params,
            allow_state_reset,
            detach,
            ..
        } => {
            if let Some(run_id) = resume {
                lines.push(format!("续训父运行：{run_id}。"));
                lines.push(format!(
                    "预算：在已保存状态后增加 {} 步。",
                    additional_steps.map_or_else(|| "未指定".into(), |steps| steps.to_string())
                ));
                lines.push("继承：最近已验证 checkpoint 中的模型、优化器与随机状态；原运行的模式、入口及未覆盖训练参数。".into());
                let callable = match mode {
                    Some(TrainingMode::ExternalLoop) => "用户循环 state_dict/load_state_dict 状态",
                    Some(TrainingMode::ManagedStep | TrainingMode::ManualStep) => {
                        "批次工厂与 training_step 自定义状态"
                    }
                    None => "用户训练入口声明的自定义状态",
                };
                lines.push(format!(
                    "可能重置：若 {callable} 未提供可恢复状态，只有显式接受时才从初始状态运行；当前选择 {}。",
                    if *allow_state_reset { "接受重置" } else { "拒绝重置" }
                ));
            } else {
                lines.push(format!(
                    "预算：{} 总步数。",
                    steps.map_or_else(|| "未指定".into(), |steps| steps.to_string())
                ));
                lines.push("本次运行从初始训练状态开始。".into());
            }
            lines.push(format!("Checkpoint 间隔：{checkpoint_every} 步。"));
            if let Some(lr) = lr {
                lines.push(format!("AdamW 学习率：{lr}"));
            }
            if let Some(weight_decay) = weight_decay {
                lines.push(format!("AdamW weight decay：{weight_decay}"));
            }
            if resume.is_some() {
                lines.push("步骤/循环参数覆盖：无；继承父运行参数。".into());
            } else {
                lines.push(format!("步骤参数：{}", parameter_values(params)));
                lines.push(format!("循环参数：{}", parameter_values(loop_params)));
            }
            lines.push(format!(
                "执行方式：{}。",
                if *detach {
                    "独立后台监督；关闭工作台不会停止训练"
                } else {
                    "前台运行；终端中断语义由训练命令处理"
                }
            ));
            lines.push("运行产物：输入与源码快照、训练报告、进度事件和可恢复 checkpoint。".into());
        }
        _ => {}
    }

    let global_overrides = [
        cli.seed.map(|value| format!("seed={value}")),
        cli.python.as_ref().map(|value| format!("python={value}")),
        cli.timeout.map(|value| format!("timeout={value}s")),
        cli.variant.as_ref().map(|value| format!("variant={value}")),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    lines.push(format!(
        "命令行全局覆盖：{}",
        if global_overrides.is_empty() {
            "无".into()
        } else {
            global_overrides.join("、")
        }
    ));
    lines.push("显存预算估算：未接入；不能据此判断任务不会 OOM。".into());
    if let Some(plan) = readiness {
        lines.push(crate::presentation::terminal::format_readiness_summary(
            plan,
        ));
    } else {
        lines.push("运行前检查：就绪计划不可用；可运行的检查范围未知。".into());
    }
    lines
}

fn command_mode(command: &Command, source_record: Option<&Value>) -> Option<TrainingMode> {
    match command {
        Command::Try {
            training_mode,
            training_loop,
            step,
            ..
        } => training_mode.or_else(|| {
            let previous = source_training_mode(source_record);
            if training_loop.is_some() {
                Some(TrainingMode::ExternalLoop)
            } else if step.is_some() && previous == Some(TrainingMode::ExternalLoop) {
                Some(TrainingMode::ManagedStep)
            } else {
                previous
            }
        }),
        Command::Train {
            training_mode,
            training_loop,
            ..
        } => training_mode.or_else(|| if training_loop.is_some() {
            Some(TrainingMode::ExternalLoop)
        } else { source_training_mode(source_record) }),
        _ => None,
    }
}

fn source_training_mode(record: Option<&Value>) -> Option<TrainingMode> {
    record.and_then(|record| {
        record["inputs"]["training_mode"].as_str()
            .or_else(|| record["record"]["training_mode"].as_str())
    }).and_then(|mode| match mode {
        "managed-step" => Some(TrainingMode::ManagedStep),
        "manual-step" => Some(TrainingMode::ManualStep),
        "external-loop" => Some(TrainingMode::ExternalLoop),
        _ => None,
    })
}

fn display_value(binding: &Value) -> String {
    if binding["sensitive"] == true {
        return "[已隐藏的敏感值]".into();
    }
    match binding["value"].as_str() {
        Some(value) if !value.is_empty() => value.to_owned(),
        _ => "未提供".into(),
    }
}

fn input_label(key: &str) -> &'static str {
    match key {
        "model" => "模型",
        "batch" => "批次",
        "training_step" => "训练步骤",
        "training_loop" => "完整循环",
        _ => "输入",
    }
}

fn source_label(source: &str) -> &'static str {
    match source {
        "project_config" => "项目配置",
        "command_line" => "本次输入/命令行",
        "user_session" => "工作台会话",
        "saved_run" => "来源运行快照",
        "model_variant" => "模型变体",
        "data_adapter" => "数据适配器",
        "generated" => "工具生成",
        "external" => "外部来源",
        _ => "未知来源",
    }
}

fn overrides(
    step_params: &[String],
    loop_params: &[String],
    lr: Option<f64>,
    weight_decay: Option<f64>,
) -> String {
    let mut values = Vec::new();
    if !step_params.is_empty() {
        values.push(format!("步骤参数 {}", parameter_values(step_params)));
    }
    if !loop_params.is_empty() {
        values.push(format!("循环参数 {}", parameter_values(loop_params)));
    }
    if let Some(lr) = lr {
        values.push(format!("学习率 {lr}"));
    }
    if let Some(weight_decay) = weight_decay {
        values.push(format!("weight decay {weight_decay}"));
    }
    values.join("；")
}

fn parameter_values(params: &[String]) -> String {
    if params.is_empty() {
        return "无".into();
    }
    params
        .iter()
        .map(|parameter| {
            let (name, value) = parameter
                .split_once('=')
                .unwrap_or((parameter.as_str(), "[未知值]"));
            let sensitive = [
                "password",
                "secret",
                "token",
                "api_key",
                "access_key",
                "credential",
                "private_key",
                "authorization",
            ]
            .iter()
            .any(|marker| name.to_ascii_lowercase().contains(marker));
            if sensitive {
                format!("{name}=[隐藏]")
            } else {
                let value = serde_json::from_str::<Value>(value)
                    .map(|value| value.to_string())
                    .unwrap_or_else(|_| value.into());
                format!("{name}={value}")
            }
        })
        .collect::<Vec<_>>()
        .join("、")
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::cli::{Cli, Command, TrainingInputs, TrainingMode};
    use serde_json::json;

    fn cli() -> Cli {
        Cli {
            describe_interface: false,
            json: false,
            no_interactive: false,
            python: None,
            python_explicit: false,
            timeout: None,
            variant: None,
            if_variant_base: None,
            if_input_content: None,
            if_source_manifest: None,
            if_source_plan: None,
            if_source_config: None,
            capture_input_to: None,
            replay_input_from: None,
            seed: None,
            verbose: false,
            command: None,
        }
    }

    #[test]
    fn external_loop_plan_explains_its_own_data_and_update_contract() {
        let command = Command::Train {
            inputs: TrainingInputs::default(),
            resume: None,
            if_checkpoint_sha256: None,
            migrate_from: None,
            migration_plan: None,
            name: None,
            request_id: None,
            training_mode: Some(TrainingMode::ExternalLoop),
            step: None,
            training_loop: Some("project:run_loop".into()),
            steps: Some(40),
            additional_steps: None,
            checkpoint_every: 10,
            lr: None,
            weight_decay: None,
            apply_supervision: false,
            params: Vec::new(),
            loop_params: Vec::new(),
            allow_state_reset: false,
            detach: false,
        };
        let readiness = json!({
            "inputs": [
                {"key":"model", "value":null, "required":false, "sensitive":false, "source":{"kind":"external"}},
                {"key":"batch", "value":null, "required":false, "sensitive":false, "source":{"kind":"external"}},
                {"key":"training_loop", "value":"project:run_loop", "required":true, "sensitive":false, "source":{"kind":"command_line"}}
            ],
            "readiness": {"state":"ready"}
        });

        let rendered = render(&command, &cli(), Some(&readiness), None).join("\n");
        assert!(rendered.contains("完整循环自行管理数据"), "{rendered}");
        assert!(rendered.contains("不要求独立 batch 或 loss"), "{rendered}");
        assert!(rendered.contains("40 总步数"), "{rendered}");
        assert!(rendered.contains("显存预算估算：未接入"), "{rendered}");
        assert!(!rendered.contains("交叉熵"), "{rendered}");
    }

    #[test]
    fn training_plan_shows_effective_sources_and_redacts_sensitive_params() {
        let command = Command::Try {
            inputs: TrainingInputs::default(),
            repeat: None,
            name: None,
            training_mode: Some(TrainingMode::ManagedStep),
            step: Some("project:step".into()),
            training_loop: None,
            steps: Some(3),
            lr: Some(0.0001),
            weight_decay: Some(0.01),
            apply_supervision: false,
            no_supervision: false,
            params: vec![
                "learning_rate=0.001".into(),
                "api_token=secret-value".into(),
            ],
            loop_params: Vec::new(),
        };
        let readiness = json!({
            "inputs": [
                {"key":"model", "value":"project:model", "required":true, "sensitive":false, "source":{"kind":"project_config", "reference":"model"}},
                {"key":"batch", "value":"project:batch", "required":true, "sensitive":false, "source":{"kind":"user_session"}},
                {"key":"training_step", "value":"project:step", "required":true, "sensitive":false, "source":{"kind":"command_line"}},
                {"key":"credential", "value":"top-secret", "required":false, "sensitive":true, "source":{"kind":"project_config"}}
            ],
            "readiness": {"state":"ready"}
        });

        let rendered = render(&command, &cli(), Some(&readiness), None).join("\n");
        assert!(rendered.contains("来源：项目配置"), "{rendered}");
        assert!(rendered.contains("来源：工作台会话"), "{rendered}");
        assert!(rendered.contains("learning_rate=0.001"), "{rendered}");
        assert!(rendered.contains("api_token=[隐藏]"), "{rendered}");
        assert!(!rendered.contains("secret-value"), "{rendered}");
        assert!(!rendered.contains("top-secret"), "{rendered}");
    }

    #[test]
    fn repeat_plan_binds_to_saved_snapshot_and_lists_explicit_overrides() {
        let command = Command::Try {
            inputs: TrainingInputs::default(),
            repeat: Some("run-parent".into()),
            name: None,
            training_mode: None,
            step: None,
            training_loop: None,
            steps: Some(5),
            lr: Some(0.0002),
            weight_decay: None,
            apply_supervision: false,
            no_supervision: false,
            params: vec!["dropout=0.2".into()],
            loop_params: Vec::new(),
        };
        let source = json!({
            "record":{"id":"run-parent","name":"Baseline","training_mode":"managed-step"},
            "inputs":{"training_mode":"managed-step"}
        });

        let rendered = render(&command, &cli(), None, Some(&source)).join("\n");
        assert!(
            rendered.contains("读取该运行保存的原始输入快照"),
            "{rendered}"
        );
        assert!(
            rendered.contains("本次显式覆盖：步骤参数 dropout=0.2；学习率 0.0002"),
            "{rendered}"
        );
        assert!(rendered.contains("命令行全局覆盖：无"), "{rendered}");
    }
}
