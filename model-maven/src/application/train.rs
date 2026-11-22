use super::*;

fn read_weight_migration_plan(path: &Path) -> Result<Value, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("无法读取权重迁移计划：{error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("权重迁移计划必须是普通文件，不能是符号链接。".into());
    }
    if metadata.len() > 1024 * 1024 {
        return Err("权重迁移计划超过 1 MiB 限制。".into());
    }
    let bytes = std::fs::read(path).map_err(|error| format!("无法读取权重迁移计划：{error}"))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("权重迁移计划 JSON 无效：{error}"))?;
    if !value.is_object() || value["schema_version"] != 1 {
        return Err("权重迁移计划必须是 schema_version=1 的 JSON 对象。".into());
    }
    Ok(value)
}

impl<B: Backend> Application<B> {
    pub(super) fn repeat_record(
        &self,
        root: &std::path::Path,
        requested: &str,
    ) -> Result<(String, Value), String> {
        let id = if requested == "latest" {
            let runs = self.runs.list(root)?;
            crate::domain::comparison::latest_run(&runs, Some("training_probe"))
                .and_then(|run| run["id"].as_str().map(str::to_owned))
                .ok_or("本项目还没有可重跑的训练试跑记录。")?
        } else {
            requested.to_owned()
        };
        let record = self.runs.show(root, &id)?;
        if record["record"]["operation"] != "training_probe"
            || record["inputs"]["operation"] != "training_probe"
        {
            return Err(format!("运行 {id} 不是可重跑的训练步骤试跑。"));
        }
        if record["inputs"]["project_root"].as_str() != root.to_str() {
            return Err(format!(
                "运行 {id} 来自其他项目目录，不能在当前项目中直接重跑。"
            ));
        }
        Ok((id, record))
    }

    pub(super) fn resume_record(
        &self,
        root: &std::path::Path,
        requested: &str,
    ) -> Result<(String, Value, String), String> {
        let id = if requested == "latest" {
            let runs = self.runs.list(root)?;
            crate::domain::comparison::latest_run(&runs, Some("train"))
                .and_then(|run| run["id"].as_str().map(str::to_owned))
                .ok_or("本项目没有可恢复的正式训练运行。")?
        } else {
            requested.to_owned()
        };
        let mut record = self.runs.show(root, &id)?;
        if record["record"]["operation"] != "train" || record["inputs"]["operation"] != "train" {
            return Err(format!("运行 {id} 不是正式训练记录。"));
        }
        if record["inputs"]["project_root"].as_str() != root.to_str() {
            return Err(format!("运行 {id} 来自其他项目目录。"));
        }
        if matches!(
            record["record"]["status"].as_str(),
            Some("running" | "stopping")
        ) {
            let supervisor = &record["record"]["supervisor"];
            let pid = supervisor["pid"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok());
            let token = supervisor["start_token"].as_str();
            if let (Some(pid), Some(token)) = (pid, token) {
                if self.supervisor.is_alive(pid, token) {
                    return Err(format!(
                        "运行 {id} 仍由 PID {pid} 监督；请等待或使用 m-model runs stop。"
                    ));
                }
                let lost = Report::error(
                    "train",
                    "RUN_PROCESS_LOST",
                    "后台监督进程已退出；运行记录现标记为中断，可从最后已提交 checkpoint 恢复。",
                    "检查 runs show 和 worker.log，确认来源与恢复点后继续。",
                );
                self.runs.finish(
                    root,
                    &id,
                    &serde_json::to_value(lost).map_err(|error| error.to_string())?,
                )?;
                record = self.runs.show(root, &id)?;
            } else {
                return Err(format!(
                    "运行 {id} 没有可验证的监督身份；状态仍为 {}，不能安全确认已停止。",
                    record["record"]["status"].as_str().unwrap_or("unknown")
                ));
            }
        }
        if !matches!(
            record["record"]["status"].as_str(),
            Some("completed" | "failed" | "interrupted")
        ) {
            return Err(format!(
                "运行 {id} 状态为 {}，当前无法确认它已停止。",
                record["record"]["status"].as_str().unwrap_or("unknown")
            ));
        }
        let checkpoint = self.runs.latest_checkpoint(root, &id)?;
        Ok((id, record, checkpoint.to_string_lossy().into_owned()))
    }

    pub(super) fn build_training_job(&self, cli: &Cli, command: &Command) -> Result<Job, Report> {
        let job = match command {
            Command::Try {
                inputs,
                repeat,
                name,
                training_mode,
                step,
                training_loop,
                steps,
                lr,
                weight_decay,
                apply_supervision,
                params,
                loop_params,
            } => {
                let requested_device = inputs.device.clone();
                let prior = if let Some(requested) = repeat {
                    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
                        Ok(root) => root,
                        Err(error) => {
                            return Err(Report::error(
                                "training_probe",
                                "PROJECT_UNAVAILABLE",
                                error.to_string(),
                                "在可访问的项目目录中重跑。",
                            ));
                        }
                    };
                    match self.repeat_record(&root, requested) {
                        Ok((id, record)) => Some((id, record)),
                        Err(error) => {
                            return Err(Report::error(
                                "training_probe",
                                "REPEAT_SOURCE_UNAVAILABLE",
                                error,
                                "使用 m-model runs list 查看本项目的训练步骤试跑。",
                            ));
                        }
                    }
                } else {
                    None
                };
                let empty_snapshot = Value::Null;
                let prior_inputs = prior
                    .as_ref()
                    .map(|(_, record)| &record["inputs"])
                    .unwrap_or(&empty_snapshot);
                let previous_device = prior_inputs["runtime"]["device"].as_str().unwrap_or("cpu");
                let resolved_device = requested_device
                    .as_deref()
                    .unwrap_or(previous_device)
                    .to_owned();
                let device_changed = prior.is_some() && resolved_device != previous_device;
                let external_loop_requested = training_mode.as_ref()
                    == Some(&TrainingMode::ExternalLoop)
                    || training_loop.is_some()
                    || prior_inputs["training_mode"].as_str() == Some("external-loop");
                if !params.is_empty() && !loop_params.is_empty() {
                    return Err(Report::error(
                        "training_probe",
                        "TRAINING_PARAMETER_INVALID",
                        "--step-param 和 --loop-param 不能同时使用。",
                        "步骤入口用 --step-param；完整循环入口用 --loop-param。",
                    ));
                }
                if external_loop_requested && !params.is_empty() {
                    return Err(Report::error(
                        "training_probe",
                        "TRAINING_PARAMETER_INVALID",
                        "external-loop 参数请使用 --loop-param。",
                        "--step-param 只传给 managed/manual-step 的 training_step。",
                    ));
                }
                if !external_loop_requested && !loop_params.is_empty() {
                    return Err(Report::error(
                        "training_probe",
                        "TRAINING_PARAMETER_INVALID",
                        "--loop-param 只能与 external-loop 一起使用。",
                        "指定 --training-mode external-loop --loop module:run_loop。",
                    ));
                }
                let supplied_params = if external_loop_requested {
                    loop_params
                } else {
                    params
                };
                let mut resolved_inputs: Inputs = inputs.clone().into();
                if resolved_inputs.model.is_none() {
                    resolved_inputs.model = prior_inputs["model"].as_str().map(str::to_owned);
                }
                if resolved_inputs.batch.is_none() {
                    resolved_inputs.batch = prior_inputs["batch"].as_str().map(str::to_owned);
                }
                if resolved_inputs.injections.is_empty()
                    && resolved_inputs.injection_params.is_empty()
                {
                    resolved_inputs.saved_injections = Some(saved_model_injections(prior_inputs));
                }

                let mut repeat_overrides = Vec::new();
                if prior.is_some()
                    && inputs.data_fingerprint.as_deref().is_some_and(|value| {
                        Some(value.trim()) != prior_inputs["data_fingerprint"].as_str()
                    })
                {
                    repeat_overrides.push("data_fingerprint".into());
                }
                let apply_supervision = *apply_supervision
                    || prior_inputs["apply_supervision"].as_bool().unwrap_or(false);
                if prior.is_some()
                    && apply_supervision
                        != prior_inputs["apply_supervision"].as_bool().unwrap_or(false)
                {
                    repeat_overrides.push("apply_supervision".into());
                }
                if device_changed {
                    repeat_overrides.push("device".into());
                }
                if let Some((_, record)) = &prior {
                    for (name, current, previous) in [
                        ("model", &inputs.model, &prior_inputs["model"]),
                        ("batch", &inputs.batch, &prior_inputs["batch"]),
                        ("step", step, &prior_inputs["training_step"]),
                        ("loop", training_loop, &prior_inputs["training_loop"]),
                    ] {
                        if current
                            .as_ref()
                            .is_some_and(|value| Some(value.as_str()) != previous.as_str())
                        {
                            repeat_overrides.push(name.to_owned());
                        }
                    }
                    let source_manifest = record["report"]["data"]["source_manifest"].clone();
                    if !source_manifest.is_object() {
                        return Err(Report::error(
                            "training_probe",
                            "REPEAT_EVIDENCE_MISSING",
                            "原运行没有足够的源码指纹，无法确认重跑使用相同代码。",
                            "用当前代码重新运行一次 try，之后即可安全使用 --repeat。",
                        ));
                    }
                }

                let model_params =
                    match snapshot_params(prior_inputs, "model_params", &inputs.model_params) {
                        Ok(values) => values,
                        Err(error) => {
                            return Err(Report::error(
                                "training_probe",
                                "PARAMETER_INVALID",
                                error,
                                "参数使用 NAME=JSON 格式。",
                            ));
                        }
                    };
                let batch_params =
                    match snapshot_params(prior_inputs, "batch_params", &inputs.batch_params) {
                        Ok(values) => values,
                        Err(error) => {
                            return Err(Report::error(
                                "training_probe",
                                "PARAMETER_INVALID",
                                error,
                                "参数使用 NAME=JSON 格式。",
                            ));
                        }
                    };
                let training_params = match snapshot_params(
                    prior_inputs,
                    "training_params",
                    supplied_params,
                ) {
                    Ok(values) => values,
                    Err(error) => {
                        return Err(Report::error(
                            "training_probe",
                            "PARAMETER_INVALID",
                            error,
                            "参数使用 NAME=JSON 格式；步骤用 --step-param，完整循环用 --loop-param。",
                        ));
                    }
                };
                let model_params_changed =
                    params_changed(prior_inputs, "model_params", &inputs.model_params)
                        .expect("model params were parsed above");
                let injection_root =
                    match std::env::current_dir().and_then(|path| path.canonicalize()) {
                        Ok(root) => root,
                        Err(error) => {
                            return Err(Report::error(
                                "training_probe",
                                "PROJECT_UNAVAILABLE",
                                error.to_string(),
                                "在可访问的项目目录中重跑。",
                            ));
                        }
                    };
                let model_injections_changed = match injections_changed(
                    &prior_inputs["model_injections"],
                    &inputs.injections,
                    &inputs.injection_params,
                    &*self.components,
                    &injection_root,
                ) {
                    Ok(changed) => changed,
                    Err(error) => {
                        return Err(Report::error(
                            "training_probe",
                            "MODEL_INJECTION_INVALID",
                            error,
                            "接缝使用 --inject SLOT=module:factory 或 component:ID@VERSION；组件参数使用 --inject-param SLOT.NAME=JSON。",
                        ));
                    }
                };
                let batch_params_changed =
                    params_changed(prior_inputs, "batch_params", &inputs.batch_params)
                        .expect("batch params were parsed above");
                let training_params_changed =
                    params_changed(prior_inputs, "training_params", supplied_params)
                        .expect("training params were parsed above");
                if has_redacted_value(&json!({
                    "model_params": model_params,
                    "batch_params": batch_params,
                    "training_params": training_params,
                    "model_injections": if model_injections_changed {
                        Value::Null
                    } else {
                        prior_inputs["model_injections"].clone()
                    },
                })) {
                    return Err(Report::error(
                        "training_probe",
                        "REPEAT_SECRET_REQUIRED",
                        "原运行包含已脱敏参数；为避免猜测凭证，重跑前需要显式重新提供这些值。",
                        "使用对应的 --model-param、--batch-param、--step-param 或 --inject-param NAME=JSON 重新传入凭证；注入接缝也需通过 --inject 声明。",
                    ));
                }
                resolved_inputs.model_params = cli_params(model_params);
                resolved_inputs.batch_params = cli_params(batch_params);

                if prior.is_some() && training_params_changed {
                    repeat_overrides.push("training_params".into());
                }
                if prior.is_some() && model_params_changed {
                    repeat_overrides.push("model_params".into());
                }
                if prior.is_some() && model_injections_changed {
                    repeat_overrides.push("model_injections".into());
                }
                if prior.is_some() && batch_params_changed {
                    repeat_overrides.push("batch_params".into());
                }
                if prior.is_some()
                    && steps.is_some_and(|value| {
                        prior_inputs["steps"].as_u64() != Some(u64::from(value))
                    })
                {
                    repeat_overrides.push("steps".into());
                }
                if prior.is_some()
                    && lr.is_some_and(|value| prior_inputs["lr"].as_f64() != Some(value))
                {
                    repeat_overrides.push("lr".into());
                }
                if prior.is_some()
                    && weight_decay
                        .is_some_and(|value| prior_inputs["weight_decay"].as_f64() != Some(value))
                {
                    repeat_overrides.push("weight_decay".into());
                }

                let mode = training_mode
                    .or_else(|| {
                        if training_loop.is_some() {
                            return Some(TrainingMode::ExternalLoop);
                        }
                        match prior_inputs["training_mode"].as_str() {
                            Some("external-loop") => Some(TrainingMode::ExternalLoop),
                            Some("manual-step") => Some(TrainingMode::ManualStep),
                            Some("managed-step") => Some(TrainingMode::ManagedStep),
                            _ => None,
                        }
                    })
                    .unwrap_or_default();
                if prior.is_some()
                    && training_mode.is_some()
                    && prior_inputs["training_mode"].as_str()
                        != Some(match mode {
                            TrainingMode::ManagedStep => "managed-step",
                            TrainingMode::ManualStep => "manual-step",
                            TrainingMode::ExternalLoop => "external-loop",
                        })
                {
                    repeat_overrides.push("training_mode".into());
                }
                let resolved_loop = training_loop
                    .clone()
                    .or_else(|| prior_inputs["training_loop"].as_str().map(str::to_owned));
                let (resolved_step, resolved_loop) = if mode == TrainingMode::ExternalLoop {
                    if step.is_some() {
                        return Err(Report::error(
                            "training_probe",
                            "TRAINING_MODE_CONFLICT",
                            "external-loop 不使用 training_step。",
                            "移除 --step，并通过 --loop 提供完整循环入口。",
                        ));
                    }
                    if resolved_inputs.batch.is_some() {
                        return Err(Report::error(
                            "training_probe",
                            "EXTERNAL_LOOP_BATCH_UNUSED",
                            "external-loop 自己管理数据与批次，不接受独立 --batch 入口。",
                            "移除 --batch；数据迭代由 --loop 指定的用户代码负责。",
                        ));
                    }
                    let Some(resolved_loop) = resolved_loop else {
                        return Err(Report::error(
                            "training_probe",
                            "EXTERNAL_LOOP_REQUIRED",
                            "缺少完整训练循环入口。",
                            "使用 --training-mode external-loop --loop module:run_loop。",
                        ));
                    };
                    (None, Some(resolved_loop))
                } else {
                    if training_loop.is_some() {
                        return Err(Report::error(
                            "training_probe",
                            "TRAINING_MODE_CONFLICT",
                            "--loop 只能与 external-loop 一起使用。",
                            "指定 --training-mode external-loop，或移除 --loop 并使用 --step。",
                        ));
                    }
                    let resolved_step = step
                        .clone()
                        .or_else(|| prior_inputs["training_step"].as_str().map(str::to_owned));
                    let Some(resolved_step) = resolved_step else {
                        return Err(Report::error(
                            "training_probe",
                            "STEP_REQUIRED",
                            "缺少训练步骤入口。",
                            "使用 --step module:training_step，或从已有运行执行 --repeat。",
                        ));
                    };
                    (Some(resolved_step), None)
                };
                let resolved_steps = steps
                    .or_else(|| prior_inputs["steps"].as_u64().map(|v| v as u32))
                    .unwrap_or(3);
                let (resolved_lr, resolved_weight_decay) = if mode == TrainingMode::ExternalLoop {
                    if lr.is_some() || weight_decay.is_some() {
                        return Err(Report::error(
                            "training_probe",
                            "TRAINING_PARAMETER_INVALID",
                            "external-loop 的优化器完全由用户代码控制，CLI 不接受 --lr 或 --weight-decay。",
                            "在 Python 循环或 --loop-param 中设置优化器参数。",
                        ));
                    }
                    (None, None)
                } else {
                    let resolved_lr = lr.or_else(|| prior_inputs["lr"].as_f64()).unwrap_or(1e-4);
                    let resolved_weight_decay = weight_decay
                        .or_else(|| prior_inputs["weight_decay"].as_f64())
                        .unwrap_or(0.01);
                    if !resolved_lr.is_finite()
                        || resolved_lr <= 0.0
                        || !resolved_weight_decay.is_finite()
                        || resolved_weight_decay < 0.0
                    {
                        return Err(Report::error(
                            "training_probe",
                            "TRAINING_PARAMETER_INVALID",
                            "学习率必须是有限正数，weight decay 必须是有限非负数。",
                            "例如 --lr 1e-4 --weight-decay 0.01。",
                        ));
                    }
                    (Some(resolved_lr), Some(resolved_weight_decay))
                };
                let (repeat_of, expected_source_manifest, expected_runtime) = match &prior {
                    Some((id, record)) => (
                        Some(id.clone()),
                        if repeat_overrides
                            .iter()
                            .any(|name| ["model", "batch", "step", "loop"].contains(&name.as_str()))
                        {
                            None
                        } else {
                            Some(record["report"]["data"]["source_manifest"].clone())
                        },
                        if device_changed
                            || cli.python.as_deref().is_some_and(|value| {
                                prior_inputs["runtime"]["python_executable"].as_str() != Some(value)
                            })
                        {
                            None
                        } else {
                            Some(record["report"]["data"]["runtime"].clone())
                        },
                    ),
                    None => (None, None, None),
                };
                let saved_runtime = &prior_inputs["runtime"];
                let runtime = json!({
                    "python_executable": cli.python.as_deref()
                        .or_else(|| saved_runtime["python_executable"].as_str())
                        .unwrap_or("python3"),
                    "timeout_seconds": cli.timeout
                        .or_else(|| saved_runtime["timeout_seconds"].as_u64())
                        .unwrap_or(30),
                    "device": resolved_device,
                });
                if prior.is_some()
                    && cli.python.as_deref().is_some_and(|value| {
                        saved_runtime["python_executable"].as_str() != Some(value)
                    })
                {
                    repeat_overrides.push("python".into());
                }
                if prior.is_some()
                    && cli.timeout.is_some_and(|value| {
                        saved_runtime["timeout_seconds"].as_u64() != Some(value)
                    })
                {
                    repeat_overrides.push("timeout".into());
                }
                let seed = cli.seed.or_else(|| prior_inputs["seed"].as_u64());
                if prior.is_some()
                    && cli.seed.is_some()
                    && prior_inputs["seed"].as_u64() != cli.seed
                {
                    repeat_overrides.push("seed".into());
                }
                Job {
                    operation: "training_probe",
                    inputs: resolved_inputs,
                    run_name: name.clone(),
                    training_step: resolved_step,
                    training_loop: resolved_loop,
                    training_mode: Some(match mode {
                        TrainingMode::ManagedStep => "managed-step",
                        TrainingMode::ManualStep => "manual-step",
                        TrainingMode::ExternalLoop => "external-loop",
                    }),
                    training_params,
                    apply_supervision,
                    batch_fingerprint: inputs
                        .data_fingerprint
                        .clone()
                        .or_else(|| prior_inputs["data_fingerprint"].as_str().map(str::to_owned)),
                    repeat_of,
                    expected_source_manifest,
                    expected_runtime,
                    repeat_overrides,
                    runtime,
                    steps: Some(resolved_steps),
                    optimizer: (mode != TrainingMode::ExternalLoop).then(|| "adamw".into()),
                    lr: resolved_lr,
                    weight_decay: resolved_weight_decay,
                    seed,
                    ..empty_job()
                }
            }
            Command::Train {
                inputs,
                resume,
                if_checkpoint_sha256,
                migrate_from,
                migration_plan,
                name,
                request_id,
                training_mode,
                step,
                training_loop,
                steps,
                additional_steps,
                checkpoint_every,
                lr,
                weight_decay,
                apply_supervision,
                params,
                loop_params,
                allow_state_reset,
                detach,
            } => {
                if if_checkpoint_sha256.as_ref().is_some_and(|digest| {
                    digest.len() != 64
                        || !digest
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                }) {
                    return Err(Report::error(
                        "train",
                        "CHECKPOINT_DIGEST_INVALID",
                        "checkpoint 摘要必须是 64 位小写十六进制 SHA-256。",
                        "通过 runs show 查看 checkpoint 摘要。",
                    ));
                }
                let requested_device = inputs.device.clone();
                let prior = if let Some(requested) = resume {
                    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
                        Ok(root) => root,
                        Err(error) => {
                            return Err(Report::error(
                                "train",
                                "PROJECT_UNAVAILABLE",
                                error.to_string(),
                                "",
                            ));
                        }
                    };
                    match self.resume_record(&root, requested) {
                        Ok((id, record, checkpoint)) => Some((id, record, checkpoint)),
                        Err(error) => {
                            return Err(Report::error(
                                "train",
                                "RESUME_UNAVAILABLE",
                                error,
                                "运行 m-model runs show <id> 查看状态与 checkpoint；活动或状态未知的运行不能恢复。",
                            ));
                        }
                    }
                } else {
                    None
                };
                if migrate_from.is_some() && (resume.is_some() || additional_steps.is_some()) {
                    return Err(Report::error(
                        "train",
                        "MIGRATION_RESUME_CONFLICT",
                        "权重迁移会创建新子运行，不能与原样恢复或额外步数一起使用。",
                        "使用 --migrate-from RUN --migration-plan FILE --steps N 创建结构变更后的新运行。",
                    ));
                }
                if migration_plan.is_some() != migrate_from.is_some() {
                    return Err(Report::error(
                        "train",
                        "MIGRATION_PLAN_REQUIRED",
                        "权重迁移必须同时提供 --migrate-from 和 --migration-plan。",
                        "不需要继承权重时，移除两个选项并创建普通新训练运行。",
                    ));
                }
                let migration_source = if let Some(requested) = migrate_from {
                    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
                        Ok(root) => root,
                        Err(error) => {
                            return Err(Report::error(
                                "train",
                                "PROJECT_UNAVAILABLE",
                                error.to_string(),
                                "",
                            ));
                        }
                    };
                    match self.resume_record(&root, requested) {
                        Ok(source) => Some(source),
                        Err(error) => {
                            return Err(Report::error(
                                "train",
                                "MIGRATION_SOURCE_UNAVAILABLE",
                                error,
                                "父运行必须是已结束、未释放 checkpoint 的本项目正式训练。",
                            ));
                        }
                    }
                } else {
                    None
                };
                let migration_plan_value = if let Some(path) = migration_plan {
                    match read_weight_migration_plan(path) {
                        Ok(value) => Some(value),
                        Err(error) => {
                            return Err(Report::error(
                                "train",
                                "MIGRATION_PLAN_INVALID",
                                error,
                                "填写完整的 preserve_compatible、reset_target 和 drop_source 状态清单；不会自动按同名/同形状复制。",
                            ));
                        }
                    }
                } else {
                    None
                };
                let empty_snapshot = Value::Null;
                let prior_inputs = prior
                    .as_ref()
                    .map(|(_, record, _)| &record["inputs"])
                    .unwrap_or(&empty_snapshot);
                if prior.is_some()
                    && inputs.data_fingerprint.as_deref().is_some_and(|value| {
                        Some(value.trim()) != prior_inputs["data_fingerprint"].as_str()
                    })
                {
                    return Err(Report::error(
                        "train",
                        "RESUME_DATA_FINGERPRINT_CHANGED",
                        "续训不能替换原运行声明的数据版本。",
                        "保持原数据版本；如需改变数据，请开始新的训练运行。",
                    ));
                }
                let previous_device = prior_inputs["runtime"]["device"].as_str().unwrap_or("cpu");
                if prior.is_some()
                    && requested_device
                        .as_deref()
                        .is_some_and(|device| device != previous_device)
                {
                    return Err(Report::error(
                        "train",
                        "RESUME_DEVICE_INCOMPATIBLE",
                        "续训不能更换 checkpoint 使用的设备。",
                        "省略 --device 以沿用原设备；如需更换设备，请从新运行开始。",
                    ));
                }
                let resolved_device = requested_device
                    .as_deref()
                    .unwrap_or(previous_device)
                    .to_owned();
                let external_loop_requested = training_mode.as_ref()
                    == Some(&TrainingMode::ExternalLoop)
                    || training_loop.is_some()
                    || prior_inputs["training_mode"].as_str() == Some("external-loop");
                if !params.is_empty() && !loop_params.is_empty() {
                    return Err(Report::error(
                        "train",
                        "TRAINING_PARAMETER_INVALID",
                        "--step-param 和 --loop-param 不能同时使用。",
                        "步骤入口用 --step-param；完整循环入口用 --loop-param。",
                    ));
                }
                if external_loop_requested && !params.is_empty() {
                    return Err(Report::error(
                        "train",
                        "TRAINING_PARAMETER_INVALID",
                        "external-loop 参数请使用 --loop-param。",
                        "--step-param 只传给 managed/manual-step 的 training_step。",
                    ));
                }
                if !external_loop_requested && !loop_params.is_empty() {
                    return Err(Report::error(
                        "train",
                        "TRAINING_PARAMETER_INVALID",
                        "--loop-param 只能与 external-loop 一起使用。",
                        "指定 --training-mode external-loop --loop module:run_loop。",
                    ));
                }
                let supplied_params = if external_loop_requested {
                    loop_params
                } else {
                    params
                };
                if prior.is_none() && additional_steps.is_some() {
                    return Err(Report::error(
                        "train",
                        "TRAINING_PARAMETER_INVALID",
                        "--additional-steps 只能与 --resume 一起使用。",
                        "新运行使用 --steps N；恢复运行使用 --resume [ID] --additional-steps N。",
                    ));
                }
                if prior.is_some() && steps.is_some() {
                    return Err(Report::error(
                        "train",
                        "TRAINING_PARAMETER_INVALID",
                        "恢复运行不能覆盖总步骤目标。",
                        "使用 --additional-steps N 明确指定续训步数。",
                    ));
                }
                let checkpoint_input = prior.as_ref().map(|(_, _, path)| path.clone());
                let (parent_run_id, resume_step, expected_source_manifest, expected_runtime) =
                    match &prior {
                        Some((id, record, _)) => {
                            if inputs.model.is_some()
                                || inputs.batch.is_some()
                                || step.is_some()
                                || training_loop.is_some()
                                || training_mode.is_some()
                            {
                                return Err(Report::error(
                                    "train",
                                    "RESUME_INPUT_CONFLICT",
                                    "恢复训练沿用原模型、批次和训练步骤入口。",
                                    "移除这些入口覆盖；如需改结构或算法，请创建新的正式训练运行。",
                                ));
                            }
                            let checkpoint_steps = record["checkpoint"]["completed_steps"]
                                .as_u64()
                                .or_else(|| record["report"]["data"]["completed_steps"].as_u64())
                                .ok_or_else(|| {
                                    Report::error(
                                        "train",
                                        "CHECKPOINT_INVALID",
                                        "checkpoint 清单缺少已完成步数。",
                                        "保留原运行记录并检查 checkpoint 文件。",
                                    )
                                });
                            let checkpoint_steps = checkpoint_steps?;
                            let Some(additional) = additional_steps else {
                                return Err(Report::error(
                                    "train",
                                    "ADDITIONAL_STEPS_REQUIRED",
                                    "恢复训练需要明确指定额外步数。",
                                    "例如 m-model train --resume --additional-steps 100。",
                                ));
                            };
                            let Some(target) = checkpoint_steps.checked_add(u64::from(*additional))
                            else {
                                return Err(Report::error(
                                    "train",
                                    "TRAINING_PARAMETER_INVALID",
                                    "续训目标步数超出范围。",
                                    "减少 --additional-steps。",
                                ));
                            };
                            if target > 1_000_000 {
                                return Err(Report::error(
                                    "train",
                                    "TRAINING_PARAMETER_INVALID",
                                    "正式训练目标最多为 1,000,000 步。",
                                    "减少 --additional-steps。",
                                ));
                            }
                            if cli.variant.is_some() {
                                return Err(Report::error(
                                    "train",
                                    "RESUME_VARIANT_CONFLICT",
                                    "恢复训练固定使用原运行中的模型变体快照。",
                                    "移除 --variant；使用新模型时创建新的训练运行。",
                                ));
                            }
                            if let Some(value) = lr
                                && prior_inputs["lr"].as_f64() != Some(*value)
                            {
                                return Err(Report::error(
                                    "train",
                                    "RESUME_OPTIMIZER_CONFLICT",
                                    "当前恢复协议要求沿用 checkpoint 中的优化器学习率。",
                                    "移除 --lr；后续版本会按具名优化器组支持显式学习率迁移。",
                                ));
                            }
                            if let Some(value) = weight_decay
                                && prior_inputs["weight_decay"].as_f64() != Some(*value)
                            {
                                return Err(Report::error(
                                    "train",
                                    "RESUME_OPTIMIZER_CONFLICT",
                                    "当前恢复协议要求沿用 checkpoint 中的 weight decay。",
                                    "移除 --weight-decay。",
                                ));
                            }
                            if cli.python.as_deref().is_some_and(|value| {
                                prior_inputs["runtime"]["python_executable"].as_str() != Some(value)
                            }) {
                                return Err(Report::error(
                                    "train",
                                    "RESUME_RUNTIME_CONFLICT",
                                    "恢复训练必须使用原 Python 解释器。",
                                    "移除 --python 并恢复原训练环境。",
                                ));
                            }
                            (
                                Some(id.clone()),
                                Some(target as u32),
                                record["report"]["data"]["source_manifest"]
                                    .as_object()
                                    .map(|_| record["report"]["data"]["source_manifest"].clone()),
                                Some(record["report"]["data"]["runtime"].clone()),
                            )
                        }
                        None => {
                            if additional_steps.is_some() {
                                unreachable!("validated above")
                            }
                            let Some(total) = steps else {
                                return Err(Report::error(
                                    "train",
                                    "TRAIN_STEPS_REQUIRED",
                                    "新训练需要明确指定目标步数。",
                                    "例如 m-model train --steps 100。",
                                ));
                            };
                            (None, Some(*total), None, None)
                        }
                    };
                let mut resolved_inputs: Inputs = inputs.clone().into();
                if resolved_inputs.model.is_none() {
                    resolved_inputs.model = prior_inputs["model"].as_str().map(str::to_owned);
                }
                if resolved_inputs.batch.is_none() {
                    resolved_inputs.batch = prior_inputs["batch"].as_str().map(str::to_owned);
                }
                if resolved_inputs.injections.is_empty()
                    && resolved_inputs.injection_params.is_empty()
                {
                    resolved_inputs.saved_injections = Some(saved_model_injections(prior_inputs));
                }
                if let Some((_, record, _)) = &prior {
                    if inputs.model_params.is_empty() {
                        resolved_inputs.model_params = record["inputs"]["model_params"]
                            .as_object()
                            .map(|values| cli_params(values.clone()))
                            .unwrap_or_default();
                    }
                    if inputs.batch_params.is_empty() {
                        resolved_inputs.batch_params = record["inputs"]["batch_params"]
                            .as_object()
                            .map(|values| cli_params(values.clone()))
                            .unwrap_or_default();
                    }
                    if inputs.injections.is_empty() && inputs.injection_params.is_empty() {
                        resolved_inputs.saved_injections =
                            Some(saved_model_injections(&record["inputs"]));
                    }
                }
                let model_params =
                    match snapshot_params(prior_inputs, "model_params", &inputs.model_params) {
                        Ok(value) => value,
                        Err(error) => {
                            return Err(Report::error(
                                "train",
                                "PARAMETER_INVALID",
                                error,
                                "参数使用 NAME=JSON 格式。",
                            ));
                        }
                    };
                let batch_params =
                    match snapshot_params(prior_inputs, "batch_params", &inputs.batch_params) {
                        Ok(value) => value,
                        Err(error) => {
                            return Err(Report::error(
                                "train",
                                "PARAMETER_INVALID",
                                error,
                                "参数使用 NAME=JSON 格式。",
                            ));
                        }
                    };
                let training_params = match snapshot_params(
                    prior_inputs,
                    "training_params",
                    supplied_params,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        return Err(Report::error(
                            "train",
                            "PARAMETER_INVALID",
                            error,
                            "参数使用 NAME=JSON 格式；步骤用 --step-param，完整循环用 --loop-param。",
                        ));
                    }
                };
                if has_redacted_value(&json!({
                    "model_params": model_params,
                    "batch_params": batch_params,
                    "training_params": training_params,
                    "model_injections": if inputs.injections.is_empty()
                        && inputs.injection_params.is_empty()
                    {
                        prior_inputs["model_injections"].clone()
                    } else {
                        Value::Null
                    },
                })) {
                    return Err(Report::error(
                        "train",
                        "RESUME_SECRET_REQUIRED",
                        "运行快照含脱敏参数；必须重新提供后才能继续训练。",
                        "使用对应的 --model-param、--batch-param、--step-param 或 --inject-param NAME=JSON 重新传入凭证；注入接缝也需通过 --inject 声明。",
                    ));
                }
                resolved_inputs.model_params = cli_params(model_params);
                resolved_inputs.batch_params = cli_params(batch_params);
                let mode = training_mode
                    .or_else(|| {
                        if training_loop.is_some() {
                            return Some(TrainingMode::ExternalLoop);
                        }
                        match prior_inputs["training_mode"].as_str() {
                            Some("external-loop") => Some(TrainingMode::ExternalLoop),
                            Some("manual-step") => Some(TrainingMode::ManualStep),
                            Some("managed-step") => Some(TrainingMode::ManagedStep),
                            _ => None,
                        }
                    })
                    .unwrap_or_default();
                let resolved_loop = training_loop
                    .clone()
                    .or_else(|| prior_inputs["training_loop"].as_str().map(str::to_owned));
                let (training_step, resolved_loop) = if mode == TrainingMode::ExternalLoop {
                    if step.is_some() {
                        return Err(Report::error(
                            "train",
                            "TRAINING_MODE_CONFLICT",
                            "external-loop 不使用 training_step。",
                            "移除 --step，并通过 --loop 提供完整循环入口。",
                        ));
                    }
                    let Some(resolved_loop) = resolved_loop else {
                        return Err(Report::error(
                            "train",
                            "EXTERNAL_LOOP_REQUIRED",
                            "缺少完整训练循环入口。",
                            "使用 --training-mode external-loop --loop module:run_loop。",
                        ));
                    };
                    (None, Some(resolved_loop))
                } else {
                    if training_loop.is_some() {
                        return Err(Report::error(
                            "train",
                            "TRAINING_MODE_CONFLICT",
                            "--loop 只能与 external-loop 一起使用。",
                            "指定 --training-mode external-loop，或移除 --loop 并使用 --step。",
                        ));
                    }
                    let training_step = step
                        .clone()
                        .or_else(|| prior_inputs["training_step"].as_str().map(str::to_owned));
                    let Some(training_step) = training_step else {
                        return Err(Report::error(
                            "train",
                            "STEP_REQUIRED",
                            "缺少 training_step 入口。",
                            "使用 --step module:training_step；恢复运行从原记录读取入口。",
                        ));
                    };
                    (Some(training_step), None)
                };
                if mode == TrainingMode::ExternalLoop && resolved_inputs.batch.is_some() {
                    return Err(Report::error(
                        "train",
                        "EXTERNAL_LOOP_BATCH_UNUSED",
                        "external-loop 自己管理数据与批次，不接受独立 --batch 入口。",
                        "移除 --batch；数据迭代由 --loop 指定的用户代码负责。",
                    ));
                }
                let (resolved_lr, resolved_weight_decay) = if mode == TrainingMode::ExternalLoop {
                    if lr.is_some() || weight_decay.is_some() {
                        return Err(Report::error(
                            "train",
                            "TRAINING_PARAMETER_INVALID",
                            "external-loop 的优化器完全由用户代码控制，CLI 不接受 --lr 或 --weight-decay。",
                            "在 Python 循环或 --loop-param 中设置优化器参数。",
                        ));
                    }
                    (None, None)
                } else {
                    let resolved_lr = lr.or_else(|| prior_inputs["lr"].as_f64()).unwrap_or(1e-4);
                    let resolved_weight_decay = weight_decay
                        .or_else(|| prior_inputs["weight_decay"].as_f64())
                        .unwrap_or(0.01);
                    if !resolved_lr.is_finite()
                        || resolved_lr <= 0.0
                        || !resolved_weight_decay.is_finite()
                        || resolved_weight_decay < 0.0
                    {
                        return Err(Report::error(
                            "train",
                            "TRAINING_PARAMETER_INVALID",
                            "学习率必须是有限正数，weight decay 必须是有限非负数。",
                            "例如 --lr 1e-4 --weight-decay 0.01。",
                        ));
                    }
                    (Some(resolved_lr), Some(resolved_weight_decay))
                };
                let previous_runtime = &prior_inputs["runtime"];
                let runtime = json!({
                    "python_executable": cli.python.as_deref()
                        .or_else(|| previous_runtime["python_executable"].as_str())
                        .unwrap_or("python3"),
                    "timeout_seconds": cli.timeout
                        .or_else(|| if *detach { Some(3_153_600_000) } else { previous_runtime["timeout_seconds"].as_u64() })
                        .unwrap_or(86_400),
                    "device": resolved_device,
                });
                Job {
                    operation: "train",
                    inputs: resolved_inputs,
                    run_name: name.clone(),
                    request_id: request_id.clone(),
                    training_step,
                    training_loop: resolved_loop,
                    training_mode: Some(match mode {
                        TrainingMode::ManagedStep => "managed-step",
                        TrainingMode::ManualStep => "manual-step",
                        TrainingMode::ExternalLoop => "external-loop",
                    }),
                    training_params,
                    apply_supervision: *apply_supervision
                        || prior_inputs["apply_supervision"].as_bool().unwrap_or(false),
                    batch_fingerprint: inputs
                        .data_fingerprint
                        .clone()
                        .or_else(|| prior_inputs["data_fingerprint"].as_str().map(str::to_owned)),
                    parent_run_id: migration_source
                        .as_ref()
                        .map(|(id, _, _)| id.clone())
                        .or(parent_run_id),
                    variant_snapshot: prior
                        .as_ref()
                        .map(|(_, record, _)| record["inputs"]["variant"].clone())
                        .filter(|value| value.is_object()),
                    expected_source_manifest,
                    expected_runtime,
                    runtime,
                    seed: cli.seed.or_else(|| prior_inputs["seed"].as_u64()),
                    checkpoint_every: Some(*checkpoint_every),
                    checkpoint_input,
                    expected_checkpoint_sha256: if_checkpoint_sha256.clone(),
                    migration_checkpoint_input: migration_source
                        .as_ref()
                        .map(|(_, _, checkpoint)| checkpoint.clone()),
                    migration_plan: migration_plan_value,
                    allow_state_reset: *allow_state_reset,
                    detach: *detach,
                    steps: resume_step,
                    optimizer: (mode != TrainingMode::ExternalLoop).then(|| "adamw".into()),
                    lr: resolved_lr,
                    weight_decay: resolved_weight_decay,
                    ..empty_job()
                }
            }
            _ => {
                return Err(Report::error(
                    "train",
                    "COMMAND_NOT_TRAINING",
                    "训练 Job 构造器收到非训练命令。",
                    "",
                ));
            }
        };
        Ok(job)
    }
}
