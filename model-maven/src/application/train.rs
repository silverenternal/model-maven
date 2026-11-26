use super::*;

fn resolve_model_initialization(
    operation: &'static str,
    inputs: &TrainingInputs,
    prior: &Value,
    resume: bool,
) -> Result<(Option<String>, Option<u64>), Report> {
    let recorded_policy = match prior.get("model_initialization_policy") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if matches!(value.as_str(), "runtime" | "seeded") => Some(value.as_str()),
        _ => return Err(Report::error(operation, "MODEL_INITIALIZATION_POLICY_INVALID", "历史模型构造策略无效。", "检查运行记录。")),
    };
    let recorded_seed = match prior.get("model_initialization_seed") {
        None | Some(Value::Null) => None,
        Some(value) => Some(value.as_u64().ok_or_else(|| Report::error(operation,
            "MODEL_INITIALIZATION_SEED_INVALID", "历史模型构造种子无效。", "检查运行记录。"))?),
    };
    let requested = inputs.model_initialization.as_deref();
    if resume && (requested.is_some_and(|value| Some(value) != recorded_policy)
        || inputs.model_initialization_seed.is_some_and(|value| Some(value) != recorded_seed)) {
        return Err(Report::error(operation, "RESUME_MODEL_INITIALIZATION_CONFLICT",
            "续训必须沿用原模型构造策略和种子。", "省略模型构造选项或创建新运行。"));
    }
    let policy = requested.or(recorded_policy);
    let seed = if requested.is_some() && requested != recorded_policy {
        inputs.model_initialization_seed
    } else {
        inputs.model_initialization_seed.or(recorded_seed)
    };
    if policy != Some("seeded") && seed.is_some() {
        return Err(Report::error(operation, "MODEL_INITIALIZATION_POLICY_CONFLICT",
            "模型构造种子需要 seeded 策略。", "选择 seeded 或移除构造种子。"));
    }
    Ok((policy.map(str::to_owned), seed))
}

fn resolve_batch_device_policy(operation: &'static str, inputs: &TrainingInputs, prior: &Value,
    historical: bool, resume: bool, external: bool) -> Result<Option<String>, Report> {
    let recorded = match prior.get("batch_device_policy") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if matches!(value.as_str(), "preserve" | "recursive-strict") => Some(value.as_str()),
        _ => return Err(Report::error(operation, "BATCH_DEVICE_POLICY_INVALID", "历史运行的批次设备策略无效或不受支持。", "检查运行记录；没有启动训练。")),
    };
    let requested = inputs.batch_device_policy.as_deref();
    if external {
        if requested.is_some() || (resume && recorded.is_some()) {
            return Err(Report::error(operation, "BATCH_DEVICE_POLICY_UNSUPPORTED", "完整用户循环自行管理批次设备。", "移除 --batch-device-policy。"));
        }
        return Ok(None);
    }
    if resume && requested.is_some() && requested != recorded {
        return Err(Report::error(operation, "RESUME_BATCH_DEVICE_POLICY_CONFLICT", "续训必须沿用原批次设备策略。", "省略 --batch-device-policy 或创建新运行。"));
    }
    Ok(requested.or(recorded).or(if historical { None } else { Some("recursive-strict") }).map(str::to_owned))
}

// Hash the same parsed inputs used to build the repeated job, not status or
// presentation annotations that can change while the user reviews the run.
pub(super) fn repeat_inputs_sha256(record: &Value) -> Option<String> {
    use sha2::{Digest, Sha256};
    if !record.get("inputs")?.is_object() { return None; }
    let bytes = serde_json::to_vec(&record["inputs"]).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

fn recorded_batch_binding(inputs: &Value) -> Result<Option<BatchBinding>, String> {
    match inputs.get("batch_binding") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => match value.as_str() {
            "envelope" => Ok(Some(BatchBinding::Envelope)),
            "single" => Ok(Some(BatchBinding::Single)),
            "single-raw" => Ok(Some(BatchBinding::SingleRaw)),
            "args" => Ok(Some(BatchBinding::Args)),
            "kwargs" => Ok(Some(BatchBinding::Kwargs)),
            _ => Err("历史运行的输入绑定不受支持。".into()),
        },
        _ => Err("历史运行的输入绑定格式无效。".into()),
    }
}

// Resolve custom managed optimizer declarations without importing user code.
// A resumed checkpoint cannot switch factory or parameters; a repeated trial
// may explicitly choose a different factory and is then a distinct observation.
fn resolve_managed_optimizer(
    operation: &'static str,
    inputs: &TrainingInputs,
    prior: &Value,
    mode: &TrainingMode,
    resume: bool,
) -> Result<(Option<String>, Map<String, Value>), Report> {
    let previous = prior["optimizer_factory"].as_str();
    let factory = inputs.optimizer_factory.as_deref().or_else(|| {
        (*mode == TrainingMode::ManagedStep).then_some(previous).flatten()
    });
    if factory.is_some() && *mode != TrainingMode::ManagedStep {
        return Err(Report::error(operation, "OPTIMIZER_MODE_UNSUPPORTED",
            "自定义受管优化器仅支持 managed-step。", "选择 managed-step，或在用户完整循环中管理优化器。"));
    }
    let Some(factory) = factory else {
        return Ok((None, Map::new()));
    };
    let Some((module, target)) = factory.split_once(':') else {
        return Err(Report::error(operation, "OPTIMIZER_FACTORY_INVALID", "优化器工厂须为 module:callable。", "填写可导入的工厂入口。"));
    };
    if module.is_empty() || target.is_empty() || target.contains(':') || factory.chars().any(char::is_whitespace) || factory.len() > 1024 {
        return Err(Report::error(operation, "OPTIMIZER_FACTORY_INVALID", "优化器工厂入口无效。", "填写有效的 module:callable。"));
    }
    let params = if inputs.optimizer_factory.as_deref().is_some_and(|value| Some(value) != previous) {
        parameters(&inputs.optimizer_params)
    } else {
        snapshot_params(prior, "optimizer_params", &inputs.optimizer_params)
    }.map_err(|error| Report::error(operation, "OPTIMIZER_PARAMETERS_INVALID", error, "使用 --optimizer-param NAME=JSON。"))?;
    if params.len() > 64 || has_redacted_value(&Value::Object(params.clone())) {
        return Err(Report::error(operation, "OPTIMIZER_PARAMETERS_INVALID", "优化器参数过多或包含已脱敏的历史值。", "检查参数并明确重新提供所需值。"));
    }
    if resume && (previous != Some(factory) || prior["optimizer_params"].as_object().cloned().unwrap_or_default() != params) {
        return Err(Report::error(operation, "RESUME_OPTIMIZER_CONFLICT", "续训必须沿用原优化器工厂与参数。", "使用原运行记录，或创建新的训练运行。"));
    }
    Ok((Some(factory.to_owned()), params))
}

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
                no_supervision,
                params,
                loop_params,
            } => {
                if *apply_supervision && *no_supervision {
                    return Err(Report::error(
                        "training_probe", "SUPERVISION_SELECTION_CONFLICT",
                        "不能同时启用和关闭本次试跑的监督。", "只选择一种监督设置。",
                    ));
                }
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
                if let Some(expected) = inputs.if_repeat_inputs_sha256.as_deref() {
                    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                        return Err(Report::error("training_probe", "REPEAT_INPUT_DIGEST_INVALID", "历史输入摘要必须是小写 SHA-256。", "重新查看历史试跑。"));
                    }
                    let actual = prior.as_ref().and_then(|(_, record)| repeat_inputs_sha256(record));
                    if actual.as_deref() != Some(expected) {
                        return Err(Report::error("training_probe", "REPEAT_INPUT_CHANGED", "历史试跑输入缺失或已变化，没有启动试跑。", "重新查看历史试跑并确认。"));
                    }
                }
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
                let external_loop_requested = match training_mode.as_ref() {
                    Some(mode) => *mode == TrainingMode::ExternalLoop,
                    None if training_loop.is_some() => true,
                    None if step.is_some() => false,
                    None => prior_inputs["training_mode"].as_str() == Some("external-loop"),
                };
                if inputs.no_model && (!external_loop_requested || prior.is_some() || cli.variant.is_some()
                    || inputs.model_initialization.is_some()) {
                    return Err(Report::error("training_probe", "NO_MODEL_UNSUPPORTED",
                        "--no-model 仅支持不使用变体的新 external-loop。",
                        "移除 --no-model，或启动不指定模型初始化策略的新完整循环。"));
                }
                if external_loop_requested && (inputs.batch.is_some() || !inputs.batch_params.is_empty() || inputs.batch_binding.is_some()) {
                    return Err(Report::error("training_probe", "EXTERNAL_LOOP_BATCH_UNUSED",
                        "完整循环自行管理输入，不接受独立批次、参数或绑定。", "移除批次相关选项，或改用受管／手动步骤。"));
                }
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
                let model_replaced = prior.is_some() && inputs.model.as_deref()
                    .is_some_and(|value| Some(value) != prior_inputs["model"].as_str());
                let batch_replaced = prior.is_some() && ((external_loop_requested && prior_inputs["batch"].as_str().is_some())
                    || inputs.batch.as_deref().is_some_and(|value| Some(value) != prior_inputs["batch"].as_str()));
                let training_replaced = prior.is_some() && (step.as_deref()
                    .is_some_and(|value| Some(value) != prior_inputs["training_step"].as_str())
                    || training_loop.as_deref()
                        .is_some_and(|value| Some(value) != prior_inputs["training_loop"].as_str())
                    || training_mode.as_ref().is_some_and(|mode| Some(match mode {
                        TrainingMode::ManagedStep => "managed-step", TrainingMode::ManualStep => "manual-step",
                        TrainingMode::ExternalLoop => "external-loop",
                    }) != prior_inputs["training_mode"].as_str()));
                let mut resolved_inputs: Inputs = inputs.clone().into();
                if prior.is_some() && prior_inputs["model"].is_null() && inputs.model.is_none() {
                    resolved_inputs.no_project_model = true;
                }
                if resolved_inputs.model.is_none() {
                    resolved_inputs.model = prior_inputs["model"].as_str().map(str::to_owned);
                }
                if resolved_inputs.batch.is_none() && !external_loop_requested {
                    resolved_inputs.batch = prior_inputs["batch"].as_str().map(str::to_owned);
                }
                if prior.is_some() && inputs.batch_binding.is_none()
                    && resolved_inputs.batch.as_deref() == prior_inputs["batch"].as_str()
                {
                    resolved_inputs.batch_binding = match recorded_batch_binding(prior_inputs) {
                        Ok(binding) => binding,
                        Err(error) => return Err(Report::error("training_probe", "REPEAT_BINDING_INVALID", error, "检查历史运行输入。")),
                    };
                }
                if resolved_inputs.injections.is_empty()
                    && resolved_inputs.injection_params.is_empty()
                    && !model_replaced
                {
                    resolved_inputs.saved_injections = Some(saved_model_injections(prior_inputs));
                }

                let mut repeat_overrides = Vec::new();
                let (model_initialization, model_initialization_seed) =
                    resolve_model_initialization("training_probe", inputs, prior_inputs, false)?;
                if prior.is_some() && (model_initialization.as_deref() != prior_inputs["model_initialization_policy"].as_str()
                    || model_initialization_seed != prior_inputs["model_initialization_seed"].as_u64()) {
                    repeat_overrides.push("model_initialization".into());
                }
                let batch_device_policy = resolve_batch_device_policy("training_probe", inputs, prior_inputs,
                    prior.is_some(), false, external_loop_requested)?;
                if prior.is_some() && batch_device_policy.as_deref() != prior_inputs["batch_device_policy"].as_str() {
                    repeat_overrides.push("batch_device_policy".into());
                }
                if batch_replaced && inputs.batch.is_none() {
                    repeat_overrides.push("batch".into());
                }
                if batch_replaced && inputs.batch_binding.is_none() && prior_inputs["batch_binding"].as_str().is_some() {
                    repeat_overrides.push("batch_binding".into());
                }
                if prior.is_some() && inputs.batch_binding.is_some() {
                    let recorded = match recorded_batch_binding(prior_inputs) {
                        Ok(binding) => binding,
                        Err(error) => return Err(Report::error("training_probe", "REPEAT_BINDING_INVALID", error, "检查历史运行输入。")),
                    };
                    if inputs.batch_binding != recorded {
                        repeat_overrides.push("batch_binding".into());
                    }
                }
                let resolved_data_fingerprint = inputs.data_fingerprint.clone().or_else(|| {
                    if batch_replaced { None } else { prior_inputs["data_fingerprint"].as_str().map(str::to_owned) }
                });
                if prior.is_some() && resolved_data_fingerprint.as_deref().map(str::trim)
                    != prior_inputs["data_fingerprint"].as_str()
                {
                    repeat_overrides.push("data_fingerprint".into());
                }
                let apply_supervision = !*no_supervision && (*apply_supervision
                    || prior_inputs["apply_supervision"].as_bool().unwrap_or(false));
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
                    match if model_replaced { parameters(&inputs.model_params) } else { snapshot_params(prior_inputs, "model_params", &inputs.model_params) } {
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
                    match if batch_replaced { parameters(&inputs.batch_params) } else { snapshot_params(prior_inputs, "batch_params", &inputs.batch_params) } {
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
                let training_params = match if training_replaced { parameters(supplied_params) }
                    else { snapshot_params(prior_inputs, "training_params", supplied_params) } {
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
                let model_params_changed = model_replaced ||
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
                let model_injections_changed = model_replaced || match injections_changed(
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
                let batch_params_changed = batch_replaced ||
                    params_changed(prior_inputs, "batch_params", &inputs.batch_params)
                        .expect("batch params were parsed above");
                let training_params_changed = training_replaced ||
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
                        if step.is_some() && prior_inputs["training_mode"].as_str() == Some("external-loop") {
                            return Some(TrainingMode::ManagedStep);
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
                    && prior_inputs["training_mode"].as_str().is_some()
                    && prior_inputs["training_mode"].as_str()
                        != Some(match mode {
                            TrainingMode::ManagedStep => "managed-step",
                            TrainingMode::ManualStep => "manual-step",
                            TrainingMode::ExternalLoop => "external-loop",
                        })
                {
                    repeat_overrides.push("training_mode".into());
                }
                let (optimizer_factory, optimizer_params) = resolve_managed_optimizer(
                    "training_probe", inputs, prior_inputs, &mode, false)?;
                if prior.is_some() && (optimizer_factory.as_deref() != prior_inputs["optimizer_factory"].as_str()
                    || prior_inputs["optimizer_params"].as_object().cloned().unwrap_or_default() != optimizer_params) {
                    repeat_overrides.push("optimizer".into());
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
                let (resolved_lr, resolved_weight_decay) = if optimizer_factory.is_some() {
                    if lr.is_some() || weight_decay.is_some() {
                        return Err(Report::error("training_probe", "OPTIMIZER_PARAMETER_CONFLICT", "自定义工厂不使用 AdamW 的 --lr 或 --weight-decay。", "使用 --optimizer-param 传递工厂参数。"));
                    }
                    (None, None)
                } else if mode == TrainingMode::ExternalLoop {
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
                            .any(|name| ["model", "batch", "step", "loop", "optimizer"].contains(&name.as_str()))
                        {
                            None
                        } else {
                            Some(record["report"]["data"]["source_manifest"].clone())
                        },
                        if device_changed
                            || (cli.python_explicit && cli.python.as_deref().is_some_and(|value| {
                                prior_inputs["runtime"]["python_executable"].as_str() != Some(value)
                            }))
                        {
                            None
                        } else {
                            Some(record["report"]["data"]["runtime"].clone())
                        },
                    ),
                    None => (None, None, None),
                };
                let saved_runtime = &prior_inputs["runtime"];
                let selected_python = if cli.python_explicit {
                    cli.python.as_deref().filter(|value| !value.trim().is_empty())
                } else {
                    saved_runtime["python_executable"].as_str().filter(|value| !value.trim().is_empty())
                        .or(cli.python.as_deref().filter(|value| !value.trim().is_empty()))
                }.ok_or_else(|| Report::error("training_probe", "PYTHON_SELECTION_REQUIRED",
                    "未记录或发现可用的 Python 选择。", "使用 --python 明确选择解释器后重试；试跑尚未启动。"))?;
                let runtime = json!({
                    "python_executable": selected_python,
                    "timeout_seconds": cli.timeout
                        .or_else(|| saved_runtime["timeout_seconds"].as_u64())
                        .unwrap_or(30),
                    "device": resolved_device,
                    "working_directory": saved_runtime["working_directory"].clone(),
                    "project_import_roots": saved_runtime["project_import_roots"].clone(),
                });
                if prior.is_some() && cli.python_explicit
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
                let mut explicit_sources = Map::new();
                if prior.is_some() {
                    for (key, supplied, flag) in [
                        ("model", inputs.model.is_some(), "--model"),
                        ("batch", inputs.batch.is_some(), "--batch"),
                        ("batch_binding", inputs.batch_binding.is_some(), "--batch-binding"),
                    ] {
                        if supplied {
                            explicit_sources.insert(key.into(), json!({"kind":"command_line", "reference":flag}));
                        }
                    }
                }
                Job {
                    operation: "training_probe",
                    inputs: resolved_inputs,
                    batch_device_policy,
                    model_initialization,
                    model_initialization_seed,
                    input_sources: explicit_sources,
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
                    batch_fingerprint: resolved_data_fingerprint,
                    repeat_of,
                    expected_source_manifest,
                    expected_runtime,
                    repeat_overrides,
                    runtime,
                    steps: Some(resolved_steps),
                    optimizer: (mode != TrainingMode::ExternalLoop && optimizer_factory.is_none()).then(|| "adamw".into()),
                    optimizer_factory,
                    optimizer_params,
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
                if inputs.if_repeat_inputs_sha256.is_some() {
                    return Err(Report::error("train", "REPEAT_INPUT_CONDITION_INVALID", "历史试跑输入摘要仅用于 try --repeat。", "续训使用检查点身份条件。"));
                }
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
                if inputs.no_model && (!external_loop_requested || prior.is_some() || cli.variant.is_some()
                    || inputs.model_initialization.is_some()) {
                    return Err(Report::error("train", "NO_MODEL_UNSUPPORTED",
                        "--no-model 仅支持不使用变体的新 external-loop。",
                        "移除 --no-model，或启动不指定模型初始化策略的新完整循环。"));
                }
                let batch_device_policy = resolve_batch_device_policy("train", inputs, prior_inputs,
                    prior.is_some(), prior.is_some(), external_loop_requested)?;
                let (model_initialization, model_initialization_seed) =
                    resolve_model_initialization("train", inputs, prior_inputs, prior.is_some())?;
                if external_loop_requested && (inputs.batch.is_some() || !inputs.batch_params.is_empty() || inputs.batch_binding.is_some()) {
                    return Err(Report::error("train", "EXTERNAL_LOOP_BATCH_UNUSED",
                        "完整循环自行管理输入，不接受独立批次、参数或绑定。", "移除批次相关选项，或改用受管／手动步骤。"));
                }
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
                                || inputs.batch_binding.is_some()
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
                            if cli.python_explicit && cli.python.as_deref().is_some_and(|value| {
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
                if prior.is_some() && prior_inputs["model"].is_null() && inputs.model.is_none() {
                    resolved_inputs.no_project_model = true;
                }
                if resolved_inputs.model.is_none() {
                    resolved_inputs.model = prior_inputs["model"].as_str().map(str::to_owned);
                }
                if resolved_inputs.batch.is_none() {
                    resolved_inputs.batch = prior_inputs["batch"].as_str().map(str::to_owned);
                }
                if prior.is_some() {
                    resolved_inputs.batch_binding = match recorded_batch_binding(prior_inputs) {
                        Ok(binding) => binding,
                        Err(error) => return Err(Report::error("train", "RESUME_BINDING_INVALID", error, "检查历史运行输入。")),
                    };
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
                let (optimizer_factory, optimizer_params) = resolve_managed_optimizer(
                    "train", inputs, prior_inputs, &mode, prior.is_some())?;
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
                let (resolved_lr, resolved_weight_decay) = if optimizer_factory.is_some() {
                    if lr.is_some() || weight_decay.is_some() {
                        return Err(Report::error("train", "OPTIMIZER_PARAMETER_CONFLICT", "自定义工厂不使用 AdamW 的 --lr 或 --weight-decay。", "使用 --optimizer-param 传递工厂参数。"));
                    }
                    (None, None)
                } else if mode == TrainingMode::ExternalLoop {
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
                let selected_python = if cli.python_explicit {
                    cli.python.as_deref().filter(|value| !value.trim().is_empty())
                } else {
                    previous_runtime["python_executable"].as_str().filter(|value| !value.trim().is_empty())
                        .or(cli.python.as_deref().filter(|value| !value.trim().is_empty()))
                }.ok_or_else(|| Report::error("train", "PYTHON_SELECTION_REQUIRED",
                    "未记录或发现可用的 Python 选择。", "使用 --python 明确选择解释器后重试；训练尚未启动。"))?;
                let runtime = json!({
                    "python_executable": selected_python,
                    "timeout_seconds": cli.timeout
                        .or_else(|| if *detach { Some(3_153_600_000) } else { previous_runtime["timeout_seconds"].as_u64() })
                        .unwrap_or(86_400),
                    "device": resolved_device,
                    "working_directory": previous_runtime["working_directory"].clone(),
                    "project_import_roots": previous_runtime["project_import_roots"].clone(),
                });
                Job {
                    operation: "train",
                    inputs: resolved_inputs,
                    batch_device_policy,
                    model_initialization,
                    model_initialization_seed,
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
                    optimizer: (mode != TrainingMode::ExternalLoop && optimizer_factory.is_none()).then(|| "adamw".into()),
                    optimizer_factory,
                    optimizer_params,
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
