use super::*;

// Values are the explicit fields of the Evaluate CLI command and saved inputs.
#[allow(clippy::too_many_arguments)]
pub(super) fn handle<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    run_id: &str,
    checkpoint: &str,
    if_checkpoint_sha256: Option<&str>,
    split: &str,
    evaluator: &str,
    data: &str,
    samples: Option<u64>,
    preview_samples: Option<u64>,
    data_fingerprint: Option<&str>,
    params: &[String],
    data_params: &[String],
) -> Report {
    if if_checkpoint_sha256.is_some_and(|digest| {
        digest.len() != 64
            || !digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }) {
        return Report::error(
            "evaluate",
            "CHECKPOINT_DIGEST_INVALID",
            "checkpoint 摘要必须是 64 位小写十六进制 SHA-256。",
            "通过 runs show 查看 checkpoint 摘要。",
        );
    }
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(root) => root,
        Err(error) => {
            return Report::error("evaluate", "PROJECT_UNAVAILABLE", error.to_string(), "");
        }
    };
    if cli.variant.is_some() {
        return Report::error(
            "evaluate",
            "EVALUATION_VARIANT_CONFLICT",
            "评估使用所选训练运行保存的模型/变体快照。",
            "移除全局 --variant；若需评估另一个结构，请基于该结构启动并训练新的运行。",
        );
    }
    if checkpoint != "latest" {
        return Report::error(
            "evaluate",
            "EVALUATION_CHECKPOINT_UNSUPPORTED",
            "当前只支持训练运行中最新且完整验证的 checkpoint。",
            "使用 --checkpoint latest；其它权重迁移需先创建有来源记录的新运行。",
        );
    }
    let (source_run_id, source_record, checkpoint_path) = match app.resume_record(&root, run_id) {
        Ok(record) => record,
        Err(error) => {
            return Report::error(
                "evaluate",
                "EVALUATION_RUN_UNAVAILABLE",
                error,
                "运行 m-model runs show <id> 核对训练状态和可用 checkpoint；活动或状态未知的运行不能评估。",
            );
        }
    };
    let mut request = source_record["inputs"].clone();
    if !request.is_object() {
        return Report::error(
            "evaluate",
            "EVALUATION_INPUTS_UNAVAILABLE",
            "训练运行缺少有效输入快照。",
            "保留原运行记录；不要用当前工作区设置猜测训练模型。",
        );
    }
    // The source training request identifies that launch only. Evaluation is
    // a separate run, linked by source_run_id, and must not reuse its dedupe key.
    request.as_object_mut().unwrap().remove("request_id");
    if request["model"].as_str().is_none() && request["variant"].is_null() {
        return Report::error(
            "evaluate",
            "EVALUATION_MODEL_REQUIRED",
            "所选训练运行没有可重建的模型工厂或变体快照。",
            "该 external-loop 运行未提供模型权重入口，不能由 Model-Maven 通用评估器加载；选择包含可重建模型的训练记录。",
        );
    }
    let evaluation_params = match parameters(params) {
        Ok(value) => value,
        Err(error) => return Report::error("evaluate", "PARAMETER_INVALID", error, ""),
    };
    let evaluation_data_params = match parameters(data_params) {
        Ok(value) => value,
        Err(error) => return Report::error("evaluate", "PARAMETER_INVALID", error, ""),
    };
    if split.trim().is_empty() || split.chars().count() > 64 || split.chars().any(char::is_control)
    {
        return Report::error(
            "evaluate",
            "EVALUATION_SPLIT_INVALID",
            "split 须为 1..64 个可见字符。",
            "例如 --split validation 或 --split test。",
        );
    }
    if data_fingerprint.is_some_and(|value| {
        value.trim().is_empty() || value.len() > 256 || value.chars().any(char::is_control)
    }) {
        return Report::error(
            "evaluate",
            "EVALUATION_DATA_FINGERPRINT_INVALID",
            "数据指纹须为 1..256 个非空可见字符。",
            "传入不含凭据的数据版本/快照标识，或省略该选项并在结果中查看较窄的身份范围。",
        );
    }
    if preview_samples
        .is_some_and(|count| count > 32 || samples.is_some_and(|sample_limit| count > sample_limit))
    {
        return Report::error(
            "evaluate",
            "EVALUATION_PREVIEW_LIMIT_INVALID",
            "preview-samples 必须在 1..=32，且不能大于 samples 评估上限。",
            "降低 --preview-samples，或提高 --samples 评估上限。",
        );
    }
    request["operation"] = json!("evaluate");
    request["required_features"] = json!(["training.evaluation", "training.checkpoint_resume"]);
    request["project_root"] = json!(root);
    request["checkpoint_input"] = json!(checkpoint_path);
    request["expected_checkpoint_sha256"] = json!(if_checkpoint_sha256);
    request["source_run_id"] = json!(source_run_id);
    request["evaluation_entrypoint"] = json!(evaluator);
    request["evaluation_data"] = json!(data);
    request["evaluation_data_params"] = json!(evaluation_data_params);
    request["evaluation_params"] = json!(evaluation_params);
    request["evaluation_split"] = json!(split.trim());
    request["evaluation_samples"] = samples.map(Value::from).unwrap_or(Value::Null);
    request["evaluation_preview_samples"] = json!(preview_samples.unwrap_or(0));
    request["evaluation_data_fingerprint"] = data_fingerprint
        .map(|value| json!(value.trim()))
        .unwrap_or(Value::Null);
    let evaluation_augment = match app.project_config.load(&root) {
        Ok(config) => match config.and_then(|config| config.get("augment").cloned()) {
            None => Vec::new(),
            Some(Value::Object(augment)) => match augment.get("eval") {
                None => Vec::new(),
                Some(Value::Array(steps)) => steps.clone(),
                Some(_) => {
                    return Report::error(
                        "evaluate",
                        "EVALUATION_AUGMENT_INVALID",
                        "augment.eval 必须是有序步骤数组。",
                        "修复 .model-maven/project.json；评估没有启动。",
                    );
                }
            },
            Some(_) => {
                return Report::error(
                    "evaluate",
                    "EVALUATION_AUGMENT_INVALID",
                    "augment 必须是 train/eval 切分对象。",
                    "修复 .model-maven/project.json；评估没有启动。",
                );
            }
        },
        Err(error) => {
            return Report::error(
                "evaluate",
                "PROJECT_CONFIG_INVALID",
                error,
                "检查项目增强配置；评估没有启动。",
            );
        }
    };
    for (index, step) in evaluation_augment.iter().enumerate() {
        if !step.is_object()
            || step["target"]
                .as_str()
                .is_none_or(|target| !super::augment::valid_target(target))
        {
            return Report::error(
                "evaluate",
                "EVALUATION_AUGMENT_INVALID",
                format!("augment.eval[{index}] 需要对象、有效 ID 和 target。"),
                "使用 m-model augment set <id> --split eval --target kwargs.images 设置数据目标。",
            );
        }
    }
    request["evaluation_augment"] = json!(evaluation_augment);
    request["expected_source_manifest"] =
        source_record["report"]["data"]["source_manifest"].clone();
    request["runtime"] = json!({
        "python_executable": cli.python.as_deref().unwrap_or("python3"),
        "timeout_seconds": cli.timeout.unwrap_or(30),
    });
    crate::domain::capability::annotate_worker_request(&mut request);

    let mut stored_inputs = request.clone();
    let mut redacted_fields = Vec::new();
    redact_run_inputs(&mut stored_inputs, "$", &mut redacted_fields);
    stored_inputs["redacted_fields"] = json!(redacted_fields);
    if let Err(report) = app.reconcile_lost_runs(&root, "evaluate") {
        return report;
    }
    let evaluation_run_id = match app.runs.create(&root, &stored_inputs) {
        Ok(id) => id,
        Err(error) => {
            if let Some(message) = error.strip_prefix("RUN_ALREADY_ACTIVE:") {
                return Report::error(
                    "evaluate",
                    "RUN_ALREADY_ACTIVE",
                    message,
                    "先查看当前运行；运行结束或明确恢复状态后再开始评估。只读查看仍可继续。",
                );
            }
            if let Some(message) = error.strip_prefix("RUN_STATE_UNKNOWN:") {
                return Report::error(
                    "evaluate",
                    "RUN_STATE_UNKNOWN",
                    message,
                    "先用 m-model runs 查看损坏或未确认的运行记录；在确认其状态前不会开始评估。",
                );
            }
            return Report::error(
                "evaluate",
                "RUN_STORE_FAILED",
                format!("无法创建评估记录：{error}"),
                "检查 .model-maven/runs 的目录权限和可用空间；评估尚未启动。",
            );
        }
    };
    match app.runs.run_directory(&root, &evaluation_run_id) {
        Ok(directory) => request["snapshot_dir"] = json!(directory.join("snapshot")),
        Err(error) => {
            let failed = Report::error(
                "evaluate",
                "RUN_DIRECTORY_UNAVAILABLE",
                error,
                "检查运行记录目录权限；评估用户代码尚未执行。",
            );
            return app.finish_run_report(&root, &evaluation_run_id, "evaluate", &request, failed);
        }
    }
    let pid = std::process::id();
    if let Some(token) = app.supervisor.identify(pid)
        && let Err(error) = app
            .runs
            .set_supervisor(&root, &evaluation_run_id, pid, &token)
    {
        let failed = Report::error(
            "evaluate",
            "RUN_SUPERVISOR_REGISTRATION_FAILED",
            error,
            "评估尚未启动；检查运行记录锁和磁盘权限。",
        );
        return app.finish_run_report(&root, &evaluation_run_id, "evaluate", &request, failed);
    }
    let mut report = app.backend.execute(&request);
    if let Some(data) = report.data.as_object_mut() {
        data.insert("evaluation_run_id".into(), json!(evaluation_run_id));
        data.insert("source_run_id".into(), json!(source_run_id));
    }
    app.finish_run_report(&root, &evaluation_run_id, "evaluate", &request, report)
}
