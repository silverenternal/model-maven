use super::*;

fn generic_run_create_error(operation: &str, error: String, request_id: Option<&str>) -> Report {
    if let Some(existing) = error.strip_prefix("RUN_REQUEST_ID_EXISTS:") {
        let mut report = Report::error(operation, "RUN_REQUEST_ID_EXISTS", "该请求已有运行记录；没有再次启动。", "查看返回的运行 ID；新运行需要新的请求标识。 ");
        report.data["run_id"] = json!(existing);
        report.data["request_id"] = json!(request_id);
        return report;
    }
    if let Some(message) = error.strip_prefix("RUN_ALREADY_ACTIVE:") {
        return Report::error(operation, "RUN_ALREADY_ACTIVE", message, "先查看运行和进程；状态未知时可用 m-model runs acknowledge-unknown <ID> 审核。 ");
    }
    if let Some(message) = error.strip_prefix("RUN_STATE_UNKNOWN:") {
        return Report::error(operation, "RUN_STATE_UNKNOWN", message, "先检查运行记录；状态不明时不会启动新任务。 ");
    }
    Report::error(operation, "RUN_STORE_FAILED", error, "检查运行记录目录、权限和空间；用户代码未执行。")
}

impl<B: Backend> Application<B> {
    pub(super) fn run_generic_formal(
        &self, cli: &Cli, training_loop: Option<&str>, raw_params: &[String], steps: u32,
        resume: Option<&str>, expected_checkpoint: Option<&str>, name: Option<&str>, detach: bool, request_id: Option<&str>, loop_interface: Option<&str>,
        context: ExecutionContext<'_>,
    ) -> Report {
        const OP: &str = "generic_loop_train";
        if cli.if_input_content.is_some() || cli.if_source_manifest.is_some() || cli.if_source_plan.is_some()
            || cli.if_source_config.is_some() || cli.capture_input_to.is_some() || cli.replay_input_from.is_some() {
            return Report::error(OP, "GENERIC_LOOP_OPTION_UNSUPPORTED", "用户循环不能使用模型输入或源码编辑条件。", "移除输入快照、输入内容和源码编辑条件；恢复时使用 checkpoint 身份保护。 ");
        }
        if detach && !cfg!(target_os = "linux") {
            return Report::error(OP, "BACKGROUND_UNAVAILABLE", "当前构建仅在 Linux 支持可验证的后台监督。", "移除 --detach 使用前台运行。 ");
        }
        if request_id.is_some_and(|id| !(16..=64).contains(&id.len()) || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')) || (detach && request_id.is_none()) {
            return Report::error(OP, "REQUEST_ID_INVALID", "后台运行需要 16 到 64 位字母、数字或连字符的唯一请求标识。", "使用 --request-id 指定；不要填入敏感信息。 ");
        }
        if cli.variant.is_some() || cli.if_variant_base.is_some() || cli.seed.is_some() {
            return Report::error(OP, "GENERIC_LOOP_OPTION_UNSUPPORTED", "用户循环不使用模型变体或工具管理的随机种子。", "移除 --variant、--if-variant-base 和 --seed。 ");
        }
        if steps == 0 || steps > 1_000_000 {
            return Report::error(OP, "TRAINING_PARAMETER_INVALID", "步数须在 1 到 1000000 之间。", "调整 --steps。 ");
        }
        let name = match name {
            Some(value) if value.trim().is_empty() || value.chars().count() > 80 || value.chars().any(char::is_control) => return Report::error(OP, "RUN_NAME_INVALID", "运行名称须为 1 到 80 个可见字符。", "调整运行名称。"),
            Some(value) => Some(value.trim()), None => None,
        };
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(value) => value, Err(error) => return Report::error(OP, "PROJECT_UNAVAILABLE", error.to_string(), "检查项目目录。"),
        };
        if let Err(report) = self.reconcile_lost_runs(&root, OP) { return report; }
        let (entry, params, mut runtime, parent_checkpoint, loop_interface) = if let Some(parent) = resume {
            if training_loop.is_some() || !raw_params.is_empty() || loop_interface.is_some() {
                return Report::error(OP, "GENERIC_RESUME_OVERRIDE_UNSUPPORTED", "恢复时不能更改循环入口、签名或参数。", "移除 --loop、--loop-interface 和 --loop-param；新配置请启动新运行。 ");
            }
            let digest = match expected_checkpoint {
                Some(value) if value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) => value,
                _ => return Report::error(OP, "GENERIC_CHECKPOINT_IDENTITY_INVALID", "恢复需要明确的 checkpoint SHA-256。", "通过 --if-checkpoint-sha256 指定所选恢复点。"),
            };
            let record = match self.runs.show(&root, parent) {
                Ok(value) => value, Err(error) => return Report::error(OP, "RUN_UNAVAILABLE", error, "检查要恢复的运行。"),
            };
            if record["record"]["operation"] != OP || !matches!(record["record"]["status"].as_str(), Some("completed" | "failed" | "interrupted")) || record["record"]["checkpoint_released"] == true || record["inputs"]["project_root"].as_str() != root.to_str() {
                return Report::error(OP, "GENERIC_RESUME_INVALID", "所选运行不可恢复或不属于当前项目。", "选择同一项目已结束且保留恢复点的用户循环运行。 ");
            }
            let entry = match record["inputs"]["training_loop"].as_str().filter(|s| python_entrypoint(s) && s.len() <= 1024) {
                Some(value) => value.to_owned(), None => return Report::error(OP, "GENERIC_RESUME_INVALID", "历史循环入口无效。", "检查运行记录。"),
            };
            let params = record["inputs"]["training_params"].clone();
            if !params.is_object() || has_redacted_value(&params) {
                return Report::error(OP, "GENERIC_RESUME_INVALID", "历史参数不可完整恢复。", "请以完整参数启动新运行，敏感信息使用运行时环境。 ");
            }
            let runtime = record["inputs"]["runtime"].clone();
            let recorded_python = match runtime["python_executable"].as_str() {
                Some(value) if !value.is_empty() => value,
                _ => return Report::error(OP, "GENERIC_RESUME_INVALID", "历史 Python 解释器缺失。", "检查运行记录。"),
            };
            if cli.python_explicit && cli.python.as_deref() != Some(recorded_python) {
                return Report::error(OP, "GENERIC_RESUME_RUNTIME_CONFLICT", "所选解释器与历史运行不一致。", "使用历史解释器恢复，或开始新运行。 ");
            }
            let checkpoint = match self.runs.checkpoint_dir(&root, parent, false) {
                Ok(value) => value, Err(error) => return Report::error(OP, "GENERIC_CHECKPOINT_UNAVAILABLE", error, "检查历史恢复点。"),
            };
            let interface = record["inputs"]["loop_interface"].as_str().unwrap_or("model-context");
            if !matches!(interface, "context" | "model-context") {
                return Report::error(OP, "GENERIC_RESUME_INVALID", "历史循环签名无效。", "检查运行记录。 ");
            }
            (entry, params, runtime, Some((checkpoint, digest.to_owned())), interface.to_owned())
        } else {
            if expected_checkpoint.is_some() { return Report::error(OP, "GENERIC_RESUME_INVALID", "没有指定恢复运行。", "同时指定 --resume，或移除 checkpoint 摘要。 "); }
            let entry = match training_loop.filter(|value| value.len() <= 1024 && python_entrypoint(value)) {
                Some(value) => value.to_owned(), None => return Report::error(OP, "GENERIC_LOOP_INVALID", "循环入口必须是 module:callable。", "通过 --loop 指定用户循环。"),
            };
            let params = match parameters(raw_params) {
                Ok(value) if value.len() <= 64 && serde_json::to_vec(&value).is_ok_and(|raw| raw.len() <= 65536) => Value::Object(value),
                _ => return Report::error(OP, "GENERIC_LOOP_PARAMS_INVALID", "循环参数最多 64 项、总计不超过 64 KiB。", "使用 --loop-param NAME=JSON；不要填写密钥。"),
            };
            let python = match require_python(cli, OP) { Ok(value) => value, Err(report) => return report };
            let mut runtime = json!({"python_executable":python});
            match self.project_config.load(&root) {
                Ok(Some(config)) => {
                    if crate::project_python_selection_changed(&root, cli, config["runtime_python"].as_str()) { return Report::error(OP, "PROJECT_CONFIG_CONFLICT", "项目 Python 选择已变化。", "重新检查配置后重试。 "); }
                    super::run::apply_project_runtime_defaults(&config, &mut runtime);
                }
                Ok(None) if crate::project_python_selection_changed(&root, cli, None) => return Report::error(OP, "PROJECT_CONFIG_CONFLICT", "项目 Python 选择已变化。", "重新检查配置后重试。 "),
                Ok(None) => {}, Err(error) => return Report::error(OP, "PROJECT_CONFIG_INVALID", error, "检查项目配置。"),
            }
            let interface = loop_interface.unwrap_or("model-context");
            if !matches!(interface, "context" | "model-context") { return Report::error(OP, "GENERIC_LOOP_INVALID", "循环签名无效。", "选择 context 或 model-context。 "); }
            (entry, params, runtime, None, interface.to_owned())
        };
        runtime["timeout_seconds"] = json!(cli.timeout.unwrap_or(if detach { 3_153_600_000 } else { 86400 }));
        if let Err(report) = super::run::resolve_worker_directory(OP, &root, &mut runtime, resume.is_some()) { return report; }
        if let Err(report) = super::run::resolve_worker_import_roots(OP, &root, &mut runtime, resume.is_some()) { return report; }
        let mut request = json!({"protocol_version":PROTOCOL_VERSION, "operation":OP, "project_root":root,
            "training_loop":entry, "loop_interface":loop_interface, "training_mode":"generic-external-loop", "training_params":params,
            "steps":steps, "run_name":name, "runtime":runtime, "parent_run_id":resume, "request_id":request_id,
            "interactive_progress":context.show_progress});
        if let Some((path, digest)) = parent_checkpoint {
            request["generic_checkpoint_input"] = json!(path);
            request["expected_checkpoint_sha256"] = json!(digest);
        }
        let mut stored = request.clone();
        let mut redacted = Vec::new();
        redact_run_inputs(&mut stored, "$", &mut redacted);
        stored["redacted_fields"] = json!(redacted);
        let id = match self.runs.create(&root, &stored) {
            Ok(id) => id, Err(error) => return generic_run_create_error(OP, error, request_id),
        };
        request["run_id"] = json!(id);
        let directory = match self.runs.run_directory(&root, &id) {
            Ok(value) => value, Err(error) => {
                let report = Report::error(OP, "RUN_DIRECTORY_UNAVAILABLE", error, "用户代码未执行。 ");
                return self.finish_run_report(&root, &id, OP, &request, report);
            }
        };
        let checkpoint = match self.runs.checkpoint_dir(&root, &id, false) {
            Ok(value) => value, Err(error) => {
                let report = Report::error(OP, "RUN_DIRECTORY_UNAVAILABLE", error, "用户代码未执行。 ");
                return self.finish_run_report(&root, &id, OP, &request, report);
            }
        };
        request["snapshot_dir"] = json!(directory.join("snapshot"));
        request["checkpoint_dir"] = json!(checkpoint);
        request["stop_file"] = json!(directory.join("stop.request"));
        request["progress_path"] = json!(directory.join("progress.jsonl"));
        request["log_path"] = json!(directory.join("worker.log"));
        if detach { return self.launch_detached_training(&root, &id, &mut request); }
        if let Some(token) = self.supervisor.identify(std::process::id()) {
            if let Err(error) = self.runs.set_supervisor(&root, &id, std::process::id(), &token) {
                let report = Report::error(OP, "RUN_SUPERVISOR_REGISTRATION_FAILED", error, "用户代码未执行。 ");
                return self.finish_run_report(&root, &id, OP, &request, report);
            }
        }
        let report = self.backend.execute(&request);
        self.finish_run_report(&root, &id, OP, &request, report)
    }

    pub(super) fn run_generic_trial(
        &self,
        cli: &Cli,
        training_loop: &str,
        loop_interface: &str,
        raw_params: &[String],
        steps: u32,
        name: Option<&str>,
        context: ExecutionContext<'_>,
    ) -> Report {
        const OPERATION: &str = "generic_loop_trial";
        if !matches!(loop_interface, "context" | "model-context") {
            return Report::error(OPERATION, "GENERIC_LOOP_INVALID", "循环签名无效。", "选择 context 或 model-context。 ");
        }
        if cli.if_input_content.is_some() || cli.if_source_manifest.is_some() || cli.if_source_plan.is_some()
            || cli.if_source_config.is_some() || cli.capture_input_to.is_some() || cli.replay_input_from.is_some() {
            return Report::error(OPERATION, "GENERIC_LOOP_OPTION_UNSUPPORTED", "用户循环不能使用模型输入或源码编辑条件。", "移除输入快照、输入内容和源码编辑条件。 ");
        }
        if cli.variant.is_some() || cli.if_variant_base.is_some() || cli.seed.is_some() {
            return Report::error(OPERATION, "GENERIC_LOOP_OPTION_UNSUPPORTED",
                "纯 Python 循环不使用模型变体或工具管理的随机种子。",
                "移除 --variant、--if-variant-base 和 --seed；需要随机种子时通过 --loop-param 交给用户循环。 ");
        }
        let python = match require_python(cli, OPERATION) {
            Ok(value) => value,
            Err(report) => return report,
        };
        if !python_entrypoint(training_loop) || training_loop.len() > 1024 {
            return Report::error(OPERATION, "GENERIC_LOOP_INVALID", "循环入口必须是 module:callable。", "提供所选 Python 中可导入的用户循环。 ");
        }
        if steps == 0 || steps > 1000 {
            return Report::error(OPERATION, "TRAINING_PARAMETER_INVALID", "试跑步数必须在 1 到 1000 之间。", "调整 --steps。 ");
        }
        let params = match parameters(raw_params) {
            Ok(value) if value.len() <= 64 && serde_json::to_vec(&value).is_ok_and(|raw| raw.len() <= 65536) => value,
            _ => return Report::error(OPERATION, "GENERIC_LOOP_PARAMS_INVALID", "循环参数必须是最多 64 项、总计不超过 64 KiB 的 JSON。", "使用 --loop-param NAME=JSON；不要填写密钥。"),
        };
        let name = match name {
            Some(value) if value.trim().is_empty() || value.chars().count() > 80 || value.chars().any(char::is_control) => {
                return Report::error(OPERATION, "RUN_NAME_INVALID", "运行名称须为 1 到 80 个可见字符。", "调整运行名称。 ");
            }
            Some(value) => Some(value.trim()),
            None => None,
        };
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(value) => value,
            Err(error) => return Report::error(OPERATION, "PROJECT_UNAVAILABLE", error.to_string(), "检查项目目录。"),
        };
        let mut runtime = json!({"python_executable":python, "timeout_seconds":cli.timeout.unwrap_or(120)});
        match self.project_config.load(&root) {
            Ok(Some(config)) => {
                if crate::project_python_selection_changed(&root, cli, config["runtime_python"].as_str()) {
                    return Report::error(OPERATION, "PROJECT_CONFIG_CONFLICT", "项目 Python 选择已变化。", "重新检查配置后重试。 ");
                }
                super::run::apply_project_runtime_defaults(&config, &mut runtime);
            }
            Ok(None) if crate::project_python_selection_changed(&root, cli, None) => {
                return Report::error(OPERATION, "PROJECT_CONFIG_CONFLICT", "项目 Python 选择已变化。", "重新检查配置后重试。 ");
            }
            Ok(None) => {}
            Err(error) => return Report::error(OPERATION, "PROJECT_CONFIG_INVALID", error, "检查项目配置。"),
        }
        if let Err(report) = super::run::resolve_worker_directory(OPERATION, &root, &mut runtime, false) { return report; }
        if let Err(report) = super::run::resolve_worker_import_roots(OPERATION, &root, &mut runtime, false) { return report; }
        let mut request = json!({"protocol_version":PROTOCOL_VERSION, "operation":OPERATION,
            "project_root":root, "training_loop":training_loop, "loop_interface":loop_interface, "training_mode":"generic-external-loop",
            "training_params":params, "steps":steps, "run_name":name, "runtime":runtime,
            "interactive_progress":context.show_progress});
        let mut stored = request.clone();
        let mut redacted = Vec::new();
        redact_run_inputs(&mut stored, "$", &mut redacted);
        stored["redacted_fields"] = json!(redacted);
        if let Err(report) = self.reconcile_lost_runs(&root, OPERATION) { return report; }
        let id = match self.runs.create(&root, &stored) {
            Ok(id) => id,
            Err(error) => return generic_run_create_error(OPERATION, error, None),
        };
        request["run_id"] = json!(id);
        let directory = match self.runs.run_directory(&root, &id) {
            Ok(value) => value,
            Err(error) => {
                let report = Report::error(OPERATION, "RUN_DIRECTORY_UNAVAILABLE", error, "用户代码未执行。 ");
                return self.finish_run_report(&root, &id, OPERATION, &request, report);
            }
        };
        request["stop_file"] = json!(directory.join("stop.request"));
        request["progress_path"] = json!(directory.join("progress.jsonl"));
        if let Some(token) = self.supervisor.identify(std::process::id()) {
            if let Err(error) = self.runs.set_supervisor(&root, &id, std::process::id(), &token) {
                let report = Report::error(OPERATION, "RUN_SUPERVISOR_REGISTRATION_FAILED", error, "用户代码未执行。 ");
                return self.finish_run_report(&root, &id, OPERATION, &request, report);
            }
        }
        let report = self.backend.execute(&request);
        self.finish_run_report(&root, &id, OPERATION, &request, report)
    }
}
