use super::*;

impl<B: Backend> Application<B> {
    pub fn run(&self, cli: &Cli, command: &Command) -> Report {
        self.run_with_context(cli, command, ExecutionContext::default())
    }

    pub fn run_with_context(
        &self,
        cli: &Cli,
        command: &Command,
        context: ExecutionContext<'_>,
    ) -> Report {
        if cli.capture_input_to.is_some()
            && !matches!(
                command,
                Command::Model {
                    command: ModelCommand::Probe { .. }
                } | Command::Assembly {
                    command: AssemblyCommand::Patch { apply: false, .. }
                } | Command::Assembly {
                    command: AssemblyCommand::Route { dry_run: true, .. }
                        | AssemblyCommand::EncoderOutput { dry_run: true, .. }
                        | AssemblyCommand::Add { dry_run: true, .. }
                        | AssemblyCommand::Set { dry_run: true, .. }
                        | AssemblyCommand::Remove { dry_run: true, .. }
                }
            )
        {
            return Report::error(
                "probe",
                "INPUT_SNAPSHOT_UNSUPPORTED",
                "输入快照只能由前向诊断创建。",
                "先诊断，再使用输入快照进行预览或重验。",
            );
        }
        let assembly_replay = matches!(
            command,
            Command::Assembly {
                command: AssemblyCommand::Patch {
                    apply: true,
                    if_input_sha256: Some(_),
                    ..
                }
            }
        ) || matches!(
            command,
            Command::Assembly {
                command: AssemblyCommand::Route {
                    dry_run: false,
                    if_input_sha256: Some(_),
                    ..
                } | AssemblyCommand::EncoderOutput {
                    dry_run: false,
                    if_input_sha256: Some(_),
                    ..
                }
            }
        );
        if cli.replay_input_from.is_some()
            && ((!assembly_replay && cli.if_input_content.is_none())
                || !matches!(
                    command,
                    Command::Model {
                        command: ModelCommand::Probe { .. }
                    } | Command::Layer { .. }
                        | Command::Assembly {
                            command: AssemblyCommand::Patch { apply: true, .. }
                        }
                        | Command::Assembly {
                            command: AssemblyCommand::Route { dry_run: false, .. }
                                | AssemblyCommand::EncoderOutput { dry_run: false, .. }
                                | AssemblyCommand::Add { dry_run: false, .. }
                                | AssemblyCommand::Set { dry_run: false, .. }
                                | AssemblyCommand::Remove { dry_run: false, .. }
                        }
                ))
        {
            return Report::error(
                "probe",
                "INPUT_SNAPSHOT_UNSUPPORTED",
                "重放快照需要受支持的前向操作和原输入指纹。",
                "指定 --if-input-content 与诊断报告中的摘要。",
            );
        }
        if let Some(expected) = &cli.if_input_content {
            if expected.len() != 64
                || !expected
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Report::error(
                    "probe",
                    "INPUT_CONTENT_INVALID",
                    "输入内容指纹格式无效。",
                    "使用先前报告中的 sha256-forward-input-v1 摘要。",
                );
            }
            if !matches!(
                command,
                Command::Model {
                    command: ModelCommand::Probe { .. }
                } | Command::Layer { .. }
                    | Command::Assembly {
                        command: AssemblyCommand::Add { .. }
                            | AssemblyCommand::Set { .. }
                            | AssemblyCommand::Remove { .. }
                    }
            ) {
                return Report::error(
                    "probe",
                    "INPUT_CONTENT_UNSUPPORTED",
                    "输入内容条件只支持前向诊断、Layer 编辑或单项装配编辑。",
                    "移除该条件，或使用支持的操作。",
                );
            }
        }
        if let Some(expected) = &cli.if_source_manifest {
            if expected.len() != 64
                || !expected
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Report::error(
                    "edit",
                    "SOURCE_MANIFEST_INVALID",
                    "源码清单摘要格式无效。",
                    "使用先前预览报告中的摘要。",
                );
            }
            if !matches!(
                command,
                Command::Layer { .. }
                    | Command::Assembly {
                        command: AssemblyCommand::Add { .. }
                            | AssemblyCommand::Set { .. }
                            | AssemblyCommand::Remove { .. }
                    }
            ) {
                return Report::error(
                    "edit",
                    "SOURCE_MANIFEST_UNSUPPORTED",
                    "源码清单条件只支持 Layer 编辑或单项装配编辑。",
                    "移除该条件，或使用受支持的编辑操作。",
                );
            }
        }
        if let Some(expected) = &cli.if_source_config {
            if expected.len() != 64
                || !expected
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Report::error(
                    "edit",
                    "PROJECT_CONFIG_REVISION_INVALID",
                    "配置修订格式无效。",
                    "使用预览返回的完整 SHA256。",
                );
            }
            if !matches!(command, Command::Layer { .. }) {
                return Report::error(
                    "edit",
                    "SOURCE_CONFIG_UNSUPPORTED",
                    "项目配置条件仅支持源码 Layer 编辑。",
                    "移除该条件，或使用 --target source。 ",
                );
            }
        }
        if let Some(expected) = &cli.if_source_plan {
            if expected.len() != 64
                || !expected
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Report::error(
                    "edit",
                    "SOURCE_PLAN_INVALID",
                    "源码计划摘要格式无效。",
                    "使用先前预览的计划摘要。",
                );
            }
            if !matches!(command, Command::Layer { .. }) {
                return Report::error(
                    "edit",
                    "SOURCE_PLAN_UNSUPPORTED",
                    "源码计划条件仅支持源码 Layer 编辑。",
                    "移除该条件或选择源码 Layer 编辑。",
                );
            }
        }
        if cli.if_variant_base.is_some()
            && !matches!(
                command,
                Command::Layer { .. }
                    | Command::Try { .. }
                    | Command::Train { .. }
                    | Command::Model {
                        command: ModelCommand::Probe { .. }
                    }
            )
        {
            return Report::error(
                "edit",
                "VARIANT_BASE_UNSUPPORTED",
                "变体基准条件只支持变体编辑提交、指定变体的前向诊断或新训练。",
                "移除该条件，或使用支持的操作。",
            );
        }
        let semantic_store_action = match command {
            Command::Model {
                command: ModelCommand::Marks { inputs },
            } if inputs.model.is_none() && inputs.batch.is_none() => {
                Some(("semantic_mark", json!({"action":"list"})))
            }
            Command::Model {
                command: ModelCommand::Unmark { id },
            } => Some(("semantic_mark", json!({"action":"remove","id":id}))),
            Command::Model {
                command:
                    ModelCommand::Supervise {
                        command:
                            SuperviseCommand::Rule {
                                command:
                                    SupervisionRuleCommand::Add {
                                        mark,
                                        kind,
                                        mode,
                                        min,
                                        max,
                                        weight,
                                        target_path,
                                        axis,
                                        id,
                                        plugin,
                                        plugin_params,
                                    },
                            },
                    },
            } => Some((
                "semantic_supervise",
                json!({
                    "action":"rule_add", "mark_id":mark,
                    "kind":match kind { SupervisionRuleKind::Finite => "finite", SupervisionRuleKind::Range => "range", SupervisionRuleKind::BatchAlignment => "batch-alignment", SupervisionRuleKind::CustomValidator => "custom-validator", SupervisionRuleKind::CustomPenalty => "custom-penalty" },
                    "mode":match mode { SupervisionRuleMode::Observe => "observe", SupervisionRuleMode::Require => "require", SupervisionRuleMode::Penalty => "penalty" },
                    "min":min,"max":max,"weight":weight,"target_path":target_path,"axis":axis,"id":id,
                    "plugin_entrypoint":plugin,"plugin_params":plugin_params
                }),
            )),
            Command::Model {
                command:
                    ModelCommand::Supervise {
                        command:
                            SuperviseCommand::Rule {
                                command: SupervisionRuleCommand::List,
                            },
                    },
            } => Some(("semantic_supervise", json!({"action":"rule_list"}))),
            Command::Model {
                command:
                    ModelCommand::Supervise {
                        command:
                            SuperviseCommand::Rule {
                                command: SupervisionRuleCommand::Remove { id, expected_rule },
                            },
                    },
            } => Some((
                "semantic_supervise",
                json!({"action":"rule_remove","id":id,"expected_rule":expected_rule}),
            )),
            _ => None,
        };
        if let Some((operation, action)) = semantic_store_action {
            let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
                Ok(root) => root,
                Err(error) => {
                    return Report::error(operation, "PROJECT_UNAVAILABLE", error.to_string(), "");
                }
            };
            return self.backend.execute(&json!({
                "protocol_version": PROTOCOL_VERSION,
                "operation": operation,
                "project_root": root,
                "edit": action,
                "runtime": {
                    "python_executable": cli.python.as_deref().unwrap_or("python3"),
                    "timeout_seconds": cli.timeout.unwrap_or(30)
                }
            }));
        }
        if let Command::Assembly { command } = command {
            return self.assembly(cli, command, context);
        }
        if let Command::Config { command } = command {
            return self.project_config_command(command);
        }
        if let Command::Create {
            preset,
            data,
            apply,
            if_plan,
        } = command
        {
            return self.create_classification(cli, preset, data, *apply, if_plan.as_deref());
        }
        if let Command::Scaffold {
            module,
            batch,
            batch_binding,
            batch_params,
            injections,
            injection_params,
            inputs,
            routes,
            native_encoders,
            output_dim,
            outputs,
            loss,
            apply,
            if_plan,
        } = command
        {
            return self.scaffold_model(
                module.as_deref(),
                batch.as_deref(),
                batch_binding.as_ref(),
                batch_params,
                injections,
                injection_params,
                inputs,
                routes,
                native_encoders,
                *output_dim,
                outputs,
                loss.as_deref(),
                *apply,
                if_plan.as_deref(),
            );
        }
        if let Command::Init {
            model,
            batch,
            injections,
            injection_params,
            batch_binding,
            dry_run,
        } = command
        {
            return self.initialize_project(InitOptions {
                model: model.as_ref(),
                batch: batch.as_ref(),
                injections,
                injection_params,
                batch_binding: batch_binding.as_ref(),
                batch_params: &[],
                dry_run: *dry_run,
            });
        }
        if let Some(report) = environment::handle(self, cli, command, context) {
            return report;
        }
        if let Command::Data { command } = command {
            return super::data_preview::handle(self, cli, command);
        }
        if let Command::Augment { command } = command {
            return self.augment(command);
        }
        if matches!(
            command,
            Command::Layer {
                command: LayerCommand::Catalog
            }
        ) {
            if cli.if_source_plan.is_some()
                || cli.if_source_config.is_some()
                || cli.if_source_manifest.is_some()
                || cli.if_variant_base.is_some()
                || cli.if_input_content.is_some()
                || cli.replay_input_from.is_some()
                || cli.capture_input_to.is_some()
            {
                return Report::error(
                    "layer_catalog",
                    "CLI_ARGUMENT_INVALID",
                    "目录查询不能携带编辑或输入条件。",
                    "移除这些条件后重试。",
                );
            }
            return super::edit::layer_catalog();
        }
        if let Command::Variant { command } = command {
            return self.handle_variant(command);
        }
        if let Command::Component { command } = command {
            return self.handle_component(cli, command, context);
        }
        if let Command::Export {
            directory,
            run,
            if_checkpoint_sha256,
            overwrite,
        } = command
        {
            return self.handle_export(
                cli,
                directory,
                run.as_deref(),
                if_checkpoint_sha256.as_deref(),
                *overwrite,
            );
        }
        if let Command::Runs { command } = command {
            return self.handle_runs(command);
        }
        if let Command::Evaluate {
            run,
            checkpoint,
            if_checkpoint_sha256,
            split,
            evaluator,
            data,
            samples,
            preview_samples,
            data_fingerprint,
            params,
            data_params,
        } = command
        {
            return super::evaluation::handle(
                self,
                cli,
                run,
                checkpoint,
                if_checkpoint_sha256.as_deref(),
                split,
                evaluator,
                data,
                *samples,
                *preview_samples,
                data_fingerprint.as_deref(),
                params,
                data_params,
            );
        }
        let expected_training_config = match command {
            Command::Try { inputs, .. } | Command::Train { inputs, .. } => {
                inputs.if_config_revision.as_deref()
            }
            _ => None,
        };
        if let Some(expected) = expected_training_config {
            if expected.len() != 64
                || !expected
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Report::error(
                    "train",
                    "PROJECT_CONFIG_REVISION_INVALID",
                    "配置修订格式无效。",
                    "使用预览返回的完整 SHA256。 ",
                );
            }
        }
        let mut job = match command {
            Command::Try { .. } | Command::Train { .. } => {
                match self.build_training_job(cli, command) {
                    Ok(job) => job,
                    Err(report) => return report,
                }
            }
            Command::Model { command } => super::model::build_model_job(command),
            Command::Data { .. } => unreachable!(),
            Command::Augment { .. } => unreachable!(),
            Command::Layer { command } => match super::edit::build_layer_job(command) {
                Ok(job) => job,
                Err(report) => return report,
            },
            Command::Doctor => unreachable!(),
            Command::Completions { .. } => unreachable!("main handles shell completion generation"),
            Command::Assembly { .. } => unreachable!(),
            Command::Config { .. } => unreachable!(),
            Command::Create { .. } => unreachable!(),
            Command::Scaffold { .. } => unreachable!(),
            Command::Env { .. } => unreachable!(),
            Command::Init { .. } => unreachable!(),
            Command::Component { .. } => unreachable!(),
            Command::Variant { .. } => unreachable!(),
            Command::Runs { .. } => unreachable!(),
            Command::Export { .. } => unreachable!(),
            Command::Evaluate { .. } => unreachable!(),
        };
        if let Command::Try { inputs, .. } | Command::Train { inputs, .. } = command {
            if let Some(expected) = inputs.if_supervision_revision.as_deref() {
                if !job.apply_supervision
                    || expected.len() != 64
                    || !expected
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Report::error(
                        job.operation,
                        "SUPERVISION_REVISION_INVALID",
                        "监督修订无效或未启用监督。",
                        "启用监督并使用规则列表返回的完整 SHA256。 ",
                    );
                }
                job.expected_supervision_revision = Some(expected.to_owned());
            }
        }
        if cli.if_variant_base.is_some()
            && matches!(job.operation, "probe" | "training_probe" | "train")
            && cli.variant.is_none()
        {
            return Report::error(
                job.operation,
                "VARIANT_BASE_UNSUPPORTED",
                "条件操作需要显式指定变体。",
                "使用 --variant 指定提交的变体名称。",
            );
        }
        if cli.if_source_manifest.is_some()
            && (job.edit.is_none()
                || !job.require_forward
                || job
                    .edit
                    .as_ref()
                    .is_some_and(|edit| matches!(edit["action"].as_str(), Some("undo" | "redo"))))
        {
            return Report::error(
                "edit",
                "SOURCE_MANIFEST_UNSUPPORTED",
                "源码清单条件要求执行前向的 Layer 编辑，不支持历史操作。",
                "指定 --require-forward 并选择 Layer 编辑。",
            );
        }
        if cli.if_source_config.is_some()
            && (job.edit.is_none()
                || !job.require_forward
                || !job
                    .edit
                    .as_ref()
                    .is_some_and(|edit| edit["target"] == "source")
                || job
                    .edit
                    .as_ref()
                    .is_some_and(|edit| matches!(edit["action"].as_str(), Some("undo" | "redo"))))
        {
            return Report::error(
                "edit",
                "SOURCE_CONFIG_UNSUPPORTED",
                "配置条件要求执行前向的源码编辑，不支持历史操作。",
                "指定 --target source 和 --require-forward。",
            );
        }
        if cli.if_source_plan.is_some()
            && (job.edit.is_none()
                || !job.require_forward
                || !job
                    .edit
                    .as_ref()
                    .is_some_and(|edit| edit["target"] == "source")
                || job
                    .edit
                    .as_ref()
                    .is_some_and(|edit| matches!(edit["action"].as_str(), Some("undo" | "redo"))))
        {
            return Report::error(
                "edit",
                "SOURCE_PLAN_UNSUPPORTED",
                "源码计划条件要求执行前向的源码编辑，不支持历史操作。",
                "指定 --target source 和 --require-forward。 ",
            );
        }
        if cli.if_variant_base.is_some()
            && !matches!(job.operation, "probe" | "training_probe" | "train")
            && (job.edit.is_none()
                || job.dry_run
                || job
                    .edit
                    .as_ref()
                    .is_some_and(|edit| edit["target"] == "source"))
        {
            return Report::error(
                "edit",
                "VARIANT_BASE_UNSUPPORTED",
                "变体基准条件只支持实际的变体编辑提交。",
                "先使用 --dry-run 获取基准，再对变体提交指定条件。",
            );
        }
        if cli.if_variant_base.is_some()
            && matches!(job.operation, "training_probe" | "train")
            && (job.variant_snapshot.is_some()
                || job.repeat_of.is_some()
                || job.parent_run_id.is_some())
        {
            return Report::error(
                job.operation,
                "VARIANT_BASE_UNSUPPORTED",
                "历史运行的变体使用已保存快照。",
                "恢复、重复或迁移不支持当前变体条件；请检查原运行快照。 ",
            );
        }
        if cli.if_input_content.is_some()
            && job.operation != "probe"
            && (job.edit.is_none()
                || !job.require_forward
                || job
                    .edit
                    .as_ref()
                    .is_some_and(|edit| matches!(edit["action"].as_str(), Some("undo" | "redo"))))
        {
            return Report::error(
                "edit",
                "INPUT_CONTENT_UNSUPPORTED",
                "输入内容条件要求 Layer 编辑执行候选前向，不支持历史操作。",
                "提供 --batch 和 --require-forward，或移除该条件。",
            );
        }
        if cli.replay_input_from.is_some()
            && job.operation != "probe"
            && (job.edit.is_none()
                || !job.require_forward
                || job
                    .edit
                    .as_ref()
                    .is_some_and(|edit| matches!(edit["action"].as_str(), Some("undo" | "redo"))))
        {
            return Report::error(
                "edit",
                "INPUT_SNAPSHOT_UNSUPPORTED",
                "输入快照仅支持需要前向的 Layer 编辑，不支持历史操作。",
                "提供 --require-forward 并选择 Layer 编辑。",
            );
        }
        if job.operation != "training_probe" && job.operation != "train" {
            job.runtime = json!({
                "python_executable": cli.python.as_deref().unwrap_or("python3"),
                "timeout_seconds": cli.timeout.unwrap_or(30),
            });
            job.seed = cli.seed;
        }
        let root = match std::env::current_dir().and_then(|p| p.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error(job.operation, "PROJECT_UNAVAILABLE", error.to_string(), "");
            }
        };
        if let Err(report) = self.resolve_project_defaults(cli, &root, &mut job) {
            return report;
        }
        if let Some(expected) = &cli.if_source_config {
            if job.config_revision.as_deref() != Some(expected.as_str()) {
                return Report::error(
                    "edit",
                    "PROJECT_CONFIG_CONFLICT",
                    "项目配置与源码编辑预览不一致或无法核对，未修改源码。",
                    "重新查看项目配置并预览源码编辑。",
                );
            }
        }
        if expected_training_config.is_some()
            && (job.repeat_of.is_some() || job.parent_run_id.is_some())
        {
            return Report::error(
                job.operation,
                "PROJECT_CONFIG_GUARD_UNSUPPORTED",
                "此配置条件仅支持新训练或新试跑，不支持重复、迁移或续跑。",
                "移除条件并核对原运行快照。 ",
            );
        }
        if let (Some(expected), Some(actual)) =
            (expected_training_config, job.config_revision.as_deref())
        {
            if expected != actual {
                return Report::error(
                    job.operation,
                    "PROJECT_CONFIG_CONFLICT",
                    "项目配置与训练预览不一致，运行未启动。",
                    "重新查看训练计划。 ",
                );
            }
        }
        if let Err(report) = self.resolve_replayed_input_sources(&root, &mut job) {
            return report;
        }
        if let Err(report) = self.validate_training_inputs(cli, &job) {
            return report;
        }
        let source_target = match self.prepare_edit_execution(cli, &mut job, &root) {
            Ok(super::edit::EditPreparation::Ready { source_target }) => source_target,
            Ok(super::edit::EditPreparation::Finished(report)) => return report,
            Err(report) => return report,
        };
        let prepared = match self.prepare_run_inputs(cli, &mut job, &root, source_target) {
            Ok(prepared) => prepared,
            Err(report) => return report,
        };
        if cli.if_input_content.is_some() && prepared.batch.entrypoint.is_none() {
            return Report::error(
                "probe",
                "BATCH_REQUIRED",
                "输入内容校验需要批次工厂。",
                "提供 --batch package:factory。",
            );
        }
        let mut request =
            match super::run_request::build_worker_request(&mut job, &root, &prepared, context) {
                Ok(request) => request,
                Err(report) => return report,
            };
        if matches!(job.operation, "training_probe" | "train") {
            if let Err(report) = self.attach_training_augmentation(
                &mut job,
                &root,
                &mut request,
                expected_training_config,
            ) {
                return report;
            }
        }
        if let Some(expected) = &cli.if_source_config {
            request["expected_source_config"] = json!(expected);
        }
        if let Some(expected) = &cli.if_input_content {
            request["expected_input_content"] = json!(expected);
            request["_disable_evidence_cache"] = json!(true);
        }
        for (key, path) in [
            ("capture_input_to", &cli.capture_input_to),
            ("replay_input_from", &cli.replay_input_from),
        ] {
            if let Some(path) = path {
                let Some(path) = path.to_str() else {
                    return Report::error(
                        job.operation,
                        "INPUT_SNAPSHOT_PATH_INVALID",
                        "输入快照路径无法以 UTF-8 表示。",
                        "使用可表示的路径。",
                    );
                };
                request[key] = json!(path);
                request["_disable_evidence_cache"] = json!(true);
            }
        }
        let run_id = match self.create_run_record(&root, &job, &mut request) {
            Ok(run_id) => run_id,
            Err(report) => return report,
        };
        if job.operation == "train" && job.detach {
            let Some(id) = run_id.as_deref() else {
                return Report::error(
                    "train",
                    "RUN_RECORD_MISSING",
                    "后台训练没有对应的运行记录。",
                    "重新启动训练；如果问题持续，请保留诊断信息。",
                );
            };
            return self.launch_detached_training(&root, id, &mut request);
        }
        if let Some(observer) = context.progress_observer {
            observer.worker_starting(
                job.operation,
                job.runtime["timeout_seconds"].as_u64().unwrap_or(30),
            );
        }
        let report = self.backend.execute(&request);
        let mut report = report;
        if report.success() && job.edit.is_some() && source_target {
            if let Some(expected) = &cli.if_input_content {
                let validation = &report.data["validation"];
                if validation["forward"] != "passed"
                    || validation["forward_evidence"] != "executed"
                    || validation["batch_content"]["status"] != "available"
                    || validation["batch_content"]["algorithm"] != "sha256-forward-input-v1"
                    || validation["batch_content"]["digest"].as_str() != Some(expected.as_str())
                    || (cli.replay_input_from.is_some()
                        && (validation["batch_snapshot"]["status"] != "replayed"
                            || validation["batch_snapshot"]["format"]
                                != "model-maven-forward-input-snapshot-v1"))
                {
                    return Report::error(
                        "edit",
                        "INPUT_CONTENT_EVIDENCE_MISSING",
                        "源码候选未证明使用了已确认的实际前向输入。",
                        "原文件没有修改；请使用支持输入条件的 CLI/worker 重新预览。",
                    );
                }
            }
            let patches = report.data["source_patches"]
                .as_array()
                .cloned()
                .unwrap_or_else(|| vec![report.data["source_patch"].clone()]);
            let valid = !patches.is_empty()
                && patches.len() <= 64
                && patches.iter().all(|patch| {
                    use sha2::{Digest, Sha256};
                    let valid_hash = |v: &Value| {
                        v.as_str().is_some_and(|s| {
                            s.len() == 64
                                && s.bytes()
                                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                        })
                    };
                    patch["path"].as_str().is_some_and(|s| !s.is_empty())
                        && valid_hash(&patch["before_sha256"])
                        && valid_hash(&patch["after_sha256"])
                        && patch["unified_diff"].is_string()
                        && patch["contents"].as_str().is_some_and(|contents| {
                            patch["after_sha256"].as_str()
                                == Some(
                                    format!("{:x}", Sha256::digest(contents.as_bytes())).as_str(),
                                )
                        })
                });
            if !valid {
                return Report::error(
                    "edit",
                    "SOURCE_PLAN_INVALID",
                    "源码候选计划不完整或内容摘要不一致。",
                    "原文件没有修改；重新预览。 ",
                );
            }
            let digest = match crate::domain::input::content_revision(&json!(patches)) {
                Ok(value) => value,
                Err(error) => {
                    return Report::error("edit", "SOURCE_PLAN_INVALID", error, "原文件没有修改。");
                }
            };
            if cli
                .if_source_plan
                .as_ref()
                .is_some_and(|expected| expected != &digest)
            {
                return Report::error(
                    "edit",
                    "SOURCE_PLAN_CONFLICT",
                    "源码候选与已确认的预览不同。",
                    "原文件没有修改；重新预览。 ",
                );
            }
            report.data["source_plan"] = json!({"algorithm":"canonical-json-sha256-v1", "scope":"planned-source-file-patches-v1", "digest":digest});
            let manifest = &report.data["source_manifest"];
            let available = manifest.as_object().is_some_and(|entries| {
                !entries.is_empty()
                    && entries.len() <= 4096
                    && entries.iter().all(|(path, hash)| {
                        let source = Path::new(path);
                        source.is_absolute()
                            && source.starts_with(&root)
                            && hash.as_str().is_some_and(|value| {
                                value.len() == 64
                                    && value
                                        .bytes()
                                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                            })
                    })
            });
            if available {
                let manifest_digest = match crate::domain::input::content_revision(manifest) {
                    Ok(value) => value,
                    Err(error) => {
                        return Report::error(
                            "edit",
                            "SOURCE_MANIFEST_UNAVAILABLE",
                            error,
                            "重新预览。",
                        );
                    }
                };
                if let Some(expected) = &cli.if_source_manifest {
                    if *expected != manifest_digest {
                        return Report::error(
                            "edit",
                            "SOURCE_MANIFEST_CONFLICT",
                            "当前项目源码清单与已确认预览不同。",
                            "原文件没有修改；重新预览。",
                        );
                    }
                    request["expected_source_files"] = manifest.clone();
                    request["expected_source_files_digest"] = json!(manifest_digest);
                }
                report.data["source_manifest_evidence"] = json!({"algorithm":"canonical-json-sha256-v1", "scope":"worker-recorded-project-sources", "digest":manifest_digest});
            } else if cli.if_source_manifest.is_some() {
                return Report::error(
                    "edit",
                    "SOURCE_MANIFEST_UNAVAILABLE",
                    "源码候选缺少可核对的项目源码清单。",
                    "原文件没有修改；检查模型来源并重新预览。",
                );
            }
        }
        if report.success() && job.edit.is_some() && !source_target {
            let manifest = &report.data["variant"]["source_manifest"];
            let available = manifest.as_object().is_some_and(|entries| {
                !entries.is_empty()
                    && entries.iter().all(|(path, digest)| {
                        !path.is_empty()
                            && digest.as_str().is_some_and(|digest| {
                                digest.len() == 64
                                    && digest
                                        .bytes()
                                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                            })
                    })
            });
            if available {
                let digest = match crate::domain::input::content_revision(manifest) {
                    Ok(digest) => digest,
                    Err(error) => {
                        return Report::error(
                            "edit",
                            "SOURCE_MANIFEST_UNAVAILABLE",
                            error,
                            "重新预览。",
                        );
                    }
                };
                if let Some(expected) = &cli.if_source_manifest {
                    if expected != &digest {
                        return Report::error(
                            "edit",
                            "SOURCE_MANIFEST_CONFLICT",
                            "候选变体记录的项目源码清单与预览不同。",
                            "变体未保存；重新预览。 ",
                        );
                    }
                    use sha2::{Digest, Sha256};
                    for (path, expected_hash) in manifest.as_object().unwrap() {
                        let matches = std::fs::read(path).ok().is_some_and(|bytes| {
                            let actual = format!("{:x}", Sha256::digest(bytes));
                            expected_hash.as_str() == Some(actual.as_str())
                        });
                        if !matches {
                            return Report::error(
                                "edit",
                                "SOURCE_MANIFEST_CONFLICT",
                                "记录的源码或组件清单在提交前发生变化。",
                                "变体未保存；重新预览。 ",
                            );
                        }
                    }
                }
                report.data["source_manifest_evidence"] = json!({"algorithm":"canonical-json-sha256-v1", "digest":digest, "scope":"worker-recorded-project-sources-and-component-manifests", "note":"不覆盖外部库、权重、动态数据或验证后发生的并发变化"});
            } else if cli.if_source_manifest.is_some() {
                return Report::error(
                    "edit",
                    "SOURCE_MANIFEST_UNAVAILABLE",
                    "候选变体没有可核对的非空源码清单。",
                    "变体未保存；先检查模型来源。 ",
                );
            }
        }
        if job.edit.is_some() && !source_target {
            match super::run_prepare::variant_base_token(prepared.variant.as_ref()) {
                Ok(token) => {
                    if let Some(data) = report.data.as_object_mut() {
                        data.insert(
                            "variant_base".into(),
                            json!({
                                "name": prepared.variant_name,
                                "token": token,
                                "algorithm": "canonical-json-sha256-v1"
                            }),
                        );
                    }
                }
                Err(error) => {
                    return Report::error("edit", "VARIANT_BASE_UNAVAILABLE", error, "重新预览。");
                }
            }
        }
        let mut report = self.finalize_edit_result(
            &root,
            &job,
            &prepared.variant_name,
            &prepared.variant_path,
            source_target,
            &request,
            report,
        );
        if report.success()
            && job.edit.is_some()
            && !source_target
            && !job.dry_run
            && report.data["committed"] == true
            && report.data["variant"].is_object()
            && let Ok(token) = super::run_prepare::variant_base_token(Some(&report.data["variant"]))
        {
            report.data["variant_committed"] = json!({
                "name": prepared.variant_name,
                "token": token,
                "algorithm": "canonical-json-sha256-v1"
            });
        }
        let report = attach_input_provenance(&job, report);
        if let Some(id) = run_id {
            return self.finish_run_report(&root, &id, job.operation, &request, report);
        }
        report
    }
}

fn attach_input_provenance(job: &Job, mut report: Report) -> Report {
    if let Some(data) = report.data.as_object_mut() {
        if let Some(revision) = &job.config_revision {
            data.insert("config_revision".into(), json!(revision));
        }
        if !job.input_sources.is_empty() {
            data.insert("input_sources".into(), json!(job.input_sources));
        }
    }
    report
}
