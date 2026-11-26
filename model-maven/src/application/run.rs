use super::*;
use clap::Parser;

// Apply saved defaults only for a new execution. Explicit workspace choices
// are resolved afterwards; historical runs retain their recorded runtime.
pub(super) fn apply_project_runtime_defaults(config: &Value, runtime: &mut Value) {
    let Some(paths) = config.get("runtime_paths").and_then(Value::as_object) else { return; };
    if runtime["working_directory"].is_null() {
        if let Some(value) = paths.get("working_directory").filter(|value| !value.is_null()) {
            runtime["working_directory"] = value.clone();
        }
    }
    if runtime["project_import_roots"].is_null() {
        if let Some(value) = paths.get("project_import_roots").filter(|value| value.as_array().is_some_and(|roots| !roots.is_empty())) {
            runtime["project_import_roots"] = value.clone();
        }
    }
}

// Private workspace transport until every mutation/preview path supports the
// same directory contract. Persist the resolved value in the worker request.
pub(super) fn resolve_worker_directory(
    operation: &'static str,
    root: &std::path::Path,
    runtime: &mut Value,
    historical: bool,
) -> Result<(), Report> {
    let requested = std::env::var_os("MODEL_MAVEN_WORKSPACE_WORKING_DIRECTORY");
    let saved = runtime.get("working_directory").and_then(Value::as_str);
    if requested.is_none() && saved.is_none() {
        if !historical && runtime.get("working_directory").is_none() {
            runtime["working_directory"] = Value::Null;
        }
        return Ok(());
    }
    if !matches!(operation, "inspect" | "layers" | "probe" | "preview" | "verify" | "edit" | "semantic_mark" | "semantic_supervise" | "training_probe" | "generic_loop_trial" | "generic_loop_train" | "train" | "evaluate" | "environment_library_probe" | "environment_devices") {
        return Err(Report::error(operation, "WORKING_DIRECTORY_UNSUPPORTED",
            "当前操作尚不支持工作区选择的运行目录。", "使用默认目录或等待此操作支持独立工作目录。"));
    }
    let candidate = match requested {
        Some(value) => value.into_string().map_err(|_| Report::error(operation,
            "WORKING_DIRECTORY_INVALID", "运行目录不能表示为 UTF-8。", "请选择可表示的绝对目录。"))?,
        None => saved.unwrap_or_default().to_owned(),
    };
    let path = std::path::Path::new(&candidate);
    if !path.is_absolute() {
        return Err(Report::error(operation, "WORKING_DIRECTORY_INVALID", "运行目录必须是绝对路径。", "请选择已有的绝对目录。"));
    }
    let resolved = path.canonicalize().map_err(|error| Report::error(operation,
        "WORKING_DIRECTORY_UNAVAILABLE", error.to_string(), "检查运行目录是否存在且可访问。"))?;
    if !resolved.is_dir() {
        return Err(Report::error(operation, "WORKING_DIRECTORY_INVALID", "运行目录不是目录。", "请选择已有目录。"));
    }
    if historical {
        let prior = saved.map(std::path::Path::new).unwrap_or(root);
        if resolved != prior {
            return Err(Report::error(operation, "WORKING_DIRECTORY_CONFLICT",
                "所选运行目录与历史运行不同。", "使用历史运行的目录；当前尚不支持改变目录后恢复或重复。"));
        }
        if saved.is_none() {
            // A legacy/default run recorded no override. Selecting the same
            // project directory must not turn it into a new runtime identity.
            runtime["working_directory"] = Value::Null;
            return Ok(());
        }
    }
    let resolved = resolved.to_str().ok_or_else(|| Report::error(operation,
        "WORKING_DIRECTORY_INVALID", "运行目录不能表示为 UTF-8。", "请选择可表示的绝对目录。"))?;
    runtime["working_directory"] = json!(resolved);
    Ok(())
}

pub(super) fn resolve_worker_import_roots(
    operation: &'static str,
    root: &std::path::Path,
    runtime: &mut Value,
    historical: bool,
) -> Result<(), Report> {
    let requested = std::env::var_os("MODEL_MAVEN_WORKSPACE_IMPORT_ROOTS");
    let saved = runtime.get("project_import_roots").filter(|value| !value.is_null()).cloned();
    if requested.is_none() && saved.is_none() {
        if !historical && runtime.get("project_import_roots").is_none() {
            runtime["project_import_roots"] = Value::Null;
        }
        return Ok(());
    }
    if !matches!(operation, "inspect" | "layers" | "probe" | "preview" | "verify" | "edit" | "semantic_mark" | "semantic_supervise" | "training_probe" | "generic_loop_trial" | "generic_loop_train" | "train" | "evaluate" | "environment_library_probe" | "environment_devices") {
        return Err(Report::error(operation, "IMPORT_ROOTS_UNSUPPORTED", "当前操作尚不支持所选模块搜索目录。", "请先使用默认搜索路径，或等待该操作支持此配置。"));
    }
    let selected = match requested {
        Some(raw) => {
            let raw = raw.into_string().map_err(|_| Report::error(operation, "IMPORT_ROOTS_INVALID", "模块搜索目录无法表示为 UTF-8。", "请重新选择项目目录。"))?;
            if raw.len() > 32768 {
                return Err(Report::error(operation, "IMPORT_ROOTS_INVALID", "模块搜索目录列表过长。", "最多选择 32 个目录。"));
            }
            serde_json::from_str::<Value>(&raw).map_err(|error| Report::error(operation, "IMPORT_ROOTS_INVALID", error.to_string(), "使用 JSON 路径数组。"))?
        }
        None => saved.clone().unwrap_or_else(|| json!([])),
    };
    let roots = selected.as_array().filter(|roots| roots.len() <= 32).ok_or_else(|| Report::error(operation,
        "IMPORT_ROOTS_INVALID", "模块搜索目录必须是最多 32 项的路径数组。", "重新选择项目内目录。"))?;
    let mut seen = std::collections::HashSet::new();
    for value in roots {
        let path = value.as_str().filter(|s| !s.is_empty() && s.len() <= 1024 && !s.contains('\\') && !s.contains('\0'))
            .ok_or_else(|| Report::error(operation, "IMPORT_ROOTS_INVALID", "模块搜索目录无效。", "使用项目内相对目录。"))?;
        if path.split('/').any(|part| part.is_empty() || part == "." || part == "..") || !seen.insert(path) {
            return Err(Report::error(operation, "IMPORT_ROOTS_INVALID", "模块搜索目录重复或包含无效路径。", "使用不重复的项目内相对目录。"));
        }
        let relative = std::path::Path::new(path);
        if relative.is_absolute() || relative.components().any(|part| !matches!(part, std::path::Component::Normal(_))) {
            return Err(Report::error(operation, "IMPORT_ROOTS_INVALID", "模块搜索目录必须位于项目内。", "使用项目内相对目录。"));
        }
        let absolute = root.join(relative);
        let resolved = absolute.canonicalize().map_err(|error| Report::error(operation, "IMPORT_ROOT_UNAVAILABLE", error.to_string(), "检查项目模块搜索目录。"))?;
        if resolved != absolute || !resolved.starts_with(root) || !resolved.is_dir() {
            return Err(Report::error(operation, "IMPORT_ROOT_UNAVAILABLE", "模块搜索目录不可用或经过符号链接。", "选择项目内的普通目录，确保训练快照能包含其源码。"));
        }
    }
    if historical {
        let prior = saved.unwrap_or_else(|| json!([]));
        if selected != prior {
            return Err(Report::error(operation, "IMPORT_ROOTS_CONFLICT", "模块搜索目录与历史运行不同。", "使用历史运行的目录后重试。"));
        }
    }
    runtime["project_import_roots"] = if roots.is_empty() { Value::Null } else { selected };
    Ok(())
}

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
        self.run_with_context_mode(cli, command, context, false, None)
    }

    fn run_with_context_mode(
        &self,
        cli: &Cli,
        command: &Command,
        context: ExecutionContext<'_>,
        authorization_identity_only: bool,
        expected_authorization_identity: Option<&str>,
    ) -> Report {
        match super::scaffold::normalize_training_command(command) {
            Ok(Some(normalized)) => return self.run_with_context_mode(cli, &normalized, context, authorization_identity_only, expected_authorization_identity),
            Ok(None) => {},
            Err(report) => return report,
        }
        if let Command::Workspace { command: workspace_command } = command {
            let (request_json, identity_only, expected_identity) = match workspace_command {
                WorkspaceCommand::AuthorizationIdentity { request_json } => (request_json, true, None),
                WorkspaceCommand::Run { request_json, if_authorization_identity } =>
                    (request_json, false, Some(if_authorization_identity.as_str())),
            };
            if authorization_identity_only || expected_authorization_identity.is_some() {
                return Report::error("authorization_identity", "REQUEST_INVALID",
                    "授权身份请求不能嵌套。", "请传入普通 CLI 操作参数。");
            }
            let request = match serde_json::from_str::<Vec<String>>(request_json) {
                Ok(request) if !request.is_empty() && request.len() <= 512
                    && request.iter().all(|arg| arg.len() <= 32_768 && !arg.contains('\0')) => request,
                _ => return Report::error("authorization_identity", "REQUEST_INVALID",
                    "目标 CLI 参数必须是有界的 JSON 字符串数组。", "重新从工作区提交操作选择。"),
            };
            let parsed = Cli::try_parse_from(std::iter::once("m-model".to_owned()).chain(request))
                .map_err(|_| ())
                .and_then(|mut target_cli| {
                    if target_cli.command.is_none() { return Err(()); }
                    target_cli.python_explicit = target_cli.python.is_some() || cli.python_explicit;
                    if target_cli.python.is_none() { target_cli.python = cli.python.clone(); }
                    if target_cli.timeout.is_none() { target_cli.timeout = cli.timeout; }
                    target_cli.verbose = cli.verbose;
                    target_cli.json = true;
                    target_cli.no_interactive = true;
                    Ok(target_cli)
                });
            let target_cli = match parsed {
                Ok(target_cli) => target_cli,
                Err(()) => return Report::error("authorization_identity", "REQUEST_INVALID",
                    "目标 CLI 参数无效或缺少操作。", "重新从工作区提交操作选择。"),
            };
            let target_command = target_cli.command.clone().expect("validated target command");
            if target_cli.capture_input_to.is_some() || target_cli.replay_input_from.is_some() || !matches!(&target_command,
                Command::Model { command: ModelCommand::Inspect { .. } | ModelCommand::Probe { .. } }
                | Command::Layer { command: LayerCommand::List { .. } | LayerCommand::Show { .. } }) {
                if !identity_only {
                    return Report::error("authorization_identity", "AUTHORIZATION_IDENTITY_UNSUPPORTED",
                        "此操作没有可复用的授权身份解析路径。", "移除身份条件并逐次确认。 ");
                }
                return Report::ok("authorization_identity", json!({"identity": {
                    "complete": false,
                    "identity_digest": null,
                    "components": {},
                    "incomplete_reasons": ["该操作当前没有可复核的授权身份解析路径。"],
                    "action_class": "user_code",
                    "display_summary": "The action still requires per-run confirmation."
                }}));
            }
            return self.run_with_context_mode(&target_cli, &target_command, context, identity_only, expected_identity);
        }
        if let Command::Model { command: ModelCommand::Probe {
            contract: Some(path), if_contract_sha256: Some(expected), ..
        } } = command {
            use sha2::{Digest, Sha256};
            use std::io::Read;
            if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
                return Report::error("probe", "MODEL_CONTRACT_IDENTITY_INVALID", "契约摘要须为 64 位小写 SHA-256。", "重新选择并确认契约；用户代码尚未执行。");
            }
            if !std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() <= 8 * 1024 * 1024) {
                return Report::error("probe", "MODEL_CONTRACT_INVALID", "契约必须是最多 8 MiB 的可读取普通文件。", "检查契约文件；用户代码尚未执行。");
            }
            let mut file = match std::fs::File::open(path) {
                Ok(file) => file,
                Err(error) => return Report::error("probe", "MODEL_CONTRACT_INVALID", error.to_string(), "检查契约文件；用户代码尚未执行。"),
            };
            if !file.metadata().is_ok_and(|metadata| metadata.is_file() && metadata.len() <= 8 * 1024 * 1024) {
                return Report::error("probe", "MODEL_CONTRACT_INVALID", "契约必须是最多 8 MiB 的普通文件。", "检查契约文件；用户代码尚未执行。");
            }
            let mut bytes = Vec::new();
            if file.by_ref().take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes).is_err() || bytes.len() > 8 * 1024 * 1024 {
                return Report::error("probe", "MODEL_CONTRACT_INVALID", "契约读取失败或超过 8 MiB。", "检查契约文件；用户代码尚未执行。");
            }
            if format!("{:x}", Sha256::digest(&bytes)).as_str() != expected.as_str() {
                return Report::error("probe", "MODEL_CONTRACT_CHANGED", "契约内容与确认时不同。", "重新查看并确认契约；用户代码尚未执行。");
            }
        }
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
                    | Command::Model {
                        command: ModelCommand::Mark { .. }
                    }
                    | Command::Model {
                        command: ModelCommand::Retag { .. }
                    }
                    | Command::Model {
                        command: ModelCommand::Supervise {
                            command: SuperviseCommand::Preview { .. }
                        }
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
            Command::Model { command: ModelCommand::Supervise { command: SuperviseCommand::Export { if_state_revision } } } =>
                Some(("semantic_supervise", json!({"action":"bundle_export","if_state_revision":if_state_revision}))),
            Command::Model { command: ModelCommand::Supervise { command: SuperviseCommand::Import { file, if_sha256, if_state_revision, apply } } } =>
                Some(("semantic_supervise", json!({"action":"bundle_import","file":file,"if_sha256":if_sha256,"if_state_revision":if_state_revision,"apply":apply}))),
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
                                        if_state_revision,
                                        plugin,
                                        plugin_source,
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
                    "min":min,"max":max,"weight":weight,"target_path":target_path,"axis":axis,"id":id,"if_state_revision":if_state_revision,
                    "plugin_entrypoint":plugin,"plugin_source":plugin_source,"plugin_params":plugin_params
                }),
            )),
            Command::Model {
                command: ModelCommand::Supervise {
                    command: SuperviseCommand::Rule {
                        command: SupervisionRuleCommand::Copy { id, mark, if_state_revision },
                    },
                },
            } => Some(("semantic_supervise", json!({
                "action":"rule_copy", "id":id, "mark_id":mark, "if_state_revision":if_state_revision
            }))),
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
            let python = match require_python(cli, operation) {
                Ok(value) => value,
                Err(report) => return report,
            };
            let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
                Ok(root) => root,
                Err(error) => {
                    return Report::error(operation, "PROJECT_UNAVAILABLE", error.to_string(), "");
                }
            };
            let custom_rule = operation == "semantic_supervise" && action["action"] == "rule_add"
                && matches!(action["kind"].as_str(), Some("custom-validator" | "custom-penalty"));
            let mut request = json!({
                "protocol_version": PROTOCOL_VERSION,
                "operation": operation,
                "project_root": root,
                "edit": action,
                "runtime": {
                    "python_executable": python,
                    "timeout_seconds": cli.timeout.unwrap_or(30)
                }
            });
            if custom_rule {
                match self.project_config.load(&root) {
                    Ok(Some(config)) => apply_project_runtime_defaults(&config, &mut request["runtime"]),
                    Ok(None) => {},
                    Err(error) => return Report::error(operation, "PROJECT_CONFIG_INVALID", error,
                        "检查项目配置；不会忽略无效的运行时路径。"),
                }
                if let Err(report) = resolve_worker_directory(operation, &root, &mut request["runtime"], false) {
                    return report;
                }
                if let Err(report) = resolve_worker_import_roots(operation, &root, &mut request["runtime"], false) {
                    return report;
                }
            }
            return self.backend.execute(&request);
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
            recipe,
            apply,
            if_plan,
        } = command
        {
            return self.create_classification(cli, preset.as_deref(), data.as_deref(), recipe.as_deref(), *apply, if_plan.as_deref());
        }
        if let Command::RecipeUpdate { training, supervision, source, runtime, if_recipe, apply, if_plan } = command {
            return self.update_classification_recipe(training.as_deref(), supervision.as_deref(), source.as_deref(), runtime.as_deref(), if_recipe, *apply, if_plan.as_deref());
        }
        if let Command::RecipeSync { data, training_profile, if_recipe, if_config_revision, apply, if_plan } = command {
            return self.sync_scaffold_recipe(*data, training_profile.as_deref(), if_recipe, if_config_revision, *apply, if_plan.as_deref());
        }
        if let Command::RecipeLayerPlan { recipe, if_recipe, moves } = command {
            return super::scaffold::plan_layer_moves(recipe, if_recipe, moves);
        }
        if let Command::RecipeLayerEditPlan { recipe, if_recipe, operations, if_operations } = command {
            return super::scaffold::plan_layer_edits(recipe, if_recipe, operations, if_operations);
        }
        if let Command::GraphPlan { graph, if_graph } = command {
            return super::graph::plan(graph, if_graph);
        }
        if let Command::GraphEditPlan { graph, if_graph, operations, if_operations } = command {
            return super::graph_edit::plan(graph, if_graph, operations, if_operations);
        }
        if let Command::GraphImportScaffold { recipe, if_recipe, model, if_project, editable } = command {
            return super::scaffold::graph_import_preview(self, recipe, if_recipe, model, if_project.as_deref(), *editable);
        }
        if let Command::GraphReorderPlan { graph, if_graph, node, before, after } = command {
            return super::graph::reorder_plan(graph, if_graph, node, before.as_deref().or(after.as_deref()).unwrap_or(""), after.is_some());
        }
        if let Command::GraphGenerate { graph, if_graph, output, if_source, apply, if_plan } = command {
            return super::graph::generate(self, graph, if_graph, output.as_deref(), if_source.as_deref(), *apply, if_plan.as_deref());
        }
        if let Command::GraphDraftProbe { graph, if_graph, snapshot, input_sha256, batch, model_device, if_config } = command {
            return super::graph::draft_probe(self, cli, graph, if_graph, snapshot, input_sha256, batch.as_deref(), model_device.as_deref(), if_config);
        }
        if let Command::GraphBind { graph, if_graph, source, if_config, include_data, apply, if_plan } = command {
            return super::graph::bind_project(self, graph, if_graph, source, if_config, *include_data, *apply, if_plan.as_deref());
        }
        if let Command::GraphSave { draft, if_draft, output, if_base, apply, if_plan } = command {
            return super::graph::save(self, draft, if_draft, output, if_base, *apply, if_plan.as_deref());
        }
        if let Command::GraphDeliveryPlan { draft, if_draft, output, if_base, source } = command {
            return super::graph::delivery_plan(self, draft, if_draft, output, if_base, source);
        }
        if let Command::GraphDeliveryStart { draft, if_draft, output, if_base, source, if_plan } = command {
            return super::graph::delivery_start(self, draft, if_draft, output, if_base, source, if_plan);
        }
        if let Command::GraphDeliveryStatus = command {
            return super::graph::delivery_status(self);
        }
        if let Command::GraphDeliveryArchive { if_plan } = command {
            return super::graph::delivery_archive(if_plan);
        }
        if let Command::GraphDeliveryResume { if_plan } = command {
            return super::graph::delivery_resume(self, if_plan.as_deref());
        }
        if let Command::RecipeBundle { graph, scaffold, output, include, apply, if_plan } = command {
            return self.bundle_classification_recipe(*graph, *scaffold, output, include, *apply, if_plan.as_deref());
        }
        if let Command::Scaffold {
            recipe,
            write_recipe,
            update_recipe,
            reconcile_assembly,
            component_params_slots,
            training_profile,
            supervision_reference,
            module,
            batch,
            batch_binding,
            batch_params,
            injections,
            injection_params,
            inputs,
            routes,
            native_encoders,
            encoder_layers,
            encoder_widths,
            output_dim,
            outputs,
            loss,
            apply,
            if_plan,
        } = command
        {
            if let Some(recipe) = recipe {
                return self.scaffold_from_recipe(recipe, *write_recipe, update_recipe.as_deref(), *reconcile_assembly, *apply, if_plan.as_deref());
            }
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
                encoder_layers,
                encoder_widths,
                *output_dim,
                outputs,
                loss.as_deref(),
                *write_recipe,
                update_recipe.as_deref(),
                *reconcile_assembly,
                component_params_slots,
                training_profile.as_deref(),
                supervision_reference.as_deref(),
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
        // Mutations can change interpreter packages even on failure. An
        // explicit environment inspection is also a request for fresh facts;
        // don't retain an earlier capability result afterwards.
        if matches!(command,
            Command::Doctor
            | Command::Env { command: EnvironmentCommand::Setup { .. } | EnvironmentCommand::Configure { .. }
                | EnvironmentCommand::Check { .. } | EnvironmentCommand::Libraries { .. }
                | EnvironmentCommand::Probe { .. } | EnvironmentCommand::Devices { .. }
                | EnvironmentCommand::Wheelhouse { command: WheelhouseCommand::Replay { .. } } }
            | Command::Component { command: ComponentCommand::Install { .. } | ComponentCommand::Uninstall { apply: true, .. }
                | ComponentCommand::Discover { .. } }) {
            self.runtime_capabilities.borrow_mut().take();
            self.runtime_declarations.borrow_mut().take();
        }
        if let Some(report) = environment::handle(self, cli, command, context) {
            return report;
        }
        if let Command::Discover { import_roots } = command {
            return super::discovery::discover(import_roots);
        }
        if let Command::LoopTrial { training_loop, loop_interface, params, steps, name } = command {
            return self.run_generic_trial(cli, training_loop, loop_interface, params, *steps, name.as_deref(), context);
        }
        if let Command::LoopRun { training_loop, loop_interface, params, steps, resume, expected_checkpoint, name, detach, request_id } = command {
            return self.run_generic_formal(cli, training_loop.as_deref(), params, *steps,
                resume.as_deref(), expected_checkpoint.as_deref(), name.as_deref(), *detach, request_id.as_deref(), loop_interface.as_deref(), context);
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
            device,
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
                device.as_deref(),
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
            Command::RecipeUpdate { .. } => unreachable!(),
            Command::RecipeSync { .. } => unreachable!(),
            Command::RecipeLayerPlan { .. } => unreachable!(),
            Command::RecipeLayerEditPlan { .. } => unreachable!(),
            Command::GraphPlan { .. } => unreachable!(),
            Command::GraphImportScaffold { .. } => unreachable!(),
            Command::GraphReorderPlan { .. } => unreachable!(),
            Command::GraphGenerate { .. } | Command::GraphDraftProbe { .. } | Command::GraphBind { .. } | Command::GraphEditPlan { .. } => unreachable!(),
            Command::GraphSave { .. } | Command::GraphDeliveryPlan { .. } | Command::GraphDeliveryStart { .. } | Command::GraphDeliveryResume { .. } | Command::GraphDeliveryStatus | Command::GraphDeliveryArchive { .. } => unreachable!(),
            Command::RecipeBundle { .. } => unreachable!(),
            Command::Scaffold { .. } => unreachable!(),
            Command::Env { .. } => unreachable!(),
            Command::Init { .. } => unreachable!(),
            Command::Component { .. } => unreachable!(),
            Command::Variant { .. } => unreachable!(),
            Command::Runs { .. } => unreachable!(),
            Command::Export { .. } => unreachable!(),
            Command::Workspace { .. } => unreachable!(),
            Command::Evaluate { .. } => unreachable!(),
            Command::Discover { .. } => unreachable!(),
            Command::LoopTrial { .. } => unreachable!(),
            Command::LoopRun { .. } => unreachable!(),
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
            && matches!(job.operation, "probe" | "semantic_mark" | "semantic_supervise" | "training_probe" | "train")
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
            && !matches!(job.operation, "probe" | "semantic_mark" | "semantic_supervise" | "training_probe" | "train")
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
            let python = match require_python(cli, job.operation) {
                Ok(value) => value,
                Err(report) => return report,
            };
            job.runtime = json!({
                "python_executable": python,
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
        let observed_config = match self.resolve_project_defaults(cli, &root, &mut job) {
            Ok(value) => value,
            Err(report) => return report,
        };
        let historical = job.repeat_of.is_some() || job.parent_run_id.is_some();
        if !historical {
            match self.project_config.load(&root) {
                Ok(Some(config)) => {
                    let revision = match crate::domain::input::content_revision(&config) {
                        Ok(revision) => revision,
                        Err(error) => return Report::error(job.operation, "PROJECT_CONFIG_REVISION_FAILED", error,
                            "无法核对项目配置；请检查配置后重新运行。"),
                    };
                    if matches!(observed_config.as_ref(), Some(None)) {
                        return Report::error(job.operation, "PROJECT_CONFIG_CONFLICT",
                            "解析模型输入后出现了新的项目配置。", "重新运行以使用同一配置修订。");
                    }
                    if job.config_revision.as_deref().is_some_and(|expected| expected != revision.as_str()) {
                        return Report::error(job.operation, "PROJECT_CONFIG_CONFLICT",
                            "读取模型配置后项目运行时配置发生变化。", "重新运行以使用同一配置修订。");
                    }
                    if crate::project_python_selection_changed(&root, cli, config["runtime_python"].as_str()) {
                        return Report::error(job.operation, "PROJECT_CONFIG_CONFLICT",
                            "启动后项目 Python 选择发生变化。", "重新运行以使用当前项目解释器。");
                    }
                    apply_project_runtime_defaults(&config, &mut job.runtime);
                },
                Ok(None) if matches!(observed_config.as_ref(), Some(Some(_))) => return Report::error(job.operation,
                    "PROJECT_CONFIG_CONFLICT", "读取模型配置后项目配置已消失。", "检查项目配置后重新运行。"),
                Ok(None) if crate::project_python_selection_changed(&root, cli, None) => return Report::error(job.operation,
                    "PROJECT_CONFIG_CONFLICT", "启动后项目 Python 选择发生变化。", "重新运行以使用当前项目解释器。"),
                Ok(None) => {},
                Err(error) => return Report::error(job.operation, "PROJECT_CONFIG_INVALID", error,
                    "检查项目配置；不会忽略无效的运行时路径。"),
            }
        }
        if let Err(report) = resolve_worker_directory(job.operation, &root, &mut job.runtime, historical) {
            return report;
        }
        if let Err(report) = resolve_worker_import_roots(job.operation, &root, &mut job.runtime, historical) {
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
        let generated_training_seed = matches!(job.operation, "training_probe" | "train")
            && job.seed.is_none() && job.repeat_of.is_none() && job.checkpoint_input.is_none();
        if generated_training_seed {
            let mut bytes = [0_u8; 8];
            if let Err(error) = getrandom::fill(&mut bytes) {
                return Report::error(job.operation, "TRAINING_SEED_UNAVAILABLE",
                    format!("无法为新运行生成种子：{error}"), "检查系统随机数来源，或使用 --seed 明确指定；训练尚未启动。");
            }
            job.seed = Some(u64::from_le_bytes(bytes));
        }
        let mut request =
            match super::run_request::build_worker_request(&mut job, &root, &prepared, context) {
                Ok(request) => request,
                Err(report) => return report,
            };
        if job.operation == "semantic_mark" && prepared.variant.is_some() {
            // A human-readable hint only; the binding itself remains the
            // digest of the full selected variant, not this mutable name.
            request["semantic_variant_name"] = json!(prepared.variant_name);
        }
        // Independent workspace rechecks opt out of cached forward evidence.
        // Carry the choice in the request as well as the adapter's process-level
        // opt-in; it applies only to probes.
        if job.operation == "probe"
            && std::env::var_os("MODEL_MAVEN_FORCE_FRESH_FORWARD").as_deref()
                == Some(std::ffi::OsStr::new("1"))
        {
            request["_disable_evidence_cache"] = json!(true);
        }
        if let Some(policy) = &job.variant_device_policy {
            request["variant_device_policy"] = json!(policy);
        } else if job.operation == "edit"
            && request["variant"].is_null()
            && request["edit"]["target"].as_str().unwrap_or("variant") == "variant"
        {
            // New variants preserve factory placement by default. Existing
            // variants retain their persisted policy, including legacy CPU.
            request["variant_device_policy"] = json!("factory");
        }
        if let Some(devices) = &job.variant_base_rng_devices {
            request["variant_base_rng_devices"] = json!(devices);
        }
        if let Some(policy) = &job.batch_device_policy {
            request["batch_device_policy"] = json!(policy);
        }
        if let Some(policy) = &job.model_initialization {
            request["model_initialization_policy"] = json!(policy);
            request["model_initialization_seed"] = json!(job.model_initialization_seed);
        }
        if matches!(job.operation, "training_probe" | "train") && job.seed.is_some() {
            let seed_source = if generated_training_seed {
                "generated_for_new_run"
            } else if cli.seed.is_some() {
                "explicit"
            } else if job.repeat_of.is_some() {
                "repeated_run"
            } else if job.checkpoint_input.is_some() {
                "resumed_run"
            } else {
                "request"
            };
            request["seed_source"] = json!(seed_source);
        }
        if let Some(device) = &job.probe_device {
            if device.trim() != device || device.is_empty() || device.len() > 128 || device.chars().any(char::is_control) {
                return Report::error(job.operation, "PROBE_DEVICE_INVALID", "观察设备标识无效。", "填写所选 PyTorch 支持的设备标识。 ");
            }
            request["probe_device"] = json!(device);
        }
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
        if expected_authorization_identity.is_some_and(|value| value.len() != 64
            || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))) {
            return Report::error(job.operation, "AUTHORIZATION_IDENTITY_INVALID",
                "授权身份条件格式无效。", "重新取得完整身份快照后再确认。 ");
        }
        if authorization_identity_only {
            let identity = super::run_prepare::authorization_identity(
                &request,
                &root,
                job.config_revision.as_deref(),
                &job.input_sources,
                cli.python_explicit,
            );
            return Report::ok("authorization_identity", json!({"identity": identity}));
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
        if let Some(expected) = expected_authorization_identity {
            let current_identity = super::run_prepare::authorization_identity(
                &request,
                &root,
                job.config_revision.as_deref(),
                &job.input_sources,
                cli.python_explicit,
            );
            if current_identity["complete"] != true
                || current_identity["identity_digest"].as_str() != Some(expected)
            {
                return Report::error(job.operation, "AUTHORIZATION_IDENTITY_CONFLICT",
                    "提交 Python 前授权身份发生变化或不再完整；Python 尚未启动。",
                    "当前授权已失效。重新检查操作范围并逐次确认。 ");
            }
        }
        let report = self.backend.execute(&request);
        let mut report = report;
        let layer_edit = job.operation == "edit";
        if report.success() && layer_edit {
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
                        "Layer 候选未证明使用了已确认的实际前向输入。",
                        "源码或变体尚未保存；请使用支持输入条件的 CLI/worker 重新预览。",
                    );
                }
            }
        }
        if report.success() && layer_edit && source_target {
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
            let directory = request["runtime"]["working_directory"].as_str();
            let import_roots = request["runtime"].get("project_import_roots").filter(|v| !v.is_null());
            let (plan_value, plan_scope) = match (directory, import_roots) {
                (directory, Some(roots)) => (json!({"patches": patches, "working_directory": directory, "project_import_roots": roots}), "planned-source-patches-and-runtime-paths-v3"),
                (Some(directory), None) => (json!({"patches": patches, "working_directory": directory}), "planned-source-file-patches-and-working-directory-v2"),
                (None, None) => (json!(patches), "planned-source-file-patches-v1"),
            };
            let digest = match crate::domain::input::content_revision(&plan_value) {
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
            report.data["source_plan"] = json!({"algorithm":"canonical-json-sha256-v1", "scope":plan_scope, "digest":digest, "working_directory":directory, "project_import_roots":import_roots});
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
        if report.success() && layer_edit && !source_target {
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
                    if let Err(error) = self.verify_source_files(&root, manifest, &[], false) {
                        return Report::error(
                            "edit",
                            "SOURCE_MANIFEST_CONFLICT",
                            format!("记录的源码或组件清单无法核对：{error}"),
                            "变体未保存；重新预览。",
                        );
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
        if layer_edit && !source_target {
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
            && layer_edit
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
