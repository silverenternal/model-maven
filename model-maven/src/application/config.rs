use super::*;

pub(super) fn valid_assembly_output_path(relative: &Path) -> bool {
    !relative.as_os_str().is_empty()
        && !relative.is_absolute()
        && relative
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
}

fn validate_project_config_history(history: &[Value]) -> Result<Vec<Value>, String> {
    let mut validated = Vec::with_capacity(history.len());
    let mut revisions = std::collections::HashSet::new();
    for (index, entry) in history.iter().enumerate() {
        let revision = entry["revision"]
            .as_str()
            .ok_or_else(|| format!("历史记录第 {} 项缺少 revision。", index + 1))?;
        let config = entry
            .get("config")
            .filter(|config| config.is_object())
            .ok_or_else(|| format!("历史记录第 {} 项缺少配置对象。", index + 1))?;
        let actual = crate::domain::input::content_revision(config)?;
        if actual != revision {
            return Err(format!(
                "历史记录第 {} 项 revision 与配置内容不匹配。",
                index + 1
            ));
        }
        if !revisions.insert(revision.to_owned()) {
            return Err(format!("历史记录包含重复 revision：{revision}。"));
        }
        validated.push(entry.clone());
    }
    Ok(validated)
}

fn redacted_config_report(operation: &str, mut data: Value) -> Report {
    let mut fields = Vec::new();
    redact_run_inputs(&mut data, "$.config", &mut fields);
    data["redacted_fields"] = json!(fields);
    Report::ok(operation, data)
}

impl<B: Backend> Application<B> {
    pub(super) fn project_config_command(&self, command: &ProjectConfigCommand) -> Report {
        let operation = match command {
            ProjectConfigCommand::History => "config_history",
            ProjectConfigCommand::SetPython { .. } => "config_set_python",
            ProjectConfigCommand::SetRuntimePaths { .. } => "config_set_runtime_paths",
            ProjectConfigCommand::SetModel { .. } => "config_set_model",
            ProjectConfigCommand::SetBatch { .. } => "config_set_batch",
            ProjectConfigCommand::Restore { .. } => "config_restore",
        };
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error(
                    "config_history",
                    "PROJECT_UNAVAILABLE",
                    error.to_string(),
                    "在可访问的项目目录中重试。",
                );
            }
        };
        let current = match self.project_config.load(&root) {
            Ok(Some(config)) => config,
            Ok(None) => {
                return Report::error(
                    operation,
                    "PROJECT_NOT_INITIALIZED",
                    "当前项目尚无已保存的模型配置。",
                    "先运行 m-model init 保存模型入口。",
                );
            }
            Err(error) => {
                return Report::error(
                    operation,
                    "PROJECT_CONFIG_INVALID",
                    error,
                    "检查 .model-maven/project.json；原文件未修改。",
                );
            }
        };
        let current_revision = match crate::domain::input::content_revision(&current) {
            Ok(revision) => revision,
            Err(error) => {
                return Report::error(
                    operation,
                    "PROJECT_CONFIG_INVALID",
                    error,
                    "检查项目配置格式；原文件未修改。",
                );
            }
        };
        let history = match self.project_config.history(&root) {
            Ok(history) => history,
            Err(error) => {
                return Report::error(
                    operation,
                    "PROJECT_CONFIG_HISTORY_INVALID",
                    error,
                    "先备份 .model-maven/project.history.json，再检查其 JSON 格式。",
                );
            }
        };

        let validated_history = match validate_project_config_history(&history) {
            Ok(history) => history,
            Err(error) => {
                return Report::error(
                    operation,
                    "PROJECT_CONFIG_HISTORY_INVALID",
                    error,
                    "历史记录没有被修改；先备份并检查 .model-maven/project.history.json。",
                );
            }
        };
        match command {
            ProjectConfigCommand::History => redacted_config_report(
                operation,
                json!({
                    "current_revision":current_revision,
                    "history":validated_history,
                    "restore_fields":["model","model_params","batch","batch_binding","batch_params","model_injections","runtime_paths","runtime_python"]
                }),
            ),
            ProjectConfigCommand::SetPython { executable, clear, apply, if_current } => {
                if !*clear && executable.as_deref().is_none_or(|value| value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control)) {
                    return Report::error(operation, "PYTHON_SELECTION_INVALID", "解释器路径或命令无效。", "提供非空解释器路径或命令，不能包含控制字符。");
                }
                let python = if *clear { None } else { executable.as_deref() };
                let mut candidate = current.clone();
                if let Some(python) = python { candidate["runtime_python"] = json!(python); }
                else if let Some(object) = candidate.as_object_mut() { object.remove("runtime_python"); }
                if *apply && if_current.as_deref() != Some(current_revision.as_str()) {
                    return Report::error(operation, "PROJECT_CONFIG_CONFLICT", "项目配置与预览修订不一致。", "重新预览；配置未修改。");
                }
                let mut result = json!({"committed":false,"unchanged":candidate == current,"from_revision":current_revision,
                    "runtime_python":python,"previous_runtime_python":current["runtime_python"],
                    "note":"只保存解释器选择，不执行 Python 或安装依赖；历史运行仍使用其记录的环境。"});
                if *apply && candidate != current {
                    match self.project_config.update_versioned(&root, &current_revision, &candidate) {
                        Ok(update) => { result["committed"] = json!(true); result["config_revision"] = json!(update.revision); },
                        Err(ProjectConfigUpdateError::Conflict { .. }) => return Report::error(operation, "PROJECT_CONFIG_CONFLICT", "项目配置在提交时发生变化。", "重新预览。"),
                        Err(error) => return Report::error(operation, "PROJECT_CONFIG_WRITE_FAILED", error.message(), "检查项目配置和目录权限。"),
                    }
                }
                redacted_config_report(operation, result)
            }
            ProjectConfigCommand::SetRuntimePaths { working_directory, import_roots, apply, if_current } => {
                let directory = match working_directory {
                    Some(value) => {
                        let path = Path::new(value);
                        if !path.is_absolute() {
                            return Report::error(operation, "WORKING_DIRECTORY_INVALID", "运行目录必须是绝对路径。", "选择已有绝对目录。");
                        }
                        match path.canonicalize() {
                            Ok(path) if path.is_dir() => match path.to_str() {
                                Some(value) => Some(value.to_owned()),
                                None => return Report::error(operation, "WORKING_DIRECTORY_INVALID", "运行目录不是有效 UTF-8。", "选择可表示的目录。"),
                            },
                            _ => return Report::error(operation, "WORKING_DIRECTORY_UNAVAILABLE", "运行目录不可用。", "选择已有目录。"),
                        }
                    }
                    None => None,
                };
                if import_roots.len() > 32 {
                    return Report::error(operation, "IMPORT_ROOTS_INVALID", "模块目录最多 32 项。", "移除多余目录。");
                }
                let mut seen = std::collections::HashSet::new();
                for value in import_roots {
                    let path = Path::new(value);
                    let valid = !value.is_empty() && value.len() <= 1024 && !value.contains('\\')
                        && !value.contains('\0') && value.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
                        && !path.is_absolute() && path.components().all(|part| matches!(part, std::path::Component::Normal(_)))
                        && seen.insert(value);
                    let full = root.join(path);
                    if !valid || !full.canonicalize().is_ok_and(|resolved| resolved == full && resolved.is_dir()) {
                        return Report::error(operation, "IMPORT_ROOTS_INVALID", format!("模块目录无效或不可用：{value}"), "使用不重复、无符号链接的项目内相对目录。");
                    }
                }
                let paths = json!({"working_directory":directory,"project_import_roots":import_roots});
                let mut candidate = current.clone();
                candidate["runtime_paths"] = paths.clone();
                if *apply && if_current.as_deref() != Some(current_revision.as_str()) {
                    return Report::error(operation, "PROJECT_CONFIG_CONFLICT", "项目配置与预览修订不一致。", "重新预览；配置未修改。");
                }
                let mut result = json!({"committed":false,"unchanged":candidate == current,"from_revision":current_revision,
                    "runtime_paths":paths,"previous_runtime_paths":current["runtime_paths"],
                    "note":"只保存路径选择，不执行项目代码或验证导入。"});
                if *apply && candidate != current {
                    match self.project_config.update_versioned(&root, &current_revision, &candidate) {
                        Ok(update) => { result["committed"] = json!(true); result["config_revision"] = json!(update.revision); },
                        Err(ProjectConfigUpdateError::Conflict { .. }) => return Report::error(operation, "PROJECT_CONFIG_CONFLICT", "项目配置在提交时发生变化。", "重新预览。"),
                        Err(error) => return Report::error(operation, "PROJECT_CONFIG_WRITE_FAILED", error.message(), "检查项目配置和目录权限。"),
                    }
                }
                redacted_config_report(operation, result)
            }
            ProjectConfigCommand::SetModel {
                model, model_params, clear_injections, apply, if_current,
            } => {
                let valid_entrypoint = model.split_once(':').is_some_and(|(module, callable)| {
                    !module.trim().is_empty() && !callable.trim().is_empty()
                        && !callable.contains(':') && !model.chars().any(char::is_whitespace)
                });
                if !valid_entrypoint {
                    return Report::error(operation, "MODEL_ENTRYPOINT_INVALID",
                        "模型入口须是无空白的 module:callable。", "例如 my_project.model:build_model。");
                }
                let params = match parameters(model_params) {
                    Ok(params) => params,
                    Err(error) => return Report::error(operation, "PARAMETER_INVALID", error,
                        "参数使用 NAME=JSON 格式；请勿保存凭据。"),
                };
                let mut candidate = current.clone();
                candidate["model"] = json!(model);
                candidate["model_params"] = json!(params);
                if *clear_injections {
                    candidate["model_injections"] = json!({});
                }
                if *apply && if_current.as_deref() != Some(current_revision.as_str()) {
                    return Report::error(operation, "PROJECT_CONFIG_CONFLICT",
                        "项目配置与预览修订不一致。", "重新预览并确认；没有写入配置。");
                }
                let mut result = json!({
                    "committed": false, "unchanged": candidate == current,
                    "from_revision": current_revision, "model": model,
                    "model_params": candidate["model_params"],
                    "model_injections": candidate["model_injections"],
                    "previous_model": current["model"],
                    "previous_model_params": current["model_params"],
                    "previous_model_injections": current["model_injections"],
                    "clear_injections": clear_injections,
                    "note": "仅更新项目配置；未执行模型、验证装配或固定源码和环境。"
                });
                if *apply && candidate != current {
                    match self.project_config.update_versioned(&root, &current_revision, &candidate) {
                        Ok(update) => {
                            result["committed"] = json!(true);
                            result["config_revision"] = json!(update.revision);
                            result["path"] = json!(update.path);
                        }
                        Err(ProjectConfigUpdateError::Conflict { .. }) => return Report::error(
                            operation, "PROJECT_CONFIG_CONFLICT", "项目配置在提交时发生变化。",
                            "重新预览并确认；没有覆盖其他更改。"),
                        Err(error) => return Report::error(operation, "PROJECT_CONFIG_WRITE_FAILED",
                            error.message(), "检查项目配置和目录权限。"),
                    }
                }
                redacted_config_report(operation, result)
            }
            ProjectConfigCommand::SetBatch {
                batch,
                batch_binding,
                batch_params,
                apply,
                if_current,
            } => {
                let Some((module, callable)) = batch.split_once(':') else {
                    return Report::error(
                        operation,
                        "BATCH_ENTRYPOINT_INVALID",
                        "批次入口须是 module:callable。",
                        "例如 my_project.data:batch。 ",
                    );
                };
                if module.trim().is_empty()
                    || callable.trim().is_empty()
                    || callable.contains(':')
                    || batch.chars().any(char::is_whitespace)
                {
                    return Report::error(
                        operation,
                        "BATCH_ENTRYPOINT_INVALID",
                        "批次入口须是无空白的 module:callable。",
                        "例如 my_project.data:batch。 ",
                    );
                }
                let params = match parameters(batch_params) {
                    Ok(params) => params,
                    Err(error) => {
                        return Report::error(
                            operation,
                            "PARAMETER_INVALID",
                            error,
                            "参数使用 NAME=JSON 格式；请勿保存凭据。",
                        );
                    }
                };
                let mut display_params = Value::Object(params.clone());
                let mut redacted_fields = Vec::new();
                redact_run_inputs(&mut display_params, "$.batch_params", &mut redacted_fields);
                let binding = batch_binding.as_ref().map(|value| match value {
                    BatchBinding::Envelope => "envelope",
                    BatchBinding::Single => "single",
                    BatchBinding::SingleRaw => "single-raw",
                    BatchBinding::Args => "args",
                    BatchBinding::Kwargs => "kwargs",
                });
                let mut candidate = current.clone();
                candidate["batch"] = json!(batch);
                candidate["batch_binding"] = json!(binding);
                candidate["batch_params"] = json!(params);
                if *apply && if_current.as_deref() != Some(current_revision.as_str()) {
                    return Report::error(
                        operation,
                        "PROJECT_CONFIG_CONFLICT",
                        "项目配置与预览修订不一致。",
                        "重新预览并确认；没有写入配置。 ",
                    );
                }
                if *apply && candidate != current {
                    match self
                        .project_config
                        .update_versioned(&root, &current_revision, &candidate)
                    {
                        Ok(update) => {
                            return redacted_config_report(
                                operation,
                                json!({"committed":true,"from_revision":current_revision,"config_revision":update.revision,"batch":batch,"batch_binding":binding,"batch_params":display_params,"redacted_fields":redacted_fields,"path":update.path,"note":"仅更新批次入口、绑定与参数；未固定数据文件或源码。"}),
                            );
                        }
                        Err(ProjectConfigUpdateError::Conflict { .. }) => {
                            return Report::error(
                                operation,
                                "PROJECT_CONFIG_CONFLICT",
                                "项目配置在提交时发生变化。",
                                "重新预览并确认；没有覆盖其他更改。",
                            );
                        }
                        Err(error) => {
                            return Report::error(
                                operation,
                                "PROJECT_CONFIG_WRITE_FAILED",
                                error.message(),
                                "检查项目配置和目录权限。",
                            );
                        }
                    }
                }
                redacted_config_report(
                    operation,
                    json!({"committed":false,"unchanged":candidate == current,"from_revision":current_revision,"batch":batch,"batch_binding":binding,"batch_params":display_params,"redacted_fields":redacted_fields,"previous_batch":current["batch"],"previous_batch_binding":current["batch_binding"],"previous_batch_params":current["batch_params"],"note":"仅预览配置；未执行批次或前向，源码和数据文件未固定。"}),
                )
            }
            ProjectConfigCommand::Restore {
                revision,
                fields,
                apply,
                if_current,
            } => {
                if fields.is_empty() {
                    return Report::error(
                        operation,
                        "CONFIG_RESTORE_FIELDS_REQUIRED",
                        "至少选择一个配置字段。",
                        "使用受支持的 --field，包括 runtime_paths 和 runtime_python。",
                    );
                }
                if fields.iter().any(|field| {
                    !matches!(
                        field.as_str(),
                        "model"
                            | "model_params"
                            | "batch"
                            | "batch_binding"
                            | "batch_params"
                            | "model_injections"
                            | "runtime_paths"
                            | "runtime_python"
                    )
                }) {
                    return Report::error(
                        operation,
                        "CONFIG_RESTORE_FIELD_UNSUPPORTED",
                        "只能恢复受支持的模型、批次、装配或运行时字段。",
                        "从受支持字段中重新选择；schema_version 等协议字段不会被历史恢复覆盖。",
                    );
                }
                let mut unique_fields = std::collections::HashSet::new();
                if fields.iter().any(|field| !unique_fields.insert(field)) {
                    return Report::error(
                        operation,
                        "CONFIG_RESTORE_FIELD_DUPLICATE",
                        "同一个字段不能重复选择。",
                        "每个字段只指定一次 --field。",
                    );
                }
                let revision = revision.trim();
                if revision.is_empty() {
                    return Report::error(
                        operation,
                        "CONFIG_REVISION_REQUIRED",
                        "历史 revision 不能为空。",
                        "运行 m-model config history 并选择一个版本。",
                    );
                }
                let matches = validated_history
                    .iter()
                    .filter(|entry| {
                        entry["revision"]
                            .as_str()
                            .is_some_and(|saved| saved.starts_with(revision))
                    })
                    .collect::<Vec<_>>();
                let snapshot = match matches.as_slice() {
                    [snapshot] => *snapshot,
                    [] => {
                        return Report::error(
                            operation,
                            "CONFIG_REVISION_NOT_FOUND",
                            format!("没有匹配的历史 revision：{revision}。"),
                            "运行 m-model config history 查看可用版本。",
                        );
                    }
                    _ => {
                        return Report::error(
                            operation,
                            "CONFIG_REVISION_AMBIGUOUS",
                            format!("revision 前缀 {revision} 匹配多个历史版本。"),
                            "从 m-model config history 复制更长的 revision 前缀。",
                        );
                    }
                };
                let source_config = &snapshot["config"];
                let mut restored = current.clone();
                let mut changes = Vec::new();
                for field in fields {
                    let previous = current.get(field).cloned();
                    let target = source_config.get(field).cloned();
                    if let Some(value) = target.clone() {
                        restored[field] = value;
                    } else if let Some(object) = restored.as_object_mut() {
                        object.remove(field);
                    }
                    changes.push(json!({
                        "field":field,
                        "current":previous,
                        "restored":target,
                        "changed":previous != target
                    }));
                }
                if *apply {
                    let Some(expected_revision) = if_current.as_deref().map(str::trim) else {
                        return Report::error(
                            operation,
                            "CONFIG_EXPECTED_REVISION_REQUIRED",
                            "提交恢复必须提供预览时看到的当前 revision。",
                            "先预览恢复，再使用 --apply --if-current <from_revision> 提交。",
                        );
                    };
                    if expected_revision != current_revision {
                        return Report::error(
                            operation,
                            "PROJECT_CONFIG_CONFLICT",
                            format!(
                                "配置在预览后已变化；预览 revision={expected_revision}，当前 revision={current_revision}。"
                            ),
                            "保留当前配置；重新预览恢复，检查差异后再确认。",
                        );
                    }
                }
                if *apply && changes.iter().all(|change| change["changed"] == false) {
                    return redacted_config_report(
                        operation,
                        json!({
                            "committed":false,
                            "unchanged":true,
                            "from_revision":current_revision,
                            "restored_from_revision":snapshot["revision"],
                            "fields":changes,
                            "note":"所选字段与当前配置相同，无需写入。"
                        }),
                    );
                }
                let committed = if *apply {
                    match self
                        .project_config
                        .update_versioned(&root, &current_revision, &restored)
                    {
                        Ok(update) => {
                            return redacted_config_report(
                                operation,
                                json!({
                                    "committed":true,
                                    "path":update.path,
                                    "from_revision":current_revision,
                                    "restored_from_revision":snapshot["revision"],
                                    "config_revision":update.revision,
                                    "fields":changes,
                                    "note":"已恢复所选字段；恢复前版本已保存在配置历史中。"
                                }),
                            );
                        }
                        Err(ProjectConfigUpdateError::Conflict {
                            expected_revision,
                            current_revision,
                        }) => {
                            return Report::error(
                                operation,
                                "PROJECT_CONFIG_CONFLICT",
                                format!(
                                    "配置在恢复前已变化；预期 {expected_revision}，当前 {current_revision}。"
                                ),
                                "重新运行 m-model config history 并检查最新配置，再预览恢复。",
                            );
                        }
                        Err(error) => {
                            return Report::error(
                                operation,
                                "PROJECT_CONFIG_WRITE_FAILED",
                                error.message(),
                                "检查项目配置目录权限；提交前的当前配置保持不变。",
                            );
                        }
                    }
                } else {
                    false
                };
                redacted_config_report(
                    operation,
                    json!({
                        "committed":committed,
                        "path":root.join(".model-maven/project.json"),
                        "from_revision":current_revision,
                        "restored_from_revision":snapshot["revision"],
                        "config_revision":crate::domain::input::content_revision(&restored).ok(),
                        "fields":changes,
                        "note":"这是预览，没有写入项目配置。确认字段后，使用 --apply --if-current <from_revision> 提交；revision 若已变化，恢复会被拒绝并要求重新预览。"
                    }),
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn assembly_candidate(
        &self,
        cli: &Cli,
        root: &Path,
        config: &Value,
        injections: &Map<String, Value>,
        inputs: &AssemblyInputs,
        operation: &str,
        variant: Option<&Value>,
        expected_input: Option<&str>,
    ) -> Report {
        let override_model_params = match parameters(&inputs.model_params) {
            Ok(params) => params,
            Err(error) => {
                return Report::error(
                    operation,
                    "PARAMETER_INVALID",
                    error,
                    "使用 --model-param NAME=JSON。",
                );
            }
        };
        let mut model_params = config["model_params"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        model_params.extend(override_model_params);
        let batch_params = match parameters(&inputs.batch_params) {
            Ok(params) => params,
            Err(error) => {
                return Report::error(
                    operation,
                    "PARAMETER_INVALID",
                    error,
                    "使用 --batch-param NAME=JSON。",
                );
            }
        };
        let batch_params = if inputs.batch.is_none() {
            let mut merged = config["batch_params"]
                .as_object()
                .cloned()
                .unwrap_or_default();
            merged.extend(batch_params);
            merged
        } else {
            batch_params
        };
        let revision = crate::domain::input::content_revision(config).ok();
        let batch_binding = |value: &str, kind, argument: &str| crate::domain::input::Candidate {
            value: Value::String(value.to_owned()),
            source: crate::domain::operation::InputSource {
                kind,
                reference: Some(
                    if kind == crate::domain::operation::InputSourceKind::CommandLine {
                        argument.into()
                    } else {
                        ".model-maven/project.json#/batch".into()
                    },
                ),
                revision: if kind == crate::domain::operation::InputSourceKind::ProjectConfig {
                    revision.clone()
                } else {
                    None
                },
            },
        };
        let resolved_batch = crate::domain::input::resolve(
            "batch",
            false,
            false,
            inputs.batch.as_deref().map(|value| {
                batch_binding(
                    value,
                    crate::domain::operation::InputSourceKind::CommandLine,
                    "--batch",
                )
            }),
            None,
            None,
            config["batch"].as_str().map(|value| {
                batch_binding(
                    value,
                    crate::domain::operation::InputSourceKind::ProjectConfig,
                    "--batch",
                )
            }),
            None,
        );
        let batch = resolved_batch.value.as_ref().and_then(Value::as_str);
        if batch.is_none() && !inputs.batch_params.is_empty() {
            return Report::error(
                operation,
                "BATCH_PARAMS_WITHOUT_BATCH",
                "--batch-param 需要 --batch 或项目默认批次入口。",
                "提供批次工厂，或移除 --batch-param。",
            );
        }
        if inputs.batch_binding.is_some() && batch.is_none() {
            return Report::error(
                operation,
                "BATCH_BINDING_WITHOUT_BATCH",
                "--batch-binding 需要同时设置 --batch 或项目默认批次。",
                "为批次工厂提供 module:factory，或省略 --batch-binding。",
            );
        }
        let explicit_binding = inputs.batch_binding.as_ref().map(|value| match value {
            BatchBinding::Envelope => "envelope",
            BatchBinding::Single => "single",
            BatchBinding::SingleRaw => "single-raw",
            BatchBinding::Args => "args",
            BatchBinding::Kwargs => "kwargs",
        });
        let resolved_binding = crate::domain::input::resolve(
            "batch_binding",
            false,
            false,
            explicit_binding.map(|value| {
                batch_binding(
                    value,
                    crate::domain::operation::InputSourceKind::CommandLine,
                    "--batch-binding",
                )
            }),
            None,
            None,
            (inputs.batch.is_none()).then(|| config["batch_binding"].as_str()).flatten().map(|value| {
                batch_binding(
                    value,
                    crate::domain::operation::InputSourceKind::ProjectConfig,
                    "--batch-binding",
                )
            }),
            None,
        );
        let binding = resolved_binding.value.as_ref().and_then(Value::as_str);
        let request_operation = if batch.is_some() { "probe" } else { "inspect" };
        if !matches!(operation, "assembly_patch" | "assembly_generate" | "assembly_show" | "assembly_encoder_output" | "assembly_route" | "assembly_add" | "assembly_set" | "assembly_remove")
            && (std::env::var_os("MODEL_MAVEN_WORKSPACE_WORKING_DIRECTORY").is_some()
                || std::env::var_os("MODEL_MAVEN_WORKSPACE_IMPORT_ROOTS").is_some()
                || !config["runtime_paths"]["working_directory"].is_null()
                || config["runtime_paths"]["project_import_roots"].as_array().is_some_and(|roots| !roots.is_empty()))
        {
            return Report::error(operation, "ASSEMBLY_RUNTIME_PATHS_UNSUPPORTED",
                "此装配操作尚不支持自定义运行时路径。", "先使用默认路径，或使用支持条件验证的批量装配。");
        }
        let (model, model_params, model_injections, seed) = if let Some(variant) = variant {
            (Value::Null, json!({}), json!({}), variant["seed"].clone())
        } else {
            (
                config["model"].clone(),
                Value::Object(model_params),
                Value::Object(injections.clone()),
                json!(cli.seed),
            )
        };
        let selected_python = cli.python.as_deref().filter(|value| !value.trim().is_empty())
            .or_else(|| config["runtime_python"].as_str().filter(|value| !value.trim().is_empty()));
        let Some(selected_python) = selected_python else {
            return Report::error(operation, "PYTHON_SELECTION_REQUIRED", "未发现或指定 Python 解释器。", "使用 --python 明确选择解释器后重试；装配尚未执行。");
        };
        let mut runtime = json!({
            "python_executable":selected_python,
            "timeout_seconds":cli.timeout.unwrap_or(30),
            "project_root":root
        });
        super::run::apply_project_runtime_defaults(config, &mut runtime);
        if let Err(report) = super::run::resolve_worker_directory(request_operation, root, &mut runtime, false) {
            return report;
        }
        if let Err(report) = super::run::resolve_worker_import_roots(request_operation, root, &mut runtime, false) {
            return report;
        }
        let runtime_paths = json!({"working_directory":runtime["working_directory"], "project_import_roots":runtime["project_import_roots"]});
        let mut report = self.backend.execute(&json!({
            "protocol_version":PROTOCOL_VERSION,
            "operation":request_operation,
            "required_features": if variant.is_none() && cli.seed.is_none() { vec!["model.runtime_rng_v1"] } else { Vec::<&str>::new() },
            "project_root":root,
            "model":model,
            "model_params":model_params,
            "model_injections":model_injections,
            "batch":batch,
            "batch_params":batch_params,
            "batch_binding":binding,
            "training":false,
            "seed":seed,
            "variant":variant,
            "edit":null,
            "require_forward":false,
            "collect_source_manifest":matches!(operation, "assembly_patch" | "assembly_encoder_output" | "assembly_route" | "assembly_add" | "assembly_set" | "assembly_remove" | "assembly_generate"),
            "expected_input_content":expected_input,
            "capture_input_to":if matches!(operation, "assembly_patch" | "assembly_encoder_output" | "assembly_route" | "assembly_add" | "assembly_set" | "assembly_remove") { cli.capture_input_to.as_ref().and_then(|path| path.to_str()) } else { None },
            "replay_input_from":if matches!(operation, "assembly_patch" | "assembly_encoder_output" | "assembly_route" | "assembly_add" | "assembly_set" | "assembly_remove") { cli.replay_input_from.as_ref().and_then(|path| path.to_str()) } else { None },
            "runtime":runtime
        }));
        if let Some(data) = report.data.as_object_mut() {
            data.insert("assembly_runtime_paths".into(), runtime_paths);
        }
        report
    }

    fn assembly_generate(&self, cli: &Cli, output: &Path, inputs: &AssemblyInputs) -> Report {
        if !valid_assembly_output_path(output) {
            return Report::error(
                "assembly_generate",
                "ASSEMBLY_OUTPUT_PATH_INVALID",
                "输出路径必须是项目目录内的相对文件路径。",
                "使用 generated/model_assembly.py 或其他不含 .. 的相对路径。",
            );
        }
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error(
                    "assembly_generate",
                    "PROJECT_UNAVAILABLE",
                    error.to_string(),
                    "在可访问的 Model-Maven 项目目录中重试。",
                );
            }
        };
        let config = match self.project_config.load(&root) {
            Ok(Some(config)) => config,
            Ok(None) => {
                return Report::error(
                    "assembly_generate",
                    "PROJECT_NOT_INITIALIZED",
                    "当前项目尚未保存模型入口。",
                    "先运行 m-model init 保存模型工厂和需要的接缝。",
                );
            }
            Err(error) => {
                return Report::error(
                    "assembly_generate",
                    "PROJECT_CONFIG_INVALID",
                    error,
                    "先修复 .model-maven/project.json；生成器不会修改项目配置。",
                );
            }
        };
        let revision = match crate::domain::input::content_revision(&config) {
            Ok(revision) => revision,
            Err(error) => {
                return Report::error(
                    "assembly_generate",
                    "PROJECT_CONFIG_REVISION_FAILED",
                    error,
                    "无法确认项目配置修订；没有生成文件。",
                );
            }
        };
        if inputs
            .if_config_revision
            .as_deref()
            .is_some_and(|expected| expected != revision)
        {
            return Report::error(
                "assembly_generate",
                "PROJECT_CONFIG_CONFLICT",
                "项目配置与确认时的修订不一致。",
                "重新查看装配并确认生成；没有写入文件。 ",
            );
        }
        let selected_variant = match cli.variant.as_deref() {
            Some(name) => match self.variants.load(&root, name) {
                Ok(Some(variant)) => Some((name.to_owned(), variant)),
                Ok(None) => {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_NOT_FOUND",
                        format!("找不到模型变体：{name}"),
                        "运行 m-model variant list 查看可生成的变体。",
                    );
                }
                Err(error) => {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_READ_FAILED",
                        error,
                        "检查变体 JSON 和 .model-maven/variants 目录权限。",
                    );
                }
            },
            None => None,
        };
        if selected_variant.is_some() && !inputs.model_params.is_empty() {
            return Report::error(
                "assembly_generate",
                "VARIANT_MODEL_PARAMS_FIXED",
                "模型变体固定了根工厂参数；不能在生成时覆盖。",
                "移除 --model-param，或复制变体并在新变体中修改模型来源。",
            );
        }
        let mut validation = if let Some((_, variant)) = selected_variant.as_ref() {
            self.assembly_candidate(
                cli,
                &root,
                &config,
                &Map::new(),
                inputs,
                "assembly_generate",
                Some(variant),
                None,
            )
        } else {
            self.assembly_show(cli, inputs, true)
        };
        if !validation.success() {
            validation.operation = "assembly_generate".into();
            return validation;
        }
        let initial_input = if selected_variant.is_some() {
            &validation.data["batch_content"]
        } else {
            &validation.data["input_evidence"]
        };

        let (model_factory, model_params, injections, seed, variant_recipe) =
            if let Some((_, variant)) = selected_variant.as_ref() {
                let Some(factory) = variant["origin"]["model"].as_str() else {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_ORIGIN_INVALID",
                        "变体没有可重建的根模型工厂入口。",
                        "检查变体来源后重试；未创建生成文件。",
                    );
                };
                let Some(params) = variant["origin"]["params"].as_object().cloned() else {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_ORIGIN_INVALID",
                        "变体根模型参数必须是对象。",
                        "检查变体来源后重试；未创建生成文件。",
                    );
                };
                let Some(saved_injections) = variant["origin"]["model_injections"].as_object()
                else {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_ORIGIN_INVALID",
                        "变体接缝必须是对象。",
                        "检查变体来源后重试；未创建生成文件。",
                    );
                };
                let mut injections = Map::new();
                for (name, binding) in saved_injections {
                    let (factory, params, component_ref) = if let Some(factory) = binding.as_str() {
                        (factory, Map::new(), None)
                    } else if let Some(specification) = binding.as_object() {
                        let Some(factory) = specification.get("factory").and_then(Value::as_str)
                        else {
                            return Report::error(
                                "assembly_generate",
                                "VARIANT_ORIGIN_INVALID",
                                format!("变体接缝 {name} 缺少工厂入口。"),
                                "检查变体来源后重试；未创建生成文件。",
                            );
                        };
                        (
                            factory,
                            specification["params"]
                                .as_object()
                                .cloned()
                                .unwrap_or_default(),
                            specification.get("component_ref").filter(|v| v.is_object()),
                        )
                    } else {
                        return Report::error(
                            "assembly_generate",
                            "VARIANT_ORIGIN_INVALID",
                            format!("变体接缝 {name} 格式无效。"),
                            "检查变体来源后重试；未创建生成文件。",
                        );
                    };
                    let mut specification = json!({"factory":factory,"params":params});
                    if let Some(reference) = component_ref {
                        specification["component_ref"] = reference.clone();
                    }
                    injections.insert(name.clone(), specification);
                }
                if variant.get("device_policy").is_some_and(|policy| !matches!(policy.as_str(), Some("legacy_cpu" | "factory"))) {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_DEVICE_POLICY_UNSUPPORTED",
                        "变体的设备重建策略无效或未知。",
                        "请保留原变体和源码并检查版本兼容。",
                    );
                }
                let Some(seed) = variant["seed"].as_u64() else {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_INVALID",
                        "变体缺少有效的初始化种子。",
                        "检查变体来源后重试；未创建生成文件。",
                    );
                };
                let Some(edits) = variant["edits"].as_array() else {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_INVALID",
                        "变体缺少 Layer 编辑历史。",
                        "检查变体来源后重试；未创建生成文件。",
                    );
                };
                let Some(cursor) = variant["cursor"].as_u64().map(|value| value as usize) else {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_INVALID",
                        "变体编辑位置无效。",
                        "检查变体来源后重试；未创建生成文件。",
                    );
                };
                if cursor > edits.len() {
                    return Report::error(
                        "assembly_generate",
                        "VARIANT_INVALID",
                        "变体编辑位置超过历史长度。",
                        "检查变体来源后重试；未创建生成文件。",
                    );
                }
                let active_edits = edits[..cursor].to_vec();
                let mut active_variant = json!({"seed":seed,"edits":active_edits});
                if let Some(policy) = variant.get("device_policy") {
                    active_variant["device_policy"] = policy.clone();
                }
                if let Some(devices) = variant.get("base_rng_devices") {
                    active_variant["base_rng_devices"] = devices.clone();
                }
                (
                    factory.to_owned(),
                    params,
                    injections,
                    Some(seed),
                    Some(active_variant),
                )
            } else {
                let Some(factory) = config["model"].as_str() else {
                    return Report::error(
                        "assembly_generate",
                        "ASSEMBLY_FACTORY_MISSING",
                        "项目配置没有可生成的根模型工厂。",
                        "使用 m-model init --model module:factory 配置命名 Python 工厂。",
                    );
                };
                let overrides = match parameters(&inputs.model_params) {
                    Ok(parameters) => parameters,
                    Err(error) => {
                        return Report::error(
                            "assembly_generate",
                            "PARAMETER_INVALID",
                            error,
                            "使用 --model-param NAME=JSON 重试。",
                        );
                    }
                };
                let mut params = config["model_params"]
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
                params.extend(overrides);
                let mut injections = Map::new();
                let Some(slots) = validation.data["slots"].as_array() else {
                    return Report::error(
                        "assembly_generate",
                        "ASSEMBLY_EVIDENCE_INVALID",
                        "装配验证没有返回可生成的接缝证据。",
                        "重新运行 m-model assembly show；当前配置未更改。",
                    );
                };
                for slot in slots {
                    let Some(name) = slot["slot"].as_str() else {
                        continue;
                    };
                    let Some(factory) = slot["factory"].as_str() else {
                        return Report::error(
                            "assembly_generate",
                            "ASSEMBLY_FACTORY_MISSING",
                            format!("接缝 {name} 没有可重建的 Python 工厂入口。"),
                            "补全接缝工厂后重新验证和生成。",
                        );
                    };
                    let params = slot["params"].as_object().cloned().unwrap_or_default();
                    let mut specification = json!({"factory":factory,"params":params});
                    if let Some(component_ref) =
                        slot.get("component_ref").filter(|value| value.is_object())
                    {
                        specification["component_ref"] = component_ref.clone();
                    }
                    injections.insert(name.to_owned(), specification);
                }
                (
                    factory.to_owned(),
                    params,
                    injections,
                    cli.seed,
                    None,
                )
            };

        let recipe = if let Some(variant) = variant_recipe {
            json!({
                "schema_version":2,
                "description":"Generated from the verified Model-Maven assembly and selected Layer variant.",
                "model":{"factory":model_factory,"params":model_params},
                "model_injections":injections,
                "variant":variant
            })
        } else {
            json!({
                "schema_version":1,
                "description":"Generated from the verified Model-Maven project assembly.",
                "model":{"factory":model_factory,"params":model_params},
                "model_injections":injections,
                "initialization_seed":seed
            })
        };
        let selected_python = cli.python.as_deref().filter(|value| !value.trim().is_empty())
            .or_else(|| config["runtime_python"].as_str().filter(|value| !value.trim().is_empty()));
        let Some(selected_python) = selected_python else {
            return Report::error("assembly_generate", "PYTHON_SELECTION_REQUIRED", "未发现或指定 Python 解释器。", "使用 --python 明确选择解释器后重试；没有生成文件。");
        };
        let mut verification_runtime = json!({"python_executable":selected_python, "timeout_seconds":cli.timeout.unwrap_or(30), "project_root":root});
        super::run::apply_project_runtime_defaults(&config, &mut verification_runtime);
        if let Err(report) = super::run::resolve_worker_directory("probe", &root, &mut verification_runtime, false) { return report; }
        if let Err(report) = super::run::resolve_worker_import_roots("probe", &root, &mut verification_runtime, false) { return report; }
        let verification_paths = json!({"working_directory":verification_runtime["working_directory"], "project_import_roots":verification_runtime["project_import_roots"]});
        if validation.data["assembly_runtime_paths"] != verification_paths {
            return Report::error("assembly_generate", "ASSEMBLY_RUNTIME_CONTEXT_CHANGED",
                "装配验证后的运行目录或模块搜索目录发生变化。", "重新查看装配并生成；没有创建输出文件。");
        }
        let mut codegen_features = vec!["edit.assembly_codegen"];
        if selected_variant.is_none() && seed.is_some() {
            codegen_features.push("edit.assembly_python_rng_v1");
        }
        if recipe["variant"]["device_policy"].as_str() == Some("factory") {
            codegen_features.push("model.variant_factory_device_v1");
        }
        if recipe["variant"].get("base_rng_devices").is_some() {
            codegen_features.push("model.variant_base_rng_v1");
        }
        let request = json!({
            "protocol_version":PROTOCOL_VERSION,
            "operation":"assembly_codegen",
            "project_root":root,
            "required_features":codegen_features,
            "assembly_recipe":recipe,
            "interactive_progress":false,
            "runtime":{
                "python_executable":selected_python,
                "timeout_seconds":cli.timeout.unwrap_or(30),
                "project_root":root
            }
        });
        let mut generated = self.backend.execute(&request);
        if !generated.success() {
            generated.operation = "assembly_generate".into();
            return generated;
        }
        let Some(source) = generated.data["source"].as_str() else {
            return Report::error(
                "assembly_generate",
                "ASSEMBLY_CODEGEN_INVALID",
                "代码生成 worker 没有返回 Python 源码。",
                "检查安装的 worker 与 CLI 协议版本。",
            );
        };
        let batch = inputs.batch.as_deref().or_else(|| config["batch"].as_str());
        let binding = inputs
            .batch_binding
            .as_ref()
            .map(|value| match value {
                BatchBinding::Envelope => "envelope",
                BatchBinding::Single => "single",
                BatchBinding::SingleRaw => "single-raw",
                BatchBinding::Args => "args",
                BatchBinding::Kwargs => "kwargs",
            })
            .or_else(|| if inputs.batch.is_none() { config["batch_binding"].as_str() } else { None });
        let mut batch_params = if inputs.batch.is_none() {
            config["batch_params"]
                .as_object()
                .cloned()
                .unwrap_or_default()
        } else {
            Map::new()
        };
        match parameters(&inputs.batch_params) {
            Ok(params) => batch_params.extend(params),
            Err(error) => {
                return Report::error(
                    "assembly_generate",
                    "PARAMETER_INVALID",
                    error,
                    "检查批次参数；没有创建输出文件。",
                );
            }
        }
        let verify_request = json!({
            "protocol_version":PROTOCOL_VERSION,
            "operation":"assembly_codegen_verify",
            "required_features":["edit.assembly_codegen_runtime_paths_v1"],
            "project_root":root,
            "generated_source":source,
            "model":model_factory,
            "batch":batch,
            "batch_binding":binding,
            "batch_params":batch_params,
            "training":false,
            "runtime":verification_runtime
        });
        let mut generated_validation = self.backend.execute(&verify_request);
        if !generated_validation.success() {
            generated_validation.operation = "assembly_generate".into();
            if let Some(diagnostic) = generated_validation.diagnostics.first_mut() {
                diagnostic.hint = Some("生成代码在独立进程中重建或前向失败；没有创建输出文件。检查工厂、批次与运行环境。".into());
            }
            return generated_validation;
        }
        if generated_validation.data["construction"] != "passed"
            || generated_validation.data["forward"]
                != if batch.is_some() { "passed" } else { "not_run" }
        {
            return Report::error(
                "assembly_generate",
                "ASSEMBLY_GENERATED_VERIFICATION_INVALID",
                "生成代码验证回执不完整。",
                "没有创建输出文件；检查 worker 版本。",
            );
        }
        let input_digest = |evidence: &Value| -> Option<String> {
            if evidence["status"] != "available"
                || evidence["algorithm"] != "sha256-forward-input-v1"
            {
                return None;
            }
            evidence["digest"]
                .as_str()
                .filter(|value| {
                    value.len() == 64
                        && value
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
                .map(str::to_owned)
        };
        let input_comparison = if batch.is_none() {
            "not_run"
        } else {
            match (
                input_digest(initial_input),
                input_digest(&generated_validation.data["input_evidence"]),
            ) {
                (Some(first), Some(second)) if first == second => "same",
                (Some(_), Some(_)) => "different",
                _ => "unknown",
            }
        };
        let complete_aliases = |value: &Value| -> Option<Value> {
            let groups = value["groups"].as_array()?;
            if value["status"] != "observed"
                || value["scope"] != "registered_module_identity_after_construction"
                || value["truncated"] != false
                || value["group_count"].as_u64()? != groups.len() as u64
                || groups.iter().any(|group| {
                    group.as_array().is_none_or(|paths| paths.len() < 2 || paths.iter().any(|path| !path.is_string()))
                })
            {
                return None;
            }
            Some(value["groups"].clone())
        };
        let alias_comparison = match (
            complete_aliases(&validation.data["module_aliases"]),
            complete_aliases(&generated_validation.data["module_aliases"]),
        ) {
            (Some(first), Some(second)) if first == second => "same",
            (Some(_), Some(_)) => "different",
            _ => "unknown",
        };
        let complete_placement = |value: &Value| -> Option<(u64, Value)> {
            if value["status"] != "observed" { return None; }
            Some((value["observed_tensors"].as_u64()?, Value::Array(value["groups"].as_array()?.clone())))
        };
        let placement_comparison = match (
            complete_placement(&validation.data["model_state_placement"]),
            complete_placement(&generated_validation.data["model_state_placement"]),
        ) {
            (Some(first), Some(second)) if first == second => "same",
            (Some(_), Some(_)) => "different",
            _ => "unknown",
        };
        let complete_shared_state = |value: &Value| -> Option<Value> {
            if value["status"] != "observed"
                || value["scope"] != "registered_parameter_buffer_objects_and_inspectable_storage" { return None; }
            Some(Value::Array(value["groups"].as_array()?.clone()))
        };
        let shared_state_comparison = match (
            complete_shared_state(&validation.data["shared_state"]),
            complete_shared_state(&generated_validation.data["shared_state"]),
        ) {
            (Some(first), Some(second)) if first == second => "same",
            (Some(_), Some(_)) => "different",
            _ => "unknown",
        };
        if alias_comparison == "different" || shared_state_comparison == "different" {
            let diff_groups = |original: &Value, generated: &Value| -> Value {
                let (Some(first), Some(second)) = (original["groups"].as_array(), generated["groups"].as_array()) else {
                    return json!({"status":"unknown"});
                };
                let encode = |value: &Value| serde_json::to_string(value).ok();
                let first_set: std::collections::BTreeSet<String> = first.iter().filter_map(encode).collect();
                let second_set: std::collections::BTreeSet<String> = second.iter().filter_map(encode).collect();
                let original_only_count = first_set.difference(&second_set).count();
                let generated_only_count = second_set.difference(&first_set).count();
                let original_only: Vec<Value> = first_set.difference(&second_set).take(8)
                    .filter_map(|raw| serde_json::from_str(raw).ok()).collect();
                let generated_only: Vec<Value> = second_set.difference(&first_set).take(8)
                    .filter_map(|raw| serde_json::from_str(raw).ok()).collect();
                json!({"status":"compared", "original_only_count":original_only_count,
                    "generated_only_count":generated_only_count,
                    "original_only":original_only, "generated_only":generated_only})
            };
            let mut report = Report::error(
                "assembly_generate", "ASSEMBLY_SHARED_STRUCTURE_MISMATCH",
                "生成代码重建出的已注册共享关系与原候选不同。",
                "未创建文件；检查工厂的动态行为或变体编辑，重新验证后生成。完整对照见 JSON 报告。",
            );
            report.data = json!({
                "module_alias_comparison":{"status":alias_comparison, "original":validation.data["module_aliases"], "generated":generated_validation.data["module_aliases"]},
                "shared_state_comparison":{"status":shared_state_comparison, "original":validation.data["shared_state"], "generated":generated_validation.data["shared_state"]},
                "module_alias_difference":if alias_comparison == "different" { diff_groups(&validation.data["module_aliases"], &generated_validation.data["module_aliases"]) } else { json!({"status":alias_comparison}) },
                "shared_state_difference":if shared_state_comparison == "different" { diff_groups(&validation.data["shared_state"], &generated_validation.data["shared_state"]) } else { json!({"status":shared_state_comparison}) },
                "input_comparison":{"status":input_comparison},
                "scope":"construction-time registered module and inspectable state sharing only"
            });
            return report;
        }
        let valid_manifest = |value: &Value| {
            value.as_object().is_some_and(|entries| {
                entries.len() <= 4096
                    && entries.iter().all(|(path, hash)| {
                        let path = Path::new(path);
                        path.is_absolute()
                            && path.starts_with(&root)
                            && hash.as_str().is_some_and(|value| {
                                value.len() == 64
                                    && value
                                        .bytes()
                                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                            })
                    })
            })
        };
        let first_sources = &validation.data["source_manifest"];
        let generated_sources = &generated_validation.data["source_manifest"];
        if !valid_manifest(first_sources) || !valid_manifest(generated_sources) {
            return Report::error(
                "assembly_generate",
                "SOURCE_EVIDENCE_UNAVAILABLE",
                "无法核对生成验证使用的项目源码清单。",
                "没有创建输出文件；检查 worker 版本后重新生成。",
            );
        }
        let first = first_sources.as_object().expect("validated manifest");
        let second = generated_sources.as_object().expect("validated manifest");
        use sha2::{Digest, Sha256};
        if !first
            .iter()
            .all(|(path, hash)| second.get(path) == Some(hash))
            || !second.iter().all(|(path, hash)| {
                std::fs::read(path).ok().is_some_and(|bytes| {
                    hash.as_str() == Some(format!("{:x}", Sha256::digest(bytes)).as_str())
                })
            })
        {
            return Report::error(
                "assembly_generate",
                "SOURCE_MANIFEST_CONFLICT",
                "生成验证期间观察到的项目源码发生变化或无法读取。",
                "没有创建输出文件；重新检查源码并确认生成。",
            );
        }
        // User factories and batch providers run during both validation passes and
        // may themselves change project state. Do not publish the old recipe if
        // that state has changed since it was read at the start of this command.
        let current_config = match self.project_config.load(&root) {
            Ok(Some(current)) => current,
            _ => {
                return Report::error(
                    "assembly_generate",
                    "PROJECT_CONFIG_CONFLICT",
                    "生成验证期间项目配置不可读取或已删除。",
                    "没有创建输出文件；检查项目配置并重新确认生成。",
                );
            }
        };
        if crate::domain::input::content_revision(&current_config)
            .ok()
            .as_deref()
            != Some(revision.as_str())
        {
            return Report::error(
                "assembly_generate",
                "PROJECT_CONFIG_CONFLICT",
                "生成验证期间项目配置发生变化。",
                "没有创建输出文件；重新查看装配并确认生成。",
            );
        }
        if let Some((name, variant)) = selected_variant.as_ref() {
            if !matches!(self.variants.load(&root, name), Ok(Some(current)) if current == *variant)
            {
                return Report::error(
                    "assembly_generate",
                    "VARIANT_CONFLICT",
                    "生成验证期间选定的变体发生变化或无法读取。",
                    "没有创建输出文件；重新查看变体并确认生成。",
                );
            }
        }
        let validated_forward = validation.data["validation"]["forward"].as_str().unwrap_or(
            if validation.operation == "probe" {
                "passed"
            } else {
                "not_run"
            },
        );
        let config_revision = crate::domain::input::content_revision(&config).ok();
        let selected_variant_name = selected_variant.as_ref().map(|(name, _)| name);
        let selected_variant_revision = selected_variant
            .as_ref()
            .map(|(_, variant)| variant["revision"].clone());
        let mut final_runtime = json!({"python_executable":selected_python, "timeout_seconds":cli.timeout.unwrap_or(30), "project_root":root});
        super::run::apply_project_runtime_defaults(&current_config, &mut final_runtime);
        if let Err(report) = super::run::resolve_worker_directory("probe", &root, &mut final_runtime, false) { return report; }
        if let Err(report) = super::run::resolve_worker_import_roots("probe", &root, &mut final_runtime, false) { return report; }
        if json!({"working_directory":final_runtime["working_directory"], "project_import_roots":final_runtime["project_import_roots"]}) != verification_paths {
            return Report::error("assembly_generate", "ASSEMBLY_RUNTIME_CONTEXT_CHANGED",
                "写入前运行目录或模块搜索目录发生变化。", "重新查看装配并生成；没有创建输出文件。");
        }
        let written = match self
            .project_source_writer
            .write_new(&root, output, source.as_bytes())
        {
            Ok(path) => path,
            Err((code, message)) => {
                return Report::error(
                    "assembly_generate",
                    code,
                    message,
                    "输出只写入项目目录内的新文件；选择另一个 generated/ 路径，不会覆盖现有源码。",
                );
            }
        };
        Report::ok(
            "assembly_generate",
            json!({
                "output":output,
                "path":written,
                "bytes":source.len(),
                "generated_source_sha256":format!("{:x}", Sha256::digest(source.as_bytes())),
                "config_revision":config_revision,
                "model":model_factory,
                "injection_slots":injections.keys().collect::<Vec<_>>(),
                "initialization_seed":seed,
                "initialization_seed_decimal":seed.map(|value| value.to_string()),
                "variant":selected_variant_name,
                "variant_revision":selected_variant_revision,
                "validated":{"construction":"passed","forward":validated_forward},
                "generated_validation":generated_validation.data,
                "source_evidence":{"status":if first.is_empty() { "unavailable" } else { "checked" }, "scope":"project-sources-observed-in-validation", "first_count":first.len(), "generated_count":second.len()},
                "input_comparison":{"status":input_comparison, "scope":"observed-forward-inputs", "algorithm":"sha256-forward-input-v1"},
                "module_alias_comparison":{"status":alias_comparison, "scope":"registered_module_identity_after_construction", "original":validation.data["module_aliases"]},
                "placement_comparison":{"status":placement_comparison, "scope":"registered_tensor_device_dtype_counts_after_construction", "original":validation.data["model_state_placement"]},
                "shared_state_comparison":{"status":shared_state_comparison, "scope":"registered_parameter_buffer_objects_and_inspectable_storage", "original":validation.data["shared_state"]},
                "note":"生成源码已在新 worker 重导入并验证；不包含训练权重，仍依赖项目工厂、包、数据及运行环境。检查生成源码后可手改。"
            }),
        )
    }

    fn assembly_config_only(&self) -> Report {
        let operation = "assembly_show";
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error(operation, "PROJECT_UNAVAILABLE", error.to_string(), "");
            }
        };
        let config = match self.project_config.load(&root) {
            Ok(Some(config)) => config,
            Ok(None) => {
                return Report::error(
                    operation,
                    "PROJECT_NOT_INITIALIZED",
                    "当前项目尚未保存模型入口。",
                    "先初始化项目。",
                );
            }
            Err(error) => {
                return Report::error(
                    operation,
                    "PROJECT_CONFIG_INVALID",
                    error,
                    "检查项目配置；没有执行或修改模型。",
                );
            }
        };
        let injections = saved_model_injections(&config);
        let Some(injections) = injections.as_object() else {
            return Report::error(
                operation,
                "MODEL_INJECTION_INVALID",
                "项目装配绑定必须是对象。",
                "检查项目配置。 ",
            );
        };
        let slots: Vec<Value> = injections.iter().map(|(slot, binding)| json!({
            "slot":slot, "binding":binding, "registered_paths":[], "sample_forward":"not_run"
        })).collect();
        let mut data = json!({"path":root.join(".model-maven/project.json"),
            "config_revision":crate::domain::input::content_revision(&config).ok(),
            "model":config["model"],"model_params":config.get("model_params").cloned().unwrap_or_else(|| json!({})),"batch":config["batch"],"batch_params":config.get("batch_params").cloned().unwrap_or_else(|| json!({})),"runtime_paths":config["runtime_paths"],"runtime_python":config["runtime_python"],"slots":slots,
            "validation":{"construction":"not_run","forward":"not_run"},
            "note":"仅显示已保存的配置；没有加载模型、验证绑定或执行前向。"});
        let mut redacted_fields = Vec::new();
        redact_run_inputs(&mut data, "$.assembly", &mut redacted_fields);
        data["redacted_fields"] = json!(redacted_fields);
        Report::ok(operation, data)
    }

    fn assembly_show(&self, cli: &Cli, inputs: &AssemblyInputs, collect_source: bool) -> Report {
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error(
                    "assembly_show",
                    "PROJECT_UNAVAILABLE",
                    error.to_string(),
                    "",
                );
            }
        };
        let config = match self.project_config.load(&root) {
            Ok(Some(config)) => config,
            Ok(None) => {
                return Report::error(
                    "assembly_show",
                    "PROJECT_NOT_INITIALIZED",
                    "当前项目尚未保存模型入口。",
                    "运行 m-model init，或在当前目录使用 m-model init --model module:factory。",
                );
            }
            Err(error) => {
                return Report::error(
                    "assembly_show",
                    "PROJECT_CONFIG_INVALID",
                    error,
                    "先备份 .model-maven/project.json；当前 CLI 只读取 schema_version=1。未知版本请用兼容版本导出或迁移。",
                );
            }
        };
        let injections = saved_model_injections(&config);
        if let Err(error) = verify_saved_model_injections(&injections, &*self.components, &root) {
            return Report::error(
                "assembly_show",
                "MODEL_INJECTION_INVALID",
                error,
                "修复或重新登记接缝所引用的组件版本。",
            );
        }
        let mut validation = self.assembly_candidate(
            cli,
            &root,
            &config,
            injections.as_object().unwrap(),
            inputs,
            if collect_source {
                "assembly_generate"
            } else {
                "assembly_show"
            },
            None,
            None,
        );
        if !validation.success() {
            validation.operation = "assembly_show".into();
            return validation;
        }
        let evidence = validation.data["origin"]["injection_evidence"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let config_revision = crate::domain::input::content_revision(&config).ok();
        let model_input = crate::domain::input::resolve(
            "model",
            true,
            false,
            None,
            None,
            None,
            config["model"]
                .as_str()
                .map(|value| crate::domain::input::Candidate {
                    value: Value::String(value.into()),
                    source: crate::domain::operation::InputSource {
                        kind: crate::domain::operation::InputSourceKind::ProjectConfig,
                        reference: Some(".model-maven/project.json#/model".into()),
                        revision: config_revision.clone(),
                    },
                }),
            None,
        );
        let batch_input = crate::domain::input::resolve(
            "batch",
            false,
            false,
            inputs
                .batch
                .as_deref()
                .map(|value| crate::domain::input::Candidate {
                    value: Value::String(value.into()),
                    source: crate::domain::operation::InputSource {
                        kind: crate::domain::operation::InputSourceKind::CommandLine,
                        reference: Some("--batch".into()),
                        revision: None,
                    },
                }),
            None,
            None,
            config["batch"]
                .as_str()
                .map(|value| crate::domain::input::Candidate {
                    value: Value::String(value.into()),
                    source: crate::domain::operation::InputSource {
                        kind: crate::domain::operation::InputSourceKind::ProjectConfig,
                        reference: Some(".model-maven/project.json#/batch".into()),
                        revision: config_revision.clone(),
                    },
                }),
            None,
        );
        let has_batch = batch_input.value.is_some();
        let readiness = match crate::domain::readiness::plan(
            format!(
                "assembly:{}",
                config_revision.as_deref().unwrap_or("unknown")
            ),
            if has_batch {
                "forward_probe"
            } else {
                "inspect_model"
            },
            vec![model_input, batch_input],
            vec![],
            vec![
                crate::domain::readiness::ReadinessCapability {
                    id: "model_construction".into(),
                    state: crate::domain::operation::FactState::Available,
                    required: true,
                    reason: None,
                    provider: Some("python-worker".into()),
                },
                crate::domain::readiness::ReadinessCapability {
                    id: "forward_observation".into(),
                    state: if has_batch {
                        crate::domain::operation::FactState::Available
                    } else {
                        crate::domain::operation::FactState::Unknown
                    },
                    required: has_batch,
                    reason: (!has_batch).then(|| "没有选择示例批次；可先检查模型结构。".into()),
                    provider: has_batch.then(|| "python-worker".into()),
                },
            ],
            vec![],
            if has_batch {
                vec![]
            } else {
                vec![crate::domain::operation::SuggestedAction {
                    id: "configure_batch".into(),
                    label: "选择示例批次以查看前向".into(),
                    availability: crate::domain::operation::ActionAvailability::Available,
                    reason: None,
                }]
            },
            config_revision.clone(),
        ) {
            Ok(plan) => plan,
            Err(error) => {
                return Report::error(
                    "assembly_show",
                    "READINESS_PLAN_INVALID",
                    error,
                    "检查当前模型入口与配置来源，再重新查看装配状态。",
                );
            }
        };
        let slots = injections
            .as_object()
            .into_iter()
            .flat_map(|values| values.iter())
            .map(|(slot, binding)| {
                let observed = evidence
                    .iter()
                    .find(|item| item["slot"].as_str() == Some(slot.as_str()));
                json!({
                    "slot":slot,
                    "binding":binding,
                    "factory":observed.map(|item| item["factory"].clone()).unwrap_or(Value::Null),
                    "params":observed.map(|item| item["params"].clone()).unwrap_or(Value::Null),
                    "component_ref":observed.and_then(|item| item.get("component_ref")).cloned(),
                    "registered_paths":observed.map(|item| item["registered_paths"].clone()).unwrap_or_else(|| json!([])),
                    "shared_with_slots":observed.and_then(|item| item.get("shared_with_slots")).cloned(),
                    "sample_forward":observed.and_then(|item| item.get("sample_forward")).cloned()
                })
            })
            .collect::<Vec<_>>();
        let mut data = json!({
            "path":root.join(".model-maven/project.json"),
            "config_revision":config_revision,
            "readiness":readiness,
            "model":config["model"],
            "batch":inputs.batch.as_deref().or_else(|| config["batch"].as_str()),
            "slots":slots,
            "validation":{
                "construction":"passed",
                "forward":if inputs.batch.is_some() || config["batch"].is_string() { "passed" } else { "not_run" }
            },
            "runtime_paths":validation.data["assembly_runtime_paths"],
            "note":"显示所选运行时中工厂显式暴露的装配接缝；Python 工厂内部图结构仍由 Python 代码控制。"
        });
        let mut redacted_fields = Vec::new();
        redact_run_inputs(&mut data, "$.assembly", &mut redacted_fields);
        data["redacted_fields"] = json!(redacted_fields);
        if collect_source {
            data["source_manifest"] = validation.data["source_manifest"].clone();
            data["input_evidence"] = validation.data["batch_content"].clone();
            data["module_aliases"] = validation.data["module_aliases"].clone();
            data["assembly_runtime_paths"] = validation.data["assembly_runtime_paths"].clone();
        }
        Report::ok("assembly_show", data)
    }

    // The assembly command has distinct optional fields and progress context;
    // keeping them explicit avoids an untyped positional request bundle.
    #[allow(clippy::too_many_arguments)]
    fn assembly_edit(
        &self,
        cli: &Cli,
        action: &str,
        slot: &str,
        component: Option<&str>,
        factory: Option<&str>,
        params: &[String],
        inputs: &AssemblyInputs,
        dry_run: bool,
        context: ExecutionContext<'_>,
    ) -> Report {
        let operation = match action {
            "add" => "assembly_add",
            "set" => "assembly_set",
            _ => "assembly_remove",
        };
        if !python_identifier(slot) {
            return Report::error(
                operation,
                "MODEL_INJECTION_INVALID",
                format!("接缝名必须是 Python 标识符：{slot}"),
                "例如 encoder、decoder 或 head。",
            );
        }
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error(operation, "PROJECT_UNAVAILABLE", error.to_string(), "");
            }
        };
        let config = match self.project_config.load(&root) {
            Ok(Some(config)) => config,
            Ok(None) => {
                return Report::error(
                    operation,
                    "PROJECT_NOT_INITIALIZED",
                    "当前项目尚未保存模型入口。",
                    "先运行 m-model init 保存根模型工厂。",
                );
            }
            Err(error) => {
                return Report::error(
                    operation,
                    "PROJECT_CONFIG_INVALID",
                    error,
                    "先备份 .model-maven/project.json；当前 CLI 只读取 schema_version=1。未知版本请用兼容版本导出或迁移。",
                );
            }
        };
        let base_revision = match crate::domain::input::content_revision(&config) {
            Ok(revision) => revision,
            Err(error) => {
                return Report::error(
                    operation,
                    "PROJECT_CONFIG_REVISION_FAILED",
                    error,
                    "没有写入候选；请检查项目配置是否可序列化。",
                );
            }
        };
        if inputs
            .if_config_revision
            .as_deref()
            .is_some_and(|expected| expected != base_revision)
        {
            return Report::error(
                operation,
                "PROJECT_CONFIG_CONFLICT",
                "项目配置与预览时的修订不一致。",
                "重新预览装配候选；没有写入项目配置。 ",
            );
        }
        if context.confirm_assembly {
            let Some(confirmed_revision) = context.confirmed_assembly_revision else {
                return Report::error(
                    operation,
                    "ASSEMBLY_CONFIRMATION_INVALID",
                    "确认提交缺少候选对应的项目配置 revision。",
                    "重新预览并确认候选；未写入任何配置。",
                );
            };
            if confirmed_revision != base_revision {
                return Report::error(
                    operation,
                    "PROJECT_CONFIG_CONFLICT",
                    format!(
                        "确认期间项目配置已变化：候选基于 {confirmed_revision}，当前为 {base_revision}。"
                    ),
                    "保留新的项目配置；请重新查看接缝并对当前状态创建候选。",
                );
            }
        }
        let existing = saved_model_injections(&config);
        let mut injections = match existing.as_object() {
            Some(values) => values.clone(),
            None => {
                return Report::error(
                    operation,
                    "MODEL_INJECTION_INVALID",
                    "已保存的模型接缝映射必须是对象。",
                    "检查 .model-maven/project.json。",
                );
            }
        };
        match action {
            "add" if injections.contains_key(slot) => {
                return Report::error(
                    operation,
                    "ASSEMBLY_SLOT_EXISTS",
                    format!("接缝 {slot} 已存在。"),
                    "使用 m-model assembly set 替换绑定，或选择另一个接缝名。",
                );
            }
            "set" | "remove" if !injections.contains_key(slot) => {
                return Report::error(
                    operation,
                    "ASSEMBLY_SLOT_NOT_FOUND",
                    format!("项目没有接缝 {slot}。"),
                    "运行 m-model assembly show 查看当前接缝。",
                );
            }
            _ => {}
        }
        if action == "remove" {
            injections.remove(slot);
        } else {
            let binding = match (component, factory) {
                (Some(value), None) => Some(format!("component:{value}")),
                (None, Some(value)) => Some(value.to_owned()),
                (Some(_), Some(_)) => unreachable!("clap rejects conflicting assembly options"),
                (None, None) => None,
            };
            let Some(binding) = binding
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
            else {
                return Report::error(
                    operation,
                    "ASSEMBLY_BINDING_REQUIRED",
                    "需要指定组件或 Python 工厂绑定。",
                    "使用 --component ID@VERSION 或 --factory module:factory。",
                );
            };
            let parsed_params = match parameters(params) {
                Ok(values) => values,
                Err(error) => {
                    return Report::error(
                        operation,
                        "MODEL_INJECTION_INVALID",
                        error,
                        "组件参数使用 --param NAME=JSON，可重复指定。",
                    );
                }
            };
            let mut parameter_values = Vec::new();
            for (name, value) in parsed_params {
                let encoded = match serde_json::to_string(&value) {
                    Ok(encoded) => encoded,
                    Err(error) => {
                        return Report::error(
                            operation,
                            "MODEL_INJECTION_INVALID",
                            error.to_string(),
                            "",
                        );
                    }
                };
                parameter_values.push(format!("{slot}.{name}={encoded}"));
            }
            let resolved = match model_injections(
                &[format!("{slot}={binding}")],
                &parameter_values,
                &*self.components,
                &root,
            ) {
                Ok(values) => values,
                Err(error) => {
                    return Report::error(
                        operation,
                        "MODEL_INJECTION_INVALID",
                        error,
                        "注册 Layer 组件须固定为 ID@精确版本，并按 Schema 传入 --param。",
                    );
                }
            };
            injections.insert(slot.to_owned(), resolved[slot].clone());
            if injections.len() > 32 {
                return Report::error(
                    operation,
                    "MODEL_INJECTION_INVALID",
                    "单个模型最多声明 32 个装配接缝。",
                    "先移除不再使用的接缝。",
                );
            }
        }
        if let Err(error) = verify_saved_model_injections(
            &Value::Object(injections.clone()),
            &*self.components,
            &root,
        ) {
            return Report::error(
                operation,
                "MODEL_INJECTION_INVALID",
                error,
                "修复项目中已保存的组件引用后重试。",
            );
        }
        if cli
            .capture_input_to
            .as_ref()
            .is_some_and(|path| path.to_str().is_none())
            || cli
                .replay_input_from
                .as_ref()
                .is_some_and(|path| path.to_str().is_none())
        {
            return Report::error(
                operation,
                "INPUT_SNAPSHOT_PATH_INVALID",
                "输入快照路径无法以 UTF-8 表示。",
                "选择受支持的临时路径。",
            );
        }
        if (cli.if_input_content.is_some()
            || cli.capture_input_to.is_some()
            || cli.replay_input_from.is_some())
            && inputs.batch.is_none()
            && !config["batch"].is_string()
        {
            return Report::error(
                operation,
                "INPUT_CONTENT_UNAVAILABLE",
                "没有代表批次，无法核对输入内容。",
                "配置批次并重新预览；没有写入项目配置。",
            );
        }
        let mut validation = self.assembly_candidate(
            cli,
            &root,
            &config,
            &injections,
            inputs,
            operation,
            None,
            cli.if_input_content.as_deref(),
        );
        if !validation.success() {
            validation.operation = operation.into();
            return validation;
        }
        let manifest = &validation.data["source_manifest"];
        let runtime_paths = &validation.data["assembly_runtime_paths"];
        let uses_runtime_paths = !runtime_paths["working_directory"].is_null() || !runtime_paths["project_import_roots"].is_null();
        let source_revision = manifest
            .as_object()
            .filter(|entries| {
                !entries.is_empty()
                    && entries.len() <= 4096
                    && entries.iter().all(|(path, hash)| {
                        let path = Path::new(path);
                        path.is_absolute()
                            && path.starts_with(&root)
                            && hash.as_str().is_some_and(|value| {
                                value.len() == 64
                                    && value
                                        .bytes()
                                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                            })
                    })
            })
            .and_then(|_| if uses_runtime_paths {
                crate::domain::input::content_revision(&json!({"sources":manifest,"runtime_paths":runtime_paths})).ok()
            } else {
                crate::domain::input::content_revision(manifest).ok()
            });
        if let Some(expected) = cli.if_source_manifest.as_deref() {
            use sha2::{Digest, Sha256};
            if source_revision.as_deref() != Some(expected)
                || !manifest.as_object().is_some_and(|entries| {
                    entries.iter().all(|(path, hash)| {
                        std::fs::read(path).ok().is_some_and(|bytes| {
                            hash.as_str() == Some(format!("{:x}", Sha256::digest(bytes)).as_str())
                        })
                    })
                })
            {
                return Report::error(
                    operation,
                    "SOURCE_MANIFEST_CONFLICT",
                    "观察到的项目源码与预览不一致或不可核对。",
                    "重新预览装配候选；没有写入项目配置。 ",
                );
            }
        }
        let confirmation_required =
            !dry_run && context.require_assembly_confirmation && !context.confirm_assembly;
        let confirmed = !confirmation_required;
        let commit = !dry_run && confirmed;
        if commit && uses_runtime_paths && cli.if_source_manifest.is_none() {
            return Report::error(operation, "ASSEMBLY_RUNTIME_GUARD_REQUIRED",
                "自定义运行时路径的装配提交需要预览返回的源码和路径摘要。",
                "先使用 --dry-run，再传入 --if-source-manifest <摘要> 提交；没有写入配置。");
        }
        let (committed_path, committed_revision) = if !commit {
            (None, None)
        } else {
            let mut updated = config.clone();
            updated["model_injections"] = Value::Object(injections.clone());
            match self
                .project_config
                .update_versioned(&root, &base_revision, &updated)
            {
                Ok(update) => (Some(update.path), Some(update.revision)),
                Err(error) => {
                    let (code, message) = match error {
                        ProjectConfigUpdateError::Conflict {
                            expected_revision,
                            current_revision,
                        } => (
                            "PROJECT_CONFIG_CONFLICT",
                            format!(
                                "配置基准已过期：期望 {expected_revision}，当前为 {current_revision}。"
                            ),
                        ),
                        ProjectConfigUpdateError::Failed(message)
                            if message.contains("另一项装配编辑") =>
                        {
                            ("PROJECT_CONFIG_LOCKED", message)
                        }
                        ProjectConfigUpdateError::Failed(message) => {
                            ("PROJECT_CONFIG_UPDATE_FAILED", message)
                        }
                    };
                    return Report::error(
                        operation,
                        code,
                        message,
                        "候选未写入；运行 m-model assembly show 确认当前设置后重试。",
                    );
                }
            }
        };
        let evidence = validation.data["origin"]["injection_evidence"].clone();
        let mut data = json!({
            "action":action,
            "slot":slot,
            "committed":commit,
            "cancelled":false,
            "confirmation_required":confirmation_required,
            "base_config_revision":base_revision,
            "path":committed_path.unwrap_or_else(|| root.join(".model-maven/project.json")),
            "config_revision":committed_revision.or_else(|| crate::domain::input::content_revision(&config).ok()),
            "model_injections":injections,
            "injection_evidence":evidence,
            "source_evidence":{"algorithm":"canonical-json-sha256-v1", "scope":if uses_runtime_paths { "worker-recorded-project-sources-and-runtime-paths-v2" } else { "worker-recorded-project-sources" }, "digest":source_revision},
            "runtime_paths":runtime_paths,
            "input_evidence":validation.data["batch_content"],
            "input_snapshot":validation.data["batch_snapshot"],
            "validation":{
                "construction":"passed",
                "forward":if inputs.batch.is_some() || config["batch"].is_string() { "passed" } else { "not_run" }
            },
            "note":if dry_run { "候选已构造验证；project.json 未更改。" } else if confirmation_required { "候选已构造验证；等待确认后写入 project.json。" } else { "候选已通过验证并写入项目默认装配；已有变体保留其创建时的装配快照。" }
        });
        let mut redacted_fields = Vec::new();
        redact_run_inputs(&mut data, "$.assembly", &mut redacted_fields);
        data["redacted_fields"] = json!(redacted_fields);
        Report::ok(operation, data)
    }

    fn assembly_patch(
        &self,
        cli: &Cli,
        sets: &[String],
        removes: &[String],
        params: &[String],
        encoder_outputs: &[String],
        route_changes: &[String],
        if_source_revision: Option<&str>,
        if_input_sha256: Option<&str>,
        inputs: &AssemblyInputs,
        apply: bool,
    ) -> Report {
        let operation = "assembly_patch";
        let fail = |code, message: String| {
            Report::error(
                operation,
                code,
                message,
                "重新查看项目配置并预览完整候选；没有写入配置。",
            )
        };
        if sets.is_empty()
            && removes.is_empty()
            && encoder_outputs.is_empty()
            && route_changes.is_empty()
        {
            return fail(
                "ASSEMBLY_PATCH_EMPTY",
                "至少指定一个 --set、--remove、--encoder-output 或 --route。".into(),
            );
        }
        if cli
            .capture_input_to
            .as_ref()
            .is_some_and(|path| path.to_str().is_none())
            || cli
                .replay_input_from
                .as_ref()
                .is_some_and(|path| path.to_str().is_none())
        {
            return fail(
                "INPUT_SNAPSHOT_PATH_INVALID",
                "输入快照路径无法以 UTF-8 表示。".into(),
            );
        }
        if !inputs.model_params.is_empty()
            || !inputs.batch_params.is_empty()
            || inputs.batch.is_some()
            || inputs.batch_binding.is_some()
        {
            return fail(
                "CLI_ARGUMENT_INVALID",
                "批量装配只使用项目保存的模型和代表批次，不能临时覆盖。".into(),
            );
        }
        if apply && inputs.if_config_revision.is_none() {
            return fail(
                "PROJECT_CONFIG_GUARD_REQUIRED",
                "提交需要预览返回的配置修订。".into(),
            );
        }
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
        };
        let config = match self.project_config.load(&root) {
            Ok(Some(config)) => config,
            Ok(None) => return fail("PROJECT_NOT_INITIALIZED", "项目尚未初始化。".into()),
            Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
        };
        if config["batch"]
            .as_str()
            .is_none_or(|value| value.is_empty())
        {
            return fail(
                "ASSEMBLY_BATCH_REQUIRED",
                "批量装配需要项目保存的代表批次，才能验证完整候选前向。".into(),
            );
        }
        let revision = match crate::domain::input::content_revision(&config) {
            Ok(revision) => revision,
            Err(error) => return fail("PROJECT_CONFIG_REVISION_FAILED", error),
        };
        if inputs
            .if_config_revision
            .as_ref()
            .is_some_and(|expected| expected != &revision)
        {
            return fail(
                "PROJECT_CONFIG_CONFLICT",
                "项目配置在预览后发生变化。".into(),
            );
        }
        let resolved = match model_injections(sets, params, &*self.components, &root) {
            Ok(values) => values,
            Err(error) => return fail("MODEL_INJECTION_INVALID", error),
        };
        let mut injections = match saved_model_injections(&config).as_object() {
            Some(values) => values.clone(),
            None => return fail("MODEL_INJECTION_INVALID", "模型接缝映射无效。".into()),
        };
        let mut removed = std::collections::HashSet::new();
        for slot in removes {
            if !python_identifier(slot) || !removed.insert(slot) || resolved.contains_key(slot) {
                return fail(
                    "ASSEMBLY_PATCH_INVALID",
                    format!("无效、重复或同时设置和移除的接缝：{slot}"),
                );
            }
            if injections.remove(slot).is_none() {
                return fail("ASSEMBLY_SLOT_NOT_FOUND", format!("项目没有接缝 {slot}。"));
            }
        }
        injections.extend(resolved);
        if injections.len() > 32 {
            return fail(
                "MODEL_INJECTION_INVALID",
                "单个模型最多声明 32 个装配接缝。".into(),
            );
        }
        if let Err(error) = verify_saved_model_injections(
            &Value::Object(injections.clone()),
            &*self.components,
            &root,
        ) {
            return fail("MODEL_INJECTION_INVALID", error);
        }
        let mut candidate = config.clone();
        if !encoder_outputs.is_empty() {
            let mut model_params = match &config["model_params"] {
                Value::Null => Map::new(),
                Value::Object(params) => params.clone(),
                _ => {
                    return fail(
                        "SCAFFOLD_ENCODER_MODE_INVALID",
                        "已保存的 model_params 不是对象。".into(),
                    );
                }
            };
            let mut modes = match model_params.get("encoder_output_modes") {
                None | Some(Value::Null) => Map::new(),
                Some(Value::Object(modes)) => modes.clone(),
                Some(_) => {
                    return fail(
                        "SCAFFOLD_ENCODER_MODE_INVALID",
                        "已保存的 encoder_output_modes 不是对象。".into(),
                    );
                }
            };
            let mut seen = std::collections::HashSet::new();
            for entry in encoder_outputs {
                let Some((slot, mode)) = entry.split_once('=') else {
                    return fail(
                        "SCAFFOLD_ENCODER_MODE_INVALID",
                        format!("编码器输出格式须为 SLOT=native|flatten：{entry}"),
                    );
                };
                if !python_identifier(slot)
                    || !matches!(mode, "native" | "flatten")
                    || !seen.insert(slot)
                {
                    return fail(
                        "SCAFFOLD_ENCODER_MODE_INVALID",
                        format!("编码器接缝重复或方式无效：{entry}"),
                    );
                }
                modes.insert(slot.into(), json!(mode));
            }
            model_params.insert("encoder_output_modes".into(), Value::Object(modes));
            candidate["model_params"] = Value::Object(model_params);
        }
        if !route_changes.is_empty() {
            let mut model_params = match &candidate["model_params"] {
                Value::Null => Map::new(),
                Value::Object(params) => params.clone(),
                _ => {
                    return fail(
                        "SCAFFOLD_ROUTE_INVALID",
                        "已保存的 model_params 不是对象。".into(),
                    );
                }
            };
            let mut routes = match model_params.get("input_routes") {
                None | Some(Value::Null) => Map::new(),
                Some(Value::Object(routes)) => routes.clone(),
                Some(_) => {
                    return fail(
                        "SCAFFOLD_ROUTE_INVALID",
                        "已保存的 input_routes 不是对象。".into(),
                    );
                }
            };
            let mut seen = std::collections::HashSet::new();
            for entry in route_changes {
                let Some((slot, input)) = entry.split_once('=') else {
                    return fail(
                        "SCAFFOLD_ROUTE_INVALID",
                        format!("输入接线格式须为 SLOT=INPUT：{entry}"),
                    );
                };
                if !python_identifier(slot) || !python_identifier(input) || !seen.insert(slot) {
                    return fail(
                        "SCAFFOLD_ROUTE_INVALID",
                        format!("接缝重复或输入名无效：{entry}"),
                    );
                }
                routes.insert(slot.into(), json!(input));
            }
            model_params.insert("input_routes".into(), Value::Object(routes));
            candidate["model_params"] = Value::Object(model_params);
        }
        for digest in [if_source_revision, if_input_sha256].into_iter().flatten() {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return fail(
                    "ASSEMBLY_EVIDENCE_INVALID",
                    "源码或输入摘要必须是 64 位小写十六进制 SHA-256。".into(),
                );
            }
        }
        let mut validation = self.assembly_candidate(
            cli,
            &root,
            &candidate,
            &injections,
            inputs,
            operation,
            None,
            if_input_sha256,
        );
        if !validation.success() {
            validation.operation = operation.into();
            return validation;
        }
        let manifest = &validation.data["source_manifest"];
        let source_available = manifest.as_object().is_some_and(|entries| {
            !entries.is_empty()
                && entries.len() <= 4096
                && entries.iter().all(|(path, hash)| {
                    let path = Path::new(path);
                    path.is_absolute()
                        && path.starts_with(&root)
                        && hash.as_str().is_some_and(|value| {
                            value.len() == 64
                                && value
                                    .bytes()
                                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                        })
                })
        });
        let runtime_paths = &validation.data["assembly_runtime_paths"];
        let uses_runtime_paths = !runtime_paths["working_directory"].is_null() || !runtime_paths["project_import_roots"].is_null();
        let source_revision = if source_available {
            if uses_runtime_paths {
                crate::domain::input::content_revision(&json!({"sources":manifest,"runtime_paths":runtime_paths})).ok()
            } else {
                crate::domain::input::content_revision(manifest).ok()
            }
        } else {
            None
        };
        if apply && uses_runtime_paths && if_source_revision.is_none() {
            return fail("ASSEMBLY_RUNTIME_GUARD_REQUIRED", "自定义运行时路径的装配提交需要预览返回的源码和路径摘要。".into());
        }
        if let Some(expected) = if_source_revision {
            if source_revision.as_deref() != Some(expected) {
                return fail(
                    "SOURCE_MANIFEST_CONFLICT",
                    "观察到的项目源码与预览不一致或不可核对。".into(),
                );
            }
            use sha2::{Digest, Sha256};
            for (path, expected_hash) in manifest.as_object().unwrap() {
                if !std::fs::read(path).ok().is_some_and(|bytes| {
                    expected_hash.as_str() == Some(format!("{:x}", Sha256::digest(bytes)).as_str())
                }) {
                    return fail(
                        "SOURCE_MANIFEST_CONFLICT",
                        format!("提交前观察到的源码已变化：{path}"),
                    );
                }
            }
        }
        let committed_revision = if apply {
            let mut updated = candidate.clone();
            updated["model_injections"] = Value::Object(injections.clone());
            match self
                .project_config
                .update_versioned(&root, &revision, &updated)
            {
                Ok(update) => Some(update.revision),
                Err(ProjectConfigUpdateError::Conflict { .. }) => {
                    return fail("PROJECT_CONFIG_CONFLICT", "提交前项目配置发生变化。".into());
                }
                Err(ProjectConfigUpdateError::Failed(error)) => {
                    return fail("PROJECT_CONFIG_UPDATE_FAILED", error);
                }
            }
        } else {
            None
        };
        let mut data = json!({"committed":apply, "base_config_revision":revision, "config_revision":committed_revision,
            "sets":sets, "removes":removes, "model_injections":injections, "encoder_outputs":encoder_outputs, "encoder_output_modes":candidate["model_params"]["encoder_output_modes"], "route_changes":route_changes, "input_routes":candidate["model_params"]["input_routes"],
            "injection_evidence":validation.data["origin"]["injection_evidence"],
            "source_evidence":{"algorithm":"canonical-json-sha256-v1", "scope":if uses_runtime_paths { "worker-recorded-project-sources-and-runtime-paths-v2" } else { "worker-recorded-project-sources" }, "digest":source_revision},
            "runtime_paths":runtime_paths,
            "input_evidence":validation.data["batch_content"],
            "input_snapshot":validation.data["batch_snapshot"],
            "validation":{"construction":"passed", "forward":"passed"},
            "note":"使用已保存代表批次执行完整候选前向；源码、环境和批次内容未固定。"});
        let mut redacted_fields = Vec::new();
        redact_run_inputs(&mut data, "$.assembly", &mut redacted_fields);
        data["redacted_fields"] = json!(redacted_fields);
        Report::ok(operation, data)
    }

    fn assembly_starter_setting(
        &self,
        cli: &Cli,
        slot: &str,
        input: &str,
        output_mode: bool,
        if_source_revision: Option<&str>,
        if_input_sha256: Option<&str>,
        inputs: &AssemblyInputs,
        dry_run: bool,
    ) -> Report {
        let operation = if output_mode {
            "assembly_encoder_output"
        } else {
            "assembly_route"
        };
        let fail = |code, message: String| {
            Report::error(operation, code, message, "重新查看装配并预览更改。 ")
        };
        if cli
            .capture_input_to
            .as_ref()
            .is_some_and(|path| path.to_str().is_none())
            || cli
                .replay_input_from
                .as_ref()
                .is_some_and(|path| path.to_str().is_none())
        {
            return fail(
                "INPUT_SNAPSHOT_PATH_INVALID",
                "输入快照路径无法以 UTF-8 表示。".into(),
            );
        }
        if !python_identifier(slot) || !python_identifier(input) {
            return fail(
                "SCAFFOLD_ROUTE_INVALID",
                "编码器接缝和目标必须是 Python 标识符。".into(),
            );
        }
        if output_mode && !matches!(input, "native" | "flatten") {
            return fail(
                "SCAFFOLD_ENCODER_MODE_INVALID",
                "编码器输出方式须为 native 或 flatten。".into(),
            );
        }
        for digest in [if_source_revision, if_input_sha256].into_iter().flatten() {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return fail(
                    "ASSEMBLY_EVIDENCE_INVALID",
                    "源码或输入摘要必须是 64 位小写十六进制 SHA-256。".into(),
                );
            }
        }
        if !inputs.model_params.is_empty()
            || !inputs.batch_params.is_empty()
            || inputs.batch.is_some()
            || inputs.batch_binding.is_some()
        {
            return fail(
                "CLI_ARGUMENT_INVALID",
                "此验证只使用项目保存的模型和批次参数，不接受临时覆盖。".into(),
            );
        }
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
        };
        let config = match self.project_config.load(&root) {
            Ok(Some(config)) => config,
            Ok(None) => return fail("PROJECT_NOT_INITIALIZED", "项目尚未初始化。".into()),
            Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
        };
        let is_generated_starter = config["model"]
            .as_str()
            .and_then(|value| value.strip_suffix(":model"))
            .filter(|module| module.split('.').all(python_identifier))
            .and_then(|module| {
                let path = root.join(format!("{}.py", module.replace('.', "/")));
                let canonical = path.canonicalize().ok()?;
                if !canonical.starts_with(&root) {
                    return None;
                }
                use std::io::Read;
                let mut prefix = String::new();
                std::fs::File::open(canonical)
                    .ok()?
                    .take(64)
                    .read_to_string(&mut prefix)
                    .ok()?;
                Some(prefix)
            })
            .is_some_and(|source| source.starts_with("# Editable Model-Maven starter."));
        if !(is_generated_starter || config["model"] == "model_starter:model")
            || !config["batch"].is_string()
        {
            return fail(
                "SCAFFOLD_ROUTE_UNSUPPORTED",
                if output_mode {
                    "编码器输出编辑需要支持 encoder_output_modes 的自定义骨架和已配置的代表批次。"
                } else {
                    "接线编辑需要支持 input_routes 的自定义骨架和已配置的代表批次。"
                }
                .into(),
            );
        }
        let revision = match crate::domain::input::content_revision(&config) {
            Ok(revision) => revision,
            Err(error) => return fail("PROJECT_CONFIG_REVISION_FAILED", error),
        };
        if inputs
            .if_config_revision
            .as_deref()
            .is_some_and(|expected| expected != revision)
        {
            return fail(
                "PROJECT_CONFIG_CONFLICT",
                "项目配置在预览后发生变化；未保存更改。".into(),
            );
        }
        if !dry_run && inputs.if_config_revision.is_none() {
            return fail(
                "PROJECT_CONFIG_GUARD_REQUIRED",
                "提交更改需要预览返回的配置修订。".into(),
            );
        }
        let mut candidate = config.clone();
        let mut params = config["model_params"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        let key = if output_mode {
            "encoder_output_modes"
        } else {
            "input_routes"
        };
        let mut routes = match params.get(key) {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(routes)) => routes.clone(),
            Some(_) => {
                return fail(
                    "SCAFFOLD_ROUTE_INVALID",
                    format!("已保存的 {key} 不是对象。"),
                );
            }
        };
        routes.insert(slot.to_owned(), json!(input));
        params.insert(key.into(), Value::Object(routes.clone()));
        candidate["model_params"] = Value::Object(params);
        let injections = saved_model_injections(&candidate);
        let Some(injections) = injections.as_object() else {
            return fail("MODEL_INJECTION_INVALID", "模型接缝配置无效。".into());
        };
        if let Err(error) = verify_saved_model_injections(
            &Value::Object(injections.clone()),
            &*self.components,
            &root,
        ) {
            return fail("MODEL_INJECTION_INVALID", error);
        }
        let validation = self.assembly_candidate(
            cli,
            &root,
            &candidate,
            injections,
            inputs,
            operation,
            None,
            if_input_sha256,
        );
        if !validation.success() {
            return validation;
        }
        let manifest = &validation.data["source_manifest"];
        let source_available = manifest.as_object().is_some_and(|entries| {
            !entries.is_empty()
                && entries.len() <= 4096
                && entries.iter().all(|(path, hash)| {
                    Path::new(path).is_absolute()
                        && Path::new(path).starts_with(&root)
                        && hash.as_str().is_some_and(|value| {
                            value.len() == 64
                                && value
                                    .bytes()
                                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                        })
                })
        });
        let runtime_paths = &validation.data["assembly_runtime_paths"];
        let uses_runtime_paths = !runtime_paths["working_directory"].is_null() || !runtime_paths["project_import_roots"].is_null();
        let source_revision = if source_available {
            if uses_runtime_paths {
                crate::domain::input::content_revision(&json!({"sources":manifest,"runtime_paths":runtime_paths})).ok()
            } else {
                crate::domain::input::content_revision(manifest).ok()
            }
        } else {
            None
        };
        if !dry_run && uses_runtime_paths && if_source_revision.is_none() {
            return fail("ASSEMBLY_RUNTIME_GUARD_REQUIRED", "自定义运行时路径的装配提交需要预览返回的源码和路径摘要。".into());
        }
        if let Some(expected) = if_source_revision {
            if source_revision.as_deref() != Some(expected) {
                return fail(
                    "SOURCE_MANIFEST_CONFLICT",
                    "观察到的项目源码与预览不一致或不可核对。".into(),
                );
            }
            use sha2::{Digest, Sha256};
            for (path, expected_hash) in manifest.as_object().unwrap() {
                if !std::fs::read(path).ok().is_some_and(|bytes| {
                    expected_hash.as_str() == Some(format!("{:x}", Sha256::digest(bytes)).as_str())
                }) {
                    return fail(
                        "SOURCE_MANIFEST_CONFLICT",
                        format!("提交前观察到的源码已变化：{path}"),
                    );
                }
            }
        }
        let committed_revision = if dry_run {
            None
        } else {
            match self
                .project_config
                .update_versioned(&root, &revision, &candidate)
            {
                Ok(update) => Some(update.revision),
                Err(ProjectConfigUpdateError::Conflict { .. }) => {
                    return fail(
                        "PROJECT_CONFIG_CONFLICT",
                        "项目配置在验证后发生变化；未保存更改。".into(),
                    );
                }
                Err(error) => return fail("PROJECT_CONFIG_UPDATE_FAILED", error.message()),
            }
        };
        let mut data = json!({"slot":slot,"committed":!dry_run,"base_config_revision":revision,"config_revision":committed_revision,"validation":{"construction":"passed","forward":"passed"},"note":"使用代表批次执行了前向；源码、环境与批次内容未固定。"});
        data["source_evidence"] = json!({"algorithm":"canonical-json-sha256-v1", "scope":if uses_runtime_paths { "worker-recorded-project-sources-and-runtime-paths-v2" } else { "worker-recorded-project-sources" }, "digest":source_revision});
        data["runtime_paths"] = runtime_paths.clone();
        data["input_evidence"] = validation.data["batch_content"].clone();
        data["input_snapshot"] = validation.data["batch_snapshot"].clone();
        if output_mode {
            data["mode"] = json!(input);
            data["encoder_output_modes"] = json!(routes);
        } else {
            data["input"] = json!(input);
            data["routes"] = json!(routes);
        }
        Report::ok(operation, data)
    }

    pub(super) fn assembly(
        &self,
        cli: &Cli,
        command: &AssemblyCommand,
        context: ExecutionContext<'_>,
    ) -> Report {
        if cli.variant.is_some() && !matches!(command, AssemblyCommand::Generate { .. }) {
            let operation = match command {
                AssemblyCommand::Show { .. } => "assembly_show",
                AssemblyCommand::Patch { .. } => "assembly_patch",
                AssemblyCommand::Add { .. } => "assembly_add",
                AssemblyCommand::Set { .. } => "assembly_set",
                AssemblyCommand::Remove { .. } => "assembly_remove",
                AssemblyCommand::Route { .. } => "assembly_route",
                AssemblyCommand::EncoderOutput { .. } => "assembly_encoder_output",
                AssemblyCommand::Generate { .. } => "assembly_generate",
            };
            return Report::error(
                operation,
                "ASSEMBLY_VARIANT_UNSUPPORTED",
                "assembly 编辑命令修改项目默认接缝，不读取或改写模型变体。",
                "移除 --variant；代码生成可读取所选变体并验证其编辑历史。",
            );
        }
        if matches!(command, AssemblyCommand::Show { inputs, .. } if inputs.if_config_revision.is_some())
        {
            return Report::error(
                "assembly",
                "CLI_ARGUMENT_INVALID",
                "--if-config-revision 不能用于装配查询。",
                "移除此参数后重试。",
            );
        }
        match command {
            AssemblyCommand::Patch {
                sets,
                removes,
                params,
                encoder_outputs,
                route_changes,
                if_source_revision,
                if_input_sha256,
                inputs,
                apply,
            } => self.assembly_patch(
                cli,
                sets,
                removes,
                params,
                encoder_outputs,
                route_changes,
                if_source_revision.as_deref(),
                if_input_sha256.as_deref(),
                inputs,
                *apply,
            ),
            AssemblyCommand::Show {
                inputs,
                config_only,
            } => {
                if *config_only {
                    if inputs.batch.is_some()
                        || inputs.batch_binding.is_some()
                        || !inputs.batch_params.is_empty()
                        || !inputs.model_params.is_empty()
                    {
                        return Report::error(
                            "assembly_show",
                            "CLI_ARGUMENT_INVALID",
                            "--config-only 不执行模型或批次，不能同时传入运行参数。",
                            "移除运行参数，或省略 --config-only。 ",
                        );
                    }
                    self.assembly_config_only()
                } else {
                    self.assembly_show(cli, inputs, false)
                }
            }
            AssemblyCommand::Add {
                slot,
                component,
                factory,
                params,
                inputs,
                dry_run,
            } => self.assembly_edit(
                cli,
                "add",
                slot,
                component.as_deref(),
                factory.as_deref(),
                params,
                inputs,
                *dry_run,
                context,
            ),
            AssemblyCommand::Set {
                slot,
                component,
                factory,
                params,
                inputs,
                dry_run,
            } => self.assembly_edit(
                cli,
                "set",
                slot,
                component.as_deref(),
                factory.as_deref(),
                params,
                inputs,
                *dry_run,
                context,
            ),
            AssemblyCommand::Remove {
                slot,
                inputs,
                dry_run,
            } => self.assembly_edit(
                cli,
                "remove",
                slot,
                None,
                None,
                &[],
                inputs,
                *dry_run,
                context,
            ),
            AssemblyCommand::Route {
                slot,
                input,
                if_source_revision,
                if_input_sha256,
                inputs,
                dry_run,
            } => self.assembly_starter_setting(
                cli,
                slot,
                input,
                false,
                if_source_revision.as_deref(),
                if_input_sha256.as_deref(),
                inputs,
                *dry_run,
            ),
            AssemblyCommand::EncoderOutput {
                slot,
                mode,
                if_source_revision,
                if_input_sha256,
                inputs,
                dry_run,
            } => self.assembly_starter_setting(
                cli,
                slot,
                mode,
                true,
                if_source_revision.as_deref(),
                if_input_sha256.as_deref(),
                inputs,
                *dry_run,
            ),
            AssemblyCommand::Generate { output, inputs } => {
                self.assembly_generate(cli, output, inputs)
            }
        }
    }

    pub(super) fn initialize_project(&self, options: InitOptions<'_>) -> Report {
        let InitOptions {
            model: requested_model,
            batch: requested_batch,
            injections: requested_injections,
            injection_params: requested_injection_params,
            batch_binding: requested_batch_binding,
            batch_params: requested_batch_params,
            dry_run,
        } = options;
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error("init", "PROJECT_UNAVAILABLE", error.to_string(), "");
            }
        };
        match self.project_config.load(&root) {
            Ok(Some(existing)) => {
                return Report::error(
                    "init",
                    "PROJECT_ALREADY_INITIALIZED",
                    format!(
                        "项目已有默认模型入口：{}。",
                        existing["model"].as_str().unwrap_or("未知")
                    ),
                    "配置文件是 .model-maven/project.json；确认后可直接编辑其 model、batch 字段。",
                );
            }
            Ok(None) => {}
            Err(error) => {
                return Report::error(
                    "init",
                    "PROJECT_CONFIG_INVALID",
                    error,
                    "先备份 .model-maven/project.json；当前 CLI 只读取 schema_version=1。未知版本请用兼容版本导出或迁移；初始化不会覆盖原文件。",
                );
            }
        }

        let model = requested_model
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        let Some(model) = model else {
            return Report::error(
                "init",
                "MODEL_REQUIRED",
                "初始化需要一个原生 PyTorch 模型工厂入口。",
                "使用 m-model init --model package.module:build_model。",
            );
        };
        let batch = requested_batch
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        if requested_batch_binding.is_some() && batch.is_none() {
            return Report::error(
                "init",
                "BATCH_BINDING_WITHOUT_BATCH",
                "--batch-binding 需要同时设置 --batch。",
                "为批次工厂提供 module:callable，或省略 --batch-binding。",
            );
        }
        let batch_binding = requested_batch_binding.map(|binding| match binding {
            BatchBinding::Envelope => "envelope",
            BatchBinding::Single => "single",
            BatchBinding::SingleRaw => "single-raw",
            BatchBinding::Args => "args",
            BatchBinding::Kwargs => "kwargs",
        });
        if batch.is_none() && !requested_batch_params.is_empty() {
            return Report::error(
                "init",
                "BATCH_PARAMS_WITHOUT_BATCH",
                "批次参数需要批次工厂。",
                "指定批次入口或移除参数。",
            );
        }
        let batch_params = match parameters(requested_batch_params) {
            Ok(params) => params,
            Err(error) => {
                return Report::error(
                    "init",
                    "PARAMETER_INVALID",
                    error,
                    "使用 NAME=JSON 格式的批次参数。",
                );
            }
        };
        let model_injections = match model_injections(
            requested_injections,
            requested_injection_params,
            &*self.components,
            &root,
        ) {
            Ok(values) => values,
            Err(error) => {
                return Report::error(
                    "init",
                    "MODEL_INJECTION_INVALID",
                    error,
                    "接缝必须是根工厂显式声明的关键字参数；每个工厂返回一个 nn.Module。",
                );
            }
        };
        let mut config = json!({
            "schema_version":1,
            "model":model,
            "batch":batch,
            "batch_binding":batch_binding,
            "model_injections":model_injections
        });
        if !batch_params.is_empty() {
            config["batch_params"] = json!(batch_params);
        }
        if dry_run {
            return Report::ok(
                "init",
                json!({
                    "committed":false,
                    "path":root.join(".model-maven/project.json"),
                    "config":config,
                    "config_revision":crate::domain::input::content_revision(&config).ok(),
                    "note":"仅展示将保存的入口；没有创建或覆盖文件。",
                }),
            );
        }
        match self.project_config.create(&root, &config) {
            Ok(path) => Report::ok(
                "init",
                json!({
                    "committed":true,
                    "path":path,
                    "config":config,
                    "config_revision":crate::domain::input::content_revision(&config).ok(),
                    "note":"默认入口只保存在项目元数据中；后续 CLI 参数仍可覆盖。没有改写用户 Python 源码或安装依赖。",
                }),
            ),
            Err(error) if error.contains("已存在") => Report::error(
                "init",
                "PROJECT_ALREADY_INITIALIZED",
                error,
                "检查 .model-maven/project.json；初始化不会覆盖项目配置。",
            ),
            Err(error) => Report::error(
                "init",
                "PROJECT_CONFIG_WRITE_FAILED",
                error,
                "检查当前项目的 .model-maven 目录权限。",
            ),
        }
    }

    pub fn project_defaults(&self, root: &Path) -> Result<Option<Value>, String> {
        self.project_config.load(root)
    }
}
