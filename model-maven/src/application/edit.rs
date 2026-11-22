use super::*;

/// Static input hints for the built-in factory allowlist. PyTorch construction
/// and candidate forward remain the authoritative compatibility checks.
pub(super) fn layer_catalog() -> Report {
    Report::ok(
        "layer_catalog",
        json!({
            "catalog_version":1,
            "scope":"common-pytorch-constructor-fields",
            "runtime_validation":"not_run",
            "layers":[
                {"name":"Linear","fields":[{"name":"in_features","type":"integer","required":true},{"name":"out_features","type":"integer","required":true},{"name":"bias","type":"boolean"}]},
                {"name":"Dropout","fields":[{"name":"p","type":"number","minimum":0,"maximum":1},{"name":"inplace","type":"boolean"}]},
                {"name":"ReLU","fields":[{"name":"inplace","type":"boolean"}]},
                {"name":"GELU","fields":[{"name":"approximate","type":"string","enum":["none","tanh"]}]},
                {"name":"LayerNorm","fields":[{"name":"normalized_shape","type":"json","required":true,"description":"integer or array of integers"},{"name":"eps","type":"number"},{"name":"elementwise_affine","type":"boolean"},{"name":"bias","type":"boolean"}]},
                {"name":"Conv2d","fields":[{"name":"in_channels","type":"integer","required":true},{"name":"out_channels","type":"integer","required":true},{"name":"kernel_size","type":"json","required":true,"description":"integer or pair of integers"},{"name":"stride","type":"json"},{"name":"padding","type":"json"},{"name":"dilation","type":"json"},{"name":"groups","type":"integer"},{"name":"bias","type":"boolean"},{"name":"padding_mode","type":"string","enum":["zeros","reflect","replicate","circular"]}]},
                {"name":"Identity","fields":[]}
            ],
            "note":"参数提示覆盖常用字段；具体 PyTorch 版本和真实输入仍须通过候选构造与前向验证。"
        }),
    )
}

impl<B: Backend> Application<B> {
    pub(super) fn handle_variant(&self, command: &VariantCommand) -> Report {
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error("variant", "PROJECT_UNAVAILABLE", error.to_string(), "");
            }
        };
        match command {
            VariantCommand::List { query } => match self.variants.list(&root) {
                Ok(mut variants) => {
                    if let Some(query) = query.as_deref().map(str::to_lowercase) {
                        variants.retain(|variant| {
                            variant["name"]
                                .as_str()
                                .unwrap_or("")
                                .to_lowercase()
                                .contains(&query)
                                || variant["model"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_lowercase()
                                    .contains(&query)
                        });
                    }
                    Report::ok("variant_list", json!({"variants":variants}))
                }
                Err(error) => Report::error(
                    "variant_list",
                    "VARIANT_LIST_FAILED",
                    error,
                    "检查 .model-maven/variants 目录和文件权限。",
                ),
            },
            VariantCommand::Show { name } => match self.variants.load(&root, name) {
                Ok(Some(mut variant)) => {
                    let token = match super::run_prepare::variant_base_token(Some(&variant)) {
                        Ok(token) => token,
                        Err(error) => {
                            return Report::error(
                                "variant_show",
                                "VARIANT_BASE_UNAVAILABLE",
                                error,
                                "重新检查变体。",
                            );
                        }
                    };
                    let mut redacted_fields = Vec::new();
                    redact_run_inputs(&mut variant, "$.variant", &mut redacted_fields);
                    Report::ok(
                        "variant_show",
                        json!({
                            "name":name,
                            "variant":variant,
                            "variant_base":{"name":name,"token":token,"algorithm":"canonical-json-sha256-v1"},
                            "redacted_fields":redacted_fields,
                            "replay_validation":"not_run",
                            "note":"详情只读取描述文件并脱敏常见凭证字段；使用 model inspect 或 Layer 编辑时会校验变体摘要、源码和组件指纹。"
                        }),
                    )
                }
                Ok(None) => Report::error(
                    "variant_show",
                    "VARIANT_NOT_FOUND",
                    format!("找不到模型变体：{name}"),
                    "先运行 m-model variant list 查看现有变体。",
                ),
                Err(error) => Report::error(
                    "variant_show",
                    "VARIANT_READ_FAILED",
                    error,
                    "检查变体 JSON 文件；详情读取不会修改它。",
                ),
            },
            VariantCommand::Copy {
                source,
                destination,
            } => {
                if source == destination {
                    return Report::error(
                        "variant_copy",
                        "VARIANT_NAME_CONFLICT",
                        "源和目标变体名称必须不同。",
                        "选择新的目标名称作为编辑分支。",
                    );
                }
                if let Err(error) = self.variants.path(&root, source) {
                    return Report::error(
                        "variant_copy",
                        "VARIANT_NAME_INVALID",
                        error,
                        "变体名只能包含字母、数字、-、_。",
                    );
                }
                if let Err(error) = self.variants.path(&root, destination) {
                    return Report::error(
                        "variant_copy",
                        "VARIANT_NAME_INVALID",
                        error,
                        "变体名只能包含字母、数字、-、_。",
                    );
                }
                let source_lock = match self.variants.lock(&root, source) {
                    Ok(lock) => lock,
                    Err(error) => {
                        return Report::error(
                            "variant_copy",
                            "VARIANT_LOCKED",
                            error,
                            "稍后重试，或检查源变体是否正被编辑。",
                        );
                    }
                };
                let variant = match self.variants.load(&root, source) {
                    Ok(Some(variant)) => variant,
                    Ok(None) => {
                        drop(source_lock);
                        return Report::error(
                            "variant_copy",
                            "VARIANT_NOT_FOUND",
                            format!("找不到模型变体：{source}"),
                            "先运行 m-model variant list 查看现有变体。",
                        );
                    }
                    Err(error) => {
                        drop(source_lock);
                        return Report::error(
                            "variant_copy",
                            "VARIANT_READ_FAILED",
                            error,
                            "源变体未复制；检查 JSON 和文件权限。",
                        );
                    }
                };
                if variant["schema_version"] != 1
                    || !variant["origin"]["model"].is_string()
                    || !variant["origin"]["project_root"].is_string()
                    || !variant["origin"]["params"].is_object()
                    || variant["seed"].as_u64().is_none()
                    || !variant["torch_version"].is_string()
                    || !variant["source_manifest"].is_object()
                    || !variant["edits"].is_array()
                    || !variant["cursor"].as_u64().is_some_and(|cursor| {
                        cursor <= variant["edits"].as_array().map_or(0, Vec::len) as u64
                    })
                    || !variant["revision"].is_string()
                {
                    drop(source_lock);
                    return Report::error(
                        "variant_copy",
                        "VARIANT_INVALID",
                        "源文件不是受支持的模型变体描述。",
                        "不会复制损坏或未知版本的变体。",
                    );
                }
                let _destination_lock = match self.variants.lock(&root, destination) {
                    Ok(lock) => lock,
                    Err(error) => {
                        drop(source_lock);
                        return Report::error(
                            "variant_copy",
                            "VARIANT_LOCKED",
                            error,
                            "稍后重试，或检查目标变体是否正被编辑。",
                        );
                    }
                };
                match self.variants.create(&root, destination, &variant) {
                    Ok(()) => Report::ok(
                        "variant_copy",
                        json!({
                            "source":source,
                            "destination":destination,
                            "revision":variant["revision"],
                            "path":self.variants.path(&root, destination).ok(),
                            "note":"这是独立结构配方分支；源码与训练权重未复制。下一步可用 --variant <目标名> 编辑。"
                        }),
                    ),
                    Err(error) => Report::error(
                        "variant_copy",
                        "VARIANT_CREATE_FAILED",
                        error,
                        "目标不会覆盖已有变体；源变体保持不变。",
                    ),
                }
            }
        }
    }
}

pub(super) fn build_layer_job(command: &LayerCommand) -> Result<Job, Report> {
    let empty = empty_job;
    let (edit_value, options, inputs) = match command {
        LayerCommand::Catalog => {
            return Err(Report::error(
                "layer_catalog",
                "CLI_ARGUMENT_INVALID",
                "目录命令不能作为模型执行任务。",
                "直接运行 m-model layer catalog。",
            ));
        }
        LayerCommand::MigrateStarter { edit } => (
            json!({"action":"migrate_starter"}),
            Some(edit),
            edit.inputs.clone(),
        ),
        LayerCommand::List {
            parent,
            inputs,
            layer_type,
            search,
        } => {
            return Ok(Job {
                operation: "layers",
                path: parent.clone(),
                inputs: inputs.clone(),
                layer_type: layer_type.clone(),
                search: search.clone(),
                layer_context: true,
                ..empty()
            });
        }
        LayerCommand::Show { path, inputs } => {
            return Ok(Job {
                operation: "layers",
                path: Some(path.clone()),
                inputs: inputs.clone(),
                layer_context: true,
                ..empty()
            });
        }
        LayerCommand::Add {
            parent,
            before,
            after,
            name,
            layer_type,
            component,
            edit,
        } => (
            json!({"action":"add", "parent": parent, "before": before, "after": after, "name":name, "layer_type":layer_type, "component":component}),
            Some(edit),
            edit.inputs.clone(),
        ),
        LayerCommand::Set { path, edit } => (
            json!({"action":"set", "path":path}),
            Some(edit),
            edit.inputs.clone(),
        ),
        LayerCommand::Replace {
            path,
            layer_type,
            component,
            edit,
        } => (
            json!({"action":"replace", "path":path, "layer_type":layer_type, "component":component}),
            Some(edit),
            edit.inputs.clone(),
        ),
        LayerCommand::Remove {
            path,
            strategy,
            edit,
        } => (
            json!({"action":"remove", "path":path, "strategy":strategy}),
            Some(edit),
            edit.inputs.clone(),
        ),
        LayerCommand::Apply { operations, edit } => {
            let mut parsed = Vec::with_capacity(operations.len());
            for operation in operations {
                let value: Value = serde_json::from_str(operation).map_err(|error| {
                    Report::error(
                        "edit",
                        "EDIT_PLAN_INVALID",
                        format!("Layer 操作不是有效 JSON：{error}"),
                        "每个 --operation 都是一个 JSON 对象，含 action/path/params。",
                    )
                })?;
                if !value.is_object()
                    || !matches!(
                        value["action"].as_str(),
                        Some("add" | "set" | "replace" | "remove" | "custom" | "sharing")
                    )
                {
                    return Err(Report::error(
                        "edit",
                        "EDIT_PLAN_INVALID",
                        "每项操作必须是 add/set/replace/remove/custom/sharing JSON 对象。",
                        "undo/redo 不可嵌套在原子编辑计划中。",
                    ));
                }
                parsed.push(value);
            }
            (
                json!({"action":"transaction", "operations":parsed}),
                Some(edit),
                edit.inputs.clone(),
            )
        }
        LayerCommand::Custom {
            provider,
            payload,
            edit,
        } => {
            let payload: Value = serde_json::from_str::<Value>(payload).map_err(|error| {
                Report::error(
                    "edit",
                    "CUSTOM_EDIT_INVALID",
                    format!("--payload 不是有效 JSON：{error}"),
                    "传入 JSON 对象，例如 --payload '{\"name\":\"silu\"}'。",
                )
            })?;
            if !payload.is_object() {
                return Err(Report::error(
                    "edit",
                    "CUSTOM_EDIT_INVALID",
                    "--payload 必须是 JSON 对象。",
                    "查看 provider 的组件 Schema 后传入对象字段。",
                ));
            }
            (
                json!({"action":"custom", "provider":provider, "payload":payload}),
                Some(edit),
                edit.inputs.clone(),
            )
        }
        LayerCommand::Undo { inputs, target, .. } => (
            json!({"action":"undo", "target":target}),
            None,
            inputs.clone(),
        ),
        LayerCommand::Redo { inputs, target, .. } => (
            json!({"action":"redo", "target":target}),
            None,
            inputs.clone(),
        ),
    };
    let mut edit_value = edit_value;
    if let Some(options) = options {
        let source_action_supported = match edit_value["action"].as_str() {
            Some("set" | "add" | "replace" | "remove" | "migrate_starter") => true,
            Some("transaction") => source_transaction_supported(&edit_value["operations"]),
            _ => false,
        };
        if options.target == "source" && !source_action_supported {
            return Err(Report::error(
                "edit",
                "SOURCE_EDIT_UNSUPPORTED",
                "源码处理器支持单项 Sequential/ModuleList CRUD、最多 64 项 set 事务，或最多 64 项结构 CRUD 事务。",
                "一个事务须全部是 set，或全部是 add/replace/remove；结构操作支持内置 Layer 和可导入的 Python 工厂。",
            ));
        }
        if edit_value["action"] == "migrate_starter" && options.target != "source" {
            return Err(Report::error(
                "edit",
                "SOURCE_TARGET_REQUIRED",
                "骨架迁移必须明确选择源码目标。",
                "使用 --target source --dry-run 先审核差异；迁移不会修改变体。",
            ));
        }
        if edit_value["action"] == "migrate_starter" && !options.params.is_empty() {
            return Err(Report::error(
                "edit",
                "EDIT_PLAN_INVALID",
                "骨架迁移不接受 Layer 参数。",
                "移除 --param 后重新预览。",
            ));
        }
        edit_value["target"] = json!(options.target);
        if edit_value["action"] == "transaction" && !options.params.is_empty() {
            return Err(Report::error(
                "edit",
                "EDIT_PLAN_INVALID",
                "原子计划的参数必须写在各自的 --operation JSON 中。",
                "每项操作使用 params 对象。",
            ));
        }
        if edit_value["action"] == "custom" && !options.params.is_empty() {
            return Err(Report::error(
                "edit",
                "EDIT_PLAN_INVALID",
                "自定义 provider 使用 --payload，不能同时传 --param。",
                "按 provider 的 payload Schema 传 JSON 对象。",
            ));
        }
        if edit_value["action"] != "transaction" && edit_value["action"] != "custom" {
            let params = parameters(&options.params).map_err(|error| {
                Report::error(
                    "edit",
                    "PARAMETER_INVALID",
                    error,
                    "参数是 JSON 数据，不是 Python 表达式。",
                )
            })?;
            edit_value["params"] = json!(params);
        }
    }
    let (history_dry_run, history_forward) = match command {
        LayerCommand::Undo {
            target,
            dry_run,
            require_forward,
            ..
        }
        | LayerCommand::Redo {
            target,
            dry_run,
            require_forward,
            ..
        } => {
            if target == "source" && (*dry_run || *require_forward) {
                return Err(Report::error(
                    "edit",
                    "SOURCE_HISTORY_PREVIEW_UNSUPPORTED",
                    "源码撤销暂不支持预览或前向验证。",
                    "仅变体历史支持这些选项。",
                ));
            }
            (*dry_run, *require_forward)
        }
        _ => (false, false),
    };
    Ok(Job {
        operation: "edit",
        inputs,
        edit: Some(edit_value),
        layer_context: true,
        dry_run: history_dry_run || options.is_some_and(|options| options.dry_run),
        require_forward: history_forward || options.is_some_and(|options| options.require_forward),
        ..empty()
    })
}
impl<B: Backend> Application<B> {
    fn verify_source_configuration(&self, root: &Path, expected: &str) -> Result<(), String> {
        let current = self
            .project_config
            .load(root)
            .map_err(|error| format!("无法读取项目配置：{error}"))?
            .ok_or_else(|| "项目配置已不存在。".to_owned())?;
        let revision = crate::domain::input::content_revision(&current)
            .map_err(|error| format!("无法计算项目配置修订：{error}"))?;
        if revision != expected {
            return Err("项目配置在源码预览或验证后发生变化。".into());
        }
        Ok(())
    }

    fn verify_source_files(
        &self,
        root: &Path,
        manifest: &Value,
        patches: &[Value],
        after: bool,
    ) -> Result<(), String> {
        use sha2::{Digest, Sha256};
        let entries = manifest
            .as_object()
            .filter(|items| !items.is_empty())
            .ok_or("项目源码清单不可用。")?;
        let mut replaced = std::collections::HashMap::new();
        if after {
            for patch in patches {
                let relative = patch["path"].as_str().ok_or("源码补丁路径不可用。")?;
                let path = root
                    .join(relative)
                    .canonicalize()
                    .map_err(|_| "修改后的源码路径不可用。")?;
                if !path.starts_with(root) {
                    return Err("源码补丁路径离开项目。".into());
                }
                let hash = patch["after_sha256"]
                    .as_str()
                    .ok_or("源码补丁摘要不可用。")?;
                replaced.insert(path, hash);
            }
        }
        for (name, hash) in entries {
            let path = Path::new(name);
            let resolved = path
                .canonicalize()
                .map_err(|_| format!("记录的项目源码不可读取：{name}"))?;
            if !path.is_absolute() || !resolved.starts_with(root) || resolved != path {
                return Err(format!("记录的项目源码路径已变化：{name}"));
            }
            let expected = replaced
                .get(&resolved)
                .copied()
                .or_else(|| hash.as_str())
                .ok_or("项目源码摘要不可用。")?;
            let bytes =
                std::fs::read(&resolved).map_err(|_| format!("记录的项目源码不可读取：{name}"))?;
            if format!("{:x}", Sha256::digest(&bytes)) != expected {
                return Err(format!("记录的项目源码已变化：{name}"));
            }
        }
        Ok(())
    }

    // Finalization must keep the target, request and candidate result explicit
    // because source and variant commits have different rollback semantics.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn finalize_edit_result(
        &self,
        root: &std::path::Path,
        job: &Job,
        name: &str,
        variant_path: &std::path::PathBuf,
        source_target: bool,
        request: &Value,
        mut report: Report,
    ) -> Report {
        if report.success() && job.edit.is_some() {
            if source_target {
                let patch = report.data["source_patch"].clone();
                let patches = report.data["source_patches"]
                    .as_array()
                    .cloned()
                    .unwrap_or_else(|| vec![patch]);
                let paths: Vec<String> = patches
                    .iter()
                    .filter_map(|item| item["path"].as_str().map(ToOwned::to_owned))
                    .collect();
                let patches_valid = !patches.is_empty()
                    && patches.len() <= 64
                    && patches.len() == paths.len()
                    && patches.iter().all(|item| {
                        item["before_sha256"]
                            .as_str()
                            .is_some_and(|value| !value.is_empty())
                            && item["contents"].as_str().is_some()
                    });
                if !patches_valid {
                    return Report::error(
                        "edit",
                        "SOURCE_PATCH_INVALID",
                        "源码处理器没有返回完整的文件变更计划。",
                        "原文件保持不变；检查源码处理器和 Python 版本。",
                    );
                }
                let shared_groups = report.data["shared_state_evidence"]["groups"]
                    .as_array()
                    .cloned();
                if report.data["shared_state_evidence"]["scope"]
                    != "in_memory_candidate_object_storage_v2"
                    || shared_groups.is_none()
                {
                    return Report::error(
                        "edit",
                        "SOURCE_SHARED_STATE_EVIDENCE_MISSING",
                        "源码候选未提供完整的共享状态证据，无法安全提交。",
                        "使用支持共享状态核验的 CLI 与 worker 后重新预览。",
                    );
                }
                let shared_groups = shared_groups.unwrap();
                if !job.dry_run {
                    let expected_config = request["expected_source_config"].as_str();
                    if let Some(expected) = expected_config {
                        if let Err(error) = self.verify_source_configuration(root, expected) {
                            return Report::error(
                                "edit",
                                "PROJECT_CONFIG_CONFLICT",
                                error,
                                "源码未修改；重新预览当前项目配置。",
                            );
                        }
                    }
                    let source_manifest = request
                        .get("expected_source_files")
                        .filter(|value| !value.is_null());
                    if let Some(manifest) = source_manifest {
                        if let Err(error) =
                            self.verify_source_files(root, manifest, &patches, false)
                        {
                            return Report::error(
                                "edit",
                                "SOURCE_MANIFEST_CONFLICT",
                                error,
                                "源码未修改；重新预览当前项目源码。",
                            );
                        }
                    }
                    let receipt = match self.source_edits.commit_batch(root, &patches) {
                        Ok(receipt) => receipt,
                        Err(error) => {
                            return Report::error(
                                "edit",
                                "SOURCE_EDIT_COMMIT_FAILED",
                                error,
                                "计划可能因文件冲突而过期；若事务日志显示恢复未完成，不要删除 pending.json，并先处理列出的文件。",
                            );
                        }
                    };
                    let mut verification_request = request.clone();
                    let verification_operation = if request["batch"].is_string() {
                        "probe"
                    } else {
                        "inspect"
                    };
                    verification_request["operation"] = json!(verification_operation);
                    verification_request["edit"] = Value::Null;
                    verification_request["variant"] = Value::Null;
                    verification_request["require_forward"] = Value::Bool(false);
                    verification_request["expected_shared_state_groups"] = json!(shared_groups);
                    let verification = self.backend.execute(&verification_request);
                    let sharing_verified = verification.data["shared_state_verification"]["status"]
                        == "passed"
                        && verification.data["shared_state_verification"]["scope"]
                            == "registered_object_and_storage_after_validation"
                        && verification.data["shared_state_verification"]["group_count"].as_u64()
                            == Some(shared_groups.len() as u64);
                    let input_verified = match request["expected_input_content"].as_str() {
                        None => true,
                        Some(expected) => {
                            verification_operation == "probe"
                                && verification.data["batch_content"]["status"] == "available"
                                && verification.data["batch_content"]["algorithm"]
                                    == "sha256-forward-input-v1"
                                && verification.data["batch_content"]["digest"].as_str()
                                    == Some(expected)
                                && verification.data["evidence_cache"]["status"] != "reused"
                                && (request["replay_input_from"].is_null()
                                    || (verification.data["batch_snapshot"]["status"]
                                        == "replayed"
                                        && verification.data["batch_snapshot"]["format"]
                                            == "model-maven-forward-input-snapshot-v1"))
                        }
                    };
                    let config_error = expected_config.and_then(|expected| {
                        self.verify_source_configuration(root, expected).err()
                    });
                    let manifest_error = source_manifest.and_then(|manifest| {
                        self.verify_source_files(root, manifest, &patches, true)
                            .err()
                    });
                    if !verification.success()
                        || !sharing_verified
                        || !input_verified
                        || config_error.is_some()
                        || manifest_error.is_some()
                    {
                        match self.source_edits.rollback(root, &receipt) {
                            Ok(rollback) => {
                                let mut failed = Report::error(
                                    "edit",
                                    "SOURCE_EDIT_VALIDATION_FAILED",
                                    config_error
                                        .clone()
                                        .or_else(|| manifest_error.clone())
                                        .or_else(|| {
                                            verification
                                                .diagnostics
                                                .first()
                                                .map(|item| item.message.clone())
                                        })
                                        .unwrap_or_else(|| "源码变更后的模型验证失败。".into()),
                                    "已自动恢复本次事务涉及的全部文件；先使用 --dry-run 查看计划与构造结果。",
                                );
                                failed.data = json!({
                                    "source_paths":paths,
                                    "rollback":rollback,
                                    "configuration_error":config_error,
                                    "source_manifest_error":manifest_error,
                                    "verification":verification
                                });
                                return failed;
                            }
                            Err(rollback_error) => {
                                let mut failed = Report::error(
                                    "edit",
                                    "SOURCE_EDIT_RECOVERY_REQUIRED",
                                    format!("验证失败，且自动回滚遇到冲突：{rollback_error}"),
                                    "不要继续编辑这些文件；根据回执中的各文件备份手动恢复，再检查 .model-maven/source-edits/pending.json。",
                                );
                                failed.data = json!({
                                    "source_paths":paths,
                                    "receipt":receipt,
                                    "configuration_error":config_error,
                                    "source_manifest_error":manifest_error,
                                    "verification":verification
                                });
                                return failed;
                            }
                        }
                    }
                    report.data["source_edit"] = receipt;
                    report.data["source_validation"] = json!({
                        "operation":verification_operation,
                        "status":"passed",
                        "modules":verification.data["modules"].as_array().map(Vec::len),
                        "outputs":verification.data["outputs"],
                        "batch_content":verification.data["batch_content"],
                        "batch_snapshot":verification.data["batch_snapshot"],
                        "source_config_revision":expected_config,
                        "source_manifest_digest":request["expected_source_files_digest"],
                        "source_manifest_status":if source_manifest.is_some() { json!("passed") } else { Value::Null },
                        "shared_state":verification.data["shared_state_verification"]
                    });
                }
                report.data["committed"] = json!(!job.dry_run);
                report.data["target"] = json!("source");
                if paths.len() == 1 {
                    report.data["source_path"] = json!(paths[0]);
                } else {
                    report.data["source_paths"] = json!(paths);
                }
                if let Some(plan) = report.data["source_patch"].as_object_mut() {
                    plan.remove("contents");
                }
                if let Some(plans) = report.data["source_patches"].as_array_mut() {
                    for plan in plans {
                        if let Some(plan) = plan.as_object_mut() {
                            plan.remove("contents");
                        }
                    }
                }
            } else {
                if !job.dry_run
                    && let Err(error) = self.variants.save(root, name, &report.data["variant"])
                {
                    return Report::error(
                        job.operation,
                        "VARIANT_WRITE_FAILED",
                        error,
                        "原源码未修改；检查目录和磁盘。",
                    );
                }
                report.data["committed"] = json!(!job.dry_run);
                report.data["variant_path"] = json!(variant_path);
                report.data["variant_name"] = json!(name);
            }
        } else if report.success() && request["variant"].is_object() {
            report.data["variant_name"] = json!(name);
        }
        report
    }
}

pub(super) enum EditPreparation {
    Ready { source_target: bool },
    Finished(Report),
}

impl<B: Backend> Application<B> {
    pub(super) fn prepare_edit_execution(
        &self,
        cli: &Cli,
        job: &mut Job,
        root: &std::path::Path,
    ) -> Result<EditPreparation, Report> {
        let source_target = job
            .edit
            .as_ref()
            .is_some_and(|edit| edit["target"] == "source");
        if source_target
            && job
                .edit
                .as_ref()
                .is_some_and(|edit| edit["action"] == "migrate_starter")
            && !job.dry_run
            && cli.if_source_plan.is_none()
        {
            return Err(Report::error(
                "edit",
                "SOURCE_PLAN_REQUIRED",
                "骨架迁移需要先预览并确认精确源码计划。",
                "先使用 --dry-run，确认差异后将返回的摘要传给 --if-source-plan。",
            ));
        }
        let source_history_action = job
            .edit
            .as_ref()
            .is_some_and(|edit| matches!(edit["action"].as_str(), Some("undo" | "redo")));
        if source_target && cli.variant.is_some() && !source_history_action {
            return Err(Report::error(
                "edit",
                "SOURCE_VARIANT_CONFLICT",
                "源码编辑针对 Python 基础源码，不能同时选择 Layer 变体。",
                "移除 --variant；需要并行试验时使用持久变体目标。",
            ));
        }
        if source_target
            && job
                .edit
                .as_ref()
                .is_some_and(|edit| match edit["action"].as_str() {
                    Some("set" | "add" | "replace" | "remove" | "migrate_starter") => true,
                    Some("transaction") => source_transaction_supported(&edit["operations"]),
                    _ => false,
                })
            && job.inputs.model.is_none()
        {
            return Err(Report::error(
                "edit",
                "MODEL_REQUIRED",
                "源码编辑需要明确的模型工厂入口。",
                "使用 --model module:factory 并确认该 Layer 来自项目内 .py 构造代码。",
            ));
        }
        if let Some(edit) = job.edit.as_ref()
            && edit["target"] == "source"
            && matches!(edit["action"].as_str(), Some("undo" | "redo"))
        {
            let _source_lock: Box<dyn SourceEditLock> =
                self.source_edits.lock(root).map_err(|error| {
                    Report::error("edit", "SOURCE_EDIT_LOCKED", error, "稍后重试源码编辑。")
                })?;
            let result = if edit["action"] == "undo" {
                self.source_edits.undo(root)
            } else {
                self.source_edits.redo(root)
            };
            return Ok(EditPreparation::Finished(match result {
                Ok(data) => Report::ok(
                    "edit",
                    json!({"target":"source","committed":true,"source_edit":data}),
                ),
                Err(error) => Report::error(
                    "edit",
                    "SOURCE_EDIT_HISTORY_FAILED",
                    error,
                    "文件指纹冲突时不会覆盖手工修改；检查 .model-maven/source-edits/history.json。",
                ),
            }));
        }
        if let Some(edit) = job.edit.as_mut()
            && let Err(error) = resolve_component_operation(&*self.components, root, edit)
        {
            return Err(Report::error(
                "edit",
                error.code,
                error.message,
                "已注册 Layer Edit Provider 必须先对相同 payload 通过 component test；直接 module:callable 入口仍可显式使用。",
            ));
        }
        Ok(EditPreparation::Ready { source_target })
    }
}
