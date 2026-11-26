use super::*;

impl<B: Backend> Application<B> {
    pub(super) fn resolve_project_defaults(
        &self,
        cli: &Cli,
        root: &Path,
        job: &mut Job,
    ) -> Result<Option<Option<String>>, Report> {
        if job.repeat_of.is_some()
            || (job.parent_run_id.is_some() && job.migration_checkpoint_input.is_none())
        {
            for (key, reference) in [
                ("model", "--model"),
                ("batch", "--batch"),
                ("batch_binding", "--batch-binding"),
            ] {
                if job.repeat_overrides.iter().any(|item| item == key) {
                    let automatically_cleared = (key == "batch" && job.inputs.batch.is_none())
                        || (key == "batch_binding" && job.inputs.batch_binding.is_none());
                    set_source(
                        job,
                        key,
                        if automatically_cleared { crate::domain::operation::InputSourceKind::Generated }
                        else { crate::domain::operation::InputSourceKind::CommandLine },
                        if automatically_cleared { "batch context cleared after mode or factory change" } else { reference },
                        None,
                    );
                }
            }
            return Ok(None);
        }
        if job.inputs.model.is_none()
            && let Some(variant) = cli.variant.as_deref()
        {
            set_source(
                job,
                "model",
                crate::domain::operation::InputSourceKind::ModelVariant,
                variant,
                job.variant_snapshot
                    .as_ref()
                    .and_then(|value| value["revision"].as_str())
                    .map(str::to_owned),
            );
        }
        if job.inputs.no_project_model && job.repeat_of.is_none() && job.parent_run_id.is_none() {
            set_source(job, "model", crate::domain::operation::InputSourceKind::CommandLine,
                "--no-model", None);
        } else if job.inputs.model.is_some() {
            set_source(
                job,
                "model",
                crate::domain::operation::InputSourceKind::CommandLine,
                "--model",
                None,
            );
        }
        if job.inputs.batch.is_some() {
            set_source(
                job,
                "batch",
                crate::domain::operation::InputSourceKind::CommandLine,
                "--batch",
                None,
            );
        }
        if job.inputs.batch_binding.is_some() {
            set_source(
                job,
                "batch_binding",
                crate::domain::operation::InputSourceKind::CommandLine,
                "--batch-binding",
                None,
            );
        }
        let loop_owns_data = matches!(job.operation, "training_probe" | "train")
            && job.training_mode == Some("external-loop");
        // An external loop owns its data, but can still consume the project's
        // selected model. Absence of a project model remains valid for it.
        let needs_model = job.inputs.model.is_none() && cli.variant.is_none() && !job.inputs.no_project_model;
        let needs_batch = job.inputs.batch.is_none() && !loop_owns_data;
        let needs_binding = job.inputs.batch_binding.is_none() && needs_batch;
        let needs_injections = cli.variant.is_none() && !job.inputs.no_project_model
            && job.inputs.injections.is_empty()
            && job.inputs.injection_params.is_empty();
        let needs_model_params = cli.variant.is_none() && !job.inputs.no_project_model
            && job.repeat_of.is_none() && job.parent_run_id.is_none();
        let needs_config = needs_model || needs_batch || needs_binding || needs_injections
            || job.inputs.no_project_model;
        if !(needs_config || needs_model_params) {
            return Ok(None);
        }
        match self.project_config.load(root) {
            Ok(Some(config)) => {
                let revision = Some(crate::domain::input::content_revision(&config).map_err(|error| {
                    Report::error(job.operation, "PROJECT_CONFIG_REVISION_FAILED", error,
                        "无法核对项目配置；请检查配置后重新运行。")
                })?);
                let observed_revision = revision.clone();
                job.config_revision = revision.clone();
                if needs_model_params
                    && (needs_model || config["model"].as_str() == job.inputs.model.as_deref())
                {
                    if let Some(configured) = config["model_params"].as_object() {
                        let mut merged = configured.clone();
                        let overrides = parameters(&job.inputs.model_params).map_err(|error| {
                            Report::error(
                                job.operation,
                                "PARAMETER_INVALID",
                                error,
                                "模型参数使用 NAME=JSON 格式。",
                            )
                        })?;
                        let overridden = !overrides.is_empty();
                        merged.extend(overrides);
                        job.inputs.model_params = cli_params(merged);
                        if overridden {
                            job.input_sources.insert("model_params".into(), json!({"kind":"command_line", "reference":"--model-param", "revision":revision, "upstream":{"kind":"project_config", "reference":".model-maven/project.json#/model_params", "revision":revision}}));
                        } else {
                            set_source(
                                job,
                                "model_params",
                                crate::domain::operation::InputSourceKind::ProjectConfig,
                                ".model-maven/project.json#/model_params",
                                revision.clone(),
                            );
                        }
                    }
                }
                if needs_model {
                    job.inputs.model = resolve_project_value(
                        job,
                        "model",
                        None,
                        config["model"].as_str(),
                        ".model-maven/project.json#/model",
                        revision.clone(),
                    );
                }
                if needs_batch
                    && job.repeat_of.is_none()
                    && (job.parent_run_id.is_none() || job.migration_checkpoint_input.is_some())
                {
                    if let Some(configured) = config["batch_params"].as_object() {
                        let mut merged = configured.clone();
                        let overrides = parameters(&job.inputs.batch_params).map_err(|error| {
                            Report::error(
                                job.operation,
                                "PARAMETER_INVALID",
                                error,
                                "批次参数使用 NAME=JSON 格式。",
                            )
                        })?;
                        let overridden = !overrides.is_empty();
                        merged.extend(overrides);
                        job.inputs.batch_params = cli_params(merged);
                        if overridden {
                            job.input_sources.insert("batch_params".into(), json!({
                                "kind":"command_line", "reference":"--batch-param", "revision":revision,
                                "upstream":{"kind":"project_config", "reference":".model-maven/project.json#/batch_params", "revision":revision},
                                "note":"CLI keys override matching project parameters"
                            }));
                        } else {
                            set_source(
                                job,
                                "batch_params",
                                crate::domain::operation::InputSourceKind::ProjectConfig,
                                ".model-maven/project.json#/batch_params",
                                revision.clone(),
                            );
                        }
                    }
                }
                if needs_batch {
                    job.inputs.batch = resolve_project_value(
                        job,
                        "batch",
                        None,
                        config["batch"].as_str(),
                        ".model-maven/project.json#/batch",
                        revision.clone(),
                    );
                }
                if needs_binding {
                    let binding = match config["batch_binding"].as_str() {
                        Some("envelope") => Some(BatchBinding::Envelope),
                        Some("single") => Some(BatchBinding::Single),
                        Some("single-raw") => Some(BatchBinding::SingleRaw),
                        Some("args") => Some(BatchBinding::Args),
                        Some("kwargs") => Some(BatchBinding::Kwargs),
                        _ => None,
                    };
                    if binding.is_some() {
                        set_source(
                            job,
                            "batch_binding",
                            crate::domain::operation::InputSourceKind::ProjectConfig,
                            ".model-maven/project.json#/batch_binding",
                            revision.clone(),
                        );
                    }
                    job.inputs.batch_binding = binding;
                }
                if needs_injections
                    && (needs_model || config["model"].as_str() == job.inputs.model.as_deref())
                {
                    job.inputs.saved_injections = Some(saved_model_injections(&config));
                    set_source(
                        job,
                        "model_injections",
                        crate::domain::operation::InputSourceKind::ProjectConfig,
                        ".model-maven/project.json#/model_injections",
                        revision,
                    );
                }
                Ok(Some(observed_revision))
            }
            Ok(None) => Ok(Some(None)),
            Err(_) if !needs_config => Ok(None),
            Err(error) => Err(Report::error(
                job.operation,
                "PROJECT_CONFIG_INVALID",
                error,
                "先备份 .model-maven/project.json；当前 CLI 只读取 schema_version=1。未知版本请用兼容版本导出或迁移，显式 --model/--batch 可绕过已损坏的默认值。",
            )),
        }
    }

    pub(super) fn resolve_variant_edit_defaults(
        &self,
        root: &Path,
        job: &mut Job,
    ) -> Result<(), Report> {
        match self.project_config.load(root) {
            Ok(Some(config)) => {
                let revision = crate::domain::input::content_revision(&config).ok();
                job.config_revision = revision.clone();
                let use_project_batch_binding = job.inputs.batch.is_none();
                if job.inputs.model.is_none() {
                    if let Some(configured) = config["model_params"].as_object() {
                        let mut merged = configured.clone();
                        merged.extend(parameters(&job.inputs.model_params).map_err(|error| {
                            Report::error(
                                job.operation,
                                "PARAMETER_INVALID",
                                error,
                                "模型参数使用 NAME=JSON 格式。",
                            )
                        })?);
                        job.inputs.model_params = cli_params(merged);
                    }
                    job.inputs.model = resolve_project_value(
                        job,
                        "model",
                        None,
                        config["model"].as_str(),
                        ".model-maven/project.json#/model",
                        revision.clone(),
                    );
                }
                if job.inputs.batch.is_none()
                    && job.repeat_of.is_none()
                    && (job.parent_run_id.is_none() || job.migration_checkpoint_input.is_some())
                {
                    if let Some(configured) = config["batch_params"].as_object() {
                        let mut merged = configured.clone();
                        let overrides = parameters(&job.inputs.batch_params).map_err(|error| {
                            Report::error(
                                job.operation,
                                "PARAMETER_INVALID",
                                error,
                                "批次参数使用 NAME=JSON 格式。",
                            )
                        })?;
                        let overridden = !overrides.is_empty();
                        merged.extend(overrides);
                        job.inputs.batch_params = cli_params(merged);
                        if overridden {
                            job.input_sources.insert("batch_params".into(), json!({
                                "kind":"command_line", "reference":"--batch-param", "revision":revision,
                                "upstream":{"kind":"project_config", "reference":".model-maven/project.json#/batch_params", "revision":revision},
                                "note":"CLI keys override matching project parameters"
                            }));
                        } else {
                            set_source(
                                job,
                                "batch_params",
                                crate::domain::operation::InputSourceKind::ProjectConfig,
                                ".model-maven/project.json#/batch_params",
                                revision.clone(),
                            );
                        }
                    }
                }
                if job.inputs.batch.is_none() {
                    job.inputs.batch = resolve_project_value(
                        job,
                        "batch",
                        None,
                        config["batch"].as_str(),
                        ".model-maven/project.json#/batch",
                        revision.clone(),
                    );
                }
                if job.inputs.batch_binding.is_none() && use_project_batch_binding {
                    let binding = match config["batch_binding"].as_str() {
                        Some("envelope") => Some(BatchBinding::Envelope),
                        Some("single") => Some(BatchBinding::Single),
                        Some("single-raw") => Some(BatchBinding::SingleRaw),
                        Some("args") => Some(BatchBinding::Args),
                        Some("kwargs") => Some(BatchBinding::Kwargs),
                        _ => None,
                    };
                    if binding.is_some() {
                        set_source(
                            job,
                            "batch_binding",
                            crate::domain::operation::InputSourceKind::ProjectConfig,
                            ".model-maven/project.json#/batch_binding",
                            revision.clone(),
                        );
                    }
                    job.inputs.batch_binding = binding;
                }
                if job.inputs.injections.is_empty()
                    && job.inputs.injection_params.is_empty()
                    && config["model"].as_str() == job.inputs.model.as_deref()
                {
                    job.inputs.saved_injections = Some(saved_model_injections(&config));
                    set_source(
                        job,
                        "model_injections",
                        crate::domain::operation::InputSourceKind::ProjectConfig,
                        ".model-maven/project.json#/model_injections",
                        revision,
                    );
                }
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(error) => Err(Report::error(
                job.operation,
                "PROJECT_CONFIG_INVALID",
                error,
                "先备份 .model-maven/project.json；当前 CLI 只读取 schema_version=1。未知版本请用兼容版本导出或迁移；新变体不会覆盖项目默认接缝。",
            )),
        }
    }

    pub(super) fn resolve_replayed_input_sources(
        &self,
        root: &Path,
        job: &mut Job,
    ) -> Result<(), Report> {
        // Weight migration creates a new run with current inputs; its parent
        // supplies weights rather than a replayed batch or factory configuration.
        if job.migration_checkpoint_input.is_some() {
            return Ok(());
        }
        let Some(run_id) = job.repeat_of.as_deref().or(job.parent_run_id.as_deref()) else {
            return Ok(());
        };
        let prior = self.runs.show(root, run_id).map_err(|error| {
            Report::error(
                job.operation,
                "INPUT_SOURCE_UNAVAILABLE",
                error,
                "无法读取原运行的输入来源；没有启动新操作。",
            )
        })?;
        let prior_inputs = &prior["inputs"];
        job.config_revision = prior_inputs["config_revision"].as_str().map(str::to_owned);
        for key in [
            "model",
            "batch",
            "batch_binding",
            "batch_params",
            "model_injections",
        ] {
            if job.repeat_overrides.iter().any(|item| item == key) || job.input_sources.contains_key(key) {
                continue;
            }
            let upstream = prior_inputs["input_sources"][key].clone();
            if !upstream.is_object() && prior_inputs[key].is_null() {
                continue;
            }
            job.input_sources.insert(
                key.into(),
                json!({
                    "kind":"saved_run",
                    "reference":run_id,
                    "revision":job.config_revision,
                    "upstream":upstream
                }),
            );
        }
        Ok(())
    }

    pub(super) fn validate_training_inputs(&self, cli: &Cli, job: &Job) -> Result<(), Report> {
        if job.operation != "train" {
            return Ok(());
        }
        let selected_variant = cli.variant.is_some() || job.variant_snapshot.is_some();
        let external_loop = job.training_mode == Some("external-loop");
        if (job.inputs.model.is_none() && !selected_variant && !external_loop)
            || (!external_loop && job.inputs.batch.is_none())
        {
            return Err(Report::error(
                "train",
                "TRAIN_INPUTS_REQUIRED",
                "正式训练需要模型；managed/manual-step 还需要批次工厂。",
                "使用 --model module:factory，managed/manual-step 再添加 --batch module:batch_factory；也可先用 m-model init 保存默认入口。",
            ));
        }
        Ok(())
    }
}

fn resolve_project_value(
    job: &mut Job,
    key: &str,
    explicit: Option<&str>,
    configured: Option<&str>,
    reference: &str,
    revision: Option<String>,
) -> Option<String> {
    let binding = crate::domain::input::resolve(
        key,
        false,
        false,
        explicit.map(|value| crate::domain::input::Candidate {
            value: Value::String(value.into()),
            source: crate::domain::operation::InputSource {
                kind: crate::domain::operation::InputSourceKind::CommandLine,
                reference: Some(format!("--{key}")),
                revision: None,
            },
        }),
        None,
        None,
        configured.map(|value| crate::domain::input::Candidate {
            value: Value::String(value.into()),
            source: crate::domain::operation::InputSource {
                kind: crate::domain::operation::InputSourceKind::ProjectConfig,
                reference: Some(reference.into()),
                revision: revision.clone(),
            },
        }),
        None,
    );
    if binding.state == crate::domain::operation::FactState::Available {
        job.input_sources.insert(key.into(), json!(binding.source));
    }
    binding
        .value
        .and_then(|value| value.as_str().map(str::to_owned))
}

fn set_source(
    job: &mut Job,
    key: &str,
    kind: crate::domain::operation::InputSourceKind,
    reference: &str,
    revision: Option<String>,
) {
    job.input_sources.insert(
        key.into(),
        json!(crate::domain::operation::InputSource {
            kind,
            reference: Some(reference.into()),
            revision,
        }),
    );
}
