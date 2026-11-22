use super::*;

pub(super) fn build_model_job(command: &ModelCommand) -> Job {
    match command {
        ModelCommand::Inspect { path, inputs } => Job {
            path: path.clone(),
            inputs: inputs.clone(),
            ..empty_job()
        },
        ModelCommand::Probe {
            path,
            inputs,
            call,
            contract,
            batch_fingerprint,
            local,
        } => Job {
            operation: "probe",
            path: path.clone(),
            inputs: inputs.clone(),
            call: call.clone(),
            contract: contract.clone(),
            batch_fingerprint: batch_fingerprint.clone(),
            local_probe: *local,
            ..empty_job()
        },
        ModelCommand::Mark {
            inputs,
            path,
            port,
            role,
            meaning,
            id,
        } => Job {
            operation: "semantic_mark",
            inputs: inputs.clone(),
            path: Some(path.clone()),
            edit: Some(serde_json::json!({
                "action": "add",
                "port": port,
                "role": role,
                "meaning": meaning,
                "id": id
            })),
            ..empty_job()
        },
        ModelCommand::Retag {
            inputs,
            id,
            path,
            port,
            role,
            meaning,
        } => Job {
            operation: "semantic_mark",
            inputs: inputs.clone(),
            path: Some(path.clone()),
            edit: Some(serde_json::json!({
                "action": "update",
                "id": id,
                "port": port,
                "role": role,
                "meaning": meaning
            })),
            ..empty_job()
        },
        ModelCommand::Marks { inputs } => Job {
            operation: "semantic_mark",
            inputs: inputs.clone(),
            edit: Some(serde_json::json!({
                "action": if inputs.model.is_some() || inputs.batch.is_some() {"audit"} else {"list"}
            })),
            ..empty_job()
        },
        ModelCommand::Unmark { id } => Job {
            operation: "semantic_mark",
            edit: Some(serde_json::json!({ "action": "remove", "id": id })),
            ..empty_job()
        },
        ModelCommand::Supervise { command } => match command {
            SuperviseCommand::Guide { inputs } => Job {
                operation: "semantic_supervise",
                inputs: inputs.clone(),
                edit: Some(serde_json::json!({ "action": "preview" })),
                ..empty_job()
            },
            SuperviseCommand::Preview { inputs } => Job {
                operation: "semantic_supervise",
                inputs: inputs.clone(),
                edit: Some(serde_json::json!({ "action": "preview" })),
                ..empty_job()
            },
            SuperviseCommand::Rule { command } => match command {
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
                } => Job {
                    operation: "semantic_supervise",
                    edit: Some(serde_json::json!({
                        "action":"rule_add",
                        "mark_id":mark,
                        "kind":match kind { SupervisionRuleKind::Finite => "finite", SupervisionRuleKind::Range => "range", SupervisionRuleKind::BatchAlignment => "batch-alignment", SupervisionRuleKind::CustomValidator => "custom-validator", SupervisionRuleKind::CustomPenalty => "custom-penalty" },
                        "mode":match mode { SupervisionRuleMode::Observe => "observe", SupervisionRuleMode::Require => "require", SupervisionRuleMode::Penalty => "penalty" },
                        "min":min,
                        "max":max,
                        "weight":weight,
                        "target_path":target_path,
                        "axis":axis,
                        "id":id,
                        "plugin_entrypoint":plugin,
                        "plugin_params":plugin_params
                    })),
                    ..empty_job()
                },
                SupervisionRuleCommand::List => Job {
                    operation: "semantic_supervise",
                    edit: Some(serde_json::json!({"action":"rule_list"})),
                    ..empty_job()
                },
                SupervisionRuleCommand::Remove { id, expected_rule } => Job {
                    operation: "semantic_supervise",
                    edit: Some(
                        serde_json::json!({"action":"rule_remove","id":id,"expected_rule":expected_rule}),
                    ),
                    ..empty_job()
                },
            },
        },
        ModelCommand::Verify { loss, inputs } => Job {
            operation: "verify",
            loss: Some(loss.clone()),
            inputs: inputs.clone(),
            ..empty_job()
        },
    }
}

pub(super) struct ModelInputs {
    pub parameters: Map<String, Value>,
    pub injections: Value,
}

impl<B: Backend> Application<B> {
    pub(super) fn resolve_model_inputs(
        &self,
        job: &Job,
        root: &Path,
    ) -> Result<ModelInputs, Report> {
        let parameters = parameters(&job.inputs.model_params)
            .map_err(|error| Report::error(job.operation, "PARAMETER_INVALID", error, ""))?;
        let injections = if let Some(saved) = &job.inputs.saved_injections {
            if !saved.is_object() {
                return Err(Report::error(
                    job.operation,
                    "MODEL_INJECTION_INVALID",
                    "已保存的模型接缝映射必须是 JSON 对象。",
                    "检查 .model-maven/project.json 或原训练运行记录。",
                ));
            }
            saved.clone()
        } else {
            Value::Object(
                model_injections(
                    &job.inputs.injections,
                    &job.inputs.injection_params,
                    &*self.components,
                    root,
                )
                .map_err(|error| {
                    Report::error(
                        job.operation,
                        "MODEL_INJECTION_INVALID",
                        error,
                        "每个注入工厂须返回 nn.Module，且根模型工厂必须显式接受同名关键字参数。",
                    )
                })?,
            )
        };
        Ok(ModelInputs {
            parameters,
            injections,
        })
    }

    pub(super) fn verify_model_inputs(
        &self,
        job: &Job,
        root: &Path,
        inputs: &ModelInputs,
        variant: Option<&Value>,
    ) -> Result<(), Report> {
        verify_saved_model_injections(&inputs.injections, &*self.components, root).map_err(
            |error| {
                model_injection_report(
                    job.operation,
                    error,
                    "注册组件参数必须匹配当前版本清单与 Schema。",
                )
            },
        )?;
        if let Some(variant) = variant {
            verify_saved_model_injections(
                &variant["origin"]["model_injections"],
                &*self.components,
                root,
            )
            .map_err(|error| {
                model_injection_report(
                    job.operation,
                    error,
                    "变体固定的注册组件参数必须匹配当前版本清单与 Schema。",
                )
            })?;
        }
        Ok(())
    }
}

fn model_injection_report(operation: &str, error: String, recovery: &str) -> Report {
    let code = if error.contains("已变化") {
        "COMPONENT_VERSION_CHANGED"
    } else {
        "MODEL_INJECTION_INVALID"
    };
    Report::error(operation, code, error, recovery)
}
