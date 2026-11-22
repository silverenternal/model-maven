use super::*;
use crate::domain::capability::{CapabilityAvailability, CapabilityCatalog};
use crate::domain::operation::{
    ActionAvailability, FactState, InputBinding, InputSource, InputSourceKind, OperationPlan,
    SuggestedAction,
};
use crate::domain::readiness::{ReadinessCapability, ReadinessInput};

struct PlanRequest {
    operation: &'static str,
    inputs: Inputs,
    require_model: bool,
    require_batch: bool,
    training_input: Option<(&'static str, Option<String>)>,
    capabilities: Vec<&'static str>,
    local_capabilities: Vec<ReadinessCapability>,
    facts: Vec<ReadinessInput>,
    conditions: Vec<String>,
    actions: Vec<SuggestedAction>,
    component: Option<ComponentReadinessRequest>,
    layer_selection: Option<String>,
    edit_target: Option<String>,
    replay: Option<(String, Value)>,
    layer_context: bool,
}

struct ComponentReadinessRequest {
    id: String,
    mode: ComponentReadinessMode,
    params: Vec<String>,
    batch: Option<String>,
}

#[derive(Clone, Copy)]
enum ComponentReadinessMode {
    Inspect,
    Check,
    Test,
    LayerProvider,
}

enum CapabilityQueryFailure {
    Interrupted,
    Unavailable(String),
}

impl<B: Backend> Application<B> {
    /// Build an input-only plan without starting a worker, changing state, or acquiring edit locks.
    pub fn plan_operation_inputs(
        &self,
        cli: &Cli,
        command: &Command,
    ) -> Result<Option<OperationPlan>, Report> {
        let Some(request) = self.readiness_request(command) else {
            return Ok(None);
        };
        let Ok(root) = std::env::current_dir().and_then(|path| path.canonicalize()) else {
            return Ok(None);
        };
        let config = match self.project_config.load(&root) {
            Ok(config) => config,
            Err(_) => return Ok(None),
        };
        let config_revision = config
            .as_ref()
            .and_then(|config| crate::domain::input::content_revision(config).ok());
        let variant_candidate = if let Some(name) = cli.variant.as_deref() {
            self.variants
                .load(&root, name)
                .ok()
                .flatten()
                .and_then(|variant| {
                    variant["origin"]["model"]
                        .as_str()
                        .or_else(|| variant["model"].as_str())
                        .map(|model| {
                            (
                                Value::String(model.to_owned()),
                                InputSource {
                                    kind: InputSourceKind::ModelVariant,
                                    reference: Some(name.to_owned()),
                                    revision: variant["revision"].as_str().map(str::to_owned),
                                },
                            )
                        })
                })
        } else if request.layer_context {
            let name = "working";
            self.variants
                .load(&root, name)
                .ok()
                .flatten()
                .and_then(|variant| {
                    variant["origin"]["model"]
                        .as_str()
                        .or_else(|| variant["model"].as_str())
                        .map(|model| {
                            (
                                Value::String(model.to_owned()),
                                InputSource {
                                    kind: InputSourceKind::ModelVariant,
                                    reference: Some(name.into()),
                                    revision: variant["revision"].as_str().map(str::to_owned),
                                },
                            )
                        })
                })
        } else {
            None
        };

        let model = if cli.variant.is_some()
            && request.inputs.model.is_none()
            && variant_candidate.is_none()
        {
            input_binding(
                "model",
                None,
                InputSource {
                    kind: InputSourceKind::ModelVariant,
                    reference: cli.variant.clone(),
                    revision: None,
                },
                request.require_model,
            )
        } else {
            self.resolve_plan_input(
                "model",
                request.inputs.model.as_deref(),
                request.replay.as_ref().and_then(|(id, record)| {
                    record["inputs"]["model"]
                        .as_str()
                        .or_else(|| record["inputs"]["variant"]["origin"]["model"].as_str())
                        .or_else(|| record["inputs"]["variant"]["model"].as_str())
                        .map(|value| {
                            (
                                id.as_str(),
                                value,
                                record["inputs"]["config_revision"]
                                    .as_str()
                                    .map(str::to_owned),
                            )
                        })
                }),
                config.as_ref().and_then(|config| config["model"].as_str()),
                variant_candidate,
                &config_revision,
                request.require_model,
            )
        };
        let batch = self.resolve_plan_input(
            "batch",
            request.inputs.batch.as_deref(),
            request.replay.as_ref().and_then(|(id, record)| {
                record["inputs"]["batch"].as_str().map(|value| {
                    (
                        id.as_str(),
                        value,
                        record["inputs"]["config_revision"]
                            .as_str()
                            .map(str::to_owned),
                    )
                })
            }),
            config.as_ref().and_then(|config| config["batch"].as_str()),
            None,
            &config_revision,
            request.require_batch,
        );

        let mut bindings = vec![model, batch];
        if let Some((key, explicit)) = request.training_input.clone() {
            let saved = request.replay.as_ref().and_then(|(_, record)| {
                record["inputs"][match key {
                    "training_step" => "training_step",
                    _ => "training_loop",
                }]
                .as_str()
            });
            let (value, source) = if let Some(value) = explicit {
                (
                    Some(Value::String(value)),
                    InputSource {
                        kind: InputSourceKind::CommandLine,
                        reference: Some(if key == "training_step" {
                            "--step".into()
                        } else {
                            "--loop".into()
                        }),
                        revision: None,
                    },
                )
            } else if let Some((run_id, value)) = request
                .replay
                .as_ref()
                .and_then(|(run_id, _)| saved.map(|value| (run_id, value)))
            {
                (
                    Some(Value::String(value.into())),
                    InputSource {
                        kind: InputSourceKind::SavedRun,
                        reference: Some(run_id.clone()),
                        revision: request.replay.as_ref().and_then(|(_, record)| {
                            record["inputs"]["config_revision"]
                                .as_str()
                                .map(str::to_owned)
                        }),
                    },
                )
            } else {
                (None, unknown_source())
            };
            bindings.push(input_binding(key, value, source, true));
        }

        let mut actions = request.actions.clone();
        if bindings.iter().any(|input| {
            input.key == "model" && input.required && input.state == FactState::Missing
        }) {
            actions.push(SuggestedAction {
                id: if cli.variant.is_some() {
                    "select_model_variant".into()
                } else {
                    "configure_model".into()
                },
                label: if cli.variant.is_some() {
                    "所选模型变体不可用；运行 m-model variant list 检查可用变体".into()
                } else {
                    "提供模型工厂入口：--model module:factory，或先运行 m-model init".into()
                },
                availability: ActionAvailability::Available,
                reason: None,
            });
        }
        if bindings.iter().any(|input| {
            input.key == "batch" && input.required && input.state == FactState::Missing
        }) {
            actions.push(SuggestedAction {
                id: "configure_batch".into(),
                label: "提供示例批次工厂：--batch module:batch_factory".into(),
                availability: ActionAvailability::Available,
                reason: None,
            });
        }
        if let Some((key, _)) = request.training_input.as_ref()
            && bindings
                .iter()
                .any(|input| input.key == *key && input.state == FactState::Missing)
        {
            actions.push(SuggestedAction {
                id: format!("configure_{key}"),
                label: if *key == "training_step" {
                    "提供训练步骤入口：--step module:training_step".into()
                } else {
                    "提供用户训练循环：--training-mode external-loop --loop module:run_loop".into()
                },
                availability: ActionAvailability::Available,
                reason: None,
            });
        }

        let mut facts = request.facts;
        let mut local_capabilities = request.local_capabilities;
        let mut conditions = request.conditions;
        if let Some(component) = request.component.as_ref() {
            let component_source = InputSource {
                kind: InputSourceKind::CommandLine,
                reference: Some("component id".into()),
                revision: None,
            };
            facts.push(ReadinessInput {
                key: "component_descriptor".into(),
                state: FactState::Unknown,
                required: true,
                reason: Some("尚未检查项目组件注册表中的描述。".into()),
                source: Some(component_source),
            });
            match self.components.inspect(&root, &component.id) {
                Ok(descriptor) => {
                    if let Some(fact) = facts
                        .iter_mut()
                        .find(|fact| fact.key == "component_descriptor")
                    {
                        fact.state = FactState::Available;
                        fact.reason = None;
                    }
                    local_capabilities.push(ReadinessCapability {
                        id: "component.static_metadata".into(),
                        state: FactState::Available,
                        required: true,
                        reason: Some("组件清单可静态检查；未导入或执行用户 Python 代码。".into()),
                        provider: Some("component-registry".into()),
                    });
                    if matches!(
                        component.mode,
                        ComponentReadinessMode::Check
                            | ComponentReadinessMode::Test
                            | ComponentReadinessMode::LayerProvider
                    ) {
                        match self.components.resolve_dependencies(&root, &component.id) {
                            Ok(_) => facts.push(ReadinessInput {
                                key: "component_dependencies".into(),
                                state: FactState::Available,
                                required: true,
                                reason: None,
                                source: None,
                            }),
                            Err(error) => facts.push(ReadinessInput {
                                key: "component_dependencies".into(),
                                state: FactState::Unsupported,
                                required: true,
                                reason: Some(error),
                                source: None,
                            }),
                        }
                    }
                    if matches!(component.mode, ComponentReadinessMode::Test) {
                        let kind = descriptor["manifest"]["kind"].as_str().unwrap_or("");
                        let fixture = descriptor["manifest"]["conformance_fixture"].as_str();
                        let capabilities = &descriptor["manifest"]["capabilities"];
                        let fixture_capabilities: &[&str] = match kind {
                            "layer-edit-provider" => &["layer_edit"],
                            "transform" | "dataset" | "loss" | "metric" => &[kind],
                            "trainer" => &["events", "stop"],
                            _ => &[],
                        };
                        let fixture_required = fixture_capabilities.iter().any(|name| {
                            let declaration = &capabilities[*name];
                            let state = declaration["state"]
                                .as_str()
                                .or_else(|| declaration.as_str())
                                .unwrap_or("unknown");
                            matches!(state, "supported" | "conditional")
                        });
                        if fixture_required && fixture.is_none() {
                            facts.push(ReadinessInput {
                                key: "component_conformance_fixture".into(),
                                state: FactState::Missing,
                                required: true,
                                reason: Some(format!("{kind} 声明 supported/conditional 时必须提供 conformance_fixture 才能运行测试。")),
                                source: None,
                            });
                            actions.push(SuggestedAction {
                                id: "configure_component_fixture".into(),
                                label: format!(
                                    "为 {kind} 组件添加 conformance_fixture 并重新登记组件"
                                ),
                                availability: ActionAvailability::Available,
                                reason: None,
                            });
                        } else if !matches!(
                            kind,
                            "layer"
                                | "layer-edit-provider"
                                | "transform"
                                | "dataset"
                                | "loss"
                                | "metric"
                                | "trainer"
                        ) {
                            facts.push(ReadinessInput {
                                key: "component_test_kind".into(),
                                state: FactState::Unsupported,
                                required: true,
                                reason: Some(format!("component test 暂不支持 kind={kind}。")),
                                source: None,
                            });
                        } else {
                            facts.push(ReadinessInput {
                                key: "component_test_kind".into(),
                                state: FactState::Available,
                                required: true,
                                reason: None,
                                source: None,
                            });
                        }
                        let forward_probe_declared = matches!(
                            descriptor["manifest"]["capabilities"]["forward_probe"]["state"]
                                .as_str(),
                            Some("supported" | "conditional")
                        );
                        if kind == "layer"
                            && forward_probe_declared
                            && fixture.is_none()
                            && component.batch.is_none()
                        {
                            conditions.push("该 Layer 未配置 conformance_fixture；前向探测需要显式传入 --batch，当前测试只验证构造和声明能力。".into());
                            actions.push(SuggestedAction {
                                id: "provide_component_test_batch".into(),
                                label: "可选：传入 --batch module:factory 增加真实前向探测".into(),
                                availability: ActionAvailability::Available,
                                reason: None,
                            });
                        }
                    }
                    if matches!(component.mode, ComponentReadinessMode::LayerProvider) {
                        let manifest = &descriptor["manifest"];
                        let kind = manifest["kind"].as_str().unwrap_or("");
                        let capability = &manifest["capabilities"]["layer_edit"];
                        let capability_state = capability
                            .as_str()
                            .or_else(|| capability["state"].as_str())
                            .unwrap_or("unknown");
                        facts.push(ReadinessInput {
                            key: "layer_provider_kind".into(),
                            state: if kind == "layer-edit-provider" {
                                FactState::Available
                            } else {
                                FactState::Unsupported
                            },
                            required: true,
                            reason: (kind != "layer-edit-provider").then(|| {
                                format!(
                                    "组件 {} 的 kind 为 {kind}，不是 layer-edit-provider。",
                                    component.id
                                )
                            }),
                            source: None,
                        });
                        facts.push(ReadinessInput {
                            key: "layer_provider_capability".into(),
                            state: match capability_state {
                                "supported" => FactState::Available,
                                "conditional" | "unsupported" => FactState::Unsupported,
                                _ => FactState::Unknown,
                            },
                            required: true,
                            reason: (capability_state != "supported").then(|| {
                                format!(
                                    "{capability_state}: {}",
                                    capability["reason"]
                                        .as_str()
                                        .unwrap_or("Provider 必须声明 supported 状态。")
                                )
                            }),
                            source: None,
                        });
                        let fixture_available = manifest["conformance_fixture"].as_str().is_some();
                        facts.push(ReadinessInput {
                            key: "layer_provider_fixture".into(),
                            state: if fixture_available {
                                FactState::Available
                            } else {
                                FactState::Missing
                            },
                            required: true,
                            reason: (!fixture_available).then(|| {
                                "Provider 缺少 conformance_fixture，不能启用扩展编辑。".into()
                            }),
                            source: None,
                        });
                        if !fixture_available {
                            actions.push(SuggestedAction {
                                id: "configure_layer_provider_fixture".into(),
                                label: "为 Provider 添加 conformance_fixture 并重新登记".into(),
                                availability: ActionAvailability::Available,
                                reason: None,
                            });
                        }
                        let verification = manifest["component_id"]
                            .as_str()
                            .zip(manifest["component_version"].as_str())
                            .and_then(|(id, version)| {
                                self.components
                                    .component_verification(&root, id, version)
                                    .ok()
                                    .flatten()
                            });
                        let resolved_payload = super::parameters(&component.params)
                            .ok()
                            .and_then(|params| {
                                self.components
                                    .resolve_layer_edit_provider(&root, &component.id, &params)
                                    .ok()
                            })
                            .map(|resolved| resolved["resolved_payload"].clone());
                        let payload_digest = resolved_payload.as_ref().and_then(|payload| {
                            crate::domain::input::worker_payload_digest(payload).ok()
                        });
                        let payload_verified = verification.as_ref().is_some_and(|record| {
                            record["verifications"].as_array().is_some_and(|items| {
                                items.iter().any(|item| {
                                    item["status"] == "verified"
                                        && item["capability"] == "layer_edit"
                                        && item["coverage"] == "complete_for_fixture_expectations"
                                        && item["suite_version"] == 2
                                        && payload_digest.as_ref().is_some_and(|digest| {
                                            item["payload_sha256"] == digest.as_str()
                                        })
                                })
                            })
                        });
                        let valid_manifest_record = verification.as_ref().is_some_and(|record| {
                            record["manifest_sha256"] == descriptor["manifest_sha256"]
                                && record["verifications"]
                                    .as_array()
                                    .is_some_and(|items| !items.is_empty())
                        });
                        let verification_state = match verification.as_ref() {
                            Some(record)
                                if record["manifest_sha256"] != descriptor["manifest_sha256"] =>
                            {
                                FactState::Unsupported
                            }
                            Some(_) if !valid_manifest_record => FactState::Unsupported,
                            Some(_) if payload_verified => FactState::Conditional,
                            Some(_) if payload_digest.is_none() => FactState::Conditional,
                            Some(_) => FactState::Missing,
                            None => FactState::Missing,
                        };
                        facts.push(ReadinessInput {
                            key: "layer_provider_verification".into(),
                            state: verification_state,
                            required: true,
                            reason: match verification_state {
                                FactState::Conditional => Some("符合性记录覆盖当前 payload（浮点 payload 将交由 Worker 核对）；执行时仍会复核源码指纹与 Python/PyTorch 版本。".into()),
                                FactState::Missing => Some("没有与当前 payload 匹配的 Provider 验证记录，需要先运行 component test。".into()),
                                FactState::Unsupported => Some("Provider 验证记录与当前清单不一致或不完整。".into()),
                                _ => None,
                            },
                            source: None,
                        });
                        if verification_state == FactState::Missing {
                            actions.push(SuggestedAction {
                                id: "test_layer_provider".into(),
                                label: format!(
                                    "运行 m-model component test {} 完成 Provider 符合性验证",
                                    component.id
                                ),
                                availability: ActionAvailability::Available,
                                reason: None,
                            });
                        }
                    }
                }
                Err(error) => {
                    if let Some(fact) = facts
                        .iter_mut()
                        .find(|fact| fact.key == "component_descriptor")
                    {
                        // Preserve the established component command diagnostic for unresolved IDs.
                        fact.reason = Some(error);
                    }
                }
            }
            bindings.push(input_binding(
                "component",
                Some(Value::String(component.id.clone())),
                InputSource {
                    kind: InputSourceKind::CommandLine,
                    reference: Some("component id".into()),
                    revision: None,
                },
                true,
            ));
            if matches!(
                component.mode,
                ComponentReadinessMode::Test | ComponentReadinessMode::LayerProvider
            ) {
                match super::parameters(&component.params) {
                    Err(error) => facts.push(ReadinessInput {
                        key: "component_parameters".into(),
                        state: FactState::Unsupported,
                        required: true,
                        reason: Some(error),
                        source: None,
                    }),
                    Ok(params)
                        if facts.iter().any(|fact| {
                            fact.key == "component_descriptor" && fact.state == FactState::Available
                        }) =>
                    {
                        if let Some(root) = std::env::current_dir()
                            .ok()
                            .and_then(|path| path.canonicalize().ok())
                        {
                            match self.components.inspect(&root, &component.id).and_then(
                                |descriptor| {
                                    let kind =
                                        descriptor["manifest"]["kind"].as_str().unwrap_or("");
                                    let missing_fixture = facts.iter().any(|fact| {
                                        fact.key == "component_conformance_fixture"
                                            && fact.state == FactState::Missing
                                    });
                                    match kind {
                                        "layer" => self.components.resolve_layer(
                                            &root,
                                            &component.id,
                                            &params,
                                        ),
                                        "layer-edit-provider" if missing_fixture => Ok(Value::Null),
                                        "layer-edit-provider" => {
                                            self.components.resolve_layer_edit_provider(
                                                &root,
                                                &component.id,
                                                &params,
                                            )
                                        }
                                        _ => Ok(Value::Null),
                                    }
                                },
                            ) {
                                Ok(_) => facts.push(ReadinessInput {
                                    key: "component_parameters".into(),
                                    state: FactState::Available,
                                    required: true,
                                    reason: None,
                                    source: None,
                                }),
                                Err(error) => facts.push(ReadinessInput {
                                    key: "component_parameters".into(),
                                    state: FactState::Unsupported,
                                    required: true,
                                    reason: Some(error),
                                    source: None,
                                }),
                            }
                        }
                    }
                    Ok(_) => {}
                }
            }
        }

        if request.operation == "plan_layer_edit" || request.operation == "apply_layer_edit" {
            let selection = request.layer_selection.clone();
            facts.push(ReadinessInput {
                key: "layer_selection".into(),
                state: if selection.is_some() {
                    FactState::Available
                } else {
                    FactState::Missing
                },
                required: true,
                reason: selection
                    .is_none()
                    .then(|| "Layer 编辑必须明确指定目标路径、父容器或 Provider。".into()),
                source: Some(InputSource {
                    kind: InputSourceKind::CommandLine,
                    reference: selection.clone(),
                    revision: None,
                }),
            });
            let target = request.edit_target.clone();
            facts.push(ReadinessInput {
                key: "edit_target".into(),
                state: if target.is_some() {
                    FactState::Available
                } else {
                    FactState::Missing
                },
                required: true,
                reason: target
                    .is_none()
                    .then(|| "Layer 编辑必须选择 variant 或 source 持久化目标。".into()),
                source: Some(InputSource {
                    kind: InputSourceKind::CommandLine,
                    reference: target,
                    revision: None,
                }),
            });
            if request.edit_target.is_none() {
                actions.push(SuggestedAction {
                    id: "select_edit_target".into(),
                    label: "指定 --target variant 或 --target source".into(),
                    availability: ActionAvailability::Available,
                    reason: None,
                });
            }
        }

        let revision = config_revision.clone().or_else(|| {
            request.replay.as_ref().and_then(|(_, record)| {
                record["inputs"]["config_revision"]
                    .as_str()
                    .map(str::to_owned)
            })
        });
        let required_inputs_missing = bindings
            .iter()
            .any(|input| input.required && input.state != FactState::Available)
            || facts
                .iter()
                .any(|fact| fact.required && fact.state != FactState::Available);
        let catalog = if request.capabilities.is_empty() {
            None
        } else if required_inputs_missing {
            let cached = self.runtime_capabilities.borrow().clone();
            if cached.is_none() {
                conditions.push(
                    "补齐必需输入后再检测当前 Python/Worker 能力；先解决静态前置条件。".into(),
                );
            }
            cached
        } else {
            match self.runtime_capability_catalog(cli) {
                Ok(catalog) => Some(catalog),
                Err(CapabilityQueryFailure::Interrupted) => {
                    return Err(Report::error(
                        "operation_readiness",
                        "USER_INTERRUPTED",
                        "检查当前 Python/Worker 能力时收到中断信号；目标操作已取消。",
                        "重新执行命令即可重试。",
                    ));
                }
                Err(CapabilityQueryFailure::Unavailable(error)) => {
                    conditions.push(format!("当前 Worker 能力未能确认：{error}"));
                    None
                }
            }
        };
        let mut capabilities = capability_states(
            &request.capabilities,
            catalog.as_ref(),
            !required_inputs_missing,
        );
        capabilities.extend(local_capabilities);
        Ok(crate::domain::readiness::plan(
            format!(
                "{}:{}",
                request.operation,
                revision.as_deref().unwrap_or("unversioned")
            ),
            request.operation,
            bindings,
            facts,
            capabilities,
            conditions,
            actions,
            revision,
        )
        .ok())
    }

    fn runtime_capability_catalog(
        &self,
        cli: &Cli,
    ) -> Result<CapabilityCatalog, CapabilityQueryFailure> {
        if let Some(catalog) = self.runtime_capabilities.borrow().as_ref() {
            return Ok(catalog.clone());
        }
        let report = self.backend.execute(&serde_json::json!({
            "protocol_version": crate::domain::PROTOCOL_VERSION,
            "operation": "doctor",
            "runtime": {
                "timeout_seconds": cli.timeout.unwrap_or(30)
            }
        }));
        if !report.success() {
            let diagnostic = report.diagnostics.first();
            if diagnostic.is_some_and(|diagnostic| diagnostic.code == "USER_INTERRUPTED") {
                return Err(CapabilityQueryFailure::Interrupted);
            }
            return Err(CapabilityQueryFailure::Unavailable(
                diagnostic
                    .map(|diagnostic| format!("{}: {}", diagnostic.code, diagnostic.message))
                    .unwrap_or_else(|| "doctor 没有返回成功的能力目录".into()),
            ));
        }
        let catalog = CapabilityCatalog::from_doctor_data(&report.data)
            .map_err(CapabilityQueryFailure::Unavailable)?;
        *self.runtime_capabilities.borrow_mut() = Some(catalog.clone());
        Ok(catalog)
    }

    fn readiness_request(&self, command: &Command) -> Option<PlanRequest> {
        match command {
            Command::Model { command } => {
                if matches!(command, ModelCommand::Unmark { .. })
                    || matches!(command, ModelCommand::Marks { inputs } if inputs.model.is_none() && inputs.batch.is_none())
                    || matches!(
                        command,
                        ModelCommand::Supervise {
                            command: SuperviseCommand::Rule { .. }
                        }
                    )
                {
                    return None;
                }
                let job = super::model::build_model_job(command);
                let (operation, require_batch) = match command {
                    ModelCommand::Inspect { .. } => ("inspect_model", false),
                    ModelCommand::Probe { .. } => ("forward_probe", true),
                    ModelCommand::Mark { .. } => ("forward_probe", true),
                    ModelCommand::Retag { .. } => ("forward_probe", true),
                    ModelCommand::Marks { .. } => ("forward_probe", true),
                    ModelCommand::Unmark { .. } => unreachable!(),
                    ModelCommand::Supervise {
                        command: SuperviseCommand::Guide { .. },
                    }
                    | ModelCommand::Supervise {
                        command: SuperviseCommand::Preview { .. },
                    } => ("forward_probe", true),
                    ModelCommand::Supervise {
                        command: SuperviseCommand::Rule { .. },
                    } => unreachable!(),
                    ModelCommand::Verify { .. } => ("loss_probe", true),
                };
                Some(PlanRequest {
                    operation,
                    inputs: job.inputs,
                    require_model: true,
                    require_batch,
                    training_input: None,
                    capabilities: match command {
                        ModelCommand::Inspect { .. } => vec!["model.inspect"],
                        ModelCommand::Probe { .. } => {
                            vec!["model.forward_probe", "data.dataset_probe"]
                        }
                        ModelCommand::Mark { .. } => {
                            vec![
                                "model.forward_probe",
                                "data.dataset_probe",
                                "model.semantic_marks",
                            ]
                        }
                        ModelCommand::Retag { .. } => {
                            vec![
                                "model.forward_probe",
                                "data.dataset_probe",
                                "model.semantic_marks",
                            ]
                        }
                        ModelCommand::Marks { .. } => vec![
                            "model.forward_probe",
                            "data.dataset_probe",
                            "model.semantic_marks",
                        ],
                        ModelCommand::Unmark { .. } => unreachable!(),
                        ModelCommand::Supervise {
                            command: SuperviseCommand::Guide { .. },
                        }
                        | ModelCommand::Supervise {
                            command: SuperviseCommand::Preview { .. },
                        } => vec![
                            "model.forward_probe",
                            "data.dataset_probe",
                            "model.semantic_marks",
                            "training.supervision_preview",
                        ],
                        ModelCommand::Supervise {
                            command: SuperviseCommand::Rule { .. },
                        } => unreachable!(),
                        ModelCommand::Verify { .. } => vec!["model.loss_probe"],
                    },
                    replay: None,
                    layer_context: false,
                    local_capabilities: vec![],
                    facts: vec![],
                    conditions: vec![],
                    actions: vec![],
                    component: None,
                    layer_selection: None,
                    edit_target: None,
                })
            }
            Command::Try {
                inputs,
                repeat,
                training_mode,
                apply_supervision,
                step,
                training_loop,
                ..
            } => {
                let replay = repeat
                    .as_deref()
                    .and_then(|id| self.readiness_training_record(id, "training_probe"));
                if repeat.is_some() && replay.is_none() {
                    return None;
                }
                let external = training_mode.as_ref() == Some(&TrainingMode::ExternalLoop)
                    || training_loop.is_some()
                    || replay.as_ref().is_some_and(|(_, record)| {
                        record["inputs"]["training_mode"] == "external-loop"
                    });
                Some(PlanRequest {
                    operation: "training_probe",
                    inputs: inputs.clone().into(),
                    require_model: !external,
                    require_batch: !external,
                    training_input: Some((
                        if external {
                            "training_loop"
                        } else {
                            "training_step"
                        },
                        if external {
                            training_loop.clone()
                        } else {
                            step.clone()
                        },
                    )),
                    capabilities: {
                        let mut required =
                            vec![training_capability(external, training_mode.as_ref())];
                        if *apply_supervision && !external {
                            required.push("training.supervision_apply");
                        }
                        required
                    },
                    replay,
                    layer_context: false,
                    local_capabilities: vec![],
                    facts: vec![],
                    conditions: vec![],
                    actions: vec![],
                    component: None,
                    layer_selection: None,
                    edit_target: None,
                })
            }
            Command::Train {
                inputs,
                resume,
                training_mode,
                apply_supervision,
                step,
                training_loop,
                ..
            } => {
                let replay = resume
                    .as_deref()
                    .and_then(|id| self.readiness_training_record(id, "train"));
                if resume.is_some() && replay.is_none() {
                    return None;
                }
                let external = training_mode.as_ref() == Some(&TrainingMode::ExternalLoop)
                    || training_loop.is_some()
                    || replay.as_ref().is_some_and(|(_, record)| {
                        record["inputs"]["training_mode"] == "external-loop"
                    });
                Some(PlanRequest {
                    operation: "train",
                    inputs: inputs.clone().into(),
                    require_model: !external,
                    require_batch: !external,
                    training_input: Some((
                        if external {
                            "training_loop"
                        } else {
                            "training_step"
                        },
                        if external {
                            training_loop.clone()
                        } else {
                            step.clone()
                        },
                    )),
                    capabilities: {
                        let mut required =
                            vec![training_capability(external, training_mode.as_ref())];
                        if resume.is_some() {
                            required.push("training.checkpoint_resume");
                        }
                        if *apply_supervision && !external {
                            required.push("training.supervision_apply");
                        }
                        if let Command::Train { detach: true, .. } = command {
                            required.push("training.detached");
                        }
                        required
                    },
                    replay,
                    layer_context: false,
                    local_capabilities: vec![],
                    facts: vec![],
                    conditions: vec![],
                    actions: vec![],
                    component: None,
                    layer_selection: None,
                    edit_target: None,
                })
            }
            Command::Layer { command } => {
                let job = super::edit::build_layer_job(command).ok()?;
                let is_edit = job.edit.is_some();
                let is_history = matches!(
                    command,
                    LayerCommand::Undo { .. } | LayerCommand::Redo { .. }
                );
                let require_model = !is_history;
                let dry_run = job.dry_run;
                let (layer_selection, edit_target) = layer_edit_context(command, &job);
                let layer_provider = match command {
                    LayerCommand::Custom { provider, .. } if !provider.contains(':') => {
                        let params = job
                            .edit
                            .as_ref()
                            .and_then(|edit| edit["payload"].as_object())
                            .into_iter()
                            .flat_map(|payload| payload.iter())
                            .map(|(key, value)| format!("{key}={value}"))
                            .collect();
                        Some(ComponentReadinessRequest {
                            id: provider.clone(),
                            mode: ComponentReadinessMode::LayerProvider,
                            params,
                            batch: None,
                        })
                    }
                    _ => None,
                };
                Some(PlanRequest {
                    operation: if is_history {
                        "layer_history"
                    } else if is_edit {
                        if dry_run {
                            "plan_layer_edit"
                        } else {
                            "apply_layer_edit"
                        }
                    } else {
                        "layer_query"
                    },
                    inputs: job.inputs,
                    require_model,
                    require_batch: job.require_forward,
                    training_input: None,
                    capabilities: if is_history {
                        vec![]
                    } else if is_edit {
                        vec!["model.layer_edit"]
                    } else {
                        vec!["model.inspect"]
                    },
                    replay: None,
                    layer_context: true,
                    local_capabilities: vec![],
                    facts: vec![],
                    conditions: vec![],
                    actions: vec![],
                    component: layer_provider,
                    layer_selection: (!is_history).then_some(layer_selection).flatten(),
                    edit_target: (!is_history).then_some(edit_target).flatten(),
                })
            }
            Command::Component { command } => match command {
                ComponentCommand::Show { id } => Some(PlanRequest {
                    operation: "component_inspect",
                    inputs: Inputs::default(),
                    require_model: false,
                    require_batch: false,
                    training_input: None,
                    capabilities: vec![],
                    local_capabilities: vec![],
                    facts: vec![],
                    conditions: vec![],
                    actions: vec![],
                    component: Some(ComponentReadinessRequest {
                        id: id.clone(),
                        mode: ComponentReadinessMode::Inspect,
                        params: vec![],
                        batch: None,
                    }),
                    layer_selection: None,
                    edit_target: None,
                    replay: None,
                    layer_context: false,
                }),
                ComponentCommand::Check { id } => Some(PlanRequest {
                    operation: "component_check",
                    inputs: Inputs::default(),
                    require_model: false,
                    require_batch: false,
                    training_input: None,
                    capabilities: vec![],
                    local_capabilities: vec![],
                    facts: vec![],
                    conditions: vec![],
                    actions: vec![],
                    component: Some(ComponentReadinessRequest {
                        id: id.clone(),
                        mode: ComponentReadinessMode::Check,
                        params: vec![],
                        batch: None,
                    }),
                    layer_selection: None,
                    edit_target: None,
                    replay: None,
                    layer_context: false,
                }),
                ComponentCommand::Test {
                    id, params, batch, ..
                } => Some(PlanRequest {
                    operation: "component_test",
                    inputs: Inputs {
                        batch: batch.clone(),
                        ..Inputs::default()
                    },
                    require_model: false,
                    require_batch: false,
                    training_input: None,
                    capabilities: vec!["extensions.component_conformance"],
                    local_capabilities: vec![],
                    facts: vec![],
                    conditions: vec![],
                    actions: vec![],
                    component: Some(ComponentReadinessRequest {
                        id: id.clone(),
                        mode: ComponentReadinessMode::Test,
                        params: params.clone(),
                        batch: batch.clone(),
                    }),
                    layer_selection: None,
                    edit_target: None,
                    replay: None,
                    layer_context: false,
                }),
                _ => None,
            },
            _ => None,
        }
    }

    fn readiness_training_record(
        &self,
        requested: &str,
        operation: &str,
    ) -> Option<(String, Value)> {
        let root = std::env::current_dir().ok()?.canonicalize().ok()?;
        let id = if requested == "latest" {
            self.runs
                .list(&root)
                .ok()?
                .into_iter()
                .find(|run| {
                    run["operation"] == operation
                        && (operation == "training_probe"
                            || matches!(
                                run["status"].as_str(),
                                Some("completed" | "failed" | "interrupted")
                            ))
                })
                .and_then(|run| run["id"].as_str().map(str::to_owned))?
        } else {
            requested.to_owned()
        };
        let record = self.runs.show(&root, &id).ok()?;
        (record["record"]["operation"] == operation || record["inputs"]["operation"] == operation)
            .then_some((id, record))
    }

    // Resolution inputs preserve their distinct precedence/provenance sources.
    #[allow(clippy::too_many_arguments)]
    fn resolve_plan_input(
        &self,
        key: &str,
        explicit: Option<&str>,
        replayed: Option<(&str, &str, Option<String>)>,
        configured: Option<&str>,
        variant: Option<(Value, InputSource)>,
        config_revision: &Option<String>,
        required: bool,
    ) -> InputBinding {
        if let Some(value) = explicit {
            return input_binding(
                key,
                Some(Value::String(value.to_owned())),
                InputSource {
                    kind: InputSourceKind::CommandLine,
                    reference: Some(format!("--{key}")),
                    revision: None,
                },
                required,
            );
        }
        if let Some((run_id, value, revision)) = replayed {
            return input_binding(
                key,
                Some(Value::String(value.to_owned())),
                InputSource {
                    kind: InputSourceKind::SavedRun,
                    reference: Some(run_id.to_owned()),
                    revision,
                },
                required,
            );
        }
        if let Some((value, source)) = variant {
            return input_binding(key, Some(value), source, required);
        }
        if let Some(value) = configured {
            return input_binding(
                key,
                Some(Value::String(value.to_owned())),
                InputSource {
                    kind: InputSourceKind::ProjectConfig,
                    reference: Some(format!(".model-maven/project.json#/{key}")),
                    revision: config_revision.clone(),
                },
                required,
            );
        }
        input_binding(key, None, unknown_source(), required)
    }
}

fn training_capability(external: bool, training_mode: Option<&TrainingMode>) -> &'static str {
    match (external, training_mode) {
        (true, _) | (_, Some(TrainingMode::ExternalLoop)) => "training.external_loop",
        (_, Some(TrainingMode::ManualStep)) => "training.manual_step",
        _ => "training.managed_step",
    }
}

fn layer_edit_context(command: &LayerCommand, job: &Job) -> (Option<String>, Option<String>) {
    let selection = match command {
        LayerCommand::MigrateStarter { .. } => Some("model starter".into()),
        LayerCommand::Add { parent, .. } => Some(parent.clone()),
        LayerCommand::Set { path, .. }
        | LayerCommand::Replace { path, .. }
        | LayerCommand::Remove { path, .. } => Some(path.clone()),
        LayerCommand::Apply { operations, .. } => operations.first().and_then(|operation| {
            let operation = serde_json::from_str::<Value>(operation).ok()?;
            operation["path"]
                .as_str()
                .or_else(|| operation["parent"].as_str())
                .or_else(|| operation["provider"].as_str())
                .map(str::to_owned)
        }),
        LayerCommand::Custom { provider, .. } => Some(provider.clone()),
        _ => None,
    };
    let target = job
        .edit
        .as_ref()
        .and_then(|edit| edit["target"].as_str())
        .map(str::to_owned);
    (selection, target)
}

fn capability_states(
    required: &[&'static str],
    catalog: Option<&CapabilityCatalog>,
    is_required: bool,
) -> Vec<ReadinessCapability> {
    required
        .iter()
        .map(|id| {
            let descriptor = catalog.and_then(|catalog| {
                catalog
                    .capabilities
                    .iter()
                    .find(|capability| capability.id == *id)
            });
            let (state, reason, provider) = match (catalog, descriptor) {
                (Some(_), None) => (
                    FactState::Unsupported,
                    Some("当前运行环境的 Worker 能力目录未声明此操作所需能力。".into()),
                    None,
                ),
                (Some(_), Some(capability)) => match capability.state {
                    CapabilityAvailability::Supported => (
                        FactState::Available,
                        capability.reason.clone(),
                        capability.provider.clone(),
                    ),
                    CapabilityAvailability::Unsupported => (
                        FactState::Unsupported,
                        capability.reason.clone(),
                        capability.provider.clone(),
                    ),
                    CapabilityAvailability::Conditional => (
                        FactState::Conditional,
                        capability.reason.clone(),
                        capability.provider.clone(),
                    ),
                    CapabilityAvailability::Unknown => (
                        FactState::Unknown,
                        capability.reason.clone(),
                        capability.provider.clone(),
                    ),
                },
                (None, _) => (
                    FactState::Unknown,
                    Some("运行时能力尚未查询；补齐输入后将检查当前 Python/Worker 环境。".into()),
                    None,
                ),
            };
            ReadinessCapability {
                id: (*id).into(),
                state,
                required: is_required,
                reason,
                provider,
            }
        })
        .collect()
}

fn unknown_source() -> InputSource {
    InputSource {
        kind: InputSourceKind::Unknown,
        reference: None,
        revision: None,
    }
}

fn input_binding(
    key: &str,
    value: Option<Value>,
    source: InputSource,
    required: bool,
) -> InputBinding {
    InputBinding {
        key: key.into(),
        state: if value.is_some() {
            FactState::Available
        } else if required {
            FactState::Missing
        } else {
            FactState::Unknown
        },
        source,
        value,
        required,
        sensitive: false,
    }
}
