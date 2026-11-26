use super::*;
use crate::domain::capability::{CapabilityAvailability, CapabilityCatalog};
use crate::domain::operation::{
    ActionAvailability, FactState, InputBinding, InputSource, InputSourceKind, OperationPlan,
    SuggestedAction,
};
use crate::domain::readiness::{ReadinessCapability, ReadinessInput};

fn is_assembly_runtime_operation(operation: &str) -> bool {
    matches!(operation, "assembly_show" | "assembly_patch" | "assembly_add" | "assembly_set"
        | "assembly_remove" | "assembly_route" | "assembly_encoder_output" | "assembly_generate")
}

fn variant_layer_rng_used(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        object.contains_key("rng_devices")
            || object.get("operations").and_then(Value::as_array).is_some_and(|items| items.iter().any(variant_layer_rng_used))
            || object.get("edits").and_then(Value::as_array).is_some_and(|items| items.iter().any(variant_layer_rng_used))
    })
}

fn assembly_requires_project_batch(operation: &str) -> bool {
    matches!(operation, "assembly_patch" | "assembly_route" | "assembly_encoder_output")
}

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
    Unavailable {
        kind: CapabilityFailureKind,
        diagnostic: String,
    },
}

#[derive(Clone, Copy)]
enum CapabilityFailureKind {
    Environment,
    Dependency,
    Protocol,
    Unknown,
}

impl CapabilityFailureKind {
    fn action_id(self) -> &'static str {
        match self {
            Self::Environment => "inspect_environment",
            Self::Dependency => "review_missing_dependency",
            Self::Protocol => "inspect_protocol_compatibility",
            Self::Unknown => "inspect_unknown_diagnostic",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Environment => "运行环境检查失败",
            Self::Dependency => "运行时依赖缺失或不可用",
            Self::Protocol => "CLI/Worker 协议或能力目录不兼容",
            Self::Unknown => "诊断类型未知",
        }
    }

    fn action_label(self) -> &'static str {
        match self {
            Self::Environment => "查看 Python/Worker 环境诊断",
            Self::Dependency => "查看缺失依赖诊断并由用户决定如何恢复",
            Self::Protocol => "查看 CLI/Worker 协议兼容诊断",
            Self::Unknown => "查看原始 CLI 诊断并检查运行环境",
        }
    }
}

impl<B: Backend> Application<B> {
    /// Build an operation plan without importing the project model or changing
    /// project state. Runtime capability discovery may start the selected worker.
    pub fn plan_operation_inputs(
        &self,
        cli: &Cli,
        command: &Command,
    ) -> Result<Option<OperationPlan>, Report> {
        if let Some(normalized) = super::scaffold::normalize_training_command(command)? {
            return self.plan_operation_inputs(cli, &normalized);
        }
        let Some(request) = self.readiness_request(command) else {
            return Ok(None);
        };
        // Assembly's own early checks provide its specific diagnostic for a
        // variant; do not start a runtime probe before that rejection.
        if is_assembly_runtime_operation(request.operation) && request.operation != "assembly_generate" && cli.variant.is_some() {
            return Ok(None);
        }
        let root = std::env::current_dir()
            .and_then(|path| path.canonicalize())
            .map_err(|error| Report::error(
                "operation_readiness",
                "PROJECT_UNAVAILABLE",
                format!("无法读取当前项目目录；目标操作没有启动：{error}"),
                "选择可访问的项目目录后重试。",
            ))?;
        let config = match self.project_config.load(&root) {
            Ok(config) => config,
            Err(error) => return Err(Report::error(
                "operation_readiness",
                "PROJECT_CONFIG_INVALID",
                format!("无法读取项目配置；目标操作没有启动：{error}"),
                "检查并修复项目配置，或选择其他项目后重试。",
            )),
        };
        if is_assembly_runtime_operation(request.operation) && config.is_none() {
            // Preserve the existing project-initialization recovery diagnostic.
            return Ok(None);
        }
        let config_revision = config
            .as_ref()
            .and_then(|config| crate::domain::input::content_revision(config).ok());
        // The execution path falls back to the working variant when no model
        // was supplied or resolved from project configuration. Keep readiness
        // consistent, without applying that fallback to assembly operations
        // or historical runs that resolve their own inputs.
        let implicit_working = request.layer_context && request.edit_target.as_deref() != Some("source")
            || (request.require_model && !is_assembly_runtime_operation(request.operation)
                && request.replay.is_none() && request.inputs.model.is_none()
                && config.as_ref().and_then(|value| value["model"].as_str()).is_none());
        let variant_name = cli.variant.as_deref().or(if implicit_working { Some("working") } else { None });
        let selected_variant = variant_name.and_then(|name| self.variants.load(&root, name).ok().flatten());
        if request.operation == "assembly_generate" && cli.variant.is_some() && selected_variant.is_none() {
            return Ok(None);
        }
        if let Command::Assembly { command: AssemblyCommand::Generate { inputs, .. } } = command {
            if (cli.variant.is_some() && !inputs.model_params.is_empty())
                || inputs.if_config_revision.as_deref().is_some_and(|expected| Some(expected) != config_revision.as_deref()) {
                return Ok(None);
            }
        }
        let variant_candidate = selected_variant.as_ref().and_then(|variant| {
            variant["origin"]["model"].as_str().or_else(|| variant["model"].as_str()).map(|model| (
                Value::String(model.to_owned()),
                InputSource { kind: InputSourceKind::ModelVariant,
                    reference: variant_name.map(str::to_owned),
                    revision: variant["revision"].as_str().map(str::to_owned) },
            ))
        });

        let wants_new_variant = request.edit_target.as_deref() == Some("variant")
            && selected_variant.is_none();
        let model = if cli.variant.is_some()
            && !wants_new_variant
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

        let framework_neutral = matches!(request.operation, "generic_loop_trial" | "generic_loop_train");
        // A user-owned framework-neutral loop does not consume the project's
        // model, batch, or their bindings, even if those defaults exist.
        let mut bindings = if framework_neutral || matches!(request.operation, "evaluate" | "preview") { vec![] } else { vec![model, batch] };
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
                id: if cli.variant.is_some() && !wants_new_variant {
                    "select_model_variant".into()
                } else {
                    "configure_model".into()
                },
                label: if cli.variant.is_some() && !wants_new_variant {
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
                id: if assembly_requires_project_batch(request.operation) { "configure_project_batch" } else { "configure_batch" }.into(),
                label: if assembly_requires_project_batch(request.operation) {
                    "先通过 m-model config set-batch 预览并保存项目默认批次，再重新预览装配".into()
                } else {
                    "提供示例批次工厂：--batch module:batch_factory".into()
                },
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
        let effective_python = if cli.python_explicit {
            cli.python.as_deref()
        } else {
            request.replay.as_ref().and_then(|(_, record)| record["inputs"]["runtime"]["python_executable"].as_str())
                .filter(|value| !value.trim().is_empty()).or(cli.python.as_deref())
        };
        // A historical run can select an interpreter different from the
        // current workspace. Its capabilities must not reuse or replace the
        // current environment's cached catalog.
        let current_environment = effective_python == cli.python.as_deref();
        let uses_raw_binding = match request.inputs.batch_binding.as_ref() {
            Some(binding) => *binding == BatchBinding::SingleRaw,
            None if request.inputs.batch.is_some() => false,
            None => match request.replay.as_ref() {
                Some((_, record)) => record["inputs"]["batch_binding"].as_str() == Some("single-raw"),
                None => config.as_ref().and_then(|config| config["batch_binding"].as_str()) == Some("single-raw"),
            },
        };
        let mut required_capabilities = request.capabilities.clone();
        if request.operation == "preview" {
            let configured_paths = config.as_ref().and_then(|value| value.get("runtime_paths"));
            let custom_paths = std::env::var_os("MODEL_MAVEN_WORKSPACE_WORKING_DIRECTORY").is_some()
                || std::env::var_os("MODEL_MAVEN_WORKSPACE_IMPORT_ROOTS").is_some()
                || configured_paths.is_some_and(|paths| {
                    !paths["working_directory"].is_null()
                        || paths["project_import_roots"].as_array().is_some_and(|roots| !roots.is_empty())
                });
            if custom_paths { required_capabilities.push("data.preview_runtime_paths_v1"); }
        }
        let replay_variant = request.replay.as_ref().and_then(|(_, record)| record["inputs"].get("variant"));
        let effective_variant = replay_variant.filter(|variant| variant.is_object()).or(selected_variant.as_ref());
        if effective_variant.is_some_and(|variant| variant["device_policy"].as_str() == Some("factory"))
            && !required_capabilities.contains(&"model.variant_factory_device_v1") {
            required_capabilities.push("model.variant_factory_device_v1");
        }
        let has_batch = bindings.iter().any(|binding| binding.key == "batch" && binding.state == FactState::Available);
        if is_assembly_runtime_operation(request.operation) {
            let uses_variant = request.operation == "assembly_generate" && selected_variant.is_some();
            if has_batch {
                required_capabilities.extend(["model.forward_probe", "data.dataset_probe"]);
                if request.operation == "assembly_generate" || (cli.seed.is_none() && !uses_variant) {
                    required_capabilities.push("data.runtime_sampling_seed_v1");
                }
            } else {
                required_capabilities.push("model.inspect");
            }
            if cli.seed.is_none() && !uses_variant { required_capabilities.push("model.runtime_rng_v1"); }
            if request.operation == "assembly_generate" {
                required_capabilities.extend(["edit.assembly_codegen", "edit.assembly_codegen_runtime_paths_v1"]);
                if !uses_variant && cli.seed.is_some() { required_capabilities.push("edit.assembly_python_rng_v1"); }
                if selected_variant.as_ref().is_some_and(variant_layer_rng_used) { required_capabilities.push("model.variant_layer_rng_v1"); }
                if selected_variant.as_ref().is_some_and(|variant| variant.get("base_rng_devices").is_some()) {
                    required_capabilities.push("model.variant_base_rng_v1");
                }
            }
            let injections = selected_variant.as_ref().and_then(|variant| variant["origin"]["model_injections"].as_object())
                .or_else(|| config.as_ref().and_then(|value| value["model_injections"].as_object()));
            if injections.is_some_and(|items| {
                items.values().any(|binding| binding.is_object() && binding.get("factory").is_some()
                    && binding.get("component_ref").is_none())
            }) && !required_capabilities.contains(&"model.direct_factory_params_v1") {
                required_capabilities.push("model.direct_factory_params_v1");
            }
            let paths = config.as_ref().and_then(|value| value.get("runtime_paths"));
            if std::env::var_os("MODEL_MAVEN_WORKSPACE_WORKING_DIRECTORY").is_some()
                || paths.is_some_and(|value| !value["working_directory"].is_null()) {
                required_capabilities.push("runtime.user_working_directory");
            }
            if std::env::var_os("MODEL_MAVEN_WORKSPACE_IMPORT_ROOTS").is_some()
                || paths.is_some_and(|value| value["project_import_roots"].as_array().is_some_and(|roots| !roots.is_empty())) {
                required_capabilities.push("runtime.project_import_roots");
            }
        }
        if has_batch && uses_raw_binding && !required_capabilities.contains(&"data.raw_batch_binding_v1") {
            required_capabilities.push("data.raw_batch_binding_v1");
        }
        if matches!(request.operation, "plan_layer_edit" | "apply_layer_edit")
            && request.edit_target.as_deref() == Some("source") {
            required_capabilities.push("edit.source_factory_device_v1");
            if cli.seed.is_none() {
                required_capabilities.push("model.runtime_rng_v1");
            }
        }
        if has_batch && cli.seed.is_none()
            && matches!(request.operation, "forward_probe" | "loss_probe" | "plan_layer_edit" | "apply_layer_edit") {
            required_capabilities.push("data.runtime_sampling_seed_v1");
        }
        let mut capability_failure = None;
        let catalog = if required_capabilities.is_empty() {
            None
        } else if required_inputs_missing {
            let cached = if current_environment { self.runtime_capabilities.borrow().clone() } else { None };
            if cached.is_none() {
                conditions.push(
                    "补齐必需输入后再检测当前 Python/Worker 能力；先解决静态前置条件。".into(),
                );
            }
            cached
        } else {
            match self.runtime_capability_catalog(cli, effective_python, current_environment, framework_neutral) {
                Ok(catalog) => Some(catalog),
                Err(CapabilityQueryFailure::Interrupted) => {
                    return Err(Report::error(
                        "operation_readiness",
                        "USER_INTERRUPTED",
                        "检查当前 Python/Worker 能力时收到中断信号；目标操作已取消。",
                        "重新执行命令即可重试。",
                    ));
                }
                Err(CapabilityQueryFailure::Unavailable { kind, diagnostic }) => {
                    capability_failure = Some((kind, diagnostic.clone()));
                    conditions.push(format!("{}：{diagnostic}", kind.label()));
                    let inspect_action_id = kind.action_id();
                    if !actions.iter().any(|action| action.id == inspect_action_id) {
                        actions.push(SuggestedAction {
                            id: inspect_action_id.into(),
                            label: kind.action_label().into(),
                            availability: ActionAvailability::Available,
                            reason: Some(diagnostic.clone()),
                        });
                    }
                    if matches!(
                        kind,
                        CapabilityFailureKind::Environment | CapabilityFailureKind::Dependency
                    )
                        && !actions
                            .iter()
                            .any(|action| action.id == "choose_python_interpreter")
                    {
                        actions.push(SuggestedAction {
                            id: "choose_python_interpreter".into(),
                            label: "显式选择 Python 解释器后重新检查".into(),
                            availability: ActionAvailability::Available,
                            reason: Some(diagnostic.clone()),
                        });
                    }
                    None
                }
            }
        };
        let mut capabilities = capability_states(
            &required_capabilities,
            catalog.as_ref(),
            !required_inputs_missing,
        );
        if let Some((kind, diagnostic)) = capability_failure {
            for capability in capabilities
                .iter_mut()
                .filter(|capability| capability.state == FactState::Unknown)
            {
                capability.reason = Some(format!("{}：{diagnostic}", kind.label()));
                capability.provider = Some(if framework_neutral { "worker capability declarations" } else { "m-model doctor" }.into());
            }
        }
        if let Some(capability) = capabilities.iter().find(|item| item.required && item.state == FactState::Unsupported)
            && !actions.iter().any(|action| action.id == "inspect_environment")
        {
            actions.push(SuggestedAction {
                id: "inspect_environment".into(),
                label: "查看所选 Python 环境与缺失能力，再决定是否切换环境".into(),
                availability: ActionAvailability::Available,
                reason: Some(format!("{}：{}", capability.id, capability.reason.as_deref().unwrap_or("运行时声明不支持"))),
            });
        }
        if let Some(capability) = capabilities.iter().find(|item| item.required && item.state == FactState::Unknown)
            && !actions.iter().any(|action| matches!(action.id.as_str(), "inspect_protocol_compatibility" | "inspect_unknown_diagnostic"))
        {
            actions.push(SuggestedAction {
                id: "inspect_unknown_diagnostic".into(),
                label: "查看 Worker 能力与版本诊断；未知不表示已支持".into(),
                availability: ActionAvailability::Available,
                reason: Some(format!("{}：{}", capability.id, capability.reason.as_deref().unwrap_or("能力状态未知"))),
            });
        }
        capabilities.extend(local_capabilities);
        let plan = crate::domain::readiness::plan(
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
        .map_err(|error| {
            Report::error(
                "operation_readiness",
                "OPERATION_PLAN_INVALID",
                format!("无法建立有效的操作准备计划；目标操作没有启动：{error}"),
                "检查操作输入和能力目录后重试；如问题持续，请保留诊断。",
            )
        })?;
        Ok(Some(plan))
    }

    fn runtime_capability_catalog(
        &self,
        cli: &Cli,
        python: Option<&str>,
        use_cache: bool,
        declarations_only: bool,
    ) -> Result<CapabilityCatalog, CapabilityQueryFailure> {
        if use_cache {
            let cached = if declarations_only { &self.runtime_declarations } else { &self.runtime_capabilities };
            if let Some(catalog) = cached.borrow().as_ref() {
                return Ok(catalog.clone());
            }
        }
        let capability_operation = if declarations_only { "runtime_capabilities" } else { "doctor" };
        let report = self.backend.execute(&serde_json::json!({
            "protocol_version": crate::domain::PROTOCOL_VERSION,
            "operation": capability_operation,
            "runtime": {
                "python_executable": python,
                "timeout_seconds": cli.timeout.unwrap_or(30)
            }
        }));
        if report.protocol_version != crate::domain::PROTOCOL_VERSION
            || report.operation != capability_operation
        {
            return Err(CapabilityQueryFailure::Unavailable {
                kind: CapabilityFailureKind::Protocol,
                diagnostic: format!(
                    "能力检查响应不兼容：protocol_version={}，operation={}，status={}；原始诊断：{}",
                    report.protocol_version,
                    report.operation,
                    report.status,
                    format_diagnostics(&report)
                ),
            });
        }
        if !report.success() {
            let diagnostic = report.diagnostics.first();
            if diagnostic.is_some_and(|diagnostic| diagnostic.code == "USER_INTERRUPTED") {
                return Err(CapabilityQueryFailure::Interrupted);
            }
            let code = diagnostic.map(|diagnostic| diagnostic.code.as_str());
            let kind = match code {
                Some(
                    "PYTHON_UNAVAILABLE"
                    | "PYTHON_SELECTION_REQUIRED"
                    | "ENVIRONMENT_MISMATCH"
                ) => CapabilityFailureKind::Environment,
                Some("TORCH_UNAVAILABLE" | "ENVIRONMENT_RUNTIME_DEPENDENCY_MISSING") => {
                    CapabilityFailureKind::Dependency
                }
                Some("PROTOCOL_ERROR" | "CAPABILITY_CATALOG_INVALID" | "OPERATION_UNSUPPORTED"
                    | "WORKER_API_UNSUPPORTED" | "STATE_FORMAT_UNSUPPORTED" | "FEATURE_UNAVAILABLE") => {
                    CapabilityFailureKind::Protocol
                }
                _ => CapabilityFailureKind::Unknown,
            };
            return Err(CapabilityQueryFailure::Unavailable {
                kind,
                diagnostic: diagnostic
                    .map(|_| format_diagnostics(&report))
                    .unwrap_or_else(|| "Worker 没有返回成功的能力目录".into()),
            });
        }
        let catalog = CapabilityCatalog::from_doctor_data(&report.data)
            .map_err(|diagnostic| CapabilityQueryFailure::Unavailable {
                kind: CapabilityFailureKind::Protocol,
                diagnostic,
            })?;
        if use_cache {
            let cache = if declarations_only { &self.runtime_declarations } else { &self.runtime_capabilities };
            *cache.borrow_mut() = Some(catalog.clone());
        }
        Ok(catalog)
    }

    fn readiness_request(&self, command: &Command) -> Option<PlanRequest> {
        match command {
            Command::LoopTrial { training_loop, loop_interface, .. } => {
                let mut capabilities = vec!["training.generic_loop_trial_v1"];
                if loop_interface == "context" { capabilities.push("training.generic_loop_context_signature_v1"); }
                Some(PlanRequest {
                    operation: "generic_loop_trial", inputs: Inputs::default(),
                    require_model: false, require_batch: false,
                    training_input: Some(("training_loop", Some(training_loop.clone()))),
                    capabilities, local_capabilities: vec![], facts: vec![], conditions: vec![],
                    actions: vec![], component: None, layer_selection: None, edit_target: None,
                    replay: None, layer_context: false,
                })
            }
            Command::LoopRun { training_loop, loop_interface, resume, detach, .. } => {
                let replay = resume.as_deref().and_then(|id| self.readiness_training_record(id, "generic_loop_train"));
                if resume.is_some() && replay.is_none() { return None; }
                let effective_interface = loop_interface.as_deref().or_else(|| replay.as_ref()
                    .and_then(|(_, record)| record["inputs"]["loop_interface"].as_str()));
                let mut capabilities = vec!["training.generic_loop_train_v1"];
                if effective_interface == Some("context") { capabilities.push("training.generic_loop_context_signature_v1"); }
                if *detach { capabilities.push("training.detached"); }
                Some(PlanRequest {
                    operation: "generic_loop_train", inputs: Inputs::default(),
                    require_model: false, require_batch: false,
                    training_input: Some(("training_loop", training_loop.clone())),
                    capabilities, local_capabilities: vec![], facts: vec![], conditions: vec![],
                    actions: vec![], component: None, layer_selection: None, edit_target: None,
                    replay, layer_context: false,
                })
            }
            Command::Evaluate { run, device, .. } => {
                // Keep invalid/missing run diagnostics in the evaluation path;
                // only a resolved historical record provides a Python source.
                let replay = self.readiness_training_record(run, "train")?;
                let mut capabilities = vec!["training.evaluation", "training.checkpoint_resume"];
                if device.is_some() { capabilities.push("evaluation.model_device_v1"); }
                Some(PlanRequest {
                    operation: "evaluate", inputs: Inputs::default(),
                    require_model: false, require_batch: false, training_input: None,
                    capabilities, local_capabilities: vec![],
                    facts: vec![ReadinessInput {
                        key: "source_run_record".into(), state: FactState::Available, required: true,
                        reason: Some("已读取训练运行记录；checkpoint、评估器与数据仍在执行时单独核对。".into()),
                        source: Some(InputSource { kind: InputSourceKind::SavedRun,
                            reference: Some(replay.0.clone()), revision: None }),
                    }],
                    conditions: vec![], actions: vec![], component: None, layer_selection: None,
                    edit_target: None, replay: Some(replay), layer_context: false,
                })
            }
            Command::Data { command: DataCommand::Preview { dataset, batch, whole_batch, .. } } => {
                let source = if dataset.is_some() { "--dataset" } else if batch.is_some() { "--batch" } else { return None; };
                let mut capabilities = vec!["data.preview"];
                if dataset.is_some() { capabilities.push("data.dataset_probe"); }
                if *whole_batch { capabilities.push("data.raw_batch_binding_v1"); }
                Some(PlanRequest {
                    operation: "preview", inputs: Inputs::default(), require_model: false,
                    require_batch: false, training_input: None, capabilities,
                    local_capabilities: vec![], facts: vec![ReadinessInput {
                        key: "data_source_selection".into(), state: FactState::Available, required: true,
                        reason: Some("已选择数据来源；工厂、样本和渲染器仍需实际执行检查。".into()),
                        source: Some(InputSource { kind: InputSourceKind::CommandLine,
                            reference: Some(source.into()), revision: None }),
                    }],
                    conditions: vec![], actions: vec![], component: None, layer_selection: None,
                    edit_target: None, replay: None, layer_context: false,
                })
            }
            Command::Assembly { command: AssemblyCommand::Show { inputs, config_only: false } }
            | Command::Assembly { command: AssemblyCommand::Patch { inputs, .. } }
            | Command::Assembly { command: AssemblyCommand::Add { inputs, .. } }
            | Command::Assembly { command: AssemblyCommand::Set { inputs, .. } }
            | Command::Assembly { command: AssemblyCommand::Remove { inputs, .. } }
            | Command::Assembly { command: AssemblyCommand::Route { inputs, .. } }
            | Command::Assembly { command: AssemblyCommand::EncoderOutput { inputs, .. } }
            | Command::Assembly { command: AssemblyCommand::Generate { inputs, .. } } => {
                let operation = match command {
                    Command::Assembly { command: AssemblyCommand::Show { .. } } => "assembly_show",
                    Command::Assembly { command: AssemblyCommand::Patch { .. } } => "assembly_patch",
                    Command::Assembly { command: AssemblyCommand::Add { .. } } => "assembly_add",
                    Command::Assembly { command: AssemblyCommand::Set { .. } } => "assembly_set",
                    Command::Assembly { command: AssemblyCommand::Remove { .. } } => "assembly_remove",
                    Command::Assembly { command: AssemblyCommand::Route { .. } } => "assembly_route",
                    Command::Assembly { command: AssemblyCommand::EncoderOutput { .. } } => "assembly_encoder_output",
                    _ => "assembly_generate",
                };
                if operation == "assembly_show" && inputs.if_config_revision.is_some() { return None; }
                if let Command::Assembly { command: AssemblyCommand::Generate { output, .. } } = command {
                    if !super::config::valid_assembly_output_path(output) { return None; }
                }
                if let Command::Assembly { command: AssemblyCommand::Patch { sets, removes, encoder_outputs, route_changes, apply, .. } } = command {
                    if (sets.is_empty() && removes.is_empty() && encoder_outputs.is_empty() && route_changes.is_empty())
                        || (*apply && inputs.if_config_revision.is_none())
                        || inputs.batch.is_some() || inputs.batch_binding.is_some()
                        || !inputs.batch_params.is_empty() || !inputs.model_params.is_empty() {
                        return None;
                    }
                }
                if let Command::Assembly { command: AssemblyCommand::Route { dry_run, .. } | AssemblyCommand::EncoderOutput { dry_run, .. } } = command {
                    if (!dry_run && inputs.if_config_revision.is_none()) || inputs.batch.is_some()
                        || inputs.batch_binding.is_some() || !inputs.batch_params.is_empty() || !inputs.model_params.is_empty() {
                        return None;
                    }
                }
                let direct_factory = matches!(command,
                    Command::Assembly { command: AssemblyCommand::Add { factory: Some(_), .. } | AssemblyCommand::Set { factory: Some(_), .. } });
                Some(PlanRequest {
                    operation,
                    inputs: Inputs { batch: inputs.batch.clone(), batch_binding: inputs.batch_binding.clone(),
                        batch_params: inputs.batch_params.clone(), model_params: inputs.model_params.clone(), ..Inputs::default() },
                    require_model: true, require_batch: assembly_requires_project_batch(operation), training_input: None,
                    capabilities: if direct_factory { vec!["model.direct_factory_params_v1"] } else { vec![] },
                    local_capabilities: vec![], facts: vec![], conditions: vec![], actions: vec![],
                    component: None, layer_selection: None, edit_target: None, replay: None, layer_context: false,
                })
            }
            Command::Model { command } => {
                if matches!(command, ModelCommand::Unmark { .. })
                    || matches!(command, ModelCommand::Marks { inputs } if inputs.model.is_none() && inputs.batch.is_none())
                    || matches!(
                        command,
                        ModelCommand::Supervise {
                            command: SuperviseCommand::Rule { .. } | SuperviseCommand::Export { .. } | SuperviseCommand::Import { .. }
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
                        command: SuperviseCommand::Rule { .. } | SuperviseCommand::Export { .. } | SuperviseCommand::Import { .. },
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
                        ModelCommand::Probe { contract, if_contract_sha256, model_device, .. } => {
                            let mut required = vec!["model.forward_probe", "data.dataset_probe"];
                            if model_device.is_some() { required.push("model.probe_device_v1"); }
                            if contract.is_some() { required.push("model.contract_port_paths_v2"); }
                            if if_contract_sha256.is_some() {
                                required.push("model.contract_identity_v1");
                                required.push("model.contract_preload_v1");
                            }
                            required
                        }
                        ModelCommand::Mark { path, model_device, .. } => {
                            let mut required = vec![
                                "model.forward_probe",
                                "data.dataset_probe",
                                "model.semantic_marks",
                                "supervision.port_path_v2",
                            ];
                            if path == "model" { required.push("supervision.root_mark_v1"); }
                            if model_device.is_some() { required.push("model.semantic_device_v1"); }
                            required
                        }
                        ModelCommand::Retag { path, model_device, .. } => {
                            let mut required = vec![
                                "model.forward_probe",
                                "data.dataset_probe",
                                "model.semantic_marks",
                                "supervision.port_path_v2",
                            ];
                            if path == "model" { required.push("supervision.root_mark_v1"); }
                            if model_device.is_some() { required.push("model.semantic_device_v1"); }
                            required
                        }
                        ModelCommand::Marks { .. } => vec![
                            "model.forward_probe",
                            "data.dataset_probe",
                            "model.semantic_marks",
                            "supervision.port_path_v2",
                        ],
                        ModelCommand::Unmark { .. } => unreachable!(),
                        ModelCommand::Supervise {
                            command: SuperviseCommand::Guide { .. },
                        } => vec![
                            "model.forward_probe",
                            "data.dataset_probe",
                            "model.semantic_marks",
                            "training.supervision_preview",
                            "supervision.port_path_v2",
                        ],
                        ModelCommand::Supervise {
                            command: SuperviseCommand::Preview { model_device, .. },
                        } => {
                            let mut required = vec!["model.forward_probe", "data.dataset_probe", "model.semantic_marks",
                                "training.supervision_preview", "supervision.port_path_v2"];
                            if model_device.is_some() { required.push("supervision.preview_device_v1"); }
                            required
                        },
                        ModelCommand::Supervise {
                            command: SuperviseCommand::Rule { .. } | SuperviseCommand::Export { .. } | SuperviseCommand::Import { .. },
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
                no_supervision,
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
                let previous_mode = replay.as_ref().and_then(|(_, record)| record["inputs"]["training_mode"].as_str());
                let external = match training_mode.as_ref() {
                    Some(mode) => *mode == TrainingMode::ExternalLoop,
                    None if training_loop.is_some() => true,
                    None if step.is_some() => false,
                    None => previous_mode == Some("external-loop"),
                };
                let inherited_mode = if step.is_some() && previous_mode == Some("external-loop") {
                    None
                } else { previous_mode };
                let supervision = !*no_supervision && (*apply_supervision || replay.as_ref().is_some_and(|(_, record)| {
                    record["inputs"]["apply_supervision"].as_bool().unwrap_or(false)
                }));
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
                            vec![training_capability(external, training_mode.as_ref(), inherited_mode)];
                        if !external && (inputs.batch_device_policy.is_some() || replay.is_none()
                            || replay.as_ref().is_some_and(|(_, record)| record["inputs"]["batch_device_policy"].as_str().is_some())) {
                            required.push("training.batch_device_policy_v1");
                        }
                        if supervision {
                            required.push("training.supervision_apply");
                            required.push("supervision.port_path_v2");
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
                let previous_mode = replay.as_ref().and_then(|(_, record)| record["inputs"]["training_mode"].as_str());
                let supervision = *apply_supervision || replay.as_ref().is_some_and(|(_, record)| {
                    record["inputs"]["apply_supervision"].as_bool().unwrap_or(false)
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
                            vec![training_capability(external, training_mode.as_ref(), previous_mode)];
                        if !external && (inputs.batch_device_policy.is_some() || replay.is_none()
                            || replay.as_ref().is_some_and(|(_, record)| record["inputs"]["batch_device_policy"].as_str().is_some())) {
                            required.push("training.batch_device_policy_v1");
                        }
                        if resume.is_some() {
                            required.push("training.checkpoint_resume");
                        }
                        if supervision {
                            required.push("training.supervision_apply");
                            required.push("supervision.port_path_v2");
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
                let uses_factory_policy = job.variant_device_policy.as_deref() == Some("factory");
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
                    } else if is_edit && uses_factory_policy {
                        vec!["model.layer_edit", "model.variant_factory_device_v1"]
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

fn training_capability(external: bool, training_mode: Option<&TrainingMode>, previous_mode: Option<&str>) -> &'static str {
    if external { return "training.external_loop"; }
    match training_mode {
        Some(TrainingMode::ManualStep) => "training.manual_step",
        Some(_) => "training.managed_step",
        None if previous_mode == Some("manual-step") => "training.manual_step",
        None => "training.managed_step",
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
                (Some(catalog), None) => match catalog.state_for(id) {
                    crate::domain::capability::CapabilityAvailability::Unsupported => (
                        FactState::Unsupported,
                        Some("当前运行环境明确声明不支持此能力。".into()),
                        None,
                    ),
                    _ => (
                        FactState::Unknown,
                        Some("当前 Worker 能力目录未包含此项；可能是旧版或不完整目录，不能据此判定明确不支持。".into()),
                        None,
                    ),
                },
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

fn format_diagnostics(report: &Report) -> String {
    let diagnostics = report
        .diagnostics
        .iter()
        .map(|diagnostic| {
            let mut text = format!("{}: {}", diagnostic.code, diagnostic.message);
            if let Some(hint) = diagnostic
                .hint
                .as_deref()
                .filter(|hint| !hint.trim().is_empty())
            {
                text.push_str("；hint: ");
                text.push_str(hint);
            }
            text
        })
        .collect::<Vec<_>>();
    if diagnostics.is_empty() {
        "未返回结构化诊断".into()
    } else {
        diagnostics.join(" | ")
    }
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
