use super::*;

impl<B: Backend> Application<B> {
    pub(super) fn handle_component(
        &self,
        cli: &Cli,
        command: &ComponentCommand,
        context: ExecutionContext<'_>,
    ) -> Report {
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error("component", "PROJECT_UNAVAILABLE", error.to_string(), "");
            }
        };
        match command {
            ComponentCommand::Install {
                path,
                no_build_isolation,
            } => {
                let installation = self.backend.execute(&json!({
                    "protocol_version":PROTOCOL_VERSION,
                    "operation":"component_install",
                    "path":path.to_string_lossy(),
                    "no_build_isolation":no_build_isolation,
                    "interactive_progress":context.show_progress,
                    "runtime":{
                        "python_executable":cli.python.as_deref().unwrap_or("python3"),
                        "timeout_seconds":cli.timeout.unwrap_or(1800),
                        "project_root":root
                    }
                }));
                if !installation.success() {
                    return installation;
                }
                let discovery = self.run(
                    cli,
                    &Command::Component {
                        command: ComponentCommand::Discover {
                            distribution: None,
                            paths: Vec::new(),
                        },
                    },
                );
                if !discovery.success() {
                    return discovery;
                }
                let expected = installation.data["components"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let mut registered = discovery.data["registered"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                registered.extend(
                    discovery.data["already_registered"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default(),
                );
                let all_registered = !expected.is_empty()
                    && expected.iter().all(|component| {
                        registered.iter().any(|item| {
                            let id = item["id"]
                                .as_str()
                                .or_else(|| item["manifest"]["component_id"].as_str());
                            let version = item["version"]
                                .as_str()
                                .or_else(|| item["manifest"]["component_version"].as_str());
                            id == component["component_id"].as_str()
                                && version == component["version"].as_str()
                        })
                    });
                if !all_registered {
                    let mut report = Report::error(
                        "component_install",
                        "COMPONENT_REGISTRATION_FAILED",
                        "wheel 已安装到项目扩展存储，但至少一个组件清单未能登记。",
                        "查看 discovery.issues；同 ID/版本内容冲突时请提升组件版本后重新安装。",
                    );
                    report.data = json!({
                        "installation":installation.data,
                        "discovery":discovery.data
                    });
                    return report;
                }
                let mut data = discovery.data;
                data["installation"] = installation.data;
                Report::ok("component_install", data)
            }
            ComponentCommand::Uninstall {
                id,
                apply,
                force,
                if_plan,
            } => {
                if if_plan.as_deref().is_some_and(|value| {
                    value.len() != 64
                        || !value
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                }) {
                    return Report::error(
                        "component_uninstall",
                        "COMPONENT_PLAN_INVALID",
                        "组件移除计划摘要无效。",
                        "重新预览移除计划。",
                    );
                }
                if *force && !*apply {
                    return Report::error(
                        "component_uninstall",
                        "COMPONENT_UNINSTALL_FLAGS_INVALID",
                        "--force 需要与 --apply 一起使用。",
                        "先查看默认预览；确认影响后再同时指定 --apply --force。",
                    );
                }
                let active_runs = match self.runs.list(&root) {
                    Ok(mut runs) => {
                        for run in &mut runs {
                            self.annotate_status(run);
                        }
                        runs.into_iter()
                            .filter(|run| {
                                matches!(
                                    run["status"].as_str(),
                                    Some("running" | "stopping" | "unknown")
                                )
                            })
                            .map(|run| json!({"id":run["id"],"status":run["status"]}))
                            .collect::<Vec<_>>()
                    }
                    Err(error) => {
                        return Report::error(
                            "component_uninstall",
                            "COMPONENT_RUN_STATE_UNAVAILABLE",
                            error,
                            "无法确认项目训练任务状态时不会移除扩展。",
                        );
                    }
                };
                if *apply && !active_runs.is_empty() {
                    let mut report = Report::error(
                        "component_uninstall",
                        "COMPONENT_ACTIVE_RUNS",
                        "项目仍有运行中或状态未知的任务，暂不能移除组件。",
                        "先停止活动任务，并确认 unknown 运行的进程状态后重试。",
                    );
                    report.data = json!({"active_runs":active_runs});
                    return report;
                }
                self.backend.execute(&json!({
                    "protocol_version":PROTOCOL_VERSION,
                    "operation":"component_uninstall",
                    "id":id,
                    "apply":*apply,
                    "force":*force,
                    "if_plan":if_plan,
                    "active_runs":active_runs,
                    "runtime":{
                        "python_executable":cli.python.as_deref().unwrap_or("python3"),
                        "timeout_seconds":cli.timeout.unwrap_or(30),
                        "project_root":root
                    }
                }))
            }
            ComponentCommand::Scaffold {
                id,
                kind,
                version,
                output,
            } => match self.scaffolder.scaffold(&root, id, kind, version, output) {
                Ok(result) => Report::ok("component_scaffold", result),
                Err(error) => Report::error(
                    "component_scaffold",
                    "COMPONENT_SCAFFOLD_FAILED",
                    error,
                    "检查组件 ID、版本和项目内输出目录；脚手架不会覆盖已有目录。",
                ),
            },
            ComponentCommand::Discover {
                distribution,
                paths,
            } => {
                let request = json!({
                    "protocol_version":PROTOCOL_VERSION,
                    "operation":"component_discover",
                    "distribution":distribution,
                    "paths":paths.iter().map(|path| path.to_string_lossy().into_owned()).collect::<Vec<_>>(),
                    "runtime":{
                        "python_executable":cli.python.as_deref().unwrap_or("python3"),
                        "timeout_seconds":cli.timeout.unwrap_or(30)
                    }
                });
                let discovery = self.backend.execute(&request);
                if !discovery.success() {
                    return discovery;
                }
                let discovered = discovery.data["components"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let mut issues = discovery.data["issues"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                let mut registered = Vec::new();
                let mut already_registered = Vec::new();
                for item in discovered.iter() {
                    let Some(mut manifest) = item.get("manifest").cloned() else {
                        issues.push(json!({"message":"发现结果缺少 manifest。"}));
                        continue;
                    };
                    let Some(package_source) = item.get("package_source").cloned() else {
                        issues.push(json!({"message":"发现结果缺少 package_source。"}));
                        continue;
                    };
                    manifest["package_source"] = package_source.clone();
                    let Some(id) = manifest["component_id"].as_str() else {
                        issues.push(json!({"distribution":package_source["distribution"],"entry_point":package_source["entry_point"],"message":"清单缺少 component_id。"}));
                        continue;
                    };
                    let Some(version) = manifest["component_version"].as_str() else {
                        issues.push(json!({"distribution":package_source["distribution"],"entry_point":package_source["entry_point"],"message":"清单缺少 component_version。"}));
                        continue;
                    };
                    let reference = format!("{id}@{version}");
                    match self.components.inspect(&root, &reference) {
                        Ok(existing) if existing["manifest"] == manifest => {
                            already_registered.push(json!({"id":id,"version":version}));
                        }
                        Ok(_) => issues.push(json!({
                            "id":id,
                            "version":version,
                            "distribution":package_source["distribution"],
                            "entry_point":package_source["entry_point"],
                            "message":"同一组件 ID 和版本已注册为不同内容；未覆盖现有清单。"
                        })),
                        Err(_) => match self.components.register(&root, &manifest) {
                            Ok(component) => registered.push(component),
                            Err(error) => issues.push(json!({
                                "id":id,
                                "version":version,
                                "distribution":package_source["distribution"],
                                "entry_point":package_source["entry_point"],
                                "message":error
                            })),
                        },
                    }
                }
                Report::ok(
                    "component_discover",
                    json!({
                        "found":discovered.len(),
                        "registered":registered,
                        "already_registered":already_registered,
                        "issues":issues,
                        "scanned_paths":discovery.data["scanned_paths"]
                    }),
                )
            }
            ComponentCommand::List { query } => {
                match self.components.list(&root, query.as_deref()) {
                    Ok(components) => Report::ok(
                        "component_list",
                        json!({"components":components,"filtered":query.is_some()}),
                    ),
                    Err(error) => Report::error(
                        "component_list",
                        "COMPONENT_SCAN_FAILED",
                        error,
                        "检查 .model-maven/components 中的清单和目录权限。",
                    ),
                }
            }
            ComponentCommand::Show { id } => match self.components.inspect(&root, id) {
                Ok(descriptor) => {
                    let mut response = json!({
                        "component":descriptor,
                        "static_validation":"passed",
                        "runtime_import":"not_run"
                    });
                    response["dependency_resolution"] =
                        match self.components.resolve_dependencies(&root, id) {
                            Ok(resolution) => json!({"status":"resolved","report":resolution}),
                            Err(error) => json!({"status":"unresolved","reason":error}),
                        };
                    if descriptor["manifest"]["kind"] == "layer-edit-provider" {
                        let component_id = descriptor["manifest"]["component_id"]
                            .as_str()
                            .unwrap_or_default();
                        let version = descriptor["manifest"]["component_version"]
                            .as_str()
                            .unwrap_or_default();
                        let verification =
                            self.components
                                .component_verification(&root, component_id, version);
                        response["edit_verification"] = match verification {
                            Ok(Some(record))
                                if record["manifest_sha256"] == descriptor["manifest_sha256"] =>
                            {
                                json!({
                                    "status":"records_present",
                                    "payload_count":record["verifications"].as_array().map_or(0, Vec::len),
                                    "runtime_recheck":"manifest/payload/provider+fixture source/Python/PyTorch"
                                })
                            }
                            Ok(Some(_)) => json!({"status":"stale"}),
                            Ok(None) => json!({"status":"unverified","payload_count":0}),
                            Err(error) => json!({"status":"unreadable","reason":error}),
                        };
                    }
                    Report::ok("component_show", response)
                }
                Err(error) => Report::error(
                    "component_show",
                    "COMPONENT_INVALID",
                    error,
                    "确认 component ID 和 manifest 字段；静态检查不会导入或执行 Python 代码。",
                ),
            },
            ComponentCommand::Check { id } => match self.components.inspect(&root, id) {
                Ok(descriptor) => match self.components.resolve_dependencies(&root, id) {
                    Ok(dependencies) => Report::ok(
                        "component_check",
                        json!({
                            "component":descriptor,
                            "static_validation":"passed",
                            "dependency_resolution":"passed",
                            "dependencies":dependencies,
                            "runtime_import":"not_run"
                        }),
                    ),
                    Err(error) => Report::error(
                        "component_check",
                        "COMPONENT_DEPENDENCY_RESOLUTION_FAILED",
                        error,
                        "补齐必需组件版本或统一依赖范围；可用 m-model component list 查看本地版本。",
                    ),
                },
                Err(error) => Report::error(
                    "component_check",
                    "COMPONENT_INVALID",
                    error,
                    "确认 component ID 和 manifest 字段；静态检查不会导入或执行 Python 代码。",
                ),
            },
            ComponentCommand::Test {
                id,
                params,
                batch,
                batch_params,
                batch_binding,
            } => {
                let requested_params = match parameters(params) {
                    Ok(params) => params,
                    Err(error) => {
                        return Report::error(
                            "component_test",
                            "COMPONENT_PARAMS_INVALID",
                            error,
                            "使用 --param NAME=JSON；省略值会采用组件 Schema 默认值。",
                        );
                    }
                };
                let raw_descriptor = match self.components.inspect(&root, id) {
                    Ok(descriptor) => descriptor,
                    Err(error) => {
                        return Report::error(
                            "component_test",
                            "COMPONENT_RESOLUTION_FAILED",
                            error,
                            "确认组件 ID 和版本；component test 只运行已有对应检查器的组件类型。",
                        );
                    }
                };
                let kind = raw_descriptor["manifest"]["kind"].as_str().unwrap_or("");
                if matches!(
                    kind,
                    "layer-edit-provider" | "transform" | "dataset" | "loss" | "metric" | "trainer"
                ) && batch.is_some()
                {
                    return Report::error(
                        "component_test",
                        "COMPONENT_BATCH_UNSUPPORTED",
                        format!(
                            "{kind} 的 conformance_fixture 自行创建测试输入，不能同时传入 --batch。"
                        ),
                        "在 fixture 中构造组件需要的输入上下文。",
                    );
                }
                let descriptor = match kind {
                    "layer" => self.components.resolve_layer(&root, id, &requested_params),
                    "layer-edit-provider" => {
                        self.components
                            .resolve_layer_edit_provider(&root, id, &requested_params)
                    }
                    "transform" => self
                        .components
                        .resolve_transform(&root, id, &requested_params),
                    "dataset" | "loss" | "metric" | "trainer" => self
                        .components
                        .resolve_component_for_test(&root, id, kind, &requested_params),
                    _ => Err(format!(
                        "component test 当前支持 layer、layer-edit-provider、transform、dataset、loss、metric 和 trainer；组件 {id} 的 kind 为 {kind}。"
                    )),
                };
                let descriptor = match descriptor {
                    Ok(descriptor) => descriptor,
                    Err(error) => {
                        return Report::error(
                            "component_test",
                            "COMPONENT_RESOLUTION_FAILED",
                            error,
                            "检查参数 Schema、组件类型和所需的 conformance_fixture。",
                        );
                    }
                };
                let manifest = &descriptor["manifest"];
                let reference = match component_reference(&root, &descriptor) {
                    Ok(reference) => reference,
                    Err(error) => {
                        return Report::error(
                            "component_test",
                            "COMPONENT_REFERENCE_INVALID",
                            error,
                            "组件运行测试前必须能够固定项目内组件清单的版本和哈希。",
                        );
                    }
                };
                let batch_params = match parameters(batch_params) {
                    Ok(params) => params,
                    Err(error) => {
                        return Report::error(
                            "component_test",
                            "BATCH_PARAMS_INVALID",
                            error,
                            "批次参数使用 --batch-param NAME=JSON。",
                        );
                    }
                };
                let binding = batch_binding.as_ref().map(|binding| match binding {
                    BatchBinding::Envelope => "envelope",
                    BatchBinding::Single => "single",
                    BatchBinding::Args => "args",
                    BatchBinding::Kwargs => "kwargs",
                });
                let request = json!({
                    "protocol_version":PROTOCOL_VERSION,
                    "operation":"component_test",
                    "project_root":root,
                    "component":manifest["entrypoint"],
                    "component_ref":reference,
                    "component_manifest":manifest,
                    "params":if kind == "layer-edit-provider" {
                        descriptor["resolved_payload"].clone()
                    } else {
                        descriptor["resolved_params"].clone()
                    },
                    "batch":batch,
                    "batch_params":batch_params,
                    "batch_binding":binding,
                    "seed":cli.seed.unwrap_or(42),
                    "runtime":{
                        "python_executable":cli.python.as_deref().unwrap_or("python3"),
                        "timeout_seconds":cli.timeout.unwrap_or(30)
                    }
                });
                let mut report = self.backend.execute(&request);
                if report.success() {
                    report.data["component"] = json!({
                        "id":manifest["component_id"],
                        "version":manifest["component_version"],
                        "kind":manifest["kind"],
                        "manifest_sha256":descriptor["manifest_sha256"],
                        "entrypoint":manifest["entrypoint"]
                    });
                    report.data["static_validation"] = json!("passed");
                    if kind == "layer-edit-provider" {
                        if report.data["activation_eligible"] == true {
                            let recorded_at_unix = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            let verification = json!({
                                "component_id":manifest["component_id"],
                                "component_version":manifest["component_version"],
                                "manifest_sha256":descriptor["manifest_sha256"],
                                "capability":"layer_edit",
                                "status":"verified",
                                "payload_sha256":report.data["verification_evidence"]["payload_sha256"],
                                "suite_version":report.data["verification_evidence"]["suite_version"],
                                "source_fingerprints":report.data["verification_evidence"]["source_fingerprints"],
                                "python_version":report.data["verification_evidence"]["python_version"],
                                "torch_version":report.data["verification_evidence"]["torch_version"],
                                "coverage":report.data["coverage"],
                                "recorded_at_unix":recorded_at_unix
                            });
                            let component_id =
                                manifest["component_id"].as_str().unwrap_or_default();
                            let version =
                                manifest["component_version"].as_str().unwrap_or_default();
                            if let Err(error) = self.components.record_component_verification(
                                &root,
                                component_id,
                                version,
                                &verification,
                            ) {
                                return Report::error(
                                    "component_test",
                                    "COMPONENT_VERIFICATION_STORE_FAILED",
                                    error,
                                    "测试本身通过，但可执行性记录没有写入；检查 .model-maven/component-verifications 的权限后重试。",
                                );
                            }
                            report.data["activation"] = json!({
                                "status":"enabled_for_tested_payloads",
                                "recorded_at_unix":recorded_at_unix
                            });
                        } else {
                            report.data["activation"] = json!({
                                "status":"not_enabled",
                                "reason":report.data["activation_reason"]
                            });
                        }
                    }
                }
                report
            }
            ComponentCommand::Add {
                id,
                kind,
                entrypoint,
                version,
                description,
                params_schema,
                capabilities,
                conformance_fixture,
                dependencies,
                required_dependencies,
                optional_dependencies,
            } => {
                let Some((namespace, _)) = id.split_once('/') else {
                    return Report::error(
                        "component_add",
                        "COMPONENT_ID_INVALID",
                        "组件 ID 必须是 namespace/name。",
                        "例如 local/swish。",
                    );
                };
                let parse_optional = |value: Option<&String>,
                                      label: &str,
                                      fallback: Value|
                 -> Result<Value, String> {
                    match value {
                        Some(value) => serde_json::from_str(value)
                            .map_err(|error| format!("{label} JSON 无效：{error}")),
                        None => Ok(fallback),
                    }
                };
                let params_schema = match parse_optional(
                    params_schema.as_ref(),
                    "params-schema",
                    json!({"type":"object","properties":{},"required":[],"additionalProperties":false}),
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        return Report::error(
                            "component_add",
                            "COMPONENT_MANIFEST_INVALID",
                            error,
                            "传入有效 JSON Schema 子集。",
                        );
                    }
                };
                let capabilities = match parse_optional(
                    capabilities.as_ref(),
                    "capabilities",
                    json!({"construct":{"state":"supported"}}),
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        return Report::error(
                            "component_add",
                            "COMPONENT_MANIFEST_INVALID",
                            error,
                            "传入有效 JSON 能力对象。",
                        );
                    }
                };
                let mut dependencies = match parse_optional(
                    dependencies.as_ref(),
                    "dependencies",
                    json!([]),
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        return Report::error(
                            "component_add",
                            "COMPONENT_MANIFEST_INVALID",
                            error,
                            "传入依赖对象数组，例如 [{\"component_id\":\"demo/kernel\",\"version\":\"^1.2\"}]。",
                        );
                    }
                };
                let Some(dependency_list) = dependencies.as_array_mut() else {
                    return Report::error(
                        "component_add",
                        "COMPONENT_MANIFEST_INVALID",
                        "dependencies 必须是 JSON 数组。",
                        "使用 --dependency ID@RANGE 或传入 JSON 依赖数组。",
                    );
                };
                for (arguments, optional) in [
                    (required_dependencies.as_slice(), false),
                    (optional_dependencies.as_slice(), true),
                ] {
                    for argument in arguments {
                        match component_dependency_argument(argument, optional) {
                            Ok(dependency) => dependency_list.push(dependency),
                            Err(error) => {
                                return Report::error(
                                    "component_add",
                                    "COMPONENT_MANIFEST_INVALID",
                                    error,
                                    "例如 --dependency 'demo/kernel@^1.2.0'；可选依赖使用 --optional-dependency。",
                                );
                            }
                        }
                    }
                }
                let mut manifest = json!({
                    "schema_version":1,
                    "namespace":namespace,
                    "component_id":id,
                    "component_version":version,
                    "extension_api_version":1,
                    "kind":kind,
                    "entrypoint":entrypoint,
                    "description":description,
                    "params_schema":params_schema,
                    "capabilities":capabilities,
                    "dependencies":dependencies
                });
                if let Some(conformance_fixture) = conformance_fixture {
                    manifest["conformance_fixture"] = json!(conformance_fixture);
                }
                match self.components.register(&root, &manifest) {
                    Ok(component) => Report::ok("component_add", json!({"component":component})),
                    Err(error) => Report::error(
                        "component_add",
                        "COMPONENT_REGISTER_FAILED",
                        error,
                        "注册只保存静态描述，不安装依赖或运行入口；先检查 Python 包在当前环境中可用。",
                    ),
                }
            }
        }
    }
}
