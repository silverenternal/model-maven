use crate::{
    cli::*,
    domain::{PROTOCOL_VERSION, Report, worker::WorkerRequest},
    ports::{
        ArtifactExporter, Backend, ComponentResolver, ComponentScaffolder, PreviewRepository,
        PreviewRepositoryError, ProjectConfigStore, ProjectConfigUpdateError, ProjectSourceWriter,
        RunRepository, RunSupervisor, SourceEditLock, SourceEditRepository, TrainingExportRequest,
        VariantLock, VariantRepository, services::ExecutionProgress,
    },
};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{cell::RefCell, collections::BTreeMap};

mod artifacts;
mod augment;
pub(crate) mod classification;
mod config;
mod data;
mod data_preview;
mod discovery;
mod edit;
mod environment;
mod evaluation;
mod extensions;
mod generic_loop;
mod graph;
mod graph_delivery_journal;
mod graph_edit;
mod model;
mod readiness;
mod run;
mod run_inputs;
mod run_lifecycle;
mod run_prepare;
mod run_query;
mod run_request;
mod scaffold;
pub(crate) use scaffold::{scaffold_encoder_layer_catalog, normalize_training_command};
mod train;

#[cfg(test)]
mod replay_tests;

#[derive(Clone, Copy, Default)]
pub struct ExecutionContext<'a> {
    pub require_assembly_confirmation: bool,
    pub show_progress: bool,
    pub confirm_assembly: bool,
    pub confirmed_assembly_revision: Option<&'a str>,
    pub progress_observer: Option<&'a dyn ExecutionProgress>,
}

pub struct Application<B> {
    pub backend: B,
    pub runtime_capabilities: RefCell<Option<crate::domain::capability::CapabilityCatalog>>,
    pub runtime_declarations: RefCell<Option<crate::domain::capability::CapabilityCatalog>>,
    pub variants: Box<dyn VariantRepository>,
    pub runs: Box<dyn RunRepository>,
    pub exporter: Box<dyn ArtifactExporter>,
    pub supervisor: Box<dyn RunSupervisor>,
    pub components: Box<dyn ComponentResolver>,
    pub scaffolder: Box<dyn ComponentScaffolder>,
    pub source_edits: Box<dyn SourceEditRepository>,
    pub project_config: Box<dyn ProjectConfigStore>,
    pub project_source_writer: Box<dyn ProjectSourceWriter>,
    pub previews: Box<dyn PreviewRepository>,
}

struct Job {
    operation: &'static str,
    inputs: Inputs,
    config_revision: Option<String>,
    input_sources: Map<String, Value>,
    path: Option<String>,
    layer_type: Option<String>,
    search: Option<String>,
    call: String,
    contract: Option<PathBuf>,
    if_contract_sha256: Option<String>,
    batch_fingerprint: Option<String>,
    probe_device: Option<String>,
    local_probe: bool,
    loss: Option<String>,
    training_step: Option<String>,
    training_loop: Option<String>,
    training_mode: Option<&'static str>,
    training_params: Map<String, Value>,
    apply_supervision: bool,
    expected_supervision_revision: Option<String>,
    repeat_of: Option<String>,
    parent_run_id: Option<String>,
    variant_snapshot: Option<Value>,
    variant_device_policy: Option<String>,
    variant_base_rng_devices: Option<Vec<String>>,
    batch_device_policy: Option<String>,
    model_initialization: Option<String>,
    model_initialization_seed: Option<u64>,
    expected_source_manifest: Option<Value>,
    expected_runtime: Option<Value>,
    repeat_overrides: Vec<String>,
    runtime: Value,
    seed: Option<u64>,
    checkpoint_every: Option<u32>,
    checkpoint_input: Option<String>,
    expected_checkpoint_sha256: Option<String>,
    migration_checkpoint_input: Option<String>,
    migration_plan: Option<Value>,
    allow_state_reset: bool,
    steps: Option<u32>,
    optimizer: Option<String>,
    optimizer_factory: Option<String>,
    optimizer_params: Map<String, Value>,
    lr: Option<f64>,
    weight_decay: Option<f64>,
    edit: Option<Value>,
    dry_run: bool,
    require_forward: bool,
    layer_context: bool,
    detach: bool,
    run_name: Option<String>,
    request_id: Option<String>,
}

struct InitOptions<'a> {
    model: Option<&'a String>,
    batch: Option<&'a String>,
    injections: &'a [String],
    injection_params: &'a [String],
    batch_binding: Option<&'a BatchBinding>,
    batch_params: &'a [String],
    dry_run: bool,
}

fn empty_job() -> Job {
    Job {
        operation: "inspect",
        inputs: Inputs::default(),
        config_revision: None,
        input_sources: Map::new(),
        path: None,
        layer_type: None,
        search: None,
        call: "all".into(),
        contract: None,
        if_contract_sha256: None,
        batch_fingerprint: None,
        probe_device: None,
        local_probe: false,
        loss: None,
        training_step: None,
        training_loop: None,
        training_mode: None,
        training_params: Map::new(),
        apply_supervision: false,
        expected_supervision_revision: None,
        repeat_of: None,
        parent_run_id: None,
        variant_snapshot: None,
        variant_device_policy: None,
        variant_base_rng_devices: None,
        batch_device_policy: None,
        model_initialization: None,
        model_initialization_seed: None,
        expected_source_manifest: None,
        expected_runtime: None,
        repeat_overrides: Vec::new(),
        runtime: json!({"python_executable":null,"timeout_seconds":30}),
        seed: None,
        checkpoint_every: None,
        checkpoint_input: None,
        expected_checkpoint_sha256: None,
        migration_checkpoint_input: None,
        migration_plan: None,
        allow_state_reset: false,
        steps: None,
        optimizer: None,
        optimizer_factory: None,
        optimizer_params: Map::new(),
        lr: None,
        weight_decay: None,
        edit: None,
        dry_run: false,
        require_forward: false,
        layer_context: false,
        detach: false,
        run_name: None,
        request_id: None,
    }
}

fn collect_component_references(edit: &Value, references: &mut BTreeMap<String, Value>) {
    if edit["action"] == "transaction" {
        if let Some(operations) = edit["operations"].as_array() {
            for operation in operations {
                collect_component_references(operation, references);
            }
        }
        return;
    }
    if let Some(reference) = edit.get("component_ref").filter(|value| value.is_object()) {
        collect_component_reference(reference, references);
    }
}

fn collect_component_reference(reference: &Value, references: &mut BTreeMap<String, Value>) {
    let key = format!(
        "{}@{}",
        reference["id"].as_str().unwrap_or(""),
        reference["version"].as_str().unwrap_or("")
    );
    references.entry(key).or_insert_with(|| reference.clone());
    if let Some(dependencies) = reference["dependencies"].as_array() {
        for dependency in dependencies {
            let key = format!(
                "{}@{}",
                dependency["id"].as_str().unwrap_or(""),
                dependency["version"].as_str().unwrap_or("")
            );
            references.entry(key).or_insert_with(|| dependency.clone());
        }
    }
}

fn active_extension_manifests(
    registry: &dyn ComponentResolver,
    root: &Path,
    run: &Value,
) -> Result<Vec<Value>, String> {
    let mut references = BTreeMap::new();
    if let Some(injections) = run["inputs"]["model_injections"].as_object() {
        for injection in injections.values() {
            if let Some(reference) = injection
                .get("component_ref")
                .filter(|value| value.is_object())
            {
                collect_component_reference(reference, &mut references);
            }
        }
    }
    if let Some(variant) = run["inputs"]
        .get("variant")
        .filter(|value| value.is_object())
    {
        if let Some(injections) = variant["origin"]["model_injections"].as_object() {
            for injection in injections.values() {
                if let Some(reference) = injection
                    .get("component_ref")
                    .filter(|value| value.is_object())
                {
                    collect_component_reference(reference, &mut references);
                }
            }
        }
        let edits = variant["edits"]
            .as_array()
            .ok_or("运行中的模型变体缺少编辑历史。")?;
        let cursor = variant["cursor"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("运行中的模型变体编辑位置无效。")?;
        if cursor > edits.len() {
            return Err("运行中的模型变体编辑位置超过历史长度。".into());
        }
        for edit in edits.iter().take(cursor) {
            collect_component_references(edit, &mut references);
        }
    }
    let mut manifests = Vec::new();
    for (id, reference) in references {
        let descriptor = registry.inspect(root, &id)?;
        if descriptor["manifest_sha256"] != reference["sha256"]
            || descriptor["manifest"]["component_id"] != reference["id"]
            || descriptor["manifest"]["component_version"] != reference["version"]
        {
            return Err(format!("变体固定的扩展清单已变化：{id}。"));
        }
        if descriptor["manifest"]["package_source"].is_object() {
            manifests.push(json!({
                "manifest":{
                    "component_id":descriptor["manifest"]["component_id"],
                    "component_version":descriptor["manifest"]["component_version"],
                    "entrypoint":descriptor["manifest"]["entrypoint"],
                    "conformance_fixture":descriptor["manifest"]["conformance_fixture"],
                    "package_source":descriptor["manifest"]["package_source"]
                },
                "descriptor_sha256":descriptor["manifest_sha256"]
            }));
        }
    }
    Ok(manifests)
}

struct ComponentResolutionError {
    code: &'static str,
    message: String,
}

impl ComponentResolutionError {
    fn generic(message: impl Into<String>) -> Self {
        Self {
            code: "COMPONENT_RESOLUTION_FAILED",
            message: message.into(),
        }
    }
}

impl From<String> for ComponentResolutionError {
    fn from(message: String) -> Self {
        Self::generic(message)
    }
}

impl From<&str> for ComponentResolutionError {
    fn from(message: &str) -> Self {
        Self::generic(message)
    }
}

fn require_python<'a>(cli: &'a Cli, operation: &'static str) -> Result<&'a str, Report> {
    cli.python.as_deref().filter(|value| !value.trim().is_empty()).ok_or_else(|| {
        Report::error(operation, "PYTHON_SELECTION_REQUIRED", "未发现或指定 Python 解释器。",
            "使用 --python 明确选择解释器后重试；操作尚未启动。")
    })
}

fn parameters(values: &[String]) -> Result<Map<String, Value>, String> {
    let mut result = Map::new();
    for value in values {
        let (name, encoded) = value
            .split_once('=')
            .ok_or("参数应为 NAME=JSON，例如 p=0.2。")?;
        if name.is_empty() || result.contains_key(name) {
            return Err(format!("参数名为空或重复：{name}"));
        }
        let parsed =
            serde_json::from_str(encoded).map_err(|e| format!("{name} 不是有效 JSON：{e}"))?;
        result.insert(name.into(), parsed);
    }
    Ok(result)
}

fn python_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars.next().is_some_and(|ch| ch == '_' || unicode_ident::is_xid_start(ch))
        && chars.all(|ch| unicode_ident::is_xid_continue(ch))
}

fn python_entrypoint(value: &str) -> bool {
    let Some((module, callable)) = value.split_once(':') else {
        return false;
    };
    !module.is_empty()
        && !callable.is_empty()
        && !value.contains(char::is_whitespace)
        && module.split('.').all(python_identifier)
        && callable.split('.').all(python_identifier)
}

fn injection_parameters(values: &[String]) -> Result<BTreeMap<String, Map<String, Value>>, String> {
    let mut result = BTreeMap::new();
    for value in values {
        let (target, encoded) = value
            .split_once('=')
            .ok_or("注入参数应为 SLOT.NAME=JSON，例如 encoder.depth=50。")?;
        let (slot, name) = target
            .split_once('.')
            .ok_or("注入参数应指定接缝和参数名：SLOT.NAME=JSON。")?;
        if !python_identifier(slot) || !python_identifier(name) {
            return Err(format!(
                "注入参数名无效：{target}；接缝和参数都必须是 Python 标识符。"
            ));
        }
        let parsed: Value = serde_json::from_str(encoded)
            .map_err(|error| format!("注入参数 {target} 不是有效 JSON：{error}"))?;
        if result
            .entry(slot.to_owned())
            .or_insert_with(Map::new)
            .insert(name.to_owned(), parsed)
            .is_some()
        {
            return Err(format!("重复指定注入参数：{target}"));
        }
    }
    Ok(result)
}

fn model_injections(
    values: &[String],
    parameter_values: &[String],
    registry: &dyn ComponentResolver,
    root: &Path,
) -> Result<Map<String, Value>, String> {
    let mut supplied_params = injection_parameters(parameter_values)?;
    let mut result = Map::new();
    for value in values {
        let (slot, factory) = value
            .split_once('=')
            .ok_or("接缝注入格式应为 SLOT=MODULE:FACTORY 或 SLOT=component:ID@VERSION。")?;
        if !python_identifier(slot) {
            return Err(format!("注入接缝名必须是 Python 标识符：{slot}"));
        }
        if result.contains_key(slot) {
            return Err(format!("重复指定模型接缝：{slot}"));
        }
        let params = supplied_params.remove(slot).unwrap_or_default();
        if let Some(component) = factory.strip_prefix("component:") {
            if component.split_once('@').is_none() {
                return Err(format!(
                    "注册组件必须固定版本：component:{component}@VERSION。"
                ));
            }
            let descriptor = registry.resolve_layer(root, component, &params)?;
            let reference = component_reference(root, &descriptor)?;
            result.insert(
                slot.into(),
                json!({
                    "factory":descriptor["manifest"]["entrypoint"],
                    "params":descriptor["resolved_params"],
                    "component_ref":reference
                }),
            );
            continue;
        }
        if !python_entrypoint(factory) {
            return Err(format!(
                "接缝 {slot} 的工厂入口无效：{factory}；格式为 package.module:factory 或 component:ID@VERSION。"
            ));
        }
        result.insert(slot.into(), if params.is_empty() { json!(factory) }
            else { json!({"factory":factory, "params":params}) });
    }
    if let Some(slot) = supplied_params.keys().next() {
        return Err(format!("--inject-param 引用了未声明的接缝：{slot}。"));
    }
    if result.len() > 32 {
        return Err("单个模型最多声明 32 个组件注入接缝。".into());
    }
    Ok(result)
}

fn saved_model_injections(value: &Value) -> Value {
    value
        .get("model_injections")
        .filter(|injections| injections.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}))
}

fn injections_changed(
    previous: &Value,
    overrides: &[String],
    parameter_values: &[String],
    registry: &dyn ComponentResolver,
    root: &Path,
) -> Result<bool, String> {
    if overrides.is_empty() && parameter_values.is_empty() {
        return Ok(false);
    }
    Ok(
        model_injections(overrides, parameter_values, registry, root)?
            != previous.as_object().cloned().unwrap_or_default(),
    )
}

fn source_transaction_supported(operations: &Value) -> bool {
    let Some(operations) = operations.as_array() else {
        return false;
    };
    if !(1..=64).contains(&operations.len()) {
        return false;
    }
    let all_actions = |allowed: &[&str]| {
        operations.iter().all(|operation| {
            operation["action"]
                .as_str()
                .is_some_and(|action| allowed.contains(&action))
        })
    };
    if all_actions(&["set"]) || all_actions(&["add", "replace", "remove"]) {
        return true;
    }
    let mut seen_structure = false;
    operations.iter().all(|operation| match operation["action"].as_str() {
        Some("set") if !seen_structure => true,
        Some("add" | "replace" | "remove") => { seen_structure = true; true },
        _ => false,
    })
}

fn secret_key(name: &str) -> bool {
    let name = name.to_ascii_lowercase().replace('-', "_");
    matches!(
        name.as_str(),
        "token" | "api_key" | "apikey" | "access_token" | "refresh_token" | "client_secret"
    ) || name.contains("password")
        || name.contains("secret")
        || name.contains("credential")
        || name.contains("authorization")
        || name.contains("private_key")
        || name.ends_with("_token")
        || name.ends_with("_api_key")
        || name.ends_with("_access_key")
}

fn redact_run_inputs(value: &mut Value, path: &str, fields: &mut Vec<String>) {
    match value {
        Value::Object(values) => {
            for (key, child) in values {
                let child_path = format!("{path}.{key}");
                if secret_key(key) {
                    *child = json!("[REDACTED]");
                    fields.push(child_path);
                } else {
                    redact_run_inputs(child, &child_path, fields);
                }
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter_mut().enumerate() {
                redact_run_inputs(child, &format!("{path}[{index}]"), fields);
            }
        }
        _ => {}
    }
}

fn snapshot_params(
    snapshot: &Value,
    key: &str,
    overrides: &[String],
) -> Result<Map<String, Value>, String> {
    let mut values = snapshot
        .get(key)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    values.extend(parameters(overrides)?);
    Ok(values)
}

fn params_changed(snapshot: &Value, key: &str, overrides: &[String]) -> Result<bool, String> {
    let previous = snapshot
        .get(key)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    Ok(parameters(overrides)?
        .into_iter()
        .any(|(name, value)| previous.get(&name) != Some(&value)))
}

fn cli_params(values: Map<String, Value>) -> Vec<String> {
    values
        .into_iter()
        .map(|(name, value)| {
            format!(
                "{name}={}",
                serde_json::to_string(&value).expect("JSON values serialize")
            )
        })
        .collect()
}

fn has_redacted_value(value: &Value) -> bool {
    match value {
        Value::String(value) => value == "[REDACTED]",
        Value::Array(values) => values.iter().any(has_redacted_value),
        Value::Object(values) => values.values().any(has_redacted_value),
        _ => false,
    }
}

fn resolve_component_operation(
    registry: &dyn ComponentResolver,
    root: &Path,
    operation: &mut Value,
) -> Result<(), ComponentResolutionError> {
    if operation["action"] == "transaction" {
        let operations = operation["operations"]
            .as_array_mut()
            .ok_or("原子 Layer 计划缺少 operations 数组。")?;
        for child in operations {
            resolve_component_operation(registry, root, child)?;
        }
        return Ok(());
    }
    if operation["action"] == "custom" {
        let Some(provider) = operation["provider"].as_str().map(str::to_owned) else {
            return Err("自定义 Layer 编辑缺少 provider 入口。".into());
        };
        if provider.contains(':') {
            if operation.get("component_ref").is_some() {
                return Err("直接 Python provider 不能携带注册组件的 component_ref。".into());
            }
            return Ok(());
        }
        if operation.get("component_ref").is_some() {
            return Err("component_ref 是 CLI 解析生成的内部字段，不能手工提交。".into());
        }
        let payload = match operation.get("payload") {
            None => Map::new(),
            Some(Value::Object(payload)) => payload.clone(),
            Some(_) => return Err("自定义 provider 的 payload 必须是 JSON 对象。".into()),
        };
        let descriptor = registry.resolve_layer_edit_provider(root, &provider, &payload)?;
        let manifest = &descriptor["manifest"];
        let mut reference = component_reference(root, &descriptor)?;
        let component_id = manifest["component_id"]
            .as_str()
            .ok_or("Layer Edit Provider manifest 缺少 component_id。")?;
        let version = manifest["component_version"]
            .as_str()
            .ok_or("Layer Edit Provider manifest 缺少 component_version。")?;
        let capability = &manifest["capabilities"]["layer_edit"];
        let capability_state = capability
            .as_str()
            .or_else(|| capability["state"].as_str())
            .unwrap_or("unknown");
        if capability_state != "supported" {
            return Err(ComponentResolutionError {
                code: if capability_state == "conditional" {
                    "COMPONENT_PROVIDER_CONDITION_UNVERIFIED"
                } else {
                    "COMPONENT_PROVIDER_UNSUPPORTED"
                },
                message: format!(
                    "Provider {component_id}@{version} 的 layer_edit 状态是 {capability_state}，当前模型/条件没有对应的可执行性证据。"
                ),
            });
        }
        let verification = registry
            .component_verification(root, component_id, version)?
            .ok_or_else(|| {
                ComponentResolutionError {
                    code: "COMPONENT_PROVIDER_UNVERIFIED",
                    message: format!(
                        "Provider {component_id}@{version} 尚无匹配的通过记录；先对本次 payload 运行 m-model component test。"
                    ),
                }
            })?;
        if verification["manifest_sha256"] != descriptor["manifest_sha256"]
            || verification["verifications"]
                .as_array()
                .is_none_or(Vec::is_empty)
        {
            return Err(ComponentResolutionError {
                code: "COMPONENT_PROVIDER_VERIFICATION_STALE",
                message: format!(
                    "Provider {component_id}@{version} 的验证记录已过期或为空；重新运行 m-model component test。"
                ),
            });
        }
        reference["verification"] = verification;
        operation["provider"] = manifest["entrypoint"].clone();
        operation["payload"] = descriptor["resolved_payload"].clone();
        operation["component_ref"] = reference;
        return Ok(());
    }
    let Some(component) = operation["component"].as_str().map(str::to_owned) else {
        if operation.get("component_ref").is_some() {
            return Err("component_ref 是 CLI 解析生成的内部字段，不能手工提交。".into());
        }
        return Ok(());
    };
    if component.contains(':') {
        if operation.get("component_ref").is_some() {
            return Err("直接 Python callable 不能携带注册组件的 component_ref。".into());
        }
        return Ok(());
    }
    if !matches!(operation["action"].as_str(), Some("add" | "replace")) {
        return Err("注册组件只能用于 add 或 replace 操作。".into());
    }
    let params = match operation.get("params") {
        None => Map::new(),
        Some(Value::Object(params)) => params.clone(),
        Some(_) => return Err("组件参数必须是 JSON 对象。".into()),
    };
    let descriptor = registry.resolve_layer(root, &component, &params)?;
    let manifest = &descriptor["manifest"];
    let reference = component_reference(root, &descriptor)?;
    operation["component"] = json!(manifest["entrypoint"]);
    operation["params"] = descriptor["resolved_params"].clone();
    operation["component_ref"] = reference;
    Ok(())
}

fn component_reference(root: &Path, descriptor: &Value) -> Result<Value, String> {
    let manifest_path = Path::new(
        descriptor["manifest_path"]
            .as_str()
            .ok_or("组件注册表未返回清单路径。")?,
    );
    let relative = manifest_path
        .strip_prefix(root)
        .map_err(|_| "组件清单必须位于当前项目目录中。")?;
    if !relative.starts_with(".model-maven/components") {
        return Err("组件清单路径不在项目组件目录内。".into());
    }
    let dependencies = descriptor["resolved_dependencies"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    Ok(json!({
        "id":descriptor["manifest"]["component_id"],
        "version":descriptor["manifest"]["component_version"],
        "path":relative.to_string_lossy().replace('\\', "/"),
        "sha256":descriptor["manifest_sha256"],
        "dependencies":dependencies
    }))
}

fn verify_saved_model_injections(
    value: &Value,
    registry: &dyn ComponentResolver,
    root: &Path,
) -> Result<(), String> {
    let Some(injections) = value.as_object() else {
        return if value.is_null() {
            Ok(())
        } else {
            Err("已保存的模型接缝映射必须是 JSON 对象。".into())
        };
    };
    for (slot, specification) in injections {
        if specification.is_string() {
            continue;
        }
        if specification.get("component_ref").is_none() {
            if !specification["factory"].as_str().is_some_and(python_entrypoint)
                || !specification["params"].is_object()
                || specification.as_object().is_none_or(|object| object.keys().any(|key| key != "factory" && key != "params"))
            {
                return Err(format!("接缝 {slot} 的直接工厂或参数无效。"));
            }
            continue;
        }
        let component_id = specification["component_ref"]["id"]
            .as_str()
            .ok_or_else(|| format!("接缝 {slot} 的组件锁缺少组件 ID。"))?;
        let version = specification["component_ref"]["version"]
            .as_str()
            .ok_or_else(|| format!("接缝 {slot} 的组件锁缺少精确版本。"))?;
        let params = specification["params"]
            .as_object()
            .ok_or_else(|| format!("接缝 {slot} 的已解析参数必须是对象。"))?;
        let descriptor =
            registry.resolve_layer(root, &format!("{component_id}@{version}"), params)?;
        let current_reference = component_reference(root, &descriptor)?;
        if current_reference != specification["component_ref"] {
            return Err(format!(
                "接缝 {slot} 固定的组件清单或传递依赖已变化：{component_id}@{version}。"
            ));
        }
        if descriptor["manifest"]["entrypoint"] != specification["factory"]
            || descriptor["resolved_params"] != specification["params"]
        {
            return Err(format!(
                "接缝 {slot} 的组件入口或参数与注册 Schema 解析结果不一致。"
            ));
        }
    }
    Ok(())
}

fn component_dependency_argument(value: &str, optional: bool) -> Result<Value, String> {
    let Some((id, version)) = value.split_once('@') else {
        return Err(format!("依赖应写为 namespace/name@SEMVER_RANGE：{value}"));
    };
    if id.trim().is_empty() || version.trim().is_empty() || version.contains('@') {
        return Err(format!("依赖应写为 namespace/name@SEMVER_RANGE：{value}"));
    }
    Ok(json!({
        "component_id":id.trim(),
        "version":version.trim(),
        "optional":optional
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplicate_and_non_json_parameters_are_rejected() {
        assert!(parameters(&["p=0.2".into(), "p=0.3".into()]).is_err());
        assert!(parameters(&["p=__import__('os')".into()]).is_err());
        assert_eq!(parameters(&["p=0.2".into()]).unwrap()["p"], json!(0.2));
    }
}
