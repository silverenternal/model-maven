//! Versioned worker compatibility and truthful runtime capability declarations.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const WORKER_API_VERSION: u32 = 1;
pub const WORKER_PROTOCOL_VERSION: u32 = 1;

// Capabilities that the application readiness planner may require. Older or
// incomplete worker catalogs must represent omitted entries as unknown before
// the readiness layer resolves them.
const READINESS_CAPABILITY_IDS: &[&str] = &[
    "data.dataset_probe",
    "data.preview",
    "data.preview_runtime_paths_v1",
    "data.raw_batch_binding_v1",
    "data.runtime_sampling_seed_v1",
    "edit.source_factory_device_v1",
    "edit.assembly_codegen",
    "edit.assembly_codegen_runtime_paths_v1",
    "edit.assembly_python_rng_v1",
    "evaluation.model_device_v1",
    "extensions.component_conformance",
    "model.contract_identity_v1",
    "model.contract_port_paths_v2",
    "model.contract_preload_v1",
    "model.direct_factory_params_v1",
    "model.forward_probe",
    "model.inspect",
    "model.layer_edit",
    "model.loss_probe",
    "model.probe_device_v1",
    "model.runtime_rng_v1",
    "model.semantic_device_v1",
    "model.semantic_marks",
    "model.variant_factory_device_v1",
    "model.variant_layer_rng_v1",
    "model.variant_base_rng_v1",
    "supervision.port_path_v2",
    "supervision.preview_device_v1",
    "supervision.root_mark_v1",
    "runtime.project_import_roots",
    "runtime.user_working_directory",
    "training.batch_device_policy_v1",
    "training.checkpoint_resume",
    "training.detached",
    "training.external_loop",
    "training.evaluation",
    "training.managed_step",
    "training.manual_step",
    "training.supervision_apply",
    "training.supervision_preview",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityAvailability {
    Supported,
    Unsupported,
    Conditional,
    Unknown,
}

/// Where a capability claim came from. Runtime declarations describe what the
/// selected worker/environment can attempt; they do not establish behavior of
/// a particular model instance or input.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityScope {
    RuntimeEnvironment,
    ModelInstance,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityDescriptor {
    pub id: String,
    pub state: CapabilityAvailability,
    /// Older workers omit scope. Keep it absent rather than implying that the
    /// declaration is model-instance evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<CapabilityScope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityCatalog {
    pub worker_protocol_version: u32,
    pub worker_api_version: u32,
    pub state_formats: BTreeMap<String, u32>,
    pub capabilities: Vec<CapabilityDescriptor>,
}

impl CapabilityCatalog {
    /// Return the declared state, treating an absent identifier as unknown.
    /// A missing entry can come from an older CLI and is not an explicit
    /// statement that the capability is unsupported.
    pub fn state_for(&self, identifier: &str) -> CapabilityAvailability {
        self.capabilities
            .iter()
            .find(|capability| capability.id == identifier)
            .map(|capability| capability.state)
            .unwrap_or(CapabilityAvailability::Unknown)
    }

    /// Parse and validate the runtime catalog returned by `m-model doctor`.
    pub fn from_doctor_data(data: &Value) -> Result<Self, String> {
        let compatibility = data
            .get("compatibility")
            .ok_or_else(|| "doctor report is missing compatibility metadata".to_owned())?;
        let mut catalog = Self {
            worker_protocol_version: compatibility["worker_protocol_version"]
                .as_u64()
                .and_then(|version| u32::try_from(version).ok())
                .ok_or_else(|| "doctor report has an invalid worker protocol version".to_owned())?,
            worker_api_version: compatibility["worker_api_version"]
                .as_u64()
                .and_then(|version| u32::try_from(version).ok())
                .ok_or_else(|| "doctor report has an invalid worker API version".to_owned())?,
            state_formats: serde_json::from_value(compatibility["state_formats"].clone())
                .map_err(|_| "doctor report has invalid state format versions".to_owned())?,
            capabilities: serde_json::from_value(
                data.get("capabilities")
                    .cloned()
                    .ok_or_else(|| "doctor report is missing capabilities".to_owned())?,
            )
            .map_err(|_| "doctor report has invalid capability descriptors".to_owned())?,
        };
        for identifier in READINESS_CAPABILITY_IDS {
            if !catalog
                .capabilities
                .iter()
                .any(|capability| capability.id == *identifier)
            {
                catalog.capabilities.push(CapabilityDescriptor {
                    id: (*identifier).into(),
                    state: CapabilityAvailability::Unknown,
                    scope: None,
                    reason: Some(
                        "所选 Worker 能力目录未声明此项；可能是旧 CLI 或目录不完整。".into(),
                    ),
                    provider: None,
                });
            }
        }
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.worker_protocol_version != WORKER_PROTOCOL_VERSION {
            return Err(format!(
                "worker protocol v{} is incompatible with supported v{}",
                self.worker_protocol_version, WORKER_PROTOCOL_VERSION
            ));
        }
        if self.worker_api_version != WORKER_API_VERSION {
            return Err(format!(
                "worker API v{} is incompatible with supported v{}",
                self.worker_api_version, WORKER_API_VERSION
            ));
        }
        if self
            .state_formats
            .iter()
            .any(|(name, version)| name.trim().is_empty() || *version == 0)
        {
            return Err("doctor report contains an invalid state format entry".into());
        }
        let mut identifiers = BTreeSet::new();
        for capability in &self.capabilities {
            if capability.id.trim().is_empty() || !identifiers.insert(&capability.id) {
                return Err("doctor report capability IDs must be non-empty and unique".into());
            }
            if capability.state != CapabilityAvailability::Supported
                && capability.reason.as_deref().is_none_or(str::is_empty)
            {
                return Err(format!(
                    "doctor report capability {} must explain why it is not supported",
                    capability.id
                ));
            }
        }
        Ok(())
    }
}

/// Add the compatibility requirements implied by the operation being sent to the worker.
/// Explicit caller requirements are preserved and combined with the operation contract.
pub fn annotate_worker_request(request: &mut Value) {
    let Some(fields) = request.as_object_mut() else {
        return;
    };
    fields
        .entry("worker_api_version")
        .or_insert_with(|| Value::from(WORKER_API_VERSION));

    let operation = fields
        .get("operation")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let feature_field_valid = !fields.contains_key("required_features")
        || fields
            .get("required_features")
            .and_then(Value::as_array)
            .is_some_and(|features| features.iter().all(Value::is_string));
    let mut required_features = fields
        .get("required_features")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let add_feature = |features: &mut Vec<String>, feature: &str| {
        if !features.iter().any(|existing| existing == feature) {
            features.push(feature.to_owned());
        }
    };
    fn layer_rng_used(value: &Value) -> bool {
        value.as_object().is_some_and(|object| {
            object.contains_key("rng_devices")
                || object.get("operations").and_then(Value::as_array)
                    .is_some_and(|items| items.iter().any(layer_rng_used))
                || object.get("edits").and_then(Value::as_array)
                    .is_some_and(|items| items.iter().any(layer_rng_used))
        })
    }
    if fields.get("edit").is_some_and(layer_rng_used)
        || fields.get("variant").is_some_and(layer_rng_used)
        || fields.get("assembly_recipe").and_then(|recipe| recipe.get("variant")).is_some_and(layer_rng_used)
    {
        add_feature(&mut required_features, "model.variant_layer_rng_v1");
    }
    if fields.get("variant_base_rng_devices").is_some_and(|value| !value.is_null())
        || fields.get("variant").and_then(|variant| variant.get("base_rng_devices")).is_some()
        || fields.get("assembly_recipe").and_then(|recipe| recipe.get("variant"))
            .and_then(|variant| variant.get("base_rng_devices")).is_some()
    {
        add_feature(&mut required_features, "model.variant_base_rng_v1");
    }
    if fields.get("variant").and_then(|variant| variant.get("device_policy")).and_then(Value::as_str) == Some("factory")
        || fields.get("variant_device_policy").and_then(Value::as_str) == Some("factory")
    {
        add_feature(&mut required_features, "model.variant_factory_device_v1");
    }
    if fields.get("batch_device_policy").is_some_and(|value| !value.is_null()) {
        add_feature(&mut required_features, "training.batch_device_policy_v1");
    }
    if fields.get("model_initialization_policy").is_some_and(|value| !value.is_null()) {
        add_feature(&mut required_features, "training.model_initialization_v1");
    }
    match operation.as_str() {
        "generic_loop_trial" => {
            add_feature(&mut required_features, "training.generic_loop_trial_v1");
            if fields.get("loop_interface").and_then(Value::as_str) == Some("context") {
                add_feature(&mut required_features, "training.generic_loop_context_signature_v1");
            }
        }
        "generic_loop_train" => {
            add_feature(&mut required_features, "training.generic_loop_train_v1");
            if fields.get("loop_interface").and_then(Value::as_str) == Some("context") {
                add_feature(&mut required_features, "training.generic_loop_context_signature_v1");
            }
        }
        "inspect" | "layers" => {
            add_feature(&mut required_features, "model.inspect");
        }
        "probe" => {
            add_feature(&mut required_features, "model.forward_probe");
            if fields.get("probe_device").is_some_and(|value| !value.is_null()) {
                add_feature(&mut required_features, "model.probe_device_v1");
            }
            if fields.get("contract").is_some_and(|contract| !contract.is_null()) {
                add_feature(&mut required_features, "model.contract_port_paths_v2");
            }
            if fields.get("if_contract_sha256").is_some_and(|value| !value.is_null()) {
                add_feature(&mut required_features, "model.contract_identity_v1");
                add_feature(&mut required_features, "model.contract_preload_v1");
            }
            if fields.get("batch").is_some_and(|batch| !batch.is_null()) {
                add_feature(&mut required_features, "data.dataset_probe");
            }
        }
        "preview" => {
            add_feature(&mut required_features, "data.preview");
            if fields.get("runtime").is_some_and(|runtime| {
                runtime.get("working_directory").is_some_and(|value| !value.is_null())
                    || runtime.get("project_import_roots").is_some_and(|value| !value.is_null())
            }) {
                add_feature(&mut required_features, "data.preview_runtime_paths_v1");
            }
            if fields
                .get("preview_source")
                .and_then(|source| source.get("kind"))
                .and_then(Value::as_str)
                == Some("dataset")
            {
                add_feature(&mut required_features, "data.dataset_probe");
            }
        }
        "verify" => {
            add_feature(&mut required_features, "model.loss_probe");
        }
        "edit" => {
            add_feature(&mut required_features, "model.layer_edit");
            if fields.get("edit").and_then(|edit| edit.get("target")).and_then(Value::as_str) == Some("source") {
                add_feature(&mut required_features, "edit.source_factory_device_v1");
            }
            if fields.get("variant").is_none_or(Value::is_null)
                && fields.get("seed").is_none_or(Value::is_null)
                && fields.get("edit").and_then(|edit| edit.get("target")).and_then(Value::as_str).unwrap_or("variant") == "variant"
            {
                add_feature(&mut required_features, "edit.random_variant_seed_v1");
            }
            if fields.get("seed").is_none_or(Value::is_null)
                && fields.get("edit").and_then(|edit| edit.get("target")).and_then(Value::as_str) == Some("source")
            {
                add_feature(&mut required_features, "model.runtime_rng_v1");
            }
            let edit = fields.get("edit");
            if edit.and_then(|value| value.get("target")).and_then(Value::as_str) == Some("source")
                && edit.and_then(|value| value.get("action")).and_then(Value::as_str) == Some("transaction")
                && edit.and_then(|value| value.get("operations")).and_then(Value::as_array).is_some_and(|operations| {
                    operations.iter().any(|operation| operation["action"] == "set")
                        && operations.iter().any(|operation| matches!(operation["action"].as_str(), Some("add" | "replace" | "remove")))
                }) {
                add_feature(&mut required_features, "edit.mixed_source_transaction_v1");
            }
        }
        "assembly_codegen" => {
            add_feature(&mut required_features, "edit.assembly_codegen");
        }
        "assembly_codegen_verify" => {
            add_feature(&mut required_features, "edit.assembly_codegen");
            if fields.get("batch").is_some_and(|batch| !batch.is_null()) {
                add_feature(&mut required_features, "model.forward_probe");
                add_feature(&mut required_features, "data.dataset_probe");
            }
        }
        "training_probe" | "train" => {
            if operation == "train" {
                add_feature(&mut required_features, "training.batch_binding_checkpoint_v1");
            }
            if fields.get("runtime").and_then(|runtime| runtime.get("device")).and_then(Value::as_str).is_some_and(|device| {
                let device = device.trim().to_ascii_lowercase();
                device != "cpu" && device != "cuda" && !device.starts_with("cuda:")
            }) {
                add_feature(&mut required_features, "training.generic_device_v1");
            }
            if fields.get("optimizer_factory").is_some_and(|value| !value.is_null()) {
                add_feature(&mut required_features, "training.custom_optimizer_worker_v1");
            }
            let feature = match fields.get("training_mode").and_then(Value::as_str) {
                Some("managed-step") => Some("training.managed_step"),
                Some("manual-step") => Some("training.manual_step"),
                Some("external-loop") => Some("training.external_loop"),
                _ => None,
            };
            if let Some(feature) = feature {
                add_feature(&mut required_features, feature);
            }
            if fields
                .get("checkpoint_input")
                .is_some_and(|checkpoint| !checkpoint.is_null())
            {
                add_feature(&mut required_features, "training.checkpoint_resume");
            }
        }
        "evaluate" => {
            add_feature(&mut required_features, "training.evaluation");
            add_feature(&mut required_features, "training.checkpoint_resume");
            if fields.get("evaluation_device").is_some_and(|value| !value.is_null()) {
                add_feature(&mut required_features, "evaluation.model_device_v1");
            }
        }
        "component_install"
        | "component_uninstall"
        | "component_discover"
        | "extension_bundle_plan" => {
            add_feature(&mut required_features, "extensions.static_discovery");
        }
        "component_test" => {
            add_feature(&mut required_features, "extensions.component_conformance");
        }
        "environment_setup" => {
            add_feature(&mut required_features, "environment.setup");
        }
        "environment_library_inventory" => {
            add_feature(&mut required_features, "environment.library_inventory");
        }
        "environment_library_probe" => {
            add_feature(&mut required_features, "environment.library_probe");
            if fields.get("layer_catalog").and_then(Value::as_bool) == Some(true) {
                add_feature(&mut required_features, "environment.torch_layer_catalog_v1");
            }
            if fields.get("operator_catalog").and_then(Value::as_bool) == Some(true) {
                add_feature(&mut required_features, "environment.operator_catalog_v1");
            }
        }
        "environment_devices" => {
            add_feature(&mut required_features, "environment.device_inventory_v1");
        }
        "environment_wheelhouse_archive"
        | "environment_wheelhouse_check"
        | "environment_wheelhouse_replay" => {
            add_feature(&mut required_features, "environment.wheelhouse");
            if operation == "environment_wheelhouse_replay" && fields.get("create_environment").and_then(Value::as_bool) == Some(true) {
                add_feature(&mut required_features, "environment.wheelhouse_create_v1");
            }
        }
        _ => {}
    }
    let direct_factory_params = |value: &Value| value.as_object().is_some_and(|injections| {
        injections.values().any(|binding| binding.as_object().is_some_and(|object| {
            object.contains_key("factory") && !object.contains_key("component_ref")
        }))
    });
    if matches!(operation.as_str(), "probe" | "verify" | "edit" | "semantic_mark" | "semantic_supervise" | "training_probe" | "train" | "assembly_codegen_verify" | "component_test")
        && fields.get("batch_binding").and_then(Value::as_str) == Some("single-raw") {
        add_feature(&mut required_features, "data.raw_batch_binding_v1");
    }
    if operation == "preview" && fields.get("preview_source").and_then(|source| source.get("binding")).and_then(Value::as_str) == Some("single-raw") {
        add_feature(&mut required_features, "data.raw_batch_binding_v1");
    }
    if matches!(operation.as_str(), "probe" | "verify" | "edit" | "semantic_mark" | "semantic_supervise" | "assembly_codegen_verify")
        && fields.get("seed").is_none_or(Value::is_null)
        && (fields.get("batch").is_some_and(|value| !value.is_null())
            || fields.get("dataset").is_some_and(|value| !value.is_null()))
    {
        add_feature(&mut required_features, "data.runtime_sampling_seed_v1");
    }
    if fields.get("model_injections").is_some_and(direct_factory_params)
        || fields.get("variant").and_then(|variant| variant.get("origin"))
            .and_then(|origin| origin.get("model_injections")).is_some_and(direct_factory_params)
    {
        add_feature(&mut required_features, "model.direct_factory_params_v1");
    }
    if fields.get("runtime").and_then(|runtime| runtime.get("working_directory"))
        .is_some_and(|directory| !directory.is_null())
    {
        add_feature(&mut required_features, "runtime.user_working_directory");
    }
    if fields.get("operation").and_then(Value::as_str) == Some("semantic_mark")
        && fields.get("probe_device").is_some_and(|value| !value.is_null())
    {
        add_feature(&mut required_features, "model.semantic_device_v1");
    }
    if operation == "semantic_supervise" && fields.get("probe_device").is_some_and(|value| !value.is_null()) {
        add_feature(&mut required_features, "supervision.preview_device_v1");
    }
    if fields.get("operation").and_then(Value::as_str) == Some("semantic_mark")
        && fields.get("edit").and_then(|edit| edit.get("if_anchor")).is_some_and(|value| !value.is_null())
    {
        add_feature(&mut required_features, "model.semantic_anchor_guard");
    }
    if operation == "semantic_mark" && fields.get("path").and_then(Value::as_str) == Some("model") {
        add_feature(&mut required_features, "supervision.root_mark_v1");
    }
    if operation == "semantic_mark" && fields.get("edit").and_then(|edit| edit.get("if_anchor"))
        .and_then(Value::as_str).and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|anchor| anchor["source_file"].as_str().map(str::to_owned))
        .is_some_and(|source| source.starts_with("@external:")) {
        add_feature(&mut required_features, "model.semantic_external_source_v1");
    }
    if fields.get("operation").and_then(Value::as_str) == Some("semantic_supervise")
        && fields.get("edit").and_then(|edit| edit.get("action")).and_then(Value::as_str) == Some("rule_add")
        && fields.get("edit").and_then(|edit| edit.get("if_state_revision")).is_some_and(|value| !value.is_null())
    {
        add_feature(&mut required_features, "supervision.conditional_rule_add_v1");
    }
    if operation == "semantic_supervise"
        && matches!(fields.get("edit").and_then(|edit| edit.get("action")).and_then(Value::as_str), Some("bundle_export" | "bundle_import"))
    {
        add_feature(&mut required_features, "supervision.bundle_v1");
    }
    if operation == "semantic_supervise"
        && fields.get("edit").and_then(|edit| edit.get("action")).and_then(Value::as_str) == Some("rule_copy")
    {
        add_feature(&mut required_features, "supervision.rule_copy_v1");
    }
    if operation == "semantic_supervise"
        && fields.get("edit").and_then(|edit| edit.get("action")).and_then(Value::as_str) == Some("rule_add")
        && fields.get("edit").and_then(|edit| edit.get("axis")).and_then(Value::as_u64).is_some_and(|axis| axis > 16)
    {
        add_feature(&mut required_features, "supervision.extended_axis_v1");
    }
    let semantic_action = fields.get("edit").and_then(|edit| edit.get("action")).and_then(Value::as_str);
    if (operation == "semantic_mark" && matches!(semantic_action, Some("add" | "update" | "audit")))
        || (operation == "semantic_supervise" && matches!(semantic_action, Some("rule_add" | "preview")))
        || (matches!(operation.as_str(), "training_probe" | "train")
            && fields.get("apply_supervision").and_then(Value::as_bool) == Some(true))
    {
        add_feature(&mut required_features, "supervision.port_path_v2");
    }
    if operation == "semantic_supervise" && fields.get("edit").and_then(|edit| edit.get("plugin_source")).and_then(Value::as_str) == Some("environment") {
        add_feature(&mut required_features, "supervision.environment_rule_v1");
    }
    if fields.get("runtime").and_then(|runtime| runtime.get("project_import_roots"))
        .is_some_and(|roots| !roots.is_null())
    {
        add_feature(&mut required_features, "runtime.project_import_roots");
    }
    if feature_field_valid
        && (!required_features.is_empty() || fields.contains_key("required_features"))
    {
        required_features.sort();
        fields.insert(
            "required_features".into(),
            serde_json::to_value(required_features).expect("feature set serializes"),
        );
    }

    let state_format_field_valid = !fields.contains_key("required_state_formats")
        || fields
            .get("required_state_formats")
            .is_some_and(Value::is_object);
    let mut required_state_formats = fields
        .get("required_state_formats")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(variant) = fields.get("variant").filter(|variant| !variant.is_null()) {
        let version = variant
            .get("schema_version")
            .cloned()
            .unwrap_or_else(|| Value::from(0));
        required_state_formats
            .entry("model_variant")
            .or_insert(version);
    }
    if fields
        .get("checkpoint_input")
        .is_some_and(|checkpoint| !checkpoint.is_null())
    {
        required_state_formats
            .entry("training_checkpoint")
            .or_insert_with(|| Value::from(1));
    }
    if operation == "generic_loop_train" && fields.get("generic_checkpoint_input").is_some_and(|checkpoint| !checkpoint.is_null()) {
        required_state_formats.entry("generic_loop_checkpoint").or_insert_with(|| Value::from(1));
    }
    if operation == "edit" && fields.get("edit").is_some_and(|edit| !edit.is_null()) {
        required_state_formats
            .entry("layer_edit_plan")
            .or_insert_with(|| Value::from(1));
    }
    if matches!(
        operation.as_str(),
        "component_install"
            | "component_uninstall"
            | "component_discover"
            | "component_test"
            | "extension_bundle_plan"
    ) {
        required_state_formats
            .entry("extension_manifest")
            .or_insert_with(|| Value::from(1));
    }
    if operation == "environment_check" || operation.starts_with("environment_wheelhouse_") {
        required_state_formats
            .entry("environment_lock")
            .or_insert_with(|| Value::from(1));
    }
    if state_format_field_valid
        && (!required_state_formats.is_empty() || fields.contains_key("required_state_formats"))
    {
        fields.insert(
            "required_state_formats".into(),
            Value::Object(required_state_formats),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> CapabilityCatalog {
        CapabilityCatalog {
            worker_protocol_version: WORKER_PROTOCOL_VERSION,
            worker_api_version: WORKER_API_VERSION,
            state_formats: BTreeMap::from([
                ("training_checkpoint".into(), 1),
                ("project_config".into(), 1),
            ]),
            capabilities: vec![
                CapabilityDescriptor {
                    id: "model.forward_probe".into(),
                    state: CapabilityAvailability::Supported,
                    scope: Some(CapabilityScope::RuntimeEnvironment),
                    reason: None,
                    provider: Some("python-worker".into()),
                },
                CapabilityDescriptor {
                    id: "train.detached".into(),
                    state: CapabilityAvailability::Unsupported,
                    scope: Some(CapabilityScope::RuntimeEnvironment),
                    reason: Some("detached training is supported on Linux only".into()),
                    provider: Some("python-worker".into()),
                },
                CapabilityDescriptor {
                    id: "data.cursor_resume".into(),
                    state: CapabilityAvailability::Unknown,
                    scope: None,
                    reason: Some("the selected adapter has no state hooks".into()),
                    provider: Some("data-adapter".into()),
                },
            ],
        }
    }

    #[test]
    fn catalog_accepts_unique_capabilities_and_rejects_unexplained_states() {
        assert!(catalog().validate().is_ok());
        let mut invalid = catalog();
        invalid.capabilities[1].reason = None;
        assert!(invalid.validate().unwrap_err().contains("must explain"));
    }

    #[test]
    fn duplicate_capability_ids_are_rejected() {
        let mut invalid = catalog();
        invalid.capabilities.push(invalid.capabilities[0].clone());
        assert!(invalid.validate().unwrap_err().contains("unique"));
    }

    #[test]
    fn worker_requests_receive_operation_features_and_state_formats() {
        let mut request = serde_json::json!({
            "operation": "train",
            "training_mode": "manual-step",
            "variant": {"schema_version": 1},
            "checkpoint_input": "runs/old/checkpoint.pt",
            "required_features": ["custom.requirement"]
        });
        annotate_worker_request(&mut request);
        assert_eq!(request["worker_api_version"], WORKER_API_VERSION);
        assert_eq!(
            request["required_features"],
            serde_json::json!([
                "custom.requirement",
                "training.checkpoint_resume",
                "training.manual_step"
            ])
        );
        assert_eq!(
            request["required_state_formats"],
            serde_json::json!({"model_variant": 1, "training_checkpoint": 1})
        );
    }

    #[test]
    fn model_and_extension_operations_receive_only_relevant_requirements() {
        let mut probe = serde_json::json!({"operation": "probe", "batch": "data:example"});
        annotate_worker_request(&mut probe);
        assert_eq!(
            probe["required_features"],
            serde_json::json!(["data.dataset_probe", "model.forward_probe"])
        );

        let mut verification =
            serde_json::json!({"operation":"assembly_codegen_verify", "batch":"fixture:batch"});
        annotate_worker_request(&mut verification);
        assert_eq!(
            verification["required_features"],
            serde_json::json!([
                "data.dataset_probe",
                "edit.assembly_codegen",
                "model.forward_probe"
            ])
        );
        let mut construction_only =
            serde_json::json!({"operation":"assembly_codegen_verify", "batch":null});
        annotate_worker_request(&mut construction_only);
        assert_eq!(
            construction_only["required_features"],
            serde_json::json!(["edit.assembly_codegen"])
        );

        let mut test = serde_json::json!({"operation": "component_test"});
        annotate_worker_request(&mut test);
        assert_eq!(
            test["required_features"],
            serde_json::json!(["extensions.component_conformance"])
        );
        assert_eq!(test["required_state_formats"]["extension_manifest"], 1);

        let mut edit = serde_json::json!({"operation": "edit", "edit": {"action": "add"}});
        annotate_worker_request(&mut edit);
        assert_eq!(
            edit["required_features"],
            serde_json::json!(["model.layer_edit"])
        );
        assert_eq!(edit["required_state_formats"]["layer_edit_plan"], 1);
    }

    #[test]
    fn malformed_or_duplicate_explicit_requirements_are_not_silently_repaired() {
        let mut malformed = serde_json::json!({
            "operation": "probe",
            "required_features": "model.inspect",
            "required_state_formats": []
        });
        annotate_worker_request(&mut malformed);
        assert_eq!(malformed["required_features"], "model.inspect");
        assert_eq!(malformed["required_state_formats"], serde_json::json!([]));

        let mut duplicates = serde_json::json!({
            "operation": "probe",
            "required_features": ["custom.feature", "custom.feature"]
        });
        annotate_worker_request(&mut duplicates);
        assert_eq!(
            duplicates["required_features"],
            serde_json::json!(["custom.feature", "custom.feature", "model.forward_probe"])
        );
    }
}
