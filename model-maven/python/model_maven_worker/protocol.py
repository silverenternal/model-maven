"""Stable JSON worker protocol primitives shared by worker domains."""

PROTOCOL_VERSION = 1
WORKER_API_VERSION = 1
PROVIDER_CONFORMANCE_SUITE_VERSION = 2
MAX_CALLS = 256
VALID_EDIT_ACTIONS = {"add", "set", "replace", "remove", "custom", "sharing"}

STATE_FORMAT_VERSIONS = {
    "project_config": 1,
    "environment_lock": 1,
    "training_checkpoint": 1,
    "generic_loop_checkpoint": 1,
    "model_variant": 1,
    "layer_edit_plan": 1,
    "extension_manifest": 1,
    "run_artifact_manifest": 2,
}

# Worker operations are registered here with the capabilities they can execute.
# The dispatcher rejects unregistered operation names, and doctor derives its
# worker capability IDs from this same table.
OPERATION_FEATURES = {
    "doctor": (),
    "runtime_capabilities": (),
    "inspect": ("model.inspect", "model.runtime_rng_v1"),
    "layers": ("model.inspect",),
    "probe": ("model.forward_probe", "data.dataset_probe", "model.contract_port_paths_v2", "model.contract_identity_v1", "model.contract_preload_v1", "model.probe_device_v1", "model.runtime_rng_v1", "data.runtime_sampling_seed_v1"),
    "graph_draft_probe": ("model.forward_probe", "data.dataset_probe", "graph.draft_replay_v1"),
    "semantic_mark": ("model.forward_probe", "data.dataset_probe", "model.semantic_marks", "model.semantic_anchor_guard", "model.semantic_external_source_v1", "supervision.port_path_v2", "supervision.root_mark_v1", "model.semantic_device_v1", "data.runtime_sampling_seed_v1"),
    "semantic_supervise": (
        "model.forward_probe",
        "data.dataset_probe",
        "model.semantic_marks",
        "training.supervision_preview",
        "supervision.preview_device_v1",
        "data.runtime_sampling_seed_v1",
        "supervision.conditional_rule_add_v1",
        "supervision.rule_copy_v1",
        "supervision.bundle_v1",
        "supervision.extended_axis_v1",
        "supervision.environment_rule_v1",
        "supervision.port_path_v2",
    ),
    "preview": ("data.preview", "data.preview_runtime_paths_v1"),
    "data_index": ("data.classification_index",),
    "preset_catalog": (),
    "verify": ("model.loss_probe", "data.runtime_sampling_seed_v1"),
    "edit": ("model.layer_edit", "edit.mixed_source_transaction_v1", "edit.random_variant_seed_v1", "data.runtime_sampling_seed_v1", "model.runtime_rng_v1", "edit.source_factory_device_v1"),
    "assembly_codegen": ("edit.assembly_codegen", "edit.assembly_python_rng_v1"),
    "assembly_codegen_verify": ("edit.assembly_codegen", "edit.assembly_codegen_runtime_paths_v1", "model.forward_probe", "data.dataset_probe", "data.runtime_sampling_seed_v1"),
    "training_probe": (
        "training.managed_step",
        "training.custom_optimizer_worker_v1",
        "training.generic_device_v1",
        "training.manual_step",
        "training.external_loop",
        "training.checkpoint_resume",
        "training.supervision_apply",
        "supervision.port_path_v2",
    ),
    "generic_loop_trial": ("training.generic_loop_trial_v1", "training.generic_loop_context_signature_v1"),
    "generic_loop_train": ("training.generic_loop_train_v1", "training.generic_loop_context_signature_v1"),
    "train": (
        "training.managed_step",
        "training.batch_binding_checkpoint_v1",
        "training.custom_optimizer_worker_v1",
        "training.generic_device_v1",
        "training.manual_step",
        "training.external_loop",
        "training.checkpoint_resume",
        "training.weight_migration",
        "training.supervision_apply",
        "supervision.port_path_v2",
    ),
    "evaluate": ("training.evaluation", "training.checkpoint_resume", "evaluation.model_device_v1"),
    "component_install": ("extensions.static_discovery",),
    "component_uninstall": ("extensions.static_discovery",),
    "component_discover": ("extensions.static_discovery",),
    "extension_bundle_plan": ("extensions.static_discovery",),
    "component_test": ("extensions.component_conformance",),
    "environment_snapshot": (),
    "environment_lock": (),
    "environment_setup": ("environment.setup",),
    "environment_configure": ("environment.setup",),
    "environment_check": (),
    "environment_library_inventory": ("environment.library_inventory",),
    "environment_library_probe": ("environment.library_probe", "environment.torch_layer_catalog_v1", "environment.operator_catalog_v1"),
    "environment_devices": ("environment.device_inventory_v1",),
    "environment_wheelhouse_archive": ("environment.wheelhouse",),
    "environment_wheelhouse_check": ("environment.wheelhouse",),
    "environment_wheelhouse_replay": ("environment.wheelhouse", "environment.wheelhouse_create_v1"),
}
for _operation in ("probe", "preview", "verify", "edit", "semantic_mark", "semantic_supervise",
                   "training_probe", "train", "assembly_codegen_verify", "component_test"):
    OPERATION_FEATURES[_operation] += ("data.raw_batch_binding_v1",)
for _operation in ("inspect", "layers", "probe", "verify", "edit", "semantic_mark",
                   "semantic_supervise", "training_probe", "train", "evaluate", "assembly_codegen"):
    OPERATION_FEATURES[_operation] += ("model.variant_factory_device_v1",)
    OPERATION_FEATURES[_operation] += ("model.variant_layer_rng_v1",)
    OPERATION_FEATURES[_operation] += ("model.variant_base_rng_v1",)
for _operation in ("training_probe", "train"):
    OPERATION_FEATURES[_operation] += ("training.batch_device_policy_v1",)
    OPERATION_FEATURES[_operation] += ("training.model_initialization_v1",)
OPERATION_FEATURES["evaluate"] += ("training.model_initialization_v1",)
HOST_FEATURES = frozenset({"training.detached"})
WORKING_DIRECTORY_OPERATIONS = frozenset({
    "inspect", "layers", "probe", "preview", "verify", "edit", "semantic_mark", "semantic_supervise", "training_probe", "generic_loop_trial", "generic_loop_train", "train", "evaluate", "environment_library_probe", "environment_devices", "assembly_codegen_verify", "graph_draft_probe",
})
PROJECT_IMPORT_ROOT_OPERATIONS = WORKING_DIRECTORY_OPERATIONS | frozenset({
    "semantic_mark", "semantic_supervise",
})
SUPPORTED_REQUIRED_FEATURES = frozenset(
    feature for features in OPERATION_FEATURES.values() for feature in features
) | HOST_FEATURES | frozenset({"runtime.user_working_directory", "runtime.project_import_roots", "model.direct_factory_params_v1", "data.raw_batch_binding_v1"})


def validate_request(request):
    """Reject incompatible API/features before an operation imports runtime code."""
    if not isinstance(request, dict):
        raise UserError("PROTOCOL_ERROR", "worker 请求必须是 JSON 对象。")
    if request.get("operation") == "environment_wheelhouse_replay" and "create_environment" in request \
            and type(request["create_environment"]) is not bool:
        raise UserError("PROTOCOL_ERROR", "create_environment 必须是布尔值。")
    policy = request.get("batch_device_policy")
    if policy is not None:
        if not isinstance(policy, str) or policy not in {"preserve", "recursive-strict"}:
            raise UserError("BATCH_DEVICE_POLICY_INVALID", "批次设备策略必须是 preserve 或 recursive-strict。")
        if request.get("operation") not in {"training_probe", "train"} or request.get("training_mode") == "external-loop":
            raise UserError("BATCH_DEVICE_POLICY_UNSUPPORTED", "此操作不使用受管或手动批次设备策略。")
    if request.get("optimizer_factory") is not None and (
        request.get("operation") not in {"train", "training_probe"}
        or request.get("training_mode") != "managed-step"
    ):
        raise UserError("OPTIMIZER_MODE_UNSUPPORTED", "自定义受管优化器工厂只支持 managed-step。")
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise UserError("PROTOCOL_ERROR", "请求协议版本不匹配。")
    operation = request.get("operation")
    if not isinstance(operation, str) or operation not in OPERATION_FEATURES:
        raise UserError(
            "OPERATION_UNSUPPORTED",
            f"当前 worker 没有注册操作：{operation!r}。",
            "升级 worker，或使用当前版本支持的操作。",
            {"operation": operation},
        )
    api_version = request.get("worker_api_version", WORKER_API_VERSION)
    if type(api_version) is not int or api_version != WORKER_API_VERSION:
        raise UserError(
            "WORKER_API_UNSUPPORTED",
            f"请求需要 worker API v{api_version}，当前支持 v{WORKER_API_VERSION}。",
            "升级 Model-Maven CLI/worker，或按当前兼容矩阵重新生成请求。",
            {"required": api_version, "supported": WORKER_API_VERSION},
        )
    required = request.get("required_features", [])
    if (
        not isinstance(required, list)
        or any(not isinstance(feature, str) or not feature.strip() for feature in required)
        or len(required) != len(set(required))
    ):
        raise UserError("PROTOCOL_ERROR", "required_features 必须是非空且不重复的字符串数组。")
    operation_features = set(OPERATION_FEATURES[operation]) | HOST_FEATURES
    if operation in {"inspect", "layers", "probe", "verify", "edit", "semantic_mark", "semantic_supervise",
                     "training_probe", "train", "evaluate", "assembly_codegen_verify"}:
        operation_features.add("model.direct_factory_params_v1")
    if operation in WORKING_DIRECTORY_OPERATIONS:
        operation_features.add("runtime.user_working_directory")
    if operation in PROJECT_IMPORT_ROOT_OPERATIONS:
        operation_features.add("runtime.project_import_roots")
    if (
        operation == "preview"
        and isinstance(request.get("preview_source"), dict)
        and request["preview_source"].get("kind") == "dataset"
    ):
        operation_features.add("data.dataset_probe")
    missing = sorted(set(required) - operation_features)
    if missing:
        raise UserError(
            "FEATURE_UNAVAILABLE",
            f"当前 worker 操作 {operation} 未注册必需功能：{', '.join(missing)}。",
            "升级 worker，或修正本次操作声明的必需 feature。",
            {"unsupported_features": missing},
        )
    state_formats = request.get("required_state_formats", {})
    if not isinstance(state_formats, dict):
        raise UserError("PROTOCOL_ERROR", "required_state_formats 必须是对象。")
    incompatible = {}
    for name, version in state_formats.items():
        supported = STATE_FORMAT_VERSIONS.get(name)
        if type(version) is not int or supported != version:
            incompatible[name] = {"required": version, "supported": supported}
    if incompatible:
        raise UserError(
            "STATE_FORMAT_UNSUPPORTED",
            "运行需要 worker 不支持的状态格式版本。",
            "使用兼容版本迁移或显式恢复状态；不会猜测字段含义。",
            {"incompatible_state_formats": incompatible},
        )


def compatibility_report():
    return {
        "worker_protocol_version": PROTOCOL_VERSION,
        "worker_api_version": WORKER_API_VERSION,
        "state_formats": dict(STATE_FORMAT_VERSIONS),
        "required_features": sorted(SUPPORTED_REQUIRED_FEATURES),
        "operations": {
            operation: list(features)
            for operation, features in sorted(OPERATION_FEATURES.items())
        },
    }


class UserError(Exception):
    def __init__(self, code, message, hint="", data=None):
        super().__init__(message)
        self.code = code
        self.hint = hint
        self.data = data or {}


class InconclusiveError(UserError):
    """The requested check ran, but produced no evidence that supports a result."""
