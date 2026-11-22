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
    "inspect": ("model.inspect",),
    "layers": ("model.inspect",),
    "probe": ("model.forward_probe", "data.dataset_probe"),
    "semantic_mark": ("model.forward_probe", "data.dataset_probe", "model.semantic_marks"),
    "semantic_supervise": (
        "model.forward_probe",
        "data.dataset_probe",
        "model.semantic_marks",
        "training.supervision_preview",
    ),
    "preview": ("data.preview",),
    "data_index": ("data.classification_index",),
    "preset_catalog": (),
    "verify": ("model.loss_probe",),
    "edit": ("model.layer_edit",),
    "assembly_codegen": ("edit.assembly_codegen",),
    "assembly_codegen_verify": ("edit.assembly_codegen", "model.forward_probe", "data.dataset_probe"),
    "training_probe": (
        "training.managed_step",
        "training.manual_step",
        "training.external_loop",
        "training.checkpoint_resume",
        "training.supervision_apply",
    ),
    "train": (
        "training.managed_step",
        "training.manual_step",
        "training.external_loop",
        "training.checkpoint_resume",
        "training.weight_migration",
        "training.supervision_apply",
    ),
    "evaluate": ("training.evaluation", "training.checkpoint_resume"),
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
    "environment_library_probe": ("environment.library_probe",),
    "environment_wheelhouse_archive": ("environment.wheelhouse",),
    "environment_wheelhouse_check": ("environment.wheelhouse",),
    "environment_wheelhouse_replay": ("environment.wheelhouse",),
}
HOST_FEATURES = frozenset({"training.detached"})
SUPPORTED_REQUIRED_FEATURES = frozenset(
    feature for features in OPERATION_FEATURES.values() for feature in features
) | HOST_FEATURES


def validate_request(request):
    """Reject incompatible API/features before an operation imports runtime code."""
    if not isinstance(request, dict):
        raise UserError("PROTOCOL_ERROR", "worker 请求必须是 JSON 对象。")
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
