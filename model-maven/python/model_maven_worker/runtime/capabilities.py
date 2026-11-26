"""Runtime capability catalog returned by doctor and operation preflight."""

from model_maven_worker.protocol import HOST_FEATURES, SUPPORTED_REQUIRED_FEATURES


_TORCH_FEATURES = SUPPORTED_REQUIRED_FEATURES - HOST_FEATURES - {
    "extensions.static_discovery",
    "environment.setup",
    "environment.library_inventory",
    "environment.library_probe",
    "environment.torch_layer_catalog_v1",
    "environment.operator_catalog_v1",
    "environment.device_inventory_v1",
    "environment.wheelhouse",
    "environment.wheelhouse_create_v1",
    "edit.assembly_codegen",
    "edit.assembly_python_rng_v1",
    "runtime.user_working_directory",
    "runtime.project_import_roots",
    "supervision.conditional_rule_add_v1",
    "supervision.rule_copy_v1",
    "supervision.bundle_v1",
    "supervision.extended_axis_v1",
    "supervision.environment_rule_v1",
    "supervision.port_path_v2",
    "training.generic_loop_trial_v1",
    "training.generic_loop_train_v1",
    "training.generic_loop_context_signature_v1",
}
_PROVIDERS = {
    "data.dataset_probe": "data-adapter",
    "data.runtime_sampling_seed_v1": "data-adapter",
    "data.raw_batch_binding_v1": "data-adapter",
    "data.preview": "data-preview",
    "data.preview_runtime_paths_v1": "data-preview",
    "environment.setup": "python-environment",
    "environment.library_inventory": "python-environment-metadata",
    "environment.library_probe": "python-environment-explicit-import-and-opt-in-cpu-smoke",
    "environment.torch_layer_catalog_v1": "python-environment-explicit-torch-introspection",
    "environment.operator_catalog_v1": "python-environment-explicit-module-introspection",
    "environment.device_inventory_v1": "python-environment-explicit-device-query",
    "environment.wheelhouse": "python-environment",
    "environment.wheelhouse_create_v1": "python-environment",
    "edit.assembly_codegen": "python-worker-code-generator",
    "edit.assembly_python_rng_v1": "python-worker-code-generator",
    "model.runtime_rng_v1": "python-model-construction",
    "edit.random_variant_seed_v1": "variant-edit-planner",
    "edit.source_factory_device_v1": "source-edit-transaction",
    "model.variant_factory_device_v1": "variant-model-reconstruction",
    "model.variant_layer_rng_v1": "variant-layer-initialization",
    "model.variant_base_rng_v1": "variant-base-initialization",
    "training.batch_device_policy_v1": "training-batch-placement",
    "edit.assembly_codegen_runtime_paths_v1": "python-worker-code-generator",
    "edit.mixed_source_transaction_v1": "source-edit-transaction",
    "runtime.user_working_directory": "python-worker-runtime",
    "runtime.project_import_roots": "python-worker-runtime",
    "training.detached": "linux-supervisor",
    "training.generic_loop_trial_v1": "python-generic-user-loop",
    "training.generic_loop_train_v1": "python-generic-user-loop-json-checkpoint",
    "training.generic_loop_context_signature_v1": "python-generic-user-loop",
    "training.evaluation": "project-evaluator",
    "evaluation.model_device_v1": "project-evaluator-device-placement",
    "training.supervision_preview": "semantic-rule-preview",
    "training.supervision_apply": "managed-step-supervision",
    "supervision.port_path_v2": "semantic-port-path-parser",
    "supervision.extended_axis_v1": "semantic-rule-storage",
    "training.batch_binding_checkpoint_v1": "training-checkpoint-signature",
    "model.direct_factory_params_v1": "python-model-construction",
}


def _capability(identifier, available, *, provider, unavailable_reason=None):
    return {
        "id": identifier,
        "state": "unknown" if available is None else "supported" if available else "unsupported",
        # This catalog describes the selected worker and Python environment.
        # Model-specific support requires inspect/probe/edit evidence.
        "scope": "runtime_environment",
        "reason": None if available else unavailable_reason,
        "provider": provider,
    }


def capability_catalog(*, torch_available, platform, torch_import_error=None):
    catalog = []
    for identifier in sorted(SUPPORTED_REQUIRED_FEATURES):
        if identifier in HOST_FEATURES:
            # Host supervision also runs framework-neutral user loops, which
            # do not require PyTorch. Individual training modes declare their
            # own runtime capabilities separately.
            available = platform.startswith("linux")
            reason = "后台运行监督目前仅在 Linux 平台提供。"
        elif identifier in _TORCH_FEATURES:
            available = torch_available
            reason = ("尚未检查 PyTorch 运行时。" if torch_available is None else
                      f"当前解释器导入 PyTorch 失败：{torch_import_error}" if torch_import_error else
                      "当前解释器未发现 PyTorch。")
        else:
            available = True
            reason = None
        catalog.append(
            _capability(
                identifier,
                available,
                provider=_PROVIDERS.get(identifier, "python-worker"),
                unavailable_reason=None if available else reason,
            )
        )
    return catalog


def doctor_report(*, python, python_version, torch_available, torch_version, platform, compatibility,
                  torch_import_error=None):
    implemented = [
        "native model inspect",
        "forward probe / per-invocation trace",
        "manually authored semantic marks bound to observed project module ports",
        "read-only preview of finite/range supervision rules on marked ports",
        "current-batch finite/range violation localization on a manually selected sample axis with user-declared stable IDs",
        "cross-batch sample and violation ID deduplication across checkpoint/resume for manually declared IDs (hash-only checkpoints, bounded summaries)",
        "explicit managed/manual-step require enforcement and weighted range penalties on manually marked ports",
        "versioned project or explicitly selected Python import-path supervision callables with single-source fingerprints, structured summaries, and explicit penalty weights",
        "opt-in external-loop supervision through scoped forward capture, context.backward, and registered optimizer step guards",
        "declared model factory component injection seams",
        "schema-validated registered Layer injection with manifest and transitive dependency locks",
        "user loss backward/gradient check",
        "bounded managed-step AdamW probe",
        "bounded manual-step with named disjoint AdamW groups",
        "bounded external-loop probe with user-reported progress and finite-parameter checks",
        "foreground training trial records, source/runtime checked repeat, and runs list/show",
        "foreground managed/manual training with atomic checkpoints and explicit resume state",
        "foreground external-loop training with explicit state hooks/reset and cooperative stop",
        "formal training timeout requests a cooperative step-boundary checkpoint before force termination",
        "task-specific evaluation from verified training checkpoints with metric/provenance reporting",
        "verified training artifact export",
        "source/runtime checked trial repeat",
        "runs list/show with parent lineage",
        "CPU",
        "Sequential/ModuleList/ModuleDict variant CRUD",
        "ModuleList indexed CRUD with renumbering",
        "ModuleDict named-key CRUD with insertion-order preservation",
        "atomic Layer edit plans",
        "local Layer Edit Provider API v1",
        "registered Layer component runtime smoke test (factory construction, finite state, declared forward probe)",
        "registered transform, dataset, loss, metric, and trainer fixture conformance through component test",
        "registered Layer extension state_dict/load_state_dict roundtrip conformance",
        "variant undo/redo",
        "source Layer set for built-in and uniquely mapped parameterless explicit-signature project nn.Module constructors",
        "source CRUD for directly assigned nn.Sequential, nn.ModuleList, and nn.ModuleDict with built-in or importable Python factory Layers",
        "multi-file source set plans with recoverable journal, fresh-worker revalidation, automatic rollback, conflict-checked undo/redo",
        "grouped source Sequential/ModuleList/ModuleDict add/replace/remove transactions across project files with recovery, revalidation, and conflict-checked undo/redo, including registered local component factories",
        "static installed Python extension discovery with package/source provenance checks",
        "project-local package scaffolds for Layer factories and Layer Edit Providers",
        "project-local Layer/Provider wheel installation without changing the selected environment lock",
        "project Python environment lock/setup/check with wheel hashes, ABI/package drift reporting, and automatic .venv selection",
        "same-ABI hash-verified wheel archives and offline dependency replay",
        "explicit import probe for user-named Python modules; optional built-in runtime probes for selected libraries, without installation or backend binding",
        "fixed-plan cleanup of ended-run logs and progress files with status, keep, and content-hash checks",
        "persistent variant CRUD for Sequential/ModuleList/ModuleDict subclasses with forward validation",
        "cooperative external-loop deadline exposure and registered Trainer timeout conformance",
    ]
    not_implemented = [
        "assembly target editing",
        "preemption of a blocked third-party Python/CUDA call before the CLI watchdog terminates its worker",
        "arbitrary-input extension correctness and OS-level isolation",
        "full extension authoring SDK and package removal/garbage collection",
        "validation of user-declared sample ID authenticity, dataset-wide coverage denominator, and task-specific label semantics",
        "OS isolation or side-effect prevention for supervision callables",
    ]
    if platform.startswith("linux"):
        implemented.append("Linux detached training supervision, attach, and cooperative stop")
    else:
        not_implemented.append("detached training supervision/attach/stop (Linux only)")
    capabilities = capability_catalog(
        torch_available=torch_available,
        torch_import_error=torch_import_error,
        platform=platform,
    )
    return {
        "python": python,
        "python_version": python_version,
        "torch_available": torch_available,
        "torch_version": torch_version,
        "torch_import_error": torch_import_error,
        "implemented": implemented,
        "not_implemented": not_implemented,
        "compatibility": compatibility,
        "capabilities": capabilities,
        "hint": "使用 --python 指定运行时。PyTorch 模型观察和受管训练需要兼容的 PyTorch；用户自有 Python 循环不要求安装 PyTorch。",
    }
