"""Embedded PyTorch adapter. Only JSON descriptors/reports cross the Rust boundary.

User factories run in this disposable process with normal user privileges.
There is deliberately no training base class, Dataset requirement or eval().
"""

import json
import os
from pathlib import Path
import sys
import time
import traceback

_WORKER_STARTED_AT = time.monotonic()


def _working_directory():
    try:
        return os.getcwd()
    except OSError:
        return None


def _select_user_working_directory(request):
    """Apply an explicitly selected directory only in supported model workers.

    The caller keeps project_root and all Model-Maven managed paths absolute;
    the directory is solely the process cwd seen by project Python code.
    """
    runtime = request.get("runtime")
    selected = runtime.get("working_directory") if isinstance(runtime, dict) else None
    if selected is None:
        return
    if request.get("operation") not in WORKING_DIRECTORY_OPERATIONS:
        raise UserError("WORKING_DIRECTORY_UNSUPPORTED", "此操作尚不支持指定运行工作目录。")
    if not isinstance(selected, str) or not selected or "\x00" in selected:
        raise UserError("WORKING_DIRECTORY_INVALID", "运行工作目录必须是非空的绝对路径。")
    path = Path(selected)
    if not path.is_absolute():
        raise UserError("WORKING_DIRECTORY_INVALID", "运行工作目录必须是绝对路径。")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))
        if resolved != path:
            raise UserError("WORKING_DIRECTORY_CONFLICT", "运行目录的规范路径在准备后发生变化。",
                            "重新选择并确认目录；本次未执行模型代码。")
        os.chdir(resolved)
    except (OSError, RuntimeError, ValueError) as error:
        raise UserError("WORKING_DIRECTORY_UNAVAILABLE", f"无法使用运行工作目录：{error}",
                        "检查目录是否存在以及当前用户是否有访问权限。") from error
    request["_selected_working_directory"] = str(resolved)

from model_maven_worker.protocol import (
    MAX_CALLS, PROTOCOL_VERSION, PROVIDER_CONFORMANCE_SUITE_VERSION,
    VALID_EDIT_ACTIONS, WORKING_DIRECTORY_OPERATIONS, InconclusiveError, UserError, compatibility_report, validate_request,
)
from model_maven_worker.runtime.capabilities import doctor_report as _doctor_report
from model_maven_worker.runtime.entrypoints import entrypoint as _entrypoint
from model_maven_worker.runtime.entrypoints import configure_project_import_roots
from model_maven_worker.environment import *  # noqa: F401,F403
from model_maven_worker.data.batch import make_batch
from model_maven_worker.data.index import run_index_operation as _run_index_operation
from model_maven_worker.data.preview import run_preview_operation as _run_preview_operation
from model_maven_worker.data.transforms import run_transform_pipeline as _run_transform_pipeline
from model_maven_worker.model.tensor_tree import tree
from model_maven_worker.model.structure import module_entries, normalize_path, resolve_module, shared_state_groups
from model_maven_worker.model.probe import probe as _probe, verify_gradients as _verify_gradients
from model_maven_worker.model.contracts import load_contract as _load_contract
from model_maven_worker.model.semantics import (
    add_from_probe as _add_semantic_mark,
    add_rule as _add_supervision_rule,
    copy_rule as _copy_supervision_rule,
    export_bundle as _export_supervision_bundle,
    import_bundle as _import_supervision_bundle,
    audit_marks as _audit_semantic_marks,
    binding_context as _semantic_binding_context,
    list_marks as _list_semantic_marks,
    list_rules as _list_supervision_rules,
    remove_mark as _remove_semantic_mark,
    remove_rule as _remove_supervision_rule,
)
from model_maven_worker.train.managed import run_managed_steps as _run_managed_steps
from model_maven_worker.train.manual import run_manual_steps as _run_manual_steps
from model_maven_worker.train.external import run_external_loop as _run_external_loop
from model_maven_worker.train.evaluation import (
    EvaluationRuntime,
    run_evaluation as _run_evaluation,
)
from model_maven_worker.train.supervision import (
    prepare_training_supervision as _prepare_training_supervision,
    preview_supervision as _preview_supervision,
)
from model_maven_worker.train.custom_rules import pin_plugin as _pin_custom_supervision_plugin
from model_maven_worker.train.checkpoint import (
    checkpoint_value_supported as _checkpoint_value_supported,
    load_verified_checkpoint_payload as _load_verified_checkpoint_payload,
    load_training_checkpoint as _load_training_checkpoint,
    model_state_digest as _model_state_digest,
    model_state_alias_groups as _model_state_alias_groups,
    save_training_checkpoint as _save_training_checkpoint,
    state_dict_digest as _state_dict_digest,
    training_run_signature,
)
from model_maven_worker.edit.migration import migrate_state_dict as _migrate_state_dict
from model_maven_worker.edit.codegen import generate_code as _generate_assembly_code
from model_maven_worker.edit.codegen import validate_recipe as _validate_assembly_recipe
from model_maven_worker.train.state import (
    StepContext,
    TrainingStateRuntime,
    add_finite_total as _add_finite_total,
    callable_checkpoint_hooks as _callable_checkpoint_hooks,
    checkpoint_rng_state as _checkpoint_rng_state,
    restore_callable_checkpoint as _restore_callable_checkpoint,
    restore_rng_state as _restore_rng_state,
    scalar_map as _scalar_map,
    scalar_value as _scalar_value,
)
from model_maven_worker.train.runtime import TrainingRuntime
from model_maven_worker.train.device import (
    is_out_of_memory as _is_out_of_memory,
    move_nested_batch as _move_nested_batch,
    out_of_memory_error as _out_of_memory_error,
    reset_peak_memory as _reset_peak_memory,
    resource_snapshot as _resource_snapshot,
    resolve_device as _resolve_device,
)
from model_maven_worker.train.runner import (
    TrainingRunRuntime,
    run_training_run as _run_training_run,
)
from model_maven_worker.train.probe import (
    TrainingModeRuntime,
    TrainingProbeRuntime,
    run_training_mode as _run_training_mode,
    training_probe_with_source_evidence as _training_probe_with_source_evidence,
)
from model_maven_worker.edit.source_candidates import (
    SourceCandidateRuntime,
    source_sequential_candidate as _source_sequential_candidate,
    source_sequential_transaction_state as _source_sequential_transaction_state,
    source_sequential_transaction_candidate as _source_sequential_transaction_candidate,
    apply_source_set_candidate as _apply_source_set_candidate,
    source_set_transaction_candidate as _source_set_transaction_candidate,
    source_mixed_transaction_candidate as _source_mixed_transaction_candidate,
)
from model_maven_worker.edit.source_set import source_set_candidate as _source_set_candidate
from model_maven_worker.edit.source_dispatch import (
    SourceEditDispatchRuntime,
    source_edit_candidate as _source_edit_candidate,
)
from model_maven_worker.extensions.conformance import (
    ExtensionConformanceRuntime,
    component_test as _component_test,
)
from model_maven_worker.edit.source_ast import (
    ast_offset as _source_ast_offset,
    call_name as _source_call_name,
    source_factory_expression as _source_source_factory_expression,
    source_module_expression as _source_source_module_expression,
    source_sequential_rebind as _source_source_sequential_rebind,
    unique_source_alias as _source_unique_source_alias,
)
from model_maven_worker.edit.plan import (
    EditPlanRuntime,
    edit_candidate as _edit_candidate,
)
from model_maven_worker.edit.variants import (
    EditValidationRuntime,
    valid_edit_operation as _valid_edit_operation,
    validate_variant as _validate_variant,
)
from model_maven_worker.edit.layers import (
    LayerEditRuntime,
    apply_edit as _apply_edit,
    apply_history_item as _apply_history_item,
    layer_container_kind as _layer_container_kind,
    new_layer as _new_layer,
    replace_module as _replace_module,
)
from model_maven_worker.model.loading import ModelLoadingRuntime, load_model as _load_model
from model_maven_worker.run.provenance import source_manifest as _source_manifest, verify_manifest as _verify_manifest
from model_maven_worker.run.snapshot import snapshot_project_code
from model_maven_worker.model.construction import (
    construct_model as _construct_model,
    validate_model_injections as _validate_model_injections,
)
from model_maven_worker.model.inspection import (
    editable_params as _editable_params,
    has_module_state,
    inspect_model as _inspect_model,
    model_state_summary as _model_state_summary,
    _state_placement as _registered_state_placement,
    source_editable_fields as _source_editable_fields,
    source_set_block_reason as _source_set_block_reason,
)
from model_maven_worker.extensions.provenance import (
    ComponentProvenanceRuntime,
    verify_component_distribution as _verify_component_distribution,
)
from model_maven_worker.extensions.references import (
    ExtensionReferenceRuntime,
    component_operations as _component_operations,
    module_source_fingerprint as _module_source_fingerprint,
    verify_component_references as _verify_component_references,
    verify_model_injection_references as _verify_model_injection_references,
)
from model_maven_worker.extensions.bundle import (
    ExtensionBundleRuntime,
    extension_bundle_plan as _extension_bundle_plan,
)
from model_maven_worker.extensions.lifecycle import (
    ComponentInstallRuntime,
    ComponentUninstallRuntime,
    component_install as _component_install,
    component_uninstall as _component_uninstall,
)
from model_maven_worker.extensions.wheels import (
    ComponentWheelRuntime,
    sha256_file as _sha256_file,
    wheel_runtime_requirements as _wheel_runtime_requirements,
)
from model_maven_worker.extensions.source_inspection import (
    component_json_files as _component_json_files,
    component_reference_paths as _component_reference_paths,
    component_source_imports,
)
from model_maven_worker.extensions.catalog import (
    discover_components,
    safe_distribution_file,
)

_ACTIVE_CODE_ROOT = None
_ACTIVE_PROJECT_ROOT = None
_ACTIVE_IMPORT_ROOTS = None

SOURCE_LOCATIONS = {}
SOURCE_SITE_COUNTS = {}


def _module_alias_evidence(model):
    """Observe registered module identity after construction, without implying forward coverage."""
    try:
        _, aliases = module_entries(model)
        shared = sorted(sorted(paths) for paths in aliases.values() if len(paths) > 1)
        return {
            "status": "observed",
            "scope": "registered_module_identity_after_construction",
            "group_count": len(shared),
            "groups": [paths[:32] for paths in shared[:32]],
            "truncated": len(shared) > 32 or any(len(paths) > 32 for paths in shared[:32]),
        }
    except Exception:
        return {"status": "unavailable"}


def _shared_state_evidence(model):
    try:
        groups = shared_state_groups(model)
        unknown = any(group.get("kind") == "storage_unavailable" for group in groups)
        return {"status": "partial" if unknown else "observed", "groups": groups,
                "scope": "registered_parameter_buffer_objects_and_inspectable_storage"}
    except Exception:
        return {"status": "unavailable"}


def _training_runtime():
    return TrainingRuntime(
        torch=torch,
        entrypoint=entrypoint,
        tree=lambda value: tree(torch, value),
        step_context=StepContext,
        scalar_value=scalar_value,
        scalar_map=scalar_map,
        add_finite_total=add_finite_total,
        move_batch=lambda batch, device, **kwargs: _move_nested_batch(torch, batch, device, **kwargs),
    )


def _prepare_training_device(model, request):
    runtime = request.get("runtime") or {}
    selection = _resolve_device(torch, runtime.get("device"))
    request["_resolved_device"] = str(selection.device)
    request["_device_selection"] = selection
    request["_device_report"] = selection.report()
    # Check a resumed or repeated run before invoking user-defined model.to().
    # Resolving the device itself can still initialize an accelerator runtime.
    _verify_expected_device(selection, request)
    _reset_peak_memory(torch, selection)
    if model is not None:
        try:
            placed = model.to(selection.device)
        except Exception as error:
            if _is_out_of_memory(torch, error):
                raise _out_of_memory_error(torch, selection, "model placement", error) from error
            raise UserError(
                "DEVICE_MODEL_PLACEMENT_FAILED",
                f"无法将模型移至设备 {selection.label}：{type(error).__name__}: {error}",
                "检查模型是否包含不能迁移的自定义状态；训练步骤尚未执行。",
                {"device": selection.label, "phase": "model placement"},
            ) from error
        if placed is not model:
            raise UserError(
                "DEVICE_MODEL_PLACEMENT_UNCONFIRMED",
                "模型的设备迁移没有返回原模型实例；训练步骤未执行。",
                "检查自定义 to() 实现；迁移可能已经产生副作用。",
                {"device": selection.label, "phase": "model placement"},
            )
    return selection


def _verify_expected_device(selection, request):
    expected = request.get("expected_runtime")
    if not isinstance(expected, dict):
        return
    changes = {}
    actual = selection.report()
    for field in ("device", "device_backend", "device_name"):
        expected_value = expected.get(field)
        if expected_value and expected_value != actual.get(field):
            changes[field] = {"expected": expected_value, "actual": actual.get(field)}
    if changes:
        operation = request.get("operation")
        raise UserError(
            "RESUME_RUNTIME_CHANGED" if operation == "train" else "REPEAT_RUNTIME_CHANGED",
            "原运行使用的设备或加速后端与当前环境不一致，本次训练步骤未执行。",
            "恢复原设备与 PyTorch 构建后重试，或在当前设备上建立新的运行。",
            {"runtime_changes": changes},
        )


def _finish_training_resources(request):
    selection = request["_device_selection"]
    request["_resource_report"] = _resource_snapshot(torch, selection)


def _raise_device_oom(request, error, phase):
    if isinstance(error, UserError) and error.code == "TRAINING_OUT_OF_MEMORY":
        return
    selection = request.get("_device_selection")
    if selection is not None and _is_out_of_memory(torch, error):
        raise _out_of_memory_error(torch, selection, phase, error) from error


def checkpoint_value_supported(value):
    return _checkpoint_value_supported(value, torch)


def model_state_digest(model):
    return _model_state_digest(torch, model)


def model_state_alias_groups(model):
    return _model_state_alias_groups(model)


def load_verified_checkpoint_payload(path):
    return _load_verified_checkpoint_payload(torch, path)


def state_dict_digest(state):
    return _state_dict_digest(torch, state)


def migrate_state_dict(source_state, target_state, specification, source_alias_groups=(), target_alias_groups=()):
    return _migrate_state_dict(
        torch, source_state, target_state, specification, source_alias_groups, target_alias_groups
    )


def save_training_checkpoint(directory, payload):
    return _save_training_checkpoint(torch, directory, payload)


def load_training_checkpoint(path, model, request, source_manifest):
    return _load_training_checkpoint(torch, path, model, request, source_manifest)


def run_external_loop(
    model, request, *, restore=None, checkpoint_callback=None,
    loop_fn=None, loop_state_status="new",
):
    return _run_external_loop(
        _training_runtime(), model, request,
        restore=restore, checkpoint_callback=checkpoint_callback,
        loop_fn=loop_fn, loop_state_status=loop_state_status,
    )


def run_evaluation(model, request):
    def prepare_evaluation_device(model, request):
        selection = _resolve_device(torch, request["evaluation_device"])
        try:
            placed = model.to(selection.device)
        except Exception as error:
            raise UserError("EVALUATION_DEVICE_PLACEMENT_FAILED",
                            f"无法将模型迁移至评估设备：{type(error).__name__}: {error}",
                            "检查设备和模型的迁移支持；评估入口尚未执行。") from error
        if placed is not model:
            raise UserError("EVALUATION_DEVICE_PLACEMENT_UNCONFIRMED",
                            "模型设备迁移未返回原模型实例；评估入口尚未执行。",
                            "检查自定义 to() 实现；迁移可能已有副作用。")
        return selection.report()

    return _run_evaluation(
        EvaluationRuntime(
            torch=torch,
            resolve_entrypoint=entrypoint,
            verify_manifest=verify_manifest,
            scalar_value=scalar_value,
            source_manifest=source_manifest,
            prepare_device=prepare_evaluation_device,
        ),
        model,
        request,
    )


def run_manual_steps(model, request, restore=None, checkpoint_callback=None):
    return _run_manual_steps(
        _training_runtime(), model, request, restore, checkpoint_callback
    )


def run_managed_steps(model, request, restore=None, checkpoint_callback=None):
    if request.get("apply_supervision") is True and "_training_supervision" not in request:
        request["_training_supervision"] = _prepare_training_supervision(model, request, torch)
        request["supervision_snapshot"] = request["_training_supervision"].snapshot()
    return _run_managed_steps(
        _training_runtime(), model, request, restore, checkpoint_callback
    )


def probe(model, request):
    return _probe(
        model, request,
        torch_module=torch,
        resolve_entrypoint=entrypoint,
        make_batch_fn=make_batch,
        module_entries_fn=module_entries,
        normalize_path_fn=normalize_path,
        resolve_module_fn=resolve_module,
        tree_fn=tree,
        max_calls=MAX_CALLS,
    )


def verify_gradients(model, request):
    return _verify_gradients(
        model, request,
        torch_module=torch,
        resolve_entrypoint=entrypoint,
        make_batch_fn=make_batch,
        tree_fn=tree,
    )


def validate_model_injections(value):
    return _validate_model_injections(value)


def construct_model(factory, params, project_root, injections=None):
    global SOURCE_LOCATIONS, SOURCE_SITE_COUNTS
    SOURCE_LOCATIONS = {}
    SOURCE_SITE_COUNTS = {}
    model, evidence, locations, site_counts = _construct_model(
        factory, params, _ACTIVE_CODE_ROOT or project_root, injections,
        torch_module=torch,
        resolve_entrypoint=entrypoint,
        module_entries_fn=module_entries,
    )
    SOURCE_LOCATIONS = locations
    SOURCE_SITE_COUNTS = site_counts
    return model, evidence


def editable_params(module):
    return _editable_params(module, torch)


def source_editable_fields(module):
    return _source_editable_fields(module, torch)


def source_set_block_reason(module):
    return _source_set_block_reason(module, torch)


def inspect_model(model, path=None, layer_type=None, search=None):
    return _inspect_model(
        model, path, layer_type, search,
        torch_module=torch,
        source_locations=SOURCE_LOCATIONS,
        source_site_counts=SOURCE_SITE_COUNTS,
        layer_container_kind_fn=layer_container_kind,
    )


def sha256_file(path):
    return _sha256_file(path)


def _component_wheel_runtime():
    return ComponentWheelRuntime(
        distribution_name_key=distribution_name_key,
        lock_file_paths=lock_file_paths,
        validate_environment_lock=validate_environment_lock,
        python_identity=python_identity,
        requirements_content_digest=requirements_content_digest,
        python_executable=sys.executable,
    )


def wheel_runtime_requirements(wheel_path):
    return _wheel_runtime_requirements(_component_wheel_runtime(), wheel_path)


def component_install(request):
    runtime_options = request.get("runtime", {})
    project_root = Path(runtime_options.get("project_root", Path.cwd())).resolve(strict=True)
    runtime = ComponentInstallRuntime(
        safe_project_path=safe_project_path,
        distribution_name_key=distribution_name_key,
        run_environment_process=run_environment_process,
        wheel_runtime_requirements=wheel_runtime_requirements,
        sha256_file=sha256_file,
        discover_components=discover_components,
        project_root=project_root,
        python_executable=sys.executable,
    )
    return _component_install(runtime, request)


def _component_source_imports(project_root, manifest, extension_root):
    return component_source_imports(
        project_root, manifest, extension_root, selected_environment_venv, safe_project_path
    )


def component_uninstall(request):
    runtime = ComponentUninstallRuntime(
        component_json_files=_component_json_files,
        component_reference_paths=_component_reference_paths,
        component_source_imports=_component_source_imports,
    )
    return _component_uninstall(runtime, request)


def environment_snapshot(capture_timing="export_time"):
    return snapshot_environment(globals().get("torch"), capture_timing)


def verify_component_distribution(manifest):
    runtime = ComponentProvenanceRuntime(
        search_path_entries=tuple(sys.path),
        current_directory=os.getcwd(),
        loaded_module_path=_loaded_module_path,
        prepend_import_path=lambda path: sys.path.insert(0, path),
    )
    return _verify_component_distribution(runtime, manifest)


def _loaded_module_path(module_name):
    loaded = sys.modules.get(module_name)
    return getattr(loaded, "__file__", None) if loaded is not None else None


def extension_bundle_plan(request):
    runtime = ExtensionBundleRuntime(
        verify_component_distribution=verify_component_distribution,
        distribution_name_key=distribution_name_key,
        safe_distribution_file=safe_distribution_file,
        sha256_file=sha256_file,
        search_path_entries=tuple(sys.path),
        current_directory=os.getcwd(),
    )
    return _extension_bundle_plan(runtime, request)


def _edit_validation_runtime():
    return EditValidationRuntime(
        valid_edit_actions=VALID_EDIT_ACTIONS,
        digest=digest,
        validate_model_injections=validate_model_injections,
    )


def valid_edit_operation(operation):
    return _valid_edit_operation(_edit_validation_runtime(), operation)


def component_operations(edit):
    yield from _component_operations(edit)


def module_source_fingerprint(module_name):
    return _module_source_fingerprint(module_name)


def _extension_reference_runtime():
    return ExtensionReferenceRuntime(
        verify_component_distribution=verify_component_distribution,
        digest=digest,
        conformance_suite_version=PROVIDER_CONFORMANCE_SUITE_VERSION,
        python_version=sys.version.split()[0],
        torch_version=str(torch.__version__),
    )


def verify_component_references(edit, project_root, allow_unverified=False):
    return _verify_component_references(
        _extension_reference_runtime(), edit, project_root, allow_unverified
    )


def verify_model_injection_references(injections, project_root):
    return _verify_model_injection_references(
        _extension_reference_runtime(), injections, project_root
    )


def entrypoint(value):
    return _entrypoint(value)


def source_manifest(root):
    return _source_manifest(
        root,
        list(sys.modules.values()),
        code_root=_ACTIVE_CODE_ROOT,
        import_roots=_ACTIVE_IMPORT_ROOTS,
    )


def verify_manifest(manifest):
    if _ACTIVE_CODE_ROOT is None or _ACTIVE_PROJECT_ROOT is None:
        return _verify_manifest(manifest)
    mapped = {}
    project_root = Path(_ACTIVE_PROJECT_ROOT).resolve()
    code_root = Path(_ACTIVE_CODE_ROOT).resolve()
    for filename, digest in manifest.items():
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(project_root)
        except ValueError:
            mapped[filename] = digest
        else:
            mapped[str(code_root / relative)] = digest
    return _verify_manifest(mapped)


def _training_state_runtime():
    return TrainingStateRuntime(torch=torch)


def scalar_value(value, field):
    return _scalar_value(_training_state_runtime(), value, field)


def scalar_map(value, field):
    return _scalar_map(_training_state_runtime(), value, field)


def add_finite_total(totals, counts, name, value, field):
    return _add_finite_total(totals, counts, name, value, field)


def callable_checkpoint_hooks(function, label):
    return _callable_checkpoint_hooks(function, label)


def restore_callable_checkpoint(function, saved, label, allow_reset):
    return _restore_callable_checkpoint(function, saved, label, allow_reset)


def checkpoint_rng_state():
    return _checkpoint_rng_state(_training_state_runtime())


def restore_rng_state(state):
    return _restore_rng_state(_training_state_runtime(), state)


def _attach_augmentation_report(result, request):
    configured = request.get("train_augment", [])
    if configured:
        result["augmentation"] = {
            "status": "applied" if request.get("_augmentation_batch_count", 0) else "not_applied",
            "configured_steps": len(configured),
            "transformed_batches": request.get("_augmentation_batch_count", 0),
            "last_batch": request.get("_last_augmentation_observations", []),
            "run_signature_includes_pipeline": True,
        }
    else:
        result["augmentation"] = {"status": "not_configured", "configured_steps": 0}
    return result


def run_training_probe(model, request):
    if request.get("apply_supervision") is True and "_training_supervision" not in request:
        request["_training_supervision"] = _prepare_training_supervision(model, request, torch)
        request["supervision_snapshot"] = request["_training_supervision"].snapshot()
    _prepare_training_device(model, request)
    try:
        result = _run_training_mode(
            TrainingModeRuntime(
                run_managed_steps=run_managed_steps,
                run_manual_steps=run_manual_steps,
                run_external_loop=run_external_loop,
            ),
            model,
            request,
        )
    except Exception as error:
        _raise_device_oom(request, error, "training probe")
        raise
    _finish_training_resources(request)
    return _attach_augmentation_report(result, request)


def training_probe_with_source_evidence(model, request):
    if request.get("apply_supervision") is True and "_training_supervision" not in request:
        request["_training_supervision"] = _prepare_training_supervision(model, request, torch)
        request["supervision_snapshot"] = request["_training_supervision"].snapshot()
    return _training_probe_with_source_evidence(
        TrainingProbeRuntime(
            environment_snapshot=environment_snapshot,
            entrypoint=entrypoint,
            source_manifest=source_manifest,
            run_training_run=run_training_run,
            run_training_probe=run_training_probe,
            python_executable=sys.executable,
            python_version=sys.version.split()[0],
            pytorch_version=str(torch.__version__),
        ),
        model,
        request,
    )


def run_training_run(model, request, source_manifest):
    if request.get("apply_supervision") is True and "_training_supervision" not in request:
        request["_training_supervision"] = _prepare_training_supervision(model, request, torch)
        request["supervision_snapshot"] = request["_training_supervision"].snapshot()
    selection = _prepare_training_device(model, request)
    try:
        result = _run_training_run(
            TrainingRunRuntime(
                entrypoint=entrypoint,
                load_training_checkpoint=load_training_checkpoint,
                load_verified_checkpoint_payload=load_verified_checkpoint_payload,
                migrate_state_dict=migrate_state_dict,
                state_dict_digest=state_dict_digest,
                model_state_alias_groups=model_state_alias_groups,
                restore_callable_checkpoint=restore_callable_checkpoint,
                restore_rng_state=restore_rng_state,
                training_run_signature=training_run_signature,
                checkpoint_value_supported=checkpoint_value_supported,
                callable_checkpoint_hooks=callable_checkpoint_hooks,
                checkpoint_rng_state=checkpoint_rng_state,
                model_state_digest=model_state_digest,
                save_training_checkpoint=save_training_checkpoint,
                run_external_loop=run_external_loop,
                run_managed_steps=run_managed_steps,
                run_manual_steps=run_manual_steps,
                emit_progress=_emit_training_progress,
                python_version=sys.version.split()[0],
                pytorch_version=str(torch.__version__),
            ),
            model,
            request,
            source_manifest,
        )
    except Exception as error:
        _raise_device_oom(request, error, "training run")
        raise
    _finish_training_resources(request)
    result["runtime"] = {**result.get("runtime", {}), **selection.report()}
    result["resources"] = request["_resource_report"]
    return _attach_augmentation_report(result, request)


def _emit_training_progress(event):
    event = dict(event)
    event.setdefault("at_ms", time.time_ns() // 1_000_000)
    print("MODEL_MAVEN_PROGRESS " + json.dumps(event), file=sys.stderr, flush=True)


def _model_loading_runtime():
    return ModelLoadingRuntime(
        torch=torch,
        validate_variant=validate_variant,
        validate_model_injections=validate_model_injections,
        verify_manifest=verify_manifest,
        verify_model_injection_references=verify_model_injection_references,
        entrypoint=entrypoint,
        construct_model=construct_model,
        verify_component_references=verify_component_references,
        apply_history_item=apply_history_item,
    )


def load_model(request, *, reset_rng=True):
    return _load_model(_model_loading_runtime(), request, reset_rng=reset_rng)


def validate_variant(variant):
    return _validate_variant(_edit_validation_runtime(), variant)


def _layer_edit_runtime():
    return LayerEditRuntime(
        torch=torch,
        entrypoint=entrypoint,
        resolve_module=resolve_module,
        module_entries=module_entries,
        normalize_path=normalize_path,
        editable_params=editable_params,
    )


def new_layer(edit, *, preserve_factory_device=False):
    return _new_layer(_layer_edit_runtime(), edit, preserve_factory_device=preserve_factory_device)


def layer_container_kind(module):
    return _layer_container_kind(_layer_edit_runtime(), module)


def replace_module(model, path, replacement):
    return _replace_module(_layer_edit_runtime(), model, path, replacement)


def apply_edit(model, edit):
    return _apply_edit(_layer_edit_runtime(), model, edit)


def apply_history_item(model, item, *, preserve_factory_device=False):
    return _apply_history_item(_layer_edit_runtime(), model, item, preserve_factory_device=preserve_factory_device)


def _edit_plan_runtime():
    return EditPlanRuntime(
        source_edit_candidate=source_edit_candidate,
        validate_variant=validate_variant,
        valid_edit_operation=valid_edit_operation,
        digest=digest,
        load_model=load_model,
        verify_component_references=verify_component_references,
        apply_history_item=apply_history_item,
        probe=probe,
        source_manifest=source_manifest,
        component_operations=component_operations,
        verify_manifest=verify_manifest,
        torch_version=str(torch.__version__),
        inspect_model=inspect_model,
    )


def edit_candidate(request):
    return _edit_candidate(_edit_plan_runtime(), request)


def ast_offset(*args, **kwargs):
    return _source_ast_offset(*args, **kwargs)


def call_name(*args, **kwargs):
    return _source_call_name(*args, **kwargs)


def unique_source_alias(*args, **kwargs):
    return _source_unique_source_alias(*args, **kwargs)


def source_module_expression(*args, **kwargs):
    return _source_source_module_expression(*args, **kwargs)


def source_factory_expression(*args, **kwargs):
    return _source_source_factory_expression(*args, **kwargs)


def source_sequential_rebind(*args, **kwargs):
    return _source_source_sequential_rebind(*args, **kwargs)


def _source_candidate_runtime():
    return SourceCandidateRuntime(
        load_model=load_model,
        source_manifest=source_manifest,
        entrypoint=entrypoint,
        normalize_path=normalize_path,
        resolve_module=resolve_module,
        module_entries=module_entries,
        source_locations=lambda: SOURCE_LOCATIONS,
        source_site_counts=lambda: SOURCE_SITE_COUNTS,
        torch=torch,
        call_name=call_name,
        unique_source_alias=unique_source_alias,
        source_factory_expression=source_factory_expression,
        source_module_expression=source_module_expression,
        source_sequential_rebind=source_sequential_rebind,
        new_layer=lambda edit: new_layer(edit, preserve_factory_device=True),
        probe=probe,
        inspect_model=inspect_model,
        editable_params=editable_params,
        apply_edit=apply_edit,
        source_editable_fields=source_editable_fields,
        has_module_state=has_module_state,
        replace_module=replace_module,
        ast_offset=ast_offset,
        apply_source_set_candidate=apply_source_set_candidate,
    )


def source_sequential_candidate(request, edit):
    return _source_sequential_candidate(_source_candidate_runtime(), request, edit)


def source_sequential_transaction_state(model, root, container_path, container, namespace):
    return _source_sequential_transaction_state(
        _source_candidate_runtime(), model, root, container_path, container, namespace
    )


def source_sequential_transaction_candidate(request, edit):
    return _source_sequential_transaction_candidate(
        _source_candidate_runtime(), request, edit
    )


def apply_source_set_candidate(model, operation, call):
    return _apply_source_set_candidate(
        _source_candidate_runtime(), model, operation, call
    )


def source_set_transaction_candidate(request, edit):
    return _source_set_transaction_candidate(
        _source_candidate_runtime(), request, edit
    )


def _source_edit_dispatch_runtime():
    from model_maven_worker.edit.starter_migration import starter_migration_candidate

    return SourceEditDispatchRuntime(
        source_set_candidate=lambda request, edit: _source_set_candidate(
            _source_candidate_runtime(), request, edit
        ),
        source_sequential_candidate=source_sequential_candidate,
        source_set_transaction_candidate=source_set_transaction_candidate,
        source_sequential_transaction_candidate=source_sequential_transaction_candidate,
        source_mixed_transaction_candidate=lambda request, edit: _source_mixed_transaction_candidate(
            _source_candidate_runtime(), request, edit
        ),
        starter_migration_candidate=lambda request, edit: starter_migration_candidate(
            _source_candidate_runtime(), request, edit
        ),
    )


def source_edit_candidate(request):
    return _source_edit_candidate(_source_edit_dispatch_runtime(), request)


def doctor():
    import importlib.util
    from model_maven_worker.environment.common import redact_environment_output

    present = importlib.util.find_spec("torch") is not None
    version = None
    import_error = None
    if present:
        try:
            import torch
            version = str(torch.__version__)
        except Exception as error:
            present = False
            import_error = redact_environment_output(f"{type(error).__name__}: {error}", limit=1000)
    return _doctor_report(
        python=sys.executable,
        python_version=sys.version.split()[0],
        torch_available=present,
        torch_version=version,
        torch_import_error=import_error,
        platform=sys.platform,
        compatibility=compatibility_report(),
    )


def component_test(request):
    return _component_test(
        ExtensionConformanceRuntime(
            torch=torch,
            add_project_import_path=lambda path: sys.path.insert(0, path),
            verify_component_references=verify_component_references,
            new_layer=new_layer,
            make_batch=make_batch,
            entrypoint=entrypoint,
            tree=tree,
            apply_edit=apply_edit,
            digest=digest,
            conformance_suite_version=PROVIDER_CONFORMANCE_SUITE_VERSION,
            module_source_fingerprint=module_source_fingerprint,
            python_version=sys.version.split()[0],
            torch_version=str(torch.__version__),
            run_transform_pipeline=_run_transform_pipeline,
            run_external_loop=lambda model, request, loop_fn: _run_external_loop(
                _training_runtime(), model, request, loop_fn=loop_fn
            ),
        ),
        request,
    )


def dispatch(request):
    validate_request(request)
    operation = request["operation"]
    if operation == "probe":
        # Capture the contract before project model/data imports. The probe
        # uses this exact validated object even if the file changes later.
        request["_resolved_contract"] = _load_contract(request)
    runtime_options = request.get("runtime", {})
    timeout_seconds = (
        runtime_options.get("timeout_seconds")
        if isinstance(runtime_options, dict)
        else None
    )
    if operation in {"training_probe", "generic_loop_trial", "generic_loop_train", "train"} and type(timeout_seconds) is int and timeout_seconds > 0:
        request["_worker_deadline_monotonic"] = _WORKER_STARTED_AT + timeout_seconds
    global _ACTIVE_CODE_ROOT, _ACTIVE_PROJECT_ROOT, _ACTIVE_IMPORT_ROOTS
    _ACTIVE_CODE_ROOT = None
    _ACTIVE_PROJECT_ROOT = None
    roots = runtime_options.get("project_import_roots") if isinstance(runtime_options, dict) else None
    _ACTIVE_IMPORT_ROOTS = roots if isinstance(roots, list) else None
    if operation == "doctor":
        return doctor()
    if operation == "runtime_capabilities":
        # Metadata only: a framework-neutral loop must not import torch as a
        # side effect of preparing its own runtime.
        result = _doctor_report(
            python=sys.executable, python_version=sys.version.split()[0],
            torch_available=None, torch_version=None, platform=sys.platform,
            compatibility=compatibility_report(),
        )
        result["capability_probe"] = "worker_declarations_only"
        return result
    if operation == "data_index":
        return _run_index_operation(request)
    if operation == "preset_catalog":
        from model_maven_worker.data.presets import describe_presets

        catalog = describe_presets()
        query = request.get("query")
        if query is None or not str(query).strip():
            return catalog
        needle = str(query).strip().casefold()
        matches = {
            name: preset
            for name, preset in catalog["presets"].items()
            if needle in name.casefold()
            or needle in preset.get("task", "").casefold()
            or needle in preset.get("model", "").casefold()
        }
        return {**catalog, "presets": matches, "query": str(query).strip()}
    if operation == "semantic_mark":
        action = request.get("edit")
        if isinstance(action, dict) and action.get("action") == "list":
            return _list_semantic_marks(request["project_root"])
        if isinstance(action, dict) and action.get("action") == "remove":
            return _remove_semantic_mark(request["project_root"], action.get("id"))
        if not isinstance(action, dict) or action.get("action") not in {"add", "update", "audit"}:
            raise UserError("SEMANTIC_ACTION_INVALID", "语义标记操作需要 add、update、list、audit 或 remove。")
    if operation == "semantic_supervise":
        action = request.get("edit")
        if not isinstance(action, dict):
            raise UserError("SUPERVISION_ACTION_INVALID", "监督规则操作缺少动作描述。")
        if action.get("action") == "rule_add":
            plugin = None
            kind = action.get("kind")
            if kind in {"custom-validator", "custom-penalty"}:
                # This metadata operation explicitly imports user code. Match
                # the chosen runtime before resolving the rule callable.
                project = str(Path(request["project_root"]).resolve())
                if project not in sys.path:
                    sys.path.insert(0, project)
                _select_user_working_directory(request)
                configure_project_import_roots(request)
                raw_params = action.get("plugin_params", [])
                if not isinstance(raw_params, list) or len(raw_params) > 64:
                    raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", "自定义规则参数须为最多 64 项的 NAME=JSON 数组。")
                plugin_params = {}
                for raw in raw_params:
                    if not isinstance(raw, str) or "=" not in raw:
                        raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", "自定义规则参数格式为 NAME=JSON。")
                    name, encoded = raw.split("=", 1)
                    if not name.isidentifier() or name in plugin_params:
                        raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", f"自定义规则参数名无效或重复：{name}。")
                    try:
                        plugin_params[name] = json.loads(encoded, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                    except (json.JSONDecodeError, ValueError) as error:
                        raise UserError("SUPERVISION_PLUGIN_PARAMS_INVALID", f"自定义规则参数 {name} 不是有效 JSON。") from error
                plugin = _pin_custom_supervision_plugin(
                    request["project_root"], kind,
                    action.get("plugin_entrypoint"), plugin_params,
                    source_kind=action.get("plugin_source") or "project",
                )
            return _add_supervision_rule(
                request["project_root"], action.get("mark_id"),
                kind=kind, mode=action.get("mode"),
                minimum=action.get("min"), maximum=action.get("max"),
                weight=action.get("weight"), target_path=action.get("target_path"),
                axis=action.get("axis"), rule_id=action.get("id"), plugin=plugin,
                expected_state_revision=action.get("if_state_revision"),
            )
        if action.get("action") == "bundle_export":
            return _export_supervision_bundle(request["project_root"], action.get("if_state_revision"))
        if action.get("action") == "bundle_import":
            return _import_supervision_bundle(request["project_root"], action.get("file"), action.get("if_sha256"), action.get("if_state_revision"), action.get("apply") is True)
        if action.get("action") == "rule_copy":
            return _copy_supervision_rule(request["project_root"], action.get("id"), action.get("mark_id"), action.get("if_state_revision"))
        if action.get("action") == "rule_list":
            return _list_supervision_rules(request["project_root"])
        if action.get("action") == "rule_remove":
            expected = action.get("expected_rule")
            if expected is not None:
                try:
                    expected = json.loads(expected)
                except (TypeError, ValueError) as error:
                    raise UserError("SUPERVISION_RULE_EXPECTATION_INVALID", "预期规则必须是有效的 JSON 对象。") from error
                if not isinstance(expected, dict):
                    raise UserError("SUPERVISION_RULE_EXPECTATION_INVALID", "预期规则必须是有效的 JSON 对象。")
            return _remove_supervision_rule(request["project_root"], action.get("id"), expected_rule=expected)
        if action.get("action") != "preview":
            raise UserError("SUPERVISION_ACTION_INVALID", "监督规则操作需要 rule_add、rule_list、rule_remove 或 preview。")
    if operation == "assembly_codegen":
        recipe = _validate_assembly_recipe(request.get("assembly_recipe"))
        return {"source": _generate_assembly_code(recipe), "recipe": recipe}
    if operation == "component_install":
        return component_install(request)
    if operation == "component_uninstall":
        return component_uninstall(request)
    if operation == "component_discover":
        return discover_components(request)
    if operation == "extension_bundle_plan":
        return extension_bundle_plan(request)
    if operation == "environment_snapshot":
        return environment_snapshot()
    if operation in {"environment_lock", "environment_setup", "environment_configure", "environment_check"}:
        return environment_operation(request)
    if operation == "environment_library_inventory":
        return inspect_environment_libraries(request)
    if operation == "environment_library_probe":
        return probe_environment_libraries(request)
    if operation == "environment_devices":
        return probe_environment_libraries({**request, "libraries": ["torch"], "device_inventory": True})
    if operation == "environment_wheelhouse_archive":
        return archive_environment_wheels(
            request.get("runtime", {}).get("python_executable") or sys.executable,
            request.get("relative"),
            request.get("interactive_progress") is True,
        )
    if operation == "environment_wheelhouse_check":
        return inspect_wheel_archive(
            request.get("runtime", {}).get("python_executable") or sys.executable,
            request.get("relative"),
        )
    if operation == "environment_wheelhouse_replay":
        return replay_wheel_archive(
            request.get("runtime", {}).get("python_executable") or sys.executable,
            request.get("venv"),
            request.get("relative"),
            request.get("interactive_progress") is True,
            request.get("create_environment") is True,
        )
    if operation in {"training_probe", "generic_loop_train", "train", "evaluate"} and request.get("snapshot_dir"):
        variant = request.get("variant")
        required_source_manifest = (
            variant.get("source_manifest", {}) if isinstance(variant, dict) else {}
        )
        snapshot = snapshot_project_code(
            request["project_root"],
            request["snapshot_dir"],
            required_source_manifest,
            request.get("runtime", {}).get("project_import_roots"),
        )
        _ACTIVE_CODE_ROOT = snapshot["root"]
        _ACTIVE_PROJECT_ROOT = str(Path(request["project_root"]).resolve())
        request["code_snapshot"] = snapshot
        request["code_snapshot_root"] = _ACTIVE_CODE_ROOT
        sys.path.insert(0, _ACTIVE_CODE_ROOT)
    _select_user_working_directory(request)
    if operation == "generic_loop_train":
        import hashlib
        from model_maven_worker.run.provenance import check_snapshot_project_imports
        from model_maven_worker.train.generic_checkpoint import load_generic_checkpoint, save_generic_checkpoint
        from model_maven_worker.train.generic_loop import GenericLoopContext, run_generic_loop

        if request.get("model") is not None or request.get("apply_supervision") is True:
            raise UserError("GENERIC_LOOP_UNSUPPORTED", "纯 Python 用户循环不接管模型或监督规则。")
        snapshot = request.get("code_snapshot")
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("sha256"), str):
            raise UserError("SOURCE_SNAPSHOT_REQUIRED", "纯 Python 正式运行需要源码快照。")
        reference, params = request.get("training_loop"), request.get("training_params", {})
        loop_interface = request.get("loop_interface", "model-context")
        if not isinstance(loop_interface, str) or loop_interface not in {"context", "model-context"}:
            raise UserError("GENERIC_LOOP_INVALID", "循环签名必须为 context 或 model-context。")
        if not isinstance(reference, str) or not reference or not isinstance(params, dict):
            raise UserError("GENERIC_LOOP_INVALID", "需要用户循环入口和参数对象。")
        directory = request.get("checkpoint_dir")
        if not isinstance(directory, str) or not directory:
            raise UserError("GENERIC_CHECKPOINT_PATH_INVALID", "纯 Python 正式运行需要 checkpoint 目录。")
        def digest_json(value):
            try:
                raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError, RecursionError) as error:
                raise UserError("GENERIC_LOOP_INVALID", "循环参数和运行路径必须是有限 JSON。") from error
            return hashlib.sha256(raw).hexdigest()
        inventory = environment_snapshot("training_start")
        runtime = request.get("runtime") or {}
        identity = {"loop": reference, "python_executable": sys.executable,
                    "python_version": sys.version.split()[0], "source_manifest_sha256": snapshot["sha256"],
                    "environment_fingerprint_sha256": inventory.get("fingerprint_sha256"),
                    "parameters_sha256": digest_json(params),
                    "runtime_paths_sha256": digest_json({"working_directory": runtime.get("working_directory"),
                                                          "project_import_roots": runtime.get("project_import_roots")})}
        if loop_interface == "context":
            identity["loop_interface"] = "context"
        prior = None
        if request.get("generic_checkpoint_input") is not None:
            prior = load_generic_checkpoint(request["generic_checkpoint_input"], expected_identity=identity,
                                            expected_sha256=request.get("expected_checkpoint_sha256"))
        code_root = request["code_snapshot_root"]
        def check_snapshot():
            from model_maven_worker.run.file_digest import sha256_file
            root = Path(code_root).resolve(strict=True)
            manifest = snapshot.get("manifest")
            if not isinstance(manifest, dict):
                raise UserError("SOURCE_SNAPSHOT_INVALID", "纯 Python 运行缺少源码快照清单。")
            for relative, expected in manifest.items():
                if not isinstance(relative, str) or not isinstance(expected, str):
                    raise UserError("SOURCE_SNAPSHOT_INVALID", "源码快照清单无效。")
                path = root / relative
                try:
                    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                            or sha256_file(path) != expected):
                        raise ValueError(relative)
                except (OSError, ValueError) as error:
                    raise UserError("SOURCE_SNAPSHOT_CHANGED", f"源码快照已变化：{relative}",
                                    "恢复点不会作为基于原源码的结果保存。") from error
            check_snapshot_project_imports(request["project_root"], code_root,
                                           import_roots=runtime.get("project_import_roots"))
        if code_root in sys.path:
            sys.path.remove(code_root)
        sys.path.insert(0, code_root)
        configure_project_import_roots(request)
        loop = entrypoint(reference)
        check_snapshot()
        latest = {"manifest": None}
        def emit(step, metrics, state):
            saved = False
            if state is not None:
                check_snapshot()
                previous = latest["manifest"]
                latest["manifest"] = save_generic_checkpoint(directory, completed_steps=step, state=state,
                    identity=identity, expected_previous=previous["sha256"] if previous else None)
                saved = True
            _emit_training_progress({"step": step, "total": request.get("steps"),
                                     "metrics": metrics, "checkpoint": saved})
        context = GenericLoopContext(request.get("steps"), completed_steps=prior["completed_steps"] if prior else 0,
                                     resume_state=prior["state"] if prior else None,
                                     stop_file=request.get("stop_file"),
                                     deadline_monotonic=request.get("_worker_deadline_monotonic"),
                                     report_callback=emit)
        try:
            result = run_generic_loop(loop, context, params, interface=loop_interface)
            check_snapshot()
        except UserError as error:
            # Preserve the original diagnostic while separating the user's
            # last accepted report from the last durable state. The stored
            # checkpoint is independently verified when a recovery is chosen.
            details = dict(error.data) if isinstance(error.data, dict) else {}
            saved_checkpoint = latest["manifest"]
            details.update({
                "training_mode": "generic-external-loop",
                "loop_interface": loop_interface,
                "completed_steps": context.completed_steps,
                "requested_steps": context.total_steps,
                "last_metrics": context._last_metrics,
                "checkpoint_saved": saved_checkpoint is not None,
                "checkpoint": saved_checkpoint,
                "checkpoint_completed_steps": saved_checkpoint["completed_steps"] if saved_checkpoint else None,
                "recovery_status": "checkpoint_recorded_verify_before_resume" if saved_checkpoint else "no_user_state_reported",
                "state_scope": "user_declared_json; external files and RNG are not captured",
            })
            error.data = details
            raise
        saved_checkpoint = latest["manifest"]
        result.update({"checkpoint_saved": saved_checkpoint is not None, "checkpoint": saved_checkpoint,
                       "checkpoint_completed_steps": saved_checkpoint["completed_steps"] if saved_checkpoint else None,
                       "source_snapshot": snapshot, "environment_snapshot": inventory,
                       "runtime": {"python_executable": sys.executable,
                                   "python_version": sys.version.split()[0], "framework": "user_managed"},
                       "state_scope": "user_declared_json; external files and RNG are not captured",
                       "recovery_status": "checkpoint_available" if saved_checkpoint else "no_user_state_reported"})
        return result
    if operation == "generic_loop_trial":
        from model_maven_worker.train.generic_loop import GenericLoopContext, run_generic_loop

        if request.get("model") is not None or request.get("apply_supervision") is True:
            raise UserError("GENERIC_LOOP_UNSUPPORTED", "纯 Python 循环试跑不接管模型或监督规则。")
        loop_interface = request.get("loop_interface", "model-context")
        if not isinstance(loop_interface, str) or loop_interface not in {"context", "model-context"}:
            raise UserError("GENERIC_LOOP_INVALID", "循环签名必须为 context 或 model-context。")
        reference = request.get("training_loop")
        if not isinstance(reference, str) or not reference:
            raise UserError("EXTERNAL_LOOP_REQUIRED", "请选择用户循环入口。")
        project_root = str(Path(request["project_root"]).resolve(strict=True))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        configure_project_import_roots(request)
        loop = entrypoint(reference)
        before = source_manifest(project_root)
        def emit(step, metrics, state):
            _emit_training_progress({"step": step, "total": request.get("steps"),
                                     "metrics": metrics, "checkpoint": False})
        context = GenericLoopContext(request.get("steps"), stop_file=request.get("stop_file"),
                                     deadline_monotonic=request.get("_worker_deadline_monotonic"),
                                     report_callback=emit)
        result = run_generic_loop(loop, context, request.get("training_params", {}), interface=loop_interface)
        after = source_manifest(project_root)
        before_paths, after_paths = set(before), set(after)
        changed_paths = sorted(path for path in before_paths & after_paths if before[path] != after[path])
        added_paths = sorted(after_paths - before_paths)
        missing_paths = sorted(before_paths - after_paths)
        result.update({"source_manifest": before, "source_changed_during_run": before != after,
                       "source_observation_delta": {
                           "scope": "loaded_project_modules_and_package_trees",
                           "baseline_timing": "after_loop_entrypoint_import",
                           "changed_content_count": len(changed_paths),
                           "newly_observed_count": len(added_paths),
                           "no_longer_observed_count": len(missing_paths),
                           "changed_content_paths": changed_paths[:32],
                           "newly_observed_paths": added_paths[:32],
                           "no_longer_observed_paths": missing_paths[:32],
                           "paths_truncated": any(len(paths) > 32 for paths in (changed_paths, added_paths, missing_paths)),
                       },
                       "runtime": {"python_executable": sys.executable,
                                   "python_version": sys.version.split()[0],
                                   "framework": "user_managed"},
                       "checkpoint_saved": False,
                       "state_persistence": "not_available_for_trial"})
        return result
    global torch
    try:
        import torch
    except ImportError as error:
        raise UserError("TORCH_UNAVAILABLE", "指定 Python 环境没有可导入的 PyTorch。", "用 --python 指定已安装 PyTorch 的解释器。") from error
    if operation == "component_test":
        return component_test(request)
    if operation == "graph_draft_probe":
        import hashlib
        import random
        import types
        source = request.get("generated_source")
        expected_source = request.get("generated_source_sha256")
        expected_input = request.get("expected_input_content")
        if not isinstance(source, str) or len(source.encode("utf-8")) > 4 * 1024 * 1024 \
                or hashlib.sha256(source.encode("utf-8")).hexdigest() != expected_source:
            raise UserError("GRAPH_DRAFT_SOURCE_INVALID", "图草稿源码缺失、过大或摘要不匹配。")
        if not isinstance(expected_input, str) or len(expected_input) != 64 or any(c not in "0123456789abcdef" for c in expected_input) \
                or not isinstance(request.get("replay_input_from"), str):
            raise UserError("GRAPH_DRAFT_INPUT_REQUIRED", "草稿预览需要可重放的已捕获输入及其摘要。")
        project_root = str(Path(request["project_root"]).resolve())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        configure_project_import_roots(request)
        module_name = "_model_maven_graph_draft_" + expected_source[:24]
        if module_name in sys.modules:
            raise UserError("GRAPH_DRAFT_MODULE_CONFLICT", "草稿模块名称已被使用。")
        module = types.ModuleType(module_name)
        module.__file__ = "<model-maven-graph-draft>"
        sys.modules[module_name] = module
        try:
            seed = request.get("seed")
            if seed is not None:
                if type(seed) is not int or not 0 <= seed < 2**64:
                    raise UserError("GRAPH_DRAFT_SEED_INVALID", "种子必须是无符号 64 位整数。")
                random.seed(seed)
                torch.manual_seed(seed)
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            model = module.model()
            if not isinstance(model, torch.nn.Module):
                raise UserError("GRAPH_DRAFT_MODEL_INVALID", "草稿工厂没有返回 torch.nn.Module。")
            requested_device = request.get("probe_device")
            if requested_device is not None:
                if not isinstance(requested_device, str) or not requested_device or len(requested_device) > 128:
                    raise UserError("PROBE_DEVICE_INVALID", "前向设备标识无效。")
                selection = _resolve_device(torch, requested_device)
                try:
                    placed = model.to(selection.device)
                except Exception as error:
                    raise UserError("PROBE_DEVICE_PLACEMENT_FAILED", f"无法迁移草稿模型：{type(error).__name__}: {error}") from error
                if placed is not model:
                    raise UserError("PROBE_DEVICE_PLACEMENT_UNCONFIRMED", "模型设备迁移没有返回原模型实例。")
                request["_probe_device_report"] = selection.report()
            request["model"] = module_name + ":model"
            request["_disable_evidence_cache"] = True
            result = probe(model, request)
            result["draft_source_sha256"] = expected_source
            result["draft_graph_sha256"] = request.get("draft_graph_sha256")
            result["preview_scope"] = "fresh-worker-graph-draft"
            result["weights_loaded"] = False
            result["runtime"] = {"python_executable": sys.executable, "python_version": sys.version.split()[0],
                                 "pytorch_version": torch.__version__, "project_import_roots": request.get("runtime", {}).get("project_import_roots"),
                                 "probe_device": request.get("_probe_device_report")}
            return result
        except UserError as error:
            error.data["draft_source_sha256"] = expected_source
            error.data["draft_graph_sha256"] = request.get("draft_graph_sha256")
            error.data["preview_scope"] = "fresh-worker-graph-draft"
            error.data["weights_loaded"] = False
            if error.code == "FORWARD_FAILED" and error.__cause__ is not None:
                frames = traceback.extract_tb(error.__cause__.__traceback__)
                lines = [frame.lineno for frame in frames
                         if frame.filename == module.__file__ and frame.name == "forward"]
                if lines:
                    error.data["draft_forward_line"] = lines[-1]
            raise
        finally:
            sys.modules.pop(module_name, None)
    if operation == "assembly_codegen_verify":
        import importlib.util
        import tempfile
        source = request.get("generated_source")
        if not isinstance(source, str) or len(source.encode("utf-8")) > 4 * 1024 * 1024:
            raise UserError("ASSEMBLY_GENERATED_SOURCE_INVALID", "生成源码缺失或超过 4 MiB。")
        project_root = str(Path(request["project_root"]).resolve())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        configure_project_import_roots(request)
        with tempfile.TemporaryDirectory(prefix="model-maven-codegen-") as directory:
            generated = Path(directory) / "generated_assembly.py"
            generated.write_text(source, encoding="utf-8")
            try:
                spec = importlib.util.spec_from_file_location("_model_maven_generated_assembly", generated)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                model = module.build_model()
                if not isinstance(model, torch.nn.Module):
                    raise TypeError("build_model() did not return torch.nn.Module")
            except Exception as error:
                raise UserError("ASSEMBLY_GENERATED_CONSTRUCTION_FAILED", f"生成代码重导入或构造失败：{type(error).__name__}: {error}", "检查组件工厂与生成适配器；没有发布输出文件。") from error
            alias_evidence = _module_alias_evidence(model)
            shared_state_evidence = _shared_state_evidence(model)
            placement_evidence = _registered_state_placement(model)
            if request.get("batch"):
                request["_disable_evidence_cache"] = True
                forward_evidence = probe(model, request).get("batch_content")
                forward = "passed"
            else:
                forward_evidence = None
                forward = "not_run"
            return {"construction":"passed", "forward":forward, "weights":"not_loaded", "scope":"fresh-worker-generated-source", "source_manifest": source_manifest(request["project_root"]), "input_evidence":forward_evidence, "module_aliases": alias_evidence, "shared_state": shared_state_evidence, "model_state_placement": placement_evidence, "weight_equality": "not_compared"}
    if operation == "preview":
        return _run_preview_operation(request, torch, entrypoint, _source_manifest)
    if operation == "edit":
        result = edit_candidate(request)
        if request.get("edit", {}).get("target") == "source":
            if request.get("edit", {}).get("action") != "migrate_starter":
                result["source_manifest"] = source_manifest(request["project_root"])
        return result
    if operation == "evaluate" and request.get("expected_source_manifest") is not None:
        verify_manifest(request["expected_source_manifest"])
    request["_working_directory_before_model"] = _working_directory()
    external_training = (
        operation in {"training_probe", "train"}
        and request.get("training_mode") == "external-loop"
    )
    if external_training and not request.get("model") and request.get("variant") is None:
        model = None
        origin = {"kind": "user_external_loop", "model": None}
        seed = request.get("seed")
        effects = []
        code_root = _ACTIVE_CODE_ROOT or str(Path(request["project_root"]).resolve())
        project_root = str(Path(request["project_root"]).resolve())
        if code_root in sys.path:
            sys.path.remove(code_root)
        sys.path.insert(0, code_root)
        configure_project_import_roots(request)
    else:
        model, origin, seed, effects = load_model(request)
    if operation in {"probe", "semantic_mark", "semantic_supervise"} and request.get("probe_device") is not None:
        requested_device = request["probe_device"]
        if not isinstance(requested_device, str) or not requested_device or len(requested_device) > 128:
            raise UserError("PROBE_DEVICE_INVALID", "前向设备标识无效。", "指定所选 PyTorch 支持的设备；输入不会自动迁移。")
        selection = _resolve_device(torch, requested_device)
        try:
            placed = model.to(selection.device)
        except Exception as error:
            raise UserError("PROBE_DEVICE_PLACEMENT_FAILED", f"无法将模型迁移到所选设备：{type(error).__name__}: {error}",
                            "检查用户模型的设备迁移支持；输入工厂也需要提供兼容设备的输入。") from error
        if placed is not model:
            raise UserError("PROBE_DEVICE_PLACEMENT_UNCONFIRMED", "模型的设备迁移没有返回原模型实例。",
                            "检查自定义 to() 实现；迁移可能已产生副作用，未执行前向。")
        request["_probe_device_report"] = selection.report()
    request["_working_directory_after_model"] = _working_directory()
    request["_model_origin"] = origin
    if operation in {"training_probe", "train"}:
        result = training_probe_with_source_evidence(model, request)
        result.setdefault("runtime", {}).update({
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "working_directory_before_model": request["_working_directory_before_model"],
            "working_directory_after_model": request["_working_directory_after_model"],
            "working_directory_after_training": _working_directory(),
            "project_import_roots": request.get("runtime", {}).get("project_import_roots"),
        })
        if request.get("_environment_fingerprint_sha256"):
            result["runtime"]["environment_fingerprint_sha256"] = request[
                "_environment_fingerprint_sha256"
            ]
        result.update({"origin": origin, "seed": seed, "variant_applied": request.get("variant") is not None,
                       "edit_effects": effects})
        result["seed_source"] = request.get("seed_source")
        result["seed_decimal"] = str(seed) if type(seed) is int else None
        if request.get("training_mode") != "external-loop":
            result["batch_device_policy"] = request.get("batch_device_policy") or "legacy-recursive"
        return result
    if operation == "evaluate":
        result = run_evaluation(model, request)
        result.update({"origin": origin, "variant_applied": request.get("variant") is not None})
        return result
    if operation == "verify":
        result = verify_gradients(model, request)
        result.update({"origin": origin, "seed": seed, "variant_applied": request.get("variant") is not None,
                       "edit_effects": effects})
        return result
    if operation == "semantic_mark":
        action = request.get("edit")
        if isinstance(action, dict) and action.get("action") == "audit":
            request["_disable_evidence_cache"] = True
            observation = probe(model, request)
            observed = {**observation, "origin": origin}
            return _audit_semantic_marks(
                request["project_root"],
                observed,
                context=_semantic_binding_context(
                    observed,
                    request.get("variant"),
                    source_manifest(request["project_root"]),
                    request.get("runtime"),
                ),
            )
        if not isinstance(action, dict) or action.get("action") not in {"add", "update"}:
            raise UserError("SEMANTIC_ACTION_INVALID", "当前语义标记操作需要 action=add 或 update。")
        request["_disable_evidence_cache"] = True
        observation = probe(model, request)
        observation["model_device"] = request.get("_probe_device_report")
        observation["reported_seed"] = seed
        observation["seed_policy"] = "runtime" if seed is None else "fixed"
        result = _add_semantic_mark(
            request["project_root"],
            {**observation, "origin": origin},
            module_path=request.get("path"),
            port=action.get("port"),
            role=action.get("role"),
            meaning=action.get("meaning"),
            mark_id=action.get("id"),
            replace_existing=action.get("action") == "update",
            expected_state_revision=action.get("if_state_revision"),
            expected_anchor=action.get("if_anchor"),
            variant_name=request.get("semantic_variant_name"),
            context=_semantic_binding_context(
                {**observation, "origin": origin},
                request.get("variant"),
                source_manifest(request["project_root"]),
                request.get("runtime"),
            ),
        )
        return {**result, "observation": {
            "probe_id": observation["probe_id"],
            "coverage": observation["coverage"],
            "batch_identity_stable": observation["batch_identity_stable"],
        }}
    if operation == "semantic_supervise":
        if request.get("batch"):
            # Match mark creation: include the batch factory module in the
            # source identity before comparing the saved semantic binding.
            entrypoint(request["batch"])
        current_source_manifest = source_manifest(request["project_root"])
        context = _semantic_binding_context(
            {"origin": origin}, request.get("variant"), current_source_manifest, request.get("runtime")
        )
        result = _preview_supervision(
            model,
            request,
            context=context,
            torch_module=torch,
            make_batch_fn=make_batch,
            resolve_entrypoint=entrypoint,
            module_entries_fn=module_entries,
        )
        result["model_device"] = request.get("_probe_device_report")
        result["seed"] = seed
        result["seed_policy"] = "runtime" if seed is None else "fixed"
        return result
    if operation not in {"inspect", "probe", "layers"}:
        raise UserError("FEATURE_UNAVAILABLE", f"当前未实现操作：{operation}")
    result = inspect_model(
        model,
        request.get("path"),
        request.get("layer_type"),
        request.get("search"),
    )
    if request.get("collect_source_manifest"):
        result["module_aliases"] = _module_alias_evidence(model)
        result["shared_state"] = _shared_state_evidence(model)
    if operation == "inspect" and request.get("collect_source_manifest"):
        result["source_manifest"] = source_manifest(request["project_root"])
    if operation == "probe":
        if request.get("collect_source_manifest"):
            request["_disable_evidence_cache"] = True
        result.update(probe(model, request))
        if result.get("forward_execution") == "executed":
            result["model_state_after_forward"] = _model_state_summary(model)
        # Runtime facts describe this worker only; they do not establish that
        # dependencies, weights, or later executions are identical.
        result["runtime"] = {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "working_directory_before_model": request["_working_directory_before_model"],
            "working_directory_after_model": request["_working_directory_after_model"],
            "working_directory_after_probe": _working_directory(),
            "project_import_roots": request.get("runtime", {}).get("project_import_roots"),
            "probe_device": request.get("_probe_device_report"),
        }
        if request.get("collect_source_manifest"):
            if request.get("replay_input_from") and request.get("batch"):
                # Replayed values do not call the batch factory, but its module
                # remains part of the observed source identity of the preview.
                entrypoint(request["batch"])
            result["source_manifest"] = source_manifest(request["project_root"])
        calls = result.get("calls", [])
        for evidence in origin.get("injection_evidence", []):
            paths = set(evidence.get("registered_paths", []))
            evidence["sample_forward"] = (
                "observed"
                if any(paths.intersection(call.get("paths", [])) for call in calls)
                else "not_observed"
            )
    expected_shared = request.get("expected_shared_state_groups")
    if expected_shared is not None:
        actual_shared = shared_state_groups(model)
        if not isinstance(expected_shared, list) or actual_shared != expected_shared:
            raise UserError("SOURCE_SHARED_STATE_MISMATCH", "源码重新加载后的共享参数或 Buffer 与候选模型不一致。", "检查构造及共享赋值的位置；原源码编辑将回滚。", {"expected": expected_shared, "actual": actual_shared})
        result["shared_state_verification"] = {"status": "passed", "scope": "registered_object_and_storage_after_validation", "group_count": len(actual_shared), "unavailable_storage_count": sum(group["kind"] == "storage_unavailable" for group in actual_shared)}
    result.update({"origin": origin, "seed": seed, "seed_policy": "runtime" if seed is None else "fixed",
                   "variant_applied": request.get("variant") is not None,
                   "edit_effects": effects})
    if operation == "probe":
        variant = request.get("variant")
        requested_seed = request.get("seed")
        construction_seed = variant.get("seed") if isinstance(variant, dict) else requested_seed
        result["randomness"] = {
            "construction_seed_decimal": str(construction_seed) if type(construction_seed) is int else None,
            "execution_seed_decimal": str(requested_seed) if isinstance(variant, dict) and type(requested_seed) is int else None,
            "forward_rng_policy": "reset_after_variant" if isinstance(variant, dict) and type(requested_seed) is int
                else "construction_stream" if type(construction_seed) is int else "runtime",
            "scope": "worker seed actions before batch construction; user code and other random libraries may change state",
        }
    return result


def main():
    # OS-level redirection also catches native libraries printing to fd 1.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    operation = "unknown"
    try:
        try:
            request = json.load(sys.stdin)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise UserError("PROTOCOL_ERROR", "worker 请求不是有效的 UTF-8 JSON。", "检查请求格式后重试；目标操作尚未启动。") from error
        if isinstance(request, dict) and isinstance(request.get("operation"), str):
            operation = request["operation"]
        result = {"protocol_version": PROTOCOL_VERSION, "operation": operation,
                  "status": "success", "data": dispatch(request), "diagnostics": []}
    except InconclusiveError as error:
        result = {"protocol_version": PROTOCOL_VERSION, "operation": operation,
                  "status": "inconclusive", "data": error.data,
                  "diagnostics": [{"code": error.code, "message": str(error), "hint": error.hint}]}
    except UserError as error:
        result = {"protocol_version": PROTOCOL_VERSION, "operation": operation,
                  "status": "error", "data": error.data,
                  "diagnostics": [{"code": error.code, "message": str(error), "hint": error.hint}]}
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        result = {"protocol_version": PROTOCOL_VERSION, "operation": operation,
                  "status": "error", "data": {},
                  "diagnostics": [{"code": "INTERNAL_ERROR", "message": f"{type(error).__name__}: {error}", "hint": "使用 --verbose 查看详情。"}]}
    json.dump(result, protocol, ensure_ascii=False, allow_nan=False)
    protocol.write("\n")
    protocol.flush()


if __name__ == "__main__":
    main()
