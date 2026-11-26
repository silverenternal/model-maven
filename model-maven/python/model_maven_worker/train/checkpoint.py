"""Atomic, source-bound checkpoints and model state fingerprints."""

import hashlib
import json
import os
import stat
import sys
import tempfile
import uuid
from contextlib import ExitStack
from pathlib import Path

from ..environment import digest
from ..protocol import UserError
from .state import authorize_checkpoint_state_migration


def checkpoint_value_supported(value, torch_module):
    if value is None or isinstance(value, (bool, int, float, str, torch_module.Tensor)):
        return True
    if isinstance(value, (list, tuple)):
        return all(checkpoint_value_supported(item, torch_module) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, (str, int)) and checkpoint_value_supported(item, torch_module)
            for key, item in value.items()
        )
    return False


def training_run_signature(request):
    variant = request.get("variant")
    signature = {
        "model": request.get("model"),
        "model_params": request.get("model_params", {}),
        "batch": request.get("batch"),
        "batch_params": request.get("batch_params", {}),
        "batch_binding": request.get("batch_binding"),
        "training_step": request.get("training_step"),
        "training_mode": request.get("training_mode"),
        "training_params": request.get("training_params", {}),
        "train_augment": request.get("train_augment", []),
        "apply_supervision": request.get("apply_supervision") is True,
        "supervision_snapshot": request.get("supervision_snapshot", []),
        "lr": request.get("lr"),
        "weight_decay": request.get("weight_decay"),
        "seed": request.get("seed"),
        "device": (request.get("runtime") or {}).get("device") or "cpu",
        "variant_digest": digest(variant) if variant is not None else None,
    }
    if request.get("training_loop") is not None:
        signature["training_loop"] = request["training_loop"]
    if request.get("batch_device_policy") is not None:
        signature["batch_device_policy"] = request["batch_device_policy"]
    if request.get("model_initialization_policy") is not None:
        signature["model_initialization_policy"] = request["model_initialization_policy"]
        signature["model_initialization_seed"] = request.get("model_initialization_seed")
    if request.get("model_injections"):
        signature["model_injections"] = request["model_injections"]
    if request.get("optimizer_factory") is not None:
        signature["optimizer_factory"] = request["optimizer_factory"]
        signature["optimizer_params"] = request.get("optimizer_params", {})
    return signature


def model_state_digest(torch_module, model):
    return state_dict_digest(torch_module, model.state_dict())


def model_state_alias_groups(model):
    """Return persisted state-dict names that refer to the same parameter/buffer."""
    state_keys = set(model.state_dict())
    aliases = {}
    for iterator_name in ("named_parameters", "named_buffers"):
        iterator = getattr(model, iterator_name)
        try:
            entries = iterator(recurse=True, remove_duplicate=False)
        except TypeError:  # Older PyTorch versions cannot expose duplicate names.
            return None
        for name, value in entries:
            if name in state_keys:
                aliases.setdefault(id(value), []).append(name)
    return sorted(
        (sorted(set(names)) for names in aliases.values() if len(set(names)) > 1),
        key=lambda group: group[0],
    )


def state_dict_digest(torch_module, state_dict):
    result = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        result.update(name.encode("utf-8"))
        if isinstance(value, torch_module.Tensor):
            tensor = value.detach().cpu().contiguous()
            result.update(str(tensor.dtype).encode())
            result.update(json.dumps(list(tensor.shape)).encode())
            result.update(tensor.reshape(-1).view(torch_module.uint8).numpy().tobytes())
        else:
            result.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return result.hexdigest()


def _sha256_file(path):
    digest_value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def atomic_json_file(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".checkpoint-", delete=False
        ) as stream:
            temporary = stream.name
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def save_training_checkpoint(torch_module, directory, payload):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise UserError("CHECKPOINT_PATH_INVALID", "checkpoint 输出目录不是可信的运行目录。")
    if not checkpoint_value_supported(payload, torch_module):
        raise UserError(
            "CHECKPOINT_STATE_UNSUPPORTED",
            "checkpoint 状态包含不能安全恢复的 Python 对象。",
            "自定义状态只保存 Tensor、字符串、数字、列表、元组和字典。",
        )
    step = payload["completed_steps"]
    filename = f"checkpoint-{step:09d}-{uuid.uuid4().hex}.pt"
    target = directory / filename
    temporary = None
    previous = None
    latest = directory / "latest.json"
    if os.path.lexists(latest):
        if latest.is_symlink() or not latest.is_file():
            raise UserError("CHECKPOINT_LATEST_INVALID", "latest.json 必须是运行目录内的普通文件。")
        try:
            previous_manifest = json.loads(latest.read_text(encoding="utf-8"))
        except Exception as error:
            raise UserError("CHECKPOINT_LATEST_INVALID", f"latest.json 无法读取：{error}") from error
        if not isinstance(previous_manifest, dict):
            raise UserError("CHECKPOINT_LATEST_INVALID", "latest.json 根值必须是对象。")
        previous = previous_manifest.get("file")
        previous_path = directory / previous if isinstance(previous, str) else None
        previous_hash = None
        if previous_path is not None and previous_path.is_file() and not previous_path.is_symlink():
            try:
                previous_hash = _sha256_file(previous_path)
            except OSError as error:
                raise UserError(
                    "CHECKPOINT_LATEST_INVALID",
                    f"无法读取现有 latest.json 指向的 checkpoint：{error}",
                    "未修改当前恢复点；检查运行目录权限和磁盘状态。",
                ) from error
        if (
            previous_manifest.get("schema_version") != 1
            or not isinstance(previous, str)
            or Path(previous).name != previous
            or not previous.startswith("checkpoint-")
            or previous_path is None
            or previous_path.is_symlink()
            or not previous_path.is_file()
            or previous_hash != previous_manifest.get("sha256")
        ):
            raise UserError("CHECKPOINT_LATEST_INVALID", "现有 latest.json 未能验证其指向的成功 checkpoint；未修改恢复点。")
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".checkpoint-", delete=False) as stream:
            temporary = stream.name
            torch_module.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        checksum = _sha256_file(target)
        manifest = {
            "schema_version": 1,
            "file": filename,
            "sha256": checksum,
            "completed_steps": step,
            "training_mode": payload["training_mode"],
            "model_state_sha256": payload["model_state_sha256"],
        }
        atomic_json_file(latest, manifest)
        if previous and previous != filename:
            previous_path = directory / previous
            if previous_path.parent == directory and previous_path.name.startswith("checkpoint-"):
                try:
                    previous_path.unlink()
                except OSError:
                    pass
        return manifest
    except UserError:
        raise
    except Exception as error:
        raise UserError(
            "CHECKPOINT_WRITE_FAILED",
            f"无法原子保存 checkpoint：{type(error).__name__}: {error}",
            "最近一个已提交 checkpoint 保持可用；检查磁盘空间与运行目录权限。",
        ) from error
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_verified_checkpoint_payload(torch_module, path, expected_sha256=None):
    """Load a checkpoint whose adjacent ``latest.json`` verifies its bytes.

    This validates artifact integrity only. Callers that resume the same run
    must still validate the run signature, source manifest, runtime and model.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise UserError("CHECKPOINT_NOT_FOUND", "恢复运行指定的 checkpoint 文件不存在。")
    directory = path.parent
    manifest_path = directory / "latest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise UserError("CHECKPOINT_INVALID", "checkpoint 缺少安全清单。")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise UserError(
            "CHECKPOINT_CORRUPT",
            f"checkpoint 清单无法读取：{error}",
            "保留原运行和其他恢复点；不要使用不可信来源的 checkpoint。",
        ) from error
    if not isinstance(manifest, dict):
        raise UserError("CHECKPOINT_CORRUPT", "latest.json 根值必须是对象。")
    if manifest.get("schema_version") != 1:
        raise UserError(
            "CHECKPOINT_VERSION_UNSUPPORTED",
            "当前构建不支持此 checkpoint 清单版本。",
            "保留原 checkpoint；使用兼容版本恢复或建立新运行，不会猜测清单字段。",
        )
    with ExitStack() as resources:
        try:
            if manifest.get("file") != path.name:
                raise ValueError("latest checkpoint 指针与请求路径不一致")
            # Always load the exact bytes checked against the manifest, even
            # when the caller did not supply an additional expected digest.
            # A live training process may replace the checkpoint between calls.
            snapshot = resources.enter_context(tempfile.TemporaryFile(mode="w+b"))
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            with os.fdopen(fd, "rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("checkpoint 不是普通文件")
                digest_value = hashlib.sha256()
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest_value.update(chunk)
                    snapshot.write(chunk)
            snapshot.flush()
            snapshot.seek(0)
            actual_hash = digest_value.hexdigest()
            load_source = snapshot
            if actual_hash != manifest.get("sha256"):
                raise ValueError("SHA-256 校验失败")
        except OSError as error:
            raise UserError(
                "CHECKPOINT_SNAPSHOT_UNAVAILABLE",
                f"无法读取 checkpoint 或创建验证副本：{error}",
                "检查 checkpoint 的读取权限和临时目录空间；未载入训练状态。",
            ) from error
        except Exception as error:
            raise UserError(
                "CHECKPOINT_CORRUPT",
                f"checkpoint 校验或安全载入失败：{error}",
                "保留原运行和其他恢复点；检查临时目录空间，不要使用不可信来源的 checkpoint。",
            ) from error
        if expected_sha256 is not None and actual_hash != expected_sha256:
            raise UserError(
                "CHECKPOINT_CHANGED",
                "恢复点与确认时的 checkpoint 摘要不一致。",
                "重新查看运行的最新 checkpoint 并确认后重试；尚未载入训练状态。",
                {"expected_sha256": expected_sha256, "actual_sha256": actual_hash},
            )
        try:
            payload = torch_module.load(load_source, map_location="cpu", weights_only=True)
        except Exception as error:
            raise UserError("CHECKPOINT_CORRUPT", f"checkpoint 安全载入失败：{error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise UserError(
            "CHECKPOINT_VERSION_UNSUPPORTED",
            "当前构建不支持此 checkpoint 格式。",
            "保留原 checkpoint；使用兼容版本恢复后导出，或从头创建新运行。不会猜测或迁移未知状态。",
        )
    if (
        type(manifest.get("completed_steps")) is not int
        or manifest["completed_steps"] != payload.get("completed_steps")
        or manifest.get("training_mode") != payload.get("training_mode")
        or manifest.get("model_state_sha256") != payload.get("model_state_sha256")
    ):
        raise UserError("CHECKPOINT_INVALID", "latest.json 的模式、完成步数或模型指纹与 checkpoint 内容不匹配。")
    return payload, manifest


def load_training_checkpoint(torch_module, path, model, request, source_manifest):
    payload, manifest = load_verified_checkpoint_payload(torch_module, path, request.get("expected_checkpoint_sha256"))
    if payload.get("training_mode") != request.get("training_mode"):
        raise UserError("RESUME_MODE_INCOMPATIBLE", "训练模式与 checkpoint 不一致。")
    saved_snapshot = payload.get("source_snapshot_sha256")
    if saved_snapshot is not None:
        snapshot = request.get("code_snapshot")
        current_snapshot = snapshot.get("sha256") if isinstance(snapshot, dict) else None
        if not isinstance(saved_snapshot, str) or saved_snapshot != current_snapshot:
            raise UserError("RESUME_SOURCE_SNAPSHOT_CHANGED", "checkpoint 对应的完整项目源码快照与当前快照不一致。",
                            "恢复原项目源码，或使用当前源码建立新的训练运行；旧检查点没有修改。")
    saved_signature = payload.get("run_signature")
    expected_signature = training_run_signature(request)
    if isinstance(saved_signature, dict) and "batch_binding" not in saved_signature:
        request["_legacy_batch_binding_unverified"] = True
        expected_signature.pop("batch_binding", None)
    if saved_signature != expected_signature:
        raise UserError(
            "RESUME_INPUTS_INCOMPATIBLE",
            "模型、批次、步骤或优化器设置与 checkpoint 不一致。",
            "恢复训练固定使用原运行输入；需要改变设置时请另建训练运行。",
        )
    source_changed = payload.get("source_manifest") != source_manifest
    runtime = payload.get("runtime", {})
    if runtime.get("python_version") != sys.version.split()[0] or runtime.get("pytorch_version") != torch_module.__version__:
        raise UserError("RESUME_RUNTIME_CHANGED", "Python/PyTorch 版本与 checkpoint 不一致。")
    # The requested label can be `auto`, and even `cuda:0` is only an index
    # in the current process's visible-device set. Compare what was actually
    # resolved, before restoring state or starting another training step.
    current_device = request.get("_device_report")
    current_device = current_device if isinstance(current_device, dict) else {}
    compared = []
    for field in ("device", "device_backend", "device_name"):
        saved = runtime.get(field)
        current = current_device.get(field)
        if isinstance(saved, str) and saved and isinstance(current, str) and current:
            if saved != current:
                raise UserError(
                    "RESUME_DEVICE_CHANGED",
                    f"实际设备的 {field} 与 checkpoint 不一致：记录为 {saved!r}，当前为 {current!r}。",
                    "检查设备可见性与运行环境；如需使用其他设备，请从新运行开始。",
                    {"field": field, "recorded": saved, "current": current},
                )
            compared.append(field)
    request["_resume_device_comparison"] = {
        "status": "matched_available_fields" if compared else "unknown",
        "compared_fields": compared,
        "note": "设备名称和可见索引不能证明物理设备、驱动或完整环境相同。",
    }
    expected_environment = runtime.get("environment_fingerprint_sha256")
    current_environment = request.get("_environment_fingerprint_sha256")
    environment_changed = bool(expected_environment and expected_environment != current_environment)
    if source_changed or environment_changed:
        engine = payload.get("engine_state")
        manual = engine.get("manual_context") if isinstance(engine, dict) else None
        groups = manual.get("optimizers") if isinstance(manual, dict) else None
        custom_manual = isinstance(groups, dict) and any(
            isinstance(group, dict) and isinstance(group.get("metadata"), dict)
            and group["metadata"].get("optimizer") == "factory" for group in groups.values()
        )
        if request.get("optimizer_factory") is not None or custom_manual:
            raise UserError(
                "RESUME_OPTIMIZER_SOURCE_CHANGED",
                "自定义优化器的训练源码或环境发生变化，不能通过其他 callable 的状态迁移恢复。",
                "恢复原源码和环境，或创建新的训练运行。",
            )
        migration = authorize_checkpoint_state_migration(payload, request, source_manifest)
        if migration is None:
            if source_changed:
                raise UserError(
                    "RESUME_SOURCE_CHANGED",
                    "checkpoint 对应的本地源码与当前源码不同，恢复已停止。",
                    "恢复原源码版本；只有具备精确版本化状态迁移声明的单一扩展 callable 才能跨源版本恢复。",
                )
            raise UserError(
                "RESUME_ENVIRONMENT_CHANGED",
                "当前 Python distribution 清单与 checkpoint 创建时不一致。",
                "恢复原运行环境；仅状态 callable 对应的单一扩展 distribution 可通过精确 StateMigrator 迁移。",
            )
        request["_checkpoint_state_migration_authorized"] = migration
    saved_model_state = payload.get("model_state")
    if model is None:
        if saved_model_state is not None:
            raise UserError(
                "RESUME_MODEL_REQUIRED",
                "该 checkpoint 包含模型权重，但当前 external-loop 没有提供模型入口。",
                "恢复此运行时提供原模型工厂，或创建不依赖模型权重的独立运行。",
            )
        if payload.get("model_state_sha256") is not None:
            raise UserError("CHECKPOINT_INVALID", "无模型 checkpoint 不应声明模型权重指纹。")
    else:
        if not isinstance(saved_model_state, dict):
            raise UserError("CHECKPOINT_INVALID", "checkpoint 缺少有效模型状态。")
        expected_model_digest = payload.get("model_state_sha256")
        if (
            not isinstance(expected_model_digest, str)
            or state_dict_digest(torch_module, saved_model_state) != expected_model_digest
        ):
            raise UserError("CHECKPOINT_CORRUPT", "checkpoint 模型状态与其 SHA-256 指纹不匹配。")
        try:
            model.load_state_dict(saved_model_state, strict=True)
        except Exception as error:
            raise UserError(
                "RESUME_MODEL_INCOMPATIBLE",
                f"模型权重与当前模型结构不兼容：{error}",
            ) from error
        if model_state_digest(torch_module, model) != expected_model_digest:
            raise UserError("CHECKPOINT_CORRUPT", "恢复后的模型权重与 checkpoint 指纹不一致。")
    request["_resume_checkpoint"] = {
        "file": manifest["file"],
        "sha256": manifest["sha256"],
        "completed_steps": manifest["completed_steps"],
        "confirmed_sha256": request.get("expected_checkpoint_sha256"),
        "loaded_from_verified_copy": True,
    }
    return payload


