"""Task-owned evaluation without assuming a classification output contract."""

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..protocol import UserError
from ..data.augment import apply_batch_augmentations


def _metric_value(runtime, value, field):
    try:
        return runtime.scalar_value(value, field)
    except UserError as error:
        raise UserError(
            "EVALUATION_METRIC_INVALID",
            f"评估指标 {field} 必须是有限数值标量：{error}",
            "检查指标计算和样本覆盖；NaN、无穷大或非数值结果不会写入评估产物。",
            {"cause_code": error.code},
        ) from error


@dataclass
class EvaluationRuntime:
    torch: Any
    resolve_entrypoint: Callable[[str], Callable[..., Any]]
    verify_manifest: Callable[[dict], None]
    scalar_value: Callable[[Any, str], float]
    source_manifest: Callable[[str], dict] | None = None


class EvaluationContext:
    """Small evaluator context; counts remain unknown unless the evaluator reports them."""

    def __init__(self, split, max_samples, runtime, max_preview_samples=0, augmentations=None, seed=0):
        self.split = split
        self.max_samples = max_samples
        self.max_preview_samples = max_preview_samples
        self._runtime = runtime
        self._metrics = {}
        self._samples_seen = None
        self._sample_previews = []
        self._preview_ids = set()
        self._augmentations = augmentations or []
        self._augmentation_seed = int(seed or 0)
        self._augmentation_calls = 0
        self._last_augmentation_observations = []

    def transform_batch(self, batch):
        """Apply the project's explicit eval augmentation targets to one evaluator batch."""
        transformed, observations = apply_batch_augmentations(
            batch,
            self._augmentations,
            resolve_entrypoint=self._runtime.resolve_entrypoint,
            seed=self._augmentation_seed + self._augmentation_calls,
        )
        if self._augmentations:
            self._augmentation_calls += 1
            self._last_augmentation_observations = observations
        return transformed

    def log_metric(self, name, value):
        self._validate_name(name, "metric")
        if len(self._metrics) >= 128 and name not in self._metrics:
            raise UserError("EVALUATION_RESULT_INVALID", "每次评估最多记录 128 个具名指标。")
        self._metrics[name] = _metric_value(self._runtime, value, f"metrics.{name}")

    def report_samples(self, count):
        if type(count) is not int or count < 0:
            raise UserError("EVALUATION_RESULT_INVALID", "评估样本数必须是非负整数。")
        if self.max_samples is not None and count > self.max_samples:
            raise UserError(
                "EVALUATION_SAMPLE_LIMIT_EXCEEDED",
                f"评估器报告了 {count} 个样本，超过请求上限 {self.max_samples}。",
                "让评估入口遵循 context.max_samples 并报告实际处理数量。",
            )
        self._samples_seen = count

    def report_prediction(self, sample_id, prediction, target=None):
        """Report one explicit JSON-safe evaluator-owned prediction for browsing."""
        if self.max_preview_samples == 0:
            raise UserError(
                "EVALUATION_PREVIEW_NOT_REQUESTED",
                "当前评估没有请求预测样本预览。",
                "使用 --preview-samples 限定需要返回的样本数。",
            )
        if len(self._sample_previews) >= self.max_preview_samples:
            raise UserError(
                "EVALUATION_PREVIEW_LIMIT_EXCEEDED",
                f"评估器报告的预测样本超过上限 {self.max_preview_samples}。",
                "只报告前 N 个样本，并继续完成指标评估。",
            )
        if (
            not isinstance(sample_id, str)
            or not sample_id.strip()
            or len(sample_id) > 256
            or any(ord(char) < 32 or ord(char) == 127 for char in sample_id)
        ):
            raise UserError(
                "EVALUATION_PREVIEW_INVALID",
                "预测样本 ID 必须是 1..256 个不含控制字符的可见字符。",
            )
        if sample_id in self._preview_ids:
            raise UserError("EVALUATION_PREVIEW_DUPLICATE", f"预测样本 ID 重复：{sample_id}")
        item = {"sample_id": sample_id}
        item["prediction"] = _preview_json_value(prediction)
        if target is not None:
            item["target"] = _preview_json_value(target)
        try:
            encoded = json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError) as error:
            raise UserError(
                "EVALUATION_PREVIEW_INVALID",
                f"预测样本必须是有限的 JSON 数据：{error}",
                "将 Tensor 显式转换为有界的 list 或标量；不要返回任意 Python 对象。",
            ) from error
        if len(encoded.encode("utf-8")) > 64 * 1024:
            raise UserError("EVALUATION_PREVIEW_TOO_LARGE", "单个预测样本超过 64 KiB。")
        self._sample_previews.append(json.loads(encoded))
        self._preview_ids.add(sample_id)

    @staticmethod
    def _validate_name(name, field):
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise UserError("EVALUATION_RESULT_INVALID", f"{field} 名称必须是 1..128 个非空字符。")


def _load_checkpoint(runtime, path, expected_sha256=None):
    checkpoint = Path(path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise UserError("CHECKPOINT_NOT_FOUND", "所选训练运行的 checkpoint 文件不存在或不是普通文件。")
    manifest_path = checkpoint.parent / "latest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise UserError("CHECKPOINT_INVALID", "checkpoint 缺少可验证的最新版本清单。")
    from .checkpoint import load_verified_checkpoint_payload
    payload, manifest = load_verified_checkpoint_payload(runtime.torch, checkpoint, expected_sha256)
    actual_digest = manifest["sha256"]
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise UserError("CHECKPOINT_VERSION_UNSUPPORTED", "当前评估器不支持此 checkpoint 格式。")
    if (
        manifest.get("completed_steps") != payload.get("completed_steps")
        or manifest.get("training_mode") != payload.get("training_mode")
        or manifest.get("model_state_sha256") != payload.get("model_state_sha256")
    ):
        raise UserError("CHECKPOINT_INVALID", "checkpoint 清单与其保存的训练状态不匹配。")
    state = payload.get("model_state")
    expected_state = payload.get("model_state_sha256")
    if not isinstance(state, dict) or not isinstance(expected_state, str):
        raise UserError("EVALUATION_MODEL_STATE_MISSING", "该训练 checkpoint 没有可评估的模型权重。")
    from .checkpoint import state_dict_digest

    if state_dict_digest(runtime.torch, state) != expected_state:
        raise UserError("CHECKPOINT_CORRUPT", "checkpoint 模型权重与其指纹不匹配。")
    return payload, actual_digest


def run_evaluation(runtime, model, request, current_source_manifest=None):
    evaluator_ref = request.get("evaluation_entrypoint")
    data_ref = request.get("evaluation_data")
    split = request.get("evaluation_split")
    max_samples = request.get("evaluation_samples")
    if not isinstance(evaluator_ref, str) or not evaluator_ref.strip():
        raise UserError("EVALUATION_ENTRYPOINT_REQUIRED", "请提供项目评估入口 module:evaluate。")
    if not isinstance(data_ref, str) or not data_ref.strip():
        raise UserError("EVALUATION_DATA_REQUIRED", "请提供验证或测试数据入口 module:build_data。")
    if not isinstance(split, str) or not split.strip() or len(split) > 64:
        raise UserError("EVALUATION_SPLIT_INVALID", "split 必须是 1..64 字符的非空名称。")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise UserError("EVALUATION_SAMPLE_LIMIT_INVALID", "samples 必须是正整数。")
    max_preview_samples = request.get("evaluation_preview_samples", 0)
    if type(max_preview_samples) is not int or not 0 <= max_preview_samples <= 32:
        raise UserError("EVALUATION_PREVIEW_LIMIT_INVALID", "preview_samples 必须在 0 到 32 之间。")
    if max_samples is not None and max_preview_samples > max_samples:
        raise UserError(
            "EVALUATION_PREVIEW_LIMIT_INVALID",
            "preview_samples 不能大于 samples 评估样本上限。",
        )

    expected_manifest = request.get("expected_source_manifest")
    if expected_manifest is not None:
        runtime.verify_manifest(expected_manifest)
    payload, checkpoint_sha256 = _load_checkpoint(runtime, request.get("checkpoint_input"), request.get("expected_checkpoint_sha256"))
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except Exception as error:
        raise UserError("EVALUATION_MODEL_INCOMPATIBLE", f"checkpoint 权重与当前模型不兼容：{error}") from error
    if request.get("model_state_sha256") is not None and request["model_state_sha256"] != payload["model_state_sha256"]:
        raise UserError("EVALUATION_MODEL_INCOMPATIBLE", "评估请求的模型指纹与 checkpoint 不一致。")

    try:
        data_factory = runtime.resolve_entrypoint(data_ref)
        data = data_factory(**request.get("evaluation_data_params", {}))
        evaluator = runtime.resolve_entrypoint(evaluator_ref)
    except UserError:
        raise
    except Exception as error:
        raise UserError("EVALUATION_INPUT_CONSTRUCTION_FAILED", f"无法构造评估入口或数据：{error}") from error

    evaluation_augment = request.get("evaluation_augment", [])
    context = EvaluationContext(
        split,
        max_samples,
        runtime,
        max_preview_samples,
        augmentations=evaluation_augment,
        seed=request.get("seed"),
    )
    was_training = model.training
    model.eval()
    try:
        with runtime.torch.inference_mode():
            result = evaluator(
                model,
                data,
                context,
                **request.get("evaluation_params", {}),
            )
    except UserError:
        raise
    except Exception as error:
        raise UserError("EVALUATION_FAILED", f"项目评估入口执行失败：{type(error).__name__}: {error}") from error
    finally:
        model.train(was_training)

    if evaluation_augment and context._augmentation_calls == 0:
        raise UserError(
            "EVALUATION_AUGMENT_NOT_APPLIED",
            "项目声明了 eval 增强，但 evaluator 没有调用 context.transform_batch()。",
            "对 evaluator 当前 batch 调用 context.transform_batch(batch)，再用返回值进行评估。",
        )

    if runtime.source_manifest is not None:
        current_source_manifest = runtime.source_manifest(request["project_root"])
    current_source_manifest = current_source_manifest or {}

    if not isinstance(result, dict):
        raise UserError("EVALUATION_RESULT_INVALID", "评估入口必须返回包含 metrics 的对象。")
    metrics = result.get("metrics", context._metrics)
    if not isinstance(metrics, dict) or not metrics:
        raise UserError("EVALUATION_METRICS_MISSING", "评估入口未返回或记录任何具名指标。")
    if len(metrics) > 128:
        raise UserError("EVALUATION_RESULT_INVALID", "每次评估最多返回 128 个具名指标。")
    normalized = {}
    for name, value in metrics.items():
        context._validate_name(name, "metric")
        normalized[name] = _metric_value(runtime, value, f"metrics.{name}")
    sample_count = result.get("samples", context._samples_seen)
    if sample_count is not None:
        context.report_samples(sample_count)
        sample_count = context._samples_seen
    preview_state = (
        "reported"
        if context._sample_previews
        else "not_reported"
        if max_preview_samples
        else "not_requested"
    )

    data_identity = {
        "entrypoint": data_ref,
        "parameters": request.get("evaluation_data_params", {}),
        "split": split,
        "sample_limit": max_samples,
        "data_fingerprint": request.get("evaluation_data_fingerprint"),
        "seed": request.get("seed"),
        "evaluation_augment": evaluation_augment,
    }
    source_digest = hashlib.sha256(
        json.dumps(current_source_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    data_digest = hashlib.sha256(
        json.dumps(data_identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    output = {
        "source_run_id": request.get("source_run_id"),
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "model_state_sha256": payload["model_state_sha256"],
            "completed_steps": payload.get("completed_steps"),
        },
        "evaluation": {
            "entrypoint": evaluator_ref,
            "split": split,
            "sample_limit": max_samples,
            "samples_evaluated": sample_count,
            "metrics": normalized,
            "custom_artifacts": result.get("artifacts", {}),
            "sample_previews": {
                "limit": max_preview_samples,
                "reported": len(context._sample_previews),
                "state": preview_state,
                "samples": context._sample_previews,
            },
            "augmentation": {
                "status": "applied" if evaluation_augment else "not_configured",
                "configured_steps": len(evaluation_augment),
                "transformed_batches": context._augmentation_calls,
                "last_batch": context._last_augmentation_observations,
            },
        },
        "provenance": {
            "evaluation_source_manifest": current_source_manifest,
            "evaluation_source_sha256": source_digest,
            "data_identity": data_identity,
            "data_identity_sha256": data_digest,
            "protocol_version": request.get("protocol_version"),
            "worker_api_version": request.get("worker_api_version"),
            "python_version": sys.version.split()[0],
            "pytorch_version": str(runtime.torch.__version__),
        },
        "coverage": {
            "checkpoint_weights": "verified",
            "metrics": "observed",
            "samples": "observed" if sample_count is not None else "unknown",
            "data_identity": (
                "declared_fingerprint_and_inputs"
                if data_identity["data_fingerprint"]
                else "entrypoint_parameters_split_and_seed_only"
            ),
            "classification_assumption": False,
            "sample_previews": preview_state,
        },
    }
    try:
        json.dumps(output, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise UserError("EVALUATION_RESULT_INVALID", f"评估产物必须是有限且可序列化的 JSON：{error}") from error
    return output


def _preview_json_value(value, depth=0, budget=None):
    if budget is None:
        budget = [10_000]
    budget[0] -= 1
    if budget[0] < 0 or depth > 16:
        raise UserError("EVALUATION_PREVIEW_TOO_LARGE", "预测样本超过 10000 个节点或 16 层嵌套。")
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UserError("EVALUATION_PREVIEW_INVALID", "预测样本不能包含 NaN 或无穷大。")
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise UserError("EVALUATION_PREVIEW_INVALID", "预测样本对象的键必须是字符串。")
        return {
            key: _preview_json_value(child, depth + 1, budget)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_preview_json_value(child, depth + 1, budget) for child in value]
    raise UserError(
        "EVALUATION_PREVIEW_INVALID",
        f"预测样本不支持 {type(value).__name__}；请显式转换为 JSON 标量、数组或对象。",
    )
