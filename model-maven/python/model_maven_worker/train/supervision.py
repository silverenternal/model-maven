"""Read-only preview of explicit rules bound to manually marked model ports."""

import re
import json
import uuid
import hashlib
from contextlib import ExitStack, contextmanager

from ..model.probe import implementation_anchor
from ..model.structure import module_entries
from ..model.semantics import implementation_identity, list_marks, runtime_paths_digest
from ..protocol import UserError
from .custom_rules import evaluate_plugin, resolve_plugin


_PATH_TOKEN = re.compile(r'(?:\.([^\.\[\]]+)|\[([0-9]+)\]|\[("(?:[^"\\]|\\.)*")\])')
_MAX_INVOCATIONS_PER_RULE = 128
_MAX_TRACKED_SAMPLE_IDS = 10_000
_MAX_TRACKED_VIOLATING_IDS = 10_000
_MISSING_PORT = object()


def _port_observation_scope(port):
    """Describe hook timing, not evidence that a particular call occurred."""
    is_input = isinstance(port, str) and (port == "inputs" or port.startswith("inputs.") or port.startswith("inputs["))
    return {
        "method": "post_forward_hook",
        "input_reference_may_be_mutated": is_input,
        "prevents_module_call": False,
        "rolls_back_user_side_effects": False,
    }


def _sample_localization_summary(localized=False, rules=None):
    if localized:
        return {
            "status": "reported",
            "stable_ids_available": True,
            "rules": rules or [],
            "reason": "已列出规则存在用户声明的样本 ID 与指定轴数量匹配的观察；不验证 ID 真实性、其他调用或数据集覆盖。",
        }
    return {
        "status": "insufficient",
        "stable_ids_available": False,
        "reason": "没有足够证据将完整的已记录调用关联到用户声明的 sample_id；不推断数据集覆盖。",
    }


def _digest(value):
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prepare_training_supervision(model, request, torch_module):
    """Resolve explicit managed-step rules to current, manually marked modules."""
    if request.get("apply_supervision") is not True:
        return None
    mode = request.get("training_mode", "managed-step")
    if mode not in {"managed-step", "manual-step", "external-loop"}:
        raise UserError(
            "SUPERVISION_MODE_UNSUPPORTED",
            "当前监督规则接入不支持该训练模式。",
            "请在 managed-step、manual-step 或显式遵守 external-loop 监督上下文协议的完整循环中使用。",
        )
    origin = request.get("_model_origin", {})
    if model is None or not isinstance(origin.get("model"), str):
        raise UserError("SUPERVISION_MODEL_REQUIRED", "监督训练需要一个可绑定语义标记的模型。")
    modules, aliases = module_entries(model)
    marks_report = list_marks(request["project_root"])
    expected_revision = request.get("expected_supervision_revision")
    if expected_revision is not None and expected_revision != marks_report["state_revision"]:
        raise UserError(
            "SUPERVISION_STATE_CONFLICT",
            "监督标记或规则在确认后发生变化，未执行训练。",
            "重新查看规则并确认训练计划。",
        )
    marks = marks_report["marks"]
    bound = []
    implementation_cache = {}
    for mark in marks:
        rules = mark.get("rules", [])
        if not rules:
            continue
        external_unverified = (mark["binding_status"] == "unverified"
                               and str(mark.get("anchor", {}).get("source_file", "")).startswith("@external:"))
        if mark["binding_status"] != "current" and not external_unverified:
            raise UserError(
                "SUPERVISION_MARK_STALE",
                f"语义标记 {mark['id']} 已过期，不能用于训练。",
                "重新 Probe 并通过 model retag 手动确认模块与端口后重试。",
                {"mark_id": mark["id"], "reason": mark.get("stale_reason")},
            )
        context = mark.get("binding_context", {})
        if context.get("runtime_paths_sha256") != runtime_paths_digest(request.get("runtime")):
            raise UserError("SUPERVISION_MARK_STALE", f"标记 {mark['id']} 的运行目录或模块搜索目录已变化。",
                            "使用当前运行时路径重新前向并手动重新绑定标记。")
        expected_model = origin.get("model")
        if context.get("model") != expected_model:
            raise UserError("SUPERVISION_MARK_STALE", f"标记 {mark['id']} 属于不同模型入口。")
        for field, value in (
            ("model_params_sha256", origin.get("params", {})),
            ("model_injections_sha256", origin.get("model_injections", {})),
        ):
            if context.get(field) != _digest(value):
                raise UserError("SUPERVISION_MARK_STALE", f"标记 {mark['id']} 的模型配置已变化。")
        variant = request.get("variant")
        variant_digest = _digest(variant) if isinstance(variant, dict) else None
        if context.get("variant_sha256") != variant_digest:
            raise UserError("SUPERVISION_MARK_STALE", f"标记 {mark['id']} 的模型变体已变化。")
        candidates = [
            (module, paths)
            for key, paths in aliases.items()
            if mark["module_path"] in paths
            for module in [modules[key]]
        ]
        if len(candidates) != 1:
            raise UserError(
                "SUPERVISION_MARK_UNRESOLVED",
                f"训练模型中无法唯一定位标记 {mark['id']} 的模块 {mark['module_path']}。",
            )
        module = candidates[0][0]
        module_type = type(module)
        if module_type not in implementation_cache:
            implementation_cache[module_type] = implementation_identity(implementation_anchor(
                module, request.get("code_snapshot_root") or request["project_root"]
            ))
        anchor = implementation_cache[module_type]
        if anchor is None or anchor != implementation_identity(mark.get("anchor")):
            raise UserError(
                "SUPERVISION_MARK_STALE",
                f"标记 {mark['id']} 当前对应的实现源码已变化。",
                "重新 Probe 并手动重新绑定后再启用监督训练。",
            )
        plugins = {}
        for rule in rules:
            if rule.get("kind") in {"custom-validator", "custom-penalty"}:
                plugins[rule["id"]] = resolve_plugin(
                    request["project_root"], rule.get("plugin"), rule["kind"], rule["mode"],
                    code_root=request.get("code_snapshot_root"),
                )
        bound.append({"mark": mark, "module": module, "plugins": plugins})
    if not bound:
        raise UserError(
            "SUPERVISION_RULES_MISSING",
            "已启用监督训练，但项目没有可用的语义标记规则。",
            "先用 model mark 手动标记实现块并添加规则，或移除 --apply-supervision。",
        )
    return TrainingSupervision(torch_module, bound, mode=mode)


class TrainingSupervision:
    """Explicit, step-scoped hooks for a managed training step."""

    def __init__(self, torch_module, bound_marks, *, mode="managed-step"):
        self.torch = torch_module
        self.bound_marks = bound_marks
        self.plugin_targets = {
            rule_id: target
            for entry in bound_marks
            for rule_id, target in entry.get("plugins", {}).items()
        }
        self.mode = mode
        self.step = 0
        self.observations = {}
        self.sample_coverage = {}
        self.prior_untracked_steps = 0
        self._active = None
        self._failure = None
        for entry in bound_marks:
            for rule in entry["mark"].get("rules", []):
                self.observations[rule["id"]] = {
                    "rule_id": rule["id"],
                    "mark_id": entry["mark"]["id"],
                    "role": entry["mark"]["role"],
                    "module_path": entry["mark"]["module_path"],
                    "port": entry["mark"]["port"],
                    "observation_scope": _port_observation_scope(entry["mark"]["port"]),
                    "kind": rule["kind"],
                    "mode": rule["mode"],
                    "expected": {key: rule[key] for key in ("min", "max", "weight", "target_path", "axis") if rule.get(key) is not None},
                    "plugin_entrypoint": rule.get("plugin", {}).get("entrypoint") if isinstance(rule.get("plugin"), dict) else None,
                    "steps_observed": 0,
                    "unmeasured_calls": 0,
                    "steps_with_unmeasured_calls": 0,
                    "element_count": 0,
                    "violation_count": 0,
                    "weighted_penalty_sum": 0.0,
                    "last_target_evidence": None,
                    "last_sample_localization": None,
                }
                if rule["kind"] in {"finite", "range"} and rule.get("axis") is not None:
                    self.sample_coverage[rule["id"]] = _new_sample_coverage()

    def restore_coverage(self, saved, *, prior_steps=0):
        if not self.sample_coverage:
            return
        if saved is None:
            self.prior_untracked_steps = max(0, int(prior_steps))
            return
        if (
            not isinstance(saved, dict)
            or saved.get("schema_version") != 1
            or not isinstance(saved.get("rules"), dict)
            or set(saved["rules"]) != set(self.sample_coverage)
            or type(saved.get("prior_untracked_steps")) is not int
            or saved["prior_untracked_steps"] < 0
        ):
            raise UserError(
                "CHECKPOINT_SUPERVISION_COVERAGE_INVALID",
                "checkpoint 中的监督样本覆盖状态与当前规则不匹配。",
            )
        self.prior_untracked_steps = saved["prior_untracked_steps"]
        for rule_id, state in saved["rules"].items():
            target = self.sample_coverage[rule_id]
            if not isinstance(state, dict):
                raise UserError("CHECKPOINT_SUPERVISION_COVERAGE_INVALID", f"规则 {rule_id} 的覆盖状态无效。")
            for field in ("localized_batches", "unlocalized_batches", "sample_occurrences"):
                if type(state.get(field)) is not int or state[field] < 0:
                    raise UserError("CHECKPOINT_SUPERVISION_COVERAGE_INVALID", f"规则 {rule_id} 的 {field} 无效。")
            for field, flag in (
                ("sample_id_hashes", "sample_ids_truncated"),
                ("violating_sample_id_hashes", "violating_ids_truncated"),
            ):
                hashes = state.get(field)
                if (
                    not isinstance(hashes, list)
                    or len(hashes) > _MAX_TRACKED_SAMPLE_IDS
                    or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes)
                    or type(state.get(flag)) is not bool
                ):
                    raise UserError("CHECKPOINT_SUPERVISION_COVERAGE_INVALID", f"规则 {rule_id} 的样本身份摘要无效。")
                target[field] = set(hashes)
                target[flag] = state[flag]
            for field in ("localized_batches", "unlocalized_batches", "sample_occurrences"):
                target[field] = state[field]

    def checkpoint_coverage(self):
        return {
            "schema_version": 1,
            "prior_untracked_steps": self.prior_untracked_steps,
            "rules": {
                rule_id: {
                    "localized_batches": state["localized_batches"],
                    "unlocalized_batches": state["unlocalized_batches"],
                    "sample_occurrences": state["sample_occurrences"],
                    "sample_id_hashes": sorted(state["sample_id_hashes"]),
                    "sample_ids_truncated": state["sample_ids_truncated"],
                    "violating_sample_id_hashes": sorted(state["violating_sample_id_hashes"]),
                    "violating_ids_truncated": state["violating_ids_truncated"],
                }
                for rule_id, state in self.sample_coverage.items()
            },
        }

    def snapshot(self):
        return [
            {"mark_id": entry["mark"]["id"],
             "role": entry["mark"]["role"],
             "module_path": entry["mark"]["module_path"],
             "port": entry["mark"]["port"],
             "rules": entry["mark"].get("rules", [])}
            for entry in self.bound_marks
        ]

    @contextmanager
    def capture_step(self, step, batch=None):
        # A user-owned loop may catch an exception outside its supervision
        # context and try to begin another step. Do not erase that failure.
        self._raise_captured_failure()
        self.step = step
        self._failure = None
        self._active_batch = batch
        self._active = {
            rule_id: {
                "penalties": [],
                "observations": [],
                "pending_calls": 0,
                "weighted_penalty": None,
                "violating_sample_ids": set(),
                "violating_ids_truncated": False,
            }
            for rule_id in self.observations
        }
        self._active_sample_ids = _sample_ids_from_batch(batch)
        self._sample_ids_changed_after_observation = False
        self._update_started = False
        self._penalty_added = False
        try:
            with ExitStack() as stack:
                for entry in self.bound_marks:
                    mark = entry["mark"]
                    module = entry["module"]

                    def before(current, args, kwargs, *, selected=entry):
                        if current is selected["module"] and self._active is not None:
                            for rule in selected["mark"].get("rules", []):
                                self._active[rule["id"]]["pending_calls"] += 1

                    def after(current, args, kwargs, output, *, selected=entry):
                        if current is not selected["module"]:
                            return
                        try:
                            selected_mark = selected["mark"]
                            inputs = {"args": args, "kwargs": kwargs}
                            value = _resolve_port((inputs, output), selected_mark["port"])
                            for rule in selected_mark.get("rules", []):
                                self._active[rule["id"]]["pending_calls"] -= 1
                                self._observe(selected_mark, rule, value)
                        except Exception as error:
                            # Project code may catch a hook exception. Retain
                            # the first failure until this step is finished so
                            # the managed backward/update cannot silently pass.
                            if self._failure is None:
                                self._failure = error
                            raise

                    pre_handle = module.register_forward_pre_hook(before, with_kwargs=True)
                    stack.callback(pre_handle.remove)
                    stack.enter_context(_hook_context(module, after))
                yield
        finally:
            self._active_batch = None

    def report_sample_ids(self, sample_ids, stable=True):
        """Accept explicit IDs from a custom batch path for this active step."""
        if self._active is None:
            raise UserError("SUPERVISION_SAMPLE_IDS_INVALID", "sample IDs 只能在监督训练步骤中报告。")
        if stable is not True:
            if self._active_sample_ids is not None and any(item["observations"] for item in self._active.values()):
                self._sample_ids_changed_after_observation = True
            self._active_sample_ids = None
            return
        validated = _validate_sample_ids(sample_ids)
        if self._active_sample_ids is not None and self._active_sample_ids != validated:
            raise UserError(
                "SUPERVISION_SAMPLE_IDS_CONFLICT",
                "batch metadata 与 context.report_sample_ids() 提供的样本 ID 不一致。",
            )
        if self._active_sample_ids != validated and any(item["observations"] for item in self._active.values()):
            self._sample_ids_changed_after_observation = True
        self._active_sample_ids = validated

    def _observe(self, mark, rule, value):
        if self._active is None:
            return
        if self._update_started:
            raise UserError(
                "SUPERVISION_FORWARD_AFTER_UPDATE",
                f"第 {self.step} 步已开始参数更新，之后又执行了受监督模块。",
                "將本步所有受監督 forward 移到第一次 optimizer.step() 之前。",
                {"mark_id": mark["id"], "rule_id": rule["id"], "step": self.step},
            )
        rule_id = rule["id"]
        penalty_tensor = None
        if value is _MISSING_PORT:
            summary = {"status": "unsupported", "reason": "标记的端口在本次调用中不存在或无法解析。"}
        elif rule["kind"] in {"custom-validator", "custom-penalty"}:
            summary = evaluate_plugin(
                self.plugin_targets[rule_id], rule["plugin"], rule["kind"], value,
                self._active_batch, self.torch, training=True,
                sample_ids=getattr(self, "_active_sample_ids", None), axis=rule.get("axis"),
            )
            penalty_tensor = summary.pop("penalty_tensor", None)
        else:
            summary = (
                _batch_alignment_summary(value, self._active_batch, rule, self.torch)
                if rule["kind"] == "batch-alignment"
                else _numeric_summary(
                    value, rule, self.torch,
                    sample_ids=getattr(self, "_active_sample_ids", None),
                )
            )
        summary["step"] = self.step
        if summary["status"] != "measured":
            if rule["mode"] in {"require", "penalty"}:
                raise UserError(
                    "SUPERVISION_PORT_UNSUPPORTED",
                    f"训练步骤 {self.step} 无法执行规则 {rule_id}：{summary.get('reason')}",
                    "规则在本次更新前失败；检查标记端口和 Tensor 类型。",
                    {"mark_id": mark["id"], "rule_id": rule_id, "step": self.step},
                )
            self._active[rule_id]["observations"].append(summary)
            return
        self._active[rule_id]["observations"].append(summary)
        localization = summary.get("sample_localization")
        if isinstance(localization, dict) and localization.get("status") == "localized":
            active = self._active[rule_id]
            active["violating_sample_ids"].update(
                item["sample_id"]
                for item in localization.get("violating_samples", [])
                if isinstance(item, dict) and isinstance(item.get("sample_id"), str)
            )
            active["violating_ids_truncated"] |= localization.get("violating_samples_truncated") is True
        if rule["mode"] == "require" and (summary["violation_count"] or summary.get("passed") is False):
            if rule["kind"] == "batch-alignment":
                message = (
                    f"端口形状 {summary['shape']} 与批次目标 {rule['target_path']} 的形状 "
                    f"{summary['target_shape']} 在 axis {rule['axis']} 不匹配。"
                )
            else:
                message = f"训练步骤 {self.step} 违反规则 {rule_id}：{summary['violation_count']}/{summary['element_count']} 个值越界或 validator 未通过。"
            raise UserError(
                "SUPERVISION_REQUIRE_FAILED",
                message,
                "本次规则在工具管理的反向和更新前拒绝；用户代码可能已有副作用，不会自动回滚。",
                {"mark_id": mark["id"], "rule_id": rule_id, "step": self.step,
                 "module_path": mark["module_path"], "port": mark["port"],
                 "observation_scope": _port_observation_scope(mark["port"]),
                 "violation_count": summary["violation_count"], "element_count": summary["element_count"],
                 "target_path": rule.get("target_path"), "axis": rule.get("axis"),
                 "sample_localization": summary.get("sample_localization"),
                 "custom_summary": summary.get("custom_summary"),
                 "shape": summary.get("shape"), "target_shape": summary.get("target_shape")},
            )
        if rule["mode"] == "penalty":
            if rule["kind"] == "custom-penalty":
                if penalty_tensor is None:
                    raise UserError("SUPERVISION_PENALTY_MISSING", f"规则 {rule_id} 没有可微 penalty Tensor。")
                self._active[rule_id]["penalties"].append(penalty_tensor * rule["weight"])
                return
            if summary["nonfinite_count"]:
                raise UserError(
                    "SUPERVISION_NONFINITE",
                    f"训练步骤 {self.step} 的 penalty 端口含非有限值。",
                    "工具管理的反向和更新前停止；用户代码可能已有副作用。先修复输出数值。",
                    {"mark_id": mark["id"], "rule_id": rule_id, "step": self.step,
                     "nonfinite_count": summary["nonfinite_count"]},
                )
            if not isinstance(value, self.torch.Tensor) or not value.is_floating_point() or not value.requires_grad:
                raise UserError(
                    "SUPERVISION_PENALTY_NOT_DIFFERENTIABLE",
                    f"规则 {rule_id} 绑定端口没有可微浮点梯度。",
                    "本步在 optimizer update 前停止；把 penalty 绑定到参与计算图的浮点输出。",
                    {"mark_id": mark["id"], "rule_id": rule_id, "step": self.step},
                )
            clean = value.reshape(-1)
            components = []
            if rule.get("min") is not None:
                components.append(self.torch.relu(rule["min"] - clean).square())
            if rule.get("max") is not None:
                components.append(self.torch.relu(clean - rule["max"]).square())
            weighted = self.torch.stack(components).sum(dim=0).mean() * rule["weight"]
            self._active[rule_id]["penalties"].append(weighted)

    def apply_manual_loss(self, loss):
        """Add named penalties only through the manual-step context.backward API."""
        if self._active is None:
            return loss, {}
        self._raise_captured_failure()
        self._check_incomplete_calls()
        for rule_id, record in self.observations.items():
            if record["mode"] == "require" and not self._active[rule_id]["observations"]:
                raise UserError(
                    "SUPERVISION_PORT_UNOBSERVED",
                    f"训练步骤 {self.step} 在 context.backward 前没有观测 require 规则 {rule_id}。",
                    "先运行受监督模块，再把任务 Loss 交给 context.backward。",
                    {"mark_id": record["mark_id"], "rule_id": rule_id,
                     "module_path": record["module_path"], "step": self.step},
                )
        penalty_rules = [
            (rule_id, record) for rule_id, record in self.observations.items()
            if record["mode"] == "penalty"
        ]
        if not penalty_rules:
            return loss, {}
        if self._penalty_added:
            raise UserError(
                "SUPERVISION_BACKWARD_REPEATED",
                "含 penalty 规则的 manual-step 只能通过 context.backward 聚合反向一次。",
                "先在用户代码中汇总本步任务 Loss，再调用一次 context.backward(loss)。",
            )
        named = {}
        terms = []
        for rule_id, record in penalty_rules:
            active = self._active[rule_id]
            if not active["observations"]:
                raise UserError(
                    "SUPERVISION_PORT_UNOBSERVED",
                    f"训练步骤 {self.step} 没有执行 penalty 规则 {rule_id} 对应的实现块。",
                    "在 context.backward 前运行该手动标记模块。",
                    {"mark_id": record["mark_id"], "rule_id": rule_id,
                     "module_path": record["module_path"], "step": self.step},
                )
            penalties = active["penalties"]
            if not penalties:
                raise UserError("SUPERVISION_PENALTY_MISSING", f"规则 {rule_id} 没有可微 penalty 项。")
            term = self.torch.stack(penalties).mean()
            active["weighted_penalty"] = term
            terms.append(term)
            named[f"supervision/{rule_id}"] = float(term.detach().item())
        total_penalty = self.torch.stack(terms).sum()
        self._penalty_added = True
        return loss + total_penalty, named

    def before_manual_update(self):
        if self._active is None:
            return
        self._raise_captured_failure()
        self._check_incomplete_calls()
        for rule_id, record in self.observations.items():
            active = self._active[rule_id]
            if record["mode"] in {"require", "penalty"} and not active["observations"]:
                raise UserError(
                    "SUPERVISION_PORT_UNOBSERVED",
                    f"训练步骤 {self.step} 在 optimizer update 前没有观测规则 {rule_id}。",
                    "将受监督模块的 forward 放到 optimizer.step() 之前。",
                    {"mark_id": record["mark_id"], "rule_id": rule_id,
                     "module_path": record["module_path"], "step": self.step},
                )
            if record["mode"] == "penalty" and not self._penalty_added:
                raise UserError(
                    "SUPERVISION_PENALTY_NOT_APPLIED",
                    f"规则 {rule_id} 的 penalty 没有通过 context.backward 加入目标函数。",
                    "使用 context.backward(total_loss)；本步不会更新参数。",
                    {"mark_id": record["mark_id"], "rule_id": rule_id, "step": self.step},
                )
        self._update_started = True

    def finish_manual_step(self):
        if self._active is None:
            return {}
        self._raise_captured_failure()
        self._check_incomplete_calls()
        named = {}
        for rule_id, record in self.observations.items():
            active = self._active[rule_id]
            measured = [item for item in active["observations"] if item["status"] == "measured"]
            unmeasured = len(active["observations"]) - len(measured) + active["pending_calls"]
            record["unmeasured_calls"] += unmeasured
            record["steps_with_unmeasured_calls"] += int(unmeasured > 0)
            if measured:
                record["steps_observed"] += 1
                record["element_count"] += sum(item["element_count"] for item in measured)
                record["violation_count"] += sum(item["violation_count"] for item in measured)
                _save_target_evidence(record, measured)
                _save_sample_localization(record, measured, active, self._active_sample_ids, self._sample_ids_changed_after_observation)
            self._record_sample_coverage(rule_id, active)
            penalty = active["weighted_penalty"]
            if penalty is not None:
                contribution = float(penalty.detach().item())
                record["weighted_penalty_sum"] += contribution
                named[f"supervision/{rule_id}"] = contribution
        self._active = None
        self._active_sample_ids = None
        return named

    def complete_step(self, loss):
        if self._active is None:
            return loss, {}
        self._raise_captured_failure()
        self._check_incomplete_calls()
        named = {}
        for rule_id, record in self.observations.items():
            active = self._active[rule_id]
            observations = active["observations"]
            if not observations and record["mode"] in {"require", "penalty"}:
                raise UserError(
                    "SUPERVISION_PORT_UNOBSERVED",
                    f"训练步骤 {self.step} 没有执行规则 {rule_id} 对应的实现块。",
                    "本步没有 backward 或参数更新；确认当前执行分支包含手动标记的模块。",
                    {"mark_id": record["mark_id"], "rule_id": rule_id,
                     "module_path": record["module_path"], "step": self.step},
                )
            measured = [item for item in observations if item["status"] == "measured"]
            unmeasured = len(observations) - len(measured) + active["pending_calls"]
            record["unmeasured_calls"] += unmeasured
            record["steps_with_unmeasured_calls"] += int(unmeasured > 0)
            if measured:
                record["steps_observed"] += 1
                record["element_count"] += sum(item["element_count"] for item in measured)
                record["violation_count"] += sum(item["violation_count"] for item in measured)
                _save_target_evidence(record, measured)
                _save_sample_localization(record, measured, active, self._active_sample_ids, self._sample_ids_changed_after_observation)
            self._record_sample_coverage(rule_id, active)
            penalties = active["penalties"]
            if penalties:
                penalty = self.torch.stack(penalties).mean()
                loss = loss + penalty
                contribution = float(penalty.detach().item())
                record["weighted_penalty_sum"] += contribution
                named[f"supervision/{rule_id}"] = contribution
        self._active = None
        self._active_sample_ids = None
        return loss, named

    def _check_incomplete_calls(self):
        if self._active is None:
            return
        for rule_id, record in self.observations.items():
            pending = self._active[rule_id]["pending_calls"]
            if pending and record["mode"] in {"require", "penalty"}:
                error = UserError("SUPERVISION_INCOMPLETE_CALL",
                                  f"训练步骤 {self.step} 的规则 {rule_id} 有 {pending} 次模块调用未成功返回。",
                                  "检查被用户代码捕获的前向异常；工具管理的更新不会继续。",
                                  {"mark_id": record["mark_id"], "rule_id": rule_id,
                                   "step": self.step, "incomplete_calls": pending})
                if self._failure is None:
                    self._failure = error
                raise error

    def _raise_captured_failure(self):
        if self._failure is None:
            return
        if isinstance(self._failure, UserError):
            raise self._failure
        raise UserError(
            "SUPERVISION_EVALUATION_FAILED",
            f"训练步骤 {self.step} 的监督检查失败：{type(self._failure).__name__}: {self._failure}",
            "即使用户代码捕获了异常，也不能将本步视为通过；检查规则和模型后重试。",
            {"step": self.step},
        ) from self._failure

    def _record_sample_coverage(self, rule_id, active):
        coverage = self.sample_coverage.get(rule_id)
        if coverage is None:
            return
        observations = active["observations"]
        measured = [item for item in observations if item.get("status") == "measured"]
        if not observations:
            if active["pending_calls"]:
                coverage["unlocalized_batches"] += 1
            return
        if not measured:
            coverage["unlocalized_batches"] += 1
            return
        fully_localized = (
            len(measured) == len(observations)
            and bool(observations)
            and all(item.get("sample_localization", {}).get("status") == "localized" for item in measured)
            and self._active_sample_ids is not None
            and not self._sample_ids_changed_after_observation
            and active["pending_calls"] == 0
        )
        if not fully_localized:
            coverage["unlocalized_batches"] += 1
            return
        sample_ids = set(self._active_sample_ids)
        coverage["localized_batches"] += 1
        coverage["sample_occurrences"] += len(sample_ids)
        _add_sample_id_hashes(
            coverage["sample_id_hashes"],
            sample_ids,
            coverage,
            "sample_ids_truncated",
            _MAX_TRACKED_SAMPLE_IDS,
        )
        _add_sample_id_hashes(
            coverage["violating_sample_id_hashes"],
            active["violating_sample_ids"],
            coverage,
            "violating_ids_truncated",
            _MAX_TRACKED_VIOLATING_IDS,
        )
        if active["violating_ids_truncated"]:
            coverage["violating_ids_truncated"] = True

    def report(self):
        return {
            "status": "applied",
            "training_mode": self.mode,
            "coverage_status": "partial",
            "rules": [
                {**record,
                 "violation_ratio": record["violation_count"] / record["element_count"] if record["element_count"] else None,
                 "mean_weighted_penalty": record["weighted_penalty_sum"] / record["steps_observed"]
                 if record["steps_observed"] and record["mode"] == "penalty" else None,
                 "sample_coverage": self._sample_coverage_report(record["rule_id"])}
                for record in self.observations.values()
            ],
            "coverage": f"仅覆盖实际执行且被手动标记的 {self.mode} 模块端口。",
            "sample_localization": _rules_sample_localization(self.observations.values(), self.step, self.sample_coverage, self.prior_untracked_steps),
        }

    def _sample_coverage_report(self, rule_id):
        state = self.sample_coverage.get(rule_id)
        if state is None:
            return {"status": "not_configured"}
        history_complete = self.prior_untracked_steps == 0
        if state["localized_batches"] == 0:
            status = "insufficient"
        elif state["unlocalized_batches"] or not history_complete:
            status = "partial"
        else:
            status = "reported"
        all_observed_batches_localized = state["localized_batches"] > 0 and state["unlocalized_batches"] == 0
        return {
            "status": status,
            "identity_source": "user_declared_stable_sample_ids",
            "localized_batches": state["localized_batches"],
            "unlocalized_batches": state["unlocalized_batches"],
            "prior_steps_without_coverage_state": self.prior_untracked_steps,
            "sample_occurrences": state["sample_occurrences"],
            "unique_sample_count_lower_bound": len(state["sample_id_hashes"]),
            "unique_sample_count_complete": history_complete and all_observed_batches_localized and not state["sample_ids_truncated"],
            "unique_violating_sample_count_lower_bound": len(state["violating_sample_id_hashes"]),
            "unique_violating_sample_count_complete": history_complete and all_observed_batches_localized and not state["violating_ids_truncated"],
            "dataset_population_size": None,
            "coverage_fraction": None,
            "limitations": "只统计规则实际观测到的批次；用户声明的 ID 未与数据集权威身份交叉验证，也没有数据集总量作为覆盖率分母。",
        }


@contextmanager
def _hook_context(module, hook):
    handle = module.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def _resolve_port(value, path):
    """Resolve a manually selected module input/output port path."""
    if not isinstance(path, str):
        return _MISSING_PORT
    if path == "inputs" or path.startswith(("inputs.", "inputs[")):
        current = value[0]
        tail = path[len("inputs"):]
    elif path == "outputs" or path.startswith(("outputs.", "outputs[")):
        current = value[1]
        tail = path[len("outputs"):]
    else:
        return _MISSING_PORT
    if not tail:
        return current
    position = 0
    while position < len(tail):
        match = _PATH_TOKEN.match(tail, position)
        if match is None:
            return _MISSING_PORT
        name, index, quoted = match.groups()
        try:
            if quoted is not None:
                name = json.loads(quoted)
            if name is not None:
                if not isinstance(current, dict) or name not in current:
                    return _MISSING_PORT
                current = current[name]
            else:
                if not isinstance(current, (list, tuple)):
                    return _MISSING_PORT
                current = current[int(index)]
        except (IndexError, KeyError, TypeError, ValueError):
            return _MISSING_PORT
        position = match.end()
    return current


def _resolve_batch_target(batch, path):
    if not isinstance(path, str) or not (path.startswith("batch.") or path.startswith("batch[")):
        return None
    tail = path[5:]
    current = batch
    position = 0
    while position < len(tail):
        match = _PATH_TOKEN.match(tail, position)
        if match is None:
            return None
        name, index, quoted = match.groups()
        try:
            if quoted is not None:
                name = json.loads(quoted)
            if name is not None:
                if not isinstance(current, dict) or name not in current:
                    return None
                current = current[name]
            else:
                if not isinstance(current, (list, tuple)):
                    return None
                current = current[int(index)]
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        position = match.end()
    return current if position == len(tail) else None


def _batch_alignment_summary(value, batch, rule, torch_module):
    target = _resolve_batch_target(batch, rule["target_path"])
    if not isinstance(value, torch_module.Tensor):
        return {"status": "unsupported", "reason": "marked model port is not a Tensor"}
    if not isinstance(target, torch_module.Tensor):
        return {
            "status": "unsupported",
            "reason": f"target path {rule['target_path']} did not resolve to a Tensor",
            "target_path": rule["target_path"],
        }
    axis = rule["axis"]
    if axis >= value.ndim or axis >= target.ndim:
        return {
            "status": "unsupported",
            "reason": f"axis {axis} is outside one or both tensor ranks",
            "shape": list(value.shape),
            "target_shape": list(target.shape),
            "axis": axis,
            "target_path": rule["target_path"],
        }
    matched = value.shape[axis] == target.shape[axis]
    return {
        "status": "measured",
        "shape": list(value.shape),
        "target_shape": list(target.shape),
        "axis": axis,
        "target_path": rule["target_path"],
        "model_axis_size": int(value.shape[axis]),
        "target_axis_size": int(target.shape[axis]),
        "element_count": 1,
        "violation_count": 0 if matched else 1,
        "violation_ratio": 0.0 if matched else 1.0,
    }


def _save_target_evidence(record, measured):
    if record["kind"] != "batch-alignment":
        return
    for item in measured:
        if "target_shape" in item:
            record["last_target_evidence"] = {
                "target_path": item["target_path"],
                "axis": item["axis"],
                "model_shape": item["shape"],
                "target_shape": item["target_shape"],
                "matched": item["violation_count"] == 0,
                "step": item.get("step"),
            }


def _save_sample_localization(record, measured, active, sample_ids, ids_changed=False):
    localized = [item.get("sample_localization") for item in measured]
    localized = [item for item in localized if isinstance(item, dict)]
    if localized:
        complete = (
            not ids_changed
            and not active["pending_calls"]
            and len(measured) == len(active["observations"])
            and len(localized) == len(measured)
            and all(item.get("status") == "localized" for item in localized)
        )
        record["last_sample_localization"] = {
            **localized[-1],
            "step": measured[-1].get("step"),
            "invocations_in_step": len(localized),
            "batch_sample_count": len(sample_ids) if sample_ids is not None else None,
            "violating_samples_observed_in_step": len(active.get("violating_sample_ids", set())),
            "violating_samples_observed_in_step_complete": complete and not active.get("violating_ids_truncated", False),
        }
        if not complete:
            reasons = []
            if ids_changed:
                reasons.append("本步观察期间样本 ID 声明发生变化，不能将所有调用归入同一个批次。")
            if active["pending_calls"] or len(measured) != len(active["observations"]) or len(localized) != len(measured):
                reasons.append("本步存在未完成或无法测量的调用。")
            if any(item.get("status") != "localized" for item in localized):
                reasons.append("至少一次调用缺少可用的样本轴或稳定样本 ID，无法完成逐样本定位。")
            record["last_sample_localization"].update({
                "status": "partial",
                "stable_ids_available": False,
                "reason": " ".join(reasons),
            })


def _new_sample_coverage():
    return {
        "localized_batches": 0,
        "unlocalized_batches": 0,
        "sample_occurrences": 0,
        "sample_id_hashes": set(),
        "sample_ids_truncated": False,
        "violating_sample_id_hashes": set(),
        "violating_ids_truncated": False,
    }


def _add_sample_id_hashes(target, sample_ids, state, truncation_field, limit):
    for sample_id in sorted(set(sample_ids)):
        digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        if digest in target:
            continue
        if len(target) >= limit:
            state[truncation_field] = True
            continue
        target.add(digest)


def _validate_sample_ids(sample_ids):
    if not isinstance(sample_ids, (list, tuple)) or not 1 <= len(sample_ids) <= 100_000:
        raise UserError("SUPERVISION_SAMPLE_IDS_INVALID", "稳定 sample IDs 必须是 1..100000 项的列表。")
    if any(
        not isinstance(sample_id, str)
        or not sample_id.strip()
        or len(sample_id) > 512
        or any(ord(character) < 32 for character in sample_id)
        for sample_id in sample_ids
    ):
        raise UserError("SUPERVISION_SAMPLE_IDS_INVALID", "稳定 sample IDs 须为 1..512 字符且不含控制字符的非空字符串。")
    values = list(sample_ids)
    if len(set(values)) != len(values):
        raise UserError("SUPERVISION_SAMPLE_IDS_INVALID", "稳定 sample IDs 在当前批次中必须唯一。")
    return values


def _sample_ids_from_batch(batch):
    if not isinstance(batch, dict):
        return None
    metadata = batch.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("stable_sample_ids") is not True:
        return None
    return _validate_sample_ids(metadata.get("sample_ids"))


def _rules_sample_localization(records, current_step=None, coverage=None, prior_untracked_steps=0):
    configured = [
        record for record in records
        if record.get("kind") in {"finite", "range"}
        and record.get("expected", {}).get("axis") is not None
    ]
    if configured and all(
        isinstance(record.get("last_sample_localization"), dict)
        and record["last_sample_localization"].get("status") == "localized"
        and record["last_sample_localization"].get("step") == current_step
        for record in configured
    ):
        report = _sample_localization_summary(True, [record["rule_id"] for record in configured])
        report["localized_batches"] = sum(
            (coverage or {}).get(record["rule_id"], {}).get("localized_batches", 0)
            for record in configured
        )
        report["prior_steps_without_coverage_state"] = prior_untracked_steps
        return report
    return _sample_localization_summary()


def _numeric_summary(tensor, rule, torch_module, *, sample_ids=None):
    if not isinstance(tensor, torch_module.Tensor):
        return {"status": "unsupported", "reason": "bound port is not a Tensor"}
    if tensor.layout != torch_module.strided:
        return {"status": "unsupported", "reason": f"numeric rules do not inspect tensor layout {tensor.layout}"}
    if tensor.device.type == "meta":
        return {"status": "unsupported", "reason": "meta tensors have no values to inspect"}
    if tensor.is_quantized:
        return {"status": "unsupported", "reason": "numeric rules do not inspect quantized tensor values"}
    if tensor.numel() == 0:
        return {"status": "unsupported", "reason": "empty tensors have no value summary"}
    complex_value = tensor.is_complex()
    if tensor.dtype == torch_module.bool:
        return {"status": "unsupported", "reason": "boolean tensors are not supported by numeric rules"}
    if complex_value and rule["kind"] != "finite":
        return {"status": "unsupported", "reason": "range comparisons require ordered real values; complex tensors support finite checks"}
    if rule["mode"] == "penalty" and not tensor.is_floating_point():
        return {"status": "unsupported", "reason": "range penalties require floating-point tensors"}
    values = tensor.detach().reshape(-1)
    finite_mask = torch_module.isfinite(values)
    finite_count = int(finite_mask.sum().item())
    count = int(values.numel())
    finite_values = values[finite_mask]
    minimum_value = float(finite_values.min().item()) if finite_count and not complex_value else None
    maximum_value = float(finite_values.max().item()) if finite_count and not complex_value else None
    violations = count - finite_count
    lower_violations = 0
    upper_violations = 0
    if rule["kind"] == "range":
        if rule.get("min") is not None:
            lower_violations = int((finite_values < rule["min"]).sum().item())
        if rule.get("max") is not None:
            upper_violations = int((finite_values > rule["max"]).sum().item())
        violations += lower_violations + upper_violations
    violation_mask = ~finite_mask
    if rule["kind"] == "range":
        if rule.get("min") is not None:
            violation_mask |= finite_mask & (values < rule["min"])
        if rule.get("max") is not None:
            violation_mask |= finite_mask & (values > rule["max"])
    penalty_value = None
    if rule["mode"] == "penalty":
        # This is a detached preview estimate only; it never enters autograd or training.
        clean = torch_module.where(finite_mask, values, torch_module.zeros_like(values))
        components = []
        if rule.get("min") is not None:
            components.append(torch_module.relu(rule["min"] - clean).square())
        if rule.get("max") is not None:
            components.append(torch_module.relu(clean - rule["max"]).square())
        if components:
            penalty_value = float(torch_module.stack(components).sum(dim=0).mean().item()) * rule["weight"]
    sample_localization = {
        "status": "insufficient",
        "stable_ids_available": False,
        "reason": "未为该规则人工指定样本轴，或本批次没有稳定 sample IDs。",
    }
    sample_axis = rule.get("axis")
    if sample_axis is not None:
        if sample_axis >= tensor.ndim:
            return {"status": "unsupported", "reason": f"sample axis {sample_axis} is outside tensor rank {tensor.ndim}", "shape": list(tensor.shape)}
        if sample_ids is None:
            return {
                "status": "unsupported",
                "reason": "规则指定了样本轴，但当前批次没有稳定 sample IDs；未宣称逐样本检查通过。",
                "shape": list(tensor.shape),
                "sample_axis": sample_axis,
            }
        elif len(sample_ids) != tensor.shape[sample_axis]:
            return {
                "status": "unsupported",
                "reason": "稳定 sample ID 数量与人工指定的 Tensor 样本轴长度不匹配",
                "shape": list(tensor.shape),
                "sample_axis": sample_axis,
                "sample_count": len(sample_ids),
                "axis_size": int(tensor.shape[sample_axis]),
            }
        else:
            per_sample = violation_mask.reshape(
                tuple(tensor.shape[:sample_axis]) + (tensor.shape[sample_axis],) + tuple(tensor.shape[sample_axis + 1 :])
            ).movedim(sample_axis, 0).reshape(len(sample_ids), -1).sum(dim=1)
            violating_mask = per_sample != 0
            violating_count = int(violating_mask.sum().item())
            # Keep the exact count, but transfer at most the 50 reported
            # positions and counts to Python instead of materializing one
            # dictionary per violating sample for large batches.
            selected_indices = violating_mask.nonzero(as_tuple=False).flatten()[:50]
            positions = selected_indices.tolist()
            selected_counts = per_sample.index_select(0, selected_indices).tolist()
            violating = [
                {"sample_id": sample_ids[index], "violation_count": int(count)}
                for index, count in zip(positions, selected_counts)
            ]
            sample_localization = {
                "status": "localized",
                "stable_ids_available": True,
                "sample_axis": sample_axis,
                "sample_count": len(sample_ids),
                "violating_sample_count": violating_count,
                "violating_samples": violating,
                "violating_samples_truncated": violating_count > len(violating),
            }
    return {
        "status": "measured",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "requires_grad": bool(tensor.requires_grad),
        "element_count": count,
        "finite_count": finite_count,
        "nonfinite_count": count - finite_count,
        "value_min": minimum_value,
        "value_max": maximum_value,
        "below_min_count": lower_violations,
        "above_max_count": upper_violations,
        "violation_count": violations,
        "violation_ratio": violations / count,
        "weighted_penalty_preview": penalty_value,
        "sample_localization": sample_localization,
    }


def preview_supervision(
    model,
    request,
    *,
    context,
    torch_module,
    make_batch_fn,
    resolve_entrypoint,
    module_entries_fn,
):
    """Run one read-only forward and summarize rules without retaining values."""
    root = request["project_root"]
    marks_report = list_marks(root)
    # Semantic block: S-supervision/binding — only manually marked ports are eligible.
    rules = [
        (mark, rule)
        for mark in marks_report["marks"]
        for rule in mark.get("rules", [])
    ]
    if not rules:
        return {
            "status": "no_rules",
            "results": [],
            "coverage_status": "partial",
            "coverage": "项目中没有绑定的监督规则；没有执行模型前向。",
            "sample_localization": _sample_localization_summary(),
        }

    results = []
    eligible = []
    modules, aliases = module_entries_fn(model)
    modules_by_path = {}
    for key, paths in aliases.items():
        for module_path in paths:
            modules_by_path.setdefault(module_path, []).append(modules[key])
    implementation_cache = {}
    for mark, rule in rules:
        result = {
            "mark_id": mark["id"],
            "rule_id": rule["id"],
            "role": mark["role"],
            "module_path": mark["module_path"],
            "port": mark["port"],
            "observation_scope": _port_observation_scope(mark["port"]),
            "kind": rule["kind"],
            "mode": rule["mode"],
            "expected": {key: rule.get(key) for key in ("min", "max", "weight", "target_path", "axis") if rule.get(key) is not None},
            "plugin_entrypoint": rule.get("plugin", {}).get("entrypoint") if isinstance(rule.get("plugin"), dict) else None,
            "observations": [],
        }
        if mark["binding_status"] == "stale":
            result.update({"status": "stale", "reason": mark.get("stale_reason")})
        elif mark.get("binding_context") != context:
            saved = mark.get("binding_context", {})
            changed = sorted(key for key in set(saved) | set(context) if saved.get(key) != context.get(key))
            result.update({"status": "stale", "reason": "模型源码、参数、注入或变体与标记绑定上下文不同。", "stale_fields": changed, "binding_context": saved, "current_context": context})
        else:
            targets = modules_by_path.get(mark["module_path"], [])
            if len(targets) != 1:
                result.update({"status": "stale", "reason": "当前模型无法唯一定位标记的模块路径。"})
                results.append(result)
                continue
            module = targets[0]
            module_type = type(module)
            if module_type not in implementation_cache:
                implementation_cache[module_type] = implementation_identity(implementation_anchor(
                    module, request.get("code_snapshot_root") or root
                ))
            current_anchor = implementation_cache[module_type]
            if current_anchor is None or current_anchor != implementation_identity(mark.get("anchor")):
                result.update({"status": "stale", "reason": "当前模块的实现来源与标记不同。"})
                results.append(result)
                continue
            if rule.get("kind") in {"custom-validator", "custom-penalty"}:
                try:
                    result["_plugin_target"] = resolve_plugin(
                        root, rule.get("plugin"), rule["kind"], rule["mode"],
                        code_root=request.get("code_snapshot_root"),
                    )
                except UserError as error:
                    result.update({"status": "stale" if error.code == "SUPERVISION_PLUGIN_STALE" else "unsupported", "reason": str(error)})
                else:
                    eligible.append((mark, rule, result))
            else:
                eligible.append((mark, rule, result))
        results.append(result)

    if not eligible:
        return {
            "status": "stale",
            "results": results,
            "coverage_status": "partial",
            "coverage": "所有规则绑定均已过期；未执行前向。",
            "sample_localization": _sample_localization_summary(),
        }

    training_states = [(module, module.training) for module in modules.values()]
    paths_by_module_id = {
        id(modules[key]): paths for key, paths in aliases.items()
    }
    handles = []
    invocation = 0
    call_id_prefix = "supervision-" + uuid.uuid4().hex
    pending_calls = {id(module): 0 for module in modules.values()}
    seen = {rule["id"]: 0 for _, rule, _ in eligible}

    def before(module, args, kwargs):
        pending_calls[id(module)] += 1

    def after(module, args, kwargs, output):
        nonlocal invocation
        pending_calls[id(module)] -= 1
        invocation += 1
        call_id = f"{call_id_prefix}:call-{invocation}"
        module_paths = paths_by_module_id.get(id(module), [])
        for mark, rule, result in eligible:
            if mark["module_path"] not in module_paths:
                continue
            if seen[rule["id"]] >= _MAX_INVOCATIONS_PER_RULE:
                result["observations_truncated"] = True
                continue
            inputs = {"args": args, "kwargs": kwargs}
            value = _resolve_port((inputs, output), mark["port"])
            if value is _MISSING_PORT:
                summary = {"status": "unsupported", "reason": "标记的端口在本次调用中不存在或无法解析。"}
            elif rule["kind"] in {"custom-validator", "custom-penalty"}:
                summary = evaluate_plugin(
                    result["_plugin_target"], rule["plugin"], rule["kind"], value,
                    _batch, torch_module, training=False,
                    sample_ids=_sample_ids_from_batch(_batch), axis=rule.get("axis"),
                )
                penalty_tensor = summary.pop("penalty_tensor", None)
                if rule["kind"] == "custom-penalty" and penalty_tensor is not None:
                    summary["weighted_penalty_preview"] = float(penalty_tensor.detach().item()) * rule["weight"]
            else:
                summary = (
                    _batch_alignment_summary(value, _batch, rule, torch_module)
                    if rule["kind"] == "batch-alignment"
                    else _numeric_summary(
                        value, rule, torch_module,
                        sample_ids=_sample_ids_from_batch(_batch),
                    )
                )
            summary["invocation_id"] = call_id
            result["observations"].append(summary)
            seen[rule["id"]] += 1

    try:
        args, kwargs, _batch = make_batch_fn(request, torch_module, resolve_entrypoint)
        for module in modules.values():
            handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(module.register_forward_hook(after, with_kwargs=True))
        model.eval()
        with torch_module.no_grad():
            model(*args, **kwargs)
    except UserError:
        raise
    except Exception as error:
        raise UserError(
            "SUPERVISION_PREVIEW_FAILED",
            f"监督预览前向失败：{type(error).__name__}: {error}",
            "检查模型和批次；工具没有执行反向或优化器更新，用户代码可能已有副作用。",
        ) from error
    finally:
        for handle in handles:
            handle.remove()
        # Restore per-module state exactly; root.train() would overwrite mixed modes.
        for module, training in training_states:
            module.training = training

    for mark, rule, result in eligible:
        failed = sum(pending_calls.get(id(modules[key]), 0) for key, paths in aliases.items()
                     if mark["module_path"] in paths)
        if failed:
            result["observations"].append({"status": "unsupported", "incomplete_calls": failed,
                                           "reason": "模块调用未成功返回，可能有前向异常被用户代码捕获。"})

    for _mark, rule, result in eligible:
        observations = result["observations"]
        measured = [item for item in observations if item["status"] == "measured"]
        result.pop("_plugin_target", None)
        if not observations:
            result.update({"status": "unobserved", "reason": "所选模块端口本次前向没有出现。"})
            continue
        if not measured:
            result.update({"status": "unsupported", "reason": observations[0].get("reason")})
            continue
        violations = sum(item.get("violation_count", 0) for item in measured)
        elements = sum(item.get("element_count", 0) for item in measured)
        result["element_count"] = elements
        result["violation_count"] = violations
        result["violation_ratio"] = violations / elements if elements else None
        validator_failed = any(item.get("passed") is False for item in measured)
        if rule["mode"] == "observe":
            result["status"] = "observed_violation" if violations or validator_failed else "observed"
        elif rule["mode"] == "require":
            result["status"] = "would_reject" if violations or validator_failed else "passed"
        else:
            result["status"] = "penalty_preview_only"
            result["objective_applied"] = False
            result["warning"] = "本次只展示加权 penalty 数值；它尚未参与训练 loss 或参数更新。"
        if result.get("observations_truncated") is True or len(measured) != len(observations):
            if result["status"] in {"passed", "observed"}:
                result["status"] = "partial"
                result["reason"] = "部分调用未记录或无法测量，已测部分没有发现违反；不能判断完整结果。"
            elif result["status"] == "penalty_preview_only":
                result["reason"] = "部分调用未记录或无法测量；惩罚数值只覆盖已测调用。"

    localized_rules, partially_localized_rules = [], []
    for _mark, rule, result in eligible:
        if rule.get("kind") not in {"finite", "range"} or rule.get("axis") is None:
            continue
        observations = result.get("observations", [])
        localized = [item for item in observations if item.get("status") == "measured"
                     and item.get("sample_localization", {}).get("status") == "localized"]
        if not localized:
            continue
        if result.get("observations_truncated") is not True and len(localized) == len(observations):
            localized_rules.append(rule["id"])
        else:
            partially_localized_rules.append(rule["id"])
    localization = _sample_localization_summary(bool(localized_rules), localized_rules)
    localization["partial_rules"] = partially_localized_rules
    return {
        "status": "observed-runtime",
        "results": results,
        "coverage_status": "partial",
        "probe_id": call_id_prefix,
        "objective_applied": False,
        "execution_mode": "eval",
        "gradients_enabled": False,
        "coverage": (
            "eval 和无梯度模式下单次真实前向的已观察模块/端口；工具未反向或更新权重，不证明训练模式或未执行分支。用户代码可能有副作用。"
            + ("已列出规则的已记录调用按用户声明的样本 ID 定位；不证明 ID 真实性或未观察调用。" if localization["stable_ids_available"] else "没有足够证据定位规则的全部已记录调用；部分调用可能仍有局部样本定位。")
        ),
        "sample_localization": localization,
    }
