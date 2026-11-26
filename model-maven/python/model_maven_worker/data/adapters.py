"""Data-source contracts shared by native datasets and direct batch factories.

The adapter can call a factory or sample a returned dataset instance. A factory
may return shared state or interact with an external stream, so independent
sampling cannot be inferred merely from calling it.
"""

from dataclasses import dataclass
import hashlib
import heapq
from typing import Any, Callable

from ..protocol import UserError


MAP_SAMPLE_ALGORITHM = "sha256-index-rank-v1"
LARGE_MAP_SAMPLE_ALGORITHM = "sha256-counter-rejection-v2"
ITERABLE_SAMPLE_ALGORITHM = "fresh-prefix-v1"
ITERABLE_WINDOW_SAMPLE_ALGORITHM = "sha256-window-rank-v1"


@dataclass(frozen=True)
class DataCapabilities:
    inspect_batch: bool
    forward_probe: bool
    independent_sampling: bool | None
    stable_sample_ids: bool
    checkpoint_save: bool
    checkpoint_load: bool
    resume_cursor: bool
    reason: str | None = None


@dataclass(frozen=True)
class DataObservation:
    source: str
    adapter: str
    adapter_version: str
    split: str | None
    dataset_fingerprint: str | None
    algorithm: str
    seed: int
    requested_samples: int
    actual_samples: int
    sample_ids: tuple[str, ...]
    stable_sample_ids: bool
    coverage: dict[str, Any]
    capabilities: DataCapabilities

    def to_dict(self):
        return {
            "source": self.source,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "split": self.split,
            "dataset_fingerprint": self.dataset_fingerprint,
            "sampling": {
                "algorithm": self.algorithm,
                "seed": self.seed,
                "seed_decimal": str(self.seed),
                "requested_samples": self.requested_samples,
                "actual_samples": self.actual_samples,
                "sample_ids": list(self.sample_ids),
                "stable_sample_ids": self.stable_sample_ids,
            },
            "coverage": dict(self.coverage),
            "capabilities": {
                "inspect_batch": self.capabilities.inspect_batch,
                "forward_probe": self.capabilities.forward_probe,
                "independent_sampling": self.capabilities.independent_sampling,
                "stable_sample_ids": self.capabilities.stable_sample_ids,
                "checkpoint_save": self.capabilities.checkpoint_save,
                "checkpoint_load": self.capabilities.checkpoint_load,
                "resume_cursor": self.capabilities.resume_cursor,
                "reason": self.capabilities.reason,
            },
        }


@dataclass(frozen=True)
class DatasetSample:
    batch: Any
    observation: DataObservation
    state: dict[str, Any] | None


class DatasetAdapter:
    """Adapter contract for a source factory that can create fresh datasets."""

    name = "python-dataset-factory"
    version = "1"

    def __init__(self, source: str, factory: Callable[..., Any], parameters=None):
        self.source = source
        self.factory = factory
        self.parameters = dict(parameters or {})

    def _factory_arguments(self, split=None, seed=0):
        parameters = dict(self.parameters)
        if split is not None and "split" in parameters and parameters["split"] != split:
            raise UserError("DATASET_SPLIT_CONFLICT", "所选切分与数据工厂参数中的 split 不一致。", "统一切分选择和工厂参数后重试；数据工厂未执行。")
        if split is not None and _accepts_keyword(self.factory, "split"):
            parameters.setdefault("split", split)
        # An explicitly supplied parameter is always forwarded. The adapter's
        # default seed is added only when the exposed signature accepts it.
        if _accepts_keyword(self.factory, "seed"):
            parameters.setdefault("seed", seed)
        return parameters

    def create(self, split=None, seed=0):
        return self.factory(**self._factory_arguments(split=split, seed=seed))

    def sample(
        self,
        *,
        count=1,
        seed=0,
        split=None,
        fingerprint=None,
        collate=None,
        scan_limit=None,
    ) -> DatasetSample:
        _validate_sampling_request(count, seed)
        _validate_fingerprint(fingerprint)
        _validate_scan_limit(count, scan_limit)
        try:
            if type(self).create is DatasetAdapter.create:
                factory_arguments = self._factory_arguments(split=split, seed=seed)
                dataset = self.factory(**factory_arguments)
                factory_seed_forwarded = "seed" in factory_arguments
                factory_seed = factory_arguments.get("seed") if type(factory_arguments.get("seed")) is int else None
                split_handling = "factory_argument" if split is not None and "split" in factory_arguments else "unverified" if split is not None else "not_requested"
            else:
                # An adapter override controls construction; do not infer what
                # it forwarded from the wrapped factory's signature.
                dataset = self.create(split=split, seed=seed)
                factory_seed_forwarded = None
                factory_seed = None
                split_handling = "unverified" if split is not None else "not_requested"
        except UserError:
            raise
        except Exception as error:
            raise UserError("DATASET_FACTORY_FAILED", f"数据工厂执行失败：{error}") from error
        if isinstance(dataset, dict) and split is not None and split in dataset:
            dataset = dataset[split]
            split_handling = "dictionary_key"
        elif isinstance(dataset, dict) and split is not None:
            raise UserError("DATASET_SPLIT_MISSING", f"数据工厂没有返回 {split!r} 切分。")
        return self.sample_instance(
            dataset,
            count=count,
            seed=seed,
            split=split,
            fingerprint=fingerprint,
            collate=collate,
            scan_limit=scan_limit,
            factory_seed_forwarded=factory_seed_forwarded,
            factory_seed=factory_seed,
            split_handling=split_handling,
        )

    def sample_instance(
        self,
        dataset,
        *,
        count=1,
        seed=0,
        split=None,
        fingerprint=None,
        collate=None,
        scan_limit=None,
        factory_seed_forwarded=None,
        factory_seed=None,
        split_handling=None,
    ):
        _validate_sampling_request(count, seed)
        _validate_fingerprint(fingerprint)
        _validate_scan_limit(count, scan_limit)
        if dataset is None:
            raise UserError("DATASET_FACTORY_INVALID", "数据工厂没有返回 Dataset 或可迭代数据源。")
        if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            if scan_limit is not None:
                raise UserError("DATA_SCAN_LIMIT_UNSUPPORTED", "扫描上限仅用于可迭代的非索引数据集。")
            return self._sample_map(dataset, count, seed, split, fingerprint, collate, factory_seed_forwarded, factory_seed, split_handling)
        if hasattr(dataset, "__iter__"):
            return self._sample_iterable(dataset, count, seed, split, fingerprint, collate, scan_limit, factory_seed_forwarded, factory_seed, split_handling)
        raise UserError("DATASET_FACTORY_INVALID", "数据工厂返回值既不可索引也不可迭代。")

    def _sample_map(self, dataset, count, seed, split, fingerprint, collate, seed_forwarded, factory_seed, split_handling):
        try:
            size = len(dataset)
        except Exception as error:
            raise UserError("DATASET_LENGTH_FAILED", f"无法读取数据集长度：{error}") from error
        if type(size) is not int or size < 0:
            raise UserError("DATASET_LENGTH_INVALID", "数据集长度必须是非负整数。")
        actual = min(count, size)
        if actual == 0:
            raise UserError("DATASET_EMPTY", "选中的数据切分没有可用样本。")
        if size > 100_000 and actual <= size // 4:
            # Hash a counter instead of ranking every index. Rejection avoids
            # modulo bias and a set provides sampling without replacement.
            # The named algorithm makes this deliberate selection change visible.
            algorithm = LARGE_MAP_SAMPLE_ALGORITHM
            accepted = set()
            counter = 0
            space = 1 << 256
            limit = space - space % size
            while len(accepted) < actual:
                raw = hashlib.sha256(
                    f"model-maven:sample-v2:{seed}:{size}:{counter}".encode("utf-8")
                ).digest()
                value = int.from_bytes(raw, "big")
                if value < limit:
                    accepted.add(value % size)
                counter += 1
            indices = sorted(accepted)
        else:
            algorithm = MAP_SAMPLE_ALGORITHM
            indices = sorted(
                heapq.nsmallest(
                    actual,
                    range(size),
                    key=lambda index: hashlib.sha256(
                        f"model-maven:{seed}:{index}".encode("utf-8")
                    ).digest(),
                )
            )
        try:
            values = [dataset[index] for index in indices]
        except Exception as error:
            raise UserError("DATASET_SAMPLE_FAILED", f"读取确定性样本失败：{error}") from error
        sample_ids, ids_stable = _sample_ids(dataset, indices, fingerprint)
        state = _state_snapshot(dataset)
        can_save = state is not None
        can_load = callable(getattr(dataset, "load_state_dict", None))
        hooks = can_save and can_load
        stateless = bool(getattr(dataset, "MODEL_MAVEN_STATELESS", False))
        resumable = hooks or stateless
        reasons = []
        reasons.append("无法确认数据实例或外部状态与训练迭代器相互独立。")
        if not ids_stable:
            reasons.append("未提供带数据指纹约束的稳定 sample_id(index)，repeat 不会声称样本身份稳定。")
        if not resumable:
            reasons.append("数据集未声明可保存/恢复的游标状态；repeat/resume 不保证无损复现。")
        capabilities = DataCapabilities(
            inspect_batch=actual > 0,
            forward_probe=actual > 0,
            independent_sampling=None,
            stable_sample_ids=ids_stable,
            checkpoint_save=can_save,
            checkpoint_load=can_load,
            resume_cursor=resumable,
            reason=" ".join(reasons) or None,
        )
        observation = DataObservation(
            source=self.source,
            adapter=self.name,
            adapter_version=self.version,
            split=split,
            dataset_fingerprint=fingerprint,
            algorithm=algorithm,
            seed=seed,
            requested_samples=count,
            actual_samples=actual,
            sample_ids=tuple(sample_ids),
            stable_sample_ids=ids_stable,
            coverage={
                "kind": "indexed-subset",
                "split_handling": split_handling or ("unverified" if split is not None else "not_requested"),
                "seed_forwarded_to_factory": seed_forwarded,
                "factory_seed": factory_seed,
                "population_size": size,
                "selected_indices": indices,
                "deterministic_selection": True,
                "complete": actual == size,
            },
            capabilities=capabilities,
        )
        collate = getattr(dataset, "collate_fn", None) or collate
        return DatasetSample(_collate(values, collate), observation, state)

    def restore_state(self, dataset, state):
        if state is None:
            if bool(getattr(dataset, "MODEL_MAVEN_STATELESS", False)):
                return "stateless"
            return "reset-required"
        hook = getattr(dataset, "load_state_dict", None)
        if not callable(hook):
            raise UserError("DATASET_STATE_RESTORE_UNSUPPORTED", "数据集没有 load_state_dict()，不能无损恢复游标。")
        if not isinstance(state, dict):
            raise UserError("DATASET_STATE_INVALID", "待恢复的数据集状态必须是字典。")
        try:
            hook(state)
        except Exception as error:
            raise UserError("DATASET_STATE_RESTORE_FAILED", f"恢复数据集状态失败：{error}") from error
        return "restored"

    def _sample_iterable(self, dataset, count, seed, split, fingerprint, collate, scan_limit, seed_forwarded, factory_seed, split_handling):
        values = []
        if scan_limit is not None and getattr(dataset, "batch_size", None) is not None:
            raise UserError("DATA_SCAN_LIMIT_UNSUPPORTED", "扫描采样不支持已分批的 DataLoader；请提供逐样本可迭代 Dataset。")
        heap = []
        examined = 0
        exhausted = False
        try:
            iterator = iter(dataset)
            for index in range(scan_limit if scan_limit is not None else count):
                try:
                    value = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                examined += 1
                if scan_limit is None:
                    values.append(value)
                else:
                    rank = int.from_bytes(hashlib.sha256(f"model-maven:stream-v1:{seed}:{index}".encode("utf-8")).digest(), "big")
                    entry = (-rank, -index, value)
                    if len(heap) < count:
                        heapq.heappush(heap, entry)
                    elif entry[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, entry)
        except Exception as error:
            raise UserError("DATASET_SAMPLE_FAILED", f"读取有界数据流样本失败：{error}") from error
        selected_offsets = None
        if scan_limit is not None:
            selected = sorted(((-negative_index, value) for _, negative_index, value in heap), key=lambda item: item[0])
            selected_offsets = [index for index, _ in selected]
            values = [value for _, value in selected]
        if not values:
            raise UserError("DATASET_EMPTY", "选中的迭代数据源没有产生任何样本。")
        state = _state_snapshot(dataset)
        can_save = state is not None
        can_load = callable(getattr(dataset, "load_state_dict", None))
        hooks = can_save and can_load
        stateless = bool(getattr(dataset, "MODEL_MAVEN_STATELESS", False))
        capabilities = DataCapabilities(
            inspect_batch=bool(values),
            forward_probe=bool(values),
            # User factories can return shared objects or consume shared streams.
            independent_sampling=None,
            stable_sample_ids=False,
            checkpoint_save=can_save,
            checkpoint_load=can_load,
            resume_cursor=hooks or stateless,
            reason=" ".join(
                reason for reason in (
                    "无法确认数据实例或外部状态与训练迭代器相互独立。",
                    "工厂即使接收 seed，也不能证明数据流顺序或内容可复现。",
                    None if hooks or stateless else "没有稳定样本 ID 或游标恢复钩子，不能声称代表性覆盖或无损 resume。",
                ) if reason
            ) or None,
        )
        observation = DataObservation(
            source=self.source,
            adapter=self.name,
            adapter_version=self.version,
            split=split,
            dataset_fingerprint=fingerprint,
            algorithm=ITERABLE_WINDOW_SAMPLE_ALGORITHM if scan_limit is not None else ITERABLE_SAMPLE_ALGORITHM,
            seed=seed,
            requested_samples=count,
            actual_samples=len(values),
            sample_ids=(),
            stable_sample_ids=False,
            coverage={
                "kind": "bounded-window-sample" if scan_limit is not None else "bounded-prefix",
                "split_handling": split_handling or ("unverified" if split is not None else "not_requested"),
                "complete": exhausted and examined <= count if scan_limit is not None else False,
                "stream_exhausted": exhausted if scan_limit is not None else None,
                "representative": False,
                "deterministic_selection": False,
                "deterministic_given_same_prefix": True,
                "seed_forwarded_to_factory": seed_forwarded,
                "factory_seed": factory_seed,
                "examined_items": examined,
                "scan_limit": scan_limit,
                "selected_offsets": selected_offsets,
                "cursor_state": "available" if hooks else "stateless" if stateless else "unsupported",
            },
            capabilities=capabilities,
        )
        # DataLoader-like sources already yield batches; collating those values
        # again would add a spurious nesting/batch dimension.
        if hasattr(dataset, "batch_size"):
            collate = None
        return DatasetSample(_collate(values, collate), observation, state)


def _accepts_keyword(factory, name):
    import inspect

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get(name)
    return (parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY
    )) or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _validate_sampling_request(count, seed):
    if type(count) is not int or count < 1:
        raise UserError("DATA_SAMPLE_LIMIT_INVALID", "采样数量必须是正整数。")
    if type(seed) is not int:
        raise UserError("DATA_SAMPLE_SEED_INVALID", "采样种子必须是整数。")


def _validate_fingerprint(fingerprint):
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or not fingerprint.strip()
    ):
        raise UserError("DATA_FINGERPRINT_INVALID", "数据指纹必须是非空字符串。")


def _sample_ids(dataset, indices, fingerprint):
    stable_id = getattr(dataset, "sample_id", None)
    if not callable(stable_id) or not isinstance(fingerprint, str) or not fingerprint:
        return [f"index:{index}" for index in indices], False
    try:
        values = [stable_id(index) for index in indices]
    except Exception as error:
        raise UserError("DATASET_SAMPLE_ID_FAILED", f"生成稳定样本 ID 失败：{error}") from error
    if any(
        not isinstance(value, (str, int)) or isinstance(value, bool)
        for value in values
    ):
        raise UserError("DATASET_SAMPLE_ID_INVALID", "sample_id(index) 必须返回字符串或整数。")
    if len({str(value) for value in values}) != len(values):
        raise UserError("DATASET_SAMPLE_ID_DUPLICATE", "sample_id(index) 必须在所选样本中唯一。")
    return [f"{fingerprint}:{value}" for value in values], True


def _state_snapshot(dataset):
    hook = getattr(dataset, "state_dict", None)
    if not callable(hook):
        return None
    try:
        state = hook()
    except Exception as error:
        raise UserError("DATASET_STATE_CAPTURE_FAILED", f"保存数据适配器状态失败：{error}") from error
    if not isinstance(state, dict):
        raise UserError("DATASET_STATE_INVALID", "数据集 state_dict() 必须返回字典。")
    return state


def _collate(values, collate):
    if collate is None:
        return values[0] if len(values) == 1 else values
    try:
        return collate(values)
    except Exception as error:
        raise UserError("DATASET_COLLATE_FAILED", f"数据批次整理失败：{error}") from error


def _validate_scan_limit(count, scan_limit):
    if scan_limit is not None and (type(scan_limit) is not int or not count <= scan_limit <= 100_000):
        raise UserError("DATA_SCAN_LIMIT_INVALID", "流式扫描上限必须不小于采样数且不超过 100000。")
