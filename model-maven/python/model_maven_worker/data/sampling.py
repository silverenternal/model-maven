"""Versioned deterministic classification split and stratified subset planning."""

import hashlib
import json

from ..protocol import UserError


SPLIT_ALGORITHM = "sha256-stratified-split-v1"
SUBSET_ALGORITHM = "sha256-stratified-subset-v1"


def _digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rank(seed, scope, value):
    return hashlib.sha256(f"model-maven:{scope}:v1:{seed}:{value}".encode("utf-8")).digest()


def _ratios(ratios):
    if not isinstance(ratios, dict) or not ratios:
        raise UserError("SPLIT_RATIOS_INVALID", "切分比例必须是 train/validation/test 比例对象。")
    unknown = set(ratios) - {"train", "validation", "test"}
    if unknown:
        raise UserError("SPLIT_RATIOS_INVALID", f"不支持切分名称：{', '.join(sorted(unknown))}。")
    values = {}
    for name in ("train", "validation", "test"):
        value = ratios.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise UserError("SPLIT_RATIOS_INVALID", f"{name} 比例必须是有限非负数。")
        values[name] = float(value)
    total = sum(values.values())
    if not 0.999999 <= total <= 1.000001:
        raise UserError("SPLIT_RATIOS_INVALID", f"切分比例总和必须为 1，当前为 {total:g}。")
    return values


def _allocate_split(count, ratios, seed, label):
    names = [name for name in ("train", "validation", "test") if ratios[name] > 0]
    raw = {name: count * ratios[name] for name in names}
    allocation = {name: int(raw[name]) for name in names}
    remaining = count - sum(allocation.values())
    order = sorted(
        names,
        key=lambda name: (-((raw[name] - allocation[name])), _rank(seed, f"split-tie:{label}", name)),
    )
    for name in order[:remaining]:
        allocation[name] += 1
    return {name: allocation.get(name, 0) for name in ("train", "validation", "test")}


def split_classification_index(index, ratios, seed=0):
    """Assign every indexed file once; never repair impossible per-class ratios."""
    if type(seed) is not int:
        raise UserError("SPLIT_SEED_INVALID", "切分种子必须是整数。")
    ratios = _ratios(ratios)
    by_label = {label: [] for label in index["classes"]}
    for sample in index["samples"]:
        by_label[sample["label"]].append(sample)
    splits = {name: [] for name in ("train", "validation", "test")}
    class_counts = {name: {label: 0 for label in index["classes"]} for name in splits}
    for label in index["classes"]:
        samples = sorted(by_label[label], key=lambda item: item["path"])
        samples.sort(key=lambda item: _rank(seed, f"split:{index['dataset_fingerprint']}:{label}", item["path"]))
        allocation = _allocate_split(len(samples), ratios, seed, label)
        cursor = 0
        for split_name in ("train", "validation", "test"):
            count = allocation[split_name]
            selected = samples[cursor:cursor + count]
            splits[split_name].extend(selected)
            class_counts[split_name][label] = len(selected)
            cursor += count
    for split_name in splits:
        splits[split_name].sort(key=lambda item: (item["label"], item["path"]))
    split_ids = {
        name: [sample["sample_id"] for sample in samples]
        for name, samples in splits.items()
    }
    fingerprint = _digest({
        "dataset_fingerprint": index["dataset_fingerprint"],
        "ratios": ratios,
        "seed": seed,
        "algorithm": SPLIT_ALGORITHM,
        "sample_ids": split_ids,
    })
    return {
        "algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "ratios": ratios,
        "classes": list(index["classes"]),
        "sample_count": len(index["samples"]),
        "dataset_fingerprint": index["dataset_fingerprint"],
        "split_fingerprint": fingerprint,
        "splits": splits,
        "sample_ids": split_ids,
        "class_counts": class_counts,
        "unavailable_splits": {
            name: [label for label in index["classes"] if ratios[name] > 0 and class_counts[name][label] == 0]
            for name in splits
        },
        "coverage": {
            "complete_partition": sum(len(items) for items in splits.values()) == len(index["samples"]),
            "disjoint": True,
            "per_class_ratios_are_exact": False,
            "note": "小类别不够支持所有比例时保留实际计数；不会从 validation/test 借样本补 train。",
        },
    }


def _allocate_budget(capacities, budget, seed):
    active = sorted(label for label, count in capacities.items() if count > 0)
    allocation = {label: 0 for label in capacities}
    if budget <= 0 or not active:
        return allocation
    if budget < len(active):
        chosen = sorted(active, key=lambda label: (_rank(seed, "subset-class-coverage", label), label))[:budget]
        for label in chosen:
            allocation[label] = 1
        return allocation
    for label in active:
        allocation[label] = 1
    remaining = min(budget - len(active), sum(capacities[label] - 1 for label in active))
    while remaining:
        eligible = [label for label in active if allocation[label] < capacities[label]]
        capacity = sum(capacities[label] - allocation[label] for label in eligible)
        if capacity <= 0:
            break
        quotas = {label: remaining * (capacities[label] - allocation[label]) / capacity for label in eligible}
        additions = {label: min(capacities[label] - allocation[label], int(quotas[label])) for label in eligible}
        added = sum(additions.values())
        for label, count in additions.items():
            allocation[label] += count
        remaining -= added
        if not remaining:
            break
        eligible = [label for label in eligible if allocation[label] < capacities[label]]
        if not eligible:
            break
        label = sorted(
            eligible,
            key=lambda item: (-(quotas.get(item, 0) % 1), _rank(seed, "subset-remainder", item)),
        )[0]
        allocation[label] += 1
        remaining -= 1
    return allocation


def stratified_sample(index, split, budget, seed=0):
    """Select a deterministic no-replacement subset and report uncovered labels."""
    if split not in {"train", "validation", "test"}:
        raise UserError("DATA_SPLIT_INVALID", "split 必须是 train、validation 或 test。")
    if type(budget) is not int or budget < 1:
        raise UserError("DATA_SAMPLE_LIMIT_INVALID", "采样预算必须是正整数。")
    if type(seed) is not int:
        raise UserError("DATA_SAMPLE_SEED_INVALID", "采样种子必须是整数。")
    samples = index["splits"].get(split)
    if samples is None:
        raise UserError("DATA_SPLIT_MISSING", f"索引中没有 {split!r} 切分。")
    if not samples:
        raise UserError(
            "DATA_SPLIT_EMPTY",
            f"{split} 切分没有样本；不会从其他切分借样本补足。",
            "修正明确的数据切分比例或类别文件，再重新建立切分协议。",
            {"split": split, "class_counts": index.get("class_counts", {}).get(split, {})},
        )
    by_label = {label: [] for label in index["classes"]}
    for sample in samples:
        by_label[sample["label"]].append(sample)
    capacities = {label: len(items) for label, items in by_label.items()}
    actual_budget = min(budget, len(samples))
    allocation = _allocate_budget(capacities, actual_budget, seed)
    selected = []
    for label in index["classes"]:
        candidates = sorted(by_label[label], key=lambda item: item["path"])
        candidates.sort(key=lambda item: _rank(seed, f"sample:{index['split_fingerprint']}:{split}:{label}", item["path"]))
        selected.extend(candidates[:allocation[label]])
    selected.sort(key=lambda item: (item["label"], item["path"]))
    class_counts = {label: allocation[label] for label in index["classes"]}
    covered = [label for label in index["classes"] if class_counts[label] > 0]
    uncovered = [label for label in index["classes"] if capacities[label] > 0 and class_counts[label] == 0]
    sample_ids = [sample["sample_id"] for sample in selected]
    protocol = {
        "dataset_fingerprint": index["dataset_fingerprint"],
        "split_fingerprint": index["split_fingerprint"],
        "split": split,
        "algorithm": SUBSET_ALGORITHM,
        "seed": seed,
        "requested_budget": budget,
        "actual_count": len(selected),
        "sample_ids": sample_ids,
        "class_counts": class_counts,
    }
    return {
        **protocol,
        "protocol_fingerprint": _digest(protocol),
        "selected": selected,
        "covered_classes": covered,
        "uncovered_classes": uncovered,
        "coverage": {
            "class_count": len(covered),
            "available_class_count": sum(value > 0 for value in capacities.values()),
            "complete_class_coverage": not uncovered,
            "budget_exhausted": len(selected) < budget,
            "budget_was_not_expanded": len(selected) <= budget,
        },
    }
