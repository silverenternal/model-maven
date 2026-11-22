"""User-confirmed class-folder verification and exclusion protocol operations."""

from pathlib import Path

from ..protocol import UserError
from .classification import (
    clear_project_exclusions,
    exclude_files,
    load_project_exclusions,
    save_project_exclusions,
    scan_class_folder,
    verify_sample_images,
)


def run_index_operation(request):
    root = Path(request.get("dataset_root", "")).expanduser().resolve()
    project_root = Path(request.get("project_root", "")).expanduser().resolve(strict=True)
    verify = request.get("verify") is True
    apply = request.get("apply_exclusions") is True
    clear = request.get("clear_exclusions") is True
    requested = request.get("exclude_files", [])
    if not isinstance(requested, list) or any(not isinstance(path, str) for path in requested):
        raise UserError("CLASS_EXCLUSION_INVALID", "排除项必须是相对路径字符串数组。")
    if (apply or clear) and not verify:
        raise UserError("CLASS_EXCLUSION_CONFIRMATION_REQUIRED", "写入排除协议前必须重新验证当前图像。")
    if apply and clear:
        raise UserError("CLASS_EXCLUSION_INVALID", "同一请求不能同时应用和清除排除协议。")
    source_index = scan_class_folder(root)
    index = source_index if clear else load_project_exclusions(project_root, root, source_index)
    damaged = []
    checked = 0
    if verify:
        try:
            result = verify_sample_images(root, index, index["samples"])
            checked = result["checked"]
        except UserError as error:
            if error.code != "CLASS_SAMPLES_DAMAGED":
                raise
            damaged = error.data.get("damaged_files", [])
            checked = index["sample_count"]
            if not apply:
                raise
    if clear:
        removed = clear_project_exclusions(project_root)
        return {
            **_summary(source_index, verified=checked, damaged=damaged),
            "exclusions_cleared": removed,
        }
    if not apply:
        return _summary(index, verified=checked, damaged=damaged)
    if not requested:
        raise UserError("CLASS_EXCLUSION_INVALID", "至少明确选择一个要排除的损坏文件。")
    damaged_paths = {item.get("path") for item in damaged if isinstance(item, dict)}
    unknown = sorted(set(requested) - damaged_paths)
    if unknown:
        raise UserError(
            "CLASS_EXCLUSION_NOT_DAMAGED",
            "排除清单必须来自本次验证确认损坏的文件。",
            "重新查看损坏清单，只选择其中的相对路径。",
            {"unknown_paths": unknown, "damaged_files": sorted(damaged_paths)},
        )
    filtered = exclude_files(index, requested)
    protocol = save_project_exclusions(project_root, root, source_index, filtered)
    return {
        **_summary(filtered, verified=checked, damaged=damaged),
        "source_dataset_fingerprint": source_index["dataset_fingerprint"],
        "exclusion_protocol": protocol,
        "excluded_without_deletion": True,
    }


def _summary(index, *, verified, damaged):
    return {
        "algorithm": index["algorithm"],
        "classes": index["classes"],
        "class_counts": index["class_counts"],
        "sample_count": index["sample_count"],
        "dataset_fingerprint": index["dataset_fingerprint"],
        "excluded_files": index["excluded_files"],
        "verified_images": verified,
        "damaged_files": damaged,
        "indexing": index["indexing"],
    }
