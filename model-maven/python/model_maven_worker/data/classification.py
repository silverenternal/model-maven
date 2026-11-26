"""Index local class-folder datasets without decoding every image first."""

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from ..protocol import UserError


INDEX_ALGORITHM = "class-folder-content-v1"
DEFAULT_IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")
EXCLUSION_PROTOCOL_PATH = Path(".model-maven/data-exclusions.json")


def _fingerprint(classes, samples, extensions, excluded_files=()):
    manifest = {
        "algorithm": INDEX_ALGORITHM,
        "classes": classes,
        "extensions": sorted(extensions),
        "samples": [
            {key: sample[key] for key in ("path", "label", "size_bytes", "sha256")}
            for sample in sorted(samples, key=lambda item: item["path"])
        ],
        "excluded_files": sorted(excluded_files),
    }
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scan_class_folder(root, extensions=DEFAULT_IMAGE_EXTENSIONS, *, digest_cache=None):
    """Index `<root>/<class>/*` paths and contents without image decoding."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise UserError("CLASS_DATASET_ROOT_INVALID", f"分类数据目录不存在或不是目录：{root}")
    normalized_extensions = tuple(sorted({
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
        if isinstance(extension, str) and extension.strip()
    }))
    if not normalized_extensions:
        raise UserError("CLASS_IMAGE_EXTENSIONS_INVALID", "至少指定一个有效图像扩展名。")
    try:
        class_dirs = sorted(
            (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".") and not path.is_symlink()),
            key=lambda path: path.name.casefold(),
        )
    except OSError as error:
        raise UserError("CLASS_DATASET_SCAN_FAILED", f"无法读取分类数据目录：{error}") from error
    if not class_dirs:
        raise UserError("CLASS_DATASET_EMPTY", "分类数据目录下没有类别子目录。", "每个类别放入独立目录；原有文件不会被移动或删除。")
    names = [path.name for path in class_dirs]
    if len({name.casefold() for name in names}) != len(names):
        raise UserError("CLASS_NAME_COLLISION", "类别目录在大小写不敏感文件系统上重名。")
    classes = sorted(names, key=lambda name: name.casefold())
    class_to_idx = {label: index for index, label in enumerate(classes)}
    samples = []
    next_cache, reused = {}, 0
    for label in classes:
        class_root = root / label
        for current, dirs, files in os.walk(class_root, followlinks=False):
            dirs[:] = sorted(name for name in dirs if not (Path(current) / name).is_symlink())
            for name in sorted(files):
                path = Path(current) / name
                if path.suffix.lower() not in normalized_extensions or path.is_symlink():
                    continue
                try:
                    before = path.stat(follow_symlinks=False)
                    if not stat.S_ISREG(before.st_mode):
                        continue
                    relative = path.relative_to(root).as_posix()
                    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    cached = digest_cache.get(relative) if isinstance(digest_cache, dict) else None
                    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == identity:
                        digest_value = cached[1]
                        reused += 1
                    else:
                        digest = hashlib.sha256()
                        with path.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(chunk)
                        digest_value = digest.hexdigest()
                    after = path.stat(follow_symlinks=False)
                except OSError as error:
                    raise UserError(
                        "CLASS_DATASET_FILE_UNREADABLE",
                        f"无法读取分类样本 {path.relative_to(root).as_posix()}：{error}",
                        "修复文件权限或显式排除该相对路径后重新建立数据协议。",
                        {"path": path.relative_to(root).as_posix()},
                    ) from error
                if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise UserError(
                        "DATASET_CHANGED_DURING_SCAN",
                        f"扫描期间样本发生变化：{path.relative_to(root).as_posix()}",
                        "等待文件写入完成后重新扫描。",
                    )
                if isinstance(digest_cache, dict) and len(next_cache) < 100_000:
                    next_cache[relative] = (identity, digest_value)
                samples.append({
                    "path": relative,
                    "label": label,
                    "class_index": class_to_idx[label],
                    "size_bytes": after.st_size,
                    "sha256": digest_value,
                    "sample_id": relative,
                })
    if not samples:
        raise UserError("CLASS_DATASET_EMPTY", "分类目录中没有支持的图像文件。", f"当前支持扩展名：{', '.join(normalized_extensions)}。")
    samples.sort(key=lambda item: (item["label"].casefold(), item["path"]))
    class_counts = {label: 0 for label in classes}
    for sample in samples:
        class_counts[sample["label"]] += 1
    fingerprint = _fingerprint(classes, samples, normalized_extensions)
    if isinstance(digest_cache, dict):
        digest_cache.clear()
        digest_cache.update(next_cache)
    return {
        "algorithm": INDEX_ALGORITHM,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "extensions": list(normalized_extensions),
        "samples": samples,
        "class_counts": class_counts,
        "sample_count": len(samples),
        "dataset_fingerprint": fingerprint,
        "indexing": {
            "decoded_images": False,
            "content_hashed": True,
            "content_hashes_reused": reused,
            "root_path_persisted": False,
        },
        "excluded_files": [],
    }


def verify_sample_images(root, index, samples):
    """Decode/verify only the selected paths; report all failures without substituting."""
    try:
        from PIL import Image
    except ImportError as error:
        raise UserError("IMAGE_DECODER_UNAVAILABLE", "当前 Python 环境没有 Pillow，无法验证图像解码。", "安装 Pillow，或先使用不解码的目录索引与结构检查。") from error
    root = Path(root).expanduser().resolve()
    failures = []
    for sample in samples:
        relative = sample["path"]
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != sample.get("sha256"):
                failures.append({"path": relative, "label": sample["label"], "error": "content changed since indexing"})
                continue
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            failures.append({"path": relative, "label": sample["label"], "error": f"{type(error).__name__}: {error}"[:300]})
    if failures:
        raise UserError(
            "CLASS_SAMPLES_DAMAGED",
            f"选中的分类样本有 {len(failures)} 个无法解码；没有自动替换样本。",
            "修复原文件，或明确排除下列相对路径并重新分层采样；排除只更新索引，不删除磁盘文件。",
            {"damaged_files": failures, "dataset_fingerprint": index["dataset_fingerprint"]},
        )
    return {"checked": len(samples), "valid": len(samples), "dataset_fingerprint": index["dataset_fingerprint"]}


def exclude_files(index, relative_paths):
    """Create a new explicit exclusion protocol; never remove files from disk."""
    requested = sorted(set(relative_paths))
    known = {sample["path"] for sample in index["samples"]}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise UserError("CLASS_EXCLUSION_UNKNOWN", "排除清单中包含当前索引不存在的文件。", "重新扫描目录，再选择有效的相对路径。", {"unknown_paths": unknown})
    all_excluded = sorted(set(index.get("excluded_files", [])) | set(requested))
    excluded_set = set(all_excluded)
    samples = [sample for sample in index["samples"] if sample["path"] not in excluded_set]
    if not samples:
        raise UserError("CLASS_EXCLUSION_EMPTY", "排除后没有剩余分类样本。", "减少排除项或修复损坏文件；不会自动删除类别或补充其他数据。")
    classes = list(index["classes"])
    result = {
        **index,
        "samples": samples,
        "sample_count": len(samples),
        "class_counts": {label: sum(sample["label"] == label for sample in samples) for label in classes},
        "excluded_files": all_excluded,
    }
    result["dataset_fingerprint"] = _fingerprint(classes, samples, index["extensions"], all_excluded)
    result["indexing"] = {**index["indexing"], "excluded_without_deletion": True}
    return result


def load_project_exclusions(project_root, dataset_root, index):
    """Apply a project-owned exclusion protocol only to its exact data root/index."""
    project_root = Path(project_root).expanduser().resolve()
    metadata = project_root / ".model-maven"
    protocol_path = metadata / "data-exclusions.json"
    try:
        metadata_stat = metadata.lstat()
    except FileNotFoundError:
        return index
    if stat.S_ISLNK(metadata_stat.st_mode) or not stat.S_ISDIR(metadata_stat.st_mode):
        raise UserError("DATA_EXCLUSION_PROTOCOL_INVALID", "项目元数据目录不是可信普通目录。")
    try:
        protocol_stat = protocol_path.lstat()
    except FileNotFoundError:
        return index
    if stat.S_ISLNK(protocol_stat.st_mode) or not stat.S_ISREG(protocol_stat.st_mode) or protocol_stat.st_size > 1024 * 1024:
        raise UserError("DATA_EXCLUSION_PROTOCOL_INVALID", "数据排除协议不是安全的小型普通文件。")
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("DATA_EXCLUSION_PROTOCOL_INVALID", f"无法读取数据排除协议：{error}") from error
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise UserError("DATA_EXCLUSION_PROTOCOL_INVALID", "数据排除协议版本无效。")
    if protocol.get("dataset_root") != str(Path(dataset_root).expanduser().resolve()):
        return index
    if protocol.get("source_dataset_fingerprint") != index.get("dataset_fingerprint"):
        raise UserError(
            "DATASET_PROTOCOL_INVALIDATED",
            "数据内容已变化，已保存的排除协议不能继续应用。",
            "重新验证当前数据并明确更新排除清单。",
            {"previous_fingerprint": protocol.get("source_dataset_fingerprint"), "current_fingerprint": index.get("dataset_fingerprint")},
        )
    excluded = protocol.get("excluded_files")
    if not isinstance(excluded, list) or any(not isinstance(path, str) for path in excluded):
        raise UserError("DATA_EXCLUSION_PROTOCOL_INVALID", "数据排除协议的文件清单格式无效。")
    result = exclude_files(index, excluded)
    if result["dataset_fingerprint"] != protocol.get("dataset_fingerprint"):
        raise UserError("DATA_EXCLUSION_PROTOCOL_INVALID", "排除协议指纹校验失败。")
    return result


def save_project_exclusions(project_root, dataset_root, source_index, result_index):
    """Atomically persist an explicit filtered protocol in the Model-Maven project."""
    project_root = Path(project_root).expanduser().resolve(strict=True)
    metadata = project_root / ".model-maven"
    try:
        metadata_stat = metadata.lstat()
    except OSError as error:
        raise UserError("DATA_EXCLUSION_PROTOCOL_WRITE_FAILED", f"无法读取项目元数据目录：{error}") from error
    if stat.S_ISLNK(metadata_stat.st_mode) or not stat.S_ISDIR(metadata_stat.st_mode):
        raise UserError("DATA_EXCLUSION_PROTOCOL_WRITE_FAILED", "项目元数据目录不是可信普通目录。")
    target = metadata / "data-exclusions.json"
    try:
        current = target.lstat()
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
            raise UserError("DATA_EXCLUSION_PROTOCOL_WRITE_FAILED", "既有数据排除协议不是普通文件；拒绝覆盖。")
    except FileNotFoundError:
        pass
    protocol = {
        "schema_version": 1,
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "source_dataset_fingerprint": source_index["dataset_fingerprint"],
        "dataset_fingerprint": result_index["dataset_fingerprint"],
        "excluded_files": result_index["excluded_files"],
    }
    encoded = (json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".data-exclusions-", suffix=".tmp", dir=metadata)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(metadata, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return protocol


def clear_project_exclusions(project_root):
    """Remove only the Model-Maven exclusion protocol after the caller re-verifies data."""
    project_root = Path(project_root).expanduser().resolve(strict=True)
    metadata = project_root / ".model-maven"
    try:
        metadata_stat = metadata.lstat()
    except OSError as error:
        raise UserError("DATA_EXCLUSION_PROTOCOL_WRITE_FAILED", f"无法读取项目元数据目录：{error}") from error
    if stat.S_ISLNK(metadata_stat.st_mode) or not stat.S_ISDIR(metadata_stat.st_mode):
        raise UserError("DATA_EXCLUSION_PROTOCOL_WRITE_FAILED", "项目元数据目录不是可信普通目录。")
    target = metadata / "data-exclusions.json"
    try:
        current = target.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise UserError("DATA_EXCLUSION_PROTOCOL_WRITE_FAILED", "既有数据排除协议不是普通文件；拒绝移除。")
    target.unlink()
    directory_fd = os.open(metadata, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return True


def compare_index_content(previous, current):
    """Return file-level repeat invalidation evidence for a changed content index."""
    before = {sample["path"]: sample for sample in previous.get("samples", [])}
    after = {sample["path"]: sample for sample in current.get("samples", [])}
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path].get("sha256") != after[path].get("sha256"))
    class_changed = previous.get("classes") != current.get("classes")
    valid = not added and not removed and not changed and not class_changed and previous.get("dataset_fingerprint") == current.get("dataset_fingerprint")
    return {
        "compatible": valid,
        "previous_fingerprint": previous.get("dataset_fingerprint"),
        "current_fingerprint": current.get("dataset_fingerprint"),
        "added_files": added,
        "removed_files": removed,
        "changed_files": changed,
        "class_mapping_changed": class_changed,
        "action": "reuse_original_protocol" if valid else "explicit_resample_required",
    }


def require_repeat_compatible(previous, current):
    result = compare_index_content(previous, current)
    if not result["compatible"]:
        raise UserError(
            "DATASET_PROTOCOL_INVALIDATED",
            "分类数据内容、类别或文件清单已变化，旧样本协议不能继续复用。",
            "查看 data.repeat_check 中的差异；确认后显式重新分层采样以生成新的数据/切分指纹。",
            result,
        )
    return result
