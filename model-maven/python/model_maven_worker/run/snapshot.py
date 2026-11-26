"""Create a private, immutable copy of project Python sources for one run."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from ..protocol import UserError
from .file_digest import sha256_file


_IGNORED_PARTS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}


def snapshot_project_code(project_root, snapshot_dir, required_source_manifest=None, import_roots=None):
    """Copy project-local Python code and pinned component descriptors.

    Data files and installed Python distributions remain owned by the project and
    selected interpreter. Python files are frozen as a project tree; non-Python
    files are copied only when a saved variant pins a registered component
    descriptor in ``.model-maven/components``.
    """
    root = Path(project_root).resolve()
    target_root = Path(snapshot_dir).resolve()
    if not root.is_dir():
        raise UserError("PROJECT_ROOT_MISSING", "项目源码目录不存在，无法固定本次运行输入。")
    if target_root == root:
        raise UserError("SOURCE_SNAPSHOT_PATH_INVALID", "源码快照目录不能覆盖项目目录。")
    if root in target_root.parents:
        relative_target = target_root.relative_to(root)
        if relative_target.parts[:2] != (".model-maven", "runs"):
            raise UserError("SOURCE_SNAPSHOT_PATH_INVALID", "项目内的源码快照只能写入 .model-maven/runs。")
    if target_root.is_symlink():
        raise UserError("SOURCE_SNAPSHOT_PATH_INVALID", "源码快照目录不能是符号链接。")
    target_root.mkdir(parents=True, exist_ok=True)
    destination = target_root / "code"
    if destination.exists() or destination.is_symlink():
        raise UserError("SOURCE_SNAPSHOT_EXISTS", "本次运行的源码快照已存在；拒绝覆盖运行输入。")

    temporary = Path(tempfile.mkdtemp(prefix=".code-snapshot-", dir=target_root))
    code = temporary / "code"
    code.mkdir()
    manifest = {}
    try:
        selected_roots = []
        if import_roots is not None and not isinstance(import_roots, list):
            raise UserError("IMPORT_ROOTS_INVALID", "项目模块搜索目录必须是相对路径数组。")
        if len(import_roots or []) > 32:
            raise UserError("IMPORT_ROOTS_INVALID", "项目模块搜索目录最多 32 项。")
        for value in import_roots or []:
            if (not isinstance(value, str) or not value or len(value) > 1024
                    or "\\" in value or "\x00" in value
                    or any(part in ("", ".", "..") for part in value.split("/"))):
                raise UserError("IMPORT_ROOTS_INVALID", "模块搜索目录必须是项目内的有效相对路径。")
            relative = Path(value)
            if (relative.is_absolute() or ".git" in relative.parts or "__pycache__" in relative.parts
                    or relative.parts[:2] == (".model-maven", "runs")):
                raise UserError("IMPORT_ROOTS_INVALID", "模块搜索目录不能包含版本控制、缓存或运行记录目录。")
            current = root
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise UserError("IMPORT_ROOTS_INVALID", f"模块搜索目录不能经过符号链接：{value}")
            if not current.is_dir() or current.resolve() != current:
                raise UserError("IMPORT_ROOT_UNAVAILABLE", f"模块搜索目录不可用：{value}")
            selected_roots.append(relative)
            # Preserve empty selected directories too; importing from one should
            # fail normally rather than because the snapshot omitted the root.
            (code / relative).mkdir(parents=True, exist_ok=True)

        def selected_path(relative):
            return any(relative == chosen or relative in chosen.parents or chosen in relative.parents
                       for chosen in selected_roots)

        interpreter_environment = Path(sys.prefix).resolve()
        excluded_roots = ({interpreter_environment}
                          if interpreter_environment != root and interpreter_environment.is_relative_to(root) else set())

        def include_directory(relative):
            candidate = root / relative
            if relative.parts[:2] == (".model-maven", "runs") or candidate.is_symlink():
                return False
            if relative.name in {".git", "__pycache__"}:
                return False
            if (relative.name in _IGNORED_PARTS or (candidate / "pyvenv.cfg").is_file()
                    or (relative.name == "target" and (candidate / ".rustc_info.json").is_file())):
                excluded_roots.add(candidate)
            if selected_path(relative):
                return True
            if candidate in excluded_roots or any(parent in excluded_roots for parent in candidate.parents):
                return False
            return True

        def raise_walk_error(error):
            raise error

        for current, directories, filenames in os.walk(root, followlinks=False, onerror=raise_walk_error):
            current_path = Path(current)
            relative_directory = current_path.relative_to(root)
            directories[:] = sorted(name for name in directories if include_directory(relative_directory / name))
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                source = current_path / filename
                if source.is_symlink() or not source.is_file():
                    continue
                relative = source.relative_to(root)
                # Ancestors of a selected root must be traversed, but their
                # sibling files are not part of the selected dependency tree.
                if not selected_path(relative) and any(parent in excluded_roots for parent in source.parents):
                    continue
                destination_file = code / relative
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                before = sha256_file(source)
                shutil.copyfile(source, destination_file, follow_symlinks=False)
                copied = sha256_file(destination_file)
                after = sha256_file(source)
                if before != copied or before != after:
                    raise UserError(
                        "SOURCE_SNAPSHOT_RACED",
                        f"固定运行源码时文件发生变化：{relative.as_posix()}。",
                        "保存代码后重新启动任务；本次任务未执行用户模型代码。",
                    )
                destination_file.chmod(0o400)
                manifest[relative.as_posix()] = copied

        required_source_manifest = required_source_manifest or {}
        if not isinstance(required_source_manifest, dict):
            raise UserError(
                "SOURCE_MANIFEST_INVALID",
                "变体的源码指纹清单必须是对象。",
            )
        for filename, expected in sorted(required_source_manifest.items()):
            if not isinstance(filename, str) or not isinstance(expected, str):
                raise UserError("SOURCE_MANIFEST_INVALID", "变体源码清单包含无效路径或指纹。")
            source = Path(filename)
            if not source.is_absolute():
                source = root / source
            try:
                relative = source.relative_to(root)
            except ValueError:
                # External sources remain bound to their original location and
                # are verified there; only project-local inputs enter the copy.
                continue
            if (
                relative.parts[:2] != (".model-maven", "components")
                or relative.suffix != ".json"
            ):
                continue
            if any(part in {"", ".", ".."} for part in relative.parts):
                raise UserError("SOURCE_SNAPSHOT_PATH_INVALID", f"变体源码路径无效：{filename}")
            current = root
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise UserError(
                        "SOURCE_SNAPSHOT_PATH_INVALID",
                        f"变体固定的组件清单不能经过符号链接：{relative.as_posix()}。",
                    )
            try:
                resolved_relative = source.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as error:
                raise UserError(
                    "SOURCE_CHANGED",
                    f"变体固定的组件清单缺失或越界：{relative.as_posix()}。",
                    "恢复原组件清单，或重新建立变体；本次训练未执行用户代码。",
                ) from error
            if resolved_relative != relative or not source.is_file():
                raise UserError(
                    "SOURCE_SNAPSHOT_PATH_INVALID",
                    f"变体固定的组件清单路径无效：{relative.as_posix()}。",
                )
            destination_file = code / relative
            if destination_file.exists():
                copied = sha256_file(destination_file)
            else:
                before = sha256_file(source)
                if before != expected:
                    raise UserError(
                        "SOURCE_CHANGED",
                        f"变体固定的组件清单已变化：{relative.as_posix()}。",
                        "保留原组件版本并重新建立变体；本次训练未执行用户代码。",
                    )
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination_file, follow_symlinks=False)
                copied = sha256_file(destination_file)
                after = sha256_file(source)
                if before != copied or before != after:
                    raise UserError(
                        "SOURCE_SNAPSHOT_RACED",
                        f"固定组件清单时文件发生变化：{relative.as_posix()}。",
                        "重新执行训练；本次任务未执行用户模型代码。",
                    )
                destination_file.chmod(0o400)
            if copied != expected:
                raise UserError(
                    "SOURCE_CHANGED",
                    f"变体固定的组件清单与源码快照不一致：{relative.as_posix()}。",
                    "保留原组件版本并重新建立变体；本次训练未执行用户代码。",
                )
            manifest[relative.as_posix()] = copied

        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        os.replace(code, destination)
        return {
            "status": "frozen",
            "root": str(destination),
            "files": len(manifest),
            "manifest": manifest,
            "sha256": digest,
            "scope": "project-python-and-pinned-component-descriptors",
            "external_dependencies_included": False,
        }
    except UserError:
        raise
    except OSError as error:
        raise UserError(
            "SOURCE_SNAPSHOT_FAILED",
            f"无法创建本次运行的源码快照：{error}",
            "检查运行目录权限和磁盘空间；本次任务未执行用户模型代码。",
        ) from error
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
