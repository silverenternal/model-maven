"""Create a private, immutable copy of project Python sources for one run."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from ..protocol import UserError


_IGNORED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "target",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}


def snapshot_project_code(project_root, snapshot_dir, required_source_manifest=None):
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
        for current, directories, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            relative_directory = current_path.relative_to(root)
            directories[:] = sorted(
                name
                for name in directories
                if name not in _IGNORED_PARTS
                and not (relative_directory / name).parts[-1].startswith(".tmp")
                and not (current_path / name).is_symlink()
                and not (relative_directory / name).parts[:2] == (".model-maven", "runs")
            )
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                source = current_path / filename
                if source.is_symlink() or not source.is_file():
                    continue
                relative = source.relative_to(root)
                destination_file = code / relative
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                before = hashlib.sha256(source.read_bytes()).hexdigest()
                shutil.copyfile(source, destination_file, follow_symlinks=False)
                copied = hashlib.sha256(destination_file.read_bytes()).hexdigest()
                after = hashlib.sha256(source.read_bytes()).hexdigest()
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
                copied = hashlib.sha256(destination_file.read_bytes()).hexdigest()
            else:
                before = hashlib.sha256(source.read_bytes()).hexdigest()
                if before != expected:
                    raise UserError(
                        "SOURCE_CHANGED",
                        f"变体固定的组件清单已变化：{relative.as_posix()}。",
                        "保留原组件版本并重新建立变体；本次训练未执行用户代码。",
                    )
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination_file, follow_symlinks=False)
                copied = hashlib.sha256(destination_file.read_bytes()).hexdigest()
                after = hashlib.sha256(source.read_bytes()).hexdigest()
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

        if not manifest:
            raise UserError(
                "SOURCE_SNAPSHOT_EMPTY",
                "项目中没有可固定的 Python 源文件。",
                "将模型、数据和训练入口放在项目源码目录内后重试。",
            )
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        os.replace(code, destination)
        return {
            "status": "frozen",
            "root": str(destination),
            "files": len(manifest),
            "manifest": manifest,
            "sha256": digest,
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
