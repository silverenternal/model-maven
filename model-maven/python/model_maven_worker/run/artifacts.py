"""Safe declaration helpers for files owned by a run-scoped producer."""

from pathlib import Path, PurePosixPath

from ..protocol import UserError


def run_artifact_directory(checkpoint_dir):
    if not checkpoint_dir:
        return None
    checkpoints = Path(checkpoint_dir)
    if checkpoints.is_symlink() or not checkpoints.is_dir():
        raise UserError("RUN_ARTIFACT_PATH_INVALID", "运行 checkpoint 目录不是可信目录。")
    run_directory = checkpoints.parent
    artifact_directory = run_directory / "artifacts"
    if artifact_directory.is_symlink():
        raise UserError("RUN_ARTIFACT_PATH_INVALID", "运行产物目录不能是符号链接。")
    artifact_directory.mkdir(mode=0o700, exist_ok=True)
    if artifact_directory.is_symlink() or not artifact_directory.is_dir():
        raise UserError("RUN_ARTIFACT_PATH_INVALID", "运行产物目录不是可信目录。")
    return artifact_directory


def managed_artifact_declaration(
    artifact_directory,
    artifact_id,
    files,
    owner,
    references,
    consumes,
    complete,
    kind,
    request_run_id,
):
    if not isinstance(artifact_id, str) or not artifact_id.strip() or len(artifact_id) > 160:
        raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "产物 id 必须是 1..160 个可见字符。")
    if not isinstance(owner, str) or not owner.strip() or len(owner) > 160:
        raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "产物 owner 必须是 1..160 个可见字符。")
    if not isinstance(files, (list, tuple)) or not files:
        raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "产物必须声明至少一个文件。")
    if not isinstance(references, (list, tuple)) or not all(
        isinstance(value, str) and value.strip() for value in references
    ):
        raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "产物引用必须是非空字符串列表。")
    if not isinstance(consumes, (list, tuple)):
        raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "产物 consumes 必须是 typed artifact identity 列表。")
    typed_consumes = []
    for value in consumes:
        if not isinstance(value, dict) or set(value) != {"run_id", "artifact_id"}:
            raise UserError(
                "RUN_ARTIFACT_DECLARATION_INVALID",
                "每个 consumes 项必须包含 run_id 和 artifact_id。",
            )
        run_id = value["run_id"]
        consumed_artifact_id = value["artifact_id"]
        for label, item in (("run_id", run_id), ("artifact_id", consumed_artifact_id)):
            if (
                not isinstance(item, str)
                or not item.strip()
                or len(item) > 160
                or any(ord(character) < 32 for character in item)
            ):
                raise UserError(
                    "RUN_ARTIFACT_DECLARATION_INVALID",
                    f"consumes 的 {label} 必须是 1..160 个可见字符。",
                )
        if run_id == request_run_id:
            raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "运行不能消费自身产物。")
        typed_consumes.append({"run_id": run_id, "artifact_id": consumed_artifact_id})
    if type(complete) is not bool:
        raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "complete 必须是布尔值。")
    if kind not in {"checkpoint", "other"}:
        raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "kind 必须是 checkpoint 或 other。")

    declared_files = []
    for value in files:
        if not isinstance(value, (str, Path)):
            raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "产物文件必须使用 artifacts/ 下的相对路径。")
        path = PurePosixPath(str(value))
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "产物文件路径必须是 artifacts/ 下的普通相对路径。")
        target = artifact_directory.joinpath(*path.parts)
        cursor = artifact_directory
        for part in path.parts[:-1]:
            cursor = cursor / part
            if cursor.is_symlink() or not cursor.is_dir():
                raise UserError("RUN_ARTIFACT_PATH_INVALID", f"产物父目录不安全：{cursor}")
        if target.is_symlink() or not target.is_file():
            raise UserError("RUN_ARTIFACT_PATH_INVALID", f"产物文件不是普通文件：{target}")
        declared_files.append("artifacts/" + path.as_posix())
    if len(set(declared_files)) != len(declared_files):
        raise UserError("RUN_ARTIFACT_DECLARATION_INVALID", "产物文件不能重复登记。")
    return {
        "id": artifact_id,
        "owner": owner,
        "kind": kind,
        "files": declared_files,
        "references": sorted(set(references)),
        "consumes": sorted(
            { (item["run_id"], item["artifact_id"]): item for item in typed_consumes }.values(),
            key=lambda item: (item["run_id"], item["artifact_id"]),
        ),
        "complete": complete,
    }
