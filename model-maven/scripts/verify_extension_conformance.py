#!/usr/bin/env python3
"""Run the public-CLI conformance path against an independently installed Provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="验收独立 Provider 的声明能力、fixture 断言和验证后编辑门禁。"
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=repository / "target/debug/m-model",
        help="已构建的 m-model 二进制",
    )
    parser.add_argument("--python", default=sys.executable, help="包含 PyTorch 的 Python 解释器")
    args = parser.parse_args()
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        parser.error(f"找不到 CLI 二进制：{binary}；先运行 cargo build。")

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/verify_provider_extension.py"),
            "--binary",
            str(binary),
            "--python",
            args.python,
        ],
        cwd=repository,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stdout, end="", file=sys.stdout)
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)
    try:
        evidence = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Provider 验收器未返回 JSON 证据。\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from error
    if evidence.get("status") != "passed":
        raise RuntimeError(f"Provider 验收器未通过：{json.dumps(evidence, ensure_ascii=False)}")
    if evidence.get("fixture_coverage") != "complete_for_fixture_expectations":
        raise RuntimeError("独立 Provider fixture 没有达到显式断言覆盖范围。")
    if evidence.get("variant_undo_redo") != "passed":
        raise RuntimeError("独立 Provider 的变体撤销/重做没有通过。")

    print(json.dumps({
        "status": "passed",
        "verification": "independent-provider-cli-conformance",
        "fixture_coverage": evidence["fixture_coverage"],
        "variant_undo_redo": evidence["variant_undo_redo"],
        "portable_export_restore": evidence.get("offline_export_restore"),
        "tampered_bundle_rejected": evidence.get("tampered_bundle_rejected"),
        "limits": [
            "该流程验证一个独立 Provider wheel 和声明的 fixture/payload。",
            "不能据此推断未声明能力、任意模型输入、状态迁移或通用持久化正确性。",
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Extension conformance acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
