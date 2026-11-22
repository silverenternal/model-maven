#!/usr/bin/env python3
"""Run available QA-02 scenarios and write provenance-bound evidence records."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "complex-model-workflow": {
        "requirement_id": "QA02-complex-model-workflow",
        "title": "复杂模型、局部契约修复、训练恢复与导出恢复",
        "script": "scripts/verify_complex_acceptance.py",
        "covered": ["多模态/嵌套批次与共享调用观测", "三个具名 loss 与双优化器阶段训练", "局部契约失败、Layer 修复、variant 新进程重载", "训练恢复等价与包含 variant 的导出恢复"],
        "uncovered": ["真实用户任务和性能对照", "GPU 性能及 OOM 压力", "独立 LayerEditProvider/状态迁移通用保证"],
    },
    "external-loop": {
        "requirement_id": "QA02-external-loop",
        "title": "用户自定义训练循环及运行控制",
        "script": "scripts/verify_external_loop.py",
        "covered": ["现有 external-loop 集成验收脚本声明的运行、检查点和控制范围"],
        "uncovered": ["跨平台终端断连和监督进程外部强杀"],
    },
    "supervision-rules": {
        "requirement_id": "S-supervision",
        "title": "手动语义标记、目标绑定与显式监督训练",
        "script": "scripts/verify_supervision_acceptance.py",
        "covered": [
            "真实 Probe 后由用户提供语义角色/含义并创建 mark",
            "人工绑定 batch Tensor 路径和 axis，真实前向只读预览",
            "为 finite/range 人工指定样本轴，并通过 stable batch metadata.sample_ids 或 context.report_sample_ids 定位当前批次违规样本",
            "显式 --apply-supervision 后训练报告保存目标 shape/匹配/step 证据",
            "样本级 require 违规和目标轴不匹配均在任何完成步骤/更新前 fail closed",
            "缺少或不匹配的 sample IDs 报 unsupported；跨批次重复 ID 在 checkpoint/resume 后去重累计，报告定位批次、出现次数、去重样本数和违规样本数；checkpoint 只存 SHA-256 摘要，不存原始 ID",
            "external-loop 显式通过 supervised_step、context.backward 和登记的 optimizer hooks 接入 observe/require/penalty；正式训练恢复后覆盖继续累计",
            "项目内 custom-validator 支持结构化判定/JSON 摘要/可选样本违规 mask；custom-penalty 返回有限可微标量并在显式训练中进入命名 Loss",
            "自定义插件规则固定项目源码相对路径、大小和 SHA-256；源码变化后预览拒绝执行",
        ],
        "uncovered": [
            "用户声明的稳定 sample IDs 与原始记录的真实性校验及数据集覆盖率分母；标签值语义验证",
            "自定义 Python 插件不做 OS 沙箱、资源预算或副作用隔离；所有 optimizer/sampler 状态由外部循环自身通过 state hooks 管理",
        ],
    },
    "extension-package": {
        "requirement_id": "QA02-extension-package",
        "title": "独立 Layer 扩展端到端流程",
        "script": "scripts/verify_extension_package.py",
        "covered": ["现有独立 Layer wheel 验收脚本声明的发现、调用、训练、导出和恢复范围"],
        "uncovered": ["多平台 ABI、OS 库与私有源凭据恢复"],
    },
    "provider-extension": {
        "requirement_id": "QA02-provider-extension",
        "title": "独立 Layer Edit Provider 扩展流程",
        "script": "scripts/verify_provider_extension.py",
        "covered": ["现有独立 Provider 验收脚本声明的编辑、undo/redo、训练、导出和离线恢复范围"],
        "uncovered": ["Provider 通用状态迁移与尚未执行的破坏注入场景"],
    },
    "environment-offline-setup": {
        "requirement_id": "QA02-environment-offline-setup",
        "title": "离线环境建立与失败恢复",
        "script": "scripts/verify_environment_setup.py",
        "covered": ["现有环境建立验收脚本实际执行并报告的离线准备范围"],
        "uncovered": ["安装发行包的新机器恢复与复杂模型独立导出恢复"],
    },
    "environment-runtime-probe": {
        "requirement_id": "QA02-environment-runtime-probe",
        "title": "显式 Python 环境 CPU runtime smoke",
        "script": "scripts/verify_environment_runtime_probe.py",
        "covered": [
            "所选解释器中显式导入 PyTorch 后执行固定 CPU tensor arithmetic",
            "import 状态与 runtime-check 状态分别报告",
            "运行时诊断明确保持 auto_installed=false 和 auto_bound=false",
        ],
        "uncovered": [
            "CUDA/ROCm/GPU 与其他操作系统设备初始化",
            "任意模型、训练器或用户环境的运行正确性",
            "OS 沙箱；第三方包初始化仍以当前用户权限运行",
        ],
    },
    "project-artifact-reference-graph": {
        "requirement_id": "QA02-project-artifact-reference-graph",
        "title": "运行产物的跨运行消费者身份与保护式清理",
        "script": "scripts/verify_run_artifact_graph.py",
        "covered": ["真实训练运行声明 typed producer/consumer identities", "预览和应用阶段的引用图解析", "新增消费者阻止固定计划释放", "不完整或未知图 fail-closed"],
        "uncovered": ["variants/extensions 的非运行资源关系", "未登记消费者与用户自行复制的文件"],
    },
    "supervisor-disconnect": {
        "requirement_id": "QA02-supervisor-disconnect",
        "title": "Linux PTY 断开、监督进程死亡与检查点恢复",
        "script": "scripts/verify_supervisor_recovery.py",
        "covered": ["PTY HUP 只结束 viewer 并支持重新附加", "监督进程 SIGKILL 后状态为 unknown", "保留 checkpoint 可恢复到独立子运行"],
        "uncovered": ["真实 SSH/网络分区", "主机或内核故障"],
    },
    "release-install": {
        "requirement_id": "QA03-release-install",
        "title": "当前源码的抽取包安装与仓库外运行",
        "script": "scripts/verify_release_install.py",
        "no_binary_argument": True,
        "covered": [
            "cargo package 归档在临时前缀安装并在源码仓库外运行",
            "安装后的嵌入式 worker doctor/解释器缺失诊断",
            "安装后的嵌入式 worker 在无项目配置情况下搜索内置模型预设",
            "复杂模型实际训练恢复、导出恢复及评估样本预览历史",
            "可选 SDK wheel 构建、离线安装与公开入口导入",
        ],
        "uncovered": [
            "GitHub hosted CI/review artifact 工作流未远端运行",
            "非 Linux x86_64、GPU/加速器以及授权/第三方再分发审查",
        ],
    },
    "independent-export-restore": {
        "requirement_id": "QA02-independent-export-restore",
        "title": "复杂工程独立离线导出和恢复",
        "script": "scripts/verify_independent_export_restore.py",
        "covered": ["源项目删除后的独立恢复", "worker/source/bundle/checkpoint 哈希验证", "weights_only training state API"],
        "uncovered": ["任意用户训练 callable 的自动续训", "数据、OS/加速器库和未登记依赖"],
    },
    "provider-conformance": {
        "requirement_id": "QA02-provider-conformance",
        "title": "声明能力驱动的扩展符合性验证",
        "script": "scripts/verify_extension_conformance_suite.py",
        "covered": ["Provider capability/payload/source fixture 证据", "注册 transform 的 CLI pipeline/port/value 检查", "注册 dataset 的有限索引与样本检查", "注册 loss 的标量、有限 loss 与反向梯度检查", "注册 metric 的有限结果和 fixture 值检查", "注册 Layer 扩展 state_dict/load_state_dict 的 CLI roundtrip", "注册 Trainer 的真实进度、协作停止和 cooperative deadline 步骤边界检查", "正式训练超时后等待用户循环的下一步骤边界并验证 checkpoint 已提交", "Linux watchdog 在确认用户 batch 已运行后终止 worker 进程组及普通子进程", "独立 Provider wheel 的真实编辑与重载流程"],
        "uncovered": ["阻塞的第三方 Python/CUDA 单步无法被协作 deadline 中断；CLI watchdog 仍负责进程级强制退出", "OS 沙箱与任意输入正确性；脱离 worker 进程组的子进程不在 cleanup 保证内"],
    },
    "edit-persistence-and-migration": {
        "requirement_id": "QA02-edit-persistence-and-migration",
        "title": "Layer 编辑事务、重载、撤销与权重迁移",
        "script": "scripts/verify_edit_persistence.py",
        "covered": ["通用 Layer CRUD 的 variant 持久化、undo/redo 与失败事务", "共享 Provider 的跨进程 replay 与补偿", "父 checkpoint 到子运行的显式结构/权重迁移、alias 保持与 optimizer reset"],
        "uncovered": ["任意自定义 forward/control-flow 语义", "不支持的 tensor reshape/slice/merge/dtype 转换", "真实用户复杂模型配对观察"],
    },
}

BLOCKED = {
    "human-value-validation": ("目标用户真实任务价值与可用性", "需要真实目标用户配对观察；仓库测试、模拟数据和开发者自测不能替代。"),
}


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def revision() -> str:
    head = git("rev-parse", "HEAD")
    paths = sorted(
        path
        for path in set(git("diff", "--name-only", "HEAD").splitlines() + git("ls-files", "--others", "--exclude-standard").splitlines())
        if not path.startswith(("docs/acceptance/results/", "target/product-acceptance/"))
    )
    if not paths:
        return f"{head};worktree=clean"
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode("utf-8", errors="surrogateescape") + b"\0")
        path = ROOT / relative
        try:
            if path.is_symlink():
                digest.update(b"symlink\0" + os.readlink(path).encode("utf-8", errors="surrogateescape"))
            elif path.is_file():
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        except OSError as error:
            digest.update(f"unreadable:{type(error).__name__}".encode())
        digest.update(b"\0")
    return f"{head};worktree=dirty:{digest.hexdigest()}"


def product_version() -> str:
    try:
        plan = json.loads((ROOT / "todo.json").read_text(encoding="utf-8"))
        # QA-02 still exercises the scenarios from this version of the product specification.
        return str(plan.get("baseline", {}).get("legacy_product_version", "unspecified"))
    except (OSError, json.JSONDecodeError):
        return "unreadable"


def runtime(python: str, binary: Path) -> dict:
    info = {"os": platform.platform(), "architecture": platform.machine(), "rust": None, "python": None, "pytorch": None, "device": None, "notes": [f"binary={binary}"]}
    rust = subprocess.run(["rustc", "--version"], cwd=ROOT, text=True, capture_output=True, check=False)
    info["rust"] = rust.stdout.strip() if rust.returncode == 0 else None
    snippet = "import sys; print(sys.version.split()[0]);\ntry:\n import torch\n print(torch.__version__)\n print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')\nexcept Exception:\n print('unavailable')\n print('unavailable')"
    probe = subprocess.run([python, "-c", snippet], cwd=ROOT, text=True, capture_output=True, check=False)
    lines = probe.stdout.splitlines()
    if probe.returncode == 0 and lines:
        info["python"] = lines[0]
        info["pytorch"] = lines[1] if len(lines) > 1 else None
        info["device"] = lines[2] if len(lines) > 2 else None
    else:
        info["notes"].append(f"runtime probe failed: {probe.stderr.strip()}")
    return info


def write_record(directory: Path, case_id: str, requirement_id: str, status: str, env: dict, command: list[str], result: str, exit_code: int | None, evidence: list[str], covered: list[str], uncovered: list[str], fixture: str | None = "repository acceptance fixture") -> Path:
    record = {
        "schema_version": "1.0.0",
        "requirement_id": requirement_id,
        "status": status,
        "product_version": product_version(),
        "revision": revision(),
        "environment": env,
        "command": command,
        "exit_code": exit_code,
        "result": result,
        "evidence_paths": evidence,
        "covered_scope": covered,
        "uncovered_scope": uncovered,
        "observations": [],
        "participant_or_fixture": fixture,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{case_id}.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="列出可运行和被阻塞场景")
    parser.add_argument("--case", action="append", help="运行场景 ID；可重复或用 all")
    parser.add_argument("--python", default=sys.executable, help="被测 PyTorch Python")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/debug/m-model")
    parser.add_argument("--output-dir", type=Path, help="结果目录；默认 target/product-acceptance/<UTC 时间>")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({"runnable": {key: value["title"] for key, value in CASES.items()}, "blocked": {key: {"title": value[0], "reason": value[1]} for key, value in BLOCKED.items()}}, ensure_ascii=False, indent=2))
        return 0
    if not args.case:
        parser.error("请使用 --list 或指定至少一个 --case；不会默认运行所有长时场景。")
    unknown = set(args.case) - {"all"} - set(CASES) - set(BLOCKED)
    if unknown:
        parser.error(f"未知场景: {', '.join(sorted(unknown))}")
    selected = list(CASES) + list(BLOCKED) if "all" in args.case else args.case
    output = args.output_dir.resolve() if args.output_dir else ROOT / "target/product-acceptance" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = runtime(args.python, args.binary)
    failures = 0
    blocked_count = 0
    for case_id in selected:
        if case_id in BLOCKED:
            title, reason = BLOCKED[case_id]
            record = write_record(output, case_id, f"QA02-{case_id}", "skipped", env, [], reason, None, [], [], [reason], fixture=None)
            print(f"BLOCKED {case_id}: {reason} ({record})")
            blocked_count += 1
            continue
        case = CASES[case_id]
        binary = args.binary.resolve()
        if not binary.is_file():
            build = subprocess.run(["cargo", "build", "--locked"], cwd=ROOT, text=True, capture_output=True, check=False)
            (logs / f"{case_id}.build.stdout.log").write_text(build.stdout, encoding="utf-8")
            (logs / f"{case_id}.build.stderr.log").write_text(build.stderr, encoding="utf-8")
            if build.returncode != 0 or not binary.is_file():
                failures += 1
                record = write_record(output, case_id, case["requirement_id"], "failed", env, ["cargo", "build", "--locked"], f"CLI build failed with exit {build.returncode}", build.returncode, [str((logs / f"{case_id}.build.stdout.log").relative_to(ROOT)), str((logs / f"{case_id}.build.stderr.log").relative_to(ROOT))], [], case["covered"] + case["uncovered"])
                print(f"FAIL {case_id}: build failed ({record})")
                continue
        command = [args.python, str(ROOT / case["script"])]
        if case.get("no_binary_argument"):
            command.extend(["--python", args.python])
        else:
            command.extend(["--binary", str(binary), "--python", args.python])
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        elapsed = time.monotonic() - started
        stdout = logs / f"{case_id}.stdout.log"
        stderr = logs / f"{case_id}.stderr.log"
        stdout.write_text(result.stdout, encoding="utf-8")
        stderr.write_text(result.stderr, encoding="utf-8")
        status = "verified" if result.returncode == 0 else "failed"
        summary = f"exit={result.returncode}; elapsed_seconds={elapsed:.3f}"
        if result.stderr.strip():
            summary += f"; stderr_tail={result.stderr.strip()[-1000:]}"
        evidence = [str(stdout.relative_to(ROOT)), str(stderr.relative_to(ROOT))]
        covered = case["covered"] if result.returncode == 0 else []
        uncovered = case["uncovered"] + ([] if result.returncode == 0 else case["covered"])
        record = write_record(output, case_id, case["requirement_id"], status, env, command, summary, result.returncode, evidence, covered, uncovered)
        print(f"{status.upper()} {case_id}: {summary} ({record})")
        failures += result.returncode != 0
    if failures:
        return 1
    return 2 if blocked_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
