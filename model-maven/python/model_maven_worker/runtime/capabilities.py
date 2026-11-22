"""Runtime capability catalog returned by doctor and operation preflight."""

from model_maven_worker.protocol import HOST_FEATURES, SUPPORTED_REQUIRED_FEATURES


_TORCH_FEATURES = SUPPORTED_REQUIRED_FEATURES - HOST_FEATURES - {
    "extensions.static_discovery",
    "environment.setup",
    "environment.library_inventory",
    "environment.library_probe",
    "environment.wheelhouse",
    "edit.assembly_codegen",
}
_PROVIDERS = {
    "data.dataset_probe": "data-adapter",
    "data.preview": "data-preview",
    "environment.setup": "python-environment",
    "environment.library_inventory": "python-environment-metadata",
    "environment.library_probe": "python-environment-explicit-import-and-opt-in-cpu-smoke",
    "environment.wheelhouse": "python-environment",
    "edit.assembly_codegen": "python-worker-code-generator",
    "training.detached": "linux-supervisor",
    "training.evaluation": "project-evaluator",
    "training.supervision_preview": "semantic-rule-preview",
    "training.supervision_apply": "managed-step-supervision",
}


def _capability(identifier, available, *, provider, unavailable_reason=None):
    return {
        "id": identifier,
        "state": "supported" if available else "unsupported",
        "reason": None if available else unavailable_reason,
        "provider": provider,
    }


def capability_catalog(*, torch_available, platform):
    catalog = []
    for identifier in sorted(SUPPORTED_REQUIRED_FEATURES):
        if identifier in HOST_FEATURES:
            available = platform.startswith("linux") and torch_available
            reason = (
                "后台训练监督需要当前解释器可导入 PyTorch。"
                if not torch_available
                else "后台训练监督仅在已验收的 Linux 平台提供。"
            )
        elif identifier in _TORCH_FEATURES:
            available = torch_available
            reason = "当前解释器没有可导入的 PyTorch。"
        else:
            available = True
            reason = None
        catalog.append(
            _capability(
                identifier,
                available,
                provider=_PROVIDERS.get(identifier, "python-worker"),
                unavailable_reason=None if available else reason,
            )
        )
    return catalog


def doctor_report(*, python, python_version, torch_available, torch_version, platform, compatibility):
    implemented = [
        "native model inspect",
        "forward probe / per-invocation trace",
        "manually authored semantic marks bound to observed project module ports",
        "read-only preview of finite/range supervision rules on marked ports",
        "current-batch finite/range violation localization on a manually selected sample axis with user-declared stable IDs",
        "cross-batch sample and violation ID deduplication across checkpoint/resume for manually declared IDs (hash-only checkpoints, bounded summaries)",
        "explicit managed/manual-step require enforcement and weighted range penalties on manually marked ports",
        "versioned project-local supervision validator and differentiable-penalty callables with source pinning, structured summaries, and explicit penalty weights",
        "opt-in external-loop supervision through scoped forward capture, context.backward, and registered optimizer step guards",
        "declared model factory component injection seams",
        "schema-validated registered Layer injection with manifest and transitive dependency locks",
        "user loss backward/gradient check",
        "bounded managed-step AdamW probe",
        "bounded manual-step with named disjoint AdamW groups",
        "bounded external-loop probe with user-reported progress and finite-parameter checks",
        "foreground training trial records, source/runtime checked repeat, and runs list/show",
        "foreground managed/manual training with atomic checkpoints and explicit resume state",
        "foreground external-loop training with explicit state hooks/reset and cooperative stop",
        "formal training timeout requests a cooperative step-boundary checkpoint before force termination",
        "task-specific evaluation from verified training checkpoints with metric/provenance reporting",
        "verified training artifact export",
        "source/runtime checked trial repeat",
        "runs list/show with parent lineage",
        "CPU",
        "Sequential/ModuleList/ModuleDict variant CRUD",
        "ModuleList indexed CRUD with renumbering",
        "ModuleDict named-key CRUD with insertion-order preservation",
        "atomic Layer edit plans",
        "local Layer Edit Provider API v1",
        "registered Layer component runtime smoke test (factory construction, finite state, declared forward probe)",
        "registered transform, dataset, loss, metric, and trainer fixture conformance through component test",
        "registered Layer extension state_dict/load_state_dict roundtrip conformance",
        "variant undo/redo",
        "source Layer set for built-in and uniquely mapped parameterless explicit-signature project nn.Module constructors",
        "source CRUD for directly assigned nn.Sequential, nn.ModuleList, and nn.ModuleDict with built-in or importable Python factory Layers",
        "multi-file source set plans with recoverable journal, fresh-worker revalidation, automatic rollback, conflict-checked undo/redo",
        "grouped source Sequential/ModuleList/ModuleDict add/replace/remove transactions across project files with recovery, revalidation, and conflict-checked undo/redo, including registered local component factories",
        "static installed Python extension discovery with package/source provenance checks",
        "project-local package scaffolds for Layer factories and Layer Edit Providers",
        "project-local Layer/Provider wheel installation without changing the selected environment lock",
        "project Python environment lock/setup/check with wheel hashes, ABI/package drift reporting, and automatic .venv selection",
        "same-ABI hash-verified wheel archives and offline dependency replay",
        "explicit import/init probe for user-named allowlisted ML/RL packages without installation or backend binding",
        "fixed-plan cleanup of ended-run logs and progress files with status, keep, and content-hash checks",
        "persistent variant CRUD for Sequential/ModuleList/ModuleDict subclasses with forward validation",
        "cooperative external-loop deadline exposure and registered Trainer timeout conformance",
    ]
    not_implemented = [
        "assembly target editing",
        "preemption of a blocked third-party Python/CUDA call before the CLI watchdog terminates its worker",
        "arbitrary-input extension correctness and OS-level isolation",
        "full extension authoring SDK and package removal/garbage collection",
        "validation of user-declared sample ID authenticity, dataset-wide coverage denominator, and task-specific label semantics",
        "OS isolation or side-effect prevention for trusted project-local supervision callables",
    ]
    if platform.startswith("linux"):
        implemented.append("Linux detached training supervision, attach, and cooperative stop")
    else:
        not_implemented.append("detached training supervision/attach/stop (Linux only)")
    capabilities = capability_catalog(
        torch_available=torch_available,
        platform=platform,
    )
    return {
        "python": python,
        "python_version": python_version,
        "torch_available": torch_available,
        "torch_version": torch_version,
        "implemented": implemented,
        "not_implemented": not_implemented,
        "compatibility": compatibility,
        "capabilities": capabilities,
        "hint": "使用 --python 指定已有 PyTorch 环境，或在项目 requirements.txt 声明 torch 后运行 m-model env setup。",
    }
