#!/usr/bin/env python3
"""Run the real CLI/TUI integration fixtures and retain auditable results."""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT.parent / "model-maven"
CASES = (
    ("journey_a", "verify_journey_a.py", "empty-directory dual-input authoring, repair, supervision, training, evaluation and isolated restore; synthetic data"),
    ("journey_b", "verify_journey_b.py", "existing model diagnosis and source repair; legacy and concurrent changes simulated by wrappers"),
    ("graph_pair", "verify_real_pair.py", "real CLI, two terminal rendering modes"),
    ("language_terminal", "verify_language_terminal.py", "40/80-column language picker and saved welcome locale in a full-screen PTY"),
    ("graph_evidence", "verify_graph_evidence_pair.py", "real shared branches, cloned and mutated tensors, unobserved nodes and truncated trace"),
    ("graph_identity_terminal", "verify_real_identity_path.py", "full-screen tensor identity and multi-consumer navigation"),
    ("capability_negotiation", "verify_capability_negotiation_pair.py", "real TUI with capability-missing, malformed and legacy CLI receipt wrappers"),
    ("large_real_terminal", "verify_large_real_terminal.py", "real 1024-layer model, truncated trace and PTY interaction timings"),
    ("guarded_commit", "verify_real_guarded_commit.py", "real CLI; some conflict/version cases use a wrapper"),
    ("snapshot_replay", "verify_real_snapshot_replay.py", "real CLI; legacy and failure cases use wrappers"),
    ("classification_create", "verify_classification_create.py", "real TUI setup and CLI forward with generated image fixture"),
    ("classification_journey", "verify_classification_journey.py", "Tiny CNN creation, image preview, forward and one trial using fixture images"),
    ("classification_recovery", "verify_classification_recovery_pair.py", "classification dependency failure and unreadable-image recovery; dependency branch depends on selected Python"),
    ("classification_resnet18", "verify_classification_journey.py", "real TUI ResNet18 creation, data preview, forward and trial; requires usable torchvision"),
    ("custom_scaffold", "verify_custom_scaffold.py", "real TUI creation and dual-input forward with synthetic data"),
    ("scaffold_drafts", "verify_scaffold_drafts_pair.py", "real TUI preserves project-scoped starter form drafts"),
    ("scaffold_drafts_terminal", "verify_scaffold_drafts_terminal.py", "full-screen PTY resumes cancelled starter form"),
    ("scaffold_draft_restart", "verify_scaffold_draft_restart.py", "Unix process termination, draft recovery, lock contention and safe cleanup"),
    ("scaffold_named_outputs", "verify_scaffold_named_outputs_pair.py", "two named output heads, trial, conflict and source preservation"),
    ("scaffold_objectives", "verify_scaffold_objectives_pair.py", "classification/regression objectives with CSV labels and invalid-target rejection"),
    ("csv_adapter", "verify_csv_adapter_pair.py", "CSV adapter creation, configuration, checkpoint resume and data change rejection"),
    ("csv_typed", "verify_csv_typed_pair.py", "typed and shaped CSV inputs, scalar targets and training"),
    ("scaffold_routes", "verify_scaffold_routes_pair.py", "real TUI creates explicit shared-input encoder routes"),
    ("legacy_starter", "verify_legacy_starter_pair.py", "real earlier starter reports unsupported or ignored route/output options without writes"),
    ("starter_migration", "verify_starter_migration_pair.py", "real earlier starter migration, cancellation, source conflict, recheck and undo/redo"),
    ("route_edit", "verify_route_edit_pair.py", "real TUI previews and conditionally edits input routes; concurrent change uses wrapper"),
    ("assembly_pair", "verify_assembly_pair.py", "real TUI assembly preview, cancel, commit and forward"),
    ("assembly_single_source_guard", "verify_single_assembly_source_guard.py", "single-seam TUI add, set, remove reject preview-to-commit project source changes"),
    ("assembly_single_input_guard", "verify_single_assembly_input_guard.py", "single-seam optional input matching rejects changed representative batches"),
    ("assembly_single_snapshot", "verify_single_assembly_snapshot.py", "single-seam dynamic input capture, replay, cancellation and cleanup"),
    ("assembly_single_capabilities", "verify_single_assembly_capabilities.py", "legacy, missing and malformed capability catalogs for single-seam edits"),
    ("component_uninstall", "verify_component_uninstall_pair.py", "guarded component removal with cancellation and concurrent manifest change"),
    ("component_package_tui", "verify_component_package_tui.py", "TUI scaffold, local installation and test of a user component"),
    ("component_plan_capacity", "verify_component_plan_capacity.py", "review and cancel remain available with eight project-scoped removal previews"),
    ("uncertain_training_launch", "verify_uncertain_training_launch.py", "lost real launch receipt with simulated long run list"),
    ("pending_start_recovery", "verify_pending_start_recovery.py", "kill TUI during real detached start, reopen and acknowledge the matching run"),
    ("request_lookup_scale", "verify_request_lookup_scale.py", "1024 synthetic runs compare exact lookup and full-list response size"),
    ("assembly_fullscreen", "verify_assembly_fullscreen.py", "full-screen dual-input creation and recovery from invalid encoder"),
    ("assembly_registered_component", "verify_assembly_catalog_pair.py", "TUI registration, assembly and reconstruction; catalog pagination wrapper injects synthetic list entries"),
    ("assembly_generation_guard", "verify_assembly_generation_guard_pair.py", "TUI rejects adapter publication when a real factory changes configuration during generated-code verification"),
    ("assembly_generation_source_guard", "verify_assembly_generation_source_guard.py", "TUI rejects adapter publication when observed source changes during generated-code verification"),
    ("assembly_generation_input_comparison", "verify_generation_input_comparison.py", "code-generation forward inputs classified as same, different or not run"),
    ("assembly_recovery", "verify_assembly_recovery.py", "real TUI repair of a previously broken assembly"),
    ("supervision_pair", "verify_supervision_pair.py", "real TUI semantic mark and read-only rule preview"),
    ("supervision_parameters", "verify_supervision_parameter_pair.py", "TUI-created fusion require rule with actual parameter comparison on reject and pass"),
    ("run_supervision_coverage", "verify_run_supervision_coverage.py", "completed TUI run distinguishes enabled observe rule from actual measurements"),
    ("native_encoder", "verify_native_encoder_pair.py", "variable-length integer encoder output reaches custom fusion intact and trains"),
    ("atomic_encoder_fusion", "verify_atomic_encoder_fusion.py", "TUI atomically switches mutually dependent encoder output and fusion"),
    ("atomic_route_head", "verify_atomic_route_head.py", "TUI atomically switches mutually dependent input route and head"),
    ("declared_route_terminal", "verify_declared_route_terminal.py", "real failed forward exposes configured seam route separately from observed flow in PTY"),
    ("training_pair", "verify_training_pair.py", "real TUI trial, detached training and stop"),
    ("training_capabilities", "verify_training_capability_guard.py", "missing or malformed training protection capabilities refuse launch"),
    ("training_variant_guard", "verify_training_variant_guard.py", "real variant guarded training and stale variant rejection"),
    ("background_trial", "verify_background_trial_pair.py", "real background trial navigation, cancellation and exit cleanup"),
    ("live_run_terminal", "verify_live_run_terminal.py", "full-screen live progress, retry and leaving a real background run active"),
    ("resume_environment_drift", "verify_resume_environment_drift.py", "isolated Python distribution inventory drift rejects and restored environment resumes"),
    ("training_data_versions", "verify_training_data_versions.py", "local CSV training with explicitly declared and unknown data versions"),
    ("training_guard", "verify_training_guard.py", "TUI stale configuration guard; wrapper simulates concurrent edit"),
    ("artifacts_pair", "verify_artifacts_pair.py", "real TUI run comparison, evaluation and export/restore"),
    ("prediction_browser_terminal", "verify_prediction_browser_terminal.py", "real saved evaluation prediction browser in 40/100-column PTYs"),
    ("export_stateful_continuation", "verify_export_stateful_continuation.py", "exported checkpoint continuation with stateful batch, RNG and optimizer"),
    ("layer_crud_pair", "verify_layer_crud_pair.py", "real TUI variant layer CRUD and recheck"),
    ("layer_guard_pair", "verify_layer_guard_pair.py", "real TUI edit with simulated concurrent variant/input changes"),
    ("layer_graph_pair", "verify_layer_graph_pair.py", "real terminal graph handoff, variant edit and recheck"),
    ("shared_parameter_graph", "verify_shared_parameter_graph_pair.py", "graph edit and recheck with parameter objects shared between modules"),
    ("shared_storage_graph", "verify_shared_storage_graph_pair.py", "graph edit and recheck with distinct shared-storage parameter views"),
    ("data_preview_pair", "verify_data_preview_pair.py", "real background data preview, cancellation and project switch"),
    ("environment_pair", "verify_environment_pair.py", "offline local-wheel setup, missing-library diagnosis and explicit repair"),
    ("components_pair", "verify_components_pair.py", "static discovery, component inspection, check and explicit runtime test"),
    ("augmentation_pair", "verify_augmentation_pair.py", "guarded augmentation edits, concurrency conflict and transformed data preview"),
    ("dataset_preview_pair", "verify_dataset_preview_pair.py", "real Dataset image artifacts and project-owned custom renderer"),
    ("linear_source_guard", "verify_linear_source_guard.py", "Linear repair refuses a changed source manifest"),
    ("source_edit_pair", "verify_source_edit_pair.py", "guarded source edit, concurrent change, post-save verification and history"),
    ("source_edit_terminal", "verify_source_edit_terminal.py", "full-screen source diff and safe cancellation"),
    ("source_graph_pair", "verify_source_graph_pair.py", "full-screen graph selection to guarded source edit"),
    ("existing_fullscreen_journey", "verify_existing_fullscreen_journey.py", "existing model diagnosis, graph edit and restart"),
    ("source_shared_graph", "verify_source_shared_graph_pair.py", "source edit shows and rechecks shared registered state"),
    ("source_snapshot", "verify_source_snapshot_pair.py", "changing batch captured and replayed for source edit"),
    ("source_snapshot_failure", "verify_source_snapshot_failure_pair.py", "repair source after captured original forward failure"),
    ("source_snapshot_cancel", "verify_source_snapshot_cancel_pair.py", "cancel source edit and remove captured input"),
    ("source_input_conflict", "verify_source_input_conflict_pair.py", "source input changes after confirmation; wrapper injects conflict"),
    ("source_config_conflict", "verify_source_config_conflict_pair.py", "source configuration changes after confirmation; wrapper injects conflict"),
    ("source_capture_config_conflict", "verify_source_capture_config_conflict_pair.py", "configuration changes during capture; wrapper injects conflict"),
    ("source_manifest_conflict", "verify_source_manifest_conflict_pair.py", "observed source dependency changes; wrapper injects conflict"),
    ("source_containers_pair", "verify_source_containers_pair.py", "source ModuleList and ModuleDict edits across restarts"),
    ("component_form_pair", "verify_component_form_pair.py", "component Schema guided fields and JSON editing"),
    ("component_form_terminal", "verify_component_form_terminal.py", "full-screen component Schema prompts and safe cancellation"),
    ("builtin_form_pair", "verify_builtin_form_pair.py", "catalog-guided built-in Layer editing"),
    ("builtin_form_terminal", "verify_builtin_form_terminal.py", "full-screen built-in Layer form and safe cancellation"),
)

# Explicit opt-in cases have extra real dependencies and are not included in
# the default suite. They never install packages or alter the selected Python.
OPTIONAL_CASES = (
    ("real_gymnasium", "verify_real_gymnasium_pair.py", "optional: real Gymnasium in a pre-existing PyTorch venv and user-owned loop"),
)


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def repository(path, output):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args])
    head = git("rev-parse", "HEAD").decode("ascii").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    diff = git("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--")
    untracked = sorted(name for name in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if name)
    try:
        excluded = output.relative_to(path)
    except ValueError:
        excluded = None
    digest = hashlib.sha256()
    digest.update(b"model-maven-git-observed-content-v1\0")
    digest.update(head.encode("ascii") + b"\0")
    digest.update(hashlib.sha256(diff).digest())
    count, total_bytes, errors = 0, 0, []
    for name in untracked:
        relative = Path(os.fsdecode(name))
        if excluded is not None and (relative == excluded or excluded in relative.parents):
            continue
        source = path / relative
        try:
            info = source.lstat()
            if stat.S_ISREG(info.st_mode):
                file_hash = hashlib.sha256()
                with source.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        file_hash.update(chunk)
                total_bytes += info.st_size
                kind = b"file"
                payload = file_hash.digest()
            elif stat.S_ISLNK(info.st_mode):
                kind = b"symlink"
                payload = hashlib.sha256(os.fsencode(os.readlink(source))).digest()
            else:
                raise OSError("unsupported file type")
            digest.update(len(name).to_bytes(8, "big") + name + kind + b"\0")
            digest.update((info.st_mode & 0o7777).to_bytes(4, "big") + payload)
            count += 1
        except OSError as error:
            errors.append({"path": os.fsdecode(name), "error": str(error)})
    return {"path": str(path), "head": head, "dirty": bool(status),
            "status_sha256": hashlib.sha256(status).hexdigest(),
            "content_identity": {"algorithm": "git-observed-content-v1", "sha256": digest.hexdigest(),
                                 "status": "incomplete" if errors else "available", "errors": errors,
                                 "untracked_files": count, "untracked_bytes": total_bytes,
                                 "excluded_output": str(excluded) if excluded is not None else None,
                                 "scope": "HEAD, tracked git diff and nonignored untracked files; not ignored files or nested repository contents"},
            "note": "Read-time observation is not an atomic filesystem snapshot."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--tui", type=Path)
    parser.add_argument("--python", type=Path, help="Python environment used by CLI workers")
    parser.add_argument("--output", type=Path, help="new directory for fixture logs and report")
    parser.add_argument("--list", action="store_true", help="list case IDs and scopes without executing")
    parser.add_argument("--case", action="append", default=[], metavar="ID", help="run selected case; repeatable (default: full suite)")
    parser.add_argument("--case-timeout", type=int, default=900, metavar="SECONDS", help="per-case timeout; default 900 seconds")
    args = parser.parse_args()
    if args.list:
        for case, script, scope in CASES + OPTIONAL_CASES:
            print(f"{case}\t{script}\t{scope}")
        return 0
    if any(value is None for value in (args.cli, args.tui, args.python, args.output)):
        parser.error("--cli, --tui, --python and --output are required unless --list is used")
    if args.case_timeout <= 0:
        parser.error("--case-timeout must be positive")
    known = {case for case, _, _ in CASES + OPTIONAL_CASES}
    if len(known) != len(CASES) + len(OPTIONAL_CASES):
        parser.error("integration case IDs must be unique")
    unknown = sorted(set(args.case) - known)
    if unknown:
        parser.error("unknown cases: " + ", ".join(unknown))
    selected = [case for case in CASES + OPTIONAL_CASES if (case[0] in args.case if args.case else case in CASES)]
    cli, tui = (p.expanduser().resolve(strict=True) for p in (args.cli, args.tui))
    # Preserve a venv's interpreter path even when it is a symlink to the base binary.
    python = args.python.expanduser().absolute()
    if not python.is_file():
        parser.error(f"Python executable not found: {python}")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cli": identity(cli), "tui": identity(tui), "python": identity(python),
        "backend_repo": repository(BACKEND, output), "tui_repo": repository(ROOT, output),
        "platform": platform.platform(),
        "invocation": sys.argv,
        "scope": "isolated synthetic fixtures; not a user study or production acceptance",
        "selection": {"requested": args.case, "selected_count": len(selected),
                      "full_suite": {case[0] for case in CASES}.issubset({case[0] for case in selected})},
        "case_timeout_seconds": args.case_timeout,
        "cases": [],
    }
    runtime = subprocess.run([str(python), "-c", "import sys, torch; print(sys.version); print('torch=' + torch.__version__)"],
                             capture_output=True, text=True, timeout=60)
    report["runtime"] = {"exit_code": runtime.returncode,
                         "stdout": runtime.stdout.strip(), "stderr": runtime.stderr.strip()}
    if runtime.returncode != 0:
        report["status"] = "runtime_unavailable"
        report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Python/PyTorch runtime unavailable; no fixtures were run.", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="maven-integration-python-") as directory:
        # The CLI selects python3 through PATH when the fixture has no project environment.
        launcher = Path(directory, "python3")
        launcher.write_text(f"#!/bin/sh\nexec {shlex.quote(str(python))} \"$@\"\n")
        launcher.chmod(0o700)
        env = os.environ.copy()
        # Focused local runs may narrow fixtures. The integration report claims
        # the full cases, so never inherit those selection switches here.
        for key in ("MODEL_MAVEN_GUARDED_SUCCESS_ONLY", "MODEL_MAVEN_SOURCE_CONTAINER",
                    "MODEL_MAVEN_LAYER_SET_ONLY", "MODEL_MAVEN_LAYER_RECHECK_ONLY",
                    "MODEL_MAVEN_FORM_TARGET", "MODEL_MAVEN_FORM_DEFAULT",
                    "MODEL_MAVEN_FORM_BUILTIN", "MODEL_MAVEN_BUILTIN_TARGET",
                    "MODEL_MAVEN_JOURNEY_EVIDENCE", "MODEL_MAVEN_JOURNEY_B_EVIDENCE",
                    "MODEL_MAVEN_CLASSIFICATION_PRESET"):
            env.pop(key, None)
        env.update(MODEL_MAVEN_CLI=str(cli), MODEL_MAVEN_TUI_BIN=str(tui),
                   PATH=directory + os.pathsep + env.get("PATH", ""))
        for case, script, scope in selected:
            command = [str(python), str(ROOT / "scripts" / script)]
            artifact_directory = output / "artifacts" / case
            case_env = {**env, "MODEL_MAVEN_INTEGRATION_ARTIFACTS_DIR": str(artifact_directory)}
            if case == "journey_a":
                case_env["MODEL_MAVEN_JOURNEY_EVIDENCE"] = str(artifact_directory)
            elif case == "journey_b":
                case_env["MODEL_MAVEN_JOURNEY_B_EVIDENCE"] = str(artifact_directory)
            elif case == "classification_resnet18":
                case_env["MODEL_MAVEN_CLASSIFICATION_PRESET"] = "resnet18"
            child = subprocess.Popen(command, env=case_env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                content, _ = child.communicate(timeout=args.case_timeout)
                code = child.returncode
            except subprocess.TimeoutExpired as error:
                os.killpg(child.pid, signal.SIGKILL)
                content, _ = child.communicate()
                code = None
                content += b"\nIntegration fixture timed out.\n"
            (output / f"{case}.log").write_bytes(content)
            artifacts = []
            if artifact_directory.is_dir():
                for artifact in sorted(artifact_directory.rglob("*")):
                    if artifact.is_file() and not artifact.is_symlink():
                        artifacts.append({"path": str(artifact.relative_to(output)), "sha256": identity(artifact)["sha256"], "bytes": artifact.stat().st_size})
            skipped = case == "classification_resnet18" and code == 77
            report["cases"].append({"id": case, "command": command, "scope": scope,
                                    "exit_code": code, "status": "skipped" if skipped else "passed" if code == 0 else "failed", "log": f"{case}.log", "artifacts": artifacts})
            print(f"{case}: {'SKIP' if skipped else 'PASS' if code == 0 else 'FAIL'} ({output / (case + '.log')})", flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    after = {}
    for name, path in (("cli", cli), ("tui", tui), ("python", python)):
        try:
            after[name] = identity(path)
        except OSError as error:
            after[name] = {"path": str(path), "error": str(error)}
    report["binary_identity_after"] = after
    report["binaries_unchanged"] = all(report[name]["sha256"] == after[name].get("sha256") for name in ("cli", "tui", "python"))
    source_after = {"backend_repo": repository(BACKEND, output), "tui_repo": repository(ROOT, output)}
    report["source_identity_after"] = source_after
    report["sources_unchanged"] = all(
        report[name]["content_identity"]["status"] == "available"
        and source_after[name]["content_identity"]["status"] == "available"
        and report[name]["content_identity"]["sha256"] == source_after[name]["content_identity"]["sha256"]
        for name in ("backend_repo", "tui_repo")
    )
    if report["binaries_unchanged"] and report["sources_unchanged"] and all(case["exit_code"] == 0 or case["status"] == "skipped" for case in report["cases"]):
        report["status"] = "incomplete" if any(case["status"] == "skipped" for case in report["cases"]) else "passed"
    else:
        report["status"] = "failed"
    report["identity_note"] = "Start/end checks cannot detect temporary changes restored between observations; git-ignored files and nested repositories are excluded."
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
