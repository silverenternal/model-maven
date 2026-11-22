use serde_json::Value;
use std::{path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn run_python(program: &str, directory: &Path) -> Value {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let worker = project.join("python");
    let sdk = project.join("python/model_maven_sdk/src");
    let python_path = format!("{}:{}", worker.display(), sdk.display());
    let output = Command::new(python())
        .arg("-c")
        .arg(program)
        .arg(directory)
        .env("PYTHONPATH", python_path)
        .output()
        .expect("start extension state checkpoint fixture");
    assert!(
        output.status.success(),
        "extension state fixture failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("fixture emits JSON")
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

#[test]
fn extension_state_is_schema_bound_and_migrates_only_one_declared_distribution_in_real_training_resume()
 {
    if !has_torch() {
        eprintln!("skipped extension state checkpoint integration: selected Python needs PyTorch");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let output = run_python(
        r#"
import hashlib, importlib, json, os, shutil, sys
from pathlib import Path
import torch
from model_maven_sdk import StateFormat, bind_state_format, parameter
from model_maven_worker.environment.snapshot import snapshot_environment
from model_maven_worker.protocol import UserError
from model_maven_worker.run.provenance import source_manifest
from model_maven_worker.train.checkpoint import (
    checkpoint_value_supported, load_training_checkpoint,
    load_verified_checkpoint_payload, save_training_checkpoint,
    state_dict_digest, training_run_signature,
)
from model_maven_worker.train.external import run_external_loop
from model_maven_worker.train.runner import TrainingRunRuntime, run_training_run
from model_maven_worker.train.state import (
    TrainingStateRuntime, add_finite_total, callable_checkpoint_hooks,
    checkpoint_environment_inventory, checkpoint_rng_state,
    restore_callable_checkpoint, restore_rng_state, scalar_map, scalar_value,
)

root = Path(sys.argv[1]).resolve()
package = root / "state_plugin"
dist_root = root / "mm_state_ext-1.0.0.dist-info"
package.mkdir()
dist_root.mkdir()
(root / "state_plugin" / "__init__.py").write_text("", encoding="utf-8")

def install_metadata(version):
    for candidate in root.glob("mm_state_ext-*.dist-info"):
        shutil.rmtree(candidate)
    info = root / f"mm_state_ext-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: mm-state-ext\nVersion: {version}\n\n",
        encoding="utf-8",
    )
    (info / "top_level.txt").write_text("state_plugin\n", encoding="utf-8")
    (info / "RECORD").write_text(
        "state_plugin/__init__.py,,\nstate_plugin/loop.py,,\n"
        f"mm_state_ext-{version}.dist-info/METADATA,,\n"
        f"mm_state_ext-{version}.dist-info/top_level.txt,,\n"
        f"mm_state_ext-{version}.dist-info/RECORD,,\n",
        encoding="utf-8",
    )

def source(version):
    if version == 1:
        return '''
from model_maven_sdk import StateFormat, bind_state_format, parameter
_STATE = {"counter": 0}
def run_loop(model, context):
    while context.remaining_steps:
        _STATE["counter"] += 1
        if not context.report_step(1.0, {"counter": _STATE["counter"]}):
            break
    return {"counter": _STATE["counter"]}
def save_state(): return dict(_STATE)
def load_state(state): _STATE.clear(); _STATE.update(state)
run_loop.state_dict = save_state
run_loop.load_state_dict = load_state
bind_state_format(run_loop, StateFormat(
    "example.loop-state", "1.0.0",
    {"type":"object", "properties":{"counter":parameter("integer",description="counter")},
     "required":["counter"], "additionalProperties":False},
))
'''
    return '''
from model_maven_sdk import StateFormat, bind_state_format, parameter
_STATE = {"counter": 0, "revision": 2}
def run_loop(model, context):
    if context.resume_state_status == "migrated":
        assert context.resume_state == _STATE, (context.resume_state, _STATE)
    while context.remaining_steps:
        _STATE["counter"] += 1
        if not context.report_step(1.0, {"counter": _STATE["counter"]}):
            break
    return {"counter": _STATE["counter"]}
def save_state(): return dict(_STATE)
def load_state(state): _STATE.clear(); _STATE.update(state)
def migrate_1_to_2(state, *, from_version, to_version):
    assert (from_version, to_version) == ("1.0.0", "2.0.0")
    return {"counter": state["counter"], "revision": 2}
run_loop.state_dict = save_state
run_loop.load_state_dict = load_state
bind_state_format(run_loop, StateFormat(
    "example.loop-state", "2.0.0",
    {"type":"object", "properties":{"counter":parameter("integer",description="counter"),
      "revision":parameter("integer",description="revision")},
     "required":["counter","revision"], "additionalProperties":False},
    migrators={"1.0.0->2.0.0":"state_plugin.loop:migrate_1_to_2"},
), migrators={"1.0.0->2.0.0":migrate_1_to_2})
'''

def load_plugin(version):
    (package / "loop.py").write_text(source(version), encoding="utf-8")
    install_metadata("1.0.0" if version == 1 else "2.0.0")
    sys.modules.pop("state_plugin.loop", None)
    module = importlib.import_module("state_plugin.loop")
    importlib.reload(module)
    return module

sys.path.insert(0, str(root))
plugin = load_plugin(1)
state_runtime = TrainingStateRuntime(torch=torch)
checkpoint_dir = root / ".model-maven" / "runs" / "run-state-migration" / "checkpoints"
checkpoint_dir.mkdir(parents=True)

def make_runtime():
    external_runtime = type("ExternalRuntime", (), {})()
    external_runtime.torch = torch
    external_runtime.move_batch = None
    external_runtime.scalar_value = lambda value, field: scalar_value(state_runtime, value, field)
    external_runtime.scalar_map = lambda value, field: scalar_map(state_runtime, value, field)
    external_runtime.add_finite_total = add_finite_total
    return TrainingRunRuntime(
        entrypoint=lambda _reference: plugin.run_loop,
        load_training_checkpoint=lambda path, model, request, manifest:
            load_training_checkpoint(torch, path, model, request, manifest),
        load_verified_checkpoint_payload=lambda path: load_verified_checkpoint_payload(torch, path),
        migrate_state_dict=lambda *args: (_ for _ in ()).throw(AssertionError("unused model migration")),
        state_dict_digest=lambda _state: "unused",
        model_state_alias_groups=lambda _model: [],
        restore_callable_checkpoint=restore_callable_checkpoint,
        restore_rng_state=lambda state: restore_rng_state(state_runtime, state),
        training_run_signature=training_run_signature,
        checkpoint_value_supported=lambda value: checkpoint_value_supported(value, torch),
        callable_checkpoint_hooks=callable_checkpoint_hooks,
        checkpoint_rng_state=lambda: checkpoint_rng_state(state_runtime),
        model_state_digest=lambda _model: "unused",
        save_training_checkpoint=lambda directory, payload: save_training_checkpoint(torch, directory, payload),
        run_external_loop=lambda model, request, **kwargs:
            run_external_loop(external_runtime, model, request, **kwargs),
        run_managed_steps=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unused managed training")),
        run_manual_steps=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unused manual training")),
        emit_progress=lambda _event: None,
        python_version=sys.version.split()[0],
        pytorch_version=str(torch.__version__),
    )

def make_request(snapshot, steps, checkpoint_input=None):
    request = {
        "operation":"train", "training_mode":"external-loop",
        "training_loop":"state_plugin.loop:run_loop", "steps":steps,
        "checkpoint_every":1, "checkpoint_dir":str(checkpoint_dir),
        "seed":11, "training_params":{}, "runtime":{"device":"cpu"},
        "project_root":str(root),
        "_environment_fingerprint_sha256":snapshot["fingerprint_sha256"],
        "_checkpoint_environment_inventory":checkpoint_environment_inventory(snapshot),
    }
    if checkpoint_input is not None:
        request["checkpoint_input"] = str(checkpoint_input)
    return request

first_snapshot = snapshot_environment(torch, "training_start")
first_request = make_request(first_snapshot, 1)
first_manifest = source_manifest(root, loaded_modules=[plugin])
first_result = run_training_run(make_runtime(), None, first_request, first_manifest)
checkpoint_path = checkpoint_dir / first_result["checkpoint"]["file"]
original_bytes = checkpoint_path.read_bytes()
saved, _ = load_verified_checkpoint_payload(torch, checkpoint_path)
assert saved["loop_state"]["kind"] == "versioned-hooks"
assert saved["loop_state"]["state_format"]["version"] == "1.0.0"
assert first_result["checkpoint_state"]["external_loop"] == "saved"
assert saved["environment_inventory"]["fingerprint_sha256"] == first_snapshot["fingerprint_sha256"]
assert saved["loop_state"]["callable_identity"]["distribution"]["name"] == "mm-state-ext"

# An unrelated package change must fail closed and leave the source checkpoint bytes unchanged.
plugin = load_plugin(2)
second_snapshot = snapshot_environment(torch, "training_start")
second_request = make_request(second_snapshot, 2, checkpoint_path)
second_manifest = source_manifest(root, loaded_modules=[plugin])
bad_inventory = json.loads(json.dumps(second_request["_checkpoint_environment_inventory"]))
bad_inventory["distributions"].append({"name":"unrelated-package","version":"9.0.0","source_kind":"registry_or_unknown"})
bad_request = {**second_request, "_checkpoint_environment_inventory":bad_inventory}
try:
    load_training_checkpoint(torch, checkpoint_path, None, bad_request, second_manifest)
except UserError as error:
    unrelated_change_code = error.code
else:
    raise AssertionError("an unrelated distribution change was allowed")
assert unrelated_change_code == "RESUME_SOURCE_CHANGED"
assert checkpoint_path.read_bytes() == original_bytes

second_result = run_training_run(make_runtime(), None, second_request, second_manifest)
assert second_result["resume_state"]["external_loop"] == "migrated"
assert second_result["completed_steps"] == 2
assert plugin.run_loop.state_dict() == {"counter": 2, "revision": 2}
latest = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
latest_path = checkpoint_dir / latest["file"]
latest_payload, _ = load_verified_checkpoint_payload(torch, latest_path)
assert latest_payload["loop_state"]["state_format"]["version"] == "2.0.0"
assert latest_payload["loop_state"]["state"] == {"counter": 2, "revision": 2}
assert json.loads(json.dumps(second_result["resume_state"]))["external_loop"] == "migrated"
print(json.dumps({
    "first_state_kind":saved["loop_state"]["kind"],
    "old_state_version":saved["loop_state"]["state_format"]["version"],
    "new_state_version":latest_payload["loop_state"]["state_format"]["version"],
    "migration_status":second_result["resume_state"]["external_loop"],
    "final_state":plugin.run_loop.state_dict(),
    "unrelated_environment_change":unrelated_change_code,
    "checkpoint_integrity":"verified",
}, sort_keys=True))
"#,
        project.path(),
    );
    assert_eq!(output["first_state_kind"], "versioned-hooks");
    assert_eq!(output["old_state_version"], "1.0.0");
    assert_eq!(output["new_state_version"], "2.0.0");
    assert_eq!(output["migration_status"], "migrated");
    assert_eq!(output["final_state"]["counter"], 2);
    assert_eq!(output["final_state"]["revision"], 2);
    assert_eq!(
        output["unrelated_environment_change"],
        "RESUME_SOURCE_CHANGED"
    );
    assert_eq!(output["checkpoint_integrity"], "verified");
}
