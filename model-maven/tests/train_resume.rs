use serde_json::Value;
use std::{path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

fn run_python(program: &str, directory: &Path) -> Value {
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let output = Command::new(python())
        .arg("-c")
        .arg(program)
        .arg(directory)
        .env("PYTHONPATH", package)
        .output()
        .expect("start checkpoint fixture");
    assert!(
        output.status.success(),
        "checkpoint fixture failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("fixture emits JSON")
}

#[test]
fn conditional_resume_loads_the_verified_snapshot_after_source_replacement() {
    if !has_torch() {
        eprintln!("skipped checkpoint integration: Python needs PyTorch");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let output = run_python(
        r#"
import hashlib, json, sys
from pathlib import Path
import torch
import model_maven_worker.train.checkpoint as checkpoint

directory = Path(sys.argv[1])
path = directory / 'checkpoint-000001.pt'
payload = {'schema_version':1, 'training_mode':'managed-step', 'completed_steps':1,
           'model_state_sha256':None, 'marker':'confirmed'}
torch.save(payload, path)
original = path.read_bytes()
sha = hashlib.sha256(original).hexdigest()
(directory / 'latest.json').write_text(json.dumps({'schema_version':1, 'file':path.name,
    'sha256':sha, 'completed_steps':1, 'training_mode':'managed-step', 'model_state_sha256':None}))
class ReplacingTorch:
    source = None
    @staticmethod
    def load(source, **kwargs):
        ReplacingTorch.source = source
        path.write_bytes(b'changed after verification')
        return torch.load(source, **kwargs)
loaded, _ = checkpoint.load_verified_checkpoint_payload(ReplacingTorch, path, sha)
path_changed = path.read_bytes() == b'changed after verification'
path.write_bytes(original)
saved_temporary_file = checkpoint.tempfile.TemporaryFile
def no_space(*args, **kwargs): raise OSError('simulated temporary disk full')
checkpoint.tempfile.TemporaryFile = no_space
error_code = None
unconditional_error = None
try:
    checkpoint.load_verified_checkpoint_payload(torch, path, sha)
except checkpoint.UserError as error:
    error_code = error.code
try:
    checkpoint.load_verified_checkpoint_payload(torch, path)
except checkpoint.UserError as error:
    unconditional_error = error.code
finally:
    checkpoint.tempfile.TemporaryFile = saved_temporary_file
print(json.dumps({'marker':loaded['marker'], 'source_closed':ReplacingTorch.source.closed,
                  'path_changed':path_changed, 'no_space_error':error_code,
                  'unconditional_no_space_error':unconditional_error}))
"#,
        directory.path(),
    );
    assert_eq!(output["marker"], "confirmed");
    assert_eq!(output["source_closed"], true);
    assert_eq!(output["path_changed"], true);
    assert_eq!(output["no_space_error"], "CHECKPOINT_SNAPSHOT_UNAVAILABLE");
    assert_eq!(
        output["unconditional_no_space_error"],
        "CHECKPOINT_SNAPSHOT_UNAVAILABLE"
    );
}

#[test]
fn resume_and_evaluation_without_explicit_digest_load_the_verified_bytes() {
    if !has_torch() {
        eprintln!("skipped checkpoint integration: Python needs PyTorch");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let output = run_python(
        r#"
import hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
import model_maven_worker.train.checkpoint as checkpoint
from model_maven_worker.train.evaluation import _load_checkpoint

directory = Path(sys.argv[1])
path = directory / 'checkpoint-000001.pt'
state = {'weight': torch.tensor([1.0])}
state_digest = checkpoint.state_dict_digest(torch, state)
payload = {'schema_version':1, 'training_mode':'managed-step', 'completed_steps':1,
           'model_state':state, 'model_state_sha256':state_digest, 'marker':'verified'}
torch.save(payload, path)
original = path.read_bytes()
sha = hashlib.sha256(original).hexdigest()
(directory / 'latest.json').write_text(json.dumps({'schema_version':1, 'file':path.name,
    'sha256':sha, 'completed_steps':1, 'training_mode':'managed-step', 'model_state_sha256':state_digest}))
class ReplacingTorch:
    def __getattr__(self, name): return getattr(torch, name)
    def load(self, source, **kwargs):
        self.source = source
        path.write_bytes(b'changed after verification')
        return torch.load(source, **kwargs)
replacement = ReplacingTorch()
resumed, _ = checkpoint.load_verified_checkpoint_payload(replacement, path)
resume_closed = replacement.source.closed
path.write_bytes(original)
evaluated, evaluation_sha = _load_checkpoint(SimpleNamespace(torch=replacement), path)
evaluation_closed = replacement.source.closed
print(json.dumps({'resume':resumed['marker'], 'evaluation':evaluated['marker'],
    'digest':evaluation_sha, 'expected':sha, 'resume_closed':resume_closed,
    'evaluation_closed':evaluation_closed, 'source_changed':path.read_bytes() != original}))
"#,
        directory.path(),
    );
    assert_eq!(output["resume"], "verified");
    assert_eq!(output["evaluation"], "verified");
    assert_eq!(output["digest"], output["expected"]);
    assert_eq!(output["resume_closed"], true);
    assert_eq!(output["evaluation_closed"], true);
    assert_eq!(output["source_changed"], true);
}

#[test]
fn failed_same_step_checkpoint_commit_keeps_previous_file_and_pointer_loadable() {
    if !has_torch() {
        eprintln!("skipped checkpoint integration: Python needs PyTorch");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let output = run_python(
        r#"
import json, sys
from pathlib import Path
import torch
import model_maven_worker.train.checkpoint as checkpoint
from model_maven_worker.protocol import UserError

directory = Path(sys.argv[1])
request = {"training_mode":"external-loop", "training_loop":"fixture:run", "seed":7}
source_manifest = {"fixture.py":"a" * 64}
payload = {
    "schema_version":1, "training_mode":"external-loop", "completed_steps":2,
    "run_signature":checkpoint.training_run_signature(request),
    "source_manifest":source_manifest,
    "runtime":{"python_version":sys.version.split()[0], "pytorch_version":str(torch.__version__)},
    "model_state":None, "model_state_sha256":None,
    "engine_state":{"metrics":{"loss_count":2}}, "batch_state":None,
    "step_state":None, "loop_state":{"kind":"stateless"},
    "rng_state":{"python":None, "torch_cpu":torch.get_rng_state()},
}
first = checkpoint.save_training_checkpoint(torch, directory, payload)
first_path = directory / first["file"]
first_bytes = first_path.read_bytes()
original = checkpoint.atomic_json_file
def fail_latest(path, value):
    if Path(path).name == "latest.json":
        raise OSError("injected latest pointer failure")
    return original(path, value)
checkpoint.atomic_json_file = fail_latest
failed_code = None
try:
    checkpoint.save_training_checkpoint(torch, directory, {**payload, "engine_state":{"metrics":{"loss_count":99}}})
except UserError as error:
    failed_code = error.code
finally:
    checkpoint.atomic_json_file = original
latest = json.loads((directory / "latest.json").read_text())
loaded = checkpoint.load_training_checkpoint(torch, first_path, None, request, source_manifest)
print(json.dumps({"failure_code":failed_code, "latest_unchanged":latest == first,
    "old_file_unchanged":first_path.read_bytes() == first_bytes,
    "loaded_count":loaded["engine_state"]["metrics"]["loss_count"],
    "checkpoint_files":len(list(directory.glob("checkpoint-*.pt")))}, sort_keys=True))
"#,
        directory.path(),
    );
    assert_eq!(output["failure_code"], "CHECKPOINT_WRITE_FAILED");
    assert_eq!(output["latest_unchanged"], true);
    assert_eq!(output["old_file_unchanged"], true);
    assert_eq!(output["loaded_count"], 2);
    assert_eq!(output["checkpoint_files"], 2);
}

#[test]
fn checkpoint_manifest_and_model_state_fingerprint_must_match_payload() {
    if !has_torch() {
        eprintln!("skipped checkpoint integration: Python needs PyTorch");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let output = run_python(
        r#"
import hashlib, json, sys
from pathlib import Path
import torch
import model_maven_worker.train.checkpoint as checkpoint
from model_maven_worker.protocol import UserError

directory = Path(sys.argv[1])
request = {"training_mode":"external-loop", "training_loop":"fixture:run", "seed":3}
source_manifest = {"fixture.py":"b" * 64}
model = torch.nn.Linear(3, 2)
state = model.state_dict()
payload = {
    "schema_version":1, "training_mode":"external-loop", "completed_steps":4,
    "run_signature":checkpoint.training_run_signature(request),
    "source_manifest":source_manifest,
    "runtime":{"python_version":sys.version.split()[0], "pytorch_version":str(torch.__version__)},
    "model_state":state, "model_state_sha256":checkpoint.state_dict_digest(torch, state),
    "engine_state":{}, "batch_state":None, "step_state":None,
    "loop_state":{"kind":"stateless"},
    "rng_state":{"python":None, "torch_cpu":torch.get_rng_state()},
}
manifest = checkpoint.save_training_checkpoint(torch, directory, payload)
path = directory / manifest["file"]
manifest["completed_steps"] = 5
checkpoint.atomic_json_file(directory / "latest.json", manifest)
try:
    checkpoint.load_training_checkpoint(torch, path, model, request, source_manifest)
except UserError as error:
    manifest_error = error.code
manifest["completed_steps"] = 4
checkpoint.atomic_json_file(directory / "latest.json", manifest)
manifest["model_state_sha256"] = "f" * 64
checkpoint.atomic_json_file(directory / "latest.json", manifest)
try:
    checkpoint.load_training_checkpoint(torch, path, model, request, source_manifest)
except UserError as error:
    model_manifest_error = error.code
manifest["model_state_sha256"] = payload["model_state_sha256"]
checkpoint.atomic_json_file(directory / "latest.json", manifest)
payload["model_state_sha256"] = "0" * 64
torch.save(payload, path)
manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest["model_state_sha256"] = payload["model_state_sha256"]
checkpoint.atomic_json_file(directory / "latest.json", manifest)
try:
    checkpoint.load_training_checkpoint(torch, path, model, request, source_manifest)
except UserError as error:
    model_error = error.code
print(json.dumps({"manifest_error":manifest_error, "model_manifest_error":model_manifest_error,
    "model_error":model_error}, sort_keys=True))
"#,
        directory.path(),
    );
    assert_eq!(output["manifest_error"], "CHECKPOINT_INVALID");
    assert_eq!(output["model_manifest_error"], "CHECKPOINT_INVALID");
    assert_eq!(output["model_error"], "CHECKPOINT_CORRUPT");
}

#[test]
fn checkpoint_restores_device_rng_and_rejects_device_drift() {
    let directory = tempfile::tempdir().unwrap();
    let output = run_python(
        r#"
import json, sys
from types import SimpleNamespace
from model_maven_worker.protocol import UserError
from model_maven_worker.train.state import TrainingStateRuntime, checkpoint_rng_state, restore_rng_state

class Cuda:
    def __init__(self, count):
        self.count = count
        self.states = [f"device-{index}" for index in range(count)]
    def is_available(self): return self.count > 0
    def device_count(self): return self.count
    def get_rng_state_all(self): return list(self.states)
    def set_rng_state_all(self, states): self.states = list(states)

class Torch:
    def __init__(self, count):
        self.cuda = Cuda(count)
        self.cpu_state = "cpu-initial"
    def get_rng_state(self): return self.cpu_state
    def set_rng_state(self, state): self.cpu_state = state

torch = Torch(2)
runtime = TrainingStateRuntime(torch=torch)
saved = checkpoint_rng_state(runtime)
torch.cuda.states = ["changed-0", "changed-1"]
torch.cpu_state = "cpu-changed"
restore_rng_state(runtime, saved)
restored = torch.cuda.states == ["device-0", "device-1"] and torch.cpu_state == "cpu-initial"

drift_code = None
try:
    restore_rng_state(TrainingStateRuntime(torch=Torch(1)), saved)
except UserError as error:
    drift_code = error.code

missing_code = None
try:
    restore_rng_state(TrainingStateRuntime(torch=Torch(1)), {"python": saved["python"], "torch_cpu": saved["torch_cpu"]})
except UserError as error:
    missing_code = error.code

print(json.dumps({"restored": restored, "saved_device_count": len(saved["torch_cuda"]),
    "drift_code": drift_code, "missing_code": missing_code}, sort_keys=True))
"#,
        directory.path(),
    );
    assert_eq!(output["restored"], true);
    assert_eq!(output["saved_device_count"], 2);
    assert_eq!(output["drift_code"], "RESUME_DEVICE_INCOMPATIBLE");
    assert_eq!(output["missing_code"], "RESUME_DEVICE_RNG_UNAVAILABLE");
}
