use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

#[test]
fn weight_migration_requires_explicit_semantics_and_accounts_for_all_state() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let script = r#"
import sys, torch
sys.path.insert(0, sys.argv[1])
from model_maven_worker.edit.migration import migrate_state_dict
from model_maven_worker.protocol import UserError

source = {"encoder.weight": torch.full((2, 2), 7.0), "legacy.bias": torch.tensor([3.0])}
target = {"encoder.weight": torch.zeros((2, 2)), "head.weight": torch.full((1, 2), 11.0)}
source_before = {key: value.clone() for key, value in source.items()}
target_before = {key: value.clone() for key, value in target.items()}
plan = {
    "schema_version": 1,
    "preserve_compatible": [{
        "source_key": "encoder.weight",
        "target_key": "encoder.weight",
        "semantic_reason": "Both tensors are the same named linear projection after the edit.",
    }],
    "reset_target": ["head.weight"],
    "drop_source": ["legacy.bias"],
}
migrated, report = migrate_state_dict(torch, source, target, plan)
assert torch.equal(migrated["encoder.weight"], source["encoder.weight"])
assert torch.equal(migrated["head.weight"], target["head.weight"])
assert report["optimizer_state"] == "reset_required"
assert report["initialized"] == ["head.weight"]
assert report["dropped"] == ["legacy.bias"]
assert all(torch.equal(source[key], source_before[key]) for key in source)
assert all(torch.equal(target[key], target_before[key]) for key in target)

def must_reject(candidate, expected_code):
    try:
        migrate_state_dict(torch, source, target, candidate)
    except UserError as error:
        assert error.code == expected_code, (error.code, expected_code)
    else:
        raise AssertionError(f"expected {expected_code}")

missing_semantics = dict(plan)
missing_semantics["preserve_compatible"] = [{"source_key":"encoder.weight", "target_key":"encoder.weight"}]
must_reject(missing_semantics, "MIGRATION_SEMANTICS_REQUIRED")
shape_mismatch = dict(plan)
shape_mismatch["preserve_compatible"] = [{
    "source_key":"encoder.weight", "target_key":"head.weight", "semantic_reason":"test mapping"
}]
must_reject(shape_mismatch, "MIGRATION_ADAPTER_REQUIRED")
unaccounted_target = dict(plan)
unaccounted_target["reset_target"] = []
must_reject(unaccounted_target, "MIGRATION_TARGET_UNACCOUNTED")
partial = dict(plan)
partial["slice"] = {"source":"encoder.weight", "target":"head.weight"}
must_reject(partial, "MIGRATION_PLAN_INVALID")
alias_plan = {
    "schema_version": 1,
    "preserve_compatible": [
        {"source_key":"a.weight", "target_key":"a.weight", "semantic_reason":"The tied projection keeps its role."},
        {"source_key":"b.weight", "target_key":"b.weight", "semantic_reason":"The tied projection keeps its alias."},
    ],
    "reset_target": [],
    "drop_source": [],
}
alias_state = {"a.weight": torch.ones(2), "b.weight": torch.ones(2)}
try:
    migrate_state_dict(torch, alias_state, alias_state, alias_plan, [["a.weight", "b.weight"]], [])
except UserError as error:
    assert error.code == "MIGRATION_SHARED_IDENTITY_MISMATCH"
else:
    raise AssertionError("migration must reject a tied-to-untied topology change")
_, alias_report = migrate_state_dict(
    torch, alias_state, alias_state, alias_plan,
    [["a.weight", "b.weight"]], [["a.weight", "b.weight"]]
)
assert alias_report["shared_identity"].startswith("preserved_for_fully_mapped_alias_groups")
assert alias_report["preserved_shared_alias_groups"] == [{
    "source_keys":["a.weight", "b.weight"],
    "target_keys":["a.weight", "b.weight"],
}]
try:
    migrate_state_dict(torch, {**source, "custom.extra": "state"}, target, plan)
except UserError as error:
    assert error.code == "MIGRATION_ADAPTER_REQUIRED"
else:
    raise AssertionError("non-Tensor extra state must require an explicit adapter")
"#;
    let python_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let output = Command::new(python())
        .args(["-c", script])
        .arg(python_source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn invoke_cli(project: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let python = python();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", &python])
        .args(args)
        .current_dir(project)
        .output()
        .expect("start m-model migration integration");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report, got {:?}: {error}; status={:?}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

#[test]
fn structure_migration_creates_immutable_parent_linked_child_and_resets_optimizer() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch CLI integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let source_v1 = r#"
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.shared_encoder = nn.Linear(4, 4)
        self.shared_encoder.weight = self.encoder.weight
        self.classifier = nn.Linear(4, 2)
    def forward(self, values):
        return self.classifier(torch.relu(self.encoder(values)))

def build_model(): return Model()
def build_batch(size=2):
    assert size == 2
    return {"inputs": (torch.ones(size, 4),), "target": torch.tensor([0, 1])}
def training_step(model, batch, context):
    (values,) = batch["inputs"]
    return {"loss": nn.functional.cross_entropy(model(values), batch["target"])}
"#;
    let source_v2 = source_v1
        .replace(
            "self.classifier = nn.Linear(4, 2)",
            "self.head = nn.Linear(4, 3)",
        )
        .replace(
            "return self.classifier(torch.relu(self.encoder(values)))",
            "return self.head(torch.relu(self.encoder(values)))",
        );
    let source_path = project.path().join("migration_fixture.py");
    fs::write(&source_path, source_v1).unwrap();
    let migration_plan = serde_json::json!({
        "schema_version":1,
        "preserve_compatible":[
            {"source_key":"encoder.weight","target_key":"encoder.weight","semantic_reason":"The encoder projection retains the same role and input/output channels."},
            {"source_key":"encoder.bias","target_key":"encoder.bias","semantic_reason":"The encoder bias retains the same affine projection semantics."},
            {"source_key":"shared_encoder.weight","target_key":"shared_encoder.weight","semantic_reason":"The second module still references the same shared encoder weight."},
            {"source_key":"shared_encoder.bias","target_key":"shared_encoder.bias","semantic_reason":"The second module bias retains the same affine projection semantics."}
        ],
        "reset_target":["head.weight","head.bias"],
        "drop_source":["classifier.weight","classifier.bias"]
    });
    let plan_path = project.path().join("migration.json");
    fs::write(
        &plan_path,
        serde_json::to_vec_pretty(&migration_plan).unwrap(),
    )
    .unwrap();

    let (parent_output, parent_report) = invoke_cli(
        project.path(),
        &[
            "train",
            "--model",
            "migration_fixture:build_model",
            "--batch",
            "migration_fixture:build_batch",
            "--step",
            "migration_fixture:training_step",
            "--steps",
            "2",
            "--checkpoint-every",
            "1",
            "--seed",
            "71",
        ],
    );
    assert!(
        parent_output.status.success(),
        "{parent_report:#}\n{}",
        String::from_utf8_lossy(&parent_output.stderr)
    );
    let parent_id = parent_report["data"]["run_id"].as_str().unwrap().to_owned();
    let parent_checkpoints = project
        .path()
        .join(".model-maven/runs")
        .join(&parent_id)
        .join("checkpoints");
    let parent_latest_path = parent_checkpoints.join("latest.json");
    let parent_latest_bytes = fs::read(&parent_latest_path).unwrap();
    let parent_latest: Value = serde_json::from_slice(&parent_latest_bytes).unwrap();
    let parent_checkpoint_path = parent_checkpoints.join(parent_latest["file"].as_str().unwrap());
    let parent_checkpoint_bytes = fs::read(&parent_checkpoint_path).unwrap();

    let (initialized, init_report) = invoke_cli(
        project.path(),
        &[
            "init",
            "--model",
            "migration_fixture:build_model",
            "--batch",
            "migration_fixture:build_batch",
        ],
    );
    assert!(initialized.status.success(), "{init_report:#}");
    let (preview_output, batch_preview) = invoke_cli(
        project.path(),
        &[
            "config",
            "set-batch",
            "--batch",
            "migration_fixture:build_batch",
            "--batch-param",
            "size=2",
        ],
    );
    assert!(preview_output.status.success(), "{batch_preview:#}");
    let (saved, saved_report) = invoke_cli(
        project.path(),
        &[
            "config",
            "set-batch",
            "--batch",
            "migration_fixture:build_batch",
            "--batch-param",
            "size=2",
            "--apply",
            "--if-current",
            batch_preview["data"]["from_revision"].as_str().unwrap(),
        ],
    );
    assert!(saved.status.success(), "{saved_report:#}");

    fs::write(&source_path, &source_v2).unwrap();
    let (child_output, child_report) = invoke_cli(
        project.path(),
        &[
            "train",
            "--model",
            "migration_fixture:build_model",
            "--step",
            "migration_fixture:training_step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
            "--seed",
            "71",
            "--migrate-from",
            &parent_id,
            "--migration-plan",
            "migration.json",
        ],
    );
    assert!(
        child_output.status.success(),
        "{child_report:#}\n{}",
        String::from_utf8_lossy(&child_output.stderr)
    );
    let child_id = child_report["data"]["run_id"].as_str().unwrap().to_owned();
    let (shown, child_snapshot) = invoke_cli(project.path(), &["runs", "show", &child_id]);
    assert!(shown.status.success(), "{child_snapshot:#}");
    assert_eq!(child_snapshot["data"]["inputs"]["batch_params"]["size"], 2);
    assert_eq!(
        child_snapshot["data"]["inputs"]["input_sources"]["batch_params"]["kind"],
        "project_config"
    );
    let migration = &child_report["data"]["weight_migration"];
    assert_eq!(migration["parent_run_id"], parent_id);
    assert_eq!(
        migration["source_checkpoint_sha256"],
        parent_latest["sha256"]
    );
    assert_eq!(migration["source_completed_steps"], 2);
    assert_eq!(migration["child_initial_step"], 0);
    assert_eq!(migration["optimizer_state"], "reset_required");
    assert_eq!(migration["preserved"].as_array().unwrap().len(), 4);
    assert_eq!(
        migration["preserved_shared_alias_groups"],
        serde_json::json!([{
            "source_keys":["encoder.weight", "shared_encoder.weight"],
            "target_keys":["encoder.weight", "shared_encoder.weight"]
        }])
    );
    assert_eq!(
        migration["initialized"],
        serde_json::json!(["head.bias", "head.weight"])
    );
    assert_eq!(
        migration["dropped"],
        serde_json::json!(["classifier.bias", "classifier.weight"])
    );
    assert_eq!(fs::read(&parent_latest_path).unwrap(), parent_latest_bytes);
    assert_eq!(
        fs::read(&parent_checkpoint_path).unwrap(),
        parent_checkpoint_bytes
    );

    let child_directory = project.path().join(".model-maven/runs").join(&child_id);
    let child_metadata: Value =
        serde_json::from_slice(&fs::read(child_directory.join("run.json")).unwrap()).unwrap();
    assert_eq!(child_metadata["parent_run_id"], parent_id);
    let child_latest: Value =
        serde_json::from_slice(&fs::read(child_directory.join("checkpoints/latest.json")).unwrap())
            .unwrap();
    let child_checkpoint_path = child_directory
        .join("checkpoints")
        .join(child_latest["file"].as_str().unwrap());

    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let state_check = Command::new(python())
        .args(["-c", r#"
import json, sys, torch
parent = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
child = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
def steps(payload):
    return sorted(int(state["step"].item()) for state in payload["engine_state"]["optimizer_state"]["state"].values())
assert steps(parent) and set(steps(parent)) == {2}, steps(parent)
assert steps(child) and set(steps(child)) == {1}, steps(child)
expected_aliases = [["encoder.weight", "shared_encoder.weight"]]
assert parent["model_state_alias_groups"] == expected_aliases
assert child["model_state_alias_groups"] == expected_aliases
assert child["weight_migration"]["parent_run_id"]
print(json.dumps({"parent_optimizer_steps":steps(parent),"child_optimizer_steps":steps(child)}))
"#])
        .arg(&parent_checkpoint_path)
        .arg(&child_checkpoint_path)
        .env("PYTHONPATH", package)
        .output()
        .unwrap();
    assert!(
        state_check.status.success(),
        "{}",
        String::from_utf8_lossy(&state_check.stderr)
    );

    let (resume_output, resumed) = invoke_cli(
        project.path(),
        &[
            "train",
            "--resume",
            &child_id,
            "--additional-steps",
            "1",
            "--checkpoint-every",
            "1",
            "--allow-state-reset",
        ],
    );
    assert!(
        resume_output.status.success(),
        "{resumed:#}\n{}",
        String::from_utf8_lossy(&resume_output.stderr)
    );
    assert_eq!(resumed["data"]["completed_steps"], 2);
    assert_eq!(resumed["data"]["resumed_from_steps"], 1);
    assert_eq!(fs::read(&parent_latest_path).unwrap(), parent_latest_bytes);
    assert_eq!(
        fs::read(&parent_checkpoint_path).unwrap(),
        parent_checkpoint_bytes
    );

    let legacy_checkpoint = Command::new(python())
        .args([
            "-c",
            r#"
import hashlib, json, sys, torch
checkpoint_path, latest_path = sys.argv[1:]
payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
payload.pop("model_state_alias_groups", None)
torch.save(payload, checkpoint_path)
manifest = json.load(open(latest_path, encoding="utf-8"))
manifest["sha256"] = hashlib.sha256(open(checkpoint_path, "rb").read()).hexdigest()
json.dump(manifest, open(latest_path, "w", encoding="utf-8"), sort_keys=True)
"#,
        ])
        .arg(&parent_checkpoint_path)
        .arg(&parent_latest_path)
        .status()
        .unwrap();
    assert!(
        legacy_checkpoint.success(),
        "could not construct legacy checkpoint fixture"
    );
    let (legacy_output, legacy_report) = invoke_cli(
        project.path(),
        &[
            "train",
            "--model",
            "migration_fixture:build_model",
            "--batch",
            "migration_fixture:build_batch",
            "--step",
            "migration_fixture:training_step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
            "--migrate-from",
            &parent_id,
            "--migration-plan",
            "migration.json",
        ],
    );
    assert!(
        !legacy_output.status.success(),
        "legacy checkpoint migration unexpectedly succeeded: {legacy_report:#}"
    );
    assert_eq!(
        legacy_report["diagnostics"][0]["code"],
        "MIGRATION_SHARED_IDENTITY_UNKNOWN"
    );
    let legacy_failed_id = legacy_report["data"]["run_id"].as_str().unwrap();
    assert!(
        !project
            .path()
            .join(".model-maven/runs")
            .join(legacy_failed_id)
            .join("checkpoints/latest.json")
            .exists()
    );
    fs::write(&parent_checkpoint_path, &parent_checkpoint_bytes).unwrap();
    fs::write(&parent_latest_path, &parent_latest_bytes).unwrap();

    let source_v2_untied = source_v2.replace(
        "        self.shared_encoder.weight = self.encoder.weight\n",
        "",
    );
    fs::write(&source_path, source_v2_untied).unwrap();
    let alias_mismatch_plan = serde_json::json!({
        "schema_version":1,
        "preserve_compatible":[
            {"source_key":"encoder.weight","target_key":"encoder.weight","semantic_reason":"The encoder projection retains its role."},
            {"source_key":"encoder.bias","target_key":"encoder.bias","semantic_reason":"The encoder bias retains its role."},
            {"source_key":"shared_encoder.weight","target_key":"shared_encoder.weight","semantic_reason":"The second projection retains its role."},
            {"source_key":"shared_encoder.bias","target_key":"shared_encoder.bias","semantic_reason":"The second bias retains its role."}
        ],
        "reset_target":[],
        "drop_source":["classifier.weight","classifier.bias"]
    });
    fs::write(
        &plan_path,
        serde_json::to_vec(&alias_mismatch_plan).unwrap(),
    )
    .unwrap();
    let (alias_output, alias_report) = invoke_cli(
        project.path(),
        &[
            "train",
            "--model",
            "migration_fixture:build_model",
            "--batch",
            "migration_fixture:build_batch",
            "--step",
            "migration_fixture:training_step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
            "--migrate-from",
            &parent_id,
            "--migration-plan",
            "migration.json",
        ],
    );
    assert!(
        !alias_output.status.success(),
        "tied-to-untied migration unexpectedly succeeded: {alias_report:#}"
    );
    assert_eq!(
        alias_report["diagnostics"][0]["code"],
        "MIGRATION_SHARED_IDENTITY_MISMATCH"
    );
    let alias_failed_id = alias_report["data"]["run_id"].as_str().unwrap();
    assert!(
        !project
            .path()
            .join(".model-maven/runs")
            .join(alias_failed_id)
            .join("checkpoints/latest.json")
            .exists()
    );
    fs::write(&source_path, &source_v2).unwrap();

    let bad_plan = serde_json::json!({
        "schema_version":1,
        "preserve_compatible":[],
        "reset_target":[],
        "drop_source":[]
    });
    fs::write(&plan_path, serde_json::to_vec(&bad_plan).unwrap()).unwrap();
    let (failed_output, failed_report) = invoke_cli(
        project.path(),
        &[
            "train",
            "--model",
            "migration_fixture:build_model",
            "--batch",
            "migration_fixture:build_batch",
            "--step",
            "migration_fixture:training_step",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
            "--migrate-from",
            &parent_id,
            "--migration-plan",
            "migration.json",
        ],
    );
    assert!(
        !failed_output.status.success(),
        "bad migration unexpectedly succeeded: {failed_report:#}"
    );
    assert_eq!(
        failed_report["diagnostics"][0]["code"],
        "MIGRATION_TARGET_UNACCOUNTED"
    );
    let failed_id = failed_report["data"]["run_id"].as_str().unwrap();
    let failed_directory = project.path().join(".model-maven/runs").join(failed_id);
    assert!(!failed_directory.join("checkpoints/latest.json").exists());
    assert_eq!(fs::read(&parent_latest_path).unwrap(), parent_latest_bytes);
    assert_eq!(
        fs::read(&parent_checkpoint_path).unwrap(),
        parent_checkpoint_bytes
    );
}
