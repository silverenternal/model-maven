use serde_json::Value;
use std::{fs, io::Write, path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

fn invoke(project: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_m-model"));
    command.args(["--json", "--python", &python()]);
    command.args(args);
    let output = command
        .current_dir(project)
        .output()
        .expect("start model evidence command");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report, got {:?}: {error}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

fn fixture() -> tempfile::TempDir {
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("fixture.py"),
        r#"import torch
from torch import nn

class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
    def forward(self, left, right):
        return self.layer(left), self.layer(right)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = Shared()
    def forward(self, left, right):
        return self.shared(left, right)

class Dataset:
    def __len__(self): return 2
    def __getitem__(self, index): return torch.full((4,), float(index + 1))
    def sample_id(self, index): return f"sample-{index}"

class DynamicModel(nn.Module):
    def forward(self, values):
        if values.sum().item() > 0: return values + 1
        return values - 1

def build_model(): return Model()
def build_dataset_model(): return nn.Linear(4, 4)
def build_dynamic_dataset_model(): return DynamicModel()
def build_dataset(token=None): return Dataset()
def batch(): return {"args": [torch.ones(2, 4), torch.ones(3, 4)], "kwargs": {}}
"#,
    )
    .unwrap();
    project
}

#[test]
fn local_probe_replays_only_the_selected_observed_invocation_on_copied_state() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    let (output, report) = invoke(
        project.path(),
        &[
            "model",
            "probe",
            "model.shared.layer",
            "--model",
            "fixture:build_model",
            "--batch",
            "fixture:batch",
            "--call",
            "2",
            "--local",
        ],
    );
    assert!(output.status.success(), "{report:#}");
    let local = &report["data"]["local_probe_reports"][0];
    assert_eq!(local["status"], "observed-runtime");
    assert_eq!(local["invocation_index"], 2);
    assert_eq!(
        local["inputs"]["items"]["args"]["items"][0]["shape"],
        serde_json::json!([3, 4])
    );
    assert_eq!(local["outputs"]["shape"], serde_json::json!([3, 4]));
    assert!(
        local["isolation"]
            .as_str()
            .unwrap()
            .contains("cloned input tensors")
    );
    assert_eq!(report["data"]["evidence_cache"]["status"], "not_stored");
}

#[test]
fn stable_evidence_reuses_only_matching_inputs_and_ignores_run_budget() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    let common = [
        "--seed",
        "7",
        "model",
        "probe",
        "--model",
        "fixture:build_dataset_model",
        "--batch",
        "fixture:build_dataset",
        "--batch-fingerprint",
        "dataset-v1",
        "--batch-binding",
        "single",
        "--batch-param",
        "token=\"private-token-value\"",
    ];
    let (first_output, first) = invoke(project.path(), &common);
    assert!(first_output.status.success(), "{first:#}");
    assert_eq!(
        first["data"]["evidence_cache"]["status"], "stored",
        "{first:#}"
    );
    let cache_dir = project.path().join(".model-maven/cache/model-evidence");
    let cache_files: Vec<_> = fs::read_dir(&cache_dir)
        .unwrap()
        .map(Result::unwrap)
        .filter(|entry| entry.file_name() != "latest.json")
        .collect();
    assert_eq!(cache_files.len(), 1);
    let cached = fs::read_to_string(cache_files[0].path()).unwrap();
    assert!(!cached.contains("private-token-value"));

    let (budget_output, budget_report) = invoke(
        project.path(),
        &[&["--timeout", "90"][..], &common].concat(),
    );
    assert!(budget_output.status.success(), "{budget_report:#}");
    let records: Vec<_> = fs::read_dir(&cache_dir)
        .unwrap()
        .map(|entry| {
            serde_json::from_str::<Value>(&fs::read_to_string(entry.unwrap().path()).unwrap())
                .unwrap()
        })
        .collect();
    let identity_differences: Vec<_> = records
        .windows(2)
        .flat_map(|pair| {
            pair[0]["identity"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(key, value)| pair[1]["identity"][*key] != **value)
                .map(|(key, _)| key.clone())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        budget_report["data"]["evidence_cache"]["status"], "reused",
        "identity changed fields: {identity_differences:?}; {budget_report:#}"
    );

    let changed = [
        "--seed",
        "7",
        "model",
        "probe",
        "--model",
        "fixture:build_dataset_model",
        "--batch",
        "fixture:build_dataset",
        "--batch-fingerprint",
        "dataset-v2",
        "--batch-binding",
        "single",
        "--batch-param",
        "token=\"private-token-value\"",
    ];
    let (changed_output, changed_report) = invoke(project.path(), &changed);
    assert!(changed_output.status.success(), "{changed_report:#}");
    assert_eq!(changed_report["data"]["evidence_cache"]["status"], "stored");
    assert!(
        changed_report["data"]["evidence_cache"]["invalidated"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "dataset_split_or_sample_identity_changed")
    );
    assert_ne!(
        first["data"]["evidence_cache"]["key"],
        changed_report["data"]["evidence_cache"]["key"]
    );

    let previous_key = changed_report["data"]["evidence_cache"]["key"]
        .as_str()
        .unwrap()
        .to_owned();
    fs::OpenOptions::new()
        .append(true)
        .open(project.path().join("fixture.py"))
        .unwrap()
        .write_all(b"\n# source revision changes cached evidence\n")
        .unwrap();
    let (source_output, source_report) = invoke(project.path(), &changed);
    assert!(source_output.status.success(), "{source_report:#}");
    assert_eq!(source_report["data"]["evidence_cache"]["status"], "stored");
    assert_ne!(source_report["data"]["evidence_cache"]["key"], previous_key);
    assert!(
        source_report["data"]["evidence_cache"]["invalidated"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "project_or_component_source_changed")
    );
    assert_eq!(
        source_report["data"]["evidence_cache"]["branch_reuse"]["status"],
        "whole_probe"
    );
    assert!(
        source_report["data"]["evidence_cache"]["branch_reuse"]["expanded_scope_reason"]
            .as_str()
            .unwrap()
            .contains("源码变化无法归属")
    );

    let reseeded = [
        "--seed",
        "8",
        "model",
        "probe",
        "--model",
        "fixture:build_dataset_model",
        "--batch",
        "fixture:build_dataset",
        "--batch-fingerprint",
        "dataset-v2",
        "--batch-binding",
        "single",
        "--batch-param",
        "token=\"private-token-value\"",
    ];
    let (weight_output, weight_report) = invoke(project.path(), &reseeded);
    assert!(weight_output.status.success(), "{weight_report:#}");
    assert_eq!(weight_report["data"]["evidence_cache"]["status"], "stored");
    assert_ne!(
        weight_report["data"]["evidence_cache"]["key"],
        source_report["data"]["evidence_cache"]["key"]
    );
}

#[test]
fn unknown_dynamic_paths_expand_to_fresh_runtime_probe_and_are_not_reused() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    let args = [
        "--seed",
        "7",
        "model",
        "probe",
        "--model",
        "fixture:build_dynamic_dataset_model",
        "--batch",
        "fixture:build_dataset",
        "--batch-fingerprint",
        "dataset-v1",
        "--batch-binding",
        "single",
    ];
    let (output, report) = invoke(project.path(), &args);
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["data"]["shape_inference"]["status"], "unsupported");
    assert_eq!(report["data"]["evidence_cache"]["status"], "not_stored");
    assert_eq!(
        report["data"]["evidence_cache"]["branch_reuse"]["status"],
        "whole_probe"
    );
    assert!(
        report["data"]["evidence_cache"]["branch_reuse"]["expanded_scope_reason"]
            .as_str()
            .unwrap()
            .contains("动态或不支持")
    );
    assert!(
        report["data"]["evidence_cache"]["reason"]
            .as_str()
            .unwrap()
            .contains("static graph evidence is unavailable")
    );
    assert!(
        report["data"]["coverage"]
            .as_str()
            .unwrap()
            .contains("不证明所有分支")
    );
}

#[test]
fn source_change_revalidates_only_affected_branch_and_its_ancestors() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    fs::write(
        project.path().join("branch_a.py"),
        r#"import torch
from torch import nn
class BranchA(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
    def forward(self, values): return self.linear(values)
"#,
    )
    .unwrap();
    fs::write(
        project.path().join("branch_b.py"),
        r#"import torch
from torch import nn
class BranchB(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
    def forward(self, values): return self.linear(values)
"#,
    )
    .unwrap();
    let mut source = fs::read_to_string(project.path().join("fixture.py")).unwrap();
    source.push_str(
        r#"
from branch_a import BranchA
from branch_b import BranchB
class BranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = BranchA()
        self.b = BranchB()
    def forward(self, values): return self.a(values) + self.b(values)
def build_branch_model(): return BranchModel()
"#,
    );
    fs::write(project.path().join("fixture.py"), source).unwrap();
    let args = [
        "--seed",
        "7",
        "model",
        "probe",
        "--model",
        "fixture:build_branch_model",
        "--batch",
        "fixture:build_dataset",
        "--batch-fingerprint",
        "dataset-v1",
        "--batch-binding",
        "single",
    ];
    let (first_output, first) = invoke(project.path(), &args);
    assert!(first_output.status.success(), "{first:#}");
    assert_eq!(first["data"]["evidence_cache"]["status"], "stored");
    assert_eq!(
        first["data"]["evidence_cache"]["branch_reuse"]["status"],
        "whole_probe"
    );

    let branch_a_path = project.path().join("branch_a.py");
    let branch_a = fs::read_to_string(&branch_a_path).unwrap();
    assert!(branch_a.contains("return self.linear(values)"));
    fs::write(
        branch_a_path,
        branch_a.replace(
            "return self.linear(values)",
            "return self.linear(values) + 1",
        ),
    )
    .unwrap();
    let (second_output, second) = invoke(project.path(), &args);
    assert!(second_output.status.success(), "{second:#}");
    let reuse = &second["data"]["evidence_cache"]["branch_reuse"];
    assert_eq!(reuse["status"], "partial_reuse", "{reuse:#}");
    assert!(reuse["root_forward_executed"].as_bool().unwrap());
    assert_eq!(reuse["tensor_reuse"], false);
    let reused = reuse["reused"].as_array().unwrap();
    assert!(reused.iter().any(|call| call["path"] == "model.b"));
    assert!(reused.iter().any(|call| call["path"] == "model.b.linear"));
    assert!(!reused.iter().any(|call| call["path"] == "model"));
    assert!(!reused.iter().any(|call| call["path"] == "model.a"));
    assert!(
        reuse["revalidated"]
            .as_array()
            .unwrap()
            .iter()
            .any(|call| call["path"] == "model.a")
    );
    let cache = fs::read_to_string(
        project
            .path()
            .join(".model-maven/cache/model-evidence")
            .join(format!(
                "{}.json",
                second["data"]["evidence_cache"]["key"].as_str().unwrap()
            )),
    )
    .unwrap();
    assert!(!cache.contains("branch_a.py"));
}

#[test]
fn repeated_shared_module_calls_fail_closed_after_source_change() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = fixture();
    let fixture_path = project.path().join("fixture.py");
    let mut source = fs::read_to_string(&fixture_path).unwrap();
    source.push_str(
        r#"
class RepeatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
    def forward(self, values): return self.layer(values) + self.layer(values)
def build_repeated_model(): return RepeatedModel()
"#,
    );
    fs::write(&fixture_path, source).unwrap();
    let args = [
        "--seed",
        "7",
        "model",
        "probe",
        "--model",
        "fixture:build_repeated_model",
        "--batch",
        "fixture:build_dataset",
        "--batch-fingerprint",
        "dataset-v1",
        "--batch-binding",
        "single",
    ];
    let (first_output, first) = invoke(project.path(), &args);
    assert!(first_output.status.success(), "{first:#}");

    fs::OpenOptions::new()
        .append(true)
        .open(&fixture_path)
        .unwrap()
        .write_all(b"\n# source changed\n")
        .unwrap();
    let (second_output, second) = invoke(project.path(), &args);
    assert!(second_output.status.success(), "{second:#}");
    let reuse = &second["data"]["evidence_cache"]["branch_reuse"];
    assert_eq!(reuse["status"], "whole_probe");
    assert!(
        reuse["expanded_scope_reason"]
            .as_str()
            .unwrap()
            .contains("重复调用或共享别名")
    );
    assert!(reuse["root_forward_executed"].as_bool().unwrap());
    assert_eq!(reuse["tensor_reuse"], false);
}
