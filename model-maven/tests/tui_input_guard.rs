use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn run(root: &Path, args: &[&str]) -> Value {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--no-interactive", "--python", &python])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "{error}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[test]
fn input_guard_rejects_changed_or_unsupported_batches_before_forward_and_write() {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(python)
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        eprintln!("PyTorch unavailable; skipping real CLI integration");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("fixture.py"),
        r#"import torch
from torch import nn
from pathlib import Path
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 8)
    def forward(self, value):
        with Path('forward.log').open('a') as stream: stream.write('forward\n')
        return self.head(value)
def model(): return Model()
def stable(): return torch.ones(2, 48)
def different(): return torch.full((2, 48), 2.0)
def unsupported(): return torch.ones(48, 2).t()
class Dataset:
    def __len__(self): return 2
    def __getitem__(self, index): return torch.full((4,), float(index + 1))
    def sample_id(self, index): return f'sample-{index}'
def dataset(): return Dataset()
def dataset_model(): return nn.Linear(4, 4)
"#,
    )
    .unwrap();
    let root = project.path();
    let diagnosis = run(
        root,
        &[
            "model",
            "probe",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:stable",
        ],
    );
    assert_eq!(diagnosis["status"], "error", "{diagnosis:#}");
    let digest = diagnosis["data"]["batch_content"]["digest"]
        .as_str()
        .unwrap();
    assert_eq!(digest.len(), 64);
    let log = root.join("forward.log");
    fs::remove_file(&log).unwrap();
    let variant = root.join(".model-maven/variants/working.json");
    let invalid = run(
        root,
        &[
            "--if-input-content",
            "invalid",
            "model",
            "probe",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:stable",
        ],
    );
    assert_eq!(
        invalid["diagnostics"][0]["code"], "INPUT_CONTENT_INVALID",
        "{invalid:#}"
    );
    let unsupported_command = run(
        root,
        &[
            "--if-input-content",
            digest,
            "model",
            "inspect",
            "--model",
            "fixture:model",
        ],
    );
    assert_eq!(
        unsupported_command["diagnostics"][0]["code"], "INPUT_CONTENT_UNSUPPORTED",
        "{unsupported_command:#}"
    );
    assert!(!log.exists());
    assert!(!variant.exists());
    for (batch, code) in [
        ("fixture:different", "INPUT_CONTENT_MISMATCH"),
        ("fixture:unsupported", "INPUT_CONTENT_UNAVAILABLE"),
    ] {
        let report = run(
            root,
            &[
                "--if-variant-base",
                "absent",
                "--if-input-content",
                digest,
                "layer",
                "set",
                "model.head",
                "--param",
                "in_features=48",
                "--model",
                "fixture:model",
                "--batch",
                batch,
                "--require-forward",
            ],
        );
        assert_eq!(report["status"], "error", "{report:#}");
        assert_eq!(report["diagnostics"][0]["code"], code, "{report:#}");
        assert_eq!(report["data"]["forward_executed"], false, "{report:#}");
        assert!(!log.exists());
        assert!(!variant.exists());
    }
    let no_forward = run(
        root,
        &[
            "--if-input-content",
            digest,
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=48",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:stable",
        ],
    );
    assert_eq!(
        no_forward["diagnostics"][0]["code"], "INPUT_CONTENT_UNSUPPORTED",
        "{no_forward:#}"
    );
    assert!(!variant.exists());
    let preview = run(
        root,
        &[
            "--if-input-content",
            digest,
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=48",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:stable",
            "--require-forward",
            "--dry-run",
        ],
    );
    assert_eq!(preview["status"], "success", "{preview:#}");
    assert_eq!(
        preview["data"]["validation"]["forward_evidence"],
        "executed"
    );
    assert_eq!(
        preview["data"]["validation"]["batch_content"]["digest"],
        digest
    );
    assert!(log.exists());
    assert!(!variant.exists());
    fs::remove_file(&log).unwrap();
    let commit = run(
        root,
        &[
            "--if-variant-base",
            "absent",
            "--if-input-content",
            digest,
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=48",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:stable",
            "--require-forward",
        ],
    );
    assert_eq!(commit["status"], "success", "{commit:#}");
    assert!(variant.exists());
    assert!(log.exists());
    let committed_token = commit["data"]["variant_committed"]["token"]
        .as_str()
        .unwrap();
    assert_eq!(commit["data"]["variant_committed"]["name"], "working");
    assert_eq!(
        commit["data"]["variant_committed"]["algorithm"],
        "canonical-json-sha256-v1"
    );
    fs::remove_file(&log).unwrap();
    let recheck = run(
        root,
        &[
            "--variant",
            "working",
            "--if-input-content",
            digest,
            "--if-variant-base",
            committed_token,
            "model",
            "probe",
            "--batch",
            "fixture:stable",
        ],
    );
    assert_eq!(recheck["status"], "success", "{recheck:#}");
    assert_eq!(recheck["data"]["batch_content"]["digest"], digest);
    assert_ne!(recheck["data"]["evidence_cache"]["status"], "reused");
    assert!(log.exists());
    fs::remove_file(&log).unwrap();
    let original = fs::read(&variant).unwrap();
    let mut altered: Value = serde_json::from_slice(&original).unwrap();
    altered["external_change"] = true.into();
    fs::write(&variant, serde_json::to_vec(&altered).unwrap()).unwrap();
    let stale = run(
        root,
        &[
            "--variant",
            "working",
            "--if-variant-base",
            committed_token,
            "--if-input-content",
            digest,
            "model",
            "probe",
            "--batch",
            "fixture:stable",
        ],
    );
    assert_eq!(
        stale["diagnostics"][0]["code"], "VARIANT_BASE_CONFLICT",
        "{stale:#}"
    );
    assert!(!log.exists());
    assert_ne!(fs::read(&variant).unwrap(), original);
    fs::write(&variant, original).unwrap();

    let common = [
        "--seed",
        "7",
        "model",
        "probe",
        "--model",
        "fixture:dataset_model",
        "--batch",
        "fixture:dataset",
        "--batch-fingerprint",
        "dataset-v1",
        "--batch-binding",
        "single",
    ];
    let first = run(root, &common);
    assert_eq!(
        first["data"]["evidence_cache"]["status"], "stored",
        "{first:#}"
    );
    let dataset_digest = first["data"]["batch_content"]["digest"].as_str().unwrap();
    let cached = run(root, &common);
    assert_eq!(
        cached["data"]["evidence_cache"]["status"], "reused",
        "{cached:#}"
    );
    let guarded = run(
        root,
        &[&["--if-input-content", dataset_digest][..], &common].concat(),
    );
    assert_eq!(guarded["status"], "success", "{guarded:#}");
    assert_ne!(
        guarded["data"]["evidence_cache"]["status"], "reused",
        "{guarded:#}"
    );
    assert_eq!(guarded["data"]["batch_content"]["digest"], dataset_digest);
    let mismatch = run(
        root,
        &[&["--if-input-content", &"0".repeat(64)][..], &common].concat(),
    );
    assert_eq!(
        mismatch["diagnostics"][0]["code"], "INPUT_CONTENT_MISMATCH",
        "{mismatch:#}"
    );
    assert_eq!(mismatch["data"]["forward_executed"], false);
}
