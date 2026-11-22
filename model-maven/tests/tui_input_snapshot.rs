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
fn captured_failed_input_is_replayed_without_calling_batch_factory() {
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
    let root = project.path();
    fs::write(
        root.join("fixture.py"),
        r#"import torch
from torch import nn
from pathlib import Path
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 8)
    def forward(self, value): return self.head(value)
def model(): return Model()
def changing():
    path = Path('counter.txt')
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    return torch.full((2, 48), float(count))
def unsupported(): return torch.ones(48, 2).t()
"#,
    )
    .unwrap();
    let captured = run(
        root,
        &[
            "--capture-input-to",
            "input.snapshot",
            "model",
            "probe",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:changing",
        ],
    );
    assert_eq!(
        captured["diagnostics"][0]["code"], "FORWARD_FAILED",
        "{captured:#}"
    );
    assert_eq!(
        captured["data"]["batch_snapshot"]["status"], "captured",
        "{captured:#}"
    );
    let digest = captured["data"]["batch_content"]["digest"]
        .as_str()
        .unwrap();
    let snapshot = root.join("input.snapshot");
    assert!(snapshot.exists());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&snapshot).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    let preview = run(
        root,
        &[
            "--if-input-content",
            digest,
            "--replay-input-from",
            "input.snapshot",
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=48",
            "--dry-run",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:changing",
            "--require-forward",
        ],
    );
    assert_eq!(preview["status"], "success", "{preview:#}");
    assert_eq!(
        preview["data"]["validation"]["batch_snapshot"]["status"], "replayed",
        "{preview:#}"
    );
    let commit = run(
        root,
        &[
            "--if-variant-base",
            "absent",
            "--if-input-content",
            digest,
            "--replay-input-from",
            "input.snapshot",
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=48",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:changing",
            "--require-forward",
        ],
    );
    assert_eq!(commit["status"], "success", "{commit:#}");
    let token = commit["data"]["variant_committed"]["token"]
        .as_str()
        .unwrap();
    let recheck = run(
        root,
        &[
            "--variant",
            "working",
            "--if-variant-base",
            token,
            "--if-input-content",
            digest,
            "--replay-input-from",
            "input.snapshot",
            "model",
            "probe",
            "--batch",
            "fixture:changing",
        ],
    );
    assert_eq!(recheck["status"], "success", "{recheck:#}");
    assert_eq!(recheck["data"]["batch_snapshot"]["status"], "replayed");
    assert_eq!(fs::read_to_string(root.join("counter.txt")).unwrap(), "1");
    let original_variant = fs::read(root.join(".model-maven/variants/working.json")).unwrap();
    fs::write(&snapshot, b"{}".as_slice()).unwrap();
    let invalid = run(
        root,
        &[
            "--variant",
            "working",
            "--if-variant-base",
            token,
            "--if-input-content",
            digest,
            "--replay-input-from",
            "input.snapshot",
            "model",
            "probe",
            "--batch",
            "fixture:changing",
        ],
    );
    assert_eq!(
        invalid["diagnostics"][0]["code"], "INPUT_SNAPSHOT_INVALID",
        "{invalid:#}"
    );
    assert_eq!(
        fs::read(root.join(".model-maven/variants/working.json")).unwrap(),
        original_variant
    );
    assert_eq!(fs::read_to_string(root.join("counter.txt")).unwrap(), "1");
    let unsupported = run(
        root,
        &[
            "--capture-input-to",
            "unsupported.snapshot",
            "model",
            "probe",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:unsupported",
        ],
    );
    assert_eq!(
        unsupported["diagnostics"][0]["code"], "FORWARD_FAILED",
        "{unsupported:#}"
    );
    assert_eq!(
        unsupported["data"]["batch_snapshot"]["status"],
        "unsupported"
    );
    assert!(!root.join("unsupported.snapshot").exists());
    let invalid_edit = run(
        root,
        &[
            "--variant",
            "working",
            "--if-variant-base",
            token,
            "--if-input-content",
            digest,
            "--replay-input-from",
            "input.snapshot",
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=48",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:changing",
            "--require-forward",
        ],
    );
    assert_eq!(
        invalid_edit["diagnostics"][0]["code"], "INPUT_SNAPSHOT_INVALID",
        "{invalid_edit:#}"
    );
    assert_eq!(
        fs::read(root.join(".model-maven/variants/working.json")).unwrap(),
        original_variant
    );
    let existing = run(
        root,
        &[
            "--capture-input-to",
            "input.snapshot",
            "model",
            "probe",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:changing",
        ],
    );
    assert_eq!(
        existing["diagnostics"][0]["code"], "INPUT_SNAPSHOT_WRITE_FAILED",
        "{existing:#}"
    );
    assert_eq!(fs::read(&snapshot).unwrap(), b"{}");
    let no_digest = run(
        root,
        &[
            "--replay-input-from",
            "input.snapshot",
            "model",
            "probe",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:changing",
        ],
    );
    assert_eq!(
        no_digest["diagnostics"][0]["code"], "INPUT_SNAPSHOT_UNSUPPORTED",
        "{no_digest:#}"
    );
}
