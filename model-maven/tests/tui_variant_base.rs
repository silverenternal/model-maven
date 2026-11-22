use serde_json::{Value, json};
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
fn variant_preview_token_guards_locked_commit_and_rejects_stale_content() {
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
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(32, 8)
    def forward(self, value): return self.head(value)
def model(): return Model()
def batch(): return torch.ones(2, 48)
def different_batch(): return torch.full((2, 48), 2.0)
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
            "fixture:batch",
        ],
    );
    assert_eq!(diagnosis["status"], "error", "{diagnosis:#}");
    assert_eq!(
        diagnosis["data"]["batch_content"]["status"], "available",
        "{diagnosis:#}"
    );
    let different = run(
        root,
        &[
            "model",
            "probe",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:different_batch",
        ],
    );
    assert_eq!(different["status"], "error");
    assert_ne!(
        different["data"]["batch_content"]["digest"],
        diagnosis["data"]["batch_content"]["digest"]
    );
    let preview_args = [
        "layer",
        "set",
        "model.head",
        "--param",
        "in_features=48",
        "--model",
        "fixture:model",
        "--batch",
        "fixture:batch",
        "--require-forward",
        "--dry-run",
    ];
    let preview = run(root, &preview_args);
    assert_eq!(preview["status"], "success", "{preview:#}");
    assert_eq!(preview["data"]["variant_base"]["token"], "absent");
    assert_eq!(preview["data"]["committed"], false);
    assert_eq!(
        preview["data"]["validation"]["batch_content"]["digest"],
        diagnosis["data"]["batch_content"]["digest"]
    );
    let variant = root.join(".model-maven/variants/working.json");
    assert!(!variant.exists());

    let competing = run(
        root,
        &[
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=40",
            "--model",
            "fixture:model",
        ],
    );
    assert_eq!(competing["status"], "success", "{competing:#}");
    let before = fs::read(&variant).unwrap();
    let conflict = run(
        root,
        &[
            "--if-variant-base",
            "absent",
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=48",
            "--model",
            "fixture:model",
        ],
    );
    assert_eq!(conflict["status"], "error", "{conflict:#}");
    assert_eq!(conflict["diagnostics"][0]["code"], "VARIANT_BASE_CONFLICT");
    assert_eq!(fs::read(&variant).unwrap(), before);

    let fresh = run(root, &preview_args);
    assert_eq!(fresh["status"], "success", "{fresh:#}");
    assert_eq!(
        fresh["data"]["variant_base"]["algorithm"],
        "canonical-json-sha256-v1"
    );
    let token = fresh["data"]["variant_base"]["token"].as_str().unwrap();
    assert_eq!(token.len(), 64);
    assert_eq!(fs::read(&variant).unwrap(), before);
    let commit = run(
        root,
        &[
            "--if-variant-base",
            token,
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=48",
            "--model",
            "fixture:model",
            "--batch",
            "fixture:batch",
            "--require-forward",
        ],
    );
    assert_eq!(commit["status"], "success", "{commit:#}");
    assert_eq!(commit["data"]["committed"], true);
    assert_eq!(commit["data"]["variant_base"]["token"], token);
    assert_ne!(fs::read(&variant).unwrap(), before);

    let concurrent_preview = run(root, &preview_args);
    let concurrent_token = concurrent_preview["data"]["variant_base"]["token"]
        .as_str()
        .unwrap()
        .to_owned();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let clients: Vec<_> = (0..2)
        .map(|_| {
            let path = root.to_owned();
            let token = concurrent_token.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                run(
                    &path,
                    &[
                        "--if-variant-base",
                        &token,
                        "layer",
                        "set",
                        "model.head",
                        "--param",
                        "in_features=48",
                        "--model",
                        "fixture:model",
                    ],
                )
            })
        })
        .collect();
    let outcomes: Vec<_> = clients
        .into_iter()
        .map(|client| client.join().unwrap())
        .collect();
    assert_eq!(
        outcomes
            .iter()
            .filter(|item| item["status"] == "success")
            .count(),
        1,
        "{outcomes:#?}"
    );
    let rejected = outcomes
        .iter()
        .find(|item| item["status"] == "error")
        .unwrap();
    assert!(
        matches!(
            rejected["diagnostics"][0]["code"].as_str(),
            Some("VARIANT_LOCKED" | "VARIANT_BASE_CONFLICT")
        ),
        "{rejected:#}"
    );

    let current = run(root, &preview_args);
    let current_token = current["data"]["variant_base"]["token"].as_str().unwrap();
    let mut altered: Value = serde_json::from_slice(&fs::read(&variant).unwrap()).unwrap();
    altered["unexpected_change"] = json!(true);
    fs::write(&variant, serde_json::to_vec(&altered).unwrap()).unwrap();
    let tampered = fs::read(&variant).unwrap();
    let conflict = run(
        root,
        &[
            "--if-variant-base",
            current_token,
            "layer",
            "set",
            "model.head",
            "--param",
            "in_features=48",
            "--model",
            "fixture:model",
        ],
    );
    assert_eq!(
        conflict["diagnostics"][0]["code"], "VARIANT_BASE_CONFLICT",
        "{conflict:#}"
    );
    assert_eq!(fs::read(&variant).unwrap(), tampered);
    let unsupported = run(
        root,
        &[
            "--if-variant-base",
            "absent",
            "model",
            "inspect",
            "--model",
            "fixture:model",
        ],
    );
    assert_eq!(
        unsupported["diagnostics"][0]["code"], "VARIANT_BASE_UNSUPPORTED",
        "{unsupported:#}"
    );
    let mut dry_run_with_condition = vec!["--if-variant-base", "absent"];
    dry_run_with_condition.extend(preview_args);
    let unsupported = run(root, &dry_run_with_condition);
    assert_eq!(
        unsupported["diagnostics"][0]["code"], "VARIANT_BASE_UNSUPPORTED",
        "{unsupported:#}"
    );
    let unsupported = run(
        root,
        &[
            "--if-variant-base",
            "absent",
            "layer",
            "undo",
            "--target",
            "source",
        ],
    );
    assert_eq!(
        unsupported["diagnostics"][0]["code"], "VARIANT_BASE_UNSUPPORTED",
        "{unsupported:#}"
    );
}
