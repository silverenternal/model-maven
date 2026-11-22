use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn run(root: &Path, args: &[&str]) -> (bool, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(root)
        .args(["--json", "--no-interactive"])
        .args(args)
        .output()
        .unwrap();
    (
        output.status.success(),
        serde_json::from_slice(&output.stdout).unwrap(),
    )
}

#[test]
fn batch_change_is_previewed_and_conditionally_preserves_other_config() {
    let project = tempfile::tempdir().unwrap();
    let (ok, _) = run(
        project.path(),
        &[
            "init",
            "--model",
            "model:build",
            "--batch",
            "data:old",
            "--inject",
            "encoder=components:build",
        ],
    );
    assert!(ok);
    let path = project.path().join(".model-maven/project.json");
    let original = fs::read(&path).unwrap();
    let (ok, preview) = run(
        project.path(),
        &[
            "config",
            "set-batch",
            "--batch",
            "data:new",
            "--batch-binding",
            "envelope",
        ],
    );
    assert!(ok, "{preview}");
    assert_eq!(preview["data"]["committed"], false);
    assert_eq!(fs::read(&path).unwrap(), original);
    let revision = preview["data"]["from_revision"].as_str().unwrap();
    let (ok, stale) = run(
        project.path(),
        &[
            "config",
            "set-batch",
            "--batch",
            "data:new",
            "--apply",
            "--if-current",
            &"0".repeat(64),
        ],
    );
    assert!(!ok, "{stale}");
    assert_eq!(stale["diagnostics"][0]["code"], "PROJECT_CONFIG_CONFLICT");
    assert_eq!(fs::read(&path).unwrap(), original);
    let (ok, applied) = run(
        project.path(),
        &[
            "config",
            "set-batch",
            "--batch",
            "data:new",
            "--batch-binding",
            "envelope",
            "--apply",
            "--if-current",
            revision,
        ],
    );
    assert!(ok, "{applied}");
    assert_eq!(applied["data"]["committed"], true);
    let config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(config["batch"], "data:new");
    assert_eq!(config["batch_binding"], "envelope");
    assert_eq!(config["model"], "model:build");
    assert_eq!(config["model_injections"]["encoder"], "components:build");
    let (ok, invalid) = run(
        project.path(),
        &["config", "set-batch", "--batch", "invalid"],
    );
    assert!(!ok);
    assert_eq!(
        invalid["diagnostics"][0]["code"],
        "BATCH_ENTRYPOINT_INVALID"
    );
}

#[test]
fn batch_parameters_are_redacted_in_preview_history_and_restore() {
    let project = tempfile::tempdir().unwrap();
    assert!(
        run(
            project.path(),
            &["init", "--model", "model:build", "--batch", "data:old"]
        )
        .0
    );
    let args = [
        "config",
        "set-batch",
        "--batch",
        "data:new",
        "--batch-param",
        "path=\"images.csv\"",
        "--batch-param",
        "api_key=\"sample-secret\"",
    ];
    let (ok, preview) = run(project.path(), &args);
    assert!(ok, "{preview}");
    assert!(!preview.to_string().contains("sample-secret"));
    assert_eq!(preview["data"]["batch_params"]["api_key"], "[REDACTED]");
    let mut apply = args.to_vec();
    apply.extend([
        "--apply",
        "--if-current",
        preview["data"]["from_revision"].as_str().unwrap(),
    ]);
    let (ok, saved) = run(project.path(), &apply);
    assert!(ok, "{saved}");
    assert!(!saved.to_string().contains("sample-secret"));
    let secret_revision = saved["data"]["config_revision"].as_str().unwrap();
    let (ok, next_preview) = run(
        project.path(),
        &["config", "set-batch", "--batch", "data:other"],
    );
    assert!(ok);
    let (ok, _) = run(
        project.path(),
        &[
            "config",
            "set-batch",
            "--batch",
            "data:other",
            "--apply",
            "--if-current",
            next_preview["data"]["from_revision"].as_str().unwrap(),
        ],
    );
    assert!(ok);
    let (ok, history) = run(project.path(), &["config", "history"]);
    assert!(ok);
    assert!(!history.to_string().contains("sample-secret"));
    let (ok, restore) = run(
        project.path(),
        &[
            "config",
            "restore",
            secret_revision,
            "--field",
            "batch_params",
        ],
    );
    assert!(ok, "{restore}");
    assert!(!restore.to_string().contains("sample-secret"));
    assert_eq!(
        restore["data"]["fields"][0]["restored"]["api_key"],
        "[REDACTED]"
    );
}
