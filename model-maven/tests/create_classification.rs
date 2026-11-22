use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn run(root: &Path, args: &[&str]) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(root)
        .args(["--json", "--no-interactive"])
        .args(args)
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        output.status.success(),
        report["status"] == "success",
        "{report}"
    );
    report
}

#[test]
fn preview_commit_and_conflict_leave_user_files_untouched() {
    let root = tempfile::tempdir().unwrap();
    for class in ["cats", "dogs"] {
        fs::create_dir_all(root.path().join("data").join(class)).unwrap();
        fs::write(
            root.path().join("data").join(class).join("image.png"),
            b"placeholder",
        )
        .unwrap();
    }
    let preview = run(
        root.path(),
        &["create", "--preset", "tiny-cnn", "--data", "data"],
    );
    assert_eq!(preview["status"], "success", "{preview}");
    assert_eq!(
        preview["data"]["plan"]["classes"],
        serde_json::json!(["cats", "dogs"])
    );
    assert!(!root.path().join("classification.py").exists());
    let revision = preview["data"]["plan_revision"].as_str().unwrap();
    fs::create_dir(root.path().join("data/birds")).unwrap();
    fs::write(root.path().join("data/birds/image.png"), b"placeholder").unwrap();
    let changed = run(
        root.path(),
        &[
            "create",
            "--preset",
            "tiny-cnn",
            "--data",
            "data",
            "--apply",
            "--if-plan",
            revision,
        ],
    );
    assert_eq!(changed["diagnostics"][0]["code"], "CREATE_PLAN_CONFLICT");
    fs::remove_dir_all(root.path().join("data/birds")).unwrap();
    let stale = run(
        root.path(),
        &[
            "create",
            "--preset",
            "tiny-cnn",
            "--data",
            "data",
            "--apply",
            "--if-plan",
            "stale",
        ],
    );
    assert_eq!(stale["diagnostics"][0]["code"], "CREATE_PLAN_CONFLICT");
    assert!(!root.path().join("classification.py").exists());
    let created = run(
        root.path(),
        &[
            "create",
            "--preset",
            "tiny-cnn",
            "--data",
            "data",
            "--apply",
            "--if-plan",
            revision,
        ],
    );
    assert_eq!(created["data"]["committed"], true, "{created}");
    assert!(
        fs::read_to_string(root.path().join("classification.py"))
            .unwrap()
            .contains("NUM_CLASSES = 2")
    );
    let config: Value =
        serde_json::from_slice(&fs::read(root.path().join(".model-maven/project.json")).unwrap())
            .unwrap();
    assert_eq!(config["model"], "classification:model");
    let source = fs::read(root.path().join("classification.py")).unwrap();
    let repeat = run(
        root.path(),
        &[
            "create",
            "--preset",
            "tiny-cnn",
            "--data",
            "data",
            "--apply",
            "--if-plan",
            revision,
        ],
    );
    assert_eq!(repeat["diagnostics"][0]["code"], "CREATE_TARGET_CONFLICT");
    assert_eq!(
        fs::read(root.path().join("classification.py")).unwrap(),
        source
    );
}
