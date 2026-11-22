use serde_json::Value;
use sha2::Digest;
use std::{fs, process::Command};

#[test]
fn init_reports_a_stable_revision_without_changing_the_legacy_config_shape() {
    let project = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "init", "--model", "example_model:build_model"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "success");
    let config_path = project.path().join(".model-maven/project.json");
    let config: Value = serde_json::from_slice(&fs::read(config_path).unwrap()).unwrap();
    assert_eq!(config["schema_version"], 1);
    assert!(config.get("config_revision").is_none());

    let digest = sha2::Sha256::digest(serde_json::to_vec(&config).unwrap());
    let expected_revision = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(report["data"]["config_revision"], expected_revision);
}

#[test]
fn unsupported_project_config_schema_is_rejected_with_recovery_guidance() {
    let project = tempfile::tempdir().unwrap();
    let metadata = project.path().join(".model-maven");
    fs::create_dir_all(&metadata).unwrap();
    let config_path = metadata.join("project.json");
    let original = br#"{"schema_version":2,"model":"example_model:build_model","batch":null,"batch_binding":null}"#;
    fs::write(&config_path, original).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "model", "inspect"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "error");
    assert_eq!(report["diagnostics"][0]["code"], "PROJECT_CONFIG_INVALID");
    assert!(
        report["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains("schema_version=2")
    );
    assert!(
        report["diagnostics"][0]["hint"]
            .as_str()
            .unwrap()
            .contains("兼容版本")
    );
    assert_eq!(fs::read(&config_path).unwrap(), original);
}
