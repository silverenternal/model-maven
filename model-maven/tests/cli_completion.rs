use std::process::Command;
use tempfile::tempdir;

#[test]
fn completion_command_emits_dynamic_shell_setup_without_running_python() {
    let project = tempdir().unwrap();
    for shell in ["bash", "elvish", "fish", "powershell", "zsh"] {
        let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args([
                "--python",
                "missing-python-is-not-used",
                "completions",
                shell,
            ])
            .current_dir(project.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{shell}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty(), "{shell}");
        let setup = String::from_utf8(output.stdout).unwrap();
        assert!(setup.contains("m-model"), "{shell}: {setup}");
        assert!(setup.contains("COMPLETE"), "{shell}: {setup}");
    }
}

#[test]
fn dynamic_component_completion_does_not_initialize_python() {
    let project = tempdir().unwrap();
    let manifest_directory = project
        .path()
        .join(".model-maven/components/example/encoder");
    std::fs::create_dir_all(&manifest_directory).unwrap();
    std::fs::write(
        manifest_directory.join("1.2.3.json"),
        serde_json::json!({
            "schema_version": 1,
            "namespace": "example",
            "component_id": "example/encoder",
            "component_version": "1.2.3",
            "extension_api_version": 1,
            "kind": "layer",
            "entrypoint": "must_not_be_imported:factory",
            "description": "test component",
            "params_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": false},
            "capabilities": {"construct": "supported"}
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .env("COMPLETE", "bash")
        .env("_CLAP_COMPLETE_INDEX", "3")
        .args(["--", "m-model", "component", "show", "example/enc"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let candidates = String::from_utf8(output.stdout).unwrap();
    assert!(candidates.contains("example/encoder"), "{candidates}");
    assert!(candidates.contains("example/encoder@1.2.3"), "{candidates}");
    assert!(output.stderr.is_empty());
}

#[test]
fn hostile_component_completion_partial_is_not_echoed_or_executed() {
    let project = tempdir().unwrap();
    let sentinel = project.path().join("completion-side-effect");
    let partial = format!("example/encoder;touch {}", sentinel.display());
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .env("COMPLETE", "bash")
        .env("_CLAP_COMPLETE_INDEX", "3")
        .args(["--", "m-model", "component", "show", partial.as_str()])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(output.stderr.is_empty());
    assert!(
        !sentinel.exists(),
        "completion interpreted partial input as shell code"
    );
}

#[test]
fn dynamic_component_parameter_completion_uses_registered_schema_without_python() {
    let project = tempdir().unwrap();
    let manifest_directory = project
        .path()
        .join(".model-maven/components/example/optimizer");
    std::fs::create_dir_all(&manifest_directory).unwrap();
    std::fs::write(
        manifest_directory.join("1.2.3.json"),
        serde_json::json!({
            "schema_version": 1,
            "namespace": "example",
            "component_id": "example/optimizer",
            "component_version": "1.2.3",
            "extension_api_version": 1,
            "kind": "layer",
            "entrypoint": "must_not_be_imported:factory",
            "description": "test component",
            "params_schema": {
                "type": "object",
                "properties": {
                    "optimizer": {"type":"string", "enum":["adamw", "sgd"]}
                },
                "required": [],
                "additionalProperties": false
            },
            "capabilities": {"construct": "supported"}
        })
        .to_string(),
    )
    .unwrap();

    for (index, args) in [
        (
            "5",
            vec![
                "--",
                "m-model",
                "component",
                "test",
                "example/optimizer",
                "--param",
                "optimizer=\"a",
            ],
        ),
        (
            "8",
            vec![
                "--",
                "m-model",
                "layer",
                "add",
                "--parent",
                "model.layers",
                "--component",
                "example/optimizer",
                "--param",
                "optimizer=\"a",
            ],
        ),
        (
            "7",
            vec![
                "--",
                "m-model",
                "layer",
                "replace",
                "model.block",
                "--component",
                "example/optimizer",
                "--param",
                "optimizer=\"a",
            ],
        ),
        (
            "7",
            vec![
                "--",
                "m-model",
                "assembly",
                "add",
                "head",
                "--component",
                "example/optimizer",
                "--param",
                "optimizer=\"a",
            ],
        ),
        (
            "7",
            vec![
                "--",
                "m-model",
                "assembly",
                "set",
                "head",
                "--component",
                "example/optimizer",
                "--param",
                "optimizer=\"a",
            ],
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .env("COMPLETE", "bash")
            .env("_CLAP_COMPLETE_INDEX", index)
            .args(["--python", "missing-python-must-not-be-started"])
            .args(args.iter().copied())
            .current_dir(project.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let candidates = String::from_utf8(output.stdout).unwrap();
        assert!(candidates.contains("optimizer=\"adamw\""), "{candidates}");
        assert!(!candidates.contains("sgd"), "{candidates}");
        assert!(output.stderr.is_empty());

        let mut field_args = args[..args.len() - 1].to_vec();
        field_args.push("op");
        let field_output = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .env("COMPLETE", "bash")
            .env("_CLAP_COMPLETE_INDEX", index)
            .args(["--python", "missing-python-must-not-be-started"])
            .args(field_args)
            .current_dir(project.path())
            .output()
            .unwrap();
        assert!(
            field_output.status.success(),
            "{}",
            String::from_utf8_lossy(&field_output.stderr)
        );
        let fields = String::from_utf8(field_output.stdout).unwrap();
        assert!(fields.contains("optimizer="), "{fields}");
        assert!(field_output.stderr.is_empty());
    }

    let injection_output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .env("COMPLETE", "bash")
        .env("_CLAP_COMPLETE_INDEX", "6")
        .args(["--python", "missing-python-must-not-be-started"])
        .args([
            "--",
            "m-model",
            "model",
            "probe",
            "--inject",
            "encoder=component:example/optimizer",
            "--inject-param",
            "encoder.optimizer=\"a",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        injection_output.status.success(),
        "{}",
        String::from_utf8_lossy(&injection_output.stderr)
    );
    let injection_candidates = String::from_utf8(injection_output.stdout).unwrap();
    assert!(
        injection_candidates.contains("encoder.optimizer=\"adamw\""),
        "{injection_candidates}"
    );
    assert!(injection_output.stderr.is_empty());
}

#[test]
fn dynamic_component_completion_suggests_nested_schema_json_without_python() {
    let project = tempdir().unwrap();
    let manifest_directory = project.path().join(".model-maven/components/example/model");
    std::fs::create_dir_all(&manifest_directory).unwrap();
    std::fs::write(
        manifest_directory.join("1.0.0.json"),
        serde_json::json!({
            "schema_version": 1,
            "namespace": "example",
            "component_id": "example/model",
            "component_version": "1.0.0",
            "extension_api_version": 1,
            "kind": "model",
            "entrypoint": "must_not_be_imported:factory",
            "description": "nested completion fixture",
            "params_schema": {
                "type": "object",
                "properties": {
                    "settings": {
                        "type": "object",
                        "required": ["optimizer", "schedule"],
                        "additionalProperties": false,
                        "properties": {
                            "optimizer": {"type":"string", "enum":["adamw", "sgd"]},
                            "schedule": {
                                "type":"object",
                                "required":["warmup"],
                                "additionalProperties":false,
                                "properties":{"warmup":{"type":"integer", "default":3}}
                            }
                        }
                    }
                },
                "required": [],
                "additionalProperties": false
            },
            "capabilities": {"construct":"supported"}
        })
        .to_string(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .env("COMPLETE", "bash")
        .env("_CLAP_COMPLETE_INDEX", "5")
        .args(["--python", "missing-python-must-not-be-started"])
        .args([
            "--",
            "m-model",
            "component",
            "test",
            "example/model@1.0.0",
            "--param",
            "settings={\"optimizer\":\"a",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let candidates = String::from_utf8(output.stdout).unwrap();
    assert!(
        candidates.contains("settings={\"optimizer\":\"adamw\",\"schedule\":{\"warmup\":3}}"),
        "{candidates}"
    );
    assert!(output.stderr.is_empty());
}
