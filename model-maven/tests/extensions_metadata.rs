use serde_json::{Value, json};
use std::{
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn invoke(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--verbose", "--python", &python()])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap()
}

fn report(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report: {error}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn python_validate(manifest: &Value) -> Output {
    let script = r#"
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('component_metadata', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = json.load(sys.stdin)
module.validate_manifest(value)
print('valid')
"#;
    let mut child = Command::new(python())
        .args([
            "-c",
            script,
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/python/model_maven_worker/extensions/metadata.py"
            ),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(manifest).unwrap().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

fn manifest() -> Value {
    json!({
        "schema_version": 1,
        "namespace": "demo",
        "component_id": "demo/layer",
        "component_version": "1.2.3",
        "extension_api_version": 1,
        "kind": "layer",
        "entrypoint": "layers:make_layer",
        "description": "A parameterized layer",
        "params_schema": {
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer", "minimum": 1, "default": 8}
                    },
                    "required": ["width"],
                    "additionalProperties": false,
                    "default": {"width": 16}
                }
            },
            "required": ["options"],
            "additionalProperties": false
        },
        "capabilities": {
            "construct": {"state": "supported", "requires": ["torch"]},
            "forward_probe": {"state": "conditional", "reason": "Requires a model batch."}
        },
        "dependencies": [
            {"component_id": "demo/base", "version": "^1.0.0"},
            {"component_id": "demo/metrics", "version": "~2.0.0", "optional": true}
        ]
    })
}

#[test]
fn python_metadata_validator_accepts_supported_schema_and_capabilities() {
    let output = python_validate(&manifest());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "valid");
}

#[test]
fn python_metadata_validator_rejects_unexplained_capabilities_and_bad_defaults() {
    let mut invalid_capability = manifest();
    invalid_capability["capabilities"]["forward_probe"] = json!("unsupported");
    assert!(!python_validate(&invalid_capability).status.success());

    let mut invalid_default = manifest();
    invalid_default["params_schema"]["properties"]["options"]["default"]["width"] = json!(0);
    assert!(!python_validate(&invalid_default).status.success());

    let mut invalid_dependency = manifest();
    invalid_dependency["dependencies"][0]["version"] = json!("^x");
    assert!(!python_validate(&invalid_dependency).status.success());
}

#[test]
fn rust_registration_uses_the_same_capability_reason_and_schema_rules() {
    let project = tempfile::tempdir().unwrap();
    let valid = invoke(
        project.path(),
        &[
            "component",
            "add",
            "demo/layer",
            "--kind",
            "layer",
            "--entrypoint",
            "layers:make_layer",
            "--version",
            "1.2.3",
            "--params-schema",
            r#"{"type":"object","properties":{"width":{"type":"integer","minimum":1,"default":8}},"required":["width"],"additionalProperties":false}"#,
            "--capabilities",
            r#"{"construct":{"state":"supported","requires":["torch"]},"forward_probe":{"state":"conditional","reason":"Requires a model batch."}}"#,
            "--dependency",
            "demo/base@^1.0.0",
            "--optional-dependency",
            "demo/metrics@~2.0.0",
        ],
    );
    assert!(valid.status.success(), "{}", report(&valid));

    let invalid = invoke(
        project.path(),
        &[
            "component",
            "add",
            "demo/bad",
            "--kind",
            "layer",
            "--entrypoint",
            "layers:make_layer",
            "--capabilities",
            r#"{"forward_probe":"unsupported"}"#,
        ],
    );
    let invalid_report = report(&invalid);
    assert!(!invalid.status.success(), "{invalid_report:#}");
    assert!(
        invalid_report["diagnostics"][0]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("reason"),
        "{invalid_report:#}"
    );

    let bad_range = invoke(
        project.path(),
        &[
            "component",
            "add",
            "demo/bad-range",
            "--kind",
            "layer",
            "--entrypoint",
            "layers:make_layer",
            "--dependency",
            "demo/base@^x",
        ],
    );
    let range_report = report(&bad_range);
    assert!(!bad_range.status.success(), "{range_report:#}");
}
