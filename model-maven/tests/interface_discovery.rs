use std::process::Command;

#[test]
fn interface_catalog_is_static_and_rejects_mixed_commands() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(dir.path())
        .args([
            "--json",
            "--no-interactive",
            "--python",
            "/does/not/exist",
            "--describe-interface",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["operation"], "interface");
    assert_eq!(report["data"]["interface_version"], 1);
    assert_eq!(
        report["data"]["report_protocol_version"],
        report["protocol_version"]
    );
    assert_eq!(
        report["data"]["capabilities"]["input_snapshot.capture_replay"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["training.request_identity"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["runs.lookup_request_id"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["project.conditional_batch_edit"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["augmentation.conditional_config_edit"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["edit.conditional_source_manifest"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["edit.conditional_source_plan"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["edit.conditional_source_input"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["edit.conditional_source_config"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["edit.conditional_source_project_manifest"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["layer.builtin_catalog"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["training.conditional_supervision"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["training.conditional_variant_start"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["training.conditional_resume_checkpoint"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["evaluation.conditional_checkpoint"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["export.conditional_checkpoint"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["assembly.atomic_patch"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["assembly.atomic_patch_evidence"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["assembly.atomic_patch_snapshot"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["assembly.generated_source_verify"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["project.scaffold_input_routes"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["project.scaffold_named_outputs"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["project.scaffold_custom_module"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["project.scaffold_external_batch"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["project.scaffold_external_batch_params"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["project.scaffold_initial_bindings"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["project.scaffold_objectives"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["project.scaffold_objective_weights"],
        true
    );
    assert_eq!(
        report["data"]["capabilities"]["data.csv_batch_template"],
        true
    );
    assert_eq!(report["data"]["capabilities"]["data.csv_batch_typed"], true);
    assert_eq!(
        report["data"]["capabilities"]["training.data_fingerprint"],
        true
    );

    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(dir.path())
        .args(["--json", "--describe-interface", "doctor"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["diagnostics"][0]["code"], "CLI_ARGUMENT_INVALID");
}

#[test]
fn built_in_layer_catalog_is_static_and_scoped_to_common_fields() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(dir.path())
        .args([
            "--json",
            "--no-interactive",
            "--python",
            "/does/not/exist",
            "layer",
            "catalog",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["operation"], "layer_catalog");
    assert_eq!(report["data"]["scope"], "common-pytorch-constructor-fields");
    assert_eq!(report["data"]["runtime_validation"], "not_run");
    let layers = report["data"]["layers"].as_array().unwrap();
    let linear = layers
        .iter()
        .find(|layer| layer["name"] == "Linear")
        .unwrap();
    assert_eq!(linear["fields"][0]["name"], "in_features");
    assert_eq!(linear["fields"][0]["required"], true);
    assert_eq!(linear["fields"][1]["name"], "out_features");
}

#[test]
fn catalog_types_construct_and_run_in_the_selected_pytorch_environment() {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(&python)
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|s| s.success())
    {
        eprintln!("skipped catalog runtime check: PyTorch unavailable");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let catalog = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .current_dir(dir.path())
        .args(["--json", "--no-interactive", "layer", "catalog"])
        .output()
        .unwrap();
    assert!(catalog.status.success());
    let program = r#"
import json, sys, torch
from types import SimpleNamespace
from model_maven_worker.edit.layers import new_layer
catalog = json.loads(sys.argv[1])["data"]["layers"]
params = {"Linear":{"in_features":4,"out_features":4}, "LayerNorm":{"normalized_shape":4}, "Conv2d":{"in_channels":3,"out_channels":3,"kernel_size":1}}
runtime = SimpleNamespace(torch=torch, entrypoint=None)
assert len({item["name"] for item in catalog}) == len(catalog) == 7
for item in catalog:
    name = item["name"]
    layer = new_layer(runtime, {"layer_type":name,"params":params.get(name,{})})
    x = torch.ones(2,3,4,4) if name == "Conv2d" else torch.ones(2,4)
    assert layer(x).shape == x.shape, name
print("ok")
"#;
    let output = Command::new(python)
        .arg("-c")
        .arg(program)
        .arg(String::from_utf8(catalog.stdout).unwrap())
        .env(
            "PYTHONPATH",
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("python"),
        )
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
