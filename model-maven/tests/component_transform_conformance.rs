use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn invoke(root: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", &python()])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected JSON report, got {:?}: {error}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

fn assert_success(output: &std::process::Output, report: &Value) {
    assert!(
        output.status.success(),
        "{report:#}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(report["status"], "success", "{report:#}");
}

#[test]
fn discovered_transform_runs_cli_conformance_with_locked_package_source_and_schema() {
    if !Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
    {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let site = tempfile::tempdir().unwrap();
    let package = site.path().join("transform_ext");
    let manifest_dir = package.join("__model_maven__");
    let dist_info = site.path().join("transform_ext-1.0.0.dist-info");
    fs::create_dir_all(&manifest_dir).unwrap();
    fs::create_dir_all(&dist_info).unwrap();

    let manifest_path = "transform_ext/__model_maven__/scale.json";
    let source_path = "transform_ext/scale.py";
    let dataset_manifest_path = "transform_ext/__model_maven__/dataset.json";
    let loss_manifest_path = "transform_ext/__model_maven__/loss.json";
    let metric_manifest_path = "transform_ext/__model_maven__/metric.json";
    let values_source_path = "transform_ext/values.py";
    let manifest = serde_json::json!({
        "schema_version":1,
        "namespace":"example",
        "component_id":"example/scale",
        "component_version":"1.0.0",
        "extension_api_version":1,
        "kind":"transform",
        "entrypoint":"transform_ext.scale:make_scale",
        "description":"Conformance-tested tensor scaling transform",
        "params_schema":{
            "type":"object",
            "properties":{"factor":{"type":"number","default":2.0}},
            "required":[],
            "additionalProperties":false
        },
        "capabilities":{"transform":{"state":"supported"}},
        "conformance_fixture":"transform_ext.scale:fixture"
    });
    fs::write(
        site.path().join(manifest_path),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    fs::write(
        site.path().join(source_path),
        "import torch\ndef make_scale(factor=2.0):\n    return lambda value: value * factor\ndef fixture():\n    return {'sample': torch.tensor([[1., 2.], [3., 4.]]), 'expected_port': {'kind': 'tensor', 'shape': [2, 2], 'dtype': 'float32'}, 'expected_output': torch.tensor([[2., 4.], [6., 8.]])}\n",
    )
    .unwrap();
    let values_manifest = |id: &str,
                           kind: &str,
                           entrypoint: &str,
                           fixture: &str,
                           capability: &str| {
        serde_json::json!({
            "schema_version":1,
            "namespace":"example",
            "component_id":format!("example/{id}"),
            "component_version":"1.0.0",
            "extension_api_version":1,
            "kind":kind,
            "entrypoint":entrypoint,
            "description":format!("Conformance-tested {kind} value component"),
            "params_schema":{"type":"object","properties":{},"required":[],"additionalProperties":false},
            "capabilities":{kind:{"state":"supported"}},
            "conformance_fixture":fixture,
            "port_contracts":{"result":{"kind":capability}}
        })
    };
    for (path, manifest) in [
        (
            dataset_manifest_path,
            values_manifest(
                "dataset",
                "dataset",
                "transform_ext.values:make_dataset",
                "transform_ext.values:dataset_fixture",
                "sample",
            ),
        ),
        (
            loss_manifest_path,
            values_manifest(
                "loss",
                "loss",
                "transform_ext.values:make_l1",
                "transform_ext.values:loss_fixture",
                "scalar_loss",
            ),
        ),
        (
            metric_manifest_path,
            values_manifest(
                "metric",
                "metric",
                "transform_ext.values:make_accuracy",
                "transform_ext.values:metric_fixture",
                "scalar_metric",
            ),
        ),
    ] {
        fs::write(
            site.path().join(path),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }
    fs::write(
        site.path().join(values_source_path),
        "import torch\ndef make_dataset():\n    return [(torch.tensor([float(i)]), i) for i in range(3)]\ndef dataset_fixture():\n    return {'expected_length': 3, 'sample_index': 1, 'expected_sample': (torch.tensor([1.]), 1)}\ndef make_l1():\n    return torch.nn.L1Loss()\ndef loss_fixture():\n    return {'args': (torch.tensor([1., 3.]), torch.tensor([0., 1.])), 'kwargs': {}, 'expected_output': 1.5}\ndef make_accuracy():\n    def accuracy(logits, targets):\n        return (logits.argmax(dim=1) == targets).float().mean()\n    return accuracy\ndef metric_fixture():\n    return {'args': (torch.tensor([[0.1, 0.9], [0.8, 0.2]]), torch.tensor([1, 0])), 'kwargs': {}, 'expected_output': 1.0}\n",
    )
    .unwrap();
    fs::write(
        dist_info.join("METADATA"),
        "Metadata-Version: 2.1\nName: transform-ext\nVersion: 1.0.0\n",
    )
    .unwrap();
    fs::write(
        dist_info.join("entry_points.txt"),
        "[model_maven.components]\nscale = transform_ext.scale:make_scale\ndataset = transform_ext.values:make_dataset\nloss = transform_ext.values:make_l1\nmetric = transform_ext.values:make_accuracy\n",
    )
    .unwrap();
    fs::write(
        dist_info.join("RECORD"),
        format!(
            "{manifest_path},,\n{source_path},,\n{dataset_manifest_path},,\n{loss_manifest_path},,\n{metric_manifest_path},,\n{values_source_path},,\ntransform_ext-1.0.0.dist-info/METADATA,,\ntransform_ext-1.0.0.dist-info/entry_points.txt,,\ntransform_ext-1.0.0.dist-info/RECORD,,\n"
        ),
    )
    .unwrap();

    let site_path = site.path().to_string_lossy().into_owned();
    let (discover_output, discovered) = invoke(
        project.path(),
        &["component", "discover", "--path", &site_path],
    );
    assert_success(&discover_output, &discovered);
    assert_eq!(
        discovered["data"]["registered"].as_array().unwrap().len(),
        4,
        "{discovered:#}"
    );

    let (test_output, tested) = invoke(
        project.path(),
        &[
            "component",
            "test",
            "example/scale",
            "--param",
            "factor=2.0",
        ],
    );
    assert_success(&test_output, &tested);
    assert_eq!(
        tested["data"]["coverage"],
        "complete_for_fixture_expectations"
    );
    assert_eq!(tested["data"]["claim_verified"], true);
    assert_eq!(
        tested["data"]["result"]["output"]["shape"],
        serde_json::json!([2, 2])
    );
    assert_eq!(tested["data"]["result"]["output_port"]["dtype"], "float32");
    assert_eq!(
        tested["data"]["capability_results"]["transform"]["expected_output"], "passed",
        "{tested:#}"
    );

    let (bad_params_output, bad_params) = invoke(
        project.path(),
        &[
            "component",
            "test",
            "example/scale",
            "--param",
            "factor=\"large\"",
        ],
    );
    assert!(!bad_params_output.status.success());
    assert_ne!(bad_params["status"], "success");

    for (id, expected) in [
        ("example/dataset", "passed"),
        ("example/loss", "finite_scalar"),
        ("example/metric", "finite_and_matched"),
    ] {
        let (output, report) = invoke(project.path(), &["component", "test", id]);
        assert_success(&output, &report);
        assert_eq!(
            report["data"]["coverage"], "complete_for_fixture_expectations",
            "{report:#}"
        );
        assert_eq!(report["data"]["claim_verified"], true, "{report:#}");
        let kind = report["data"]["component"]["kind"].as_str().unwrap();
        assert_eq!(
            report["data"]["capability_results"][kind]["status"], "passed",
            "{report:#}"
        );
        if kind == "loss" {
            assert_eq!(
                report["data"]["capability_results"][kind]["output"], expected,
                "{report:#}"
            );
            assert!(
                report["data"]["capability_results"][kind]["gradient_values"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
        } else if kind == "metric" {
            assert_eq!(
                report["data"]["capability_results"][kind]["output"], expected,
                "{report:#}"
            );
        } else {
            assert_eq!(
                report["data"]["capability_results"][kind]["sample"], expected,
                "{report:#}"
            );
        }
    }
}
