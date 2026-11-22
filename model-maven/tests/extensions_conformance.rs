use std::{path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

#[test]
fn provider_conformance_is_scoped_to_declared_capabilities_and_fixture_evidence() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let script = r#"
import sys, tempfile, torch
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from model_maven_worker.extensions.conformance import (
    ExtensionConformanceRuntime, component_test_layer_edit_provider,
)
from model_maven_worker.protocol import UserError

class Model(torch.nn.Module):
    def __init__(self, finite=True):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(2, 2))
        if not finite:
            self.register_buffer("bad", torch.tensor(float("nan")))

def fixture(expect=True, finite=True):
    value = {"model": Model(finite)}
    if expect:
        value["expected_result"] = {"added": "encoder.act", "type": "ReLU"}
        value["expected_modules"] = {"encoder.act": "ReLU"}
    return value

def make_runtime(fixture_factory, source_fingerprint=None):
    return ExtensionConformanceRuntime(
        torch=torch,
        add_project_import_path=lambda _path: None,
        verify_component_references=lambda *_args, **_kwargs: None,
        new_layer=lambda _manifest: None,
        make_batch=lambda _batch: None,
        entrypoint=lambda _reference: fixture_factory,
        tree=lambda model: model,
        apply_edit=lambda model, _operation: (
            model.encoder.add_module("act", torch.nn.ReLU())
            or {"result": {"added": "encoder.act", "type": "ReLU"}}
        ),
        digest=lambda value: "sha256:" + str(sorted(value.items())),
        conformance_suite_version=7,
        module_source_fingerprint=source_fingerprint or (lambda name: {"module": name, "sha256": "abc"}),
        python_version="test-python",
        torch_version="test-torch",
        run_transform_pipeline=lambda *_args, **_kwargs: {},
    )

def run(capabilities, *, expected=True, finite=True, fixture_ref="fixture:check", source_fingerprint=None):
    runtime = make_runtime(lambda: fixture(expected, finite), source_fingerprint)
    manifest = {
        "kind": "layer-edit-provider", "component_id": "sample/provider",
        "component_version": "1.2.3", "entrypoint": "provider:edit",
        "capabilities": capabilities,
    }
    if fixture_ref:
        manifest["conformance_fixture"] = fixture_ref
    return component_test_layer_edit_provider(
        runtime, {"component": "provider:edit", "params": {"activation": "relu"}},
        Path(tempfile.gettempdir()), manifest,
    )

supported = run({"layer_edit": {"state": "supported"}})
assert supported["coverage"] == "complete_for_fixture_expectations"
assert supported["activation_eligible"] is True
evidence = supported["verification_evidence"]
assert evidence["payload_sha256"].startswith("sha256:")
assert evidence["suite_version"] == 7
assert evidence["source_fingerprints"]["provider"]["module"] == "provider"
assert evidence["source_fingerprints"]["fixture"]["module"] == "fixture"
assert evidence["python_version"] == "test-python"
assert evidence["torch_version"] == str(torch.__version__)

source_reads = {}
def drifting_source_fingerprint(name):
    source_reads[name] = source_reads.get(name, 0) + 1
    digest = "before" if source_reads[name] == 1 else "changed"
    return {"module": name, "sha256": digest}

drifted_source = run(
    {"layer_edit": {"state": "supported"}},
    source_fingerprint=drifting_source_fingerprint,
)
assert drifted_source["coverage"] == "partial"
assert drifted_source["activation_eligible"] is False
assert drifted_source["verification_evidence"] is None
assert "源码在符合性测试期间发生变化" in drifted_source["activation_reason"]

no_assertions = run({"layer_edit": {"state": "supported"}}, expected=False)
assert no_assertions["coverage"] == "partial"
assert no_assertions["activation_eligible"] is False
assert no_assertions["verification_evidence"] is None

conditional = run({"layer_edit": {"state": "conditional", "reason": "model-specific"}})
assert conditional["checks"]["layer_edit"]["status"] == "passed"
assert conditional["activation_eligible"] is False
assert conditional["verification_evidence"] is None
assert "conditional" in conditional["activation_reason"]

unsupported = run({"layer_edit": {"state": "unsupported", "reason": "not implemented"}})
assert unsupported["coverage"] == "complete_for_declared_checks"
assert unsupported["checks"]["layer_edit"]["status"] == "skipped"
assert unsupported["activation_eligible"] is False

unverified_claim = run({
    "layer_edit": {"state": "supported"},
    "state_roundtrip": {"state": "supported"},
    "events": {"state": "supported"},
    "stop": {"state": "supported"},
    "timeout": {"state": "supported"},
})
assert unverified_claim["coverage"] == "partial"
for capability in ("state_roundtrip", "events", "stop", "timeout"):
    assert unverified_claim["capability_results"][capability]["status"] == "not_tested"
assert unverified_claim["activation_eligible"] is False

try:
    run({"layer_edit": {"state": "supported"}}, finite=False)
except UserError as error:
    assert error.code == "COMPONENT_STATE_NONFINITE"
else:
    raise AssertionError("non-finite fixture state was accepted")
"#;
    let python_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let output = Command::new(python())
        .args(["-c", script])
        .arg(python_source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn custom_transform_conformance_executes_the_data_runtime_and_respects_capabilities() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let script = r#"
import importlib, pathlib, sys, tempfile, torch
sys.path.insert(0, sys.argv[1])
from model_maven_worker.data.conformance import TransformConformanceRuntime, component_test_transform
from model_maven_worker.data.transforms import run_transform_pipeline
from model_maven_worker.protocol import UserError

root = pathlib.Path(tempfile.mkdtemp())
(root / "transform_fixture.py").write_text('''import torch\ndef make_scale(factor=2.0):\n    return lambda value: value * factor\ndef fixture():\n    return {"sample": torch.tensor([[1., 2.], [3., 4.]]),\n            "expected_port": {"kind": "tensor", "shape": [2, 2], "dtype": "float32"},\n            "expected_output": torch.tensor([[2., 4.], [6., 8.]])}\n''')
runtime = TransformConformanceRuntime(
    add_project_import_path=lambda path: sys.path.insert(0, path),
    verify_component_references=lambda *_args, **_kwargs: None,
    entrypoint=lambda ref: getattr(importlib.import_module(ref.split(':')[0]), ref.split(':')[1]),
    run_pipeline=run_transform_pipeline,
    torch=torch,
    tree=lambda _torch, value: value.detach().cpu().tolist(),
)
manifest = {
    "kind": "transform",
    "entrypoint": "transform_fixture:make_scale",
    "conformance_fixture": "transform_fixture:fixture",
    "capabilities": {"transform": {"state": "supported"}},
}
passed = component_test_transform(
    runtime, {"component": "transform_fixture:make_scale", "params": {"factor": 2.0}}, root, manifest
)
assert passed["coverage"] == "complete_for_fixture_expectations"
assert passed["claim_verified"] is True
assert passed["result"]["output"] == [[2.0, 4.0], [6.0, 8.0]]
assert passed["capability_results"]["transform"]["status"] == "passed"

conditional = dict(manifest, capabilities={"transform": {"state": "conditional", "reason": "runtime-specific"}})
conditional_result = component_test_transform(
    runtime, {"component": "transform_fixture:make_scale", "params": {"factor": 2.0}}, root, conditional
)
assert conditional_result["coverage"] == "complete_for_fixture_expectations"
assert conditional_result["claim_verified"] is False

unsupported = dict(manifest, capabilities={"transform": {"state": "unsupported", "reason": "not implemented"}})
skipped = component_test_transform(runtime, {"component": "missing:entry", "component_ref": {}}, root, unsupported)
assert skipped["coverage"] == "complete_for_declared_checks"
assert skipped["capability_results"]["transform"]["status"] == "skipped"

bad_port = dict(manifest, conformance_fixture="transform_fixture:bad_fixture")
(root / "transform_fixture.py").write_text((root / "transform_fixture.py").read_text() + '\ndef bad_fixture():\n    return {"sample": torch.ones((2, 2)), "expected_port": {"kind": "tensor", "shape": [3, 2]}, "expected_output": torch.ones((3, 2))}\n')
importlib.invalidate_caches()
importlib.reload(importlib.import_module("transform_fixture"))
try:
    component_test_transform(runtime, {"component": "transform_fixture:make_scale"}, root, bad_port)
except UserError as error:
    assert error.code == "TRANSFORM_PORT_MISMATCH"
else:
    raise AssertionError("transform output violating fixture port contract was accepted")
"#;
    let python_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let output = Command::new(python())
        .args(["-c", script])
        .arg(python_source)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
