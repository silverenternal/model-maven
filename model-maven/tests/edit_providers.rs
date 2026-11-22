use std::{fs, process::Command};

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
fn named_module_sharing_and_provider_lifecycle_are_explicit_and_replayable() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let root = env!("CARGO_MANIFEST_DIR")
        .replace('\\', "\\\\")
        .replace('\'', "\\'");
    let script = format!(
        r#"
import importlib
import sys
import torch
sys.path.insert(0, r'{root}/python')
sys.path.insert(0, r'{root}')
from model_maven_worker.edit.layers import LayerEditRuntime, apply_edit
from model_maven_worker.edit.sharing import aliases_for

def entries(model):
    modules = dict(model.named_modules(remove_duplicate=False))
    aliases = {{}}
    for path, module in modules.items(): aliases.setdefault(id(module), []).append(path)
    return modules, aliases

runtime = LayerEditRuntime(
    torch=torch,
    entrypoint=lambda ref: getattr(importlib.import_module(ref.split(':')[0]), ref.split(':')[1]),
    resolve_module=lambda model, path: model.get_submodule(path.removeprefix('model.')) if path != 'model' else model,
    module_entries=entries,
    normalize_path=lambda path: path if path.startswith('model') else f'model.{{path}}',
    editable_params=lambda module: None,
)

class Blocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared = torch.nn.Linear(3, 3)
        self.encoder = shared
        self.decoder = shared

model = Blocks()
before = model.encoder.weight.detach().clone()
effect = apply_edit(runtime, model, {{'action':'sharing','mode':'clone-selected','path':'model.decoder'}})
assert model.encoder is not model.decoder
assert torch.equal(model.encoder.weight, model.decoder.weight)
assert effect['retained_aliases'] == ['model.encoder']
assert effect['cloned_aliases'] == ['model.decoder']
assert torch.equal(model.decoder.weight, before)
effect = apply_edit(runtime, model, {{'action':'sharing','mode':'share-selected','source':'model.encoder','targets':['model.decoder']}})
assert model.encoder is model.decoder
assert effect['shared_aliases'] == ['model.decoder', 'model.encoder']
assert effect['replaced_target_state'][0]['state_policy'] == 'replaced_by_source_module_state'
assert effect['replaced_target_state'][0]['state_keys'] == ['bias', 'weight']

fresh = Blocks()
operation = {{'action':'custom','provider':'examples.edit_providers.named_branch_rewire:NamedBranchRewire',
              'payload':{{'source':'model.encoder','target':'model.decoder'}}}}
effect = apply_edit(runtime, fresh, operation)
assert fresh.encoder is fresh.decoder
assert effect['result']['shared'] is True
assert operation['provider_plan'] == {{'source':'model.encoder','target':'model.decoder'}}
assert operation['provider_persistence']['schema_version'] == 1

reloaded = Blocks()
replay = {{key:value for key,value in operation.items()}}
effect = apply_edit(runtime, reloaded, replay)
assert reloaded.encoder is reloaded.decoder
assert effect['plan']['source'] == 'model.encoder'

failed_model = Blocks()
old_decoder = failed_model.decoder
provider = importlib.import_module('examples.edit_providers.named_branch_rewire').NamedBranchRewire()
original_candidate = provider.candidate
def candidate_then_fail(model, plan):
    original_candidate(model, plan)
    raise RuntimeError('injected candidate failure after mutation')
provider.candidate = candidate_then_fail
runtime = LayerEditRuntime(
    torch=torch,
    entrypoint=lambda ref: provider,
    resolve_module=lambda model, path: model.get_submodule(path.removeprefix('model.')) if path != 'model' else model,
    module_entries=entries,
    normalize_path=lambda path: path if path.startswith('model') else f'model.{{path}}',
    editable_params=lambda module: None,
)
failed = {{'action':'custom','provider':'examples.edit_providers.named_branch_rewire:NamedBranchRewire',
          'payload':{{'source':'model.encoder','target':'model.decoder'}}}}
try:
    apply_edit(runtime, failed_model, failed)
except Exception as error:
    assert getattr(error, 'code', None) == 'CUSTOM_EDIT_FAILED', error
else:
    raise AssertionError('expected injected Provider failure')
assert failed_model.decoder is old_decoder, 'rollback did not restore target module identity'
"#
    );
    let output = Command::new(python())
        .args(["-c", &script])
        .output()
        .expect("could not start the test Python interpreter");
    assert!(
        output.status.success(),
        "Provider lifecycle test failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn provider_source_is_readable_and_does_not_mutate_user_sources() {
    let provider = fs::read_to_string("examples/edit_providers/named_branch_rewire.py")
        .expect("provider example should be checked in");
    assert!(provider.contains("MODEL_MAVEN_LAYER_EDIT_API = 2"));
    assert!(provider.contains("def plan("));
    assert!(provider.contains("def rollback("));
}

#[test]
fn sharing_operation_is_persisted_and_replayed_by_the_cli() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "from torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        shared = nn.Linear(2, 2)\n        self.encoder = shared\n        self.decoder = shared\n    def forward(self, x):\n        return self.decoder(self.encoder(x))\ndef build_model():\n    return Net()\n",
    )
    .unwrap();
    fs::write(
        project.path().join("data.py"),
        "import torch\ndef batch():\n    return torch.ones((1, 2))\n",
    )
    .unwrap();
    let operation = serde_json::json!({
        "action":"sharing", "mode":"clone-selected", "path":"model.decoder"
    })
    .to_string();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "layer",
            "apply",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--operation",
            &operation,
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        output.status.success(),
        "{report:#}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(report["data"]["committed"], true);
    assert_eq!(
        report["data"]["edit_effects"][0]["action"],
        "clone-selected"
    );
    assert_eq!(report["data"]["validation"]["forward"], "passed");

    let variant: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/variants/working.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(variant["edits"][0]["operations"][0]["action"], "sharing");
    assert_eq!(
        variant["edits"][0]["operations"][0]["path"],
        "model.decoder"
    );

    let list = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "layer",
            "list",
            "--model",
            "model:build_model",
            "--variant",
            "working",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    let listed: serde_json::Value = serde_json::from_slice(&list.stdout).unwrap();
    assert!(
        list.status.success(),
        "{listed:#}\n{}",
        String::from_utf8_lossy(&list.stderr)
    );
    assert!(
        listed["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|module| module["path"] == "model.decoder")
    );
}

#[test]
fn api_v2_provider_variant_replays_through_cli_undo_redo_and_recovers_failures() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("model.py"),
        "from torch import nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.encoder = nn.Linear(2, 2)\n        self.decoder = nn.Linear(2, 2)\n    def forward(self, x):\n        return self.decoder(self.encoder(x))\ndef build_model():\n    return Net()\n",
    )
    .unwrap();
    fs::write(
        project.path().join("data.py"),
        "import torch\ndef batch():\n    return torch.ones((1, 2))\n",
    )
    .unwrap();
    fs::write(
        project.path().join("provider_fixture.py"),
        r#"MODEL_MAVEN_LAYER_EDIT_API = 2

def _parent(model, path):
    parent_path, _, name = path.removeprefix("model.").rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return parent, name

class Rewire:
    def __init__(self): self.previous = None
    def describe(self, model, operation): return {"name": "rewire"}
    def can_edit(self, model, operation): return {"state": "supported"}
    def plan(self, model, operation): return {"source": "model.encoder", "target": "model.decoder"}
    def persist(self, plan): return {"schema_version": 1, **plan}
    def candidate(self, model, plan):
        source_parent, source_name = _parent(model, plan["source"])
        target_parent, target_name = _parent(model, plan["target"])
        self.previous = (target_parent, target_name, target_parent._modules[target_name])
        target_parent._modules[target_name] = source_parent._modules[source_name]
        if __import__("os").environ.get("PROVIDER_FAIL_AFTER_MUTATION") == "1":
            raise RuntimeError("injected candidate failure")
        return {"shared": target_parent._modules[target_name] is source_parent._modules[source_name]}
    def rollback(self, model, plan):
        if self.previous is not None:
            parent, name, module = self.previous
            parent._modules[name] = module
            self.previous = None
        marker = __import__("os").environ.get("PROVIDER_ROLLBACK_MARKER")
        if marker: open(marker, "w").write("called")
        return {"restored": True}
"#,
    )
    .unwrap();

    let invoke = |args: &[&str], fail_candidate: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_m-model"));
        command
            .args(["--json", "--python", &python()])
            .args(args)
            .current_dir(project.path());
        if fail_candidate {
            command.env("PROVIDER_FAIL_AFTER_MUTATION", "1");
            command.env(
                "PROVIDER_ROLLBACK_MARKER",
                project.path().join("rollback-called").as_os_str(),
            );
        }
        command.output().unwrap()
    };
    let operation = serde_json::json!({
        "action":"custom",
        "provider":"provider_fixture:Rewire",
        "payload":{}
    })
    .to_string();
    let applied = invoke(
        &[
            "layer",
            "apply",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--operation",
            &operation,
        ],
        false,
    );
    let report: serde_json::Value = serde_json::from_slice(&applied.stdout).unwrap();
    assert!(
        applied.status.success(),
        "{report:#}\n{}",
        String::from_utf8_lossy(&applied.stderr)
    );
    assert_eq!(report["data"]["committed"], true);
    let variant_path = project.path().join(".model-maven/variants/working.json");
    let variant: serde_json::Value =
        serde_json::from_slice(&fs::read(&variant_path).unwrap()).unwrap();
    let stored_operation = &variant["edits"][0]["operations"][0];
    assert_eq!(stored_operation["provider_plan"]["source"], "model.encoder");
    assert_eq!(
        stored_operation["provider_persistence"]["target"],
        "model.decoder"
    );

    let undo = invoke(&["layer", "undo"], false);
    let undo_report: serde_json::Value = serde_json::from_slice(&undo.stdout).unwrap();
    assert!(
        undo.status.success(),
        "{undo_report:#}\n{}",
        String::from_utf8_lossy(&undo.stderr)
    );
    assert_eq!(undo_report["data"]["variant"]["cursor"], 0);
    let redo = invoke(&["layer", "redo"], false);
    let redo_report: serde_json::Value = serde_json::from_slice(&redo.stdout).unwrap();
    assert!(
        redo.status.success(),
        "{redo_report:#}\n{}",
        String::from_utf8_lossy(&redo.stderr)
    );
    assert_eq!(redo_report["data"]["variant"]["cursor"], 1);
    assert_eq!(
        redo_report["data"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .find(|module| module["path"] == "model.encoder")
            .unwrap()["aliases"],
        serde_json::json!(["model.encoder", "model.decoder"])
    );

    let before_failure = fs::read(&variant_path).unwrap();
    let failed = invoke(
        &[
            "layer",
            "apply",
            "--model",
            "model:build_model",
            "--batch",
            "data:batch",
            "--operation",
            &operation,
        ],
        true,
    );
    let failed_report: serde_json::Value = serde_json::from_slice(&failed.stdout).unwrap();
    assert!(
        !failed.status.success(),
        "unexpected success: {failed_report:#}"
    );
    assert_eq!(
        failed_report["diagnostics"][0]["code"],
        "CUSTOM_EDIT_FAILED"
    );
    assert_eq!(
        fs::read_to_string(project.path().join("rollback-called")).unwrap(),
        "called"
    );
    assert_eq!(
        fs::read(&variant_path).unwrap(),
        before_failure,
        "failed candidate changed the persisted variant"
    );
}
