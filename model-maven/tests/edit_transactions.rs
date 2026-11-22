use std::process::Command;

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
fn variant_layer_initialization_is_replayable_and_preserves_global_rng() {
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
import sys
sys.path.insert(0, r'{root}/python')
import torch
from model_maven_worker.edit.layers import LayerEditRuntime, apply_edit, new_layer

def module_entries(model):
    modules = dict(model.named_modules(remove_duplicate=False))
    aliases = {{}}
    for path, module in modules.items():
        aliases.setdefault(id(module), []).append(path)
    return modules, aliases

def editable_params(module):
    if isinstance(module, torch.nn.Linear):
        return {{"in_features": module.in_features, "out_features": module.out_features, "bias": module.bias is not None}}
    return None

runtime = LayerEditRuntime(
    torch=torch,
    entrypoint=lambda _: None,
    resolve_module=lambda model, path: model.get_submodule(path),
    module_entries=module_entries,
    normalize_path=lambda value: value,
    editable_params=editable_params,
)
torch.manual_seed(23)
before = torch.random.get_rng_state().clone()
edit = {{"layer_type": "Linear", "params": {{"in_features": 8, "out_features": 4}}, "initialization_seed": 912345}}
first = new_layer(runtime, edit)
assert torch.equal(before, torch.random.get_rng_state()), "layer creation advanced the global PyTorch RNG"
second = new_layer(runtime, edit)
assert torch.equal(first.weight, second.weight), "replaying a persisted seed changed initialized weights"
assert torch.equal(first.bias, second.bias)
different = new_layer(runtime, {{**edit, "initialization_seed": 912346}})
assert not torch.equal(first.weight, different.weight), "distinct edit seeds produced identical weights"

model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
untouched_parameter = model[1].weight
apply_edit(runtime, model, {{"action": "set", "path": "0", "params": {{"out_features": 4}}}})
assert model[1].weight is untouched_parameter, "editing one layer replaced an untouched module parameter object"
"#
    );
    let output = Command::new(python())
        .args(["-c", &script])
        .output()
        .expect("could not start the test Python interpreter");
    assert!(
        output.status.success(),
        "seeded variant initialization failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn edits_preserve_shared_parameter_identity_or_reject_unsafe_changes() {
    if !has_torch() {
        return;
    }
    let root = env!("CARGO_MANIFEST_DIR")
        .replace('\\', "\\\\")
        .replace('\'', "\\'");
    let script = format!(
        r#"
import sys
sys.path.insert(0, r'{root}/python')
import torch
from model_maven_worker.edit.layers import LayerEditRuntime, apply_edit
from model_maven_worker.model.structure import module_entries, resolve_module, normalize_path
from model_maven_worker.model.inspection import editable_params
from model_maven_worker.protocol import UserError
runtime = LayerEditRuntime(torch, lambda _: None, resolve_module, module_entries, normalize_path, lambda m: editable_params(m, torch))
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(4, 4)
        self.right = torch.nn.Linear(4, 4)
        self.right.weight = self.left.weight
    def forward(self, x): return self.right(self.left(x))
model = Net()
original_weight = model.left.weight
before = original_weight.detach().clone()
effect = apply_edit(runtime, model, {{'action':'set', 'path':'model.left', 'params':{{'bias':False}}}})
assert model.left.weight is model.right.weight is original_weight
assert torch.equal(model.left.weight, before)
assert effect['preserved_external_state_sharing'] == [{{'kind':'parameter', 'name':'weight', 'external_paths':['model.right.weight']}}]
assert effect['preserved'] == ['weight']
model(torch.ones(2,4)).sum().backward()
assert model.left.weight.grad is model.right.weight.grad
old = model.left
for edit in [{{'action':'set','path':'model.left','params':{{'out_features':3}}}}, {{'action':'replace','path':'model.left','layer_type':'Linear','params':{{'in_features':4,'out_features':4}}}}, {{'action':'remove','path':'model.left','strategy':'bypass'}}]:
    try: apply_edit(runtime, model, edit)
    except UserError as error:
        assert error.code == 'SHARED_STATE_EDIT_UNSUPPORTED', (error.code, str(error))
    else: raise AssertionError('unsafe edit accepted')
    assert model.left is old and model.left.weight is model.right.weight
buffer_net = Net()
shared_buffer = torch.ones(2)
buffer_net.left.register_buffer('stats', shared_buffer)
buffer_net.right.register_buffer('stats', shared_buffer)
old_left = buffer_net.left
try: apply_edit(runtime, buffer_net, {{'action':'set', 'path':'model.left', 'params':{{'bias':False}}}})
except UserError as error:
    assert error.code == 'SHARED_STATE_EDIT_UNSUPPORTED' and 'model.right.stats' in str(error)
else: raise AssertionError('shared buffer removed silently')
assert buffer_net.left is old_left and buffer_net.left.stats is buffer_net.right.stats
view_net = Net()
view_net.right.weight = torch.nn.Parameter(view_net.left.weight.detach().view(4, 4).t())
original = view_net.left.weight
assert view_net.right.weight is not original
assert view_net.right.weight.untyped_storage()._cdata == original.untyped_storage()._cdata
effect = apply_edit(runtime, view_net, {{'action':'set', 'path':'model.left', 'params':{{'bias':False}}}})
assert view_net.left.weight is original
assert view_net.right.weight.untyped_storage()._cdata == original.untyped_storage()._cdata
assert effect['preserved_external_state_sharing'][0]['sharing'] == 'storage'
assert effect['preserved_external_state_sharing'][0]['external_paths'] == ['model.right.weight']
from model_maven_worker.model.structure import shared_state_groups
storage_group = next(group for group in shared_state_groups(view_net) if group['kind'] == 'storage')
assert storage_group['paths'] == ['model.left.weight', 'model.right.weight']
assert storage_group['views'][0]['stride'] != storage_group['views'][1]['stride']
old = view_net.left
try: apply_edit(runtime, view_net, {{'action':'set', 'path':'model.left', 'params':{{'out_features':3}}}})
except UserError as error: assert error.code == 'SHARED_STATE_EDIT_UNSUPPORTED'
else: raise AssertionError('storage sharing silently detached')
assert view_net.left is old
empty = torch.nn.Module()
empty.register_buffer('zero', torch.empty(0))
assert shared_state_groups(empty) == [{{'kind':'storage_unavailable','paths':['model.zero'],'state_kind':'buffer'}}]
empty_layer = torch.nn.Sequential(torch.nn.Linear(0, 3))
empty_effect = apply_edit(runtime, empty_layer, {{'action':'set', 'path':'model.0', 'params':{{'bias':False}}}})
assert empty_effect['unverified_old_state_storage'] == [{{'kind':'parameter','name':'weight'}}]
print('shared identity and rejected changes verified')
"#
    );
    let result = Command::new(python())
        .args(["-c", &script])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
