use std::{path::Path, process::Command};

#[test]
fn migration_preserves_legacy_starter_behavior_and_rejects_ambiguous_code() {
    let python = std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into());
    if !Command::new(&python)
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
    {
        eprintln!("skipped starter migration integration: Python needs PyTorch");
        return;
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(python)
        .arg("-c")
        .arg(
            r#"
from pathlib import Path
import sys
import torch
from model_maven_worker.edit.starter_migration import migrate_starter_source
from model_maven_worker.protocol import UserError
source = Path(sys.argv[1]).read_text()
source += '\n# user-owned comment and helper must survive\ndef user_helper(): return 17\n'
candidate = migrate_starter_source(source)
assert '# user-owned comment and helper must survive\ndef user_helper(): return 17' in candidate
original, updated = {'__name__':'old'}, {'__name__':'new'}
exec(compile(source, 'old.py', 'exec'), original)
exec(compile(candidate, 'new.py', 'exec'), updated)
torch.manual_seed(42); before = original['model']()
torch.manual_seed(42); after = updated['model']()
batch = original['batch']()['kwargs']
assert torch.equal(before(**batch), after(**batch))
assert list(before.state_dict()) == list(after.state_dict())
assert all(torch.equal(v, after.state_dict()[k]) for k,v in before.state_dict().items())
native = updated['model'](encoder_output_modes={'image_encoder':'native'})
assert native._m_model_encode('image_encoder', batch).shape == batch['image'].shape
routed = updated['model'](input_routes={'image_encoder':'tabular'})
assert routed._m_model_encode('image_encoder', batch).shape == batch['tabular'].shape
for invalid in (
    source.replace('.flatten(start_dim=1)', '.mean(dim=1)', 1),
    source.replace('        self.encoders =', '        self.input_routes = {}\n        self.encoders =', 1),
    source.replace('        self.encoders =', '        _m_model_inputs = 1\n        self.encoders =', 1),
):
    try: migrate_starter_source(invalid)
    except UserError as error: assert error.code == 'STARTER_MIGRATION_UNSUPPORTED'
    else: raise AssertionError('ambiguous starter must be rejected')
windows = source.replace('\n', '\r\n')
converted = migrate_starter_source(windows)
assert '\r\n' in converted and '\n' not in converted.replace('\r\n', '')
print('migration candidate checks passed')
"#,
        )
        .arg(root.join("docs/acceptance/results/journey-a-20261121/model_starter.py"))
        .env("PYTHONPATH", root.join("python"))
        .output()
        .expect("start migration integration");
    assert!(
        output.status.success(),
        "migration fixture failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
