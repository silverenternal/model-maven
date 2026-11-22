#!/usr/bin/env python3
"""Check actual parameters for a TUI-created require rule, on reject and pass."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
tui = os.environ.get('MODEL_MAVEN_TUI_BIN', str(root / 'target/debug/m-model-tui'))
cli = os.environ.get('MODEL_MAVEN_CLI', str(root.parent / 'model-maven/target/debug/m-model'))
with tempfile.TemporaryDirectory(prefix='maven-supervision-parameters-') as directory:
    project = Path(directory)
    env = dict(os.environ, MODEL_MAVEN_TUI_PREFERENCES=str(project / 'prefs.json'))
    proc = subprocess.run([tui, '--line-mode', '--lang', 'en', '--cli', cli], cwd=project, env=env,
        input='\nm\nimage=3,4,4;tabular=5\n3\ny\n2\n\n\nt\n2\n\nfused\nFused features\ny\nrange\nrequire\n0\n0.5\n\ny\nq\n',
        capture_output=True, text=True, timeout=120, check=True)
    assert 'would_reject' in proc.stdout and 'Sample localization: insufficient' in proc.stdout, proc.stdout
    state = json.loads((project / '.model-maven/semantic-marks.json').read_text())
    mark = state['marks'][0]
    rule = mark['rules'][0]
    assert mark['module_path'] == 'model.fusion' and rule['mode'] == 'require', state
    (project / 'checked_step.py').write_text('''import atexit, json
from pathlib import Path
import torch
import model_starter

def step(model, batch, context):
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    def record():
        after = dict(model.named_parameters())
        changed = [name for name, parameter in before.items() if not torch.equal(parameter, after[name].detach())]
        Path("parameter-observation.json").write_text(json.dumps({"names": list(before), "changed": changed}))
    atexit.register(record)
    return model_starter.step(model, batch, context)
step.MODEL_MAVEN_STATELESS = True
''')
    def command(*args):
        result = subprocess.run([cli, '--json', '--no-interactive', *args], cwd=project, env=env,
                                capture_output=True, text=True, timeout=120)
        return result, json.loads(result.stdout)
    def attempt():
        observation = project / 'parameter-observation.json'
        observation.unlink(missing_ok=True)
        result, report = command('try', '--step', 'checked_step:step', '--steps', '1', '--apply-supervision')
        assert observation.exists(), (result.stdout, result.stderr)
        return result, report, json.loads(observation.read_text())
    result, report, observed = attempt()
    assert result.returncode != 0 and report['data']['cause_code'] == 'SUPERVISION_REQUIRE_FAILED', report
    assert observed['names'] and not observed['changed'], observed
    result, report = command('model', 'supervise', 'rule', 'remove', '--id', rule['id'])
    assert result.returncode == 0, report
    result, report = command('model', 'supervise', 'rule', 'add', '--mark', mark['id'], '--id', rule['id'],
                             '--kind', 'range', '--mode', 'require', '--min=-1000000', '--max=1000000')
    assert result.returncode == 0, report
    result, report = command('model', 'supervise', 'preview')
    assert result.returncode == 0 and 'would_reject' not in json.dumps(report), report
    result, report, observed = attempt()
    assert result.returncode == 0 and report['data']['completed_steps'] == 1, report
    assert observed['changed'], observed
    print('TUI-created fusion rule: rejected batch preserves actual parameters; legal batch updates them')
