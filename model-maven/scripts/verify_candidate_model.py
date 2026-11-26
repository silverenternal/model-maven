#!/usr/bin/env python3
"""Exercise a candidate CLI with a declared graph and a real local CSV batch.

The caller supplies an existing Python environment with PyTorch; this script
never creates an environment or installs a framework in a user's project.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

cli, graph, data = map(Path, sys.argv[1:4])
launcher = ['node', str(cli.resolve())] if cli.suffix == '.cjs' else [str(cli.resolve())]
graph, data = graph.resolve(), data.resolve()
env = dict(os.environ)
env.pop('LD_LIBRARY_PATH', None)
with tempfile.TemporaryDirectory(prefix='mm-candidate-model-') as temp:
    root = Path(temp)
    rows = list(csv.DictReader(data.open(newline='', encoding='utf-8')))
    if len(rows) < 2:
        raise RuntimeError('expected at least two CSV rows')
    (root / 'batch.csv').write_bytes(data.read_bytes())
    env['MM_CANDIDATE_CSV'] = str(root / 'batch.csv')
    (root / 'real_batch.py').write_text('''import csv\nimport os\nfrom pathlib import Path\nimport torch\n\ndef tensor(columns):\n    with Path(os.environ["MM_CANDIDATE_CSV"]).open(newline="", encoding="utf-8") as source:\n        rows = list(csv.DictReader(source))[:2]\n    return torch.tensor([[float(row[c]) for c in columns] for row in rows], dtype=torch.float32)\n\ndef batch():\n    return {"args": [], "kwargs": {"input": tensor(["a", "b"]), "input_2": tensor(["c", "d"])}}\n\ndef training_batch():\n    result = batch()\n    result["loss_inputs"] = {"targets": {"output": tensor(["target", "target2"])}}\n    return result\n\ndef loss(outputs, batch):\n    target = tensor(["target", "target2"])\n    return torch.nn.functional.mse_loss(outputs["output"], target) + torch.nn.functional.l1_loss(outputs["output_copy"], target)\n''', encoding='utf-8')
    def call(args):
        completed = subprocess.run(launcher + ['--json', '--python', sys.executable] + args,
                                   cwd=root, env=env, text=True, encoding='utf-8',
                                   capture_output=True, timeout=120)
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'non-JSON response (exit {completed.returncode}): {completed.stderr} {completed.stdout[:3000]}') from exc
        if completed.returncode or report.get('status') != 'success':
            raise RuntimeError(f'operation failed (exit {completed.returncode}): {json.dumps(report, ensure_ascii=False)} {completed.stderr}')
        return report
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    generated = call(['graph-generate', '--graph', str(graph), '--if-graph', digest])
    (root / 'graph_model.py').write_text(generated['data']['source'], encoding='utf-8')
    report = call(['model', 'probe', '--model', 'graph_model:model', '--batch', 'real_batch:batch'])
    calls = report['data'].get('calls', [])
    if not calls or report['data'].get('summary', {}).get('status') == 'failure':
        raise RuntimeError('probe did not report successful model calls')
    call(['model', 'verify', '--model', 'graph_model:model', '--batch', 'real_batch:batch', '--loss', 'real_batch:loss'])
    trial = call(['try', '--model', 'graph_model:model', '--batch', 'real_batch:training_batch',
                  '--step', 'graph_model:step', '--training-mode', 'managed-step', '--steps', '2'])
    training = call(['train', '--model', 'graph_model:model', '--batch', 'real_batch:training_batch',
                     '--step', 'graph_model:step', '--training-mode', 'managed-step', '--steps', '2',
                     '--checkpoint-every', '1'])
    runs = call(['runs', 'list'])
    trial_data = trial['data']
    if (trial_data.get('completed_steps') != 2 or trial_data.get('optimizer_updates') != 2
            or not trial_data.get('parameters_with_gradient')
            or not math.isfinite(trial_data.get('last_loss', float('nan')))):
        raise RuntimeError(f'incomplete training update: {trial_data}')
    if not any(run.get('id') == trial_data.get('run_id') and run.get('status') == 'completed'
               for run in runs['data'].get('runs', [])):
        raise RuntimeError(f'completed trial not present in run history: {runs}')
    training_data = training['data']
    if training_data.get('completed_steps') != 2 or not training_data.get('checkpoint_saved'):
        raise RuntimeError(f'training did not save a checkpoint: {training_data}')
    if not any(run.get('id') == training_data.get('run_id') and run.get('status') == 'completed'
               for run in runs['data'].get('runs', [])):
        raise RuntimeError(f'completed training not present in run history: {runs}')
    print(json.dumps({'platform': platform.platform(), 'python': sys.version.split()[0],
                      'graph_sha256': digest, 'csv_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                      'rows': 2, 'calls': len(calls), 'status': report['status'],
                      'completed_steps': trial_data['completed_steps'],
                      'optimizer_updates': trial_data['optimizer_updates'],
                      'parameters_with_gradient': trial_data['parameters_with_gradient'],
                      'last_loss': trial_data['last_loss'], 'run_id': trial_data['run_id'],
                      'run_record': 'completed', 'training_run': training_data['run_id'],
                      'training_steps': training_data['completed_steps'],
                      'checkpoint_saved': training_data['checkpoint_saved']}, ensure_ascii=False))
