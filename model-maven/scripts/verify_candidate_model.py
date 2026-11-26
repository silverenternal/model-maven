#!/usr/bin/env python3
"""Exercise a candidate CLI with a declared graph and a real local CSV batch.

The caller supplies an existing Python environment with PyTorch; this script
never creates an environment or installs a framework in a user's project.
"""
import csv
import hashlib
import json
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
    (root / 'real_batch.py').write_text('''import csv\nfrom pathlib import Path\nimport torch\n\ndef batch():\n    with (Path(__file__).parent / "batch.csv").open(newline="", encoding="utf-8") as source:\n        rows = list(csv.DictReader(source))[:2]\n    def tensor(columns):\n        return torch.tensor([[float(row[c]) for c in columns] for row in rows], dtype=torch.float32)\n    return {"args": [], "kwargs": {"input": tensor(["a", "b"]), "input_2": tensor(["c", "d"])}}\n''', encoding='utf-8')
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
    print(json.dumps({'platform': platform.platform(), 'python': sys.version.split()[0],
                      'graph_sha256': digest, 'csv_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                      'rows': 2, 'calls': len(calls), 'status': report['status']}, ensure_ascii=False))
