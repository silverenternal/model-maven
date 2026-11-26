#!/usr/bin/env python3
"""Drive the installed Windows TUI through a small actual authoring journey."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import pyte
import torch
from winpty import PtyProcess

native = Path(sys.argv[1]).resolve()
cli, tui = native / 'bin' / 'm-model.exe', native / 'bin' / 'm-model-tui.exe'
with tempfile.TemporaryDirectory(prefix='mm-windows-authoring-') as temp:
    root = Path(temp)
    project = root / 'project'
    project.mkdir()
    env = dict(os.environ, MODEL_MAVEN_NVIM=os.environ['MM_CI_NVIM'], NO_COLOR='1',
               MODEL_MAVEN_REDUCE_MOTION='1', XDG_CONFIG_HOME=str(root/'config'),
               XDG_STATE_HOME=str(root/'state'), XDG_CACHE_HOME=str(root/'cache'))
    proc = PtyProcess.spawn([str(tui), '--skip-welcome', '--lang', 'en', '--cli', str(cli),
                             '--project', str(project)], cwd=str(project), env=env, dimensions=(40,140))
    screen = pyte.Screen(140,40)
    stream = pyte.Stream(screen)
    lock = threading.Lock()
    output = []
    def consume():
        try:
            while True:
                chunk = proc.read(4096)
                with lock:
                    output.append(chunk)
                    stream.feed(chunk)
        except (EOFError, OSError):
            pass
    threading.Thread(target=consume, daemon=True).start()
    def snapshot():
        with lock:
            return '\n'.join(screen.display)
    def wait_for(phrase, timeout=60):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            text = snapshot()
            if phrase in text:
                return
            if not proc.isalive():
                raise RuntimeError(f'TUI exited before {phrase!r}:\n{text}')
            time.sleep(.1)
        raise RuntimeError(f'TUI did not show {phrase!r}:\n{snapshot()}')
    try:
        wait_for('MODEL GRAPH')
        proc.write('H')
        wait_for('Two inputs + add')
        proc.write('2\r')
        wait_for('Draft')
        time.sleep(.5)
        proc.write('s')
        wait_for('Proposed graph save')
        proc.write('a')
        wait_for('Graph saved; model not run.')
        proc.write('g')
        wait_for('New Python module filename')
        proc.write('\r')
        wait_for('Proposed graph source')
        proc.write('a')
        wait_for('Source created. Configure its model factory?')
        graph, source = project/'model-graph.json', project/'graph_model.py'
        assert graph.is_file() and source.is_file(), 'TUI did not persist both artifacts'
        spec=importlib.util.spec_from_file_location('authored_model', source)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x=torch.tensor([[1.,2.],[2.,1.]])
        y=torch.tensor([[.1,.2],[.4,.3]])
        assert torch.allclose(module.model()(input=x,input_2=y)['output'], x+y), 'generated model output differs'
        print(json.dumps({'status':'success','action_count':7,'graph_bytes':graph.stat().st_size,
                          'source_bytes':source.stat().st_size,'matching_output':True}))
    finally:
        if proc.isalive(): proc.terminate(force=True)
