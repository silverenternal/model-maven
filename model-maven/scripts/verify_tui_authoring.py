#!/usr/bin/env python3
"""Drive an installed TUI through a small actual authoring journey."""
import codecs
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
if os.name == "nt":
    from winpty import PtyProcess
else:
    from ptyprocess import PtyProcess

native = Path(sys.argv[1]).resolve()
suffix = '.exe' if os.name == 'nt' else ''
cli, tui = native / 'bin' / ('m-model' + suffix), native / 'bin' / ('m-model-tui' + suffix)
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
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        try:
            while True:
                chunk = proc.read(4096)
                if isinstance(chunk, bytes):
                    chunk = decoder.decode(chunk)
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
    def send(text):
        proc.write(text if os.name == 'nt' else text.encode('utf-8'))
    try:
        wait_for('MODEL GRAPH')
        send('H')
        wait_for('Two inputs + add')
        send('2\r')
        wait_for('Draft')
        time.sleep(.5)
        send('s')
        wait_for('Proposed graph save')
        send('a')
        wait_for('Graph saved; model not run.')
        send('g')
        wait_for('New Python module filename')
        send('\r')
        wait_for('Proposed graph source')
        send('a')
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
