#!/usr/bin/env python3
"""Check call identity evidence in line mode using a real CLI and PyTorch."""
import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
tui = os.environ.get('MODEL_MAVEN_TUI_BIN', str(root / 'target/debug/m-model-tui'))
cli = os.environ.get('MODEL_MAVEN_CLI', str(root.parent / 'model-maven/target/debug/m-model'))

for mode, language in (('stable', 'en'), ('cloned', 'en'), ('mutating', 'en'), ('shared', 'en'), ('fanout', 'en'), ('stable', 'zh-cn'), ('stable', 'zh-tw')):
    with tempfile.TemporaryDirectory(prefix='maven-identity-line-') as directory:
        project = Path(directory)
        (project / 'fixture.py').write_text('''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.src = nn.Linear(4, 48)
        self.identity = nn.Identity()
        self.head = nn.Linear(32, 8)
        if mode == "shared": self.again = self.src
        if mode == "fanout": self.branches = nn.ModuleList([nn.Linear(48, 48) for _ in range(32)])
        self.mode = mode
    def forward(self, x):
        value = self.identity(self.src(x))
        if self.mode == "cloned": value = value.clone()
        if self.mode == "mutating": value.add_(1)
        if self.mode == "fanout":
            for branch in self.branches: branch(value)
        return self.head(value)
def stable(): return Model("stable")
def cloned(): return Model("cloned")
def mutating(): return Model("mutating")
def shared(): return Model("shared")
def fanout(): return Model("fanout")
def batch(): return torch.ones(2, 4)
''', encoding='utf-8')
        env = os.environ.copy()
        env.update(MODEL_MAVEN_TUI_PREFERENCES=str(project / 'prefs.json'), COLUMNS='160')
        head_call = 36 if mode == 'fanout' else 4
        result = subprocess.run([tui, '--line-mode', '--lang', language, '--cli', cli],
            cwd=project, env=env, input=f'\n1\nfixture:{mode}\n2\n\nfixture:batch\nk\n/\nmodel.head\n{head_call}\n@2\n@3\nb\nq\n',
            text=True, capture_output=True, timeout=120, check=True)
        output = result.stdout
        assert {'en': 'Earlier tensor matches', 'zh-cn': '较早的张量匹配', 'zh-tw': '較早的張量符合'}[language] in output, output[-6000:]
        assert {'en': 'not proof of a producer or dataflow', 'zh-cn': '不能证明生产者或数据流', 'zh-tw': '不能證明生產者或資料流'}[language] in output, output[-6000:]
        heading = {'en': 'Earlier tensor matches', 'zh-cn': '较早的张量匹配', 'zh-tw': '較早的張量符合'}[language]
        later = {'en': 'Later input matches', 'zh-cn': '后续输入匹配', 'zh-tw': '後續輸入符合'}[language]
        head_matches = output.split(heading, 1)[1].split(later, 1)[0]
        source_matches = output.split(heading, 2)[2].split(heading, 1)[0]
        assert 'outputs → model.identity #1 · inputs.args[0] · @3' in source_matches, source_matches
        assert 'Filter  model.head' in output if language == 'en' else True
        if mode == 'fanout':
            assert 'outputs → model.branches.31 #1 · inputs.args[0] · @35' in source_matches, source_matches
            assert 'outputs → model.head #1 · inputs.args[0] · @36' in source_matches, source_matches
        if mode in ('stable', 'shared', 'fanout'):
            assert 'inputs.args[0] ← model.src #1 · outputs' in head_matches, head_matches
            assert 'inputs.args[0] ← model.identity #1 · outputs · @3' in head_matches, head_matches
            if mode == 'shared': assert 'shared instance; registration unknown' in output
        else:
            assert 'No validated matches recorded; a source may still exist.' in output, output[-6000:]
            assert 'inputs.args[0] ← model.src' not in head_matches
        print(f'{mode} {language}: validated line-mode partial identity evidence')
