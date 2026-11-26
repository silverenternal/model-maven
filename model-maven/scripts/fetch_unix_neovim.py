#!/usr/bin/env python3
"""Prepare a checksum-verified Neovim for isolated Unix CI use."""
import hashlib
import os
from pathlib import Path
import platform
import re
import tarfile
import tempfile
import urllib.request

system = {'Linux': 'linux', 'Darwin': 'macos'}[platform.system()]
arch = {'x86_64': 'x86_64', 'aarch64': 'arm64', 'arm64': 'arm64'}[platform.machine().lower()]
asset = f'nvim-{system}-{arch}.tar.gz'
base = f'https://github.com/neovim/neovim/releases/download/v0.10.4/{asset}'
def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'model-maven-ci'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()
match = re.match(r'\s*([0-9a-fA-F]{64})\b', fetch(base+'.sha256sum').decode('ascii'))
if not match:
    raise RuntimeError('invalid Neovim checksum')
payload = fetch(base)
if hashlib.sha256(payload).hexdigest() != match.group(1).lower():
    raise RuntimeError('Neovim checksum mismatch')
root = Path(tempfile.mkdtemp(prefix='mm-ci-nvim-'))
archive = root / asset
archive.write_bytes(payload)
with tarfile.open(archive, 'r:gz') as tar:
    for item in tar.getmembers():
        path = Path(item.name)
        link = Path(item.linkname)
        if path.is_absolute() or '..' in path.parts or (item.issym() or item.islnk()) and (link.is_absolute() or '..' in link.parts):
            raise RuntimeError('unsafe Neovim archive entry')
    tar.extractall(root)
executable = root / f'nvim-{system}-{arch}' / 'bin' / 'nvim'
if not executable.is_file():
    raise RuntimeError('Neovim executable missing')
with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as f:
    f.write(f'MM_CI_NVIM={executable}\n')
print(f'Prepared Neovim 0.10.4 for {system}-{arch}')
