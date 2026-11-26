#!/usr/bin/env python3
"""Fetch Neovim for the native Windows preview workflow; do not install it for users."""
import hashlib
import os
from pathlib import Path
import re
import tempfile
import urllib.request
import zipfile

version = 'v0.10.4'
asset = 'nvim-win64.zip'
base = f'https://github.com/neovim/neovim/releases/download/{version}/{asset}'
headers = {'User-Agent': 'model-maven-ci'}
def download(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        return response.read()
expected_text = download(base + '.sha256sum').decode('ascii')
match = re.match(r'\s*([0-9a-fA-F]{64})\b', expected_text)
if not match:
    raise RuntimeError('Neovim checksum response is invalid')
payload = download(base)
if hashlib.sha256(payload).hexdigest() != match.group(1).lower():
    raise RuntimeError('Neovim archive checksum mismatch')
root = Path(tempfile.mkdtemp(prefix='mm-ci-nvim-'))
archive = root / asset
archive.write_bytes(payload)
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or ':' in name:
            raise RuntimeError('Unsafe Neovim archive entry')
    z.extractall(root)
executable = root / 'nvim-win64' / 'bin' / 'nvim.exe'
if not executable.is_file():
    raise RuntimeError('Neovim executable is missing')
with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as f:
    f.write(f'MM_CI_NVIM={executable}\n')
print(f'Prepared Neovim {version} x64 (Windows ARM uses emulation).')
