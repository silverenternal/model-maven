#!/usr/bin/env python3
"""Verify real npm optional-dependency selection against local release tarballs."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

artifacts = Path(sys.argv[1]).resolve()
packages = {}
for path in artifacts.glob('*.tgz'):
    with tarfile.open(path, 'r:gz') as tar:
        manifest = json.load(tar.extractfile('package/package.json'))
    packages[manifest['name']] = (manifest, path, hashlib.sha1(path.read_bytes()).hexdigest())
base = '@silverenternal/model-maven'
assert base in packages, 'launcher tarball missing'
assert set(packages[base][0]['optionalDependencies']) <= set(packages), 'platform tarballs missing'

class Registry(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        route = unquote(urlparse(self.path).path).lstrip('/')
        if route.startswith('tarballs/'):
            name = route.removeprefix('tarballs/')
            item = next((p for _, p, _ in packages.values() if p.name == name), None)
            if item:
                body, content_type = item.read_bytes(), 'application/octet-stream'
            else:
                self.send_error(404); return
        elif route in packages:
            manifest, path, sha1 = packages[route]
            version = manifest['version']
            data = dict(manifest, dist={'tarball': f'http://127.0.0.1:{self.server.server_port}/tarballs/{path.name}', 'shasum': sha1})
            body = json.dumps({'name': route, 'dist-tags': {'latest': version}, 'versions': {version: data}}).encode()
            content_type = 'application/json'
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = ThreadingHTTPServer(('127.0.0.1', 0), Registry)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    with tempfile.TemporaryDirectory(prefix='maven-npm-verify-') as tmp:
        env = dict(os.environ, npm_config_registry=f'http://127.0.0.1:{server.server_port}', npm_config_cache=str(Path(tmp) / 'cache'))
        subprocess.run(['npm', 'install', base, '--ignore-scripts', '--no-audit', '--no-fund'], cwd=tmp, env=env, check=True, shell=sys.platform == 'win32')
        installed = Path(tmp) / 'node_modules' / '@silverenternal'
        cpu = os.environ['MM_TARGET_ARCH']
        expected = f'model-maven-{sys.platform}-{cpu}'
        actual = {p.name for p in installed.glob('model-maven-*') if p.is_dir()}
        assert actual == {expected}, f'expected {expected}, installed {actual}'
        ext = '.cmd' if sys.platform == 'win32' else ''
        binaries = Path(tmp) / 'node_modules' / '.bin'
        for command in ('m-model', 'm-model-tui'):
            subprocess.run([str(binaries / (command + ext)), '--version'], env=env, check=True, shell=sys.platform == 'win32')
        script = Path(__file__).with_name('verify_windows_tui_authoring.py')
        subprocess.run([sys.executable, str(script), str(installed / expected)], env=env, check=True)
        print(f'npm selected and used {expected}')
finally:
    server.shutdown()
