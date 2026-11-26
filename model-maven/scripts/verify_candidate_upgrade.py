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
        public_env = dict(os.environ, npm_config_registry='https://registry.npmjs.org/', npm_config_cache=str(Path(tmp) / 'cache'))
        project = Path(tmp) / 'project'
        project.mkdir()
        saved = {'model-graph.json': b'{\"nodes\":[]}\n', 'data.csv': b'x,y\n1,2\n', 'model.py': b'def model(): return None\n'}
        for name, content in saved.items(): (project / name).write_bytes(content)
        def install(version, chosen_env):
            subprocess.run(['npm', 'install', base + '@' + version, '--ignore-scripts', '--no-audit', '--no-fund'], cwd=tmp, env=chosen_env, check=True, shell=sys.platform == 'win32')
        def version():
            executable = Path(tmp) / 'node_modules' / '.bin' / ('m-model.cmd' if sys.platform == 'win32' else 'm-model')
            result = subprocess.run([str(executable), '--version'], env=public_env, cwd=project, check=True, shell=sys.platform == 'win32', capture_output=True, text=True)
            return result.stdout.strip()
        install('0.1.0', public_env)
        assert version().endswith('0.1.0'), 'old version did not start'
        install(packages[base][0]['version'], env)
        installed = Path(tmp) / 'node_modules' / '@silverenternal'
        cpu = os.environ['MM_TARGET_ARCH']
        expected = f'model-maven-{sys.platform}-{cpu}'
        actual = {p.name for p in installed.glob('model-maven-*') if p.is_dir()}
        assert actual == {expected}, f'expected {expected}, installed {actual}'
        ext = '.cmd' if sys.platform == 'win32' else ''
        binaries = Path(tmp) / 'node_modules' / '.bin'
        for command in ('m-model', 'm-model-tui'):
            subprocess.run([str(binaries / (command + ext)), '--version'], env=env, check=True, shell=sys.platform == 'win32')
        assert version().endswith(packages[base][0]['version']), 'candidate did not start'
        install('0.1.0', public_env)
        assert version().endswith('0.1.0'), 'rollback did not start'
        assert all((project / name).read_bytes() == content for name, content in saved.items()), 'project contents changed'
        print(json.dumps({'platform': expected, 'upgrade': packages[base][0]['version'], 'rollback': '0.1.0', 'project_files_unchanged': True}))
finally:
    server.shutdown()
