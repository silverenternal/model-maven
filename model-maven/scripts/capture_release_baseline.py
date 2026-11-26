#!/usr/bin/env python3
"""Record source and artifact identities without changing or running the product."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tomllib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL, timeout=30)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def repository(root, excluded_output):
    git = lambda *args: command(['git', '-C', str(root), *args])
    head = git('rev-parse', '--verify', 'HEAD^{commit}').decode().strip()
    patch = git('diff', '--binary', '--no-ext-diff', 'HEAD', '--')
    staged = git('diff', '--binary', '--no-ext-diff', '--cached', 'HEAD', '--')
    unstaged = git('diff', '--binary', '--no-ext-diff', '--')
    status = git('status', '--porcelain=v1', '--untracked-files=normal', '-z')
    raw_names = git('ls-files', '--others', '--exclude-standard', '-z')
    entries = []
    for raw in sorted(filter(None, raw_names.split(b'\0'))):
        name = os.fsdecode(raw)
        path = root / name
        if path.absolute() == excluded_output:
            continue
        info = path.lstat()
        if stat.S_ISREG(info.st_mode):
            identity = {'kind': 'file', 'sha256': digest_file(path), 'size': info.st_size,
                        'executable': bool(info.st_mode & 0o111)}
        elif stat.S_ISLNK(info.st_mode):
            identity = {'kind': 'symlink', 'target_sha256': hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()}
        else:
            raise ValueError(f'unsupported untracked filesystem entry: {name!r}')
        entries.append({'path': name, **identity})
    canonical = json.dumps(entries, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()
    return {'head': head, 'dirty': bool(status), 'tracked_patch_sha256': hashlib.sha256(patch).hexdigest(),
            'staged_patch_sha256': hashlib.sha256(staged).hexdigest(),
            'unstaged_patch_sha256': hashlib.sha256(unstaged).hexdigest(),
            'untracked_files': len(entries), 'untracked_manifest_sha256': hashlib.sha256(canonical).hexdigest(),
            'untracked_manifest': entries}


def version(*args):
    try:
        return command(list(args)).decode(errors='replace').strip()
    except (OSError, subprocess.SubprocessError):
        return None


def artifact(path):
    if not path.is_file():
        return {'path': str(path), 'status': 'absent'}
    return {'path': str(path), 'status': 'present_unverified', 'size': path.stat().st_size,
            'sha256': digest_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tui-source', type=Path, default=ROOT.parent / 'model-maven-tui')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--published-version', help='Observed npm version; omit if not independently checked')
    args = parser.parse_args()
    tui = args.tui_source.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error('output already exists; choose a new file')
    backend = repository(ROOT, output)
    workspace = repository(tui, output)
    contract = json.loads((ROOT / 'docs/contracts/compatibility.json').read_text(encoding='utf-8'))
    manifest = {
        'schema_version': 1,
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Local snapshot only. Untracked manifest excludes this output; artifact hashes do not prove their build inputs.',
        'repositories': {'backend': backend, 'workspace': workspace},
        'package_versions': {'backend': tomllib.loads((ROOT / 'Cargo.toml').read_text())['package']['version'],
                             'workspace': tomllib.loads((tui / 'Cargo.toml').read_text())['package']['version'],
                             'published_npm_observed': args.published_version},
        'protocol_contracts': [{'id': item.get('id'), 'version': item.get('version')}
                               for item in contract.get('interfaces', [])],
        'toolchains': {'rustc': version('rustc', '--version'), 'cargo': version('cargo', '--version'),
                       'python': sys.version.split()[0], 'node': version('node', '--version'), 'npm': version('npm', '--version')},
        'artifacts': {'backend': artifact(ROOT / 'target/release/m-model'),
                      'workspace': artifact(tui / 'target/release/m-model-tui')},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=True, sort_keys=True, indent=2)
        stream.write('\n')
    print(f'Wrote local baseline: {output}')


if __name__ == '__main__':
    main()
