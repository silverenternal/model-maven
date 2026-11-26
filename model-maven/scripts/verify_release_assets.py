import hashlib,json,tarfile
from pathlib import Path
root=Path('release')
expected={'@silverenternal/model-maven'} | {f'@silverenternal/model-maven-{p}' for p in ('linux-x64','linux-arm64','darwin-x64','darwin-arm64','win32-x64','win32-arm64')}
files=sorted(root.glob('*.tgz'))
if len(files)!=7: raise RuntimeError(f'expected seven package archives, got {len(files)}')
names=set();digests=set();rows=[]
for path in files:
    manifest_path=Path(str(path)+'.manifest.json')
    manifest=json.loads(manifest_path.read_text())
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get('sha256')!=sha or manifest.get('local_test_only') is not False:
        raise RuntimeError(f'package provenance or checksum invalid: {path.name}')
    with tarfile.open(path,'r:gz') as tar:
        package=json.load(tar.extractfile('package/package.json'))
    if package.get('version')!='0.2.0-alpha.1' or manifest.get('version')!=package['version']:
        raise RuntimeError(f'package version invalid: {path.name}')
    if package['name']!=manifest.get('package') or package['name'] in names:
        raise RuntimeError(f'package identity invalid: {path.name}')
    names.add(package['name'])
    digests.add(manifest['workspace']['digest'])
    rows.append(f'{sha}  {path.name}')
if names!=expected or len(digests)!=1:
    raise RuntimeError(f'package set or workspace identity mismatch: {names}, {digests}')
(root/'SHA256SUMS').write_text('\n'.join(rows)+'\n',encoding='ascii')
print('Validated seven matching release packages and their SHA-256 manifests')
