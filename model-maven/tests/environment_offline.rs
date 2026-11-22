use std::{fs, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn run_python(project: &std::path::Path, code: &str, args: &[&str]) -> std::process::Output {
    Command::new(python())
        .arg("-c")
        .arg(code)
        .args(args)
        .current_dir(project)
        .output()
        .expect("start Python offline environment fixture")
}

fn build_wheel(project: &std::path::Path, name: &str, package: &str, version: &str) {
    let script = r#"
import base64, csv, hashlib, io, pathlib, sys, zipfile
root = pathlib.Path(sys.argv[1])
distribution, package, version = sys.argv[2:]
normalized = distribution.replace('-', '_')
dist_info = f'{normalized}-{version}.dist-info'
files = {
    f'{package}/__init__.py': f"__version__ = '{version}'\n",
    f'{dist_info}/METADATA': f'Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n\n',
    f'{dist_info}/WHEEL': 'Wheel-Version: 1.0\nGenerator: model-maven-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n',
}
record = io.StringIO(newline='')
writer = csv.writer(record, lineterminator='\n')
filename = f'{normalized}-{version}-py3-none-any.whl'
with zipfile.ZipFile(root / filename, 'w', zipfile.ZIP_DEFLATED) as wheel:
    for name, content in files.items():
        raw = content.encode()
        wheel.writestr(name, raw)
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()
        writer.writerow([name, 'sha256=' + digest, len(raw)])
    record_name = f'{dist_info}/RECORD'
    writer.writerow([record_name, '', ''])
    wheel.writestr(record_name, record.getvalue().encode())
"#;
    let output = run_python(
        project,
        script,
        &[project.to_str().unwrap(), name, package, version],
    );
    assert!(
        output.status.success(),
        "wheel creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn torch_wheelhouse_archives_and_replays_offline() {
    let project = tempfile::tempdir().unwrap();
    let wheelhouse = project.path().join("wheelhouse");
    fs::create_dir(&wheelhouse).unwrap();
    build_wheel(&wheelhouse, "torch", "torch", "1.0.0");
    fs::write(project.path().join("requirements.txt"), "torch==1.0.0\n").unwrap();

    let exercise = r#"
import hashlib, json, os, pathlib, subprocess, sys
binary = sys.argv[1]
def run(*args):
    result = subprocess.run([binary, '--python', sys.executable, '--json', 'env', *args], capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert result.returncode == 0, (args, report, result.stderr)
    return report
locked = run('lock')
assert locked['data']['package_count'] == 1, locked
initial = run('wheelhouse', 'check')['data']
assert initial['index_available'] is False and initial['archive_available'] is False, initial
archived = run('wheelhouse', 'archive')['data']
assert archived['created'] and archived['archive_available'], archived
bundle = pathlib.Path(archived['path'])
index = json.loads((bundle / 'index.json').read_text())
wheel = bundle / index['files'][0]['filename']
original = wheel.read_bytes()
wheel.write_bytes(original + b'tampered')
corrupt = run('wheelhouse', 'check')['data']
assert corrupt['archive_available'] is False and corrupt['corrupt'], corrupt
wheel.write_bytes(original)
venv = pathlib.Path('.offline-venv')
subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
replay = run('wheelhouse', 'replay', '--venv', '.offline-venv')['data']
assert replay['offline'] and replay['environment_check']['matches'], replay
python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
subprocess.run([str(python), '-c', "import torch; assert torch.__version__ == '1.0.0'"], check=True)
print(json.dumps({'locked': locked['data'], 'replay': replay}))
"#;
    let output = Command::new(python())
        .arg("-c")
        .arg(exercise)
        .arg(env!("CARGO_BIN_EXE_m-model"))
        .env("PIP_NO_INDEX", "1")
        .env("PIP_FIND_LINKS", &wheelhouse)
        .env("PIP_NO_CACHE_DIR", "1")
        .env("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        .current_dir(project.path())
        .output()
        .expect("run torch wheelhouse scenario");
    assert!(
        output.status.success(),
        "torch wheelhouse scenario failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn explicit_lock_and_setup_support_non_torch_environment_without_binding_a_backend() {
    let project = tempfile::tempdir().unwrap();
    let wheelhouse = project.path().join("wheelhouse");
    fs::create_dir(&wheelhouse).unwrap();
    build_wheel(&wheelhouse, "sample-dep", "sample_dep", "1.0.0");
    fs::write(
        project.path().join("requirements.txt"),
        "sample-dep==1.0.0\n",
    )
    .unwrap();

    let exercise = r#"
import json, pathlib, subprocess, sys
binary = sys.argv[1]
def run(*args, python=False, success=True):
    command = [binary]
    if python:
        command += ['--python', sys.executable]
    command += ['--json', *args]
    result = subprocess.run(command, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert (result.returncode == 0) == success, (args, report, result.stderr)
    return report
locked = run('env', 'lock', '--requirements', 'requirements.txt', python=True)
metadata = json.loads(pathlib.Path('.model-maven/environment.lock.json').read_text())
assert locked['data']['package_count'] == 1, locked
assert [item['name'] for item in metadata['packages']] == ['sample-dep'], metadata
setup = run('env', 'setup', '--requirements', 'requirements.txt', python=True)
assert setup['data']['environment_check']['matches'] and setup['data']['package_count'] == 1, setup
doctor = run('doctor')
assert doctor['data']['torch_available'] is False, doctor
inventory = run('env', 'libraries', '--library', 'sample-dep')
assert inventory['data']['libraries'][0]['installed'] is True, inventory
model = run('model', 'inspect', '--model', 'example:build_model', success=False)
assert model['diagnostics'][0]['code'] == 'OPERATION_CAPABILITY_UNAVAILABLE', model
assert 'PyTorch' in model['data']['readiness']['readiness']['capabilities'][0]['reason'], model
print(json.dumps({'packages': metadata['packages'], 'model_error': model['diagnostics'][0]['code']}))
"#;
    let output = Command::new(python())
        .arg("-c")
        .arg(exercise)
        .arg(env!("CARGO_BIN_EXE_m-model"))
        .env("PIP_NO_INDEX", "1")
        .env("PIP_FIND_LINKS", &wheelhouse)
        .env("PIP_NO_CACHE_DIR", "1")
        .env("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        .current_dir(project.path())
        .output()
        .expect("run isolated non-Torch environment scenario");
    assert!(
        output.status.success(),
        "non-Torch environment scenario failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
