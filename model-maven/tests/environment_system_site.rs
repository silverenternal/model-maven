use std::{fs, path::Path, process::Command, time::Duration};

#[cfg(unix)]
#[path = "cli_ui_support/mod.rs"]
mod cli_ui_support;

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

fn build_wheel(root: &Path) {
    let script = r#"
import base64, csv, hashlib, io, pathlib, sys, zipfile
root = pathlib.Path(sys.argv[1])
distribution, package, version = 'fixture-dependency', 'fixture_dependency', '1.0.0'
dist_info = 'fixture_dependency-1.0.0.dist-info'
files = {
    f'{package}/__init__.py': "READY = True\n",
    f'{dist_info}/METADATA': f'Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n\n',
    f'{dist_info}/WHEEL': 'Wheel-Version: 1.0\nGenerator: model-maven-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n',
}
record = io.StringIO(newline='')
writer = csv.writer(record, lineterminator='\n')
filename = 'fixture_dependency-1.0.0-py3-none-any.whl'
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
    let output = Command::new(python())
        .arg("-c")
        .arg(script)
        .arg(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "wheel fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn system_site_policy_installs_project_wheel_and_checks_inherited_packages() {
    let project = tempfile::tempdir().unwrap();
    let wheelhouse = project.path().join("wheelhouse");
    fs::create_dir(&wheelhouse).unwrap();
    build_wheel(&wheelhouse);
    fs::write(
        project.path().join("requirements.txt"),
        "fixture-dependency==1.0.0\n",
    )
    .unwrap();
    let binary = env!("CARGO_BIN_EXE_m-model");
    let configured = Command::new(binary)
        .args(["--json", "env", "configure", "--system-site-packages"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        configured.status.success(),
        "{}",
        String::from_utf8_lossy(&configured.stdout)
    );
    let setup = Command::new(binary)
        .args(["--python", &python(), "--json", "env", "setup"])
        .env("PIP_NO_INDEX", "1")
        .env("PIP_FIND_LINKS", &wheelhouse)
        .env("PIP_NO_CACHE_DIR", "1")
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        setup.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&setup.stdout),
        String::from_utf8_lossy(&setup.stderr)
    );
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/environment.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(config["system_site_packages"], true);
    let pyvenv = fs::read_to_string(project.path().join(".venv/pyvenv.cfg")).unwrap();
    assert!(pyvenv.contains("include-system-site-packages = true"));
    let venv_python = project.path().join(".venv/bin/python");
    let inventory = Command::new(&venv_python)
        .args(["-c", "import fixture_dependency; import importlib.metadata as m; print(m.version('fixture-dependency'))"])
        .output()
        .unwrap();
    assert!(inventory.status.success());
    assert_eq!(String::from_utf8_lossy(&inventory.stdout).trim(), "1.0.0");
    let metadata: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/environment.lock.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(metadata["system_site_packages"]["enabled"], true);
    assert!(
        !metadata["system_site_packages"]["distributions"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let checked = Command::new(binary)
        .args(["--json", "env", "check"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&checked.stdout),
        String::from_utf8_lossy(&checked.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["data"]["environment_check"]["matches"], true);
    let locked_policy = Command::new(binary)
        .args([
            "--json",
            "env",
            "configure",
            "--isolate-system-site-packages",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    let locked_report: serde_json::Value = serde_json::from_slice(&locked_policy.stdout).unwrap();
    assert_eq!(
        locked_report["diagnostics"][0]["code"],
        "ENVIRONMENT_POLICY_LOCKED"
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(
            &fs::read(project.path().join(".model-maven/environment.json")).unwrap()
        )
        .unwrap()["system_site_packages"],
        true
    );

    let alternate_policy = Command::new(binary)
        .args([
            "--json",
            "env",
            "configure",
            "--isolate-system-site-packages",
            "--venv",
            ".isolated-venv",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        alternate_policy.status.success(),
        "{}",
        String::from_utf8_lossy(&alternate_policy.stdout)
    );
    let reset: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/environment.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(reset["system_site_packages"], false);
    assert_eq!(reset["venv"], ".isolated-venv");
    assert!(
        fs::read_to_string(project.path().join(".venv/pyvenv.cfg"))
            .unwrap()
            .contains("include-system-site-packages = true")
    );
    let isolated_setup = Command::new(binary)
        .args(["--json", "env", "setup", "--update-lock"])
        .env("PIP_NO_INDEX", "1")
        .env("PIP_FIND_LINKS", &wheelhouse)
        .env("PIP_NO_CACHE_DIR", "1")
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        isolated_setup.status.success(),
        "{}",
        String::from_utf8_lossy(&isolated_setup.stdout)
    );
    let isolated_cfg =
        fs::read_to_string(project.path().join(".isolated-venv/pyvenv.cfg")).unwrap();
    assert!(isolated_cfg.contains("include-system-site-packages = false"));
}

#[cfg(unix)]
#[test]
fn e4_environment_recovery_installs_missing_wheel_and_retries_same_preview() {
    if !has_torch() {
        eprintln!(
            "skipped E4 isolated-environment recovery: a real PyTorch base runtime is required"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let wheelhouse = project.path().join("wheelhouse");
    fs::create_dir(&wheelhouse).unwrap();
    build_wheel(&wheelhouse);
    fs::write(
        project.path().join("requirements.txt"),
        "fixture-dependency==1.0.0\n",
    )
    .unwrap();
    fs::write(
        project.path().join("recovery_model.py"),
        r#"
import torch
from torch import nn
from model_maven_worker.protocol import UserError

def build_model(): return nn.Linear(2, 2)
def failing_batch():
    try:
        import fixture_dependency
    except ImportError as error:
        raise UserError("IMAGE_DECODER_UNAVAILABLE", "fixture dependency unavailable", "prepare the project environment") from error
    return torch.ones(1, 2)
"#,
    )
    .unwrap();
    let binary = env!("CARGO_BIN_EXE_m-model");
    let configured = Command::new(binary)
        .args(["--json", "env", "configure", "--system-site-packages"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        configured.status.success(),
        "{}",
        String::from_utf8_lossy(&configured.stdout)
    );
    let init = Command::new(binary)
        .args([
            "--json",
            "init",
            "--model",
            "recovery_model:build_model",
            "--batch",
            "recovery_model:failing_batch",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stdout)
    );
    let input = "6\n4\nbatch\nrecovery_model:failing_batch\n\n\n\n\n\n\n\nstart\n4\nwait\nenv\nrequirements.txt\nsetup\nretry\nstart\n4\nwait\n返回首页\nq\n";
    let run = cli_ui_support::run_tty_with_env(
        Path::new(binary),
        project.path(),
        &[],
        &[
            ("PIP_NO_INDEX", "1"),
            ("PIP_FIND_LINKS", wheelhouse.to_str().unwrap()),
            ("PIP_NO_CACHE_DIR", "1"),
        ],
        input.as_bytes(),
        Duration::from_secs(180),
    )
    .unwrap();
    assert!(!run.timed_out, "{}", run.transcript);
    assert_eq!(run.exit_code, Some(0), "{}", run.transcript);
    for text in [
        "IMAGE_DECODER_UNAVAILABLE",
        "项目虚拟环境已创建并验证",
        "锁定 wheel 安装、pip check 和包指纹核对均通过",
        "后台预览重试已启动",
        "当前预览 · 本次请求已发布",
    ] {
        assert!(
            run.transcript.contains(text),
            "missing {text}: {}",
            run.transcript
        );
    }
    let current: serde_json::Value = serde_json::from_slice(
        &fs::read(project.path().join(".model-maven/previews/current.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(current["current"]["status"], "succeeded", "{current:#}");
    assert_eq!(current["current"]["generation"], 2, "{current:#}");
    assert_eq!(current["generation"], 2, "{current:#}");
    assert_eq!(
        current["current"]["request_id"],
        current["last_success"]["request_id"]
    );
    let venv_python = project.path().join(".venv/bin/python");
    let torch_check = Command::new(venv_python)
        .args([
            "-c",
            "import torch,fixture_dependency; print(torch.__version__, fixture_dependency.READY)",
        ])
        .output()
        .unwrap();
    assert!(
        torch_check.status.success(),
        "{}",
        String::from_utf8_lossy(&torch_check.stderr)
    );
}

#[cfg(unix)]
#[test]
fn empty_requirements_keep_the_empty_resolution_refusal_and_create_no_venv() {
    let project = tempfile::tempdir().unwrap();
    fs::write(project.path().join("requirements.txt"), "\n# no packages\n").unwrap();
    let binary = env!("CARGO_BIN_EXE_m-model");
    let output = Command::new(binary)
        .args(["--json", "env", "configure", "--system-site-packages"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let setup = Command::new(binary)
        .args(["--python", &python(), "--json", "env", "setup"])
        .current_dir(project.path())
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&setup.stdout).unwrap();
    assert_eq!(
        report["diagnostics"][0]["code"],
        "ENVIRONMENT_RESOLUTION_EMPTY"
    );
    assert!(!project.path().join(".venv").exists());
}
