use serde_json::Value;
use std::{fs, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn venv_python(venv: &std::path::Path) -> std::path::PathBuf {
    if cfg!(windows) {
        venv.join("Scripts/python.exe")
    } else {
        venv.join("bin/python")
    }
}

#[test]
fn library_inventory_reads_metadata_without_importing_or_binding_user_packages() {
    let project = tempfile::tempdir().unwrap();
    let packages = project.path().join("metadata-only-path");
    fs::create_dir_all(packages.join("model_maven_env_spy")).unwrap();
    fs::create_dir_all(packages.join("model_maven_env_spy-0.4.2.dist-info")).unwrap();
    fs::write(
        packages.join("model_maven_env_spy/__init__.py"),
        "import os\nfrom pathlib import Path\nPath(os.environ['MM_ENV_SPY_MARKER']).write_text('imported')\n",
    )
    .unwrap();
    fs::write(
        packages.join("model_maven_env_spy-0.4.2.dist-info/METADATA"),
        "Metadata-Version: 2.1\nName: model-maven-env-spy\nVersion: 0.4.2\n\n",
    )
    .unwrap();
    let marker = project.path().join("spy-was-imported");
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "--json",
            "env",
            "libraries",
            "--library",
            "model-maven-env-spy",
        ])
        .current_dir(project.path())
        .env("PYTHONPATH", &packages)
        .env("MM_ENV_SPY_MARKER", &marker)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["operation"], "environment_library_inventory");
    assert_eq!(report["status"], "success");
    let data = &report["data"];
    assert_eq!(data["inspection"], "distribution-metadata-only");
    assert_eq!(data["third_party_code_imported"], false);
    assert_eq!(data["selected_by"], "explicit-python");
    let library = data["libraries"].as_array().unwrap().first().unwrap();
    assert_eq!(library["distribution"], "model-maven-env-spy");
    assert_eq!(library["installed"], true);
    assert_eq!(library["installed_versions"][0]["version"], "0.4.2");
    assert_eq!(library["adapter_status"], "unknown");
    assert_eq!(library["auto_installed"], false);
    assert_eq!(library["auto_bound"], false);
    assert!(
        !marker.exists(),
        "inventory imported an optional third-party package"
    );
}

#[test]
fn library_inventory_rejects_invalid_names_and_unsafe_venv_paths() {
    let project = tempfile::tempdir().unwrap();
    let invalid_name = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "--json",
            "env",
            "libraries",
            "--library",
            "x;__import__('os')",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!invalid_name.status.success());
    let invalid_report: Value = serde_json::from_slice(&invalid_name.stdout).unwrap();
    assert_eq!(
        invalid_report["diagnostics"][0]["code"],
        "ENVIRONMENT_LIBRARY_QUERY_INVALID"
    );

    let unsafe_venv = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "env", "libraries", "--venv", "../outside"])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!unsafe_venv.status.success());
    let unsafe_report: Value = serde_json::from_slice(&unsafe_venv.stdout).unwrap();
    assert_eq!(
        unsafe_report["diagnostics"][0]["code"],
        "ENVIRONMENT_PATH_INVALID"
    );

    let missing_venv = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "env",
            "libraries",
            "--venv",
            "missing-environment",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!missing_venv.status.success());
    let missing_report: Value = serde_json::from_slice(&missing_venv.stdout).unwrap();
    assert_eq!(
        missing_report["diagnostics"][0]["code"],
        "ENVIRONMENT_VENV_NOT_FOUND"
    );
}

#[test]
fn library_inventory_runs_in_selected_interpreter_without_torch() {
    let project = tempfile::tempdir().unwrap();
    let venv = project.path().join("plain-python");
    let created = Command::new(python())
        .args(["-m", "venv"])
        .arg(&venv)
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        created.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&created.stdout),
        String::from_utf8_lossy(&created.stderr)
    );
    let selected_python = venv_python(&venv);
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            selected_python.to_str().unwrap(),
            "--json",
            "env",
            "libraries",
            "--library",
            "gymnasium",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "success");
    assert_eq!(report["data"]["selected_by"], "explicit-python");
    assert_eq!(
        report["data"]["python_executable"],
        selected_python.to_str().unwrap()
    );
    assert_eq!(report["data"]["libraries"][0]["installed"], false);
    assert_eq!(report["data"]["libraries"][0]["auto_bound"], false);

    let doctor = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            selected_python.to_str().unwrap(),
            "--json",
            "doctor",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(doctor.status.success());
    let doctor: Value = serde_json::from_slice(&doctor.stdout).unwrap();
    assert_eq!(doctor["data"]["torch_available"], false);
    assert!(
        doctor["data"]["capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["id"] == "environment.library_probe" && item["state"] == "supported")
    );
}

#[test]
fn explicit_library_probe_imports_only_named_candidates_and_reports_init_failure() {
    let project = tempfile::tempdir().unwrap();
    let packages = project.path().join("probe-path");
    let gym_module = packages.join("gymnasium");
    let gym_dist = packages.join("gymnasium-0.29.0.dist-info");
    let jax_module = packages.join("jax");
    let jax_dist = packages.join("jax-0.4.0.dist-info");
    fs::create_dir_all(&gym_module).unwrap();
    fs::create_dir_all(&gym_dist).unwrap();
    fs::create_dir_all(&jax_module).unwrap();
    fs::create_dir_all(&jax_dist).unwrap();
    let marker = project.path().join("gymnasium-imported");
    fs::write(
        gym_module.join("__init__.py"),
        format!(
            "from pathlib import Path\nprint('package startup log')\nPath({marker:?}).write_text('initialized')\n__version__ = 'probe-test'\n"
        ),
    )
    .unwrap();
    fs::write(
        gym_dist.join("METADATA"),
        "Metadata-Version: 2.1\nName: gymnasium\nVersion: 0.29.0\n\n",
    )
    .unwrap();
    fs::write(
        jax_module.join("__init__.py"),
        "raise RuntimeError('simulated native initialization failure')\n",
    )
    .unwrap();
    fs::write(
        jax_dist.join("METADATA"),
        "Metadata-Version: 2.1\nName: jax\nVersion: 0.4.0\n\n",
    )
    .unwrap();

    let inventory = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "--json",
            "env",
            "libraries",
            "--library",
            "gymnasium",
        ])
        .current_dir(project.path())
        .env("PYTHONPATH", &packages)
        .output()
        .unwrap();
    assert!(inventory.status.success());
    let inventory: Value = serde_json::from_slice(&inventory.stdout).unwrap();
    assert_eq!(inventory["data"]["third_party_code_imported"], false);
    assert_eq!(inventory["data"]["libraries"][0]["installed"], true);
    assert!(!marker.exists(), "metadata inventory imported gymnasium");

    let probe = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "--json",
            "env",
            "probe",
            "--library",
            "gymnasium",
            "--library",
            "jax",
        ])
        .current_dir(project.path())
        .env("PYTHONPATH", &packages)
        .output()
        .unwrap();
    assert!(
        probe.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&probe.stdout),
        String::from_utf8_lossy(&probe.stderr)
    );
    let report: Value = serde_json::from_slice(&probe.stdout).unwrap();
    assert_eq!(report["operation"], "environment_library_probe");
    assert_eq!(report["data"]["selected_by"], "explicit-python");
    assert_eq!(report["data"]["auto_installed"], false);
    assert_eq!(report["data"]["auto_bound"], false);
    assert_eq!(report["data"]["third_party_code_imported"], true);
    assert_eq!(report["data"]["probes"][0]["status"], "imported");
    assert_eq!(report["data"]["probes"][0]["import_version"], "probe-test");
    assert_eq!(
        report["data"]["probes"][0]["adapter_status"],
        "package-only"
    );
    assert_eq!(report["data"]["probes"][0]["auto_bound"], false);
    assert_eq!(report["data"]["probes"][1]["status"], "import_failed");
    assert_eq!(report["data"]["probes"][1]["failure_type"], "RuntimeError");
    assert!(
        report["data"]["probes"][1]["failure_reason"]
            .as_str()
            .unwrap()
            .contains("simulated native initialization failure")
    );
    assert!(
        marker.exists(),
        "explicit probe did not initialize gymnasium"
    );

    let unsupported = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "--json",
            "env",
            "probe",
            "--library",
            "model-maven-env-spy",
        ])
        .current_dir(project.path())
        .env("PYTHONPATH", &packages)
        .output()
        .unwrap();
    assert!(!unsupported.status.success());
    let unsupported: Value = serde_json::from_slice(&unsupported.stdout).unwrap();
    assert_eq!(
        unsupported["diagnostics"][0]["code"],
        "ENVIRONMENT_LIBRARY_PROBE_UNSUPPORTED"
    );

    let venv = project.path().join("probe-empty-environment");
    let created = Command::new(python())
        .args(["-m", "venv"])
        .arg(&venv)
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        created.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&created.stdout),
        String::from_utf8_lossy(&created.stderr)
    );
    let probe_python = venv_python(&venv);
    let absent = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            probe_python.to_str().unwrap(),
            "--json",
            "env",
            "probe",
            "--library",
            "timm",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(absent.status.success());
    let absent: Value = serde_json::from_slice(&absent.stdout).unwrap();
    assert_eq!(absent["data"]["probes"][0]["status"], "not_installed");
    assert_eq!(absent["data"]["probes"][0]["auto_installed"], false);
}

#[test]
fn opt_in_runtime_probe_reports_cpu_smoke_results_without_binding_a_backend() {
    let project = tempfile::tempdir().unwrap();
    let packages = project.path().join("runtime-probe-path");
    let torch = packages.join("torch");
    let torch_dist = packages.join("torch-0.0.1.dist-info");
    fs::create_dir_all(&torch).unwrap();
    fs::create_dir_all(&torch_dist).unwrap();
    fs::write(
        torch.join("__init__.py"),
        r#"
import os
__version__ = "runtime-probe-fixture"
float32 = object()
class Tensor:
    def __init__(self, values): self.values = values
    def __mul__(self, other):
        if os.environ.get("MM_TORCH_SMOKE_FAIL"):
            raise RuntimeError("simulated CPU arithmetic failure")
        return Tensor([a * b for a, b in zip(self.values, other.values)])
    def sum(self): return Tensor([sum(self.values)])
    def item(self): return self.values[0]
def tensor(values, dtype=None, device=None):
    assert device == "cpu"
    return Tensor(values)
"#,
    )
    .unwrap();
    fs::write(
        torch_dist.join("METADATA"),
        "Metadata-Version: 2.1\nName: torch\nVersion: 0.0.1\n\n",
    )
    .unwrap();
    fs::create_dir_all(packages.join("ray-2.0.0.dist-info")).unwrap();
    fs::create_dir_all(packages.join("ray")).unwrap();
    fs::write(packages.join("ray/__init__.py"), "__version__ = '2.0.0'\n").unwrap();
    fs::write(
        packages.join("ray-2.0.0.dist-info/METADATA"),
        "Metadata-Version: 2.1\nName: ray\nVersion: 2.0.0\n\n",
    )
    .unwrap();

    let run = |fail: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_m-model"));
        command
            .args([
                "--python",
                &python(),
                "--json",
                "env",
                "probe",
                "--library",
                "torch",
                "--check-runtime",
            ])
            .current_dir(project.path())
            .env("PYTHONPATH", &packages);
        if fail {
            command.env("MM_TORCH_SMOKE_FAIL", "1");
        }
        command.output().unwrap()
    };

    let passed = run(false);
    assert!(
        passed.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&passed.stdout),
        String::from_utf8_lossy(&passed.stderr)
    );
    let report: Value = serde_json::from_slice(&passed.stdout).unwrap();
    let data = &report["data"];
    assert_eq!(
        data["inspection"],
        "explicit-allowlisted-import-and-runtime-smoke"
    );
    assert_eq!(data["auto_installed"], false);
    assert_eq!(data["auto_bound"], false);
    assert_eq!(data["probes"][0]["status"], "imported");
    assert_eq!(data["probes"][0]["adapter_status"], "implemented");
    assert_eq!(data["probes"][0]["runtime_check"]["status"], "passed");
    assert_eq!(
        data["probes"][0]["runtime_check"]["profile"],
        "torch-cpu-tensor-arithmetic"
    );
    assert_eq!(data["probes"][0]["runtime_check"]["device"], "cpu");
    assert_eq!(data["probes"][0]["runtime_check"]["result"], 5.0);

    let failed = run(true);
    assert!(
        failed.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&failed.stdout),
        String::from_utf8_lossy(&failed.stderr)
    );
    let report: Value = serde_json::from_slice(&failed.stdout).unwrap();
    let probe = &report["data"]["probes"][0];
    assert_eq!(probe["status"], "imported");
    assert_eq!(probe["runtime_check"]["status"], "failed");
    assert_eq!(probe["runtime_check"]["failure_type"], "RuntimeError");
    assert!(
        probe["runtime_check"]["failure_reason"]
            .as_str()
            .unwrap()
            .contains("simulated CPU arithmetic failure")
    );
    assert_eq!(probe["auto_bound"], false);

    let unsupported = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "--json",
            "env",
            "probe",
            "--library",
            "ray",
            "--check-runtime",
        ])
        .current_dir(project.path())
        .env("PYTHONPATH", &packages)
        .output()
        .unwrap();
    assert!(unsupported.status.success());
    let report: Value = serde_json::from_slice(&unsupported.stdout).unwrap();
    assert_eq!(report["data"]["probes"][0]["status"], "imported");
    assert_eq!(
        report["data"]["probes"][0]["runtime_check"]["status"],
        "unsupported"
    );
    assert_eq!(report["data"]["probes"][0]["auto_bound"], false);

    let too_many = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python(),
            "--json",
            "env",
            "probe",
            "--library",
            "torch",
            "--library",
            "jax",
            "--library",
            "torchvision",
            "--library",
            "gymnasium",
            "--library",
            "ray",
            "--check-runtime",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!too_many.status.success());
    let report: Value = serde_json::from_slice(&too_many.stdout).unwrap();
    assert_eq!(
        report["diagnostics"][0]["code"],
        "ENVIRONMENT_RUNTIME_CHECK_LIMIT"
    );
}

#[test]
fn non_torch_runtime_profiles_execute_their_bounded_cpu_fixture_operations() {
    let project = tempfile::tempdir().unwrap();
    let packages = project.path().join("optional-runtime-fixtures");
    let write_distribution = |name: &str, import_name: &str, source: &str| {
        let package = packages.join(import_name);
        fs::create_dir_all(&package).unwrap();
        fs::write(package.join("__init__.py"), source).unwrap();
        let dist = packages.join(format!("{name}-0.0.1.dist-info"));
        fs::create_dir_all(&dist).unwrap();
        fs::write(
            dist.join("METADATA"),
            format!("Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.1\n\n"),
        )
        .unwrap();
    };

    write_distribution(
        "torch",
        "torch",
        "__version__='fixture'\nfloat32=object()\nclass Tensor:\n def __init__(self, values): self.values=values\n def tolist(self): return self.values\ndef tensor(values, **kwargs): return Tensor(values)\n",
    );
    write_distribution(
        "torchvision",
        "torchvision",
        "__version__='fixture'\nfrom . import ops\n",
    );
    fs::write(
        packages.join("torchvision/ops.py"),
        "class Indices:\n def tolist(self): return [0]\ndef nms(boxes,scores,threshold): return Indices()\n",
    )
    .unwrap();
    write_distribution("jax", "jax", "__version__='fixture'\n");
    fs::write(
        packages.join("jax/numpy.py"),
        "import builtins\nfloat32=object()\nclass Value:\n def __init__(self, values): self.values=values\n def __mul__(self, other): return Value([a*b for a,b in zip(self.values,other.values)])\n def block_until_ready(self): return self\n def __float__(self): return float(self.values[0])\ndef asarray(values, **kwargs): return Value(values)\ndef sum(value): return Value([builtins.sum(value.values)])\n",
    )
    .unwrap();
    write_distribution(
        "gymnasium",
        "gymnasium",
        "__version__='fixture'\nfrom . import spaces\n",
    );
    fs::write(
        packages.join("gymnasium/spaces.py"),
        "class Sample:\n shape=(2,)\nclass Box:\n def __init__(self, **kwargs): pass\n def sample(self): return Sample()\n",
    )
    .unwrap();
    write_distribution(
        "numpy",
        "numpy",
        "float32=object()\nclass Finite:\n def all(self): return True\ndef isfinite(value): return Finite()\n",
    );

    for (library, profile) in [
        ("torchvision", "torchvision-cpu-nms"),
        ("jax", "jax-cpu-array-arithmetic"),
        ("gymnasium", "gymnasium-box-sample"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args([
                "--python",
                &python(),
                "--json",
                "env",
                "probe",
                "--library",
                library,
                "--check-runtime",
            ])
            .current_dir(project.path())
            .env("PYTHONPATH", &packages)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{library}: stdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        let probe = &report["data"]["probes"][0];
        assert_eq!(probe["status"], "imported", "{library}: {probe}");
        assert_eq!(
            probe["runtime_check"]["status"], "passed",
            "{library}: {probe}"
        );
        assert_eq!(probe["runtime_check"]["profile"], profile);
        assert_eq!(probe["auto_bound"], false);
        assert_eq!(report["data"]["auto_installed"], false);
        assert_eq!(report["data"]["auto_bound"], false);
    }
}
