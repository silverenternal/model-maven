#[cfg(unix)]
#[path = "cli_ui_support/mod.rs"]
mod cli_ui_support;

#[cfg(unix)]
mod unix_tests {
    use crate::cli_ui_support;
    use std::{
        fs,
        path::{Path, PathBuf},
        process::Command,
        time::Duration,
    };

    const VERIFIED_TORCHVISION_FIXTURE: &str = "/home/hugo/.cache/uv/archive-v0/sE1cpT8a1i1YS2hm";

    fn python() -> String {
        std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
    }

    fn fixture_path() -> PathBuf {
        std::env::var_os("MODEL_MAVEN_TEST_TORCHVISION_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(VERIFIED_TORCHVISION_FIXTURE))
    }

    fn python_path() -> String {
        let mut paths = vec![
            fixture_path(),
            Path::new(env!("CARGO_MANIFEST_DIR")).join("python"),
        ];
        if let Some(existing) = std::env::var_os("PYTHONPATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        std::env::join_paths(paths)
            .expect("fixture and repository paths must be valid PYTHONPATH entries")
            .to_string_lossy()
            .into_owned()
    }

    fn compatible_resnet_environment(python: &str) -> bool {
        Command::new(python)
            .args([
                "-c",
                "import torch, torchvision; from torchvision.models import resnet18; m=resnet18(weights=None); assert m.fc.in_features == 512",
            ])
            .env("PYTHONPATH", python_path())
            .status()
            .is_ok_and(|status| status.success())
    }

    fn write_class_images(root: &Path, python: &str) {
        for label in ["cats", "dogs"] {
            fs::create_dir_all(root.join("data").join(label)).unwrap();
        }
        let script = r#"from pathlib import Path
from PIL import Image
import sys
root = Path(sys.argv[1])
Image.new("RGB", (12, 12), (220, 20, 35)).save(root / "data/cats/sample.png")
Image.new("RGB", (12, 12), (10, 50, 220)).save(root / "data/dogs/sample.png")
"#;
        let status = Command::new(python)
            .args(["-c", script])
            .arg(root)
            .env("PYTHONPATH", python_path())
            .status()
            .unwrap();
        assert!(status.success(), "could not write real class-folder images");
    }

    #[test]
    fn resnet18_onboarding_builds_project_and_completes_real_forward_loss_backward() {
        let python = python();
        if !fixture_path().join("torchvision").is_dir() || !compatible_resnet_environment(&python) {
            eprintln!(
                "skipped real ResNet18 onboarding: verified torchvision fixture is unavailable or incompatible; set MODEL_MAVEN_TEST_TORCHVISION_PATH"
            );
            return;
        }

        let project = tempfile::tempdir().unwrap();
        write_class_images(project.path(), &python);
        let run = cli_ui_support::run_tty_with_env(
            Path::new(env!("CARGO_BIN_EXE_m-model")),
            project.path(),
            &["--python", python.as_str()],
            &[("PYTHONPATH", &python_path())],
            b"3\n2\ndata\n1\n1\nback\nq\n",
            Duration::from_secs(90),
        )
        .unwrap();
        assert!(!run.timed_out, "{}", run.transcript);
        assert_eq!(run.exit_code, Some(0), "{}", run.transcript);
        assert!(
            run.transcript.contains("model  ResNet"),
            "{}",
            run.transcript
        );
        assert!(
            run.transcript.contains("检测到 2 个类别目录"),
            "{}",
            run.transcript
        );
        assert!(
            run.transcript.contains("已创建可编辑的 ResNet18"),
            "{}",
            run.transcript
        );
        assert!(run.transcript.contains("前向证据"), "{}", run.transcript);

        let config: serde_json::Value = serde_json::from_slice(
            &fs::read(project.path().join(".model-maven/project.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config["model"], "classification:model");
        assert_eq!(config["batch"], "classification:batch");
        let source = fs::read_to_string(project.path().join("classification.py")).unwrap();
        assert!(source.contains("build_resnet18"));
        assert!(source.contains("NUM_CLASSES = 2"));
        assert!(project.path().join("data/cats/sample.png").is_file());
        assert!(project.path().join("data/dogs/sample.png").is_file());

        let trial = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args([
                "--python",
                python.as_str(),
                "--json",
                "try",
                "--step",
                "classification:step",
                "--steps",
                "1",
                "--training-mode",
                "managed-step",
            ])
            .env("PYTHONPATH", python_path())
            .current_dir(project.path())
            .output()
            .unwrap();
        assert!(
            trial.status.success(),
            "stdout={}\nstderr={}",
            String::from_utf8_lossy(&trial.stdout),
            String::from_utf8_lossy(&trial.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&trial.stdout).unwrap();
        assert_eq!(report["status"], "success", "{report:#}");
        assert_eq!(report["data"]["completed_steps"], 1, "{report:#}");
        assert!(report["data"]["mean_loss"].as_f64().is_some(), "{report:#}");
        assert!(
            report["data"]["gradient_observations"]
                .as_u64()
                .is_some_and(|count| count > 0),
            "{report:#}"
        );
        assert!(
            report["data"]["parameters_with_gradient"]
                .as_u64()
                .is_some_and(|count| count > 0),
            "{report:#}"
        );
    }
}
