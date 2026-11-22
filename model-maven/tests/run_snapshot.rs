use std::{path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

#[test]
fn snapshots_copy_variant_pinned_component_descriptors_and_reject_drift() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    let component_dir = project
        .path()
        .join(".model-maven/components/example/encoder/1.0.0");
    std::fs::create_dir_all(&component_dir).unwrap();
    let descriptor = component_dir.join("manifest.json");
    std::fs::write(
        &descriptor,
        b"{\"component_id\":\"example/encoder\",\"component_version\":\"1.0.0\"}\n",
    )
    .unwrap();
    std::fs::write(
        project.path().join("model.py"),
        b"def build_model(): return None\n",
    )
    .unwrap();

    let snapshot_root = project
        .path()
        .join(".model-maven/runs/snapshot-test/snapshot");
    let script = r#"
import hashlib, json, sys
from pathlib import Path
from model_maven_worker.run.snapshot import snapshot_project_code
root, target, descriptor = map(Path, sys.argv[1:])
digest = hashlib.sha256(descriptor.read_bytes()).hexdigest()
result = snapshot_project_code(root, target, {str(descriptor): digest})
copied = target / "code" / descriptor.relative_to(root)
assert copied.is_file() and hashlib.sha256(copied.read_bytes()).hexdigest() == digest
assert copied.stat().st_mode & 0o222 == 0
print(json.dumps(result))
"#;
    let repository_python = Path::new(env!("CARGO_MANIFEST_DIR")).join("python");
    let output = Command::new(python())
        .args([
            "-c",
            script,
            project.path().to_str().unwrap(),
            snapshot_root.to_str().unwrap(),
            descriptor.to_str().unwrap(),
        ])
        .env("PYTHONPATH", &repository_python)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["status"], "frozen");
    assert_eq!(report["files"], 2);
    assert!(report["manifest"].get("model.py").is_some());
    assert!(
        report["manifest"]
            .get(".model-maven/components/example/encoder/1.0.0/manifest.json")
            .is_some()
    );

    std::fs::write(&descriptor, b"{\"changed\":true}\n").unwrap();
    let drifted = Command::new(python())
        .args([
            "-c",
            r#"import hashlib, sys; from pathlib import Path; from model_maven_worker.protocol import UserError; from model_maven_worker.run.snapshot import snapshot_project_code; root,target,descriptor=map(Path,sys.argv[1:]);
try: snapshot_project_code(root,target,{str(descriptor): hashlib.sha256(b'old').hexdigest()})
except UserError as error: print(error.code, file=sys.stderr); raise SystemExit(2)"#,
            project.path().to_str().unwrap(),
            project.path()
                .join(".model-maven/runs/snapshot-test/drifted")
                .to_str()
                .unwrap(),
            descriptor.to_str().unwrap(),
        ])
        .env("PYTHONPATH", &repository_python)
        .output()
        .unwrap();
    assert!(!drifted.status.success());
    let stderr = String::from_utf8_lossy(&drifted.stderr);
    assert!(stderr.contains("SOURCE_CHANGED"), "{stderr}");
}
