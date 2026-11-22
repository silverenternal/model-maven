use std::{fs, path::Path, process::Command};

#[test]
fn architecture_guard_rejects_domain_to_presentation_imports() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let fixture = tempfile::tempdir().unwrap();
    let offender = fixture.path().join("src/domain/offender.rs");
    fs::create_dir_all(offender.parent().unwrap()).unwrap();
    fs::write(
        &offender,
        "use crate::presentation::terminal;\nfn invalid() { let _ = terminal::render(); }\n",
    )
    .unwrap();
    let output = Command::new(python)
        .arg(repository.join("scripts/check_architecture.py"))
        .arg("--root")
        .arg(fixture.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("architecture boundary violations"),
        "{stderr}"
    );
    assert!(stderr.contains("presentation"), "{stderr}");
}

#[test]
fn architecture_guard_accepts_a_clean_domain_fixture() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let fixture = tempfile::tempdir().unwrap();
    let output = Command::new(python)
        .arg(repository.join("scripts/check_architecture.py"))
        .arg("--root")
        .arg(fixture.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn compatibility_manifest_matches_rust_and_worker_versions() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let output = Command::new(python)
        .arg(repository.join("scripts/check_compatibility.py"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
