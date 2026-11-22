use std::{
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

fn session(project: &Path, input: &str, language: Option<&str>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_m-model-tui"));
    command
        .args([
            "--line-mode",
            "--cli",
            "/no-such-model-maven-cli",
            "--project",
        ])
        .arg(project)
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            project.join("preferences.json"),
        )
        .env("LC_ALL", "C")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(language) = language {
        command.args(["--lang", language]);
    }
    let mut child = command.spawn().unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn language_selection_is_immediate_saved_and_overridden_by_cli() {
    let project = tempfile::tempdir().unwrap();
    let path = project.path();
    let first = session(path, "\ns\n5\n2\nb\nq\n", None);
    assert!(String::from_utf8_lossy(&first.stdout).contains("语言  简体中文"));
    let prefs = path.join("preferences.json");
    let read =
        || serde_json::from_slice::<serde_json::Value>(&std::fs::read(&prefs).unwrap()).unwrap();
    assert_eq!(read()["language"], "zh-cn");
    let chinese = session(path, "\nq\n", None);
    assert!(String::from_utf8_lossy(&chinese.stderr).contains("进入工作台"));
    let overridden = session(path, "\ns\n5\n3\nb\nq\n", Some("en"));
    assert!(String::from_utf8_lossy(&overridden.stderr).contains("Open workbench"));
    assert!(String::from_utf8_lossy(&overridden.stdout).contains("語言  繁體中文"));
    assert_eq!(read()["language"], "zh-tw");
    let traditional = session(path, "\nq\n", None);
    assert!(String::from_utf8_lossy(&traditional.stderr).contains("進入工作臺"));
    let system = session(path, "\ns\n5\n0\nb\nq\n", None);
    assert!(String::from_utf8_lossy(&system.stdout).contains("Language  Follow system"));
    assert_eq!(read()["language"], "system");
    let english = session(path, "\ns\n5\n1\nb\nq\n", None);
    assert!(String::from_utf8_lossy(&english.stdout).contains("Language  English"));
    assert_eq!(read()["language"], "en");
}
