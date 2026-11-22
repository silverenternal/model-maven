use std::{
    fs,
    io::{Read, Write},
    os::fd::FromRawFd,
    path::Path,
    process::{Command, Stdio},
};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn has_torch() -> bool {
    Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
}

fn fixture(root: &Path) {
    fs::write(
        root.join("interactive_component.py"),
        r#"
from torch import nn

def make_layer(width=0):
    if width < 1:
        return "invalid layer"
    return nn.Linear(width, width)
"#,
    )
    .unwrap();
}

fn register_component(root: &Path) {
    let schema = r#"{"type":"object","properties":{"width":{"type":"integer","minimum":0,"default":0,"description":"Feature width"}},"required":["width"],"additionalProperties":false}"#;
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "component",
            "add",
            "local/interactive-layer",
            "--kind",
            "layer",
            "--entrypoint",
            "interactive_component:make_layer",
            "--version",
            "1.0.0",
            "--params-schema",
            schema,
            "--capabilities",
            r#"{"construct":{"state":"supported"}}"#,
        ])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn register_resumable_component(root: &Path) {
    fs::write(
        root.join("resumable_component.py"),
        r#"
from torch import nn

def make_layer(a_width=0, z_out_features=0):
    if a_width < 1 or z_out_features < 1:
        return "invalid layer"
    return nn.Linear(a_width, z_out_features)
"#,
    )
    .unwrap();
    let schema = r#"{"type":"object","properties":{"a_width":{"type":"integer","minimum":1},"z_out_features":{"type":"integer","minimum":1}},"required":["a_width","z_out_features"],"additionalProperties":false}"#;
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "component",
            "add",
            "local/resumable-layer",
            "--kind",
            "layer",
            "--entrypoint",
            "resumable_component:make_layer",
            "--version",
            "1.0.0",
            "--params-schema",
            schema,
            "--capabilities",
            r#"{"construct":{"state":"supported"}}"#,
        ])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn init_with_component(root: &Path) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "init",
            "--model",
            "interactive_component:build_model",
            "--inject",
            "encoder=component:local/interactive-layer@1.0.0",
            "--inject-param",
            "encoder.width=4",
        ])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn write_unknown_run(root: &Path) {
    let run = root.join(".model-maven/runs/run-lost");
    fs::create_dir_all(&run).unwrap();
    fs::write(
        run.join("run.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version":1,
            "id":"run-lost",
            "name":"Lost supervisor",
            "kept":false,
            "operation":"train",
            "status":"running",
            "started_at_ms":1_000,
            "finished_at_ms":null,
            "requested_steps":10,
            "summary":{"completed_steps":1,"requested_steps":10},
            "inputs_file":"inputs.json"
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        run.join("inputs.json"),
        serde_json::to_vec(&serde_json::json!({
            "operation":"train",
            "project_root":root,
            "model":"interactive_component:build_model",
            "model_params":{},
            "model_injections":{},
            "runtime":{"python_executable":"python3"}
        }))
        .unwrap(),
    )
    .unwrap();
}

#[cfg(unix)]
fn workbench(root: &Path, input: &[u8]) -> String {
    let (mut master_fd, mut slave_fd) = (-1, -1);
    let opened = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(opened, 0);
    let slave = unsafe { fs::File::from_raw_fd(slave_fd) };
    let stdout = slave.try_clone().unwrap();
    let stderr = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--python", &python()])
        .current_dir(root)
        .stdin(Stdio::from(slave))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap();
    let mut terminal = unsafe { fs::File::from_raw_fd(master_fd) };
    let mut scripted_input = Vec::new();
    if root.join(".model-maven/ui-page-session.json").is_file() {
        scripted_input.extend_from_slice(b"2\n");
    } else if !root.join(".model-maven/project.json").is_file() {
        scripted_input.extend_from_slice(b"5\n");
    }
    scripted_input.extend_from_slice(input);
    terminal.write_all(&scripted_input).unwrap();
    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut descriptor = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut descriptor, 1, 60_000) };
        assert!(
            ready > 0,
            "workbench timeout:\n{}",
            String::from_utf8_lossy(&transcript)
        );
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read PTY: {error}"),
        }
    }
    let status = child.wait().unwrap();
    assert!(status.success(), "workbench exited with {status}");
    String::from_utf8_lossy(&transcript).into_owned()
}

#[cfg(unix)]
#[test]
fn component_catalog_details_and_retry_keep_selected_component_and_schema_parameters() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    register_component(project.path());

    let transcript = workbench(
        project.path(),
        b"4\n1\n\nshow\nlocal/interactive-layer\n1\n\ntest\nlocal/interactive-layer\n0\nretry\n1\n3\n8\n",
    );
    assert!(transcript.contains("组件目录 · 1 个组件"), "{transcript}");
    assert!(
        transcript.contains("local/interactive-layer@1.0.0"),
        "{transcript}"
    );
    assert!(transcript.contains("描述来源："), "{transcript}");
    assert!(
        transcript.contains("width · integer · 必填"),
        "{transcript}"
    );
    assert!(transcript.contains("forward"), "{transcript}");
    assert!(
        transcript.contains("组件详情 · local/interactive-layer@1.0.0"),
        "{transcript}"
    );
    assert!(transcript.contains("静态清单：passed"), "{transcript}");
    assert!(transcript.contains("运行时导入：not_run"), "{transcript}");
    assert!(transcript.contains("LAYER_TYPE_INVALID"), "{transcript}");
    assert!(transcript.contains("组件测试未通过"), "{transcript}");
    assert!(
        transcript.contains("构造通过 · 覆盖范围 complete_for_declared_checks"),
        "{transcript}"
    );
}

#[cfg(unix)]
#[test]
fn interrupted_component_schema_form_can_be_restored_and_explicitly_cancelled() {
    if !has_torch() {
        eprintln!("skipped PyTorch integration: component runtime needs PyTorch");
        return;
    }
    let project = tempfile::tempdir().unwrap();
    register_resumable_component(project.path());

    let interrupted = workbench(
        project.path(),
        b"4\n1\n\ntest\nlocal/resumable-layer@1.0.0\n8\n\x04\x04",
    );
    assert!(interrupted.contains("a_width"), "{interrupted}");
    let draft_path = project.path().join(".model-maven/ui-form-drafts.json");
    assert!(draft_path.is_file());
    let draft: serde_json::Value = serde_json::from_slice(&fs::read(&draft_path).unwrap()).unwrap();
    assert!(draft.to_string().contains("8"));
    assert!(!draft.to_string().contains("token"));

    let restored = workbench(
        project.path(),
        b"4\n1\n\ntest\nlocal/resumable-layer@1.0.0\n1\n\n4\n3\n8\n",
    );
    assert!(restored.contains("发现上次未提交的表单草稿"), "{restored}");
    assert!(
        restored.contains("当前值：8 · 来源：上次未提交的表单草稿"),
        "{restored}"
    );
    assert!(restored.contains("构造通过"), "{restored}");
    assert!(!draft_path.exists());

    let interrupted_again = workbench(
        project.path(),
        b"4\n1\n\ntest\nlocal/resumable-layer@1.0.0\n9\n\x04\x04",
    );
    assert!(
        interrupted_again.contains("z_out_features"),
        "{interrupted_again}"
    );
    assert!(draft_path.is_file());
    let cancelled = workbench(
        project.path(),
        b"4\n1\n\ntest\nlocal/resumable-layer@1.0.0\n1\n:cancel\n3\n8\n",
    );
    assert!(cancelled.contains("组件参数表单已取消"), "{cancelled}");
    assert!(!draft_path.exists());
}

#[cfg(unix)]
#[test]
fn component_install_failure_stays_recoverable_in_the_workbench() {
    let project = tempfile::tempdir().unwrap();
    let transcript = workbench(project.path(), b"4\n1\n\ninstall\nnot-a-package\n\n3\n8\n");
    assert!(transcript.contains("组件目录 · 0 个组件"), "{transcript}");
    assert!(
        transcript.contains("COMPONENT_INSTALL_PATH_INVALID"),
        "{transcript}"
    );
}

#[cfg(unix)]
#[test]
fn component_uninstall_preview_explains_project_reference_and_requires_explicit_force() {
    if !has_torch() {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    register_component(project.path());
    init_with_component(project.path());

    let transcript = workbench(
        project.path(),
        b"4\n1\n\nuninstall\nlocal/interactive-layer@1.0.0\nno\n3\n8\n",
    );
    assert!(
        transcript.contains("local/interactive-layer@1.0.0"),
        "{transcript}"
    );
    assert!(transcript.contains("project.json"), "{transcript}");
    assert!(transcript.contains("输入 FORCE"), "{transcript}");
    assert!(
        transcript.contains("已取消；组件和扩展文件未改变"),
        "{transcript}"
    );

    let still_registered = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python(),
            "component",
            "show",
            "local/interactive-layer@1.0.0",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(still_registered.status.success());
}

#[cfg(unix)]
#[test]
fn component_uninstall_preview_blocks_when_run_status_is_unknown() {
    let project = tempfile::tempdir().unwrap();
    fixture(project.path());
    register_component(project.path());
    write_unknown_run(project.path());

    let transcript = workbench(
        project.path(),
        b"4\n1\n\nuninstall\nlocal/interactive-layer@1.0.0\n3\n8\n",
    );
    assert!(transcript.contains("COMPONENT_ACTIVE_RUNS"), "{transcript}");
    assert!(transcript.contains("run-lost"), "{transcript}");
    assert!(transcript.contains("状态未知"), "{transcript}");
    assert!(
        !transcript.contains("输入完整组件 ID@版本确认移除"),
        "{transcript}"
    );
    assert!(!transcript.contains("输入 FORCE"), "{transcript}");
}
