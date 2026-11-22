use std::{
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn explicit_line_diagnosis_runs_with_redirected_input_and_output() {
    let directory = tempfile::tempdir().unwrap();
    let preferences = directory.path().join("preferences.json");
    for (locale, columns, trace, rechecked) in [
        ("en_US.UTF-8", "80", "Source trace", "Passed · 2×48"),
        ("zh_CN.UTF-8", "40", "来源追踪", "✓ 重验通过"),
        ("zh_TW.UTF-8", "40", "來源追蹤", "✓ 重驗通過"),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args(["--diagnosis-demo", "--line-mode"])
            .env("TERM", "xterm-ghostty")
            .env("MODEL_MAVEN_GRAPHICS", "pixels")
            .env("MODEL_MAVEN_TUI_PREFERENCES", &preferences)
            .env("LC_ALL", locale)
            .env("COLUMNS", columns)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let commands = concat!(
            "\r\nI\r\nU\r\nB\r\nB\r\nK\r\nM\r\nY\r\n",
            "\r\nJ\r\nE\r\nY\r\n1\r\nM\r\n",
            "\r\nC\r\nC\r\nB\r\nN\r\nE\r\nY\r\nX\r\nY\r\nB\r\n",
            "R\r\n3\r\nC\r\nC\r\nB\r\nL\r\nQ\r\n",
        );
        child
            .stdin
            .take()
            .unwrap()
            .write_all(commands.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        let transcript = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{transcript}");
        assert!(output.stdout.is_empty());
        assert!(!transcript.contains('\x1b'));
        assert!(!transcript.lines().any(|line| line.starts_with("> ")));
        assert!(
            !transcript
                .chars()
                .any(|c| ('\u{2800}'..='\u{28ff}').contains(&c))
        );
        assert!(transcript.contains(trace));
        assert!(transcript.contains(rechecked));
        assert!(transcript.contains("model.encoder.out[0] → net.head.in[0]"));
        let sessions: Vec<_> = transcript.split("Model-Maven").skip(1).collect();
        assert_eq!(sessions.len(), 3);
        assert!(
            sessions[1]
                .split("model.head #2")
                .next()
                .unwrap()
                .contains("model.head #1")
        );
        assert!(sessions[1].contains("net.head.weight"));
        let final_session = sessions[2];
        assert!(final_session.contains("net.merge.weight"));
        let v2 = final_session
            .find("structure snapshot v2")
            .or_else(|| final_session.find("结构快照 v2"))
            .or_else(|| final_session.find("結構快照 v2"))
            .unwrap();
        let v3 = final_session.find("v3 · repair-2").unwrap();
        let rejected = final_session.find("net.adapter.out[9]").unwrap();
        let truncated = final_session.find("rewire-truncated").unwrap();
        assert!(v2 < v3 && v3 < rejected && rejected < truncated);
        assert!(
            final_session[rejected..truncated].contains("net.adapter.out[0] → net.merge.in[0]")
        );
        assert!(final_session.contains("v2 · repair-1 · net.head.in[0]"));
        assert!(final_session.contains("v3 · repair-2 · net.merge.in[0]"));
        assert!(!transcript.contains("v4"));
    }
}

#[test]
fn redirected_input_requires_the_explicit_offline_line_mode() {
    let command = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_m-model-tui"));
        command.stdin(Stdio::null()).env("LC_ALL", "en_US.UTF-8");
        command
    };
    let output = command().arg("--diagnosis-demo").output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires an interactive terminal"));

    let output = command().arg("--line-mode").output().unwrap();
    assert!(output.status.success());
    assert!(!output.stderr.contains(&0x1b));

    let output = command()
        .args(["--diagnosis-demo", "--line-mode"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("model.head"));
}

#[test]
fn explicit_language_overrides_environment_in_line_welcome() {
    for (language, environment, expected) in [
        ("en", "zh_CN.UTF-8", "Open workbench"),
        ("zh-cn", "en_US.UTF-8", "进入工作台"),
        ("zh-tw", "en_US.UTF-8", "進入工作臺"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args([
                "--line-mode",
                "--lang",
                language,
                "--cli",
                "/does/not/exist/m-model",
            ])
            .env("LC_ALL", environment)
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected));
    }
}

#[cfg(unix)]
#[test]
fn line_workbench_browses_cli_reports_and_cancels_preview() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
printf '%s %s\n' "$3" "$4" >> calls.log
case "$3 $4" in
  'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"error","diagnostics":[{"code":"PROJECT_NOT_INITIALIZED","message":"fixture"}]}' ; exit 2 ;;
  'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.very_long_encoder_block_with_same_name.head","type":"Linear","editable_params":{"in_features":32}}]}}' ;;
  'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","data":{"calls":[{"alias_paths":["model.very_long_encoder_block_with_same_name.head"],"invocation_index":1,"inputs":{"kind":"dict","items":{"args":{"kind":"tuple","items":[{"kind":"tensor","shape":[2,32]}]}}}},{"alias_paths":["model.very_long_encoder_block_with_same_name.head"],"invocation_index":2,"inputs":{"kind":"dict","items":{"args":{"kind":"tuple","items":[{"kind":"tensor","shape":[2,64]}]}}}}],"failure_location":{"last_observed_module_call":{"alias_paths":["model.very_long_encoder_block_with_same_name.head"],"invocation_index":2,"module_type":"Linear","inputs":{"items":{"args":{"items":[{"kind":"tensor","shape":[2,48]}]}}}}}},"diagnostics":[{"code":"FORWARD_FAILED","message":"shape mismatch"}]}' ; exit 4 ;;
  'doctor ') printf '%s\n' '{"protocol_version":1,"operation":"doctor","status":"error","diagnostics":[{"code":"ENV_FAILED","message":"fixture environment diagnostic","hint":"verify fixture environment"}]}' ; exit 3 ;;
  'layer set') case " $* " in *' --batch-binding single '*) ;; *) exit 1 ;; esac
    case " $* " in *' --dry-run '*' --require-forward '*) ;; *) exit 1 ;; esac
    printf '%s\n' '{"protocol_version":1,"operation":"edit","status":"success","data":{"committed":false,"variant_name":"working","validation":{"forward":"passed"},"edit_effects":[{"path":"model.very_long_encoder_block_with_same_name.head","initialized":["weight"]}]}}' ;;
  *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    for locale in ["en_US.UTF-8", "zh_CN.UTF-8", "zh_TW.UTF-8"] {
        let _ = std::fs::remove_file(directory.path().join("calls.log"));
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args(["--line-mode", "--cli", backend.to_str().unwrap()])
            .current_dir(directory.path())
            .env("LC_ALL", locale)
            .env("COLUMNS", "40")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                directory.path().join("preferences.json"),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(b"\n1\nb\n1\nfixture:build\n6\n2\nfixture:build\nb\n\n:options\nsingle\ndataset-v1\nfixture:batch\n3\n9\nk\n/\nENV_FAILED\nd2\n/\nFORWARD_FAILED\nd1\n/\n2x64\n2\n/\n\nb\n8\n7\nb\n7\n48\n3\n0\nq\n").unwrap();
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{stdout}\n{stderr}");
        assert!(!stdout.contains('\x1b') && !stderr.contains('\x1b'));
        assert!(
            !stdout
                .chars()
                .chain(stderr.chars())
                .any(|c| ('\u{2800}'..='\u{28ff}').contains(&c))
        );
        assert!(
            stdout
                .replace('\n', "")
                .contains("model.very_long_encoder_block_with_same_name.head")
        );
        assert!(stdout.matches("FORWARD_FAILED").count() >= 2);
        assert!(stdout.contains("2x64"), "{stdout}");
        assert!(stdout.contains("d2. [doctor] ENV_FAILED"), "{stdout}");
        assert!(stdout.contains("d1. [probe] FORWARD_FAILED"), "{stdout}");
        let compact: String = stdout.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(compact.contains("fixtureenvironmentdiagnostic"), "{stdout}");
        assert!(compact.contains("verifyfixtureenvironment"), "{stdout}");
        let notice = match locale {
            "en_US.UTF-8" => "Report-leveldiagnostic;nocallassociationisasserted.",
            "zh_CN.UTF-8" => "这是报告级诊断，未关联到具体调用。",
            _ => "這是報告級診斷，未關聯至具體呼叫。",
        };
        assert!(compact.matches(notice).count() >= 2, "{stdout}");
        assert!(stdout.contains("\"invocation_index\": 2"), "{stdout}");
        assert_eq!(stdout.matches("committed  false").count(), 2);
        let saved_notice = match locale {
            "en_US.UTF-8" => "Saved preview from this session; it has not been revalidated.",
            "zh_CN.UTF-8" => "本次会话保存的预览；尚未重新验证。",
            _ => "本次工作階段儲存的預覽；尚未重新驗證。",
        };
        let unwrapped: String = stdout
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        let saved_notice: String = saved_notice
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        assert!(unwrapped.contains(&saved_notice));
        let identity_notice = match locale {
            "en_US.UTF-8" => {
                "Inspection snapshot is not verified against the preview base revision."
            }
            "zh_CN.UTF-8" => "检查快照尚未与预览基准版本核对。",
            _ => "檢查快照尚未與預覽基準版本核對。",
        };
        let identity_notice: String = identity_notice
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect();
        assert!(unwrapped.contains(&identity_notice));
        assert!(stdout.contains("2 × 48"));
        assert!(stderr.contains("fixture:build"));
        let calls = std::fs::read_to_string(directory.path().join("calls.log")).unwrap();
        assert_eq!(calls.matches("model inspect").count(), 1);
        assert_eq!(calls.matches("model probe").count(), 1);
        assert_eq!(calls.matches("doctor ").count(), 2);
        assert_eq!(calls.matches("layer set").count(), 1);
    }
}

#[cfg(unix)]
#[test]
fn running_probe_announces_context_before_the_cli_finishes() {
    use std::{
        io::{BufRead, BufReader},
        os::unix::fs::PermissionsExt,
        sync::mpsc,
        time::Duration,
    };
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r#"#!/bin/sh
case "$3 $4" in
  'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
  'model probe')
    i=0
    while [ ! -f release ] && [ "$i" -lt 100 ]; do sleep 0.05; i=$((i+1)); done
    printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"success","data":{}}' ;;
  *) exit 1 ;;
esac
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--line-mode", "--cli", backend.to_str().unwrap()])
        .current_dir(directory.path())
        .env("LC_ALL", "en_US.UTF-8")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\n2\nfixture:build\nfixture:batch\nq\n")
        .unwrap();
    let stderr = child.stderr.take().unwrap();
    let (sender, receiver) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let mut visible = String::new();
    while !visible.contains("Batch  fixture:batch") {
        match receiver.recv_timeout(Duration::from_secs(3)) {
            Ok(line) => {
                visible.push_str(&line);
                visible.push('\n');
            }
            Err(_) => break,
        }
    }
    let still_running = child.try_wait().unwrap().is_none();
    std::fs::write(directory.path().join("release"), "").unwrap();
    let output = child.wait_with_output().unwrap();
    drop(receiver);
    reader.join().unwrap();
    assert!(output.status.success());
    assert!(still_running, "CLI finished before the status was observed");
    assert!(visible.contains("Running CLI"), "{visible}");
    assert!(visible.contains("Probe a batch"), "{visible}");
    assert!(visible.contains("Model  fixture:build"), "{visible}");
    assert!(visible.contains("Batch  fixture:batch"), "{visible}");
}

#[cfg(unix)]
#[test]
fn line_workbench_reuses_model_and_batch_only_for_the_same_model() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r#"#!/bin/sh
printf '%s\n' "$*" >> calls.log
case "$3 $4" in
  'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
  'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{}}' ;;
  'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"success","data":{}}' ;;
  *) exit 1 ;;
esac
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--line-mode", "--cli", backend.to_str().unwrap()])
        .current_dir(directory.path())
        .env("LC_ALL", "en_US.UTF-8")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\n1\nfirst:build\n2\n\nfirst:batch\n2\n\n\n2\nsecond:build\n\n2\nfirst:build\nfirst:batch\n2\n\n:default\nq\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Current model: first:build"), "{stderr}");
    assert!(stderr.contains("Current batch: first:batch"), "{stderr}");
    let log = std::fs::read_to_string(directory.path().join("calls.log")).unwrap();
    let probes: Vec<_> = log
        .lines()
        .filter(|line| line.contains("model probe"))
        .collect();
    assert_eq!(probes.len(), 5, "{log}");
    assert_eq!(probes[0], probes[1]);
    assert!(probes[1].contains("--model first:build --batch first:batch"));
    assert!(probes[2].contains("--model second:build"));
    assert!(!probes[2].contains("--batch"));
    assert!(probes[3].contains("--model first:build --batch first:batch"));
    assert!(probes[4].contains("--model first:build"));
    assert!(!probes[4].contains("--batch"));
}

#[cfg(unix)]
#[test]
fn opening_another_project_resets_evidence_and_uses_the_new_directory() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let first = root.path().join("first project");
    let second = root.path().join("second project");
    std::fs::create_dir(&first).unwrap();
    std::fs::create_dir(&second).unwrap();
    let backend = root.path().join("m-model");
    let log = root.path().join("calls.log");
    std::fs::write(&backend, r#"#!/bin/sh
printf '%s|%s\n' "$PWD" "$*" >> "$TEST_LOG"
case "$3 $4" in
  'config history')
    case "$PWD" in *'second project') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"error","diagnostics":[{"code":"PROJECT_NOT_INITIALIZED","message":"fixture"}]}' ; exit 2 ;; esac
    printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
  'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear"}]}}' ;;
  'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","data":{"failure_location":{"last_observed_module_call":{"alias_paths":["model.head"],"module_type":"Linear"}}},"diagnostics":[{"code":"FORWARD_FAILED","message":"fixture"}]}' ; exit 4 ;;
  'layer set') printf '%s\n' '{"protocol_version":1,"operation":"edit","status":"success","data":{"committed":false}}' ;;
  *) exit 1 ;;
esac
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--line-mode", "--cli", backend.to_str().unwrap()])
        .current_dir(&first)
        .env("LC_ALL", "en_US.UTF-8")
        .env("TEST_LOG", &log)
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            root.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"\n1\nfirst:build\n2\n\nfirst:batch\n7\n48\no\nmissing\nb\n9\no\n.\n9\no\n../second project\n9\n0\n4\n2\n\n\nq\n").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Could not open project"), "{stdout}");
    assert!(
        stdout.matches("Probe failed  FORWARD_FAILED").count() >= 3,
        "{stdout}"
    );
    let after_switch = stdout.split_once("Project not configured").unwrap().1;
    assert!(
        after_switch.contains("Model  Not selected"),
        "{after_switch}"
    );
    assert!(
        after_switch.contains("Batch  Not selected"),
        "{after_switch}"
    );
    assert_eq!(
        after_switch.matches("Probe failed  FORWARD_FAILED").count(),
        1,
        "{after_switch}"
    );
    let log = std::fs::read_to_string(log).unwrap();
    let lines: Vec<_> = log.lines().collect();
    assert_eq!(
        lines
            .iter()
            .filter(|line| line.contains("config history"))
            .count(),
        2,
        "{log}"
    );
    let last_probe = lines
        .iter()
        .rev()
        .find(|line| line.contains("model probe"))
        .unwrap();
    assert!(last_probe.starts_with(second.canonicalize().unwrap().to_str().unwrap()), "{log}");
    assert!(
        !last_probe.contains("--model") && !last_probe.contains("--batch"),
        "{log}"
    );
    assert_eq!(
        lines
            .iter()
            .filter(|line| line.contains("layer set"))
            .count(),
        1,
        "{log}"
    );
}

#[cfg(unix)]
#[test]
fn project_initialization_requires_matching_preview_and_explicit_confirmation() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/usr/bin/python3
import json, os, pathlib, sys
args=sys.argv[3:]
project=pathlib.Path.cwd()
path=project/'.model-maven'/'project.json'
def out(operation,status,data=None,diagnostics=None):
 print(json.dumps({'protocol_version':1,'operation':operation,'status':status,'data':data or {},'diagnostics':diagnostics or []}))
if args[:2]==['config','history']:
 if path.exists(): out('config_history','success')
 else: out('config_history','error',diagnostics=[{'code':'PROJECT_NOT_INITIALIZED','message':'fixture'}]); sys.exit(2)
elif args[0]=='init':
 model=args[args.index('--model')+1]
 batch=args[args.index('--batch')+1] if '--batch' in args else None
 config={'schema_version':1,'model':model,'batch':batch,'batch_binding':None,'model_injections':{}}
 if os.environ.get('TEST_MALFORMED'): config['model']='other:factory'
 dry='--dry-run' in args
 if not dry: path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(config))
 out('init','success',{'committed':not dry,'path':str(path),'config':config})
else: sys.exit(1)
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let run = |commands: &[u8], malformed: bool| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args(["--line-mode", "--cli", backend.to_str().unwrap()])
            .current_dir(directory.path())
            .env("LC_ALL", "en_US.UTF-8")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                directory.path().join("preferences.json"),
            )
            .env("TEST_MALFORMED", if malformed { "1" } else { "" })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(commands).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap()
    };
    let config = directory.path().join(".model-maven/project.json");
    let cancelled = run(b"\nn\nfixture:build\nfixture:batch\n\nq\n", false);
    assert!(cancelled.contains("Project setup preview"));
    assert!(cancelled.contains("Preview does not import or validate"));
    assert!(!config.exists());
    let malformed = run(b"\nn\nfixture:build\nfixture:batch\ny\nq\n", true);
    assert!(malformed.contains("Unexpected initialization preview"));
    assert!(!config.exists());
    let saved = run(b"\nn\nfixture:build\nfixture:batch\ny\nq\n", false);
    assert!(saved.contains("Project defaults saved"), "{saved}");
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(config).unwrap()).unwrap();
    assert_eq!(data["model"], "fixture:build");
    assert_eq!(data["batch"], "fixture:batch");
}

#[cfg(unix)]
#[test]
fn project_check_reports_missing_cli_and_can_retry_without_restart() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    let preferences = directory.path().join("preferences.json");
    let run = |backend: &std::path::Path, input: &[u8]| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args(["--line-mode", "--cli", backend.to_str().unwrap()])
            .current_dir(directory.path())
            .env("LC_ALL", "en_US.UTF-8")
            .env("MODEL_MAVEN_TUI_PREFERENCES", &preferences)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(input).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap()
    };
    let missing = run(&backend, b"\nc\n\nq\n");
    assert!(missing.contains("Project status unavailable"), "{missing}");
    assert!(
        missing.contains("Next  c. Project and CLI status"),
        "{missing}"
    );
    assert!(missing.contains("Could not run the CLI"), "{missing}");
    assert!(missing.contains("MODEL_MAVEN_CLI"), "{missing}");
    std::fs::write(&backend, r#"#!/bin/sh
if [ ! -f .first_check ]; then
  touch .first_check
  printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"error","diagnostics":[{"code":"TEMPORARY","message":"transient failure","hint":"retry the check"}]}'
  exit 2
fi
printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}'
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let recovered = run(&backend, b"\nc\nr\n\nq\n");
    assert!(
        recovered.contains("[TEMPORARY] transient failure"),
        "{recovered}"
    );
    assert!(recovered.contains("retry the check"), "{recovered}");
    assert!(recovered.contains("Project configured"), "{recovered}");
    assert!(recovered.contains("Next  2. Probe a batch"), "{recovered}");
    std::fs::write(&backend, r#"#!/bin/sh
case "$3 $4" in
  'config history')
    if [ -f .revision_seen ]; then revision=v2; else revision=v1; touch .revision_seen; fi
    printf '{"protocol_version":1,"operation":"config_history","status":"success","data":{"current_revision":"%s"}}\n' "$revision" ;;
  'model probe')
    printf '%s\n' "$*" >> probes.log
    printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","diagnostics":[{"code":"FORWARD_FAILED","message":"fixture"}]}' ; exit 4 ;;
  *) exit 1 ;;
esac
"#).unwrap();
    let changed = run(
        &backend,
        b"\n2\nold:build\nold:batch\nc\nr\n\n9\n2\n\n\nq\n",
    );
    assert!(
        changed
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .contains("sessionevidencewascleared"),
        "{changed}"
    );
    let probes = std::fs::read_to_string(directory.path().join("probes.log")).unwrap();
    let calls: Vec<_> = probes.lines().collect();
    assert_eq!(calls.len(), 2, "{probes}");
    assert!(
        calls[0].contains("--model old:build --batch old:batch"),
        "{probes}"
    );
    assert!(
        !calls[1].contains("--model") && !calls[1].contains("--batch"),
        "{probes}"
    );
}

#[cfg(unix)]
#[test]
fn truncated_input_search_requires_an_explicit_recorded_input_choice() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r#"#!/bin/sh
case "$3 $4" in
  'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
  'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear"}]}}' ;;
  'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"success","data":{"calls":[{"alias_paths":["model.head"],"invocation_index":1,"inputs":{"kind":"dict","items":{"args":{"kind":"tuple","truncated":3,"items":[{"kind":"tensor","shape":[2,64]}]}}}}]}}' ;;
  *) exit 1 ;;
esac
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--line-mode", "--cli", backend.to_str().unwrap()])
        .current_dir(directory.path())
        .env("LC_ALL", "en_US.UTF-8")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\n1\nfixture:build\n2\n\nfixture:batch\nk\n/\n2x64\n1\nb\n1\n1\nb\nq\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.matches("Choose matching input").count() >= 2,
        "{stdout}"
    );
    assert!(stdout.contains("Input trace incomplete"), "{stdout}");
    assert!(
        stdout.contains("Selected input  inputs.args[0]"),
        "{stdout}"
    );
}

#[cfg(unix)]
#[test]
fn timed_out_probe_stops_its_process_group_and_preserves_prior_report_as_stale() {
    use std::{
        os::unix::fs::PermissionsExt,
        time::{Duration, Instant},
    };
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r#"#!/usr/bin/python3
import json, os, pathlib, sys, time
args=sys.argv[3:]
def out(operation,status,data=None,diagnostics=None):
 print(json.dumps({'protocol_version':1,'operation':operation,'status':status,'data':data or {},'diagnostics':diagnostics or []}),flush=True)
if args[:2]==['config','history']: out('config_history','success')
elif args[0]=='doctor': out('doctor','success')
elif args[:2]==['model','probe']:
 marker=pathlib.Path('probe-count')
 if not marker.exists():
  marker.write_text('1')
  out('probe','error',{'padding':'x'*200000,'failure_location':{'last_observed_module_call':{'alias_paths':['model.head'],'module_type':'Linear'}}},[{'code':'FORWARD_FAILED','message':'original evidence'}]); sys.exit(4)
 child=os.fork()
 if child==0:
  time.sleep(0.6); pathlib.Path('child-survived').write_text('unexpected'); os._exit(0)
 time.sleep(3)
else: sys.exit(1)
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let start = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--line-mode", "--cli", backend.to_str().unwrap()])
        .current_dir(directory.path())
        .env("LC_ALL", "en_US.UTF-8")
        .env("MODEL_MAVEN_TUI_CLI_TIMEOUT_MS", "200")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\n2\nfixture:build\nfixture:batch\n2\n\n\n9\n3\nq\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(start.elapsed() < Duration::from_secs(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("CLI operation timed out"), "{stdout}");
    assert!(
        stdout.contains("MODEL_MAVEN_TUI_CLI_TIMEOUT_MS"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Previous diagnosis · latest attempt timed out"),
        "{stdout}"
    );
    assert!(stdout.matches("original evidence").count() >= 2, "{stdout}");
    let stale = stdout
        .split_once("Previous diagnosis · latest attempt timed out")
        .unwrap()
        .1;
    assert!(!stale.contains("7. Preview Linear input size"), "{stale}");
    assert!(stale.contains("Completed"), "{stale}");
    std::thread::sleep(Duration::from_millis(700));
    assert!(
        !directory.path().join("child-survived").exists(),
        "timed-out descendant survived"
    );
}

#[cfg(unix)]
#[test]
fn confirmed_project_save_is_not_interrupted_by_the_read_only_timeout() {
    use std::{
        os::unix::fs::PermissionsExt,
        time::{Duration, Instant},
    };
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r#"#!/usr/bin/python3
import json,pathlib,sys,time
args=sys.argv[3:]; path=pathlib.Path.cwd()/'.model-maven'/'project.json'
def out(operation,status,data=None,diagnostics=None):
 print(json.dumps({'protocol_version':1,'operation':operation,'status':status,'data':data or {},'diagnostics':diagnostics or []}),flush=True)
if args[:2]==['config','history']:
 if path.exists(): out('config_history','success')
 else: out('config_history','error',diagnostics=[{'code':'PROJECT_NOT_INITIALIZED','message':'fixture'}]);sys.exit(2)
elif args[0]=='init':
 config={'schema_version':1,'model':'fixture:build','batch':None,'batch_binding':None,'model_injections':{}}
 dry='--dry-run' in args
 if not dry:
  time.sleep(2.0);path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(config))
 out('init','success',{'committed':not dry,'path':str(path),'config':config})
else: sys.exit(1)
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let started = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--line-mode", "--cli", backend.to_str().unwrap()])
        .current_dir(directory.path())
        .env("LC_ALL", "en_US.UTF-8")
        .env("MODEL_MAVEN_TUI_CLI_TIMEOUT_MS", "1500")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\nn\nfixture:build\n\ny\nq\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(started.elapsed() >= Duration::from_millis(2000), "stdout={} stderr={}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("Project defaults saved")
    );
    assert!(directory.path().join(".model-maven/project.json").exists());
}

#[cfg(unix)]
#[test]
fn reused_probe_evidence_is_labeled_in_report_list_and_call_detail() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
  'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{"current_revision":"v1"}}' ;;
  'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear"}]}}' ;;
  'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"success","data":{"evidence_cache":{"status":"reused"},"calls":[{"alias_paths":["model.head"],"invocation_index":1}]}}' ;;
  *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    for (locale, notice) in [
        ("en_US.UTF-8", "Reusedevidence·nonewforward"),
        ("zh_CN.UTF-8", "复用证据·本次未执行前向"),
        ("zh_TW.UTF-8", "重用證據·本次未執行前向"),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args(["--line-mode", "--cli", backend.to_str().unwrap()])
            .current_dir(directory.path())
            .env("LC_ALL", locale)
            .env("COLUMNS", "40")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                directory.path().join("preferences.json"),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"\n1\nfixture:build\n2\n\nfixture:batch\nk\n1\nb\nq\n")
            .unwrap();
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        let compact: String = stdout.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(output.status.success(), "{stdout}");
        assert!(compact.contains("model.head"), "{stdout}");
        // The list and selected-call detail must independently retain the warning.
        assert!(compact.matches(notice).count() >= 2, "{stdout}");
        let report_notice = match locale {
            "en_US.UTF-8" => "Reusedevidence·nonewforwardwasrun.",
            "zh_CN.UTF-8" => "复用历史证据·本次未执行前向。",
            _ => "重用歷史證據·本次未執行前向。",
        };
        assert!(compact.contains(report_notice), "{stdout}");
    }
}

#[cfg(unix)]
#[test]
fn probe_options_follow_matching_batch_and_do_not_leak_to_another_batch() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
  'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{"current_revision":"v1"}}' ;;
  'model probe') printf '%s\n' "$*" >> calls.log
    printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"success","data":{}}' ;;
  *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(directory.path())
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"\n2\nfixture:model\n:options\nsingle\ndataset-v1\nfixture:batch\n2\n\n\n2\n\nfixture:other\n2\n\n:options\nargs\nchanged\nb\nfixture:another\nfixture:other\nq\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(stderr.contains("update when data changes"), "{stderr}");
    let calls = std::fs::read_to_string(directory.path().join("calls.log")).unwrap();
    let calls: Vec<_> = calls.lines().collect();
    assert_eq!(calls.len(), 4, "{calls:?}");
    for call in &calls[..2] {
        assert!(
            call.contains(
                "--batch fixture:batch --batch-binding single --batch-fingerprint dataset-v1"
            ),
            "{call}"
        );
    }
    assert!(calls[2].contains("--batch fixture:other"));
    assert!(!calls[2].contains("--batch-binding") && !calls[2].contains("--batch-fingerprint"));
    assert!(calls[3].contains("--model fixture:another --batch fixture:other"));
    assert!(!calls[3].contains("--batch-binding") && !calls[3].contains("--batch-fingerprint"));
}

#[cfg(unix)]
#[test]
fn repeated_probe_compares_only_reported_stable_batch_identities() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model probe')
   count=0; test ! -f count || read -r count < count
   count=$((count + 1)); printf '%s\n' "$count" > count
   case "$count" in
    1|2) data='{"batch_id":"batch-a","batch_identity_stable":true}' ;;
    3) data='{"batch_id":"batch-b","batch_identity_stable":true}' ;;
    4) data='{"batch_id":"batch-b","batch_identity_stable":false}' ;;
    *) data='{}' ;;
   esac
   printf '{"protocol_version":1,"operation":"probe","status":"success","data":%s}\n' "$data" ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(directory.path())
        .env("COLUMNS", "100")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\n2\nfixture:model\nfixture:batch\n2\n\n\n2\n\n\n2\n\n\n2\n\n\n9\n2\n\n:options\nsingle\nchanged-data\n\nq\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}");
    assert_eq!(
        stdout
            .matches("Batch ID matches previous probe report")
            .count(),
        1,
        "{stdout}"
    );
    assert_eq!(
        stdout
            .matches("Batch ID differs from previous probe report")
            .count(),
        1,
        "{stdout}"
    );
    // Last diagnosis can be revisited; its uncertainty must remain visible.
    assert_eq!(
        stdout
            .matches("Batch identity cannot be compared with previous probe")
            .count(),
        3,
        "{stdout}"
    );
    assert!(stdout.contains("Reported batch ID  batch-a"), "{stdout}");
    assert!(
        stdout.contains("Batch identity is not stable across runs."),
        "{stdout}"
    );
}

#[cfg(unix)]
#[test]
fn workbench_command_search_respects_availability_and_custom_keys() {
    use std::os::unix::fs::PermissionsExt;
    let project = tempfile::tempdir().unwrap();
    let backend = project.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model inspect') printf '%s\n' "$*" >> calls.log
  printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{}}' ;;
 'layer set') printf '%s\n' 'UNEXPECTED EDIT' >> calls.log; exit 1 ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let prefs = project.path().join("preferences.json");
    std::fs::write(&prefs, r#"{"workbench_keys":{"default":{"inspect":"v"}}}"#).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(project.path())
        .env("MODEL_MAVEN_TUI_PREFERENCES", &prefs)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\n/\npreview\n7\nb\n/\ninspect\nv\nfixture:model\nq\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let screen = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{screen}");
    assert!(
        screen.contains("! 7. Preview Linear input size"),
        "{screen}"
    );
    assert!(screen.contains("Command unavailable"), "{screen}");
    assert!(screen.contains("v. Inspect model structure"), "{screen}");
    let calls = std::fs::read_to_string(project.path().join("calls.log")).unwrap();
    assert!(
        calls.contains("model inspect --model fixture:model"),
        "{calls}"
    );
    assert!(!calls.contains("UNEXPECTED EDIT"), "{calls}");
}

#[cfg(unix)]
#[test]
fn workbench_key_customization_rejects_conflicts_and_survives_restart() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model inspect') printf '%s\n' "$*" >> calls.log
   printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let preferences = directory.path().join("preferences.json");
    let run = |commands: &[u8]| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args([
                "--line-mode",
                "--lang",
                "en",
                "--cli",
                backend.to_str().unwrap(),
            ])
            .current_dir(directory.path())
            .env("MODEL_MAVEN_TUI_PREFERENCES", &preferences)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(commands).unwrap();
        let result = child.wait_with_output().unwrap();
        assert!(result.status.success());
        String::from_utf8(result.stdout).unwrap()
    };
    let output = run(b"\ns\n3\n1\n3\n1\nv\nb\nb\nv\nfixture:model\nq\n");
    assert!(
        output.contains("Already assigned 3: Check Python and PyTorch"),
        "{output}"
    );
    assert!(output.contains("v. Inspect model structure"), "{output}");
    let output = run(b"\nv\nfixture:model\nq\n");
    assert!(output.contains("v. Inspect model structure"), "{output}");
    assert_eq!(
        std::fs::read_to_string(directory.path().join("calls.log"))
            .unwrap()
            .lines()
            .count(),
        2
    );
    let settings: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&preferences).unwrap()).unwrap();
    assert_eq!(settings["workbench_keys"]["default"]["inspect"], "v");
    let restored = run(b"\ns\nr\nb\nq\n");
    assert!(
        restored.contains("1. Inspect model structure"),
        "{restored}"
    );
    let settings: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&preferences).unwrap()).unwrap();
    assert!(
        settings["workbench_keys"]["default"]
            .as_object()
            .unwrap()
            .is_empty()
    );
}

#[cfg(unix)]
#[test]
fn selected_variant_is_shared_by_inspection_probe_and_preview_and_clears_old_context() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
printf '%s\n' "$*" >> calls.log
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'variant list') printf '%s\n' '{"protocol_version":1,"operation":"variant_list","status":"success","data":{"variants":[{"name":"branch","readable":true,"revision":"revision-123456"},{"name":"broken","readable":false},{"name":"duplicate","readable":true},{"name":"duplicate","readable":true}]}}' ;;
 'variant show') printf '%s\n' '{"protocol_version":1,"operation":"variant_show","status":"success","data":{"name":"branch","variant":{"revision":"revision-123456"}}}' ;;
 'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear","editable_params":{"in_features":32}}]}}' ;;
 'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","data":{"calls":[{"alias_paths":["model.head"],"module_type":"Linear","invocation_index":1}],"failure_location":{"last_observed_module_call":{"alias_paths":["model.head"],"module_type":"Linear","invocation_index":1}}},"diagnostics":[{"code":"FORWARD_FAILED","message":"fixture"}]}' ; exit 4 ;;
 'layer set') printf '%s\n' '{"protocol_version":1,"operation":"edit","status":"success","data":{"committed":false,"variant_name":"branch","validation":{"forward":"passed"}}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(directory.path())
        .env("COLUMNS", "100")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"\n1\nfixture:model\n2\n\nfixture:batch\ns\n4\n2\n3\n1\nb\n1\nfixture:model\n2\n\nfixture:batch\n7\n48\ns\n4\n1\nb\ns\n4\n0\nb\n1\nfixture:model\nq\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(
        stdout.matches("Variant not selected").count() >= 2,
        "{stdout}"
    );
    assert!(
        stdout.contains("Listing is not validation or a pin"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Selecting clears session evidence"),
        "{stdout}"
    );
    assert!(stdout.contains("Variant  branch"), "{stdout}");
    assert!(stdout.contains("Selected variant  branch"), "{stdout}");
    let after_selection = stdout.split("4. Variant  branch").nth(1).unwrap();
    let before_inspect = after_selection.split("Completed").next().unwrap();
    assert!(
        !before_inspect.contains("9. Last diagnosis"),
        "{before_inspect}"
    );
    let calls = std::fs::read_to_string(directory.path().join("calls.log")).unwrap();
    let inspect: Vec<_> = calls
        .lines()
        .filter(|line| line.contains("model inspect"))
        .collect();
    let probe: Vec<_> = calls
        .lines()
        .filter(|line| line.contains("model probe"))
        .collect();
    let preview: Vec<_> = calls
        .lines()
        .filter(|line| line.contains("layer set"))
        .collect();
    assert_eq!(inspect.len(), 3, "{calls}");
    assert!(
        !inspect[0].contains("--variant")
            && inspect[1].contains("--variant branch")
            && !inspect[2].contains("--variant"),
        "{calls}"
    );
    assert_eq!(probe.len(), 2, "{calls}");
    assert!(
        !probe[0].contains("--variant") && probe[1].contains("--variant branch"),
        "{calls}"
    );
    assert_eq!(preview.len(), 1, "{calls}");
    assert!(
        preview[0].contains("--variant branch")
            && preview[0].contains("--dry-run")
            && preview[0].contains("--require-forward"),
        "{calls}"
    );
}

#[cfg(unix)]
#[test]
fn externally_changed_variant_clears_cached_context_before_it_can_be_opened() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
printf '%s\n' "$*" >> calls.log
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'variant list') printf '%s\n' '{"protocol_version":1,"operation":"variant_list","status":"success","data":{"variants":[{"name":"branch","readable":true,"revision":"rev-1"}]}}' ;;
 'variant show') printf '%s\n' '{"protocol_version":1,"operation":"variant_show","status":"success","data":{"name":"branch","variant":{"revision":"rev-2"}}}' ;;
 'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear"}]}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(directory.path())
        .env("COLUMNS", "100")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\ns\n4\n1\nb\n1\nfixture:model\n6\n1\nfixture:model\n6\nq\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("Variant evidence is stale"), "{stdout}");
    assert!(stdout.contains("Session evidence was cleared"), "{stdout}");
    assert_eq!(stdout.matches("Inspected modules").count(), 1, "{stdout}");
    let calls = std::fs::read_to_string(directory.path().join("calls.log")).unwrap();
    assert_eq!(
        calls
            .lines()
            .filter(|line| line.contains("variant show branch"))
            .count(),
        2,
        "{calls}"
    );
    assert_eq!(
        calls
            .lines()
            .filter(|line| line.contains("model inspect"))
            .count(),
        2,
        "{calls}"
    );
}

#[cfg(unix)]
#[test]
fn missing_variant_revisions_do_not_validate_cached_evidence() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'variant list') printf '%s\n' '{"protocol_version":1,"operation":"variant_list","status":"success","data":{"variants":[{"name":"branch","readable":true}]}}' ;;
 'variant show') printf '%s\n' '{"protocol_version":1,"operation":"variant_show","status":"success","data":{"name":"branch","variant":{}}}' ;;
 'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear"}]}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(directory.path())
        .env("COLUMNS", "100")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\ns\n4\n1\nb\n1\nfixture:model\n6\nq\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("Variant revision could not be read"),
        "{stdout}"
    );
    assert!(!stdout.contains("Inspected modules"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn line_mode_enter_opens_the_recommended_setup_without_writing() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"error","diagnostics":[{"code":"PROJECT_NOT_INITIALIZED","message":"missing"}]}' ; exit 2 ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(directory.path())
        .env("COLUMNS", "100")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"\n\nb\nq\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(stderr.contains("Connect a model"), "{stderr}");
    assert!(
        stderr.contains("Model factory (or :preset to create a model)"),
        "{stderr}"
    );
    assert!(!directory.path().join(".model-maven").exists());
}

#[cfg(unix)]
#[test]
fn keyword_linear_input_is_visible_in_the_real_workbench_report() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear","editable_params":{"in_features":32}}]}}' ;;
 'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","data":{"failure_location":{"last_observed_module_call":{"alias_paths":["model.head"],"module_type":"Linear","inputs":{"kind":"dict","items":{"args":{"kind":"tuple","items":[]},"kwargs":{"kind":"dict","items":{"input":{"kind":"tensor","shape":[2,48]}}}}}}}}}' ; exit 4 ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    for (language, expected) in [
        ("en", "Keyword input · observed"),
        ("zh-cn", "关键字 input · 实际观测"),
        ("zh-tw", "關鍵字 input · 實際觀測"),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args([
                "--line-mode",
                "--lang",
                language,
                "--cli",
                backend.to_str().unwrap(),
            ])
            .current_dir(directory.path())
            .env("COLUMNS", "100")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                directory.path().join("preferences.json"),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"\n1\nfixture:model\n2\n\nfixture:batch\nq\n")
            .unwrap();
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(output.status.success(), "{stdout}");
        assert!(stdout.contains(&format!("{expected}  2 × 48")), "{stdout}");
        assert!(stdout.contains("32"), "{stdout}");
    }
}

#[cfg(unix)]
#[test]
fn preview_surfaces_all_reported_aliases_in_three_languages() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear","editable_params":{"in_features":32}}]}}' ;;
 'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","data":{"failure_location":{"last_observed_module_call":{"alias_paths":["model.head","model.alias"],"module_type":"Linear"}}}}' ; exit 4 ;;
 'layer set') printf '%s\n' '{"protocol_version":1,"operation":"edit","status":"success","data":{"committed":false,"edit_effects":[{"path":"model.head","affected_aliases":["model.head","model.alias"],"initialized":["weight"],"preserved":["bias"]}],"validation":{"forward":"passed"}}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    for (language, label, mode_unknown) in [
        (
            "en",
            "Affected aliases · CLI report",
            "Candidate mode unverified",
        ),
        ("zh-cn", "受影响别名 · CLI 报告", "候选模式无法核对"),
        ("zh-tw", "受影響別名 · CLI 報告", "候選模式無法核對"),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args([
                "--line-mode",
                "--lang",
                language,
                "--cli",
                backend.to_str().unwrap(),
            ])
            .current_dir(directory.path())
            .env("COLUMNS", "100")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                directory.path().join("preferences.json"),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"\n1\nfixture:model\n2\n\nfixture:batch\n7\n48\nq\n")
            .unwrap();
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(output.status.success(), "{stdout}");
        assert!(stdout.contains(mode_unknown), "{stdout}");
        let section = stdout.split_once(label).expect("alias summary").1;
        let lines: Vec<_> = section.lines().take(4).collect();
        assert!(
            lines.iter().any(|line| line.contains("model.head")),
            "{stdout}"
        );
        assert!(
            lines.iter().any(|line| line.contains("model.alias")),
            "{stdout}"
        );
    }
}

#[cfg(unix)]
#[test]
fn preview_uses_unique_inspected_alias_and_blocks_ambiguous_snapshots() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/usr/bin/python3
import json, os, sys
args=sys.argv[3:]
with open('calls.log','a') as log: log.write(' '.join(args)+'\n')
def out(operation,status,data): print(json.dumps({'protocol_version':1,'operation':operation,'status':status,'data':data}))
if args[:2]==['config','history']: out('config_history','success',{})
elif args[:2]==['model','inspect']:
 modules=[{'path':'model.head','type':'Linear','editable_params':{'in_features':32}}]
 if os.environ.get('TEST_AMBIGUOUS')=='1': modules.insert(0,{'path':'ghost','type':'Linear'})
 out('inspect','success',{'modules':modules})
elif args[:2]==['model','probe']:
 out('probe','error',{'failure_location':{'last_observed_module_call':{'alias_paths':['ghost','model.head'],'module_type':'Linear'}}}); sys.exit(4)
elif args[:2]==['layer','set']: out('edit','success',{'committed':False,'validation':{'forward':'passed'}})
else: sys.exit(1)
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let run = |ambiguous: bool, language: &str, input: &[u8]| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args([
                "--line-mode",
                "--lang",
                language,
                "--cli",
                backend.to_str().unwrap(),
            ])
            .current_dir(directory.path())
            .env("COLUMNS", "100")
            .env("TEST_AMBIGUOUS", if ambiguous { "1" } else { "0" })
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                directory.path().join("preferences.json"),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(input).unwrap();
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(output.status.success(), "{stdout}");
        stdout
    };
    let valid = run(
        false,
        "en",
        b"\n1\nfixture:model\n2\n\nfixture:batch\n7\n48\nq\n",
    );
    assert!(
        valid.contains("Back, then choose Preview Linear input size"),
        "{valid}"
    );
    let calls = std::fs::read_to_string(directory.path().join("calls.log")).unwrap();
    assert!(
        calls
            .lines()
            .any(|line| line.starts_with("layer set model.head ")),
        "{calls}"
    );
    std::fs::write(directory.path().join("calls.log"), "").unwrap();
    for (language, hint) in [
        ("en", "not uniquely matched"),
        ("zh-cn", "未唯一匹配"),
        ("zh-tw", "未唯一符合"),
    ] {
        let output = run(
            true,
            language,
            b"\n1\nfixture:model\n2\n\nfixture:batch\n7\nq\n",
        );
        assert!(output.contains(hint), "{output}");
    }
    let calls = std::fs::read_to_string(directory.path().join("calls.log")).unwrap();
    assert!(
        !calls.lines().any(|line| line.starts_with("layer set")),
        "{calls}"
    );
}

#[cfg(unix)]
#[test]
fn batch_factory_parameters_are_validated_and_carried_into_preview_without_leaking() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/usr/bin/python3
import json, sys
args=sys.argv[3:]
with open('calls.jsonl','a') as log: log.write(json.dumps(args)+'\n')
def out(operation,status,data): print(json.dumps({'protocol_version':1,'operation':operation,'status':status,'data':data}))
if args[:2]==['config','history']: out('config_history','success',{})
elif args[:2]==['model','inspect']: out('inspect','success',{'modules':[{'path':'model.head','type':'Linear'}]})
elif args[:2]==['model','probe']:
 out('probe','error',{'failure_location':{'last_observed_module_call':{'alias_paths':['model.head'],'module_type':'Linear'}}}); sys.exit(4)
elif args[:2]==['layer','set']: out('edit','success',{'committed':False,'validation':{'forward':'passed'}})
else: sys.exit(1)
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(directory.path())
        .env("COLUMNS", "100")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"\n1\nfixture:model\n2\n\n:batch-params\n[]\n{\"size\":1,\"size\":2}\n{\"size\":48,\"config\":{\"name\":\"a b\"}}\n:options\nsingle\ndataset-v1\nfixture:batch\n7\n48\n2\n\nfixture:other\nq\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}");
    assert_eq!(
        stdout.matches("Invalid batch parameters").count(),
        2,
        "{stdout}"
    );
    let calls: Vec<Vec<String>> = std::fs::read_to_string(directory.path().join("calls.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let probe: Vec<_> = calls
        .iter()
        .filter(|args| args.starts_with(&["model".into(), "probe".into()]))
        .collect();
    assert_eq!(probe.len(), 2, "{calls:?}");
    let preview = calls
        .iter()
        .find(|args| args.starts_with(&["layer".into(), "set".into()]))
        .unwrap();
    for args in [probe[0], preview] {
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--batch-param", "size=48"]),
            "{args:?}"
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--batch-param", r#"config={"name":"a b"}"#]),
            "{args:?}"
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--batch-binding", "single"]),
            "{args:?}"
        );
    }
    assert!(probe[0].contains(&"--batch-fingerprint".into()));
    assert!(!preview.contains(&"--batch-fingerprint".into()));
    assert!(
        !probe[1].contains(&"--batch-param".into()),
        "{:?}",
        probe[1]
    );
    assert!(
        !probe[1].contains(&"--batch-fingerprint".into()),
        "{:?}",
        probe[1]
    );
}

#[cfg(unix)]
#[test]
fn model_factory_parameters_follow_inspect_probe_preview_and_invalidate_old_context() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/usr/bin/python3
import json, sys
args=sys.argv[3:]
with open('calls.jsonl','a') as log: log.write(json.dumps(args)+'\n')
def out(operation,status,data): print(json.dumps({'protocol_version':1,'operation':operation,'status':status,'data':data}))
if args[:2]==['config','history']: out('config_history','success',{})
elif args[:2]==['model','inspect']: out('inspect','success',{'modules':[{'path':'model.head','type':'Linear','editable_params':{'in_features':32}}]})
elif args[:2]==['model','probe']:
 out('probe','error',{'failure_location':{'last_observed_module_call':{'alias_paths':['model.head'],'module_type':'Linear'}}}); sys.exit(4)
elif args[:2]==['layer','set']: out('edit','success',{'committed':False,'validation':{'forward':'passed'}})
else: sys.exit(1)
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(directory.path())
        .env("COLUMNS", "100")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"\n1\n:model-params\n{\"width\":32}\nfixture:model\n2\n\n:batch-params\n{\"size\":48}\nfixture:batch\n7\n48\n2\n:model-params\n{\"width\":64}\n\nfixture:batch\n6\nq\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.contains("No inspected structure available"),
        "{stdout}"
    );
    let calls: Vec<Vec<String>> = std::fs::read_to_string(directory.path().join("calls.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let inspect = calls
        .iter()
        .find(|args| args.starts_with(&["model".into(), "inspect".into()]))
        .unwrap();
    let probes: Vec<_> = calls
        .iter()
        .filter(|args| args.starts_with(&["model".into(), "probe".into()]))
        .collect();
    let preview = calls
        .iter()
        .find(|args| args.starts_with(&["layer".into(), "set".into()]))
        .unwrap();
    assert_eq!(probes.len(), 2, "{calls:?}");
    for args in [inspect, probes[0], preview] {
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--model-param", "width=32"]),
            "{args:?}"
        );
    }
    assert!(
        probes[0]
            .windows(2)
            .any(|pair| pair == ["--batch-param", "size=48"])
    );
    assert!(
        preview
            .windows(2)
            .any(|pair| pair == ["--batch-param", "size=48"])
    );
    assert!(
        probes[1]
            .windows(2)
            .any(|pair| pair == ["--model-param", "width=64"]),
        "{:?}",
        probes[1]
    );
    assert!(
        !probes[1].contains(&"--batch-param".into()),
        "{:?}",
        probes[1]
    );
    std::fs::write(directory.path().join("calls.jsonl"), "").unwrap();
    for (replacement, should_keep) in [("", true), ("fixture:other", false)] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args([
                "--line-mode",
                "--lang",
                "en",
                "--cli",
                backend.to_str().unwrap(),
            ])
            .current_dir(directory.path())
            .env("COLUMNS", "100")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                directory.path().join("preferences.json"),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let commands = format!(
            "\n2\n:model-params\n{{\"width\":32}}\nfixture:model\nb\n{replacement}\nfixture:batch\nq\n"
        );
        child
            .stdin
            .take()
            .unwrap()
            .write_all(commands.as_bytes())
            .unwrap();
        assert!(child.wait_with_output().unwrap().status.success());
        let calls = std::fs::read_to_string(directory.path().join("calls.jsonl")).unwrap();
        let probe: Vec<String> = calls
            .lines()
            .map(|line| serde_json::from_str::<Vec<String>>(line).unwrap())
            .filter(|args| args.starts_with(&["model".into(), "probe".into()]))
            .last()
            .unwrap();
        assert_eq!(
            probe
                .windows(2)
                .any(|pair| pair == ["--model-param", "width=32"]),
            should_keep,
            "{probe:?}"
        );
    }
}

#[cfg(unix)]
#[test]
fn forward_mode_is_retained_for_preview_and_isolates_diagnosis_comparisons() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/usr/bin/python3
import json, sys
args=sys.argv[3:]
with open('calls.jsonl','a') as log: log.write(json.dumps(args)+'\n')
def out(operation,status,data): print(json.dumps({'protocol_version':1,'operation':operation,'status':status,'data':data}))
mode='train' if '--training' in args else 'eval'
if args[:2]==['config','history']: out('config_history','success',{})
elif args[:2]==['model','inspect']: out('inspect','success',{'modules':[{'path':'model.head','type':'Linear'}]})
elif args[:2]==['model','probe']:
 out('probe','error',{'observed_mode':mode,'batch_id':'batch-a','batch_identity_stable':True,'failure_location':{'last_observed_module_call':{'alias_paths':['model.head'],'module_type':'Linear'}}}); sys.exit(4)
elif args[:2]==['layer','set']: out('edit','success',{'committed':False,'validation':{'forward':'passed','mode':mode}})
else: sys.exit(1)
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args([
            "--line-mode",
            "--lang",
            "en",
            "--cli",
            backend.to_str().unwrap(),
        ])
        .current_dir(directory.path())
        .env("COLUMNS", "100")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            directory.path().join("preferences.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"\n1\nfixture:model\n2\n\n:mode\nbad\ntrain\n:batch-params\n{\"size\":48}\nfixture:batch\n7\n48\n2\n\n:mode\neval\n\n7\n48\n2\n\n\nq\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("Invalid forward mode"), "{stdout}");
    assert!(
        stdout.contains("Observed mode  train") && stdout.contains("Observed mode  eval"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Candidate mode · CLI report  train")
            && stdout.contains("Candidate mode · CLI report  eval"),
        "{stdout}"
    );
    assert_eq!(
        stdout
            .matches("Batch ID matches previous probe report")
            .count(),
        1,
        "{stdout}"
    );
    let calls: Vec<Vec<String>> = std::fs::read_to_string(directory.path().join("calls.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let probes: Vec<_> = calls
        .iter()
        .filter(|args| args.starts_with(&["model".into(), "probe".into()]))
        .collect();
    let previews: Vec<_> = calls
        .iter()
        .filter(|args| args.starts_with(&["layer".into(), "set".into()]))
        .collect();
    assert_eq!(probes.len(), 3, "{calls:?}");
    assert_eq!(previews.len(), 2, "{calls:?}");
    assert!(probes[0].contains(&"--training".into()) && previews[0].contains(&"--training".into()));
    assert!(
        !probes[1].contains(&"--training".into())
            && !previews[1].contains(&"--training".into())
            && !probes[2].contains(&"--training".into())
    );
    assert!(
        previews[0]
            .windows(2)
            .any(|pair| pair == ["--batch-param", "size=48"])
    );
}

#[cfg(unix)]
#[test]
fn mode_mismatch_blocks_old_diagnosis_and_flags_candidate_validation() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let backend = directory.path().join("m-model");
    std::fs::write(&backend, r##"#!/usr/bin/python3
import json, sys
args=sys.argv[3:]
with open('calls.jsonl','a') as log: log.write(json.dumps(args)+'\n')
def out(operation,status,data): print(json.dumps({'protocol_version':1,'operation':operation,'status':status,'data':data}))
if args[:2]==['config','history']: out('config_history','success',{})
elif args[:2]==['model','inspect']: out('inspect','success',{'modules':[{'path':'model.head','type':'Linear'}]})
elif args[:2]==['model','probe']:
 out('probe','error',{'observed_mode':'eval','failure_location':{'last_observed_module_call':{'alias_paths':['model.head'],'module_type':'Linear'}}}); sys.exit(4)
elif args[:2]==['layer','set']: out('edit','success',{'committed':False,'validation':{'forward':'passed','mode':'train'}})
else: sys.exit(1)
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    for (language, probe_warning, preview_warning) in [
        (
            "en",
            "Mode mismatch · preview unavailable",
            "Candidate mode differs from request",
        ),
        ("zh-cn", "模式不一致 · 无法预览", "候选模式与请求不一致"),
        ("zh-tw", "模式不一致 · 無法預覽", "候選模式與請求不一致"),
    ] {
        std::fs::write(directory.path().join("calls.jsonl"), "").unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args([
                "--line-mode",
                "--lang",
                language,
                "--cli",
                backend.to_str().unwrap(),
            ])
            .current_dir(directory.path())
            .env("COLUMNS", "100")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                directory.path().join("preferences.json"),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(b"\n1\nfixture:model\n2\n\n:mode\ntrain\nfixture:batch\n7\n2\n\n:mode\neval\n\n7\n48\nq\n").unwrap();
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(output.status.success(), "{stdout}");
        assert!(stdout.contains(probe_warning), "{stdout}");
        assert!(stdout.contains(preview_warning), "{stdout}");
        let calls: Vec<Vec<String>> = std::fs::read_to_string(directory.path().join("calls.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let previews: Vec<_> = calls
            .iter()
            .filter(|args| args.starts_with(&["layer".into(), "set".into()]))
            .collect();
        assert_eq!(previews.len(), 1, "{calls:?}");
        assert!(!previews[0].contains(&"--training".into()));
    }
}
