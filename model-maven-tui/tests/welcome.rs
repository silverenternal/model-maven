#![cfg(unix)]

use std::{
    fs::File,
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::PermissionsExt,
    },
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[test]
fn monochrome_welcome_keeps_animating_until_enter() {
    let project = tempfile::tempdir().unwrap();
    let backend = project.path().join("m-model");
    std::fs::write(&backend, r#"#!/bin/sh
test "$1" = --json && test "$2" = --no-interactive || exit 1
case "$3 $4" in
    "config history")
        printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"error","diagnostics":[{"code":"PROJECT_NOT_INITIALIZED","message":"fixture"}]}'
        exit 2 ;;
    "model inspect")
        test "$5" = --model && test "$6" = demo:build || exit 1
        printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"net.head","type":"Linear","shape":[64,32]},{"path":"model.head","type":"Linear","editable_params":{"in_features":32}}]}}'
        exit 0 ;;
    "model probe")
        test "$5" = --model && test "$6" = demo:build && test "$7" = --batch && test "$8" = demo:batch || exit 1
        printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","data":{"failure_location":{"last_observed_module_call":{"alias_paths":["model.head"],"invocation_index":2,"module_type":"Linear","inputs":{"kind":"dict","items":{"args":{"kind":"tuple","items":[{"kind":"tensor","shape":[2,48]}]},"kwargs":{"kind":"dict","items":{"mask":{"kind":"tensor","shape":[2,7],"dtype":"torch.bool"}}}}}},"project_stack":[{"file":"demo.py","line":12}]},"calls":[{"alias_paths":["net.head"],"invocation_index":1,"module_type":"Linear","inputs":{"kind":"dict","items":{"args":{"kind":"tuple","items":[{"kind":"tensor","shape":[2,32]}]}}}},{"alias_paths":["net.head"],"invocation_index":2,"module_type":"Linear","inputs":{"kind":"dict","items":{"args":{"kind":"tuple","items":[{"kind":"tensor","shape":[2,64]}]},"kwargs":{"kind":"dict","items":{"mask":{"kind":"tensor","dtype":"torch.bool","shape":[2,7]}}}}}},{"alias_paths":["model.head"],"invocation_index":2,"module_type":"Linear","inputs":{"shape":[2,48]}}]},"diagnostics":[{"code":"FORWARD_FAILED","message":"fixture shape mismatch: actual 48, expected 32"}]}'
        exit 4 ;;
    "layer set")
        test "$5" = model.head && test "$6" = --param && test "$7" = in_features=48 && test "$8" = --dry-run && test "$9" = --model || exit 1
        shift 9
        test "$1" = demo:build && test "$2" = --batch && test "$3" = demo:batch && test "$4" = --require-forward || exit 1
        printf '%s\n' '{"protocol_version":1,"operation":"edit","status":"success","data":{"committed":false,"variant_name":"working","validation":{"forward":"passed"},"modules":[{"path":"model.head","type":"Linear","editable_params":{"in_features":48,"out_features":8}}],"edit_effects":[{"path":"model.head","initialized":["weight"],"preserved":["bias"]}]}}'
        exit 0 ;;
    *) exit 1 ;;
esac
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 80,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: openpty receives two valid output pointers and a valid size.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    // SAFETY: openpty transferred both descriptors to this process.
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut original_mode = std::mem::MaybeUninit::<libc::termios>::uninit();
    // SAFETY: original_mode points to writable storage for the terminal settings.
    assert_eq!(
        unsafe { libc::tcgetattr(slave.as_raw_fd(), original_mode.as_mut_ptr()) },
        0
    );
    let original_mode = unsafe { original_mode.assume_init() };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .current_dir(project.path())
        .arg("--cli")
        .arg(&backend)
        .env("TERM", "xterm-256color")
        .env("NO_COLOR", "1")
        .env("LC_ALL", "zh_CN.UTF-8")
        .env("HOME", project.path())
        .env("XDG_CONFIG_HOME", project.path().join("config"))
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();

    let mut welcome = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !String::from_utf8_lossy(&welcome).contains("[Enter]  进入工作台") {
        assert!(Instant::now() < deadline, "welcome screen did not appear");
        assert!(read_frame(&mut terminal, &mut welcome));
    }
    let first_frame = String::from_utf8_lossy(&welcome);
    assert!(first_frame.contains("PyTorch 模型调试与实验工作台"));
    assert!(first_frame.contains(" · 接入模型"));
    terminal.write_all(b"x").unwrap();

    // Pass the first 120° speed cycle as well as the previous one-second intro.
    let end_of_cycle = Instant::now() + Duration::from_millis(2350);
    while Instant::now() < end_of_cycle {
        assert!(read_frame(&mut terminal, &mut welcome));
    }
    let mut following_frames = Vec::new();
    let deadline = Instant::now() + Duration::from_millis(180);
    while Instant::now() < deadline {
        assert!(read_frame(&mut terminal, &mut following_frames));
    }
    welcome.extend_from_slice(&following_frames);
    let welcome = String::from_utf8_lossy(&welcome);
    assert!(
        String::from_utf8_lossy(&following_frames)
            .chars()
            .any(|ch| ('⠁'..='⣿').contains(&ch))
    );
    assert!(!welcome.contains("选择操作:") && !welcome.contains("\x1b[?1049l"));
    assert!(!welcome.contains("\x1b[0m") && !welcome.contains("\x1b[96m"));

    let mut after_enter = Vec::new();
    terminal
        .write_all(b"\n2\x1b[200~demo:bu\x04Xild\x1b[201~\x1b[D\x1b[D\x1b[D\x7f\n")
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !String::from_utf8_lossy(&after_enter).contains("批次工厂") {
        assert!(Instant::now() < deadline, "batch form did not appear");
        assert!(read_frame(&mut terminal, &mut after_enter));
    }
    terminal.write_all(b"\x1b").unwrap();
    let mut back_to_model = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !String::from_utf8_lossy(&back_to_model).contains("demo:build▏") {
        assert!(
            Instant::now() < deadline,
            "model form did not restore the draft"
        );
        assert!(read_frame(&mut terminal, &mut back_to_model));
    }
    after_enter.extend_from_slice(&back_to_model);
    terminal
        .write_all(b"\ndemo:batch\nbs11\x1b[B\nbp\n\nblbv1bbq")
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while read_frame(&mut terminal, &mut after_enter) {
        if Instant::now() >= deadline {
            let _ = child.kill();
            break;
        }
    }
    assert!(child.wait().unwrap().success());
    let mut restored_mode = std::mem::MaybeUninit::<libc::termios>::uninit();
    // SAFETY: restored_mode points to writable storage for the terminal settings.
    assert_eq!(
        unsafe { libc::tcgetattr(terminal.as_raw_fd(), restored_mode.as_mut_ptr()) },
        0
    );
    let restored_mode = unsafe { restored_mode.assume_init() };
    // Darwin may set PENDIN when queued raw input is retyped in canonical mode.
    #[cfg(target_os = "macos")]
    let mode_mask = !libc::PENDIN;
    #[cfg(not(target_os = "macos"))]
    let mode_mask = !0;
    assert_eq!(
        restored_mode.c_lflag & mode_mask,
        original_mode.c_lflag & mode_mask
    );
    let after_enter = String::from_utf8_lossy(&after_enter);
    let frame = after_enter
        .split_once("\x1b[?2026h\x1b[?1049l\x1b[?25h")
        .and_then(|(_, rest)| rest.split_once("\x1b[?2026l"))
        .map(|(frame, _)| frame)
        .expect("expected a synchronized workbench frame");
    assert!(frame.contains("Model-Maven 工作台"));
    assert!(frame.contains("项目尚未配置"));
    assert!(frame.contains("> n. 设置项目入口"));
    assert!(frame.contains("2. 前向诊断"));
    assert!(after_enter.contains("↑/↓ 移动"), "{after_enter}");
    assert!(after_enter.contains("> 2. 前向诊断"), "{after_enter}");
    assert!(after_enter.contains("视觉风格  紧凑"), "{after_enter}");
    assert!(after_enter.contains("视觉风格  叙事"), "{after_enter}");
    assert!(after_enter.contains("╭─ 导航"), "{after_enter}");
    assert!(after_enter.contains("操作预设  快捷键"), "{after_enter}");
    assert!(after_enter.contains("> 2. 操作预设"), "{after_enter}");
    assert!(after_enter.contains("p. 前向诊断"), "{after_enter}");
    assert!(after_enter.contains("FORWARD_FAILED"), "{after_enter}");
    assert!(
        after_enter.matches("FORWARD_FAILED").count() >= 3,
        "{after_enter}"
    );
    assert!(after_enter.contains("模型  demo:build"), "{after_enter}");
    assert!(after_enter.contains("批次  demo:batch"), "{after_enter}");
    assert!(after_enter.contains("报告详情"), "{after_enter}");
    assert!(after_enter.contains("\"shape\""), "{after_enter}");
    assert!(
        after_enter.contains("actual 48, expected 32"),
        "{after_enter}"
    );
    assert!(after_enter.contains("model.head  #2"), "{after_enter}");
    assert!(after_enter.contains("demo.py:12"), "{after_enter}");

    let (mut master, mut slave) = (-1, -1);
    // SAFETY: both descriptors and the window size are valid for openpty.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    // SAFETY: openpty transferred ownership of both descriptors.
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .current_dir(project.path())
        .arg("--cli")
        .arg(&backend)
        .env("TERM", "xterm-256color")
        .env("PATH", project.path())
        .env("NO_COLOR", "1")
        .env("LC_ALL", "zh_CN.UTF-8")
        .env("HOME", project.path())
        .env("XDG_CONFIG_HOME", project.path().join("config"))
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();
    let mut restarted = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !String::from_utf8_lossy(&restarted).contains("[Enter]  进入工作台") {
        assert!(
            Instant::now() < deadline,
            "restarted welcome screen did not appear"
        );
        assert!(read_frame(&mut terminal, &mut restarted));
    }
    terminal.write_all(b"\ng").unwrap();
    let graph_output = editor_frame(&mut terminal, "源码 · Drifting 静态图");
    let braille = |ch| ('\u{2801}'..='\u{28ff}').contains(&ch);
    assert!(graph_output.chars().any(braille));
    assert!(graph_output.contains("DitGen.backbone"));
    assert!(graph_output.contains("models/generator.py:553"));
    let mut after_restart = graph_output.into_bytes();
    let motion = wait_for_terminal_text(&mut terminal, "\x1b[2K");
    assert!(!motion.windows(4).any(|bytes| bytes == b"\x1b[2J"));
    terminal.write_all(b" ").unwrap();
    let drain_until = Instant::now() + Duration::from_millis(150);
    while Instant::now() < drain_until {
        assert!(read_frame(&mut terminal, &mut Vec::new()));
    }
    let mut paused = Vec::new();
    let paused_until = Instant::now() + Duration::from_millis(250);
    while Instant::now() < paused_until {
        assert!(read_frame(&mut terminal, &mut paused));
    }
    assert!(paused.is_empty());
    terminal.write_all(b" ").unwrap();
    wait_for_terminal_text(&mut terminal, "\x1b[2K");
    terminal.write_all(b"b").unwrap();
    after_restart.extend(wait_for_terminal_text(
        &mut terminal,
        "✦  Model-Maven 工作台",
    ));

    terminal.write_all(b"idemo:build\nbg").unwrap();
    let mut inspected = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !String::from_utf8_lossy(&inspected).contains("shape: [64,32]") {
        assert!(
            Instant::now() < deadline,
            "inspected structure did not appear"
        );
        assert!(read_frame(&mut terminal, &mut inspected));
    }
    assert!(String::from_utf8_lossy(&inspected).contains("检查报告 · 模块列表"));
    assert!(String::from_utf8_lossy(&inspected).contains("net.head"));
    after_restart.extend_from_slice(&inspected);
    terminal
        .write_all(b"bp\x15demo:build\ndemo:batch\n")
        .unwrap();
    let probe = wait_for_terminal_text(&mut terminal, "FORWARD_FAILED");
    let probe_output = String::from_utf8_lossy(&probe);
    assert!(probe_output.contains("CLI 正在运行"), "{probe_output}");
    assert!(probe_output.contains("demo:build"), "{probe_output}");
    assert!(probe_output.contains("demo:batch"), "{probe_output}");
    after_restart.extend(probe);
    terminal.write_all(b"b").unwrap();
    let menu_after_probe = wait_for_terminal_text(&mut terminal, "> p. 前向诊断");
    let menu_text = String::from_utf8_lossy(&menu_after_probe);
    assert!(menu_text.contains("模型  demo:build"), "{menu_text}");
    assert!(menu_text.contains("批次  demo:batch"), "{menu_text}");
    after_restart.extend(menu_after_probe);
    size.ws_col = 40;
    size.ws_row = 16;
    // SAFETY: terminal and size refer to this test's pseudoterminal.
    assert_eq!(
        unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ, &size) },
        0
    );
    terminal.write_all(b"\x1b[A\x1b[A").unwrap();
    let bottom = wait_for_terminal_text(&mut terminal, "> q.");
    let bottom = String::from_utf8_lossy(&bottom);
    assert!(bottom.contains("模型  demo:build"), "{bottom}");
    assert!(bottom.contains("↑ …"), "{bottom}");
    size.ws_col = 80;
    size.ws_row = 24;
    // SAFETY: terminal and size refer to this test's pseudoterminal.
    assert_eq!(
        unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ, &size) },
        0
    );
    terminal.write_all(b"k").unwrap();
    let calls = wait_for_terminal_text(&mut terminal, "> 1. net.head");
    let calls = String::from_utf8_lossy(&calls);
    assert!(calls.contains("检查快照身份尚未与调用核对"), "{calls}");
    terminal.write_all(b"/").unwrap();
    wait_for_terminal_text(&mut terminal, "筛选已记录调用");
    terminal.write_all(b"2x64\n").unwrap();
    let filtered = wait_for_terminal_text(&mut terminal, "> 2. net.head");
    let filtered = String::from_utf8_lossy(&filtered);
    assert!(filtered.contains("1/4"), "{filtered}");
    terminal.write_all(b"/").unwrap();
    wait_for_terminal_text(&mut terminal, "筛选已记录调用");
    terminal.write_all(b"\x15not-found\n").unwrap();
    wait_for_terminal_text(&mut terminal, "没有匹配的结果");
    terminal.write_all(b"/").unwrap();
    wait_for_terminal_text(&mut terminal, "筛选已记录调用");
    terminal.write_all(b"\x15FORWARD_FAILED\n").unwrap();
    wait_for_terminal_text(&mut terminal, "> d1. [probe] FORWARD_FAILED");
    terminal.write_all(b"\n").unwrap();
    let diagnostic = wait_for_terminal_text(&mut terminal, "报告级诊断");
    assert!(String::from_utf8_lossy(&diagnostic).contains("未关联到具体调用"));
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "> d1. [probe] FORWARD_FAILED");
    terminal.write_all(b"/").unwrap();
    wait_for_terminal_text(&mut terminal, "筛选已记录调用");
    terminal.write_all(b"\x152x64\n").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. net.head");
    terminal.write_all(b"/").unwrap();
    wait_for_terminal_text(&mut terminal, "筛选已记录调用");
    terminal.write_all(b"\x15cancelled\x1b").unwrap();
    let cancelled = wait_for_terminal_text(&mut terminal, "> 2. net.head");
    assert!(String::from_utf8_lossy(&cancelled).contains("2x64"));
    terminal.write_all(b"\n").unwrap();
    let direct = wait_for_terminal_text(&mut terminal, "2×64");
    let direct = String::from_utf8_lossy(&direct);
    assert!(direct.contains("inputs.args[0]"), "{direct}");
    assert!(direct.contains("未记录来源"), "{direct}");
    terminal.write_all(b"l").unwrap();
    let returned = wait_for_terminal_text(&mut terminal, "> 2. net.head");
    assert!(String::from_utf8_lossy(&returned).contains("2x64"));
    terminal.write_all(b"/").unwrap();
    wait_for_terminal_text(&mut terminal, "筛选已记录调用");
    terminal.write_all(b"\x15inputs.kwargs.mask\n").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. net.head");
    terminal.write_all(b"\n").unwrap();
    let mask = wait_for_terminal_text(&mut terminal, "inputs.kwargs.mask");
    let mask = String::from_utf8_lossy(&mask);
    assert!(mask.contains("输入 2/2"), "{mask}");
    assert!(mask.contains("2×7"), "{mask}");
    terminal.write_all(b"l").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. net.head");
    terminal.write_all(b"/").unwrap();
    wait_for_terminal_text(&mut terminal, "筛选已记录调用");
    terminal.write_all(b"\x15tensor\n").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. net.head");
    size.ws_col = 40;
    // SAFETY: terminal and size refer to this test's pseudoterminal.
    assert_eq!(
        unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ, &size) },
        0
    );
    terminal.write_all(b"\n").unwrap();
    let choices = wait_for_terminal_text(&mut terminal, "kwargs.mask");
    let choices = String::from_utf8_lossy(&choices);
    assert!(choices.contains("选择匹配的输入"), "{choices}");
    assert!(choices.contains("inputs.args[0]"), "{choices}");
    assert!(
        choices.contains("2×64") && choices.contains("2×7"),
        "{choices}"
    );
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. net.head");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "kwargs.mask");
    terminal.write_all(b"\x1b[B\n").unwrap();
    let chosen = wait_for_terminal_text(&mut terminal, "未记录来源");
    let chosen = String::from_utf8_lossy(&chosen);
    assert!(chosen.contains("输入 2/2"), "{chosen}");
    assert!(chosen.contains("kwargs.mask"), "{chosen}");
    terminal.write_all(b"l").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. net.head");
    size.ws_col = 80;
    // SAFETY: terminal and size refer to this test's pseudoterminal.
    assert_eq!(
        unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ, &size) },
        0
    );
    terminal.write_all(b"/").unwrap();
    wait_for_terminal_text(&mut terminal, "筛选已记录调用");
    terminal.write_all(b"\x15\n").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. net.head");
    terminal.write_all(b"\x1b[A").unwrap();
    wait_for_terminal_text(&mut terminal, "> 1. net.head");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "net.head");
    terminal.write_all(b"F").unwrap();
    let call = wait_for_terminal_text(&mut terminal, "调用路径 · 调用证据");
    let call = String::from_utf8_lossy(&call);
    assert!(call.contains("观测调用 · net.head #1"), "{call}");
    assert!(call.contains("未记录来源"), "{call}");
    terminal.write_all(b"Fx").unwrap();
    wait_for_terminal_text(&mut terminal, "展开切片");
    terminal.write_all(b"+").unwrap();
    wait_for_terminal_text(&mut terminal, "110%");
    terminal.write_all(b"l").unwrap();
    wait_for_terminal_text(&mut terminal, "> 1. net.head");
    terminal.write_all(b"\x1b[B\n").unwrap();
    let second_call = wait_for_terminal_text(&mut terminal, "2×7");
    let second_call = String::from_utf8_lossy(&second_call);
    assert!(second_call.contains("inputs.kwargs.mask"), "{second_call}");
    terminal.write_all(b"F").unwrap();
    let second_graph = wait_for_terminal_text(&mut terminal, "观测调用 #2");
    let second_graph = String::from_utf8_lossy(&second_graph);
    assert!(!second_graph.contains("展开切片"), "{second_graph}");
    assert!(second_graph.contains("2×64"), "{second_graph}");
    terminal.write_all(b"l").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. net.head");
    terminal.write_all(b"\x1b[A\n").unwrap();
    let restored = wait_for_terminal_text(&mut terminal, "2×32");
    let restored = String::from_utf8_lossy(&restored);
    assert!(restored.contains("110%"), "{restored}");
    assert!(restored.contains("展开切片"), "{restored}");
    assert!(restored.contains("net.head"), "{restored}");
    assert!(restored.contains("2×32"), "{restored}");
    terminal.write_all(b"lb").unwrap();
    wait_for_terminal_text(&mut terminal, "> k. 定位已记录调用");
    terminal.write_all(b"k").unwrap();
    wait_for_terminal_text(&mut terminal, "> 1. net.head");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "> k. 定位已记录调用");
    terminal.write_all(b"f").unwrap();
    wait_for_terminal_text(&mut terminal, "model.head");
    terminal.write_all(b"/kwargs.mask").unwrap();
    wait_for_terminal_text(&mut terminal, "inputs.kwargs.mask");
    terminal.write_all(b"\n").unwrap();
    let found_input = wait_for_terminal_text(&mut terminal, "2×7");
    assert!(String::from_utf8_lossy(&found_input).contains("inputs.kwargs.mask"));
    terminal.write_all(b"F").unwrap();
    wait_for_terminal_text(&mut terminal, "model.head");
    terminal.write_all(b"/FORWARD_FAILED").unwrap();
    wait_for_terminal_text(&mut terminal, "FORWARD_FAILED");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "CLI 报告原文");
    terminal.write_all(b"\x1bF").unwrap();
    wait_for_terminal_text(&mut terminal, "model.head");
    terminal.write_all(b"F").unwrap();
    let path = wait_for_terminal_text(&mut terminal, "inputs.kwargs.mask");
    let path = String::from_utf8_lossy(&path);
    assert!(path.contains("最后观测"), "{path}");
    assert!(path.contains("inputs.kwargs.mask"), "{path}");
    terminal.write_all(b"\x1b[A").unwrap();
    let positional = wait_for_terminal_text(&mut terminal, "2×48");
    assert!(String::from_utf8_lossy(&positional).contains("inputs.args[0]"));
    terminal.write_all(b"\x1b[B").unwrap();
    let keyword = wait_for_terminal_text(&mut terminal, "预期 ?");
    let keyword = String::from_utf8_lossy(&keyword);
    assert!(keyword.contains("inputs.kwargs.mask"), "{keyword}");
    assert!(keyword.contains("2×7"), "{keyword}");
    assert!(keyword.contains("预期 ?"), "{keyword}");
    terminal.write_all(b"v").unwrap();
    let raw = wait_for_terminal_text(&mut terminal, "CLI 报告原文");
    assert!(String::from_utf8_lossy(&raw).contains("protocol_version"));
    terminal.write_all(b"\x1bFq").unwrap();
    wait_for_terminal_text(&mut terminal, "> f. 定位最后观测调用");
    terminal.write_all(b"e").unwrap();
    let preview_form = wait_for_terminal_text(&mut terminal, "model.head");
    after_restart.extend(preview_form);
    terminal.write_all(b"48\n").unwrap();
    let preview = wait_for_terminal_text(&mut terminal, "committed  false");
    assert!(String::from_utf8_lossy(&preview).contains("model.head.weight"));
    after_restart.extend(preview);
    terminal.write_all(b"d").unwrap();
    let difference = wait_for_terminal_text(&mut terminal, "net.head · Linear");
    let difference = String::from_utf8_lossy(&difference);
    assert!(difference.contains("in_features  32"), "{difference}");
    assert!(difference.contains("in_features  48"), "{difference}");
    assert!(difference.contains("未对照运行时连线"), "{difference}");
    assert!(difference.contains("报告的模块树"), "{difference}");
    assert!(difference.contains("model.head · Linear"), "{difference}");
    resize_terminal(&terminal, 10, 40);
    fn diff_page(terminal: &mut File) -> String {
        let mut output = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            assert!(
                Instant::now() < deadline,
                "missing complete difference page"
            );
            assert!(read_frame(terminal, &mut output));
            let text = String::from_utf8_lossy(&output);
            if text
                .rsplit_once("修复差异")
                .is_some_and(|(_, tail)| tail.contains("n/p 滚动"))
            {
                return text.into_owned();
            }
        }
    }
    let mut tree = String::new();
    for _ in 0..8 {
        terminal.write_all(b"n").unwrap();
        tree.push_str(&diff_page(&mut terminal));
        if tree.contains("model.head · Linear") && tree.contains("net.head · Linear") {
            break;
        }
    }
    assert!(
        tree.contains("model.head · Linear") && tree.contains("net.head · Linear"),
        "{tree}"
    );
    terminal.write_all(b"\x1b[Z").unwrap();
    let other_side = diff_page(&mut terminal);
    assert!(
        other_side.contains("model.head") || other_side.contains("net.head"),
        "{other_side}"
    );
    resize_terminal(&terminal, 24, 80);
    terminal.write_all(b"v").unwrap();
    let evidence = wait_for_terminal_text(&mut terminal, "不能确认样本相同");
    assert!(String::from_utf8_lossy(&evidence).contains("不能确认样本相同"));
    terminal.write_all(b"bb").unwrap();
    wait_for_terminal_text(&mut terminal, "d 差异图");
    terminal.write_all(b"b").unwrap();
    let menu_after_preview = wait_for_terminal_text(&mut terminal, "> e. 预览 Linear 输入维度");
    after_restart.extend(menu_after_preview);
    terminal.write_all(b"g").unwrap();
    let mut after_probe = wait_for_terminal_text(&mut terminal, "检查报告 · 模块列表");
    if !String::from_utf8_lossy(&after_probe).contains("net.head") {
        after_probe.extend(wait_for_terminal_text(&mut terminal, "net.head"));
    }
    after_restart.extend(after_probe);
    terminal.write_all(b"bq").unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while read_frame(&mut terminal, &mut after_restart) {
        if Instant::now() >= deadline {
            let _ = child.kill();
            break;
        }
    }
    assert!(child.wait().unwrap().success());
    let after_restart = String::from_utf8_lossy(&after_restart);
    assert!(
        after_restart.contains("✦  Model-Maven 工作台"),
        "{after_restart}"
    );
    assert!(after_restart.contains("p. 前向诊断"), "{after_restart}");
    assert!(after_restart.contains("model.head  #2"), "{after_restart}");
    assert!(after_restart.contains("\x1b[?1003h"), "{after_restart}");
    assert!(after_restart.contains("\x1b[?1003l"), "{after_restart}");
}

fn read_frame(terminal: &mut File, output: &mut Vec<u8>) -> bool {
    let mut fd = libc::pollfd {
        fd: terminal.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: fd is a live descriptor with one pollfd to fill.
    let ready = unsafe { libc::poll(&mut fd, 1, 80) };
    assert!(ready >= 0, "PTY poll failed");
    if ready == 0 {
        return true;
    }
    let mut buffer = [0; 8192];
    match terminal.read(&mut buffer) {
        Ok(0) => false,
        Ok(length) => {
            output.extend_from_slice(&buffer[..length]);
            true
        }
        Err(error) if error.raw_os_error() == Some(libc::EIO) => false,
        Err(error) if error.kind() == io::ErrorKind::Interrupted => true,
        Err(error) => panic!("cannot read welcome PTY: {error}"),
    }
}

fn wait_for_terminal_text(terminal: &mut File, expected: &str) -> Vec<u8> {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut output = Vec::new();
    while !String::from_utf8_lossy(&output).contains(expected) {
        assert!(
            Instant::now() < deadline,
            "Missing {expected:?}\n{}",
            String::from_utf8_lossy(&output)
        );
        assert!(read_frame(terminal, &mut output));
    }
    output
}

#[test]
fn keyboard_project_switch_clears_context_at_common_sizes() {
    let root = tempfile::tempdir().unwrap();
    let first = root.path().join("first");
    let second = root.path().join("second");
    std::fs::create_dir(&first).unwrap();
    std::fs::create_dir(&second).unwrap();
    let backend = root.path().join("m-model");
    std::fs::write(&backend, r#"#!/bin/sh
case "$3 $4" in
  'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
  'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","diagnostics":[{"code":"FORWARD_FAILED","message":"fixture"}]}' ; exit 4 ;;
  *) exit 1 ;;
esac
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    for columns in [40, 100] {
        let (mut master, mut slave) = (-1, -1);
        let mut size = libc::winsize {
            ws_row: if columns == 40 { 24 } else { 30 },
            ws_col: columns,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // SAFETY: openpty receives initialized output pointers and window size.
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut size,
                )
            },
            0
        );
        // SAFETY: openpty transferred ownership of the descriptors.
        let mut terminal = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .current_dir(&first)
            .arg("--cli")
            .arg(&backend)
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .env("LC_ALL", "en_US.UTF-8")
            .env("MODEL_MAVEN_REDUCE_MOTION", "1")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                root.path().join(format!("preferences-{columns}.json")),
            )
            .env_remove("COLUMNS")
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave))
            .spawn()
            .unwrap();
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n").unwrap();
        wait_for_terminal_text(&mut terminal, "> 2. Probe");
        terminal.write_all(b"2first:build\nfirst:batch\n").unwrap();
        wait_for_terminal_text(&mut terminal, "FORWARD_FAILED");
        terminal.write_all(b"b").unwrap();
        wait_for_terminal_text(&mut terminal, "> 2. Probe");
        terminal.write_all(b"o").unwrap();
        wait_for_terminal_text(&mut terminal, "Project directory");
        terminal.write_all(b"../second\n").unwrap();
        let switched = wait_for_terminal_text(&mut terminal, "Batch  Not selected");
        let switched = String::from_utf8_lossy(&switched);
        assert!(switched.contains("second"), "{switched}");
        assert!(switched.contains("Batch  Not selected"), "{switched}");
        assert!(!switched.contains("first:batch"), "{switched}");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn setup_preview_needs_an_explicit_save_at_common_sizes() {
    use std::os::unix::process::CommandExt;
    let root = tempfile::tempdir().unwrap();
    let backend = root.path().join("m-model");
    std::fs::write(&backend, r#"#!/bin/sh
case "$3 $4" in
  'config history')
    if [ -e .saved ]; then printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}'
    else printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"error","diagnostics":[{"code":"PROJECT_NOT_INITIALIZED","message":"fixture"}]}' ; exit 2; fi ;;
  'init --dry-run') printf '{"protocol_version":1,"operation":"init","status":"success","data":{"committed":false,"path":"%s/.model-maven/project.json","config":{"schema_version":1,"model":"fixture:build","batch":"fixture:batch","batch_binding":null,"model_injections":{}}}}\n' "$PWD" ;;
  'init --model') touch .saving; sleep 0.4; touch .saved; printf '{"protocol_version":1,"operation":"init","status":"success","data":{"committed":true,"path":"%s/.model-maven/project.json"}}\n' "$PWD" ;;
  *) exit 1 ;;
esac
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    for columns in [40, 100] {
        let project = root.path().join(columns.to_string());
        std::fs::create_dir(&project).unwrap();
        let (mut master, mut slave) = (-1, -1);
        let mut size = libc::winsize {
            ws_row: if columns == 40 { 24 } else { 30 },
            ws_col: columns,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // SAFETY: openpty receives initialized output pointers and window size.
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut size,
                )
            },
            0
        );
        // SAFETY: openpty transferred ownership of the descriptors.
        let mut terminal = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        let mut command = Command::new(env!("CARGO_BIN_EXE_m-model-tui"));
        command
            .current_dir(&project)
            .arg("--cli")
            .arg(&backend)
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .env("LC_ALL", "en_US.UTF-8")
            .env("MODEL_MAVEN_REDUCE_MOTION", "1")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                root.path().join(format!("prefs-{columns}.json")),
            )
            .env_remove("COLUMNS")
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave));
        // SAFETY: only async-signal-safe system calls run after fork and before exec.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(io::Error::last_os_error());
                }
                if libc::ioctl(libc::STDIN_FILENO, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command.spawn().unwrap();
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n").unwrap();
        wait_for_terminal_text(&mut terminal, "> n. Set up");
        for confirm in [false, true] {
            terminal
                .write_all(b"nfixture:build\nfixture:batch\n")
                .unwrap();
            wait_for_terminal_text(&mut terminal, "Project setup preview");
            assert!(!project.join(".saved").exists());
            terminal.write_all(b"b").unwrap();
            wait_for_terminal_text(&mut terminal, "Save project defaults?");
            terminal
                .write_all(if confirm { b"y" } else { b"\n" })
                .unwrap();
            if confirm {
                wait_for_terminal_text(&mut terminal, "Ctrl+C waits for completion");
                let deadline = Instant::now() + Duration::from_secs(3);
                while !project.join(".saving").exists() {
                    assert!(Instant::now() < deadline, "save did not start");
                    std::thread::sleep(Duration::from_millis(10));
                }
                terminal.write_all(b"\x03").unwrap();
                wait_for_terminal_text(&mut terminal, "Project defaults saved");
                assert!(project.join(".saved").exists());
                terminal.write_all(b"b").unwrap();
            } else {
                wait_for_terminal_text(&mut terminal, "> n. Set up project");
                assert!(!project.join(".saved").exists());
            }
        }
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn missing_cli_status_is_readable_and_retryable_in_a_small_terminal() {
    let project = tempfile::tempdir().unwrap();
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 40,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: openpty receives initialized output pointers and window size.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    // SAFETY: openpty transferred ownership of the descriptors.
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .current_dir(project.path())
        .arg("--cli")
        .arg(project.path().join("missing-cli"))
        .env("TERM", "xterm-256color")
        .env("NO_COLOR", "1")
        .env("LC_ALL", "en_US.UTF-8")
        .env("MODEL_MAVEN_REDUCE_MOTION", "1")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            project.path().join("prefs.json"),
        )
        .env_remove("COLUMNS")
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Next  c.");
    terminal.write_all(b"c").unwrap();
    wait_for_terminal_text(&mut terminal, "Could not run the CLI");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "r. Retry");
    terminal.write_all(b"r").unwrap();
    wait_for_terminal_text(&mut terminal, "Could not run the CLI");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "r. Retry");
    terminal.write_all(b"\nq").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn dumb_terminal_uses_line_controls_without_an_explicit_flag() {
    let project = tempfile::tempdir().unwrap();
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 40,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: openpty receives initialized output pointers and window size.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    // SAFETY: openpty transferred ownership of the descriptors.
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .current_dir(project.path())
        .arg("--cli")
        .arg(project.path().join("missing-cli"))
        .env("TERM", "dumb")
        .env("LC_ALL", "en_US.UTF-8")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            project.path().join("prefs.json"),
        )
        .env_remove("COLUMNS")
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();
    let mut transcript = wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    transcript.extend(wait_for_terminal_text(
        &mut terminal,
        "1. Inspect model structure",
    ));
    terminal.write_all(b"1\n").unwrap();
    transcript.extend(wait_for_terminal_text(&mut terminal, "b back"));
    terminal.write_all(b"b\n").unwrap();
    transcript.extend(wait_for_terminal_text(
        &mut terminal,
        "1. Inspect model structure",
    ));
    terminal.write_all(b"q\n").unwrap();
    assert!(child.wait().unwrap().success());
    assert!(!transcript.contains(&0x1b));
    assert!(!String::from_utf8_lossy(&transcript).contains("Model  b"));
}

#[test]
fn timed_out_probe_returns_to_an_operable_workbench() {
    let project = tempfile::tempdir().unwrap();
    let backend = project.path().join("m-model");
    std::fs::write(&backend, r#"#!/bin/sh
case "$3 $4" in
  'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
  'model probe') sleep 3; printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"success","data":{}}' ;;
  'doctor ') printf '%s\n' '{"protocol_version":1,"operation":"doctor","status":"success","data":{}}' ;;
  *) exit 1 ;;
esac
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 40,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: openpty receives initialized output pointers and window size.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    // SAFETY: openpty transferred ownership of the descriptors.
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .current_dir(project.path())
        .arg("--cli")
        .arg(&backend)
        .env("TERM", "xterm-256color")
        .env("NO_COLOR", "1")
        .env("LC_ALL", "en_US.UTF-8")
        .env("MODEL_MAVEN_REDUCE_MOTION", "1")
        .env("MODEL_MAVEN_TUI_CLI_TIMEOUT_MS", "200")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            project.path().join("prefs.json"),
        )
        .env_remove("COLUMNS")
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. Probe");
    terminal
        .write_all(b"2fixture:build\nfixture:batch\n")
        .unwrap();
    wait_for_terminal_text(&mut terminal, "CLI operation timed out");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. Probe");
    terminal.write_all(b"3").unwrap();
    wait_for_terminal_text(&mut terminal, "Completed");
    terminal.write_all(b"bq").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn ctrl_c_cancels_running_probe_without_leaving_the_workbench() {
    use std::os::unix::process::CommandExt;
    let project = tempfile::tempdir().unwrap();
    let backend = project.path().join("m-model");
    std::fs::write(&backend, r#"#!/bin/sh
case "$3 $4" in
  'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
  'model probe')
    if [ ! -e .probe-seen ]; then
      touch .probe-seen
      printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","data":{"failure_location":{"last_observed_module_call":{"alias_paths":["model.head"],"module_type":"Linear"}}},"diagnostics":[{"code":"FORWARD_FAILED","message":"original evidence"}]}'
      exit 4
    fi
    touch probe-started; sleep 5; printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"success","data":{}}' ;;
  'doctor ') printf '%s\n' '{"protocol_version":1,"operation":"doctor","status":"success","data":{}}' ;;
  *) exit 1 ;;
esac
"#).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 40,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: openpty receives initialized output pointers and window size.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    // SAFETY: openpty transferred ownership of the descriptors.
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut command = Command::new(env!("CARGO_BIN_EXE_m-model-tui"));
    command
        .current_dir(project.path())
        .arg("--cli")
        .arg(&backend)
        .env("TERM", "xterm-256color")
        .env("NO_COLOR", "1")
        .env("LC_ALL", "en_US.UTF-8")
        .env("MODEL_MAVEN_REDUCE_MOTION", "1")
        .env("MODEL_MAVEN_TUI_CLI_TIMEOUT_MS", "0")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            project.path().join("prefs.json"),
        )
        .env_remove("COLUMNS")
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave));
    // SAFETY: only async-signal-safe system calls run after fork and before exec.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::ioctl(libc::STDIN_FILENO, libc::TIOCSCTTY as _, 0) < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command.spawn().unwrap();
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. Probe");
    terminal
        .write_all(b"2fixture:build\nfixture:batch\n")
        .unwrap();
    wait_for_terminal_text(&mut terminal, "FORWARD_FAILED");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. Probe");
    terminal.write_all(b"2\n\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Ctrl+C cancels this operation");
    let deadline = Instant::now() + Duration::from_secs(3);
    while !project.path().join("probe-started").exists() {
        assert!(Instant::now() < deadline, "probe did not start");
        std::thread::sleep(Duration::from_millis(10));
    }
    let started = Instant::now();
    terminal.write_all(b"\x03").unwrap();
    wait_for_terminal_text(&mut terminal, "CLI operation cancelled");
    assert!(started.elapsed() < Duration::from_secs(2));
    terminal.write_all(b"b").unwrap();
    let menu = wait_for_terminal_text(&mut terminal, "> 2. Probe");
    assert!(!String::from_utf8_lossy(&menu).contains("7. Preview Linear input size"));
    terminal.write_all(b"9").unwrap();
    let previous = wait_for_terminal_text(&mut terminal, "retry cancelled");
    assert!(!String::from_utf8_lossy(&previous).contains("latest attempt timed out"));
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "> 9. Last diagnosis");
    terminal.write_all(b"3").unwrap();
    wait_for_terminal_text(&mut terminal, "Completed");
    terminal.write_all(b"bq").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn demo_editor_publishes_each_confirmation_once_and_keeps_rejected_revision() {
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 80,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: output pointers and window size are valid for the duration of openpty.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    // SAFETY: openpty transferred both descriptors.
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--editor-demo", "--cli", "/does/not/exist/m-model"])
        .env("TERM", "xterm-ghostty")
        .env("MODEL_MAVEN_GRAPHICS", "pixels")
        .env("MODEL_MAVEN_REDUCE_MOTION", "1")
        .env("LC_ALL", "en_US.UTF-8")
        .env_remove("COLUMNS")
        .env_remove("NO_COLOR")
        .env_remove("TMUX")
        .env_remove("STY")
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();

    let first = editor_frame(&mut terminal, "DEMO EDITOR · editor · v1");
    assert_eq!(first.matches("\x1b_Ga=T,").count(), 1);

    terminal.write_all(b"e").unwrap();
    editor_frame(&mut terminal, "2× click focus");
    let click_fanout = b"\x1b[<0;14;10M\x1b[<0;14;10m";
    terminal.write_all(click_fanout).unwrap();
    editor_frame(&mut terminal, "encoder.embedding → encoder.blocks");
    terminal.write_all(click_fanout).unwrap();
    editor_frame(&mut terminal, "encoder.embedding → encoder.attention");
    terminal.write_all(b"w").unwrap();
    editor_frame(&mut terminal, "encoder.embedding → encoder.blocks");
    terminal.write_all(b"\x1b[<0;1;6M\x1b[<0;1;6me").unwrap();
    editor_frame(&mut terminal, "DEMO EDITOR · editor · v1");

    terminal.write_all(b"1128\n").unwrap();
    let review = editor_frame(&mut terminal, "Y Confirm");
    assert!(review.contains("Confirm segment from v1"));
    assert!(!review.contains("\x1b_Ga=T,"));
    terminal.write_all(b"\n\ny").unwrap();
    let shape = editor_frame(&mut terminal, "Committed v2");
    assert!(shape.contains("tensor flow v2"));
    assert!(shape.contains("2×32×256 → 2×32×128  ·  shape mismatch"));
    assert_eq!(shape.matches("\x1b_Ga=T,").count(), 1);

    terminal.write_all(b"22\ny").unwrap();
    let wire = editor_frame(&mut terminal, "Committed v3");
    assert!(wire.contains("encoder.attention.out[0] → merge.in[1]"));
    assert!(wire.contains("tensor flow v3"));
    assert_eq!(wire.matches("\x1b_Ga=T,").count(), 1);

    terminal.write_all(b"23\ny").unwrap();
    let rejected = editor_frame(&mut terminal, "Rejected · missing port · v3");
    assert!(!rejected.contains("\x1b_Ga=T,"));
    assert!(!rejected.contains("v4"));

    terminal.write_all(b"\x1b").unwrap();
    let previous = editor_frame(&mut terminal, "encoder.attention.out[0] → merge.in[1]");
    assert!(!previous.contains("\x1b_Ga=T,"));
    terminal.write_all(b"1256\ny21\ny").unwrap();
    let burst = editor_frame(&mut terminal, "Committed v5");
    let v4 = burst.find("tensor flow v4").unwrap();
    let v5 = burst.find("tensor flow v5").unwrap();
    assert!(v4 < v5);
    assert_eq!(burst[v4..v5].matches("\x1b_Ga=T,").count(), 1);
    assert_eq!(burst[v5..].matches("\x1b_Ga=T,").count(), 1);

    terminal.write_all(b"1128\t").unwrap();
    let graph_focus = editor_frame(&mut terminal, "DEMO EDITOR · graph · v5");
    assert!(!graph_focus.contains("\x1b_Ga=T,"));

    size.ws_col = 48;
    // SAFETY: the master descriptor is open and points at this test's PTY.
    assert_eq!(
        unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ, &size) },
        0
    );
    editor_frame(&mut terminal, "1–4/5");
    terminal.write_all(b"\x1b[6~").unwrap();
    let next_page = editor_frame(&mut terminal, "5–5/5");
    assert!(next_page.contains("2×32×256 → 2×32×128"));
    terminal
        .write_all(b"\x1b[<0;4;21M\x1b[<0;4;21m\x1b[5~\x1b[A\x1b[<66;4;21M\n")
        .unwrap();
    let retained = editor_frame(&mut terminal, "Confirm segment from v5");
    assert!(retained.contains("DEMO EDITOR · editor · v5"));
    assert!(!retained.contains("1–4/5"));
    assert!(!retained.contains("\x1b_Ga=T,"));
    terminal.write_all(b"\x1b").unwrap();
    let revised = editor_frame(&mut terminal, "Type the last dimension");
    assert!(revised.contains("128▏"));
    assert!(!revised.contains("\x1b_Ga=T,"));
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn offline_diagnosis_hands_off_and_rechecks_the_same_batch() {
    let preferences = tempfile::tempdir().unwrap();
    let preferences_path = preferences.path().join("preferences.json");
    let (mut terminal, mut child) =
        open_diagnosis(80, 24, &preferences_path, "en_US.UTF-8", "cells", &[]);
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    let first = editor_frame(&mut terminal, "Observed 2×48 · Expected …×32");
    assert!(first.contains("Offline sample · Failed · Details"));
    assert!(first.contains("model.head #2 · demo.py:12"));
    assert!(!first.contains("\x1b[?1049l"));
    terminal.write_all(b"\x1b[<64;6;22M").unwrap();
    let previous_call = editor_frame(&mut terminal, "model.head #1");
    assert!(!previous_call.contains("110%  ·"));
    terminal.write_all(b"\x1b[<65;6;22M").unwrap();
    editor_frame(&mut terminal, "model.head #2 · demo.py:12");
    terminal.write_all(b"\t\x1b[<64;30;10M").unwrap();
    let zoomed = editor_frame(&mut terminal, "110%  ·  OFFLINE");
    assert!(zoomed.contains("model.head #2"));
    terminal.write_all(b"\x1b[<65;30;10M\t").unwrap();
    editor_frame(&mut terminal, "Offline sample · Failed · Details");
    terminal.write_all(b"a").unwrap();
    let axes = editor_frame(&mut terminal, "Last-axis alignment");
    assert!(axes.contains("Tensor cross-sections"));
    assert!(axes.contains("2×48 (observed) ↔ …×32 (declared)"));
    assert!(axes.contains("changing-batch-01 · failed-48-32"));
    assert!(axes.contains("demo.py:12"));
    let artwork_width = |frame: &str| {
        frame
            .split("Last-axis alignment")
            .next()
            .unwrap()
            .lines()
            .map(|line| {
                line.chars()
                    .filter(|character| ('\u{2801}'..='\u{28ff}').contains(character))
                    .count()
            })
            .max()
            .unwrap()
    };
    assert!(artwork_width(&axes) > 20);
    terminal.write_all(b"+").unwrap();
    let enlarged = editor_frame(&mut terminal, "Last-axis alignment");
    assert!(artwork_width(&enlarged) > artwork_width(&axes));
    terminal.write_all(b"\x1b").unwrap();
    let restored = editor_frame(&mut terminal, "Offline sample · Failed · Details");
    assert!(restored.contains("model.head #2 · demo.py:12"));
    assert!(restored.contains("100%  ·"));
    assert!(restored.contains("observed tensor flow v1"));
    terminal.write_all(b"s12b").unwrap();
    let shortcuts = editor_frame(&mut terminal, "p Preview");
    assert!(shortcuts.contains("model.head #2"));
    let saved: serde_json::Value =
        serde_json::from_slice(&std::fs::read(preferences_path).unwrap()).unwrap();
    assert_eq!(saved["style"], "compact");
    assert_eq!(saved["preset"], "shortcuts");
    terminal.write_all(b"\x1b[5~").unwrap();
    editor_frame(&mut terminal, "model.head #1");
    terminal.write_all(b"\x1b[6~p").unwrap();
    let review = editor_frame(&mut terminal, "Confirm · Esc cancel");
    assert!(review.contains("…×32 → …×48"));
    assert!(!review.contains("Awaiting recheck"));
    terminal.write_all(b"y").unwrap();
    let committed = editor_frame(&mut terminal, "Awaiting recheck");
    assert!(committed.contains("v2 · repair-1"));
    assert!(committed.contains("Observed 2×48 · Expected …×32"));
    terminal.write_all(b"t").unwrap();
    editor_frame(&mut terminal, "Passed · 2×48 → …×48");
    terminal.write_all(b"f").unwrap();
    editor_frame(&mut terminal, "Still failing · 2×64 → …×48");
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn offline_diagnosis_reopens_at_the_same_call_and_keeps_the_diagnostic_session() {
    const ENTER_SCREEN: &str = "\x1b[?1049h";
    const LEAVE_SCREEN: &str = "\x1b[?1049l";
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            100,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &[],
        );
        let resize_handle = terminal.try_clone().unwrap();
        let mut pending = Vec::new();
        let welcome = checkpoint_frame(&mut terminal, "Open workbench", &mut pending);
        assert_eq!(welcome.matches(ENTER_SCREEN).count(), 1);
        assert!(welcome.contains("\x1b[?2026h\x1b[2J"));
        assert!(!welcome.contains(LEAVE_SCREEN));
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            let frame = checkpoint_frame(&mut terminal, expected, &mut pending);
            assert!(!frame.contains(ENTER_SCREEN));
            assert!(!frame.contains(LEAVE_SCREEN));
            frame
        };

        let initial = step(b"\n\x1b[A", "model.head #2 · demo.py:12");
        assert!(initial.contains("Observed 2×48"));
        assert!(initial.contains("e Preview"));
        assert!(!initial.contains("y Confirm"));
        assert!(step(b"", "model.head #1").contains("Observed"));
        step(b"\t+", "110%  ·  OFFLINE");
        let returned = step(b"m", "Open workbench");
        assert_eq!(returned.contains("\x1b_Ga=d,"), mode == "pixels");
        assert!(returned.contains("\x1b[?2026h"));
        assert!(!returned.contains("\x1b[?2026l\x1b[2J"));

        let reopened = step(b"\n\x1b[B", "model.head #1");
        assert!(reopened.contains("Observed call · Details"));
        assert!(reopened.contains("110%  ·  OFFLINE"));
        assert!(!reopened.contains("y Confirm"));
        step(b"", "model.head #2 · demo.py:12");
        step(b"e", "Confirm · Esc cancel");
        step(b"m", "Open workbench");
        resize_terminal(&resize_handle, 17, 48);
        step(b"", "Open workbench");
        let draft = step(b"\n", "Confirm · Esc cancel");
        assert!(draft.contains("…×32 → …×48"));
        assert!(!draft.contains("observed tensor flow v2"));
        resize_terminal(&resize_handle, 30, 100);
        let wide = step(b"", "110%  ·  OFFLINE · observed tensor flow v1");
        assert!(wide.contains("Confirm · Esc cancel"));
        step(b"y1", "Passed · 2×48 → …×48");
        step(b"m", "Open workbench");
        let rechecked = step(b"\n", "Passed · 2×48 → …×48");
        assert!(rechecked.contains("110%  ·  OFFLINE · observed tensor flow v2"));
        step(b"l", "v2 · repair-1 · net.head.in[0]");
        step(b"\x1b", "Passed · 2×48 → …×48");
        step(b"m", "Open workbench");
        terminal.write_all(b"q").unwrap();
        let cleanup = wait_for_terminal_text(&mut terminal, "\x1b[?1049l\x1b[?25h");
        assert_eq!(
            String::from_utf8_lossy(&cleanup)
                .matches(LEAVE_SCREEN)
                .count(),
            1
        );
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn motion_choice_is_kept_when_reopening_the_workbench() {
    let directory = tempfile::tempdir().unwrap();
    let (mut terminal, mut child) = open_diagnosis(
        100,
        30,
        &directory.path().join("preferences.json"),
        "en_US.UTF-8",
        "cells",
        &[],
    );
    let mut pending = Vec::new();
    checkpoint_frame(&mut terminal, "Open workbench", &mut pending);
    terminal.write_all(b"\n").unwrap();
    checkpoint_frame(&mut terminal, "Observed 2×48", &mut pending);
    terminal.write_all(b" m").unwrap();
    checkpoint_frame(&mut terminal, "Open workbench", &mut pending);
    let moving_deadline = Instant::now() + Duration::from_millis(240);
    while Instant::now() < moving_deadline {
        assert!(read_frame(&mut terminal, &mut pending));
    }
    assert!(
        String::from_utf8_lossy(&pending)
            .chars()
            .any(|ch| ('⠁'..='⣿').contains(&ch))
    );

    terminal.write_all(b"\n").unwrap();
    checkpoint_frame(&mut terminal, "Observed 2×48", &mut pending);
    terminal.write_all(b" m").unwrap();
    checkpoint_frame(&mut terminal, "Open workbench", &mut pending);
    let idle_deadline = Instant::now() + Duration::from_millis(240);
    while Instant::now() < idle_deadline {
        assert!(read_frame(&mut terminal, &mut pending));
    }
    assert!(pending.is_empty(), "paused welcome produced more frames");
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn line_diagnosis_can_reenter_the_previous_call() {
    for (terminal_mode, args) in [("dumb", &[][..]), ("pixels", &["--line-mode"][..])] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            24,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            terminal_mode,
            args,
        );
        let welcome = wait_for_terminal_text(&mut terminal, "Open workbench");
        assert!(!welcome.contains(&27));
        terminal.write_all(b"\n").unwrap();
        wait_for_terminal_text(&mut terminal, "\r\n> ");
        terminal.write_all(b"k\n").unwrap();
        wait_for_terminal_text(&mut terminal, "\r\n> ");
        terminal.write_all(b"m\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n").unwrap();
        let reopened = wait_for_terminal_text(&mut terminal, "\r\n> ");
        assert!(String::from_utf8_lossy(&reopened).contains("model.head #1"));
        assert!(!reopened.contains(&27));
        terminal.write_all(b"q\n").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn short_workbench_switches_between_graph_and_details_and_preserves_the_draft() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            24,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &[],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\ne").unwrap();
        editor_frame(&mut terminal, "y Confirm · Esc cancel");
        resize_terminal(&terminal, 17, 48);
        let compact = editor_frame(&mut terminal, "v1 · in[0] …×32 → …×48");
        assert!(!compact.contains("\x1b_Ga=T,"));
        assert_eq!(compact.contains("\x1b_Ga=d,"), mode == "pixels");
        terminal.write_all(b"\ty+").unwrap();
        let graph = editor_frame(&mut terminal, "110%  ·  OFFLINE · observed tensor flow v1");
        assert!(graph.contains("Tab Details"));
        assert!(!graph.contains("Confirm"));
        terminal.write_all(b"\x1b").unwrap();
        let draft = editor_frame(&mut terminal, "y Confirm · Esc cancel");
        assert!(draft.contains("model.head #2"));
        assert!(draft.contains("v1 · in[0] …×32 → …×48"));
        terminal.write_all(b"\t").unwrap();
        editor_frame(&mut terminal, "110%  ·  OFFLINE · observed tensor flow v1");
        resize_terminal(&terminal, 24, 80);
        editor_frame(&mut terminal, "Graph");
        terminal.write_all(b"\t\x1b").unwrap();
        let cancelled = editor_frame(&mut terminal, "e Preview");
        assert!(!cancelled.contains("repair-1 · Awaiting recheck"));
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

fn resize_terminal(terminal: &File, rows: u16, columns: u16) {
    let mut size = libc::winsize {
        ws_row: rows,
        ws_col: columns,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: the descriptor is this test's PTY and the size lives through ioctl.
    assert_eq!(
        unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ, &size) },
        0
    );
}

#[test]
fn hidden_commit_starts_feedback_when_the_graph_is_opened() {
    let directory = tempfile::tempdir().unwrap();
    let (mut terminal, mut child) = open_diagnosis(
        48,
        17,
        &directory.path().join("preferences.json"),
        "en_US.UTF-8",
        "cells/dark",
        &[],
    );
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    editor_frame(&mut terminal, "Observed 2×48");
    terminal.write_all(b" ey").unwrap();
    let receipt = editor_frame(&mut terminal, "v2 · repair-1");
    assert!(!receipt.contains("observed tensor flow"));
    std::thread::sleep(Duration::from_millis(350));
    terminal.write_all(b"\t").unwrap();
    let graph = editor_frame(&mut terminal, "observed tensor flow v2");
    assert!(
        graph.contains("\x1b[38;5;157m"),
        "missing confirmation mark"
    );
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn offline_commit_settles_with_motion_and_keeps_the_static_mark_with_reduced_motion() {
    const CONFIRMATION_INK: &str = "\x1b[38;5;157m";
    for animated in [true, false] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            100,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            "cells/dark",
            &[],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n").unwrap();
        editor_frame(&mut terminal, "Observed 2×48 · Expected …×32");
        if animated {
            terminal.write_all(b" \t").unwrap();
            editor_frame(&mut terminal, "Failed · Graph");
            editor_frame(&mut terminal, "\x1b[2K");
            terminal.write_all(b"\t").unwrap();
            editor_frame(&mut terminal, "Failed · Details");
        }
        terminal.write_all(b"ey").unwrap();
        let committed = editor_frame(&mut terminal, "v2 · repair-1 · Awaiting recheck");
        assert!(committed.contains("observed tensor flow v2"));
        assert!(committed.contains(CONFIRMATION_INK));
        assert!(committed.contains("Observed 2×48 · Expected …×32"));
        if animated {
            let changed = editor_frame(&mut terminal, "\x1b[2K");
            assert!(!changed.contains("\x1b[2J"));
        }
        let after_feedback = Instant::now() + Duration::from_millis(330);
        let mut feedback = Vec::new();
        while Instant::now() < after_feedback {
            assert!(read_frame(&mut terminal, &mut feedback));
        }
        assert!(!feedback.windows(4).any(|bytes| bytes == b"\x1b[2J"));
        terminal.write_all(b"\t+").unwrap();
        let zoomed = editor_frame(&mut terminal, "110%  ·  OFFLINE · observed tensor flow v2");
        assert_eq!(zoomed.contains(CONFIRMATION_INK), !animated);
        assert!(!zoomed.contains("tensor flow v3"));
        terminal.write_all(b"\tl").unwrap();
        let receipts = editor_frame(&mut terminal, "Offline receipts");
        assert!(receipts.contains("v2 · repair-1 · net.head.in[0]"));
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

fn open_diagnosis(
    columns: u16,
    rows: u16,
    preferences_path: &std::path::Path,
    locale: &str,
    mode: &str,
    extra: &[&str],
) -> (File, std::process::Child) {
    open_offline(
        columns,
        rows,
        preferences_path,
        locale,
        mode,
        std::iter::once("--diagnosis-demo").chain(extra.iter().copied()),
    )
}

fn open_offline<'a>(
    columns: u16,
    rows: u16,
    preferences_path: &std::path::Path,
    locale: &str,
    mode: &str,
    arguments: impl IntoIterator<Item = &'a str>,
) -> (File, std::process::Child) {
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: rows,
        ws_col: columns,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: openpty initializes the two descriptors with the given size.
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    // SAFETY: both initialized descriptors belong to this test.
    let terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut command = Command::new(env!("CARGO_BIN_EXE_m-model-tui"));
    let (mode, backdrop) = mode
        .split_once('/')
        .map_or((mode, None), |(mode, backdrop)| (mode, Some(backdrop)));
    command
        .env_remove("MODEL_MAVEN_BACKGROUND")
        .env_remove("COLORFGBG");
    if let Some(backdrop) = backdrop {
        command.env("MODEL_MAVEN_BACKGROUND", backdrop);
    }
    if mode == "pixels" || mode == "auto-cells" || backdrop.is_some() {
        command.env_remove("NO_COLOR");
    } else {
        command.env("NO_COLOR", "1");
    }
    let child = command
        .args(["--cli", "/does/not/exist/m-model"])
        .args(arguments)
        .env(
            "TERM",
            if mode == "dumb" {
                "dumb"
            } else {
                "xterm-ghostty"
            },
        )
        .env(
            "MODEL_MAVEN_GRAPHICS",
            if mode == "auto-cells" { "cells" } else { mode },
        )
        .env("MODEL_MAVEN_REDUCE_MOTION", "1")
        .env("LC_ALL", locale)
        .env("MODEL_MAVEN_TUI_PREFERENCES", preferences_path)
        // A stale shell column count must not override the real grid or a resize.
        .env("COLUMNS", "120")
        .env_remove("TMUX")
        .env_remove("STY")
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();

    (terminal, child)
}

#[test]
fn drifting_import_reports_the_unreadable_path_and_working_commands() {
    let directory = tempfile::tempdir().unwrap();
    let missing_root = directory.path().join("missing");
    let incomplete_root = directory.path().join("incomplete");
    let missing_preset = directory.path().join("another.yaml");
    std::fs::create_dir_all(incomplete_root.join("models")).unwrap();
    std::fs::write(incomplete_root.join("models/generator.py"), "").unwrap();

    let incomplete = incomplete_root.to_str().unwrap();
    for (arguments, option, missing) in [
        (
            vec!["--drifting-source", missing_root.to_str().unwrap()],
            "--drifting-source",
            missing_root.join("models/generator.py"),
        ),
        (
            vec!["--drifting-source", incomplete],
            "--drifting-source",
            incomplete_root.join("configs/gen/latent_sota_L.yaml"),
        ),
        (
            vec![
                "--drifting-source",
                incomplete,
                "--drifting-config",
                missing_preset.to_str().unwrap(),
            ],
            "--drifting-config",
            missing_preset.clone(),
        ),
    ] {
        let (mut terminal, mut child) = open_offline(
            120,
            36,
            &directory.path().join("preferences.json"),
            "zh_CN.UTF-8",
            "cells",
            arguments,
        );
        let output = wait_for_terminal_text(
            &mut terminal,
            "cargo run --release --locked -- --drifting-demo",
        );
        let output = String::from_utf8_lossy(&output);
        assert!(output.contains(&format!("Drifting 导入无法读取文件，参数为 {option}")));
        assert!(output.contains(missing.to_str().unwrap()));
        assert!(
            output
                .contains("cargo run --release --locked -- --drifting-source ./fixtures/drifting")
        );
        assert_eq!(child.wait().unwrap().code(), Some(1));
    }
}

#[test]
#[ignore = "requires Python 3.9+ to parse the bundled source"]
fn drifting_import_rejects_an_invalid_attention_preset_with_its_path() {
    let directory = tempfile::tempdir().unwrap();
    let preset = directory.path().join("invalid-heads.yaml");
    let base = include_str!("../fixtures/drifting/configs/gen/latent_sota_B.yaml");
    std::fs::write(&preset, base.replace("num_heads: 12", "num_heads: 7")).unwrap();
    let (mut terminal, mut child) = open_offline(
        120,
        36,
        &directory.path().join("preferences.json"),
        "en_US.UTF-8",
        "cells",
        [
            "--drifting-source",
            "./fixtures/drifting",
            "--drifting-config",
            preset.to_str().unwrap(),
        ],
    );
    let response = wait_for_terminal_text(&mut terminal, preset.to_str().unwrap());
    let response = String::from_utf8_lossy(&response);
    assert!(response.contains("model.hidden_size must be divisible by model.num_heads"));
    assert!(!response.contains("SOURCE · static Drifting graph"));
    assert_eq!(child.wait().unwrap().code(), Some(1));
}

#[test]
#[ignore = "requires Python 3.9+ to parse the bundled source"]
fn drifting_source_traces_a_boundary_and_restores_the_selected_page() {
    let base = [
        "--drifting-source",
        "./fixtures/drifting",
        "--drifting-config",
        "./fixtures/drifting/configs/gen/latent_sota_B.yaml",
    ];
    for (mode, arguments, depth) in [
        ("cells", ["--drifting-demo"].as_slice(), 24usize),
        ("pixels", ["--demo-graph"].as_slice(), 24),
        (
            "cells",
            ["--drifting-source", "./fixtures/drifting"].as_slice(),
            24,
        ),
        ("cells", base.as_slice(), 12),
        ("pixels", base.as_slice(), 12),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_offline(
            120,
            36,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            arguments.iter().copied(),
        );
        let mut pending = Vec::new();
        let home = checkpoint_frame(
            &mut terminal,
            "SOURCE · static Drifting graph",
            &mut pending,
        );
        assert!(home.contains("models/generator.py:553"));
        assert_eq!(home.contains("\x1b_Ga=T,"), mode == "pixels");
        let mut focus = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            checkpoint_frame(&mut terminal, expected, &mut pending)
        };
        let select = |focus: &mut dyn FnMut(&[u8], &str) -> String,
                      keys: &[u8],
                      starts: &str,
                      expected: &str| {
            for _ in 0..16 {
                let output = focus(keys, starts);
                if output.contains(expected) {
                    return output;
                }
            }
            panic!("Did not select {expected}");
        };
        let labels = "DitGen.noise_labels[0] → DitGen.conditioning[0]";
        select(&mut focus, b"w", " → ", labels);
        focus(
            b"d\n",
            "DitGen.noise_labels[0] → DitGen.conditioning.noise_embeds[1]",
        );
        let coordinate = "DitGen.noise_labels[0] → DitGen.conditioning.noise_embeds.coord_00[1]";
        let coordinates = focus(b"\n", coordinate);
        assert!(coordinates.contains("/32"));
        let label = "DitGen.noise_labels[0] → DitGen.conditioning.noise_embeds.coord_00.label[0]";
        let slice = focus(b"\n", label);
        assert!(slice.contains("models/generator.py:617 → models/generator.py:585"));
        assert!(slice.contains("scope boundary"));
        focus(b"u", labels);
        focus(b"\x1b", label);
        focus(b"\x1b", coordinate);
        focus(
            b"\x1b",
            "DitGen.noise_labels[0] → DitGen.conditioning.noise_embeds[1]",
        );
        focus(b"\x1b", labels);
        select(
            &mut focus,
            b"w",
            " → ",
            "DitGen.conditioning[0] → DitGen.backbone[1]",
        );
        let backbone = focus(b"d\n", "scope boundary");
        if !backbone.contains("DitGen.conditioning[0] → DitGen.backbone.blocks[1]") {
            select(
                &mut focus,
                b"\x1b[B",
                "SOURCE · static Drifting graph",
                "DitGen.conditioning[0] → DitGen.backbone.blocks[1]",
            );
        }
        focus(
            b"d\n",
            "DitGen.conditioning[0] → DitGen.backbone.blocks.block_00[1]",
        );
        let elapsed = Instant::now();
        let page = focus(
            b"\x1b[6~",
            "DitGen.conditioning[0] → DitGen.backbone.blocks.block_07[1]",
        );
        let page_time = elapsed.elapsed();
        let second_page = format!("8–{}/{depth}", depth.min(14));
        assert!(page.contains(&second_page));
        let selected = "DitGen.conditioning[0] → DitGen.backbone.blocks.block_07[1]";
        let exploded = focus(b"x", selected);
        assert!(exploded.contains("exploded slices"));
        focus(b"x", selected);
        focus(b"x", selected);
        let elapsed = Instant::now();
        let inside = focus(
            b"\n",
            "DitGen.conditioning[0] → DitGen.backbone.blocks.block_07.adaLN[0]",
        );
        let expand_time = elapsed.elapsed();
        assert!(inside.contains("scope boundary"));
        let attention_edge = "DitGen.backbone.blocks.block_07.modulate_attn[0] → DitGen.backbone.blocks.block_07.attention[0]";
        select(&mut focus, b"w", " → ", attention_edge);
        let projection = focus(b"d\n", "DitGen.backbone.blocks.block_07.attention.qkv[0]");
        assert!(projection.contains("scope boundary"));
        let (heads, width) = if depth == 24 { (16, 1024) } else { (12, 768) };
        let projected = select(
            &mut focus,
            b"w",
            " → ",
            "attention.qkv[0] → DitGen.backbone.blocks.block_07.attention.reshape_qkv[0]",
        );
        assert!(projected.contains(&format!("B×N×{}", 3 * width)));
        assert!(projected.contains("models/generator.py:182 → models/generator.py:183"));
        for arm in ["q", "k", "v"] {
            let slice = format!(
                "attention.reshape_qkv[0] → DitGen.backbone.blocks.block_07.attention.{arm}[0]"
            );
            let tensor = select(&mut focus, b"w", " → ", &slice);
            assert!(tensor.contains(&format!("B×N×3×{heads}×64")));
            assert!(tensor.contains("shape unresolved"));
        }
        select(
            &mut focus,
            b"w",
            " → ",
            "attention.rope[0] → DitGen.backbone.blocks.block_07.attention.head_axes[0]",
        );
        let rope_q = focus(
            b"u\n",
            "attention.rope.q_rotated[0] → DitGen.backbone.blocks.block_07.attention.head_axes[0]",
        );
        assert!(rope_q.contains("models/generator.py:131 → models/generator.py:211"));
        let rope_k = select(
            &mut focus,
            b"w",
            " → ",
            "attention.rope.k_rotated[0] → DitGen.backbone.blocks.block_07.attention.head_axes[1]",
        );
        assert!(rope_k.contains("models/generator.py:132 → models/generator.py:212"));
        let moved_key = focus(
            b"d\n",
            "attention.rope[1] → DitGen.backbone.blocks.block_07.attention.head_axes.k_heads[0]",
        );
        assert!(moved_key.contains(&format!("B×N×{heads}×64")));
        assert!(moved_key.contains("models/generator.py:197 → models/generator.py:212"));
        let key_wire =
            "head_axes.k_transposed[0] → DitGen.backbone.blocks.block_07.attention.logits[1]";
        let transposed = select(&mut focus, b"w", " → ", key_wire);
        assert!(transposed.contains(&format!("B×{heads}×64×N")));
        assert!(transposed.contains("models/generator.py:215 → models/generator.py:215"));
        focus(
            b"d",
            "attention.head_axes[1] → DitGen.backbone.blocks.block_07.attention.logits[1]",
        );
        focus(b"\x1b", key_wire);
        focus(
            b"\x1b",
            "attention.rope[1] → DitGen.backbone.blocks.block_07.attention.head_axes[1]",
        );
        focus(
            b"\x1b",
            "attention.rope.k_rotated[0] → DitGen.backbone.blocks.block_07.attention.head_axes[1]",
        );
        focus(b"bbb", attention_edge);
        focus(b"\x1b", "\x1b[?2026h");
        select(
            &mut focus,
            b"w",
            " → ",
            "DitGen.conditioning[0] → DitGen.backbone.blocks.block_07.adaLN[0]",
        );
        focus(b"+", "110%  ·");
        focus(b"u", "DitGen.conditioning[0] → DitGen.backbone[1]");
        let restored = focus(
            b"\x1b",
            "DitGen.conditioning[0] → DitGen.backbone.blocks.block_07.adaLN[0]",
        );
        assert!(restored.contains("110%  ·"));
        let page = focus(
            b"\x1b",
            "DitGen.conditioning[0] → DitGen.backbone.blocks.block_07[1]",
        );
        assert!(page.contains(&second_page));
        let remaining_pages: Vec<_> = (15..=depth)
            .step_by(7)
            .map(|first| format!("{first}–{}/{depth}", (first + 6).min(depth)))
            .collect();
        if let Some((first, rest)) = remaining_pages.split_first() {
            focus(&b"\x1b[6~".repeat(remaining_pages.len()), first);
            for page in rest {
                focus(b"", page);
            }
        }
        let last_page = remaining_pages.last().unwrap_or(&second_page);
        let last_edge = format!(
            "DitGen.backbone.blocks.block_{:02}[0] → DitGen.backbone.final[0]",
            depth - 1
        );
        select(&mut focus, b"w", " → ", &last_edge);
        focus(b"d", "DitGen.backbone.blocks[0] → DitGen.backbone.final[0]");
        let restored = focus(b"\x1b", &last_edge);
        assert!(restored.contains(last_page));
        eprintln!(
            "Drifting ({depth} blocks) {mode}: page to frame {page_time:?}, expand to frame {expand_time:?}"
        );
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn offline_style_and_control_combinations_preserve_the_diagnosis() {
    let directory = tempfile::tempdir().unwrap();
    let prefs_path = directory.path().join("preferences.json");
    let sizes = [(40, 24), (80, 24), (100, 30)];
    for (style_index, (style, divider)) in [
        ("default", "──────"),
        ("compact", "┄┄┄┄┄┄"),
        ("story", "━━━━━━"),
    ]
    .into_iter()
    .enumerate()
    {
        for (preset_index, (preset, preview, recheck)) in
            [("default", b'e', b'1'), ("shortcuts", b'p', b't')]
                .into_iter()
                .enumerate()
        {
            for (mode_index, mode) in ["cells", "cells/light", "cells/dark"]
                .into_iter()
                .enumerate()
            {
                let (columns, rows) = sizes[(style_index + mode_index) % sizes.len()];
                let chinese = (style_index + preset_index + mode_index) % 2 == 0;
                let locale = if chinese {
                    "zh_CN.UTF-8"
                } else {
                    "en_US.UTF-8"
                };
                let preferences = serde_json::json!({"style": style, "preset": preset});
                std::fs::write(&prefs_path, preferences.to_string()).unwrap();
                let (mut terminal, mut child) =
                    open_diagnosis(columns, rows, &prefs_path, locale, mode, &[]);
                wait_for_terminal_text(
                    &mut terminal,
                    if chinese {
                        "进入工作台"
                    } else {
                        "Open workbench"
                    },
                );
                terminal.write_all(b"\n").unwrap();
                let first = editor_frame(
                    &mut terminal,
                    if chinese {
                        "实际 2×48 · 预期 …×32"
                    } else {
                        "Observed 2×48 · Expected …×32"
                    },
                );
                assert!(first.contains(divider), "{style}, {mode}, {columns}×{rows}");
                assert!(first.contains("model.head #2 · demo.py:12"));
                assert_eq!(first.contains("\x1b[38;5;"), mode != "cells");

                terminal.write_all(b"?").unwrap();
                let help = editor_frame(
                    &mut terminal,
                    if chinese {
                        "当前面板操作"
                    } else {
                        "Actions in this pane"
                    },
                );
                let preview_label = if chinese { "预览" } else { "Preview" };
                assert!(
                    help.contains(&format!("{} / Enter  {preview_label}", char::from(preview)))
                );
                terminal.write_all(&[b'?', preview]).unwrap();
                let draft = editor_frame(
                    &mut terminal,
                    if chinese {
                        "确认本段"
                    } else {
                        "Confirm · Esc cancel"
                    },
                );
                assert!(draft.contains("…×32 → …×48"));
                assert!(draft.contains("net.head.in[0]"));
                terminal.write_all(b"\x1b").unwrap();
                editor_frame(
                    &mut terminal,
                    &format!("{} {preview_label}", char::from(preview)),
                );
                terminal.write_all(&[preview, b'y']).unwrap();
                editor_frame(
                    &mut terminal,
                    if chinese {
                        "待重验"
                    } else {
                        "Awaiting recheck"
                    },
                );
                terminal.write_all(&[recheck]).unwrap();
                editor_frame(
                    &mut terminal,
                    if chinese {
                        "重验通过"
                    } else {
                        "Passed · 2×48"
                    },
                );
                terminal.write_all(b"q").unwrap();
                assert!(child.wait().unwrap().success());
            }
        }
    }
}

#[test]
fn graph_background_reply_changes_ink_without_resetting_the_draft() {
    let directory = tempfile::tempdir().unwrap();
    let prefs_path = directory.path().join("preferences.json");
    let (mut terminal, mut child) =
        open_diagnosis(100, 30, &prefs_path, "en_US.UTF-8", "auto-cells", &[]);
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    let first = editor_frame(&mut terminal, "Observed 2×48 · Expected …×32");
    assert!(first.contains("\x1b]11;?\x1b\\"));
    assert!(first.contains("\x1b[38;5;216m"));
    terminal.write_all(b"e").unwrap();
    editor_frame(&mut terminal, "Confirm · Esc cancel");
    terminal.write_all(b"\x1b]11;rgb:fa/ff/f4\x1b\\").unwrap();
    let light = editor_frame(&mut terminal, "Confirm · Esc cancel");
    assert!(light.contains("\x1b[38;5;124m"));
    assert!(light.contains("observed tensor flow v1"));
    assert!(light.contains("net.head.in[0]"));

    terminal
        .write_all(b"\x1b]11;rgb:1414/1818/2424\x07")
        .unwrap();
    let dark = editor_frame(&mut terminal, "Confirm · Esc cancel");
    assert!(dark.contains("\x1b[38;5;216m"));
    assert!(dark.contains("observed tensor flow v1"));
    assert!(dark.contains("net.head.in[0]"));
    terminal.write_all(b"y").unwrap();
    editor_frame(&mut terminal, "v2 · repair-1 · Awaiting recheck");
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn offline_settings_recover_from_unknown_values_and_restore_the_saved_preset() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    std::fs::write(&path, r#"{"style":"future-style","preset":"future-keys"}"#).unwrap();
    let (mut terminal, mut child) = open_diagnosis(100, 30, &path, "en_US.UTF-8", "cells", &[]);
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    let initial = editor_frame(&mut terminal, "Could not read settings. Using defaults.");
    assert!(initial.contains("e Preview"));
    terminal.write_all(b"s2b").unwrap();
    let updated = editor_frame(&mut terminal, "p Preview");
    assert!(updated.contains("model.head #2"));
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(
        saved,
        serde_json::json!({"style":"default","preset":"shortcuts"})
    );
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());

    let (mut terminal, mut child) = open_diagnosis(100, 30, &path, "en_US.UTF-8", "cells", &[]);
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    let reopened = editor_frame(&mut terminal, "p Preview");
    assert!(reopened.contains("model.head #2"));
    assert!(reopened.contains("observed tensor flow v1"));
    terminal.write_all(b"p").unwrap();
    editor_frame(&mut terminal, "Confirm · Esc cancel");
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn graph_settings_and_shortcuts_return_to_the_same_wire_and_camera() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preferences.json");
        let (mut terminal, mut child) = open_diagnosis(48, 17, &path, "en_US.UTF-8", mode, &[]);
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            checkpoint_frame(&mut terminal, expected, &mut pending)
        };
        step(b"\n", "model.head #2 · demo.py:12");
        step(b"\t", "Tab Details · ? Controls");
        let old_help = step(b"?", "Graph controls");
        assert!(old_help.contains("Tab Details · ↑↓ Select"));
        assert!(old_help.contains("Enter Expand"));
        assert!(old_help.contains("M Port matrix · F Failure path"));
        let wire = "model.encoder[0] → net.head[0]";
        let graph = step(b"?", "Tab Details · ? Controls");
        assert!(graph.contains(wire));
        let zoomed = step(b"+", "110%  ·  OFFLINE · observed tensor flow v1");
        if mode == "pixels" {
            assert!(!zoomed.contains("\x1b[2J"));
        } else {
            assert!(zoomed.contains(wire));
        }

        step(b"s", "1  Style · Default");
        step(b"1", "1  Style · Compact");
        step(b"2", "2  Controls · Shortcuts");
        let returned = step(b"goj\x1b[Ab", "110%  ·  OFFLINE · observed tensor flow v1");
        assert!(returned.contains("g Details · ? Controls"));
        assert!(returned.contains(wire));
        assert!(returned.contains("┄┄┄┄"));
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            saved,
            serde_json::json!({"style":"compact","preset":"shortcuts"})
        );

        let new_help = step(b"?", "Graph controls");
        assert!(new_help.contains("Tab / g Details · ↑↓ / j/k Select"));
        assert!(new_help.contains("Enter / o Expand"));
        assert!(new_help.contains("M Port matrix · F Failure path"));
        step(b"\x1b", "110%  ·  OFFLINE · observed tensor flow v1");
        step(b"o", "Zoom · 0");
        let rejoined = step(b"b", "110%  ·  OFFLINE · observed tensor flow v1");
        assert!(rejoined.contains(wire));
        let details = step(b"g", "j/k calls · p Preview · g");
        assert!(details.contains("model.head #2"));
        step(b"k", "model.head #1");
        step(b"j", "model.head #2");
        step(b"p", "Confirm · Esc cancel");
        step(b"sy", "v2 · repair-1 · Awaiting recheck");
        step(b"t", "✓ Passed · 2×48");
        let before = step(b"c", "› Before · × Target failed");
        assert!(before.contains("j/k/c Compare"));
        step(b"j", "› After · ✓ Sample passed");
        step(b"k", "› Before · × Target failed");
        step(b"\x1b", "✓ Passed · 2×48");
        step(b"g", "observed tensor flow v2");
        step(b"s", "2  Controls · Shortcuts");
        let restored = step(b"r", "1  Style · Default");
        assert!(restored.contains("2  Controls · Default"));
        step(b"b", "Tab Details · ? Controls");
        step(b"\t", "✓ Passed · 2×48");
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            saved,
            serde_json::json!({"style":"default","preset":"default"})
        );
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn a_bad_setting_preserves_the_other_saved_choice() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    for (settings, style, controls, key, expected) in [
        (
            serde_json::json!({"style":"future-style","preset":"shortcuts","layout_version":2}),
            "Default",
            "Shortcuts",
            b'1',
            serde_json::json!({"style":"compact","preset":"shortcuts","layout_version":2}),
        ),
        (
            serde_json::json!({"style":"story","preset":"future-keys"}),
            "Story",
            "Default",
            b'2',
            serde_json::json!({"style":"story","preset":"shortcuts"}),
        ),
    ] {
        std::fs::write(&path, settings.to_string()).unwrap();
        let (mut terminal, mut child) = open_diagnosis(100, 30, &path, "en_US.UTF-8", "cells", &[]);
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n").unwrap();
        let start = editor_frame(&mut terminal, "Invalid setting replaced by its default.");
        assert!(start.contains("model.head #2"));
        terminal.write_all(b"s").unwrap();
        let settings = editor_frame(&mut terminal, &format!("2  Controls · {controls}"));
        assert!(settings.contains(&format!("1  Style · {style}")));
        terminal.write_all(&[key, b'b']).unwrap();
        let after = editor_frame(&mut terminal, "p Preview");
        assert!(after.contains("model.head #2"));
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved, expected);
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn rejected_settings_write_keeps_the_active_style_and_report() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    std::fs::create_dir(&path).unwrap();
    let (mut terminal, mut child) = open_diagnosis(100, 30, &path, "en_US.UTF-8", "cells", &[]);
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    editor_frame(&mut terminal, "Could not read settings. Using defaults.");
    terminal.write_all(b"s").unwrap();
    editor_frame(&mut terminal, "1  Style · Default");
    terminal.write_all(b"1").unwrap();
    let failed = editor_frame(&mut terminal, "Could not save settings.");
    assert!(!failed.contains("Style · Compact"));
    terminal.write_all(b"b").unwrap();
    let restored = editor_frame(&mut terminal, "Observed 2×48 · Expected …×32");
    assert!(restored.contains("model.head #2"));
    assert!(restored.contains("e Preview"));
    terminal.write_all(b"s").unwrap();
    editor_frame(&mut terminal, "1  Style · Default");
    assert!(path.is_dir());
    assert_eq!(directory.path().read_dir().unwrap().count(), 1);
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn call_evidence_can_follow_the_producer_without_losing_the_failed_invocation() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            100,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "large"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            checkpoint_frame(&mut terminal, expected, &mut pending)
        };
        step(b"\n", "model.head #2 · demo.py:12");
        step(b"\t+", "110%  ·");
        step(b"\t", "Offline sample · Failed · Details");
        let evidence = step(b"i", "Call evidence · failed-48-32");
        assert!(evidence.contains("model.head #2 · demo.py:12"));
        assert!(evidence.contains("Input 2×48 (observed) · Expected …×32 (declared)"));
        assert!(evidence.contains("Output ?"));
        assert!(evidence.contains("model.encoder.out[0] → net.head.in[0]"));
        assert!(evidence.contains("changing-batch-01 · trace captured"));
        assert!(evidence.contains("u Producer"));
        let full = step(b"v", "Full evidence");
        assert!(full.contains("model.head #2 · demo.py:12"));
        assert!(full.contains("model.encoder.out[0] → net.head.in[0]"));
        assert_eq!(full.contains("\x1b_Ga=d,"), mode == "pixels");
        step(b"\x1b", "Call evidence · failed-48-32");
        let trace = step(b"u", "Source trace · failed-48-32");
        assert!(trace.contains("1–1/1"));
        assert!(trace.contains("model.head #2 · demo.py:12"));
        let restored = step(b"\x1b", "Call evidence · failed-48-32");
        assert!(restored.contains("110%  ·  OFFLINE · observed tensor flow v1  |  1–3/3"));
        assert!(restored.contains("model.encoder.out[0] → net.head.in[0]"));
        step(b"\x1b", "Offline sample · Failed · Details");
        step(b"\x1b[A", "model.head #1");
        let first = step(b"i", "Call evidence · failed-48-32");
        assert!(first.contains("model.head #1 · ?"));
        assert!(first.contains("Output 2×16 (observed)"));
        assert!(first.contains("Source unknown"));
        assert!(!first.contains("u Producer"));
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn call_evidence_switches_ports_and_stops_at_a_replaced_edge() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    std::fs::write(&path, br#"{"preset":"shortcuts"}"#).unwrap();
    let (mut terminal, mut child) = open_diagnosis(
        100,
        30,
        &path,
        "en_US.UTF-8",
        "cells",
        &["--diagnosis-case", "crossed-ports"],
    );
    wait_for_terminal_text(&mut terminal, "Open workbench");
    let mut pending = Vec::new();
    let mut step = |keys: &[u8], expected: &str| {
        terminal.write_all(keys).unwrap();
        checkpoint_frame(&mut terminal, expected, &mut pending)
    };
    step(b"\npyn", "model.merge #1");
    let first = step(b"i", "Call evidence · failed-48-32");
    assert!(first.contains("Input evidence 1/2"));
    assert!(first.contains("model.encoder.out[1] → net.merge.in[0]"));
    let second = step(b"j", "Input evidence 2/2");
    assert!(second.contains("model.encoder.out[0] → net.merge.in[1]"));
    let scrolled = step(b"\x1b[<64;45;28M", "Input evidence 1/2");
    assert!(scrolled.contains("model.encoder.out[1] → net.merge.in[0]"));
    step(b"\x1b[6~", "Input evidence 2/2");
    step(b"u", "Source trace · failed-48-32");
    let restored = step(b"\x1b", "Call evidence · failed-48-32");
    assert!(restored.contains("Input evidence 2/2"));
    step(b"ipy", "v3 · repair-2 · Awaiting recheck");
    step(b"i", "model.encoder.out[1] → net.merge.in[0]");
    let stale = step(b"u", "Edge unavailable in current graph");
    assert!(!stale.contains("tensor flow v4"));
    let historical = step(b"v", "Full evidence");
    assert!(historical.contains("model.encoder.out[1] → net.merge.in[0]"));
    assert!(historical.contains("Edge unavailable in current graph"));
    step(b"\x1b", "Edge unavailable in current graph");
    step(b"j", "model.encoder.out[0] → net.merge.in[1]");
    step(b"u", "Source trace · failed-48-32");
    step(b"\x1b", "Call evidence · failed-48-32");
    step(b"i", "v3 · repair-2 · Awaiting recheck");
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn full_call_evidence_wraps_endpoints_and_scrolls_without_switching_input_ports() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("preferences.json");
        std::fs::write(&path, br#"{"preset":"shortcuts"}"#).unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            100,
            30,
            &path,
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let resize_handle = terminal.try_clone().unwrap();
        let mut pending = Vec::new();
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            checkpoint_frame(&mut terminal, expected, &mut pending)
        };
        step(b"\npyn", "model.merge #1");
        step(b"g+", "110%  ·");
        step(b"gij", "Input evidence 2/2");
        resize_terminal(&resize_handle, 8, 26);
        let compact = step(b"", "Inputs 2/2");
        assert!(compact.contains("v Full"));
        let first = step(b"v", "Full evidence · 1–6/");
        assert!(first.contains("Report failed-48-32"));
        assert!(first.contains("↑↓ Scroll · Esc Back"));
        step(b"\x1b[<65;4;4M", "Full evidence · 2–7/");
        let lower = step(b"\x1b[6~", "net.merge.in[1]");
        assert!(lower.contains("model.encoder.out[0] →"));
        assert!(!lower.contains("net.merge.in[0]"));
        step(b"\x1b[5~", "Full evidence · 1–6/");

        resize_terminal(&resize_handle, 30, 100);
        let expanded = step(b"", "Full evidence · 1–6/6");
        assert!(expanded.contains("model.encoder.out[0] → net.merge.in[1]"));
        let restored = step(b"\x1b", "Call evidence · failed-48-32");
        assert!(restored.contains("Input evidence 2/2"));
        assert!(restored.contains("110%  ·"));
        assert!(restored.contains("observed tensor flow v2"));
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn unavailable_snapshot_leaves_focus_on_the_call_evidence() {
    for (columns, rows, mode) in [(100, 30, "pixels"), (48, 17, "cells")] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            columns,
            rows,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "snapshot-missing"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            editor_frame(&mut terminal, expected)
        };
        let opened = step(b"\n", "Matching snapshot unavailable");
        assert!(opened.contains("model.head #2 · demo.py:12"));
        assert!(opened.contains("i Call evidence"));
        assert!(!opened.contains("Empty graph"));
        assert!(!opened.contains("\x1b_Ga=T,"));
        assert!(!opened.contains("Tab graph"));

        let previous = step(b"\t\x1b[A", "model.head #1");
        assert!(previous.contains("Observed call · Details"));
        step(b"\x1b[B", "model.head #2 · demo.py:12");
        let evidence = step(b"i", "Source unknown");
        assert!(evidence.contains("changing-batch-01"));
        let full = step(b"v", "Full evidence");
        assert!(full.contains("demo.py:12"));
        step(b"\x1b", "Call evidence");
        step(b"\x1b", "Matching snapshot unavailable");
        let axes = step(b"a", "Last-axis alignment");
        assert!(axes.contains("2×48 (observed) ↔ ?"));
        assert_eq!(axes.contains("Tensor cross-sections"), mode == "pixels");
        let returned = step(b"\x1b", "Matching snapshot unavailable");
        assert!(!returned.contains("\x1b_Ga=T,"));
        assert_eq!(returned.contains("\x1b_Ga=d,"), mode == "pixels");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn explicit_line_workbench_avoids_graphics_even_in_a_capable_terminal() {
    let directory = tempfile::tempdir().unwrap();
    let (mut terminal, mut child) = open_offline(
        80,
        24,
        &directory.path().join("preferences.json"),
        "en_US.UTF-8",
        "pixels",
        ["--line-mode"],
    );
    let mut transcript = wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    transcript.extend(wait_for_terminal_text(&mut terminal, "Enter: Next"));
    terminal.write_all(b"q\n").unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
        read_frame(&mut terminal, &mut transcript);
    }
    if child.try_wait().unwrap().is_none() {
        child.kill().unwrap();
        panic!("line workbench did not exit");
    }
    assert!(child.wait().unwrap().success());
    assert!(!transcript.contains(&0x1b));
    assert!(
        !String::from_utf8_lossy(&transcript)
            .chars()
            .any(|c| ('\u{2800}'..='\u{28ff}').contains(&c))
    );
}

#[test]
fn offline_line_mode_reports_missing_snapshot_without_fabricating_a_graph() {
    let directory = tempfile::tempdir().unwrap();
    let (mut terminal, mut child) = open_diagnosis(
        80,
        24,
        &directory.path().join("preferences.json"),
        "en_US.UTF-8",
        "dumb",
        &["--diagnosis-case", "snapshot-missing"],
    );
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    let frame = wait_for_terminal_text(&mut terminal, "Line mode");
    let frame = String::from_utf8_lossy(&frame);
    assert!(frame.contains("matching snapshot unavailable"));
    assert!(frame.contains("Offline sample · Last observed"));
    assert!(frame.contains("Observed 2×48 · Expected ?"));
    assert!(frame.contains("truncated"));
    assert!(!frame.contains("e Preview"));
    terminal.write_all(b"a\n").unwrap();
    let axes = wait_for_terminal_text(&mut terminal, "Expected  ┆");
    assert!(String::from_utf8_lossy(&axes).contains("2×48 (observed) ↔ ?"));
    terminal.write_all(b"b\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Offline sample · Last observed");
    terminal.write_all(b"k\n").unwrap();
    wait_for_terminal_text(&mut terminal, "model.head #1");
    terminal.write_all(b"i\n").unwrap();
    let evidence = wait_for_terminal_text(&mut terminal, "changing-batch-01 · trace truncated");
    let evidence = String::from_utf8_lossy(&evidence);
    assert!(evidence.contains("Source unknown"));
    assert!(evidence.contains("Output 2×16 (observed)"));
    assert!(!evidence.contains("Source trace"));
    terminal.write_all(b"q\n").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn line_diagnosis_preserves_the_full_evidence_for_every_style_and_preset() {
    for (style_index, style) in ["default", "compact", "story"].into_iter().enumerate() {
        for (preset_index, (preset, preview, pass)) in
            [("default", b'e', b'1'), ("shortcuts", b'p', b't')]
                .into_iter()
                .enumerate()
        {
            let chinese = (style_index + preset_index) % 2 == 0;
            let locale = if chinese {
                "zh_CN.UTF-8"
            } else {
                "en_US.UTF-8"
            };
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("preferences.json");
            let settings = serde_json::json!({"style": style, "preset": preset});
            std::fs::write(&path, settings.to_string()).unwrap();
            let (mut terminal, mut child) = open_diagnosis(40, 24, &path, locale, "dumb", &[]);
            wait_for_terminal_text(
                &mut terminal,
                if chinese {
                    "进入工作台"
                } else {
                    "Open workbench"
                },
            );

            let terminal_flags = |terminal: &File| {
                let mut mode = std::mem::MaybeUninit::<libc::termios>::uninit();
                // SAFETY: open PTY and writable termios storage.
                assert_eq!(
                    unsafe { libc::tcgetattr(terminal.as_raw_fd(), mode.as_mut_ptr()) },
                    0
                );
                let mode = unsafe { mode.assume_init() };
                (mode.c_lflag, mode.c_iflag, mode.c_oflag)
            };
            let before = terminal_flags(&terminal);
            assert_ne!(before.0 & libc::ICANON, 0);
            let step = |terminal: &mut File, input: &[u8]| {
                terminal.write_all(input).unwrap();
                terminal.write_all(b"\n").unwrap();
                let frame = wait_for_terminal_text(terminal, "\r\n> ");
                let frame = String::from_utf8(frame).unwrap();
                assert!(!frame.contains('\x1b'));
                assert!(!frame.contains("Esc"));
                assert!(
                    !frame
                        .chars()
                        .any(|c| ('\u{2800}'..='\u{28ff}').contains(&c))
                );
                frame
            };
            let baseline = step(&mut terminal, b"");
            assert!(baseline.contains("model.head #2 · demo.py:12"));
            assert!(baseline.contains("2×48"));
            let records = step(&mut terminal, b"i");
            assert!(records.contains("model.encoder.out[0] → net.head.in[0]"));
            let trace = step(&mut terminal, b"u");
            assert!(trace.contains(if chinese {
                "来源追踪"
            } else {
                "Source trace"
            }));
            let returned = step(&mut terminal, b"b");
            assert!(returned.contains(if chinese {
                "调用证据"
            } else {
                "Call evidence"
            }));
            step(&mut terminal, b"b");
            let help = step(&mut terminal, b"?");
            let action = if chinese { "预览" } else { "Preview" };
            let help = help.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(help.contains(&format!("{} / Enter {action}", char::from(preview))));
            step(&mut terminal, b"b");
            let draft = step(&mut terminal, &[preview]);
            assert!(draft.contains("net.head.in[0]"));
            assert!(draft.contains("…×32 → …×48"));
            assert!(draft.contains("net.head.weight"));
            assert!(draft.contains(if chinese { "b 取消" } else { "b cancel" }));
            let cancelled = step(&mut terminal, b"b");
            assert!(cancelled.contains("v1"));

            step(&mut terminal, &[preview]);
            let v2 = step(&mut terminal, b"y");
            assert!(v2.contains("v2 · repair-1"));
            assert!(v2.contains(if chinese {
                "待重验"
            } else {
                "Awaiting recheck"
            }));
            step(&mut terminal, b"n");
            let rewire = step(&mut terminal, &[preview]);
            for evidence in [
                "net.merge.in[0]",
                "model.encoder.out[0]",
                "net.adapter.out[0]",
                "net.merge.weight",
            ] {
                assert!(
                    rewire.contains(evidence),
                    "{evidence} is absent in {rewire}"
                );
            }
            let v3 = step(&mut terminal, b"y");
            assert!(v3.contains("v3 · repair-2"));
            step(&mut terminal, b"x");
            let rejected = step(&mut terminal, b"y");
            assert!(rejected.contains("net.adapter.out[9]"));
            assert!(rejected.contains(if chinese {
                "来源端口不存在"
            } else {
                "source port unavailable"
            }));
            assert!(rejected.contains("net.adapter.out[0] → net.merge.in[0]"));
            assert!(rejected.contains("v3") && !rejected.contains("v4"));
            step(&mut terminal, b"b");
            let passed = step(&mut terminal, &[pass]);
            assert!(passed.contains("rewire-passed"));
            step(&mut terminal, b"r");
            let partial = step(&mut terminal, b"3");
            assert!(partial.contains("rewire-truncated"));
            let original = step(&mut terminal, b"c");
            assert!(original.contains("failed-48-32"));
            assert!(original.contains("model.encoder.out[0]"));
            assert!(!original.contains("contract-repair / baseline · failed-48-32"));

            let mut size = libc::winsize {
                ws_row: 24,
                ws_col: 80,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            // SAFETY: the live PTY accepts a pointer to this window size.
            assert_eq!(
                unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ, &size) },
                0
            );
            let after = step(&mut terminal, b"c");
            assert!(after.contains("contract-repair / repair-2 · rewire-truncated"));
            assert!(after.contains("net.adapter.out[0] → net.merge.in[0]"));
            assert!(after.contains(if chinese {
                "已记录目标 · 其余调用未知"
            } else {
                "Target captured · remaining trace unknown"
            }));
            let result = step(&mut terminal, b"b");
            assert!(result.contains("rewire-truncated"));
            let receipts = step(&mut terminal, b"l");
            assert!(receipts.contains("v2 · repair-1 · net.head.in[0]"));
            assert!(receipts.contains("v3 · repair-2 · net.merge.in[0]"));

            terminal.write_all(b"q\n").unwrap();
            assert!(child.wait().unwrap().success());
            assert_eq!(terminal_flags(&terminal), before);
        }
    }
}

#[test]
fn narrow_graph_preview_shows_the_target_source_and_weight_impact() {
    let directory = tempfile::tempdir().unwrap();
    let (mut terminal, mut child) = open_diagnosis(
        40,
        24,
        &directory.path().join("preferences.json"),
        "en_US.UTF-8",
        "cells",
        &[],
    );
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    editor_frame(&mut terminal, "Observed 2×48");
    terminal.write_all(b"e").unwrap();
    let shape = editor_frame(&mut terminal, "Confirm · Esc cancel");
    assert!(shape.contains("v1 · in[0] …×32 → …×48"));
    assert!(shape.contains("model.encoder[0] → net.head.in[0]"));
    assert!(shape.contains("reset net.head.weight"));
    terminal.write_all(b"yne").unwrap();
    let rewire = editor_frame(&mut terminal, "Confirm · x Source");
    assert!(rewire.contains("v2 · net.merge.in[0]"));
    assert!(rewire.contains("model.encoder[0] → net.adapter[0]"));
    assert!(rewire.contains("reuse net.merge.weight"));
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn offline_pixel_edits_publish_once_and_rejection_preserves_the_graph() {
    let directory = tempfile::tempdir().unwrap();
    let (mut terminal, mut child) = open_diagnosis(
        100,
        30,
        &directory.path().join("preferences.json"),
        "en_US.UTF-8",
        "pixels",
        &[],
    );
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    let first = editor_frame(&mut terminal, "Observed 2×48");
    assert_eq!(first.matches("\x1b_Ga=T,").count(), 1);
    terminal.write_all(b"e").unwrap();
    let preview = editor_frame(&mut terminal, "Confirm · Esc cancel");
    assert_eq!(preview.matches("\x1b_Ga=T,").count(), 1);
    assert!(preview.contains("observed tensor flow v1"));
    terminal.write_all(b"a").unwrap();
    let axes = editor_frame(&mut terminal, "Proposed last-axis alignment");
    assert!(axes.contains("Tensor cross-sections"));
    assert!(axes.contains("Candidate"));
    assert!(axes.contains("2×48 (observed) ↔ …×48 (preview)"));
    assert_eq!(axes.matches("\x1b_Ga=T,").count(), 1);
    terminal.write_all(b"\x1b[<64;51;8M").unwrap();
    let enlarged = editor_frame(&mut terminal, "\x1b_Ga=T,");
    let payload = |frame: &str| {
        frame
            .split_once("\x1b_Ga=T,")
            .unwrap()
            .1
            .split_once(';')
            .unwrap()
            .1
            .split_once("\x1b\\")
            .unwrap()
            .0
            .to_owned()
    };
    assert_ne!(payload(&enlarged), payload(&axes));
    assert_eq!(enlarged.matches("\x1b_Ga=T,").count(), 1);
    terminal.write_all(b"a").unwrap();
    let resumed = editor_frame(&mut terminal, "Confirm · Esc cancel");
    assert!(resumed.contains("100%  ·"));
    assert!(resumed.contains("observed tensor flow v1"));
    terminal.write_all(b"y").unwrap();
    let committed = editor_frame(&mut terminal, "Awaiting recheck");
    assert_eq!(committed.matches("\x1b_Ga=T,").count(), 1);
    terminal.write_all(b"1").unwrap();
    let rechecked = editor_frame(&mut terminal, "Passed · 2×48");
    assert_eq!(rechecked.matches("\x1b_Ga=T,").count(), 0);
    terminal.write_all(b"n").unwrap();
    let next = editor_frame(&mut terminal, "model.merge #1");
    assert!(next.contains("observed tensor flow v2"));
    terminal.write_all(b"e").unwrap();
    let rewired_preview = editor_frame(&mut terminal, "Confirm · x Source");
    assert!(
        rewired_preview.contains("net.merge.in[0] · model.encoder.out[0] → net.adapter.out[0]")
    );
    assert!(rewired_preview.contains("observed tensor flow v2"));
    terminal.write_all(b"y").unwrap();
    let rewired = editor_frame(&mut terminal, "v3 · repair-2 · Awaiting recheck");
    assert!(rewired.contains("observed tensor flow v3"));
    assert_eq!(rewired.matches("\x1b_Ga=T,").count(), 1);
    terminal.write_all(b"x").unwrap();
    let invalid = editor_frame(&mut terminal, "net.adapter.out[9]");
    assert!(invalid.contains("net.merge.in[0] · net.adapter.out[0] → net.adapter.out[9]"));
    assert!(invalid.contains("observed tensor flow v3"));
    terminal.write_all(b"y").unwrap();
    let rejected = editor_frame(&mut terminal, "Rejected · source port unavailable · v3");
    assert!(!rejected.contains("v4"));
    assert_eq!(rejected.matches("\x1b_Ga=T,").count(), 0);
    terminal.write_all(b"x").unwrap();
    editor_frame(&mut terminal, "net.adapter.out[0] → net.adapter.out[0]");
    terminal.write_all(b"\x1b").unwrap();
    editor_frame(&mut terminal, "v3 · repair-2 · Awaiting recheck");
    terminal.write_all(b"2").unwrap();
    let result = editor_frame(&mut terminal, "Still failing · 2×64 → …×48 · rewire-failed");
    assert_eq!(result.matches("\x1b_Ga=T,").count(), 0);
    terminal.write_all(b"l").unwrap();
    let receipts = editor_frame(&mut terminal, "Offline receipts");
    assert!(receipts.contains("v2 · repair-1 · net.head.in[0] · ✓ Sample passed"));
    assert!(receipts.contains("v3 · repair-2 · net.merge.in[0] · × Sample failed"));
    assert_eq!(receipts.matches("\x1b_Ga=T,").count(), 0);
    terminal.write_all(b"l").unwrap();
    editor_frame(&mut terminal, "Still failing · 2×64 → …×48 · rewire-failed");
    terminal.write_all(b"c").unwrap();
    let before = editor_frame(&mut terminal, "Before · • Target observed");
    assert!(before.contains("contract-repair / baseline · failed-48-32"));
    assert!(before.contains("model.encoder.out[0] → net.merge.in[0]"));
    assert!(before.contains("observed tensor flow v3"));
    terminal.write_all(b"c").unwrap();
    let after = editor_frame(&mut terminal, "After · × Target failed");
    assert!(after.contains("contract-repair / repair-2 · rewire-failed"));
    assert!(after.contains("net.adapter.out[0] → net.merge.in[0]"));
    assert!(after.contains("observed tensor flow v3"));
    terminal.write_all(b"\x1b").unwrap();
    editor_frame(&mut terminal, "Still failing · 2×64 → …×48 · rewire-failed");
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn recheck_compares_batch_evidence_in_wide_and_narrow_workbenches() {
    for (columns, rows, mode) in [(100, 30, "cells"), (40, 24, "pixels")] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            columns,
            rows,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &[],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            editor_frame(&mut terminal, expected)
        };
        step(b"\n", "Observed 2×48");
        step(b"e", "Confirm · Esc cancel");
        let receipt = step(b"y", "Awaiting recheck ◌");
        assert!(receipt.contains("v2 · repair-1"));
        assert!(receipt.contains("r Recheck samples"));

        let choices = step(b"r", "4  Different batch");
        assert!(choices.contains("3  Truncated trace"));
        let rejected = step(b"4", "Recheck batch mismatch");
        assert!(rejected.contains("changing-batch-01 ≠ different-batch-02"));
        let pending = step(b"l", "Offline receipts");
        let receipt_identity = if columns == 100 {
            "v2 · repair-1 · net.head.in[0] · Awaiting recheck"
        } else {
            "v2 · net.head.in[0] · Awaiting recheck"
        };
        assert!(pending.contains(receipt_identity));
        step(b"l", "Awaiting recheck ◌");

        step(b"r3", "? Coverage incomplete · 2×48");
        if columns == 100 {
            step(b"\th+", "110%  ·  OFFLINE · structure snapshot v2");
            step(b"\t", "Details");
        }
        let before = step(b"c", "› Before · × Target failed");
        assert!(before.contains("changing-batch-01 · v2"));
        assert!(before.contains("demo.py:12"));
        if columns == 100 {
            assert!(before.contains("100%  ·  OFFLINE · observed tensor flow v2"));
            assert!(before.contains("│   After · ? Coverage incomplete"));
            assert!(before.contains("repair-1 · recheck-truncated"));
            assert!(before.contains("2×48 → …×32"));
            assert!(before.contains("2×48 → …×48"));
        } else {
            assert!(!before.contains("After · ? Coverage incomplete"));
            assert!(before.contains("baseline · failed-48-32"));
        }
        let forward = if columns == 100 {
            b"\x1b[6~".as_slice()
        } else {
            b"\x1b[<65;6;21M".as_slice()
        };
        let after = step(forward, "› After · ? Coverage incomplete");
        if columns == 100 {
            assert!(after.contains("100%  ·  OFFLINE · observed tensor flow v2"));
            assert!(after.contains("Target captured · remaining trace"));
        } else {
            assert!(after.contains("Target captured · rest unknown"));
            assert!(after.contains("repair-1 · recheck-truncated"));
        }
        assert!(after.contains("v Full"));
        step(b"\x1b[A", "› Before · × Target failed");
        step(b"\x1b[6~", "› After · ? Coverage incomplete");
        let full = step(b"\x1b[<64;12;10Mv", "Recheck evidence");
        assert!(full.contains("Target net.head.in[0]"));
        assert!(full.contains("demo.py:12"));
        assert!(full.contains("recheck-truncated"));
        assert!(full.contains("remaining trace"));
        assert!(full.contains("› After · ? Coverage incomplete"));
        let restored_comparison = step(b"\x1b", "› After · ? Coverage incomplete");
        assert!(restored_comparison.contains("100%  ·  OFFLINE"));
        let restored = step(b"\x1b", "? Coverage incomplete · 2×48");
        if columns == 100 {
            assert!(restored.contains("110%  ·  OFFLINE · structure snapshot v2"));
        }

        let rejected = step(b"r4", "Recheck batch mismatch");
        assert!(rejected.contains("? Coverage incomplete · 2×48"));
        step(b"\x1b", "Observed 2×48");
        step(b"2", "× Still failing · 2×64");
        let before_failure = step(b"c", "› Before · × Target failed");
        let failed = step(b"c", "› After · × Target failed");
        let comparison = if columns == 100 {
            &before_failure
        } else {
            &failed
        };
        assert!(comparison.contains("2×64 → …×48"));
        let full_failure = step(b"v", "Recheck evidence");
        assert!(full_failure.contains("recheck-failed"));
        assert!(full_failure.contains("2×64 → …×48"));
        step(b"\x1b", "› After · × Target failed");
        step(b"\x1b", "× Still failing · 2×64");
        step(b"1", "✓ Passed · 2×48");
        step(b"c", "› Before · × Target failed");
        step(b"c", "› After · ✓ Sample passed");
        let full_pass = step(b"v", "Recheck evidence");
        assert!(full_pass.contains("recheck-passed"));
        assert!(full_pass.contains("2×48 → …×48"));
        step(b"\x1b", "› After · ✓ Sample passed");
        step(b"\x1b", "✓ Passed · 2×48");
        step(b"\x1b", "Offline receipts");
        step(b"\x1b", "✓ Passed · 2×48");

        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn matrix_mismatch_jump_is_visible_and_keeps_the_selected_connection() {
    for mode in ["cells", "pixels"] {
        for width in [40, 80] {
            let directory = tempfile::tempdir().unwrap();
            let (mut terminal, mut child) = open_diagnosis(
                width,
                24,
                &directory.path().join("preferences.json"),
                "en_US.UTF-8",
                mode,
                &["--diagnosis-case", "crossed-ports"],
            );
            wait_for_terminal_text(&mut terminal, "Open workbench");
            let mut pending = Vec::new();
            terminal.write_all(b"\n\tM").unwrap();
            let matrix = checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
            assert!(
                matrix
                    .lines()
                    .any(|line| line.starts_with('>') && line.contains('[')),
                "{matrix}"
            );
            terminal.write_all(b"n").unwrap();
            let selected = checkpoint_frame(&mut terminal, "Mismatch cell", &mut pending);
            assert!(selected.contains("[×]"), "{selected}");
            assert!(
                selected.contains("model.encoder[0] → net.head[0]"),
                "{selected}"
            );
            terminal.write_all(b"p\na").unwrap();
            checkpoint_frame(&mut terminal, "Axis index", &mut pending);
            terminal.write_all(b"\x1b").unwrap();
            checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
            terminal.write_all(b"\x1b").unwrap();
            let graph = checkpoint_frame(&mut terminal, "model.encoder", &mut pending);
            assert!(graph.contains("model.encoder[0] → net.head[0]"), "{graph}");
            terminal.write_all(b"q").unwrap();
            assert!(child.wait().unwrap().success());
        }
    }
}

#[test]
fn matrix_axis_index_navigates_without_losing_the_current_cell() {
    for mode in ["cells", "pixels"] {
        for width in [40, 80] {
            let directory = tempfile::tempdir().unwrap();
            let (mut terminal, mut child) = open_diagnosis(
                width,
                24,
                &directory.path().join("preferences.json"),
                "en_US.UTF-8",
                mode,
                &["--diagnosis-case", "crossed-ports"],
            );
            wait_for_terminal_text(&mut terminal, "Open workbench");
            let mut pending = Vec::new();
            terminal.write_all(b"\n\tM").unwrap();
            checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
            terminal.write_all(b"a").unwrap();
            let rows = checkpoint_frame(&mut terminal, "Axis index · output rows", &mut pending);
            assert!(rows.contains("model.encoder.out[0]"), "{rows}");
            terminal.write_all(b"v").unwrap();
            let detail = checkpoint_frame(&mut terminal, "Axis port details", &mut pending);
            assert!(
                detail.contains("model.encoder") && detail.contains("out[0]"),
                "{detail}"
            );
            resize_terminal(&terminal, 14, 34);
            checkpoint_frame(&mut terminal, "Axis port details", &mut pending);
            terminal.write_all(b"\x1b[<65;10;5M").unwrap();
            resize_terminal(&terminal, 24, width);
            checkpoint_frame(&mut terminal, "Axis port details", &mut pending);
            terminal.write_all(b"v").unwrap();
            checkpoint_frame(&mut terminal, "Axis index · output rows", &mut pending);
            terminal.write_all(b"\x1b[B\x1b").unwrap();
            checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
            terminal.write_all(b"a\t").unwrap();
            let columns =
                checkpoint_frame(&mut terminal, "Axis index · input columns", &mut pending);
            assert!(columns.contains(".in[0]"), "{columns}");
            terminal.write_all(b"\x1b[B\n").unwrap();
            checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
            terminal.write_all(b"ga\t").unwrap();
            let grouped =
                checkpoint_frame(&mut terminal, "Axis index · input columns", &mut pending);
            assert!(grouped.contains(".in[*]"), "{grouped}");
            terminal.write_all(b"v").unwrap();
            let detail = checkpoint_frame(&mut terminal, "Axis port details", &mut pending);
            assert!(
                detail.contains("Group members") && detail.contains("in[0]"),
                "{detail}"
            );
            terminal.write_all(b"\x1b").unwrap();
            checkpoint_frame(&mut terminal, "Axis index · input columns", &mut pending);
            // Dragging away cancels the press; Tab proves the index stayed open.
            terminal
                .write_all(b"\x1b[<0;3;3M\x1b[<32;3;4M\x1b[<0;3;4m\t")
                .unwrap();
            checkpoint_frame(&mut terminal, "Axis index · output rows", &mut pending);
            terminal.write_all(b"\x1b[<0;3;4M\x1b[<0;3;4m").unwrap();
            checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
            terminal.write_all(b"q").unwrap();
            assert!(child.wait().unwrap().success());
        }
    }
}

#[test]
fn matrix_groups_preserve_port_selection_when_expanded() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\tM").unwrap();
        checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
        terminal.write_all(b"\x1b[<0;15;4M\x1b[<0;15;4m").unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[0] → net.merge[1]",
            &mut pending,
        );
        terminal.write_all(b"g").unwrap();
        let grouped = checkpoint_frame(&mut terminal, "Pairs 2/4", &mut pending);
        assert!(grouped.contains("remaining unknown"));
        assert!(grouped.contains("consumers 2"));
        terminal.write_all(b"w").unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[1] → net.merge[0]",
            &mut pending,
        );
        terminal.write_all(b"\x1bM").unwrap();
        let restored = checkpoint_frame(&mut terminal, "Pairs 2/4", &mut pending);
        assert!(restored.contains("model.encoder[1] → net.merge[0]"));
        terminal.write_all(b"g\x1b").unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[1] → net.merge[0]",
            &mut pending,
        );
        terminal.write_all(b"q").unwrap();
        wait_for_terminal_text(&mut terminal, "\x1b[?1049l");
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn port_matrix_selects_and_returns_to_the_same_wire() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\tM").unwrap();
        let frame = checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
        assert!(frame.contains("Rows output · columns input"));
        // Dragging out and returning to a cell must not select it.
        terminal
            .write_all(b"\x1b[<0;15;4M\x1b[<32;2;2M\x1b[<0;15;4mM")
            .unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[0] → net.head[0]",
            &mut pending,
        );
        terminal.write_all(b"M").unwrap();
        checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
        // A release after focus loss cannot confirm the old press.
        terminal
            .write_all(b"\x1b[<0;15;4M\x1b[O\x1b[I\x1b[<0;15;4mM")
            .unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[0] → net.head[0]",
            &mut pending,
        );
        terminal.write_all(b"M").unwrap();
        checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
        terminal.write_all(b"\x1b[<65;15;4M").unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[1] → net.head[0]",
            &mut pending,
        );
        terminal.write_all(b"\x1b[<64;15;4M").unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[0] → net.head[0]",
            &mut pending,
        );
        terminal.write_all(b"\x1b[<0;15;4M\x1b[<0;15;4m").unwrap();
        let chosen = checkpoint_frame(&mut terminal, "→", &mut pending);
        assert!(
            chosen.contains("model.encoder[0] → net.merge[1]"),
            "{chosen}"
        );
        terminal.write_all(b"\x1b").unwrap();
        let graph = checkpoint_frame(&mut terminal, "model.encoder", &mut pending);
        assert!(graph.contains("model.encoder[0] → net.merge[1]"));
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn search_opens_a_loaded_port_and_cancel_restores_graph_in_both_renderers() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\t/").unwrap();
        checkpoint_frame(&mut terminal, "Search · loaded", &mut pending);
        terminal.write_all(b"mer\x1b[Zge.in[1]").unwrap();
        checkpoint_frame(&mut terminal, "net.merge.in[1]", &mut pending);
        terminal.write_all(b"\x1b").unwrap();
        checkpoint_frame(&mut terminal, "Tab Details", &mut pending);
        terminal.write_all(b"/merge.in[1]").unwrap();
        checkpoint_frame(&mut terminal, "net.merge.in[1]", &mut pending);
        terminal.write_all(b"\n").unwrap();
        let graph = checkpoint_frame(
            &mut terminal,
            "model.encoder[0] → net.merge[1]",
            &mut pending,
        );
        assert!(graph.contains("net.merge"));
        terminal.write_all(b"b").unwrap();
        checkpoint_frame(&mut terminal, "Tab Details", &mut pending);
        terminal.write_all(b"/").unwrap();
        let restored = checkpoint_frame(&mut terminal, "/ [merge.in[1]]", &mut pending);
        assert!(restored.contains("net.merge.in[1]"), "{restored}");
        terminal.write_all(b"\x15").unwrap();
        let cleared = checkpoint_frame(&mut terminal, "Search · loaded", &mut pending);
        assert!(cleared.contains("/ "), "{cleared}");
        terminal.write_all(b"\x1b").unwrap();
        terminal.write_all(b"q").unwrap();
        wait_for_terminal_text(&mut terminal, "\x1b[?1049l");
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn repair_diff_reviews_preview_and_receipt_in_both_renderers() {
    for (mode, width) in [
        ("cells", 100),
        ("pixels", 100),
        ("cells", 40),
        ("pixels", 40),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            width,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "mismatch"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\ne").unwrap();
        checkpoint_frame(&mut terminal, "Preview", &mut pending);
        terminal.write_all(b"D").unwrap();
        let before = checkpoint_frame(&mut terminal, "Repair difference", &mut pending);
        assert!(before.contains("Preview · not committed"));
        assert!(before.contains("Before · recorded"));
        if width < 72 {
            terminal.write_all(b"\x1b[Z").unwrap();
            checkpoint_frame(&mut terminal, "Candidate · preview", &mut pending);
        } else {
            assert!(before.contains("Candidate · preview"));
        }
        terminal.write_all(b"v").unwrap();
        checkpoint_frame(&mut terminal, "Repair evidence", &mut pending);
        terminal.write_all(b"\x1b").unwrap();
        checkpoint_frame(&mut terminal, "Repair difference", &mut pending);
        terminal.write_all(b"Dy").unwrap();
        checkpoint_frame(&mut terminal, "Awaiting recheck", &mut pending);
        terminal.write_all(b"D").unwrap();
        checkpoint_frame(&mut terminal, "Committed · awaiting recheck", &mut pending);
        terminal.write_all(b"Dq").unwrap();
        wait_for_terminal_text(&mut terminal, "\x1b[?1049l");
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn failure_path_preserves_the_selected_input_when_returning_to_overview() {
    for mode in ["cells", "pixels"] {
        for (width, height) in [(80, 30), (40, 24)] {
            let directory = tempfile::tempdir().unwrap();
            let (mut terminal, mut child) = open_diagnosis(
                width,
                height,
                &directory.path().join("preferences.json"),
                "en_US.UTF-8",
                mode,
                &["--diagnosis-case", "crossed-ports"],
            );
            wait_for_terminal_text(&mut terminal, "Open workbench");
            let mut pending = Vec::new();
            terminal.write_all(b"\nF").unwrap();
            let failure =
                checkpoint_frame(&mut terminal, "Failure path · call evidence", &mut pending);
            assert!(failure.contains("model.head #2"));
            assert!(failure.contains("2×48 (observed)"));
            assert!(failure.contains("…×32 (declared)"));
            assert!(
                failure.contains(if width < 48 {
                    "↵ graph · Esc"
                } else {
                    "Enter graph · F/Esc back"
                }),
                "{failure}"
            );
            terminal.write_all(b"F\x1b[B").unwrap();
            checkpoint_frame(&mut terminal, "model.merge #1", &mut pending);
            terminal.write_all(b"F").unwrap();
            checkpoint_frame(&mut terminal, "Input 1/2", &mut pending);
            terminal.write_all(b"\x1b[B").unwrap();
            let second = checkpoint_frame(&mut terminal, "Input 2/2", &mut pending);
            assert!(second.contains("net.merge.in[1]"));
            terminal.write_all(b"v").unwrap();
            let evidence = checkpoint_frame(&mut terminal, "Recorded call evidence", &mut pending);
            assert!(evidence.contains("model.merge #1"));
            terminal.write_all(b"\x1b").unwrap();
            checkpoint_frame(&mut terminal, "Input 2/2", &mut pending);
            terminal.write_all(b"FF").unwrap();
            checkpoint_frame(&mut terminal, "Input 2/2", &mut pending);
            terminal.write_all(b"Fq").unwrap();
            wait_for_terminal_text(&mut terminal, "\x1b[?1049l");
            assert!(child.wait().unwrap().success());
        }
    }
}

#[test]
fn localized_narrow_failure_path_keeps_its_return_hint_visible() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            40,
            24,
            &directory.path().join("preferences.json"),
            "zh_CN.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "进入工作台");
        let mut pending = Vec::new();
        terminal.write_all(b"\nF").unwrap();
        let failure = checkpoint_frame(&mut terminal, "故障路径 · 调用证据", &mut pending);
        assert!(failure.contains("2×48 (观测)"), "{failure}");
        assert!(failure.contains("…×32 (声明)"), "{failure}");
        assert!(failure.contains("↵ 图 · Esc 返回"), "{failure}");
        terminal.write_all(b"Fq").unwrap();
        wait_for_terminal_text(&mut terminal, "\x1b[?1049l");
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn exploded_graph_module_details_return_to_the_selected_view() {
    for mode in ["cells", "pixels"] {
        for width in [40, 100] {
            let directory = tempfile::tempdir().unwrap();
            let (mut terminal, mut child) = open_offline(
                width,
                30,
                &directory.path().join("preferences.json"),
                "en_US.UTF-8",
                mode,
                ["--drifting-demo"],
            );
            let mut pending = Vec::new();
            checkpoint_frame(
                &mut terminal,
                "SOURCE · static Drifting graph",
                &mut pending,
            );
            terminal.write_all(b"/block_07.attention.qkv").unwrap();
            checkpoint_frame(&mut terminal, "[source]", &mut pending);
            terminal.write_all(b"\n").unwrap();
            checkpoint_frame(&mut terminal, "v detail", &mut pending);
            terminal.write_all(b"x").unwrap();
            checkpoint_frame(&mut terminal, "exploded slices 1/2", &mut pending);
            terminal.write_all(b"v").unwrap();
            let details = checkpoint_frame(&mut terminal, "Module details", &mut pending);
            assert!(
                details.contains("DitGen.backbone") && details.contains("qkv"),
                "{details}"
            );
            assert!(details.contains("Type"), "{details}");
            resize_terminal(&terminal, 14, 34);
            checkpoint_frame(&mut terminal, "Module details", &mut pending);
            terminal.write_all(b"\x1b[<65;10;5M").unwrap();
            resize_terminal(&terminal, 30, width);
            checkpoint_frame(&mut terminal, "Module details", &mut pending);
            terminal.write_all(b"v").unwrap();
            let graph = checkpoint_frame(&mut terminal, "exploded slices 1/2", &mut pending);
            assert!(
                graph.contains("qkv") && graph.contains("v detail"),
                "{graph}"
            );
            terminal.write_all(b"q").unwrap();
            assert!(child.wait().unwrap().success());
        }
    }
}

#[test]
fn source_search_reveals_a_folded_module_in_both_renderers() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_offline(
            120,
            36,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            ["--drifting-demo"],
        );
        let mut pending = Vec::new();
        checkpoint_frame(
            &mut terminal,
            "SOURCE · static Drifting graph",
            &mut pending,
        );
        terminal.write_all(b"/block_07.attention.qkv").unwrap();
        let results = checkpoint_frame(
            &mut terminal,
            "DitGen.backbone.blocks.block_07.attention.qkv",
            &mut pending,
        );
        assert!(results.contains("[source]"));
        terminal.write_all(b"\n").unwrap();
        let graph = checkpoint_frame(
            &mut terminal,
            "SOURCE · static Drifting graph",
            &mut pending,
        );
        assert!(graph.contains("block_07.attention.qkv"), "{graph}");
        terminal.write_all(b"\x1b").unwrap();
        checkpoint_frame(
            &mut terminal,
            "SOURCE · static Drifting graph",
            &mut pending,
        );
        terminal.write_all(b"q").unwrap();
        wait_for_terminal_text(&mut terminal, "\x1b[?1049l");
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn exploded_qkv_remains_readable_across_spacing_levels_and_pages() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_offline(
            120,
            36,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            ["--drifting-demo"],
        );
        let mut pending = Vec::new();
        let root = checkpoint_frame(
            &mut terminal,
            "SOURCE · static Drifting graph",
            &mut pending,
        );
        assert!(root.contains("DitGen.backbone"), "{root}");
        for level in [1, 2] {
            terminal.write_all(b"x").unwrap();
            let frame = checkpoint_frame(
                &mut terminal,
                &format!("exploded slices {level}/2"),
                &mut pending,
            );
            for ordinal in 1..=3 {
                assert!(
                    frame.contains(&format!("s{ordinal}")),
                    "{mode} root explosion hides slice s{ordinal}: {frame}"
                );
            }
        }
        terminal.write_all(b"x").unwrap();
        checkpoint_frame(&mut terminal, "CAM 032°/+30°", &mut pending);
        terminal.write_all(b"/block_07.attention.qkv").unwrap();
        checkpoint_frame(
            &mut terminal,
            "DitGen.backbone.blocks.block_07.attention.qkv",
            &mut pending,
        );
        terminal.write_all(b"\n").unwrap();

        let spatial = checkpoint_frame(
            &mut terminal,
            "SOURCE · static Drifting graph",
            &mut pending,
        );
        assert!(
            spatial.contains("qkv") && spatial.contains("reshape_qkv"),
            "{spatial}"
        );
        for level in [1, 2] {
            terminal.write_all(b"x").unwrap();
            let frame = checkpoint_frame(
                &mut terminal,
                &format!("exploded slices {level}/2"),
                &mut pending,
            );
            assert!(frame.contains("qkv"), "{frame}");
            assert!(frame.contains("reshape_qkv"), "{frame}");
        }

        terminal.write_all(b"\x1b[6~").unwrap();
        let next_page = checkpoint_frame(&mut terminal, "q_norm", &mut pending);
        assert!(next_page.contains("k_norm"), "{next_page}");
        assert!(next_page.contains("q_norm"), "{next_page}");
        assert!(next_page.contains("exploded slices 2/2"), "{next_page}");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn graph_commands_open_matrix_and_enter_exploded_view_from_2d() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\t?").unwrap();
        checkpoint_frame(&mut terminal, "/ Search and commands", &mut pending);
        terminal.write_all(b"/").unwrap();
        checkpoint_frame(&mut terminal, "Open port matrix", &mut pending);
        terminal.write_all(b"\x1b").unwrap();
        checkpoint_frame(&mut terminal, "Graph controls", &mut pending);
        terminal.write_all(b"/matrix").unwrap();
        checkpoint_frame(&mut terminal, "Open port matrix", &mut pending);
        terminal.write_all(b"\n").unwrap();
        checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
        terminal.write_all(b"\x1bz").unwrap();
        checkpoint_frame(&mut terminal, "2D", &mut pending);
        terminal.write_all(b"/spacing").unwrap();
        checkpoint_frame(&mut terminal, "Spread to exploded x1", &mut pending);
        terminal.write_all(b"\n").unwrap();
        checkpoint_frame(&mut terminal, "exploded slices 1/2", &mut pending);
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn narrow_flat_flow_survives_paging_and_resize() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            40,
            24,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let resize_handle = terminal.try_clone().unwrap();
        let mut pending = Vec::new();
        terminal.write_all(b"\n\tz").unwrap();
        checkpoint_frame(&mut terminal, "2D", &mut pending);
        terminal.write_all(b"]").unwrap();
        checkpoint_frame(&mut terminal, "2D", &mut pending);
        terminal.write_all(b"[").unwrap();
        checkpoint_frame(&mut terminal, "2D", &mut pending);
        terminal.write_all(b"w").unwrap();
        checkpoint_frame(&mut terminal, "net.merge", &mut pending);
        resize_terminal(&resize_handle, 30, 100);
        let wide = checkpoint_frame(&mut terminal, "net.merge", &mut pending);
        assert!(wide.contains("2D"));
        terminal.write_all(b"z").unwrap();
        checkpoint_frame(&mut terminal, "net.merge", &mut pending);
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn exploded_level_stays_visible_and_selected_wire_survives_resize_and_rotation() {
    const WIRES: [&str; 2] = [
        "model.encoder[1] → net.merge[0]",
        "model.encoder[0] → net.merge[1]",
    ];
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            100,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let resize_handle = terminal.try_clone().unwrap();
        let mut pending = Vec::new();
        terminal.write_all(b"\n\t").unwrap();
        let unpinned = checkpoint_frame(&mut terminal, "Tab Details · ? Controls", &mut pending);
        terminal.write_all(b"w").unwrap();
        let selected = checkpoint_frame(&mut terminal, "→ net.merge[", &mut pending);
        let wire = WIRES
            .into_iter()
            .find(|wire| selected.contains(wire))
            .expect("selected wire");
        if mode == "cells" {
            let braille_signature = |frame: &str| {
                frame
                    .chars()
                    .filter(|ch| ('\u{2801}'..='\u{28ff}').contains(ch))
                    .collect::<String>()
            };
            assert!(
                braille_signature(&selected) != braille_signature(&unpinned),
                "pinning a crossed port should update the monochrome line rendering"
            );
        }
        terminal.write_all(b"x").unwrap();
        let first = checkpoint_frame(&mut terminal, "exploded slices 1/2", &mut pending);
        assert!(first.contains(wire), "{first}");
        terminal.write_all(b"x\x1b[C").unwrap();
        let second = checkpoint_frame(&mut terminal, "exploded slices 2/2", &mut pending);
        assert!(second.contains(wire), "{second}");
        resize_terminal(&resize_handle, 24, 30);
        checkpoint_frame(&mut terminal, "x2/2", &mut pending);
        resize_terminal(&resize_handle, 24, 40);
        checkpoint_frame(&mut terminal, "exploded slices 2/2", &mut pending);
        terminal.write_all(b"L").unwrap();
        let layout = checkpoint_frame(&mut terminal, "L Layout on", &mut pending);
        assert!(layout.contains("[nodes]"), "{layout}");
        terminal.write_all(b"L").unwrap();
        checkpoint_frame(&mut terminal, "L Layout off", &mut pending);
        terminal.write_all(b"x").unwrap();
        let collapsed = checkpoint_frame(&mut terminal, "100%  ·", &mut pending);
        assert!(!collapsed.contains("exploded slices"), "{collapsed}");
        resize_terminal(&resize_handle, 30, 100);
        let restored = checkpoint_frame(&mut terminal, wire, &mut pending);
        assert!(!restored.contains("exploded slices"), "{restored}");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn crossed_ports_can_both_be_selected_and_one_can_be_rewired() {
    const REWIRE: &str = "model.encoder[1] → net.merge[0]";
    const RETAIN: &str = "model.encoder[0] → net.merge[1]";
    const PICK: &[u8] = b"\x1b[<0;27;8M\x1b[<0;27;8m";
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            checkpoint_frame(&mut terminal, expected, &mut pending)
        };
        step(b"\n", "model.head #2 · demo.py:12");
        step(b"\t", "Tab Details · ? Controls");
        step(b"z", "2D");
        let first_flat = step(b"w", "→ net.merge[");
        let second_flat = step(b"w", "→ net.merge[");
        assert!(first_flat.contains(REWIRE) || second_flat.contains(REWIRE));
        assert!(first_flat.contains(RETAIN) || second_flat.contains(RETAIN));
        step(b"z", "Tab Details · ? Controls");
        let picked = step(PICK, "→ net.merge[");
        let (first, second) = if picked.contains(REWIRE) {
            (REWIRE, RETAIN)
        } else {
            assert!(picked.contains(RETAIN));
            (RETAIN, REWIRE)
        };
        step(PICK, second);
        step(b"w", first);
        step(b"\t", "Offline sample · Failed · Details");
        step(b"ey", "v2 · repair-1 · Awaiting recheck");
        let candidate = step(b"ne", "Confirm · x Source");
        assert!(candidate.contains("net.merge.in[0] · model.encoder.out[1] → net.adapter.out[0]"));
        let committed = step(b"y", "v3 · repair-2 · Awaiting recheck");
        assert!(committed.contains("net.adapter[0] → net.merge[0]"));
        step(b"\t", "Tab Details · ? Controls");
        let mut retained = false;
        for _ in 0..4 {
            let frame = step(b"w", "→ ");
            if frame.contains(RETAIN) {
                retained = true;
                break;
            }
        }
        assert!(
            retained,
            "retained wire must still be selectable after rewire"
        );
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn twin_merge_preview_names_the_target_in_a_narrow_window_and_cancels_back_to_the_tree() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            120,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "twin-merge"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let resize_handle = terminal.try_clone().unwrap();
        let mut pending = Vec::new();
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            checkpoint_frame(&mut terminal, expected, &mut pending)
        };
        step(b"\n", "model.merge_a #2");
        step(b"\th+", "110%  ·  OFFLINE · structure snapshot v1");
        let preview = step(b"\te", "Confirm · Esc cancel");
        assert!(preview.contains("100%  ·  OFFLINE · observed tensor flow v1"));
        assert!(preview.contains("net.merge_a.in[0]"));
        resize_terminal(&resize_handle, 14, 34);
        step(b"", "v Full");
        let shape = step(b"v", "Edit preview");
        assert!(shape.contains("Target net.merge_a.in[0]"));
        assert!(shape.contains("…×32 → …×48"));
        assert!(shape.contains("net.merge_a.weight"));
        step(b"y\x1b", "v Full · y Confirm");
        resize_terminal(&resize_handle, 30, 120);
        step(b"\x1b", "110%  ·  OFFLINE · structure snapshot v1");
        step(b"ey", "v2 · repair-1 · Awaiting recheck");
        step(b"ne\th", "structure snapshot v2");
        let rejection = step(b"\txy", "Rejected · source port unavailable · v2");
        assert!(!rejection.contains("observed tensor flow v2"));
        let rejected = step(b"v", "Edit preview");
        assert!(rejected.contains("Target net.merge_b.in[0]"));
        assert!(rejected.contains("net.adapter.out[9]"));
        assert!(rejected.contains("net.merge_b.weight"));
        assert!(rejected.contains("Rejected · source port unavailable · v2"));
        let committed = step(b"vxy", "v3 · repair-2 · Awaiting recheck");
        assert!(committed.contains("observed tensor flow v3"));
        assert!(committed.contains("net.adapter[0] → net.merge_b[0]"));
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn twin_merge_terminal_commits_and_rejects_at_the_named_input() {
    let directory = tempfile::tempdir().unwrap();
    let (mut terminal, mut child) = open_diagnosis(
        120,
        30,
        &directory.path().join("preferences.json"),
        "en_US.UTF-8",
        "pixels",
        &["--diagnosis-case", "twin-merge"],
    );
    wait_for_terminal_text(&mut terminal, "Open workbench");
    let mut step = |key: u8, expected: &str| {
        terminal.write_all(&[key]).unwrap();
        editor_frame(&mut terminal, expected)
    };
    let first = step(b'\n', "Observed 2×48 · Expected …×32");
    assert!(first.contains("model.merge_a #2"));
    assert!(first.contains("observed tensor flow v1"));
    let shape = step(b'e', "Confirm · Esc cancel");
    assert!(shape.contains("model.encoder.out[0] → net.merge_a.in[0]"));
    assert!(shape.contains("v1 · in[0] …×32 → …×48"));
    let v2 = step(b'y', "v2 · repair-1 · Awaiting recheck");
    assert!(v2.contains("net.merge_a.in[0] · …×32 → …×48"));
    assert!(v2.contains("observed tensor flow v2"));
    assert_eq!(v2.matches("\x1b_Ga=T,").count(), 1);

    step(b'n', "model.merge_b #1");
    let next = step(b'e', "Confirm · x Source");
    let change = "net.merge_b.in[0] · model.encoder.out[0] → net.adapter.out[0]";
    assert!(next.contains(change));
    assert!(next.contains("observed tensor flow v2"));
    let v3 = step(b'y', "v3 · repair-2 · Awaiting recheck");
    assert!(v3.contains(change));
    assert!(v3.contains("observed tensor flow v3"));
    assert_eq!(v3.matches("\x1b_Ga=T,").count(), 1);

    let invalid = "net.merge_b.in[0] · net.adapter.out[0] → net.adapter.out[9]";
    let draft = step(b'x', invalid);
    assert!(draft.contains("observed tensor flow v3"));
    let rejection = step(b'y', "Rejected · source port unavailable · v3");
    assert!(!rejection.contains("v4"));
    assert!(!rejection.contains("\x1b[2J"));
    assert_eq!(rejection.matches("\x1b_Ga=T,").count(), 0);
    step(
        b'x',
        "net.merge_b.in[0] · net.adapter.out[0] → net.adapter.out[0]",
    );
    step(27, "v3 · repair-2 · Awaiting recheck");
    let receipts = step(b'l', "Offline receipts");
    let shape = receipts.find("v2 · repair-1 · net.merge_a.in[0]").unwrap();
    let source = receipts.find("v3 · repair-2 · net.merge_b.in[0]").unwrap();
    assert!(shape < source);
    assert!(!receipts.contains("v4"));
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn wire_candidates_select_a_recorded_connection_from_another_page() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            40,
            24,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "large"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\t/block00.layer00").unwrap();
        checkpoint_frame(&mut terminal, "block00.layer00", &mut pending);
        terminal.write_all(b"\n").unwrap();
        checkpoint_frame(&mut terminal, "block00.layer00", &mut pending);
        terminal.write_all(b"W").unwrap();
        checkpoint_frame(&mut terminal, "Wire candidates", &mut pending);
        terminal.write_all(b"s").unwrap();
        let scope = checkpoint_frame(&mut terminal, "Loaded scope · s local", &mut pending);
        let count = scope
            .split("Wire candidates · ")
            .last()
            .unwrap()
            .split('/')
            .nth(1)
            .unwrap()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<usize>()
            .unwrap();
        assert!(count > 20, "{scope}");
        for _ in 0..8 {
            terminal.write_all(b"\x1b[6~").unwrap();
        }
        let last = checkpoint_frame(
            &mut terminal,
            &format!("Wire candidates · {count}/{count}"),
            &mut pending,
        );
        assert!(last.contains(".out[") && last.contains(".in["), "{last}");
        terminal.write_all(b"\n").unwrap();
        let graph = checkpoint_frame(&mut terminal, "block00", &mut pending);
        assert!(graph.contains("→"), "{graph}");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn overlapping_wire_candidates_can_be_reviewed_and_selected() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\t\x1b[<35;27;8M").unwrap();
        checkpoint_frame(&mut terminal, "model.encoder", &mut pending);
        terminal.write_all(b"W").unwrap();
        let menu = checkpoint_frame(&mut terminal, "Wire candidates", &mut pending);
        assert!(menu.contains("model.encoder.out[0]"));
        assert!(menu.contains("model.encoder.out[1]"));
        assert!(menu.contains("net.merge.in[0]"));
        assert!(menu.contains("net.merge.in[1]"));
        assert!(menu.contains("At pointer · s scope"));
        terminal.write_all(b"s").unwrap();
        checkpoint_frame(&mut terminal, "Loaded scope · s local", &mut pending);
        terminal.write_all(b"s").unwrap();
        checkpoint_frame(&mut terminal, "At pointer · s scope", &mut pending);
        // Dragging to another candidate cancels the press; scrolling must still
        // act on the open list, not the graph behind it.
        terminal
            .write_all(b"\x1b[<0;2;3M\x1b[<32;2;5M\x1b[<0;2;5m\x1b[<65;2;5M")
            .unwrap();
        checkpoint_frame(&mut terminal, "Wire candidates · 2/2", &mut pending);
        terminal.write_all(b"\x1b[<64;2;5M").unwrap();
        checkpoint_frame(&mut terminal, "Wire candidates · 1/2", &mut pending);
        // Focus loss also cancels a held press.
        terminal
            .write_all(b"\x1b[<0;2;3M\x1b[O\x1b[I\x1b[<0;2;3m\x1b[6~")
            .unwrap();
        checkpoint_frame(&mut terminal, "Wire candidates · 2/2", &mut pending);
        terminal.write_all(b"\n").unwrap();
        let chosen = checkpoint_frame(
            &mut terminal,
            "model.encoder[1] → net.merge[0]",
            &mut pending,
        );
        assert!(chosen.contains("model.encoder[1] → net.merge[0]"));
        terminal.write_all(b"\x1b[<35;27;8MW").unwrap();
        checkpoint_frame(&mut terminal, "Wire candidates", &mut pending);
        terminal.write_all(b"\x1b[<0;2;3M\x1b[<0;2;3m").unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[0] → net.merge[1]",
            &mut pending,
        );
        terminal.write_all(b"q").unwrap();
        wait_for_terminal_text(&mut terminal, "\x1b[?1049l");
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn graph_modes_toggle_layout_edit_and_release_mouse_capture() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            100,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\t").unwrap();
        checkpoint_frame(&mut terminal, "L Layout off", &mut pending);
        terminal.write_all(b"L").unwrap();
        let layout = checkpoint_frame(&mut terminal, "L Layout on", &mut pending);
        assert!(layout.contains("[nodes]"), "{layout}");
        terminal.write_all(b"T").unwrap();
        let released = checkpoint_frame(&mut terminal, "T Mouse off", &mut pending);
        assert!(released.contains("\x1b[?1003l"));
        assert!(released.contains("\x1b[?1006l"));
        terminal.write_all(b"T").unwrap();
        let enabled = checkpoint_frame(&mut terminal, "T Mouse on", &mut pending);
        assert!(enabled.contains("\x1b[?1003h"));
        assert!(enabled.contains("\x1b[?1006h"));
        terminal.write_all(b"q").unwrap();
        let restored = wait_for_terminal_text(&mut terminal, "\x1b[?1049l");
        assert!(String::from_utf8_lossy(&restored).contains("\x1b[?1003l"));
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn module_drag_moves_the_terminal_hit_target_and_can_be_reset() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_offline(
            100,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            ["--editor-demo"],
        );
        let mut pending = Vec::new();
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            checkpoint_frame(&mut terminal, expected, &mut pending)
        };
        step(b"", "tensor flow v1");
        step(b"\tL", "L Layout on");
        step(b"\x1b[<0;38;12M\x1b[<32;47;10M", "╭─ encoder.attention");
        step(b"\x1b", "tensor flow v1");
        let moving = step(b"\x1b[<0;38;12M\x1b[<32;47;10M", "╭─ encoder.attention");
        assert_eq!(moving.contains("\x1b_Ga=T,"), mode == "pixels");
        let released = step(b"\x1b[<0;47;10m", "tensor flow v1");
        assert!(!released.contains("tensor flow v2"));
        step(b"\x1b[<35;47;10M", "╭─ encoder.attention");
        step(
            b"\x1b[<0;47;10M\x1b[<0;47;10m\x1b[<0;47;10M\x1b[<0;47;10m",
            "Zoom · 0  |  encoder.attention",
        );
        step(b"\x1b", "tensor flow v1");
        step(b"\x1b[<35;47;10M", "╭─ encoder.attention");
        step(b"r", "tensor flow v1");
        step(b"\x1b[<35;38;12M", "╭─ encoder.attention");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn zoom_refines_the_latest_pixel_view_without_moving_the_selected_wire() {
    let directory = tempfile::tempdir().unwrap();
    let (mut terminal, mut child) = open_diagnosis(
        100,
        30,
        &directory.path().join("preferences.json"),
        "en_US.UTF-8",
        "pixels",
        &["--diagnosis-case", "large"],
    );
    wait_for_terminal_text(&mut terminal, "Open workbench");
    let mut pending = Vec::new();
    let width = |frame: &str| -> u32 {
        let (_, packet) = frame.split_once("\x1b_Ga=T,").unwrap();
        let (header, _) = packet.split_once(';').unwrap();
        header
            .split(',')
            .find_map(|field| field.strip_prefix("s="))
            .unwrap()
            .parse()
            .unwrap()
    };
    terminal.write_all(b"\n").unwrap();
    let before = checkpoint_frame(&mut terminal, "Observed 2×48", &mut pending);
    assert!(before.contains("model.encoder[0] → net.head[0]"));
    let base_width = width(&before);
    terminal
        .write_all(b"\x1b[<64;51;12M\x1b[<64;51;12M\x1b[<64;51;12M")
        .unwrap();
    let zoomed = checkpoint_frame(&mut terminal, "133%  ·", &mut pending);
    assert!(zoomed.contains("\x1b[2;1H\x1b[2K133%  ·"));
    assert!(!zoomed.contains("\x1b[2J"));
    assert!(!zoomed.contains("model.encoder[0] → net.head[0]"));
    let preview_width = width(&zoomed);
    assert!(preview_width < base_width);
    let refined = checkpoint_frame(&mut terminal, &format!(",s={base_width},"), &mut pending);
    assert_eq!(width(&refined), base_width);
    assert!(!refined.contains("\x1b[2J"));
    assert!(!refined.contains("tensor flow v2"));

    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn large_offline_workbench_measures_input_and_retains_each_edit() {
    const ZOOM_IN: &[u8] = b"\x1b[<64;51;12M";
    const ZOOM_OUT: &[u8] = b"\x1b[<65;51;12M";
    let capture_root = std::env::var_os("MODEL_MAVEN_TUI_REPLAY_DIR").map(std::path::PathBuf::from);
    for mode in ["cells", "pixels"] {
        let capture = capture_root.as_ref().map(|root| root.join(mode));
        if let Some(path) = &capture {
            std::fs::create_dir(path).unwrap();
        }
        let mut stream = capture
            .as_ref()
            .map(|path| File::create(path.join("stream.ansi")).unwrap());
        let mut index = capture
            .as_ref()
            .map(|path| File::create(path.join("checkpoints.txt")).unwrap());
        let mut offset = 0;
        let mut save = |name: &str, bytes: &[u8]| {
            if let Some(path) = &capture {
                if !name.is_empty() {
                    std::fs::write(path.join(name), bytes).unwrap();
                }
                if name != "timing.txt" {
                    stream.as_mut().unwrap().write_all(bytes).unwrap();
                    offset += bytes.len();
                    if !name.is_empty() {
                        writeln!(index.as_mut().unwrap(), "{offset} {name}").unwrap();
                    }
                }
            }
        };
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            100,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "large"],
        );
        let welcome = wait_for_terminal_text(&mut terminal, "Open workbench");
        save("01-welcome.ansi", &welcome);
        let first_frame_started = Instant::now();
        terminal.write_all(b"\n").unwrap();
        let failure = editor_frame(&mut terminal, "Observed 2×48 · Expected …×32");
        let first_frame_time = first_frame_started.elapsed();
        assert!(failure.contains("model.head #2"));
        save("02-failure.ansi", failure.as_bytes());

        let mut refinement = Vec::new();
        let settle_limit = Instant::now() + Duration::from_secs(2);
        let idle_span = loop {
            let bytes = refinement.len();
            let start = Instant::now();
            assert!(read_frame(&mut terminal, &mut refinement));
            if refinement.len() == bytes && start.elapsed() >= Duration::from_millis(70) {
                break start.elapsed();
            }
            assert!(
                Instant::now() < settle_limit,
                "stationary view did not settle"
            );
        };
        save("", &refinement);
        let panel_started = Instant::now();
        terminal.write_all(b"?").unwrap();
        let panel = editor_frame(&mut terminal, "Actions in this pane");
        let panel_time = panel_started.elapsed();
        assert!(!panel.contains("\x1b_Ga=T,"));
        assert!(!panel.contains("\x1b[2J"));
        save("", panel.as_bytes());
        terminal.write_all(b"?").unwrap();
        let details = editor_frame(&mut terminal, "Observed 2×48 · Expected …×32");
        save("", details.as_bytes());

        terminal.write_all(b"a").unwrap();
        let ratios = editor_frame(&mut terminal, "Last-axis alignment");
        assert!(ratios.contains("2×48 (observed) ↔ …×32 (declared)"));
        save("03-axes-mismatch.ansi", ratios.as_bytes());
        terminal.write_all(b"a\tu").unwrap();
        let upstream = editor_frame(&mut terminal, "1–1/1");
        assert!(upstream.contains("model.encoder[0] → net.head[0]"));
        assert!(upstream.contains("model.head #2"));
        save("04-source.ansi", upstream.as_bytes());
        terminal.write_all(b"\x1b").unwrap();
        let returned = editor_frame(&mut terminal, "1–3/3");
        assert!(returned.contains("model.encoder[0] → net.head[0]"));
        assert!(returned.contains("model.head #2"));
        save("05-return.ansi", returned.as_bytes());
        terminal.write_all(b"b").unwrap();
        let overview = editor_frame(&mut terminal, "/34");
        assert!(overview.contains("structure snapshot v1"));
        save("06-groups.ansi", overview.as_bytes());
        terminal.write_all(b"\x1b[B\nh").unwrap();
        let group = editor_frame(&mut terminal, "generated tensor flow v1");
        assert!(group.contains("/32"));
        save("07-group-flow.ansi", group.as_bytes());

        let mut samples = Vec::new();
        let mut intervals = Vec::new();
        let mut last_finished = None;
        let mut peak_bytes = 0;
        for step in 0..100 {
            let (event, zoom) = if step % 2 == 0 {
                (ZOOM_IN, "110%  ·")
            } else {
                (ZOOM_OUT, "100%  ·")
            };
            let started = Instant::now();
            terminal.write_all(event).unwrap();
            let response = editor_frame(&mut terminal, zoom);
            let finished = Instant::now();
            samples.push(finished.duration_since(started));
            if let Some(last) = last_finished.replace(finished) {
                intervals.push(finished.duration_since(last));
            }
            peak_bytes = peak_bytes.max(response.len());
            save(
                if step == 0 { "08-zoom.ansi" } else { "" },
                response.as_bytes(),
            );
        }
        samples.sort_unstable();
        intervals.sort_unstable();
        let timing = format!(
            "100×30 {mode}, 1024 generated layers: first frame {first_frame_time:?}; 100 events: PTY response p50 {:?}, p95 {:?}, max {:?}, frame interval p50 {:?}, p95 {:?}, max {:?}, max {peak_bytes} bytes",
            samples[49], samples[94], samples[99], intervals[49], intervals[94], intervals[98],
        );
        eprintln!("{timing}");

        let mut burst = b"\x1b[<0;51;12M".to_vec();
        for column in 52..=63 {
            burst.extend_from_slice(format!("\x1b[<32;{column};12M").as_bytes());
        }
        burst.extend_from_slice(b"\x1b[<0;63;12m");
        burst.extend_from_slice(&ZOOM_IN.repeat(12));
        burst.extend_from_slice(b"\teyneyxy");
        let burst_started = Instant::now();
        terminal.write_all(&burst).unwrap();
        let mut pending = Vec::new();
        let v2 = checkpoint_frame(
            &mut terminal,
            "v2 · repair-1 · Awaiting recheck",
            &mut pending,
        );
        let v2_time = burst_started.elapsed();
        let second = checkpoint_frame(&mut terminal, "Confirm · x Source", &mut pending);
        assert!(v2.contains("Confirm · Esc cancel"));
        assert!(second.contains("model.merge #1"));
        let first_commit = v2
            .split("\x1b[?2026l")
            .find(|frame| frame.contains("v2 · repair-1 · Awaiting recheck"))
            .unwrap();
        assert!(first_commit.contains("net.head.in[0] · …×32 → …×48"));
        if mode == "pixels" {
            assert_eq!(first_commit.matches("\x1b_Ga=T,").count(), 1);
        }
        save("09-first-commit.ansi", v2.as_bytes());
        save("10-second-preview.ansi", second.as_bytes());
        let committed = checkpoint_frame(
            &mut terminal,
            "v3 · repair-2 · Awaiting recheck",
            &mut pending,
        );
        assert!(!committed.contains("v2 · repair-1 · Awaiting recheck"));
        if mode == "pixels" {
            assert_eq!(committed.matches("\x1b_Ga=T,").count(), 1);
        }
        let v3_time = burst_started.elapsed();
        save("11-second-commit.ansi", committed.as_bytes());
        let invalid = checkpoint_frame(&mut terminal, "net.adapter.out[9]", &mut pending);
        let rejected = checkpoint_frame(
            &mut terminal,
            "Rejected · source port unavailable · v3",
            &mut pending,
        );
        assert!(!rejected.contains("observed tensor flow v4"));
        assert!(!rejected.contains("\x1b_Ga=T,"));
        let editor_timing = format!(
            "{mode}: idle {idle_span:?}, 0 bytes; details only {panel_time:?}, {} bytes; burst to v2 {v2_time:?}, v3 {v3_time:?}; rejection {} bytes",
            panel.len(),
            rejected.len(),
        );
        eprintln!("{editor_timing}");
        save(
            "timing.txt",
            format!("{timing}\n{editor_timing}\n").as_bytes(),
        );
        save("12-invalid-port.ansi", invalid.as_bytes());
        save("13-rejected.ansi", rejected.as_bytes());
        save("", &pending);
        terminal.write_all(b"a").unwrap();
        let missing_axis = editor_frame(&mut terminal, "? ↔ …×48 (declared)");
        assert!(missing_axis.contains("Candidate ?"));
        save("14-axes-unknown.ansi", missing_axis.as_bytes());
        terminal.write_all(b"a\x1b1c").unwrap();
        let before = editor_frame(&mut terminal, "Before · • Target observed");
        assert!(before.contains("model.encoder.out[0] → net.merge.in[0]"));
        save("15-before-recheck.ansi", before.as_bytes());
        terminal.write_all(b"c").unwrap();
        let after = editor_frame(&mut terminal, "After · ✓ Sample passed");
        assert!(after.contains("net.adapter.out[0] → net.merge.in[0]"));
        save("16-after-recheck.ansi", after.as_bytes());
        terminal.write_all(b"v").unwrap();
        let full = editor_frame(&mut terminal, "Recheck evidence");
        assert!(full.contains("Target net.merge.in[0]"));
        assert!(full.contains("model.encoder.out[0] → net.merge.in[0]"));
        assert!(full.contains("net.adapter.out[0] → net.merge.in[0]"));
        assert!(!full.contains("observed tensor flow"));
        save("17-recheck-evidence.ansi", full.as_bytes());
        terminal.write_all(b"\x1b").unwrap();
        let compared = editor_frame(&mut terminal, "› After · ✓ Sample passed");
        assert!(compared.contains("observed tensor flow v3"));
        save("18-return-to-comparison.ansi", compared.as_bytes());
        terminal.write_all(b"\x1b").unwrap();
        let receipt = editor_frame(&mut terminal, "Passed · 2×48");
        assert!(!receipt.contains("Target observed"));
        save("19-return-to-receipt.ansi", receipt.as_bytes());
        terminal.write_all(b"a").unwrap();
        let aligned = editor_frame(&mut terminal, "Last-axis alignment");
        assert!(aligned.contains("2×48 (observed) ↔ …×48 (declared)"));
        save("20-axes-equal.ansi", aligned.as_bytes());
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

fn checkpoint_frame(terminal: &mut File, expected: &str, pending: &mut Vec<u8>) -> String {
    const FRAME_END: &[u8] = b"\x1b[?2026l";
    let expected = expected.as_bytes();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(end) = pending
            .windows(expected.len())
            .position(|bytes| bytes == expected)
            .map(|start| start + expected.len())
            .and_then(|start| {
                pending[start..]
                    .windows(FRAME_END.len())
                    .position(|bytes| bytes == FRAME_END)
                    .map(|offset| start + offset + FRAME_END.len())
            })
        {
            let frame: Vec<_> = pending.drain(..end).collect();
            return String::from_utf8_lossy(&frame).into_owned();
        }
        assert!(
            Instant::now() < deadline,
            "Missing checkpoint {:?}: {}",
            String::from_utf8_lossy(expected),
            String::from_utf8_lossy(pending)
        );
        assert!(read_frame(terminal, pending));
    }
}

fn editor_frame(terminal: &mut File, expected: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut output = Vec::new();
    loop {
        assert!(read_frame(terminal, &mut output));
        let data = String::from_utf8_lossy(&output);
        if data
            .split_once(expected)
            .is_some_and(|(_, tail)| tail.contains("\x1b[?2026l"))
        {
            return data.into_owned();
        }
        assert!(
            Instant::now() < deadline,
            "Missing editor frame {expected:?}: {}",
            data.chars().take(400).collect::<String>()
        );
    }
}

#[test]
fn fullscreen_probe_options_are_editable_and_return_to_the_batch_form() {
    let project = tempfile::tempdir().unwrap();
    let backend = project.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{"current_revision":"v1"}}' ;;
 'model probe') printf '%s\n' "$*" >> calls.log
   printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"success","data":{"batch_id":"batch-a","batch_identity_stable":true}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 80,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--lang", "en", "--cli", backend.to_str().unwrap()])
        .current_dir(project.path())
        .env("TERM", "xterm-256color")
        .env("NO_COLOR", "1")
        .env("MODEL_MAVEN_REDUCE_MOTION", "1")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            project.path().join("preferences.json"),
        )
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Probe a batch");
    terminal.write_all(b"2").unwrap();
    wait_for_terminal_text(&mut terminal, "Model factory");
    terminal.write_all(b":model-params\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Model factory parameters");
    terminal.write_all(b"\x15{\"width\":32}\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Model factory");
    terminal.write_all(b"fixture:model\n").unwrap();
    wait_for_terminal_text(&mut terminal, ":options advanced");
    terminal.write_all(b":options\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Batch binding:");
    terminal.write_all(b"single\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Dataset fingerprint");
    terminal.write_all(b"dataset-v1\n").unwrap();
    wait_for_terminal_text(&mut terminal, ":options advanced");
    terminal.write_all(b":batch-params\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Batch factory parameters");
    terminal.write_all(b"\x15{\"size\":48}\n").unwrap();
    wait_for_terminal_text(&mut terminal, ":options advanced");
    terminal.write_all(b":mode\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Forward mode:");
    terminal.write_all(b"\x15train\n").unwrap();
    wait_for_terminal_text(&mut terminal, ":options advanced");
    terminal.write_all(b"fixture:batch\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Completed");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "Probe a batch");
    resize_terminal(&terminal, 24, 40);
    terminal.write_all(b"2").unwrap();
    wait_for_terminal_text(&mut terminal, "Model factory");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Batch factory");
    terminal.write_all(b"\x15:batch-params\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Batch factory parameters");
    terminal.write_all(b"\x1b").unwrap();
    wait_for_terminal_text(&mut terminal, "Batch factory");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Batch ID matches previous probe report");
    terminal.write_all(b"bq").unwrap();
    assert!(child.wait().unwrap().success());
    let calls = std::fs::read_to_string(project.path().join("calls.log")).unwrap();
    for line in calls.lines() {
        assert!(line.contains("--batch fixture:batch"), "{calls}");
        assert!(line.contains("--batch-binding single"), "{calls}");
        assert!(line.contains("--batch-fingerprint dataset-v1"), "{calls}");
        assert!(line.contains("--batch-param size=48"), "{calls}");
        assert!(line.contains("--model-param width=32"), "{calls}");
        assert!(line.contains("--training"), "{calls}");
    }
    assert_eq!(calls.lines().count(), 2, "{calls}");
}

#[test]
fn graph_toolbar_supports_mouse_zoom_and_reset_in_both_renderers() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &[],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n\t").unwrap();
        wait_for_terminal_text(&mut terminal, "[-][+][<][>][^][v][R][E]");
        // SGR coordinates are one-based; the zoom and reset buttons are on row one.
        terminal.write_all(b"\x1b[<0;61;1M\x1b[<0;61;1m").unwrap();
        wait_for_terminal_text(&mut terminal, "110%");
        terminal.write_all(b"\x1b[<0;76;1M\x1b[<0;76;1m").unwrap();
        wait_for_terminal_text(&mut terminal, "100%");
        // Focus loss cancels the pending press; a later release must not zoom in.
        terminal
            .write_all(b"\x1b[<0;61;1M\x1b[O\x1b[I\x1b[<0;61;1m\x1b[<0;58;1M\x1b[<0;58;1m")
            .unwrap();
        wait_for_terminal_text(&mut terminal, "91%");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn wide_graph_toolbar_mouse_buttons_switch_visual_presets() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            100,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n\t").unwrap();
        wait_for_terminal_text(&mut terminal, "[-][+][<][>][^][v][R][E]{1}[2][3][4]");

        // The toolbar is right-aligned on row one. Click preset 4, 3, 1, then 2.
        terminal.write_all(b"\x1b[<0;99;1M\x1b[<0;99;1m").unwrap();
        let exploded_x2 = wait_for_terminal_text(&mut terminal, "exploded slices 2/2");
        assert!(
            String::from_utf8_lossy(&exploded_x2).contains("CAM 045°/+35°"),
            "{}",
            String::from_utf8_lossy(&exploded_x2)
        );
        terminal.write_all(b"\x1b[<0;96;1M\x1b[<0;96;1m").unwrap();
        let exploded_x1 = wait_for_terminal_text(&mut terminal, "exploded slices 1/2");
        assert!(
            String::from_utf8_lossy(&exploded_x1).contains("CAM 045°/+35°"),
            "{}",
            String::from_utf8_lossy(&exploded_x1)
        );
        terminal.write_all(b"\x1b[<0;90;1M\x1b[<0;90;1m").unwrap();
        wait_for_terminal_text(&mut terminal, "CAM");
        terminal.write_all(b"\x1b[<0;93;1M\x1b[<0;93;1m").unwrap();
        wait_for_terminal_text(&mut terminal, "2D");

        terminal.write_all(b"q").unwrap();
        wait_for_terminal_text(&mut terminal, "\x1b[?1049l");
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn graph_mode_toolbar_uses_the_exact_92_column_boundary() {
    let directory = tempfile::tempdir().unwrap();
    let (mut narrow, mut narrow_child) = open_diagnosis(
        91,
        30,
        &directory.path().join("narrow-preferences.json"),
        "en_US.UTF-8",
        "cells",
        &["--diagnosis-case", "crossed-ports"],
    );
    wait_for_terminal_text(&mut narrow, "Open workbench");
    narrow.write_all(b"\n\t").unwrap();
    let compact = wait_for_terminal_text(&mut narrow, "[-][+][<][>][^][v][R][E]");
    assert!(!String::from_utf8_lossy(&compact).contains("[1][2][3][4]"));
    narrow.write_all(b"q").unwrap();
    assert!(narrow_child.wait().unwrap().success());

    let (mut wide, mut wide_child) = open_diagnosis(
        92,
        30,
        &directory.path().join("wide-preferences.json"),
        "en_US.UTF-8",
        "cells",
        &["--diagnosis-case", "crossed-ports"],
    );
    wait_for_terminal_text(&mut wide, "Open workbench");
    wide.write_all(b"\n\t").unwrap();
    wait_for_terminal_text(&mut wide, "[-][+][<][>][^][v][R][E]{1}[2][3][4]");
    // At 92 columns, the rightmost preset is the final terminal cell.
    wide.write_all(b"\x1b[<0;92;1M\x1b[<0;92;1m").unwrap();
    wait_for_terminal_text(&mut wide, "exploded slices 2/2");
    wide.write_all(b"q").unwrap();
    wait_for_terminal_text(&mut wide, "\x1b[?1049l");
    assert!(wide_child.wait().unwrap().success());
}

#[test]
fn recorded_call_nesting_can_be_filtered_opened_and_restored_at_40_columns() {
    let project = tempfile::tempdir().unwrap();
    let backend = project.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model","type":"Net"},{"path":"model.block","type":"Sequential"},{"path":"model.block.0","type":"Linear"}]}}' ;;
 'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"success","data":{"calls":[{"invocation_id":"f:1","forward_id":"f","parent_invocation_id":null,"alias_paths":["model"],"invocation_index":1},{"invocation_id":"f:2","forward_id":"f","parent_invocation_id":"f:1","alias_paths":["model.block"],"invocation_index":1},{"invocation_id":"f:3","forward_id":"f","parent_invocation_id":"f:2","alias_paths":["model.block.0"],"invocation_index":1}]}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 40,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--lang", "en", "--cli", backend.to_str().unwrap()])
        .current_dir(project.path())
        .env("TERM", "xterm-256color")
        .env_remove("COLUMNS")
        .env("NO_COLOR", "1")
        .env("MODEL_MAVEN_REDUCE_MOTION", "1")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            project.path().join("preferences.json"),
        )
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Inspect model");
    terminal.write_all(b"1").unwrap();
    wait_for_terminal_text(&mut terminal, "Model factory");
    terminal.write_all(b"fixture:model\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Completed");
    terminal.write_all(b"b2\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Batch factory");
    terminal.write_all(b"fixture:batch\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Completed");
    terminal.write_all(b"bk").unwrap();
    let list = wait_for_terminal_text(&mut terminal, "3.   ↳ model.block.0");
    let list = String::from_utf8_lossy(&list);
    assert!(list.contains("> 1. model"));
    assert!(
        list.contains("execution nesting")
            && list.contains("Nesting does not establish tensor flow"),
        "{list}"
    );
    assert!(
        list.contains("2. ↳ model.block") && list.contains("3.   ↳ model.block.0"),
        "{list}"
    );
    terminal.write_all(b"/").unwrap();
    wait_for_terminal_text(&mut terminal, "Filter recorded calls");
    let long_filter = format!("{}末端", "模型".repeat(30));
    terminal.write_all(long_filter.as_bytes()).unwrap();
    wait_for_terminal_text(&mut terminal, "末端▏");
    terminal.write_all(b"\x01").unwrap();
    let at_start = wait_for_terminal_text(&mut terminal, "▏模型");
    assert!(String::from_utf8_lossy(&at_start).contains('›'));
    size.ws_col = 80;
    assert_eq!(
        unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ, &size) },
        0
    );
    wait_for_terminal_text(&mut terminal, &format!("▏{}", "模型".repeat(12)));
    size.ws_col = 40;
    assert_eq!(
        unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ, &size) },
        0
    );
    wait_for_terminal_text(&mut terminal, "▏模型模型模型模型模型");
    terminal.write_all(b"\x15").unwrap();
    terminal.write_all(b"model.block.0\n").unwrap();
    wait_for_terminal_text(&mut terminal, "> 3.   ↳ model.block.0");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "model.block.0");
    terminal.write_all(b"l").unwrap();
    wait_for_terminal_text(&mut terminal, "> 3.   ↳ model.block.0");
    terminal.write_all(b"bq").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn fullscreen_workbench_key_editor_checks_conflicts_at_40_columns() {
    let project = tempfile::tempdir().unwrap();
    let backend = project.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model inspect') printf '%s\n' "$*" >> calls.log
   printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 40,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--lang", "en", "--cli", backend.to_str().unwrap()])
        .current_dir(project.path())
        .env("TERM", "xterm-256color")
        .env("NO_COLOR", "1")
        .env("MODEL_MAVEN_REDUCE_MOTION", "1")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            project.path().join("preferences.json"),
        )
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\n").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. Probe a batch");
    terminal.write_all(b"\x1b[Z").unwrap();
    wait_for_terminal_text(&mut terminal, "> 1. Inspect model");
    terminal.write_all(b"\t").unwrap();
    wait_for_terminal_text(&mut terminal, "> 2. Probe a batch");
    terminal.write_all(b"s").unwrap();
    wait_for_terminal_text(&mut terminal, "Workbench keys");
    terminal.write_all(b"3").unwrap();
    wait_for_terminal_text(&mut terminal, "Workbench menu keys");
    terminal.write_all(b"1").unwrap();
    wait_for_terminal_text(&mut terminal, "New key");
    terminal.write_all(b"\x152\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Already assigned 2");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "Workbench menu keys");
    terminal.write_all(b"1").unwrap();
    wait_for_terminal_text(&mut terminal, "New key");
    terminal.write_all(b"\x15v\n").unwrap();
    wait_for_terminal_text(&mut terminal, "v. Inspect model structure");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "Workbench keys");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "v. Inspect model structure");
    terminal.write_all(b"v").unwrap();
    wait_for_terminal_text(&mut terminal, "Model factory");
    terminal.write_all(b"fixture:q\x1b[Zmodel\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Completed");
    terminal.write_all(b"bq").unwrap();
    assert!(child.wait().unwrap().success());
    assert!(
        std::fs::read_to_string(project.path().join("calls.log"))
            .unwrap()
            .contains("--model fixture:qmodel")
    );
}

#[test]
fn reverse_tab_switches_graph_focus_and_stays_inside_modal_contexts() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            80,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &[],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        let mut step = |keys: &[u8], expected: &str| {
            terminal.write_all(keys).unwrap();
            checkpoint_frame(&mut terminal, expected, &mut pending)
        };
        step(b"\n", "model.head #2");
        step(b"\x1b[Z", "Tab Details · ? Controls");
        let help = step(b"?", "Graph controls");
        assert!(help.contains("Shift+Tab"), "{help}");
        step(b"?\x1b[Z", "Offline sample · Failed · Details");
        step(b"e", "Confirm · Esc cancel");
        step(b"\x1b[Z", "Tab Details · ? Controls");
        step(b"\x1b[Z", "Offline sample · Failed · Details");
        step(b"v", "Edit preview");
        step(b"\x1b[Z\x1b", "Confirm · Esc cancel");
        let back = step(b"\x1b", "Offline sample · Failed · Details");
        assert!(!back.contains("Awaiting recheck"));
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn fullscreen_variant_selection_scopes_real_workbench_commands() {
    let project = tempfile::tempdir().unwrap();
    let backend = project.path().join("m-model");
    std::fs::write(&backend, r##"#!/bin/sh
printf '%s\n' "$*" >> calls.log
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'variant list') printf '%s\n' '{"protocol_version":1,"operation":"variant_list","status":"success","data":{"variants":[{"name":"branch","readable":true,"revision":"rev-123456789"}]}}' ;;
 'variant show') printf '%s\n' '{"protocol_version":1,"operation":"variant_show","status":"success","data":{"name":"branch","variant":{"revision":"rev-changed"}}}' ;;
 'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear"}]}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
    std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (mut master, mut slave) = (-1, -1);
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 40,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        },
        0
    );
    let mut terminal = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
        .args(["--lang", "en", "--cli", backend.to_str().unwrap()])
        .current_dir(project.path())
        .env("TERM", "xterm-256color")
        .env("NO_COLOR", "1")
        .env("MODEL_MAVEN_REDUCE_MOTION", "1")
        .env(
            "MODEL_MAVEN_TUI_PREFERENCES",
            project.path().join("preferences.json"),
        )
        .stdin(Stdio::from(slave.try_clone().unwrap()))
        .stdout(Stdio::from(slave.try_clone().unwrap()))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();
    wait_for_terminal_text(&mut terminal, "Open workbench");
    terminal.write_all(b"\ns").unwrap();
    wait_for_terminal_text(&mut terminal, "4. Variant");
    terminal.write_all(b"4").unwrap();
    let listing = wait_for_terminal_text(&mut terminal, "1. branch");
    assert!(String::from_utf8_lossy(&listing).contains("Listing is not validation or a pin"));
    terminal.write_all(b"1").unwrap();
    wait_for_terminal_text(&mut terminal, "4. Variant  branch");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "Variant  branch");
    terminal.write_all(b"1").unwrap();
    wait_for_terminal_text(&mut terminal, "Model factory");
    terminal.write_all(b"fixture:model\n").unwrap();
    wait_for_terminal_text(&mut terminal, "Completed");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "Variant  branch");
    terminal.write_all(b"6").unwrap();
    wait_for_terminal_text(&mut terminal, "Variant evidence is stale");
    terminal.write_all(b"b").unwrap();
    wait_for_terminal_text(&mut terminal, "Variant  branch");
    terminal.write_all(b"q").unwrap();
    assert!(child.wait().unwrap().success());
    let calls = std::fs::read_to_string(project.path().join("calls.log")).unwrap();
    assert!(
        calls
            .lines()
            .any(|line| line.contains("model inspect") && line.contains("--variant branch")),
        "{calls}"
    );
    assert!(calls.contains("variant show branch"), "{calls}");
}

#[test]
fn welcome_enter_focuses_recommended_initialization_in_both_sizes() {
    for width in [40, 100] {
        let project = tempfile::tempdir().unwrap();
        let backend = project.path().join("m-model");
        std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"error","diagnostics":[{"code":"PROJECT_NOT_INITIALIZED","message":"missing"}]}' ; exit 2 ;;
 *) exit 1 ;;
esac
"##).unwrap();
        std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
        let (mut master, mut slave) = (-1, -1);
        let mut size = libc::winsize {
            ws_row: 24,
            ws_col: width,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut size,
                )
            },
            0
        );
        let mut terminal = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args(["--lang", "en", "--cli", backend.to_str().unwrap()])
            .current_dir(project.path())
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .env("MODEL_MAVEN_REDUCE_MOTION", "1")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                project.path().join("preferences.json"),
            )
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave))
            .spawn()
            .unwrap();
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n").unwrap();
        let menu = wait_for_terminal_text(&mut terminal, "Set up project");
        let menu = String::from_utf8_lossy(&menu);
        assert!(menu.contains("Next"), "{menu}");
        terminal.write_all(b"\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Model factory");
        terminal.write_all(b"\x1b").unwrap();
        wait_for_terminal_text(&mut terminal, "Set up project");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
        assert!(!project.path().join(".model-maven").exists());
    }
}

#[test]
fn matrix_distinguishes_declared_absence_from_unknown_in_both_renderers() {
    for (mode, width) in [("cells", 40), ("pixels", 40), ("cells", 80), ("pixels", 80)] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            width,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\tM").unwrap();
        checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
        terminal.write_all(b"\x1b[<0;11;6M\x1b[<0;11;6m").unwrap();
        let absent = checkpoint_frame(&mut terminal, "Declared no wire", &mut pending);
        assert!(absent.contains("[-]"), "{absent}");
        assert!(absent.contains("? unknown"), "{absent}");
        terminal.write_all(b"M").unwrap();
        let graph = checkpoint_frame(&mut terminal, "Port pair · Declared no wire", &mut pending);
        assert!(
            graph.contains("Output ") && graph.contains("Input "),
            "{graph}"
        );
        assert!(graph.contains("u/d ports · v evidence"), "{graph}");
        terminal.write_all(b"v").unwrap();
        let evidence = checkpoint_frame(&mut terminal, "Port pair evidence", &mut pending);
        assert!(
            evidence.contains("net.adapter.out[0]") && evidence.contains("net.merge.in[0]"),
            "{evidence}"
        );
        assert!(evidence.contains("Declared no wire"), "{evidence}");
        resize_terminal(&terminal, 14, 34);
        let resized = checkpoint_frame(&mut terminal, "Port pair evidence", &mut pending);
        assert!(resized.contains("Declared no wire"), "{resized}");
        resize_terminal(&terminal, 30, width);
        checkpoint_frame(&mut terminal, "Port pair evidence", &mut pending);
        terminal.write_all(b"v").unwrap();
        checkpoint_frame(&mut terminal, "Port pair · Declared no wire", &mut pending);
        terminal.write_all(b"u").unwrap();
        checkpoint_frame(&mut terminal, "net.adapter.out[0]", &mut pending);
        terminal.write_all(b"b").unwrap();
        checkpoint_frame(&mut terminal, "Port pair · Declared no wire", &mut pending);
        terminal.write_all(b"d").unwrap();
        checkpoint_frame(&mut terminal, "net.merge.in[0]", &mut pending);
        terminal.write_all(b"b").unwrap();
        checkpoint_frame(&mut terminal, "Port pair · Declared no wire", &mut pending);
        terminal.write_all(b"M").unwrap();
        checkpoint_frame(&mut terminal, "Declared no wire", &mut pending);
        terminal.write_all(b"\x1b[C").unwrap();
        let unknown = checkpoint_frame(&mut terminal, "No recorded wire", &mut pending);
        assert!(unknown.contains("[?]"), "{unknown}");
        terminal.write_all(b"\nM").unwrap();
        let graph = checkpoint_frame(
            &mut terminal,
            "Port pair · Connection unknown",
            &mut pending,
        );
        assert!(
            graph.contains("Output ") && graph.contains("Input "),
            "{graph}"
        );
        terminal.write_all(b"M").unwrap();
        checkpoint_frame(&mut terminal, "No recorded wire", &mut pending);
        terminal.write_all(b"c").unwrap();
        let grouped = checkpoint_frame(&mut terminal, "no wire 1", &mut pending);
        assert!(grouped.contains("? 1"), "{grouped}");
        terminal.write_all(b"M").unwrap();
        checkpoint_frame(&mut terminal, "Tab Details", &mut pending);
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn keyboard_matrix_selects_a_wire_from_another_graph_page() {
    let mut modules = vec![serde_json::json!({"path":"model", "type":"Sequential"})];
    modules.extend(
        (0..20).map(|i| serde_json::json!({"path": format!("model.{i}"), "type":"Linear"})),
    );
    let report = serde_json::json!({"protocol_version":1, "operation":"inspect", "status":"success", "data":{"modules":modules}}).to_string();
    for mode in ["cells", "pixels"] {
        let project = tempfile::tempdir().unwrap();
        let backend = project.path().join("m-model");
        let script = format!(
            "#!/bin/sh\ncase \"$3 $4\" in\n 'config history') printf '%s\\n' '{{\"protocol_version\":1,\"operation\":\"config_history\",\"status\":\"success\"}}' ;;\n 'model inspect') printf '%s\\n' '{report}' ;;\n *) exit 1 ;;\nesac\n"
        );
        std::fs::write(&backend, script).unwrap();
        std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
        let (mut master, mut slave) = (-1, -1);
        let mut size = libc::winsize {
            ws_row: 24,
            ws_col: 40,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut size,
                )
            },
            0
        );
        let mut terminal = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args(["--lang", "en", "--cli", backend.to_str().unwrap()])
            .current_dir(project.path())
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .env("MODEL_MAVEN_REDUCE_MOTION", "1")
            .env("MODEL_MAVEN_GRAPHICS", mode)
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                project.path().join("preferences.json"),
            )
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave))
            .spawn()
            .unwrap();
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n1fixture:model\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Completed");
        terminal.write_all(b"b6").unwrap();
        let mut pending = Vec::new();
        checkpoint_frame(&mut terminal, "model.0", &mut pending);
        terminal.write_all(b"M").unwrap();
        checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
        for _ in 0..15 {
            terminal.write_all(b"\x1b[B").unwrap();
        }
        for _ in 0..16 {
            terminal.write_all(b"\x1b[C").unwrap();
        }
        let target = checkpoint_frame(&mut terminal, "model.15[0] → model.16[0]", &mut pending);
        assert!(target.contains("[=]") || target.contains("[o]"), "{target}");
        terminal.write_all(b"\nM").unwrap();
        let graph = checkpoint_frame(&mut terminal, "Sequential · declared flow", &mut pending);
        assert!(graph.contains("model.15[0] → model.16[0]"), "{graph}");
        terminal.write_all(b"q").unwrap();
        wait_for_terminal_text(&mut terminal, "Model-Maven Workbench");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn workbench_command_search_opens_available_actions_and_explains_disabled_ones() {
    for width in [40, 100] {
        let project = tempfile::tempdir().unwrap();
        let backend = project.path().join("m-model");
        std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model inspect') printf '%s\n' "$*" >> calls.log
  printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{}}' ;;
 *) exit 1 ;;
esac
"##).unwrap();
        std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
        let (mut master, mut slave) = (-1, -1);
        let mut size = libc::winsize {
            ws_row: 24,
            ws_col: width,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut size,
                )
            },
            0
        );
        let mut terminal = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args(["--lang", "en", "--cli", backend.to_str().unwrap()])
            .current_dir(project.path())
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .env("MODEL_MAVEN_REDUCE_MOTION", "1")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                project.path().join("preferences.json"),
            )
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave))
            .spawn()
            .unwrap();
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n/").unwrap();
        wait_for_terminal_text(&mut terminal, "Search commands");
        terminal.write_all(b"preview").unwrap();
        wait_for_terminal_text(&mut terminal, "! 7. Preview Linear input size");
        terminal.write_all(b"\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Command results");
        terminal.write_all(b"\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Command unavailable");
        terminal.write_all(b"b").unwrap();
        wait_for_terminal_text(&mut terminal, "Command results");
        terminal.write_all(b"b").unwrap();
        wait_for_terminal_text(&mut terminal, "/ Search commands");
        terminal.write_all(b"/").unwrap();
        wait_for_terminal_text(&mut terminal, "Search commands");
        terminal.write_all(b"inspect").unwrap();
        wait_for_terminal_text(&mut terminal, "1. Inspect model structure");
        terminal.write_all(b"\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Command results");
        terminal.write_all(b"\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Model factory");
        let model = format!("pkg.{}:build", "网络".repeat(30));
        terminal.write_all(model.as_bytes()).unwrap();
        wait_for_terminal_text(&mut terminal, ":build▏");
        terminal.write_all(b"\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Completed");
        terminal.write_all(b"bq").unwrap();
        assert!(child.wait().unwrap().success());
        let calls = std::fs::read_to_string(project.path().join("calls.log")).unwrap();
        assert!(
            calls.contains(&format!("model inspect --model {model}")),
            "{calls}"
        );
    }
}

#[test]
fn unconnected_search_keeps_port_identity_across_graph_history() {
    for (mode, width) in [("cells", 40), ("pixels", 40), ("cells", 80), ("pixels", 80)] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            width,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "crossed-ports"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\t/adapter.out[0]").unwrap();
        checkpoint_frame(&mut terminal, "net.adapter.out[0]", &mut pending);
        terminal.write_all(b"\n").unwrap();
        let port = checkpoint_frame(&mut terminal, "No recorded wire", &mut pending);
        assert!(port.contains("net.adapter.out[0]"), "{port}");
        assert!(port.contains("status unknown"), "{port}");
        terminal.write_all(b"M").unwrap();
        let matrix = checkpoint_frame(&mut terminal, "Port matrix", &mut pending);
        assert!(matrix.contains("net.adapter[0] →"), "{matrix}");
        terminal.write_all(b"M").unwrap();
        let port = checkpoint_frame(&mut terminal, "No recorded wire", &mut pending);
        assert!(port.contains("net.adapter.out[0]"), "{port}");
        terminal.write_all(b"/merge.in[1]").unwrap();
        checkpoint_frame(&mut terminal, "net.merge.in[1]", &mut pending);
        terminal.write_all(b"\n").unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[0] → net.merge[1]",
            &mut pending,
        );
        terminal.write_all(b"b").unwrap();
        let restored = checkpoint_frame(&mut terminal, "No recorded wire", &mut pending);
        assert!(restored.contains("net.adapter.out[0]"), "{restored}");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn source_search_reveals_an_unloaded_port_and_its_chosen_consumer() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_offline(
            120,
            36,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            ["--drifting-demo"],
        );
        let mut pending = Vec::new();
        checkpoint_frame(
            &mut terminal,
            "SOURCE · static Drifting graph",
            &mut pending,
        );
        terminal
            .write_all(b"/block_07.attention.reshape_qkv.out[0]")
            .unwrap();
        let results = checkpoint_frame(
            &mut terminal,
            "/ block_07.attention.reshape_qkv.out[0]",
            &mut pending,
        );
        assert!(results.contains("[source]"), "{results}");
        terminal.write_all(b"\x1b[B").unwrap();
        let selected = checkpoint_frame(&mut terminal, "> [source]", &mut pending);
        assert!(selected.contains("k.in[0]"), "{selected}");
        terminal.write_all(b"\n").unwrap();
        let graph = checkpoint_frame(
            &mut terminal,
            "SOURCE · static Drifting graph",
            &mut pending,
        );
        assert!(graph.contains("reshape_qkv[0]"), "{graph}");
        assert!(graph.contains("k[0]"), "{graph}");
        terminal.write_all(b"b").unwrap();
        checkpoint_frame(
            &mut terminal,
            "SOURCE · static Drifting graph",
            &mut pending,
        );
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn ambiguous_inspection_disables_preview_with_recheck_guidance() {
    for width in [40, 100] {
        let project = tempfile::tempdir().unwrap();
        let backend = project.path().join("m-model");
        std::fs::write(&backend, r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.first","type":"Linear"},{"path":"model.second","type":"Linear"}]}}' ;;
 'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","data":{"failure_location":{"last_observed_module_call":{"alias_paths":["model.first","model.second"],"module_type":"Linear"}}}}' ; exit 4 ;;
 *) exit 1 ;;
esac
"##).unwrap();
        std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
        let (mut master, mut slave) = (-1, -1);
        let mut size = libc::winsize {
            ws_row: 24,
            ws_col: width,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut size,
                )
            },
            0
        );
        let mut terminal = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        let mut child = Command::new(env!("CARGO_BIN_EXE_m-model-tui"))
            .args(["--lang", "en", "--cli", backend.to_str().unwrap()])
            .current_dir(project.path())
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .env("MODEL_MAVEN_REDUCE_MOTION", "1")
            .env(
                "MODEL_MAVEN_TUI_PREFERENCES",
                project.path().join("preferences.json"),
            )
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave))
            .spawn()
            .unwrap();
        wait_for_terminal_text(&mut terminal, "Open workbench");
        terminal.write_all(b"\n1fixture:model\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Completed");
        terminal.write_all(b"b").unwrap();
        wait_for_terminal_text(&mut terminal, "Model-Maven Workbench");
        terminal.write_all(b"2\nfixture:batch\n").unwrap();
        wait_for_terminal_text(&mut terminal, "Preview unavailable:");
        terminal.write_all(b"b").unwrap();
        let menu = wait_for_terminal_text(&mut terminal, "↑/↓ Move");
        let menu = String::from_utf8_lossy(&menu);
        assert!(!menu.contains("7. Preview Linear"), "{menu}");
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn command_palette_opens_failure_and_repair_views_with_context_guards() {
    for (mode, width) in [("cells", 40), ("pixels", 40), ("cells", 80), ("pixels", 80)] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            width,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "mismatch"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\t/repair").unwrap();
        checkpoint_frame(&mut terminal, "need preview", &mut pending);
        terminal.write_all(b"\n").unwrap();
        checkpoint_frame(&mut terminal, "need preview", &mut pending);
        terminal.write_all(b"\x1b").unwrap();
        checkpoint_frame(&mut terminal, "Tab Details", &mut pending);
        terminal.write_all(b"/failure").unwrap();
        let failure_name = if width < 58 {
            "Failure path"
        } else {
            "Open failure / call path"
        };
        let failure_search = checkpoint_frame(&mut terminal, failure_name, &mut pending);
        assert!(failure_search.contains("[F]"), "{failure_search}");
        terminal.write_all(b"\n").unwrap();
        checkpoint_frame(&mut terminal, "Failure path", &mut pending);
        terminal.write_all(b"F").unwrap();
        checkpoint_frame(&mut terminal, "Tab Details", &mut pending);
        terminal.write_all(b"\te").unwrap();
        checkpoint_frame(&mut terminal, "y Confirm", &mut pending);
        terminal.write_all(b"/repair").unwrap();
        let repair_name = if width < 58 {
            "Repair diff"
        } else {
            "Open repair difference"
        };
        let available = checkpoint_frame(&mut terminal, repair_name, &mut pending);
        assert!(available.contains("[D]"), "{available}");
        assert!(!available.contains("need preview"), "{available}");
        terminal.write_all(b"\n").unwrap();
        let diff = checkpoint_frame(&mut terminal, "Repair difference", &mut pending);
        assert!(diff.contains("Preview · not committed"), "{diff}");
        terminal.write_all(b"D").unwrap();
        checkpoint_frame(&mut terminal, "y Confirm", &mut pending);
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn twin_merge_wire_candidates_support_keyboard_and_mouse_selection() {
    for mode in ["cells", "pixels"] {
        let directory = tempfile::tempdir().unwrap();
        let (mut terminal, mut child) = open_diagnosis(
            100,
            30,
            &directory.path().join("preferences.json"),
            "en_US.UTF-8",
            mode,
            &["--diagnosis-case", "twin-merge"],
        );
        wait_for_terminal_text(&mut terminal, "Open workbench");
        let mut pending = Vec::new();
        terminal.write_all(b"\n\tW").unwrap();
        let menu = checkpoint_frame(&mut terminal, "Wire candidates", &mut pending);
        assert!(menu.contains("net.merge_a.in[0]"), "{menu}");
        assert!(menu.contains("net.merge_b.in[1]"), "{menu}");
        terminal.write_all(b"\x1b[6~\n").unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.residual[0] → net.merge_b[1]",
            &mut pending,
        );
        terminal.write_all(b"W").unwrap();
        checkpoint_frame(&mut terminal, "Wire candidates", &mut pending);
        terminal.write_all(b"\x1b[<0;2;3M\x1b[<0;2;3m").unwrap();
        checkpoint_frame(
            &mut terminal,
            "model.encoder[0] → net.merge_a[0]",
            &mut pending,
        );
        terminal.write_all(b"q").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

#[test]
fn mode_mismatch_is_visible_and_disables_preview_at_common_widths() {
    for (
        language,
        welcome,
        completed,
        mode_prompt,
        batch_prompt,
        warning,
        modes,
        footer,
        preview,
    ) in [
        (
            "en",
            "Open workbench",
            "Completed",
            "Forward mode:",
            "Batch factory",
            "Mode mismatch · preview unavailable",
            "Requested train · reported eval",
            "↑/↓ Move",
            "Preview Linear input size",
        ),
        (
            "zh-cn",
            "进入工作台",
            "操作完成",
            "前向模式：",
            "批次工厂",
            "模式不一致 · 无法预览",
            "请求 train · 报告 eval",
            "↑/↓ 移动",
            "预览 Linear 输入维度",
        ),
        (
            "zh-tw",
            "進入工作臺",
            "操作完成",
            "前向模式：",
            "批次工廠",
            "模式不一致 · 無法預覽",
            "請求 train · 報告 eval",
            "↑/↓ 移動",
            "預覽 Linear 輸入維度",
        ),
    ] {
        for (width, height) in [(40, 24), (80, 24), (100, 30)] {
            for background in ["none", "light", "dark"] {
                let project = tempfile::tempdir().unwrap();
                let backend = project.path().join("m-model");
                std::fs::write(&backend,r##"#!/bin/sh
case "$3 $4" in
 'config history') printf '%s\n' '{"protocol_version":1,"operation":"config_history","status":"success","data":{}}' ;;
 'model inspect') printf '%s\n' '{"protocol_version":1,"operation":"inspect","status":"success","data":{"modules":[{"path":"model.head","type":"Linear"}]}}' ;;
 'model probe') printf '%s\n' '{"protocol_version":1,"operation":"probe","status":"error","data":{"observed_mode":"eval","failure_location":{"last_observed_module_call":{"alias_paths":["model.head"],"module_type":"Linear"}}}}' ; exit 4 ;;
 *) exit 1 ;;
esac
"##).unwrap();
                std::fs::set_permissions(&backend, std::fs::Permissions::from_mode(0o755)).unwrap();
                let (mut master, mut slave) = (-1, -1);
                let mut size = libc::winsize {
                    ws_row: height,
                    ws_col: width,
                    ws_xpixel: 0,
                    ws_ypixel: 0,
                };
                assert_eq!(
                    unsafe {
                        libc::openpty(
                            &mut master,
                            &mut slave,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            &mut size,
                        )
                    },
                    0
                );
                let mut terminal = unsafe { File::from_raw_fd(master) };
                let slave = unsafe { File::from_raw_fd(slave) };
                let mut command = Command::new(env!("CARGO_BIN_EXE_m-model-tui"));
                command
                    .args(["--lang", language, "--cli", backend.to_str().unwrap()])
                    .current_dir(project.path())
                    .env("TERM", "xterm-256color")
                    .env_remove("NO_COLOR")
                    .env_remove("COLORFGBG")
                    .env_remove("MODEL_MAVEN_BACKGROUND");
                if background == "none" {
                    command.env("NO_COLOR", "1");
                } else {
                    command.env("MODEL_MAVEN_BACKGROUND", background);
                }
                let mut child = command
                    .env("MODEL_MAVEN_REDUCE_MOTION", "1")
                    .env(
                        "MODEL_MAVEN_TUI_PREFERENCES",
                        project.path().join("preferences.json"),
                    )
                    .stdin(Stdio::from(slave.try_clone().unwrap()))
                    .stdout(Stdio::from(slave.try_clone().unwrap()))
                    .stderr(Stdio::from(slave))
                    .spawn()
                    .unwrap();
                wait_for_terminal_text(&mut terminal, welcome);
                terminal.write_all(b"\n1fixture:model\n").unwrap();
                wait_for_terminal_text(&mut terminal, completed);
                terminal.write_all(b"b").unwrap();
                wait_for_terminal_text(&mut terminal, footer);
                terminal.write_all(b"2\n:mode\n").unwrap();
                wait_for_terminal_text(&mut terminal, mode_prompt);
                terminal.write_all(b"\x15train\n").unwrap();
                wait_for_terminal_text(&mut terminal, batch_prompt);
                terminal.write_all(b"fixture:batch\n").unwrap();
                let mut report = wait_for_terminal_text(&mut terminal, warning);
                let deadline = Instant::now() + Duration::from_secs(5);
                while !String::from_utf8_lossy(&report).contains(modes) {
                    assert!(
                        Instant::now() < deadline,
                        "mode comparison missing from report"
                    );
                    assert!(read_frame(&mut terminal, &mut report));
                }
                let report = String::from_utf8_lossy(&report);
                assert!(
                    report.contains(modes),
                    "{language} {width}x{height} {background}: {report}"
                );
                terminal.write_all(b"b").unwrap();
                let menu = wait_for_terminal_text(&mut terminal, footer);
                assert!(!String::from_utf8_lossy(&menu).contains(preview));
                terminal.write_all(b"q").unwrap();
                assert!(child.wait().unwrap().success());
            }
        }
    }
}
