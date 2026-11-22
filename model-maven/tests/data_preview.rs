#[path = "../src/adapters/previews.rs"]
mod previews;

use serde_json::Value;
#[cfg(unix)]
use std::{
    io::{Read, Write},
    os::fd::FromRawFd,
    process::{ExitStatus, Stdio},
};
use std::{path::Path, process::Command};
use tempfile::tempdir;

use previews::FilePreviewStore;

fn is_active(project: &Path, request_id: &str, generation: u64) -> bool {
    let state = FilePreviewStore::current(project).unwrap();
    state["current"]["request_id"] == request_id
        && state["current"]["generation"] == generation
        && matches!(
            state["current"]["status"].as_str(),
            Some("pending" | "running")
        )
}

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn run_python(program: &str, args: &[&str]) -> Value {
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let package = project.join("python");
    let output = Command::new(python())
        .arg("-c")
        .arg(program)
        .args(args)
        .env("PYTHONPATH", package)
        .output()
        .expect("start Python preview fixture");
    assert!(
        output.status.success(),
        "Python fixture failed: {}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("fixture emits JSON")
}

#[cfg(unix)]
fn run_preview_pty(project: &Path, input: &[u8]) -> (ExitStatus, String) {
    run_preview_pty_with_prefix(project, input, &[])
}

#[cfg(unix)]
fn run_preview_pty_with_prefix(
    project: &Path,
    input: &[u8],
    prefix: &[&str],
) -> (ExitStatus, String) {
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
    assert_eq!(opened, 0, "openpty failed");
    let slave = unsafe { std::fs::File::from_raw_fd(slave_fd) };
    let stdout = slave.try_clone().unwrap();
    let stderr = slave.try_clone().unwrap();
    let python = python();
    let mut child = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(prefix)
        .args(["--python", &python, "data", "preview"])
        .env_remove("DISPLAY")
        .current_dir(project)
        .stdin(Stdio::from(slave))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap();
    let mut terminal = unsafe { std::fs::File::from_raw_fd(master_fd) };
    terminal.write_all(input).unwrap();
    let mut transcript = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let mut descriptor = libc::pollfd {
            fd: master_fd,
            events: libc::POLLIN | libc::POLLHUP,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut descriptor, 1, 60_000) };
        if ready <= 0 {
            let _ = child.kill();
            panic!(
                "interactive data preview timed out:\n{}",
                String::from_utf8_lossy(&transcript)
            );
        }
        match terminal.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => transcript.extend_from_slice(&buffer[..count]),
            Err(error) if error.raw_os_error() == Some(libc::EIO) => break,
            Err(error) => panic!("cannot read preview terminal: {error}"),
        }
    }
    let status = child.wait().unwrap();
    (status, String::from_utf8_lossy(&transcript).into_owned())
}

fn preview_signature(seed: u64, sample_id: &str) -> String {
    let result = run_python(
        r#"
import json, sys
from model_maven_worker.data.preview import preview_signature
identity = {"dataset_fingerprint":"data-v1", "sample_ids":[sys.argv[1]], "seed":int(sys.argv[2]), "transforms":[{"type":"Resize", "size":8}]}
print(json.dumps({"signature":preview_signature(identity)}))
"#,
        &[sample_id, &seed.to_string()],
    );
    result["signature"].as_str().unwrap().to_owned()
}

#[test]
fn cli_preview_samples_project_batch_and_publishes_real_artifacts() {
    let python = python();
    let available = Command::new(&python)
        .args(["-c", "import torch, PIL"])
        .status()
        .is_ok_and(|status| status.success());
    if !available {
        eprintln!("skipped CLI preview integration: Python needs PyTorch and Pillow");
        return;
    }
    let project = tempfile::Builder::new()
        .prefix("model maven preview ")
        .tempdir()
        .unwrap();
    std::fs::write(
        project.path().join("fixture.py"),
        r#"
import torch

def build_batch(count=3):
    return torch.arange(count * 3 * 16 * 12, dtype=torch.float32).reshape(count, 3, 16, 12) / (count * 3 * 16 * 12)

def failing_batch():
    raise RuntimeError("intentional preview failure")
"#,
    )
    .unwrap();
    std::fs::create_dir_all(project.path().join(".model-maven")).unwrap();
    std::fs::write(
        project.path().join(".model-maven/environment.lock.json"),
        r#"{"schema_version":1}"#,
    )
    .unwrap();

    for args in [
        vec!["init", "--model", "fixture:build_model"],
        vec![
            "augment", "add", "Resize", "--id", "resize", "--target", "args.0", "--param", "size=8",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args(["--json", "--python", &python])
            .args(args)
            .current_dir(project.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let invoke = |fingerprint: &str| {
        Command::new(env!("CARGO_BIN_EXE_m-model"))
            .args([
                "--json",
                "--python",
                &python,
                "data",
                "preview",
                "--batch",
                "fixture:build_batch",
                "--fingerprint",
                fingerprint,
                "--samples",
                "2",
            ])
            .current_dir(project.path())
            .output()
            .unwrap()
    };
    let output = invoke("fixture-v1");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: Value = serde_json::from_slice(stdout.trim().as_bytes()).unwrap_or_else(|error| {
        panic!(
            "expected one JSON report, got {stdout:?}: {error}; status={:?}; stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["operation"], "preview");
    assert_eq!(report["data"]["status"], "succeeded");
    assert_eq!(report["data"]["sample_count"], 2);
    assert_eq!(
        report["data"]["transform_observations"][0]["steps"][0]["type"], "Resize",
        "{report:#}"
    );
    assert_eq!(
        report["data"]["transform_observations"][0]["output_port"]["shape"],
        serde_json::json!([3, 8, 8])
    );
    assert_eq!(report["data"]["rendered_image_count"], 2);
    assert_eq!(report["data"]["scope"]["stable_sample_ids"], true);
    let artifacts = report["data"]["artifacts"].as_array().unwrap();
    assert_eq!(artifacts.len(), 2);
    let png = artifacts
        .iter()
        .find(|item| item["media_type"] == "image/png")
        .unwrap();
    let html = artifacts
        .iter()
        .find(|item| {
            item["media_type"]
                .as_str()
                .unwrap()
                .starts_with("text/html")
        })
        .unwrap();
    assert!(project.path().join(png["uri"].as_str().unwrap()).is_file());
    let html_path = project.path().join(html["uri"].as_str().unwrap());
    let html_contents = std::fs::read_to_string(html_path).unwrap();
    assert!(html_contents.contains("data:image/png;base64,"));
    assert!(report["data"]["delivery"]["opened"] == false);

    let cached_output = invoke("fixture-v1");
    let cached: Value = serde_json::from_slice(&cached_output.stdout).unwrap();
    assert!(cached_output.status.success(), "{cached:#}");
    assert_eq!(cached["data"]["cached"], true);
    assert_eq!(
        cached["data"]["input_signature"],
        report["data"]["input_signature"]
    );

    let changed_output = invoke("fixture-v2");
    let changed: Value = serde_json::from_slice(&changed_output.stdout).unwrap();
    assert!(changed_output.status.success(), "{changed:#}");
    assert_eq!(changed["data"]["cached"], false);
    assert_ne!(
        changed["data"]["input_signature"],
        report["data"]["input_signature"]
    );

    let failed = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python,
            "data",
            "preview",
            "--batch",
            "fixture:failing_batch",
            "--fingerprint",
            "fixture-failed-v1",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!failed.status.success());
    let failed_report: Value = serde_json::from_slice(&failed.stdout).unwrap();
    assert_eq!(failed_report["status"], "error", "{failed_report:#}");
    assert_eq!(failed_report["data"]["previous_result_stale"], true);
    assert_eq!(
        failed_report["data"]["previous_preview"]["request_id"],
        changed["data"]["request_id"]
    );
    assert!(
        failed_report["data"]["previous_preview"]["artifacts"]
            .as_array()
            .is_some_and(|artifacts| !artifacts.is_empty())
    );

    let failed_human = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python,
            "data",
            "preview",
            "--batch",
            "fixture:failing_batch",
            "--fingerprint",
            "fixture-failed-human-v1",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!failed_human.status.success());
    let failed_human_text = String::from_utf8_lossy(&failed_human.stdout);
    assert!(
        failed_human_text.contains("预览失败 · 本次请求没有生成可确认的新结果。"),
        "{failed_human_text}"
    );
    assert!(
        failed_human_text.contains("上一版预览已过期；不能代表当前输入。"),
        "{failed_human_text}"
    );
    assert!(
        failed_human_text.contains("上一版预览（已过期）"),
        "{failed_human_text}"
    );
    assert!(
        failed_human_text.contains(changed["data"]["artifacts"][0]["uri"].as_str().unwrap()),
        "{failed_human_text}"
    );
    assert!(
        failed_human_text.contains("内容不代表这次设置"),
        "{failed_human_text}"
    );
    assert!(
        !failed_human_text.contains("当前预览 ·"),
        "{failed_human_text}"
    );
    assert!(
        !failed_human_text.contains("已发布产物："),
        "{failed_human_text}"
    );

    let ssh_output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python,
            "data",
            "preview",
            "--batch",
            "fixture:build_batch",
            "--fingerprint",
            "fixture-v1",
            "--samples",
            "2",
            "--delivery",
            "ssh",
            "--ssh-host",
            "preview.example",
            "--ssh-user",
            "developer",
            "--ssh-port",
            "2222",
            "--destination",
            "./downloaded preview.html",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    let ssh_report: Value = serde_json::from_slice(&ssh_output.stdout).unwrap();
    assert!(ssh_output.status.success(), "{ssh_report:#}");
    let delivery = &ssh_report["data"]["delivery"];
    assert_eq!(delivery["transferred"], false);
    assert_eq!(delivery["opened"], false);
    assert!(
        delivery["remote_path"]
            .as_str()
            .unwrap()
            .starts_with(project.path().to_str().unwrap())
    );
    assert!(
        delivery["scp_command"]
            .as_str()
            .unwrap()
            .contains("developer@preview.example:")
    );
    assert!(
        delivery["scp_command"]
            .as_str()
            .unwrap()
            .contains("-P 2222")
    );
    assert!(
        delivery["scp_command"]
            .as_str()
            .unwrap()
            .contains("developer@preview.example:'")
    );
    assert!(
        delivery["scp_command"]
            .as_str()
            .unwrap()
            .contains("'./downloaded preview.html'")
    );

    let before_missing_ssh = FilePreviewStore::current(project.path()).unwrap();
    let ssh_missing = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python,
            "data",
            "preview",
            "--batch",
            "fixture:build_batch",
            "--fingerprint",
            "fixture-ssh-missing",
            "--samples",
            "2",
            "--delivery",
            "ssh",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(!ssh_missing.status.success());
    let ssh_missing_report: Value = serde_json::from_slice(&ssh_missing.stdout).unwrap();
    assert_eq!(ssh_missing_report["status"], "error");
    assert_eq!(
        ssh_missing_report["diagnostics"][0]["code"],
        "CLI_ARGUMENT_INVALID"
    );
    assert!(
        ssh_missing_report["diagnostics"][0]["message"]
            .as_str()
            .unwrap()
            .contains("--ssh-host")
    );
    assert!(ssh_missing.stderr.is_empty());
    let after_missing_ssh = FilePreviewStore::current(project.path()).unwrap();
    assert_eq!(
        before_missing_ssh["current"]["request_id"],
        after_missing_ssh["current"]["request_id"]
    );
    assert_eq!(
        before_missing_ssh["current"]["generation"],
        after_missing_ssh["current"]["generation"]
    );

    let human_output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--python",
            &python,
            "data",
            "preview",
            "--batch",
            "fixture:build_batch",
            "--fingerprint",
            "fixture-v1",
            "--samples",
            "2",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    assert!(
        human_output.status.success(),
        "{}",
        String::from_utf8_lossy(&human_output.stderr)
    );
    let human = String::from_utf8_lossy(&human_output.stdout);
    assert!(human.contains("数据预览"), "{human}");
    assert!(human.contains("当前预览 · 复用已验证结果"), "{human}");
    assert!(human.contains("2 个实际样本"), "{human}");
    assert!(human.contains("HTML 预览："), "{human}");
    assert!(human.contains("尚未自动打开"), "{human}");

    #[cfg(unix)]
    {
        let (non_interactive_status, non_interactive_output) =
            run_preview_pty_with_prefix(project.path(), b"", &["--no-interactive"]);
        assert!(
            !non_interactive_status.success(),
            "{non_interactive_output}"
        );
        assert!(
            !non_interactive_output.contains("预览来源："),
            "--no-interactive must not prompt on a TTY: {non_interactive_output}"
        );

        let (status, interactive) = run_preview_pty(
            project.path(),
            b"batch\nfixture:build_batch\ninvalid\ncount=2\n\nvalidation\n2\nfixture-interactive-v1\n\nsummary\nlocal\n",
        );
        assert!(
            status.success(),
            "interactive preview exited with {status}:\n{interactive}"
        );
        assert!(interactive.contains("格式应为 NAME=JSON"), "{interactive}");
        assert!(
            interactive.contains("预览计划：批次工厂 · 切分 validation"),
            "{interactive}"
        );
        assert!(interactive.contains("已设置 1 个工厂参数"), "{interactive}");
        assert!(interactive.contains("2 个实际样本"), "{interactive}");
        assert!(
            interactive.contains("当前预览 · 本次请求已发布"),
            "{interactive}"
        );

        let before_retry = FilePreviewStore::current(project.path()).unwrap();
        let (retry_status, retried) = run_preview_pty(
            project.path(),
            b"batch\nfixture:failing_batch\n\n-\n\n\nfixture-retry-v1\n\nauto\nlocal\ninvalid\n1\nbatch\nfixture:build_batch\n\n\n\n\n\nauto\nlocal\n",
        );
        assert!(retry_status.success(), "{retried}");
        assert!(retried.contains("intentional preview failure"), "{retried}");
        assert!(retried.contains("切分名称不能为空"), "{retried}");
        assert!(retried.contains("请输入 1、2 或 back"), "{retried}");
        assert!(retried.contains("1 修改设置并重试"), "{retried}");
        assert!(retried.contains("当前预览 · 本次请求已发布"), "{retried}");
        let after_retry = FilePreviewStore::current(project.path()).unwrap();
        assert_eq!(after_retry["current"]["status"], "succeeded");
        assert_ne!(
            after_retry["current"]["request_id"],
            before_retry["current"]["request_id"]
        );

        let (ssh_status, ssh_preview) = run_preview_pty(
            project.path(),
            b"batch\nfixture:build_batch\n\n\n2\nfixture-ssh-interactive\n\nsummary\nssh\npreview.example\ndeveloper\n2244\n./downloaded preview.html\n",
        );
        assert!(ssh_status.success(), "{ssh_preview}");
        assert!(ssh_preview.contains("远端预览文件："), "{ssh_preview}");
        assert!(
            ssh_preview.contains("scp -P 2244 developer@preview.example:"),
            "{ssh_preview}"
        );
        assert!(
            ssh_preview.contains("scp -P 2244 developer@preview.example:'"),
            "{ssh_preview}"
        );
        assert!(ssh_preview.contains("model maven preview"), "{ssh_preview}");
        assert!(
            ssh_preview.contains("'./downloaded preview.html'"),
            "{ssh_preview}"
        );
        assert!(ssh_preview.contains("不会自动连接或传输"), "{ssh_preview}");
        assert!(
            ssh_preview.contains("尚未自动打开") || !ssh_preview.contains("已自动打开"),
            "{ssh_preview}"
        );

        let before_cancel = FilePreviewStore::current(project.path()).unwrap();
        let (cancel_status, cancelled) =
            run_preview_pty(project.path(), b"batch\nfixture:build_batch\n\nback\n");
        assert!(!cancel_status.success(), "{cancelled}");
        assert!(cancelled.contains("已取消数据预览设置"), "{cancelled}");
        let after_cancel = FilePreviewStore::current(project.path()).unwrap();
        assert_eq!(
            before_cancel["current"]["request_id"],
            after_cancel["current"]["request_id"]
        );
        assert_eq!(
            before_cancel["current"]["generation"],
            after_cancel["current"]["generation"]
        );
    }
}

#[test]
fn cli_dataset_preview_accepts_dataset_probe_feature_and_reports_observed_ids() {
    let python = python();
    let available = Command::new(&python)
        .args(["-c", "import torch, PIL"])
        .status()
        .is_ok_and(|status| status.success());
    if !available {
        eprintln!("skipped Dataset preview integration: Python needs PyTorch and Pillow");
        return;
    }
    let project = tempdir().unwrap();
    std::fs::write(
        project.path().join("fixture.py"),
        r#"
import torch

class TinyDataset:
    def __len__(self):
        return 4
    def __getitem__(self, index):
        return torch.full((3, 16, 12), float(index) / 4), index
    def sample_id(self, index):
        return f"sample-{index}"

def build_dataset():
    return TinyDataset()
"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python,
            "data",
            "preview",
            "--dataset",
            "fixture:build_dataset",
            "--fingerprint",
            "dataset-v1",
            "--samples",
            "2",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: Value = serde_json::from_slice(stdout.trim().as_bytes()).unwrap_or_else(|error| {
        panic!(
            "expected one JSON report, got {stdout:?}: {error}; status={:?}; stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["data"]["status"], "succeeded");
    assert_eq!(report["data"]["scope"]["source_kind"], "dataset");
    assert_eq!(report["data"]["sample_count"], 2);
    assert_eq!(report["data"]["scope"]["stable_sample_ids"], true);
    assert!(report["data"]["scope"]["data_observation"].is_object());
    assert!(
        report["data"]["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["media_type"] == "image/png" })
    );
}

#[test]
fn cli_custom_renderer_is_versioned_and_artifact_scoped() {
    let python = python();
    let available = Command::new(&python)
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success());
    if !available {
        eprintln!("skipped custom renderer integration: Python needs PyTorch");
        return;
    }
    let project = tempdir().unwrap();
    std::fs::write(
        project.path().join("fixture.py"),
        r#"
import torch

def build_batch():
    return torch.ones((2, 4))

def render_samples(samples, output_directory, request):
    path = output_directory / "custom.html"
    path.write_text("<html><body>custom preview</body></html>", encoding="utf-8")
    return {
        "renderer_version": request["renderer_version"],
        "artifacts": [{"kind": "preview_document", "path": path.name, "media_type": "text/html"}],
    }
"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args([
            "--json",
            "--python",
            &python,
            "data",
            "preview",
            "--batch",
            "fixture:build_batch",
            "--fingerprint",
            "custom-data-v1",
            "--samples",
            "2",
            "--renderer",
            "custom",
            "--renderer-entrypoint",
            "fixture:render_samples",
            "--renderer-version",
            "fixture-v1",
        ])
        .current_dir(project.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: Value = serde_json::from_slice(stdout.trim().as_bytes()).unwrap_or_else(|error| {
        panic!(
            "expected one JSON report, got {stdout:?}: {error}; status={:?}; stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["data"]["artifacts"][0]["kind"], "preview_document");
    assert_eq!(report["data"]["artifacts"][0]["media_type"], "text/html");
    let path = project
        .path()
        .join(report["data"]["artifacts"][0]["uri"].as_str().unwrap());
    assert_eq!(
        std::fs::read_to_string(path).unwrap(),
        "<html><body>custom preview</body></html>"
    );
}

fn render(directory: &Path, request_id: &str, generation: u64, signature: &str, seed: u64) {
    run_python(
        r#"
import json, sys
from pathlib import Path
from PIL import Image
from model_maven_worker.data.preview import preview_signature, render_preview
seed = int(sys.argv[4])
identity = {"dataset_fingerprint":"data-v1", "sample_ids":["sample-" + str(seed)], "seed":seed, "transforms":[{"type":"Resize", "size":8}]}
request = {"request_id":sys.argv[1], "generation":int(sys.argv[2]), "base_signature":sys.argv[3], "input_signature":preview_signature(identity), "identity":identity, "scope":{"stable_sample_ids":True}}
samples = [
  {"sample_id":"sample-" + str(seed), "label":"flower", "image":Image.new("RGB", (9, 7), (80, 120, 30))},
  {"label":"flower", "image":Image.new("RGB", (5, 11), (140, 80, 20))},
]
manifest = render_preview(request, samples, Path(sys.argv[5]))
print(json.dumps(manifest, sort_keys=True))
"#,
        &[
            request_id,
            &generation.to_string(),
            signature,
            &seed.to_string(),
            directory.to_str().unwrap(),
        ],
    );
}

#[test]
fn real_image_preview_writes_png_self_contained_html_and_redacted_manifest() {
    let project = tempdir().unwrap();
    let output = run_python(
        r#"
import json, sys
from pathlib import Path
from PIL import Image
from model_maven_worker.data.preview import preview_signature, render_preview

root = Path(sys.argv[1]); identity = {"dataset_fingerprint":"data-123", "sample_ids":["private-person-17"], "seed":7, "weights":"weight-hash", "transforms":[{"type":"Resize", "size":[8,9]}]}
signature = preview_signature(identity)
request = {"request_id":"render-check", "generation":3, "input_signature":signature, "identity":identity, "scope":{"split":"train", "count":1}}
manifest = render_preview(request, [{"sample_id":"private-person-17", "label":"rose", "image":Image.new("RGB", (7,5), (200,30,80))}], root)
png = Image.open(root / "preview.png")
html = (root / "preview.html").read_text()
complete = (root / "complete.json").read_text()
print(json.dumps({
  "artifacts":manifest["artifacts"], "sample_count":manifest["sample_count"],
  "png_format":png.format, "png_size":list(png.size),
  "html_embeds_png":"data:image/png;base64," in html,
  "html_has_title":"Model-Maven data preview" in html,
  "manifest_has_private_id":"private-person-17" in complete,
  "signature":signature,
}, sort_keys=True))
"#,
        &[project.path().to_str().unwrap()],
    );
    assert_eq!(output["sample_count"], 1);
    assert_eq!(output["png_format"], "PNG");
    assert_eq!(output["png_size"], serde_json::json!([240, 282]));
    assert_eq!(output["html_embeds_png"], true);
    assert_eq!(output["html_has_title"], true);
    assert_eq!(output["manifest_has_private_id"], false);
    assert_eq!(output["artifacts"].as_array().unwrap().len(), 2);
    assert_eq!(
        FilePreviewStore::begin(project.path(), output["signature"].as_str().unwrap()).unwrap()["status"],
        "pending"
    );
}

#[test]
fn latest_generation_wins_and_cache_failure_and_cancel_preserve_last_success() {
    let project = tempdir().unwrap();
    let signature_a = preview_signature(1, "sample-1");
    let signature_b = preview_signature(2, "sample-2");
    let signature_c = preview_signature(3, "sample-3");
    let request_a = FilePreviewStore::begin(project.path(), &signature_a).unwrap();
    let id_a = request_a["request_id"].as_str().unwrap();
    let generation_a = request_a["generation"].as_u64().unwrap();
    assert!(FilePreviewStore::mark_running(project.path(), id_a, generation_a).unwrap());
    let request_b = FilePreviewStore::begin(project.path(), &signature_b).unwrap();
    let id_b = request_b["request_id"].as_str().unwrap();
    let generation_b = request_b["generation"].as_u64().unwrap();
    assert!(!is_active(project.path(), id_a, generation_a));
    assert!(is_active(project.path(), id_b, generation_b));
    render(
        Path::new(request_a["request_directory"].as_str().unwrap()),
        id_a,
        generation_a,
        &signature_a,
        1,
    );
    render(
        Path::new(request_b["request_directory"].as_str().unwrap()),
        id_b,
        generation_b,
        &signature_b,
        2,
    );
    let published_b =
        FilePreviewStore::complete(project.path(), id_b, generation_b, &signature_b).unwrap();
    let late_a =
        FilePreviewStore::complete(project.path(), id_a, generation_a, &signature_a).unwrap();
    assert_eq!(published_b["published"], true);
    assert_eq!(late_a["published"], false);
    assert_eq!(late_a["status"], "superseded");
    assert_eq!(
        FilePreviewStore::current(project.path()).unwrap()["publication"]["request_id"],
        id_b
    );

    let cached = FilePreviewStore::begin(project.path(), &signature_b).unwrap();
    assert_eq!(cached["cached"], true);
    assert_eq!(cached["publication"]["request_id"], id_b);

    let request_c = FilePreviewStore::begin(project.path(), &signature_c).unwrap();
    let id_c = request_c["request_id"].as_str().unwrap();
    let generation_c = request_c["generation"].as_u64().unwrap();
    render(
        Path::new(request_c["request_directory"].as_str().unwrap()),
        id_c,
        generation_c,
        &signature_c,
        3,
    );
    let run_marker = project.path().join(".model-maven/runs/training-marker");
    std::fs::create_dir_all(run_marker.parent().unwrap()).unwrap();
    std::fs::write(&run_marker, b"training remains owned by run service").unwrap();
    FilePreviewStore::cancel(project.path(), id_c, generation_c).unwrap();
    let late_cancelled =
        FilePreviewStore::complete(project.path(), id_c, generation_c, &signature_c).unwrap();
    let current = FilePreviewStore::current(project.path()).unwrap();
    assert_eq!(late_cancelled["published"], false);
    assert_eq!(late_cancelled["status"], "cancelled");
    assert_eq!(current["current"]["status"], "cancelled");
    assert_eq!(current["last_success"]["request_id"], id_b);
    assert_eq!(current["last_success_stale"], true);
    assert_eq!(
        std::fs::read(run_marker).unwrap(),
        b"training remains owned by run service"
    );

    let request_d = FilePreviewStore::begin(project.path(), &signature_a).unwrap();
    let id_d = request_d["request_id"].as_str().unwrap();
    let generation_d = request_d["generation"].as_u64().unwrap();
    FilePreviewStore::fail(project.path(), id_d, generation_d, "renderer failed").unwrap();
    let after_failure = FilePreviewStore::current(project.path()).unwrap();
    assert_eq!(after_failure["current"]["status"], "failed");
    assert_eq!(after_failure["last_success"]["request_id"], id_b);

    let signature_e = preview_signature(5, "sample-5");
    let request_e = FilePreviewStore::begin(project.path(), &signature_e).unwrap();
    let id_e = request_e["request_id"].as_str().unwrap();
    let generation_e = request_e["generation"].as_u64().unwrap();
    let directory_e = Path::new(request_e["request_directory"].as_str().unwrap());
    render(directory_e, id_e, generation_e, &signature_e, 5);
    std::fs::write(
        directory_e.join("preview.png"),
        b"tampered after completion marker",
    )
    .unwrap();
    let integrity_error =
        FilePreviewStore::complete(project.path(), id_e, generation_e, &signature_e).unwrap_err();
    assert!(integrity_error.contains("完整性"));
    assert_eq!(
        FilePreviewStore::current(project.path()).unwrap()["last_success"]["request_id"],
        id_b
    );
}

#[test]
fn summary_renderer_avoids_payload_values_and_custom_renderer_cannot_escape_request_directory() {
    let output = run_python(
        r#"
import json, tempfile
from pathlib import Path
from model_maven_worker.data.preview import preview_signature, render_preview
from model_maven_worker.protocol import UserError

identity = {"dataset_fingerprint":"other-data", "seed":10, "renderer":"custom"}
signature = preview_signature(identity, "project.custom", "fixture-v1")
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    summary_request = {"request_id":"summary-1", "generation":1, "identity":{"dataset_fingerprint":"other-data", "seed":10}, "renderer":"builtin.summary"}
    summary = render_preview(summary_request, [{"value":{"patient_name":"do-not-render", "scores":[1,2,3]}}], root / "summary")
    content = (root / "summary" / "preview.html").read_text()
    traversal = None
    def bad_renderer(**kwargs):
        return {"artifacts":[{"kind":"leak", "path":"../outside.html", "media_type":"text/html"}]}
    try:
        custom_request = {"request_id":"summary-2", "generation":2, "identity":identity, "input_signature":signature, "renderer":"project.custom", "renderer_version":"fixture-v1"}
        render_preview(custom_request, [{"value":1}], root / "second", bad_renderer)
    except UserError as error:
        traversal = error.code
    print(json.dumps({"artifact_count":len(summary["artifacts"]), "contains_secret":"do-not-render" in content, "traversal":traversal}, sort_keys=True))
"#,
        &[],
    );
    assert_eq!(output["artifact_count"], 1);
    assert_eq!(output["contains_secret"], false);
    assert_eq!(output["traversal"], "PREVIEW_ARTIFACT_INVALID");
}
