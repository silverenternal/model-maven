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

fn run_workbench(project: &Path, input: &str) -> String {
    run_workbench_variant(project, None, input)
}

fn run_workbench_variant(project: &Path, variant: Option<&str>, input: &str) -> String {
    let python = python();
    let mut args = vec!["--python", python.as_str()];
    if let Some(variant) = variant {
        args.extend(["--variant", variant]);
    }
    let run = cli_ui_support::run_tty(
        Path::new(env!("CARGO_BIN_EXE_m-model")),
        project,
        &args,
        input.as_bytes(),
        Duration::from_secs(120),
    )
    .unwrap();
    assert!(!run.timed_out, "{}", run.transcript);
    assert_eq!(run.exit_code, Some(0), "{}", run.transcript);
    run.transcript
}

fn run_json(project: &Path, args: &[&str]) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", &python()])
        .args(args)
        .current_dir(project)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|error| {
        panic!(
            "expected JSON report: {error}; stdout={stdout}; stderr={}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert!(output.status.success(), "{report:#}");
    assert_eq!(report["status"], "success", "{report:#}");
    report
}

#[cfg(unix)]
#[test]
fn existing_custom_model_journey_repairs_previews_trains_evaluates_compares_and_exports() {
    if !has_torch() {
        eprintln!("skipped custom model journey: PyTorch is required");
        return;
    }
    let workspace = tempfile::tempdir().unwrap();
    let project = workspace.path().join("custom-project");
    fs::create_dir(&project).unwrap();
    let export = workspace.path().join("trained-export");
    fs::write(
        project.join("journey.py"),
        r#"
import time
import torch
from torch import nn

def build_model():
    # Deliberately wrong: the real image batch flattens to 192 features.
    return nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

def example_batch():
    time.sleep(0.5)
    return torch.ones(2, 3, 8, 8)

def training_step(model, batch, context, scale=1.0):
    return {"loss": model(batch).square().mean() * scale}

def validation_data():
    return {"values": torch.ones(2, 3, 8, 8)}

def evaluate(model, data, context):
    output = model(data["values"])
    context.report_samples(2)
    if context.max_preview_samples:
        context.report_prediction("validation-0", output[0].tolist(), 0)
    return {"metrics": {"journey/mean_score": float(output.mean().item())}}
"#,
    )
    .unwrap();
    let original_source = fs::read(project.join("journey.py")).unwrap();

    // Attach an existing Python project, save model/batch defaults, and run
    // the first actual forward pass. The shape mismatch remains visible.
    let onboarding = run_workbench(
        &project,
        "1\njourney:build_model\njourney:example_batch\n2\n2\nback\nq\n",
    );
    assert!(onboarding.contains("接入已有 Python 模型"), "{onboarding}");
    assert!(onboarding.contains("已保存默认入口"), "{onboarding}");
    assert!(onboarding.contains("前向失败"), "{onboarding}");
    assert!(onboarding.contains("FORWARD_FAILED"), "{onboarding}");
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(project.join(".model-maven/project.json")).unwrap())
            .unwrap();
    assert_eq!(config["model"], "journey:build_model");
    assert_eq!(config["batch"], "journey:example_batch");

    // The candidate edit must pass a real forward before it is committed.
    let edit = run_workbench(&project, "2\n2\nmodel.1\n\n192\n\nback\nq\n");
    assert!(edit.contains("修改变体 Layer 参数"), "{edit}");
    assert!(edit.contains("\"forward\":\"passed\""), "{edit}");
    assert!(edit.contains("Layer 参数更新：model.1"), "{edit}");
    assert_eq!(
        fs::read(project.join("journey.py")).unwrap(),
        original_source
    );
    assert!(project.join(".model-maven/variants/working.json").is_file());

    // Return to the model page and confirm that the persisted repaired
    // variant now passes the real forward probe.
    let repaired_probe =
        run_workbench_variant(&project, Some("working"), "1\n2\n\n\n\nback\nback\nq\n");
    assert!(repaired_probe.contains("前向证据"), "{repaired_probe}");
    assert!(
        !repaired_probe.contains("前向失败 · 没有通过验证"),
        "{repaired_probe}"
    );

    // Generate a real background batch preview and wait for its worker result.
    let preview = run_workbench_variant(
        &project,
        Some("working"),
        "6\n4\n\n\n\n\n2\n\n\n\n\nstart\n4\nwait\nback\nback\nq\n",
    );
    assert!(preview.contains("后台预览已启动"), "{preview}");
    assert!(preview.contains("preview.html"), "{preview}");
    let preview_html = walkdir_preview(&project.join(".model-maven/previews"));
    assert!(
        preview_html.is_file(),
        "expected real preview artifact; {preview}"
    );

    // A real, explicitly confirmed one-step managed training run.
    let training = run_workbench_variant(
        &project,
        Some("working"),
        "3\n3\n1\njourney:training_step\n1\n1\n\n\n\n\nno\nstart\nback\nq\n",
    );
    assert!(
        training.contains("确认按以上来源、预算和覆盖启动任务"),
        "{training}"
    );
    assert!(training.contains("正式训练完成：1/1 步"), "{training}");
    let runs_root = project.join(".model-maven/runs");
    let train_id = most_recent_train_id(&runs_root);
    assert!(
        runs_root
            .join(&train_id)
            .join("checkpoints/latest.json")
            .is_file()
    );

    // Export the newly trained checkpoint to a separate directory through
    // the human output page, leaving original model code unchanged.
    let export_ui = run_workbench_variant(
        &project,
        Some("working"),
        &format!("6\n1\n{}\nback\nq\n", export.display()),
    );
    assert!(export_ui.contains("导出完成"), "{export_ui}");
    assert!(export.is_dir());
    assert!(
        export.join("export.json").is_file() || export.join("manifest.json").is_file(),
        "export files: {:?}; {export_ui}",
        fs::read_dir(&export)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        fs::read(project.join("journey.py")).unwrap(),
        original_source
    );

    // Evaluate the retained training checkpoint and browse a real reported
    // sample through the run-detail page.
    let evaluation = run_workbench(
        &project,
        &format!(
            "3\n5\n{train_id}\nevaluate\njourney:evaluate\njourney:validation_data\n\nvalidation\n\n2\n\n\n\ny\nback\nback\nback\nback\nq\n"
        ),
    );
    assert!(evaluation.contains("评估结果"), "{evaluation}");
    assert!(evaluation.contains("journey/mean_score"), "{evaluation}");
    assert!(evaluation.contains("validation-0"), "{evaluation}");
    assert!(project.join(".model-maven/runs").is_dir());

    // Two distinct real try runs are then selected and compared in the
    // workbench; different step parameters make the observed losses differ.
    let first = run_json(
        &project,
        &[
            "--variant",
            "working",
            "try",
            "--model",
            "journey:build_model",
            "--batch",
            "journey:example_batch",
            "--step",
            "journey:training_step",
            "--step-param",
            "scale=1.0",
            "--steps",
            "1",
            "--name",
            "baseline",
        ],
    );
    let second = run_json(
        &project,
        &[
            "--variant",
            "working",
            "try",
            "--model",
            "journey:build_model",
            "--batch",
            "journey:example_batch",
            "--step",
            "journey:training_step",
            "--step-param",
            "scale=2.0",
            "--steps",
            "1",
            "--name",
            "scaled",
        ],
    );
    assert_ne!(first["data"]["run_id"], second["data"]["run_id"]);
    assert_ne!(first["data"]["mean_loss"], second["data"]["mean_loss"]);
    let comparison =
        run_workbench_variant(&project, Some("working"), "3\n10\n1\n2\nback\nback\nq\n");
    assert!(comparison.contains("运行对照"), "{comparison}");
    assert!(
        comparison.contains(first["data"]["run_id"].as_str().unwrap()),
        "{comparison}"
    );
    assert!(
        comparison.contains(second["data"]["run_id"].as_str().unwrap()),
        "{comparison}"
    );
    assert_eq!(
        fs::read(project.join("journey.py")).unwrap(),
        original_source
    );
}

fn walkdir_preview(root: &Path) -> std::path::PathBuf {
    fn visit(path: &Path) -> Option<std::path::PathBuf> {
        for entry in fs::read_dir(path).ok()? {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = visit(&path) {
                    return Some(found);
                }
            } else if path.file_name().is_some_and(|name| name == "preview.html") {
                return Some(path);
            }
        }
        None
    }
    visit(root).unwrap_or_else(|| root.join("missing-preview.html"))
}

fn most_recent_train_id(runs_root: &Path) -> String {
    let mut runs = fs::read_dir(runs_root)
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let value: serde_json::Value =
                serde_json::from_slice(&fs::read(path.join("run.json")).ok()?).ok()?;
            if value["operation"] == "train" && value["status"] == "completed" {
                Some((
                    value["started_at_ms"].as_u64().unwrap_or_default(),
                    value["id"].as_str()?.to_owned(),
                ))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    runs.sort_by_key(|run| run.0);
    runs.pop().expect("a completed training run exists").1
}
