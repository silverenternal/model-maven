use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn python() -> String {
    std::env::var("MODEL_MAVEN_TEST_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn invoke(root: &Path, args: &[&str]) -> (std::process::Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_m-model"))
        .args(["--json", "--python", &python()])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report = serde_json::from_slice(stdout.trim().as_bytes()).unwrap_or_else(|error| {
        panic!(
            "expected one JSON report, got {stdout:?}: {error}; stderr={}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output, report)
}

fn assert_success(output: &std::process::Output, report: &Value) {
    assert!(
        output.status.success(),
        "{report:#}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(report["status"], "success", "{report:#}");
}

#[test]
fn finished_run_registers_checkpoint_with_hash_and_storage_category() {
    if !Command::new(python())
        .args(["-c", "import torch"])
        .status()
        .is_ok_and(|status| status.success())
    {
        eprintln!(
            "skipped PyTorch integration: set MODEL_MAVEN_TEST_PYTHON to a PyTorch environment"
        );
        return;
    }
    let project = tempfile::tempdir().unwrap();
    fs::write(
        project.path().join("external_loop.py"),
        "def run_loop(model, context):\n    context.report_step(0.5, {'updates': 1})\n    path = context.artifact_dir / 'weights.bin'\n    path.write_bytes(b'custom checkpoint bytes')\n    context.register_artifact('custom-weights', ['weights.bin'], owner='example.external-loop')\n    (context.artifact_dir / 'model.bin').write_bytes(b'model-state')\n    (context.artifact_dir / 'optimizer.bin').write_bytes(b'optimizer-state')\n    context.register_artifact('disposable-checkpoint', ['model.bin', 'optimizer.bin'], owner='example.external-loop')\n    (context.artifact_dir / 'late.bin').write_bytes(b'late consumer weights')\n    context.register_artifact('late-checkpoint', ['late.bin'], owner='example.external-loop')\n    return {'finished': True}\nrun_loop.MODEL_MAVEN_STATELESS = True\n",
    )
    .unwrap();
    let (output, trained) = invoke(
        project.path(),
        &[
            "train",
            "--training-mode",
            "external-loop",
            "--loop",
            "external_loop:run_loop",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    );
    assert_success(&output, &trained);
    let run_id = trained["data"]["run_id"].as_str().unwrap();
    let run_directory = project.path().join(".model-maven/runs").join(run_id);
    fs::write(
        run_directory.join("artifacts/unregistered-checkpoint.pt"),
        b"leave unregistered user data alone",
    )
    .unwrap();
    let manifest: Value =
        serde_json::from_slice(&fs::read(run_directory.join("artifacts.json")).unwrap()).unwrap();
    assert_eq!(manifest["status"], "complete");
    let checkpoint = manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["id"] == "training-checkpoint")
        .unwrap();
    assert_eq!(checkpoint["complete"], true);
    assert_eq!(
        checkpoint["references"],
        serde_json::json!(["latest-checkpoint"])
    );
    assert!(checkpoint["files"][0]["sha256"].as_str().is_some());
    let custom = manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["id"] == "custom-weights")
        .unwrap();
    assert_eq!(custom["owner"], "example.external-loop");
    assert_eq!(custom["complete"], true);
    assert_eq!(custom["references"], serde_json::json!([]));
    assert_eq!(custom["files"][0]["path"], "artifacts/weights.bin");
    let disposable = manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["id"] == "disposable-checkpoint")
        .unwrap();
    assert_eq!(disposable["kind"], "checkpoint");
    assert_eq!(disposable["references"], serde_json::json!([]));
    assert_eq!(disposable["files"].as_array().unwrap().len(), 2);

    fs::write(
        project.path().join("consumer_loop.py"),
        format!(
            "def run_loop(model, context):\n    context.report_step(0.25, {{'updates': 1}})\n    path = context.artifact_dir / 'result.bin'\n    path.write_bytes(b'consumer result')\n    context.register_artifact('consumer-result', ['result.bin'], owner='example.consumer', kind='other', consumes=[{{'run_id': '{run_id}', 'artifact_id': 'custom-weights'}}])\n    return {{'finished': True}}\nrun_loop.MODEL_MAVEN_STATELESS = True\n"
        ),
    )
    .unwrap();
    let (consumer_output, consumer_report) = invoke(
        project.path(),
        &[
            "train",
            "--training-mode",
            "external-loop",
            "--loop",
            "consumer_loop:run_loop",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    );
    assert_success(&consumer_output, &consumer_report);
    let consumer_run_id = consumer_report["data"]["run_id"].as_str().unwrap();
    let consumer_dir = project
        .path()
        .join(".model-maven/runs")
        .join(consumer_run_id);
    let consumer_manifest: Value =
        serde_json::from_slice(&fs::read(consumer_dir.join("artifacts.json")).unwrap()).unwrap();
    assert_eq!(
        consumer_manifest["artifacts"][0]["consumes"],
        serde_json::json!([{"run_id":run_id,"artifact_id":"custom-weights"}])
    );

    let (late_plan_output, late_plan) = invoke(
        project.path(),
        &[
            "runs",
            "clean",
            "--run",
            run_id,
            "--include-checkpoints",
            "--artifact",
            "late-checkpoint",
            "--dry-run",
        ],
    );
    assert_success(&late_plan_output, &late_plan);
    assert_eq!(late_plan["data"]["items"].as_array().unwrap().len(), 1);
    let late_plan_id = late_plan["data"]["id"].as_str().unwrap();
    fs::write(
        project.path().join("late_consumer_loop.py"),
        format!(
            "def run_loop(model, context):\n    context.report_step(0.25, {{'updates': 1}})\n    path = context.artifact_dir / 'result.bin'\n    path.write_bytes(b'late consumer result')\n    context.register_artifact('late-consumer-result', ['result.bin'], owner='example.consumer', kind='other', consumes=[{{'run_id': '{run_id}', 'artifact_id': 'late-checkpoint'}}])\n    return {{'finished': True}}\nrun_loop.MODEL_MAVEN_STATELESS = True\n"
        ),
    )
    .unwrap();
    let (late_consumer_output, late_consumer_report) = invoke(
        project.path(),
        &[
            "train",
            "--training-mode",
            "external-loop",
            "--loop",
            "late_consumer_loop:run_loop",
            "--steps",
            "1",
            "--checkpoint-every",
            "1",
        ],
    );
    assert_success(&late_consumer_output, &late_consumer_report);
    let (late_apply_output, late_apply) = invoke(
        project.path(),
        &[
            "runs",
            "clean",
            "--apply",
            late_plan_id,
            "--release-checkpoints",
        ],
    );
    assert_success(&late_apply_output, &late_apply);
    assert!(
        late_apply["data"]["apply_skipped"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| {
                item["reason"]
                    .as_str()
                    .unwrap_or("")
                    .contains("预览后出现 typed consumer")
            })
    );
    assert!(run_directory.join("artifacts/late.bin").is_file());

    #[cfg(unix)]
    {
        let checkpoint_path = run_directory.join(checkpoint["files"][0]["path"].as_str().unwrap());
        fs::hard_link(&checkpoint_path, project.path().join("external-weight.pt")).unwrap();
        let (clean_output, clean_plan) = invoke(
            project.path(),
            &[
                "runs",
                "clean",
                "--run",
                run_id,
                "--include-checkpoints",
                "--dry-run",
            ],
        );
        assert_success(&clean_output, &clean_plan);
        let items = clean_plan["data"]["items"].as_array().unwrap();
        assert!(
            !items
                .iter()
                .any(|item| item["kind"] == "latest_checkpoint_release")
        );
        let managed_plan_item = items
            .iter()
            .find(|item| item["kind"] == "managed_checkpoint_release")
            .unwrap_or_else(|| panic!("missing managed checkpoint item: {clean_plan:#}"));
        assert_eq!(managed_plan_item["artifact_id"], "disposable-checkpoint");
        assert!(
            !items.iter().any(|item| {
                item["path"] == "artifacts/unregistered-checkpoint.pt"
                    || item["files"].as_array().is_some_and(|files| {
                        files
                            .iter()
                            .any(|file| file["path"] == "artifacts/unregistered-checkpoint.pt")
                    })
            }),
            "unregistered .pt files must not be inferred as disposable checkpoints: {items:#?}"
        );
        assert!(
            !items
                .iter()
                .any(|item| item["artifact_id"] == "custom-weights")
        );
        assert!(
            clean_plan["data"]["skipped"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["reason"]
                    .as_str()
                    .unwrap_or("")
                    .contains("硬链接共享引用"))
        );
        assert!(
            clean_plan["data"]["skipped"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["artifact_id"] == "custom-weights"
                    && item["consumers"].as_array().is_some_and(|consumers| {
                        consumers.iter().any(|consumer| {
                            consumer == &format!("{consumer_run_id}/consumer-result")
                        })
                    }))
        );

        let (referenced_output, referenced_plan) = invoke(
            project.path(),
            &[
                "runs",
                "clean",
                "--run",
                run_id,
                "--include-checkpoints",
                "--artifact",
                "custom-weights",
                "--dry-run",
            ],
        );
        assert_success(&referenced_output, &referenced_plan);
        assert!(
            referenced_plan["data"]["items"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(
            referenced_plan["data"]["skipped"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["artifact_id"] == "custom-weights"
                    && item["consumers"]
                        .as_array()
                        .is_some_and(|consumers| !consumers.is_empty()))
        );

        let (selected_output, selected_plan) = invoke(
            project.path(),
            &[
                "runs",
                "clean",
                "--run",
                run_id,
                "--include-checkpoints",
                "--artifact",
                "disposable-checkpoint",
                "--dry-run",
            ],
        );
        assert_success(&selected_output, &selected_plan);
        assert_eq!(selected_plan["data"]["items"].as_array().unwrap().len(), 1);
        assert_eq!(
            selected_plan["data"]["items"][0]["artifact_id"],
            "disposable-checkpoint"
        );
        let plan_id = selected_plan["data"]["id"].as_str().unwrap();
        let tampered_relative = selected_plan["data"]["items"][0]["files"][0]["path"]
            .as_str()
            .unwrap();
        let tampered_path = run_directory.join(tampered_relative);
        let original = fs::read(&tampered_path).unwrap();
        fs::write(&tampered_path, b"changed-after-plan").unwrap();
        let (apply_output, applied) = invoke(
            project.path(),
            &["runs", "clean", "--apply", plan_id, "--release-checkpoints"],
        );
        assert_success(&apply_output, &applied);
        assert!(
            applied["data"]["apply_skipped"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["reason"].as_str().unwrap_or("").contains("内容已变化"))
        );
        assert!(run_directory.join("artifacts/model.bin").is_file());
        assert!(run_directory.join("artifacts/optimizer.bin").is_file());
        fs::write(&tampered_path, original).unwrap();

        let (release_plan_output, release_plan) = invoke(
            project.path(),
            &[
                "runs",
                "clean",
                "--run",
                run_id,
                "--include-checkpoints",
                "--artifact",
                "disposable-checkpoint",
                "--dry-run",
            ],
        );
        assert_success(&release_plan_output, &release_plan);
        let release_plan_id = release_plan["data"]["id"].as_str().unwrap();
        let (release_output, released) = invoke(
            project.path(),
            &[
                "runs",
                "clean",
                "--apply",
                release_plan_id,
                "--release-checkpoints",
            ],
        );
        assert_success(&release_output, &released);
        assert!(
            released["data"]["applied_items"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["artifact_id"] == "disposable-checkpoint")
        );
        assert!(!run_directory.join("artifacts/model.bin").exists());
        assert!(!run_directory.join("artifacts/optimizer.bin").exists());
        assert!(
            run_directory
                .join("artifacts/unregistered-checkpoint.pt")
                .is_file()
        );
        let after_release: Value =
            serde_json::from_slice(&fs::read(run_directory.join("artifacts.json")).unwrap())
                .unwrap();
        let released_artifact = after_release["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|artifact| artifact["id"] == "disposable-checkpoint")
            .unwrap();
        assert_eq!(released_artifact["availability"], "released");
        assert_eq!(released_artifact["complete"], false);
        assert_eq!(released_artifact["files"][0]["available"], false);
    }

    let (show_output, show) = invoke(project.path(), &["runs", "show", run_id]);
    assert_success(&show_output, &show);
    assert!(
        show["data"]["artifacts"]["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|artifact| artifact["id"] == "training-checkpoint")
    );

    let (storage_output, storage) = invoke(project.path(), &["runs", "storage", run_id]);
    assert_success(&storage_output, &storage);
    assert!(
        storage["data"]["categories"]["checkpoints"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        storage["data"]["categories"]["artifacts"],
        (b"custom checkpoint bytes".len() + b"late consumer weights".len()) as u64
    );
}
