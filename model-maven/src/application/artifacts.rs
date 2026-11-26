use super::*;

impl<B: Backend> Application<B> {
    pub(super) fn handle_export(
        &self,
        cli: &Cli,
        directory: &std::path::Path,
        run: Option<&str>,
        if_checkpoint_sha256: Option<&str>,
        overwrite: bool,
    ) -> Report {
        if if_checkpoint_sha256.is_some_and(|digest| {
            digest.len() != 64
                || !digest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }) {
            return Report::error(
                "export",
                "CHECKPOINT_DIGEST_INVALID",
                "checkpoint 摘要必须是 64 位小写十六进制 SHA-256。",
                "通过 runs show 查看 checkpoint 摘要。",
            );
        }
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error("export", "PROJECT_UNAVAILABLE", error.to_string(), "");
            }
        };
        let run_id = if let Some(id) = run {
            id.to_owned()
        } else {
            match self.runs.list(&root) {
                Ok(records) => match records.iter().find(|record| {
                    record["operation"] == "train"
                        && record["checkpoint"].is_object()
                        && matches!(
                            record["status"].as_str(),
                            Some("completed" | "failed" | "interrupted")
                        )
                }) {
                    Some(record) => record["id"].as_str().unwrap_or_default().to_owned(),
                    None => {
                        return Report::error(
                            "export",
                            "EXPORT_RUN_REQUIRED",
                            "本项目没有带可用 checkpoint 的正式训练记录。",
                            "先运行 m-model train，或使用 --run <id> 指定训练记录。",
                        );
                    }
                },
                Err(error) => {
                    return Report::error("export", "RUN_STORE_FAILED", error, "");
                }
            }
        };
        let run_record = match self.runs.show(&root, &run_id) {
            Ok(record) => record,
            Err(error) => {
                return Report::error(
                    "export",
                    "EXPORT_RUN_UNAVAILABLE",
                    error,
                    "使用 m-model runs list 查看可用运行 ID。",
                );
            }
        };
        if let Some(expected) = if_checkpoint_sha256 {
            if run_record["checkpoint"]["sha256"].as_str() != Some(expected) {
                return Report::error(
                    "export",
                    "CHECKPOINT_CHANGED",
                    "所选运行的 checkpoint 与确认的摘要不同。",
                    "重新查看检查点并确认后重试；没有创建导出目录。",
                );
            }
        }
        if run_record["record"]["operation"] != "train"
            || run_record["inputs"]["operation"] != "train"
        {
            return Report::error(
                "export",
                "EXPORT_RUN_INVALID",
                "只能导出正式训练运行；try 试跑不包含模型 checkpoint。",
                "使用 m-model train 创建可导出的运行。",
            );
        }
        if !matches!(
            run_record["record"]["status"].as_str(),
            Some("completed" | "failed" | "interrupted")
        ) {
            return Report::error(
                "export",
                "EXPORT_RUN_ACTIVE",
                "活动或状态未知的运行不能导出。",
                "先等待训练结束，再使用 m-model runs show 检查 checkpoint。",
            );
        }
        let checkpoint = match self.runs.latest_checkpoint(&root, &run_id) {
            Ok(path) => path,
            Err(error) => {
                return Report::error(
                    "export",
                    "EXPORT_CHECKPOINT_UNAVAILABLE",
                    error,
                    "只有带有效 checkpoint 的正式训练运行可导出。",
                );
            }
        };
        let recorded_python = run_record["report"]["data"]["runtime"]["python_executable"]
            .as_str()
            .filter(|value| !value.is_empty())
            .or_else(|| run_record["inputs"]["runtime"]["python_executable"].as_str().filter(|value| !value.is_empty()));
        // A project/PATH fallback is not evidence of the historical runtime.
        // Only an explicit user choice can substitute for a missing record.
        let python_executable = recorded_python.or_else(|| {
            cli.python_explicit.then(|| cli.python.as_deref()).flatten().filter(|value| !value.is_empty())
        });
        let mut environment = if run_record["report"]["data"]["environment_snapshot"].is_object() {
            run_record["report"]["data"]["environment_snapshot"].clone()
        } else if python_executable.is_none() {
            json!({"status":"unavailable", "capture_timing":"not_attempted",
                "reason":"历史运行未记录 Python；未自动选择当前环境。"})
        } else {
            let environment_report = self.backend.execute(&json!({
                "protocol_version":PROTOCOL_VERSION,
                "operation":"environment_snapshot",
                "runtime":{
                    "python_executable":python_executable,
                    "timeout_seconds":cli.timeout.unwrap_or(30)
                }
            }));
            if environment_report.success() {
                let mut data = environment_report.data;
                if let Some(fields) = data.as_object_mut() {
                    fields.insert("capture_python_source".into(), json!(if recorded_python.is_some() { "recorded" } else { "explicit_export_override" }));
                }
                data
            } else {
                json!({
                    "status":"unavailable",
                    "capture_timing":"export_time",
                    "python_executable":python_executable,
                    "capture_python_source":if recorded_python.is_some() { "recorded" } else { "explicit_export_override" },
                    "diagnostics":environment_report.diagnostics
                })
            }
        };
        let extension_manifests =
            match active_extension_manifests(self.components.as_ref(), &root, &run_record) {
                Ok(manifests) => manifests,
                Err(error) => {
                    return Report::error(
                        "export",
                        "EXPORT_COMPONENT_CHANGED",
                        error,
                        "检查该训练变体固定的组件版本；导出不会静默替换扩展来源。",
                    );
                }
            };
        let (extension_packages, runtime_requirements) = if extension_manifests.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            if python_executable.is_none() {
                return Report::error("export", "EXPORT_PYTHON_REQUIRED",
                    "历史运行未记录 Python，无法核验扩展包。",
                    "明确使用 --python 选择用于核验扩展的解释器后重试；这不证明其与历史环境相同。");
            }
            let package_plan = self.backend.execute(&json!({
                "protocol_version":PROTOCOL_VERSION,
                "operation":"extension_bundle_plan",
                "components":extension_manifests,
                "runtime":{
                    "python_executable":python_executable,
                    "timeout_seconds":cli.timeout.unwrap_or(30)
                }
            }));
            if !package_plan.success() {
                let message = package_plan
                    .diagnostics
                    .first()
                    .map(|diagnostic| diagnostic.message.clone())
                    .unwrap_or_else(|| "无法核验训练使用的扩展包。".into());
                return Report::error(
                    "export",
                    "EXPORT_COMPONENT_PACKAGE_UNAVAILABLE",
                    message,
                    "恢复训练时使用的 Python 扩展版本后重试；不完整的扩展导出不会提交。",
                );
            }
            (
                package_plan.data["packages"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default(),
                package_plan.data["runtime_requirements"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default(),
            )
        };
        if !extension_packages.is_empty() {
            if let Some(fields) = environment.as_object_mut() {
                fields.insert("extension_verification_python".into(), json!({
                    "executable": python_executable,
                    "source": if recorded_python.is_some() { "recorded" } else { "explicit_export_override" },
                    "same_as_training_environment": "unverified"
                }));
            }
        }
        match self.exporter.export_training_run(TrainingExportRequest {
            project_root: &root,
            destination: directory,
            run: &run_record,
            checkpoint: &checkpoint,
            environment: &environment,
            extension_packages: &extension_packages,
            runtime_requirements: &runtime_requirements,
            overwrite,
        }) {
            Ok(data) => Report::ok("export", data),
            Err(error) => Report::error(
                "export",
                "EXPORT_FAILED",
                error,
                "导出先在临时目录构建并校验；失败时原目标和训练记录保持不变。",
            ),
        }
    }
}
