use super::*;

impl<B: Backend> Application<B> {
    pub(super) fn reconcile_lost_runs(
        &self,
        root: &std::path::Path,
        operation: &str,
    ) -> Result<(), Report> {
        let records = self.runs.list(root).map_err(|error| {
            Report::error(
                operation,
                "RUN_STORE_FAILED",
                format!("无法核对项目活动任务：{error}"),
                "检查 .model-maven/runs 记录；状态不明时不会启动新任务。",
            )
        })?;
        for item in records {
            if !matches!(item["status"].as_str(), Some("running" | "stopping")) {
                continue;
            }
            let Some(id) = item["id"].as_str() else {
                continue;
            };
            let record = self.runs.show(root, id).map_err(|error| {
                Report::error(
                    operation,
                    "RUN_STATE_UNKNOWN",
                    format!("无法读取活动运行 {id} 的监督身份：{error}"),
                    "先恢复运行记录可读性；在身份不明时不会启动新任务。",
                )
            })?;
            let supervisor = &record["record"]["supervisor"];
            let identity = supervisor["pid"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .zip(supervisor["start_token"].as_str());
            let Some((pid, token)) = identity else {
                continue;
            };
            if self.supervisor.is_alive(pid, token) {
                continue;
            }
            let lost = Report::error(
                "run",
                "RUN_PROCESS_LOST",
                format!("运行 {id} 的监督进程已退出；已保留其输入和最近可验证恢复点。"),
                "检查 runs show 与 worker.log；确认恢复点后可从该运行继续或重跑。",
            );
            let serialized = serde_json::to_value(lost).map_err(|error| {
                Report::error(
                    operation,
                    "RUN_REPORT_INVALID",
                    error.to_string(),
                    "运行记录保持不变；请保留诊断信息。",
                )
            })?;
            if let Err(error) = self.runs.finish(root, id, &serialized) {
                let latest = self.runs.show(root, id).map_err(|read_error| {
                    Report::error(
                        operation,
                        "RUN_STORE_FAILED",
                        format!("无法确认运行 {id} 的中断状态：{error}; {read_error}"),
                        "保留原运行记录；检查运行目录权限和磁盘空间。",
                    )
                })?;
                if matches!(
                    latest["record"]["status"].as_str(),
                    Some("running" | "stopping")
                ) {
                    return Err(Report::error(
                        operation,
                        "RUN_STORE_FAILED",
                        format!("无法记录运行 {id} 的中断状态：{error}"),
                        "保留原运行记录；检查运行目录权限和磁盘空间。",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish_run_report(
        &self,
        root: &std::path::Path,
        id: &str,
        operation: &str,
        request: &Value,
        mut report: Report,
    ) -> Report {
        if let Some(data) = report.data.as_object_mut() {
            data.insert("run_id".into(), json!(id));
            if let Some(parent) = request["repeat_of"].as_str() {
                data.insert("repeat_of".into(), json!(parent));
            }
            if let Some(parent) = request["parent_run_id"].as_str() {
                data.insert("parent_run_id".into(), json!(parent));
            }
            if request["repeat_overrides"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
            {
                data.insert(
                    "repeat_overrides".into(),
                    request["repeat_overrides"].clone(),
                );
            }
        }
        let serialized = match serde_json::to_value(&report) {
            Ok(value) => value,
            Err(error) => {
                return Report::error(
                    operation,
                    "RUN_REPORT_INVALID",
                    error.to_string(),
                    "试跑结果未能保存到运行记录。",
                );
            }
        };
        if let Err(error) = self.runs.finish(root, id, &serialized) {
            let mut failed = Report::error(
                operation,
                "RUN_STORE_FAILED",
                format!("试跑结束，但运行结果未能完整保存：{error}"),
                "保留本次终端/JSON结果，并检查 .model-maven/runs 磁盘和权限。",
            );
            failed.data["run_id"] = json!(id);
            failed.data["training_result"] = report.data;
            return failed;
        }
        report
    }

    pub(super) fn create_run_record(
        &self,
        root: &std::path::Path,
        job: &Job,
        request: &mut Value,
    ) -> Result<Option<String>, Report> {
        if !matches!(job.operation, "training_probe" | "train") {
            return Ok(None);
        }
        let mut inputs = request.clone();
        if let Some(revision) = &job.config_revision {
            inputs["config_revision"] = json!(revision);
        }
        if !job.input_sources.is_empty() {
            inputs["input_sources"] = json!(job.input_sources);
        }
        let mut redacted_fields = Vec::new();
        redact_run_inputs(&mut inputs, "$", &mut redacted_fields);
        if let Some(inputs) = inputs.as_object_mut() {
            inputs.insert("redacted_fields".into(), json!(redacted_fields));
        }
        self.reconcile_lost_runs(root, job.operation)?;
        let id = self.runs.create(root, &inputs).map_err(|error| {
            if let Some(id) = error.strip_prefix("RUN_REQUEST_ID_EXISTS:") {
                let mut report = Report::error(
                    job.operation,
                    "RUN_REQUEST_ID_EXISTS",
                    "该启动请求标识已有运行记录；没有再次启动。",
                    "查看返回的 run_id 及当前状态；需要新训练时生成新的请求标识。",
                );
                report.data["run_id"] = json!(id);
                report.data["request_id"] = json!(job.request_id);
                return report;
            }
            if let Some(message) = error.strip_prefix("RUN_ALREADY_ACTIVE:") {
                return Report::error(
                    job.operation,
                    "RUN_ALREADY_ACTIVE",
                    message,
                    "先查看当前运行；运行结束或明确恢复状态后再启动新任务。只读查看仍可继续。",
                );
            }
            if let Some(message) = error.strip_prefix("RUN_STATE_UNKNOWN:") {
                return Report::error(
                    job.operation,
                    "RUN_STATE_UNKNOWN",
                    message,
                    "先用 m-model runs 查看损坏或未确认的运行记录；在确认其状态前不会启动新任务。",
                );
            }
            Report::error(
                job.operation,
                "RUN_STORE_FAILED",
                format!("无法创建运行记录：{error}"),
                "检查 .model-maven/runs 的目录权限和可用空间；任务尚未启动。",
            )
        })?;
        request["run_id"] = json!(id);
        let directory = match self.runs.run_directory(root, &id) {
            Ok(directory) => directory,
            Err(error) => {
                let failed = Report::error(
                    job.operation,
                    "RUN_DIRECTORY_UNAVAILABLE",
                    error,
                    "检查运行记录目录权限；用户代码尚未执行。",
                );
                self.finish_failed_run(root, &id, &failed);
                return Err(failed);
            }
        };
        request["snapshot_dir"] = json!(directory.join("snapshot"));
        if job.operation == "train" {
            let checkpoint_dir = match self.runs.checkpoint_dir(root, &id, false) {
                Ok(directory) => directory,
                Err(error) => {
                    let failed = Report::error(
                        "train",
                        "CHECKPOINT_STORE_FAILED",
                        error,
                        "训练尚未启动；检查运行目录权限和磁盘空间。",
                    );
                    self.finish_failed_run(root, &id, &failed);
                    return Err(failed);
                }
            };
            request["checkpoint_dir"] = json!(checkpoint_dir);
            request["progress_path"] = json!(directory.join("progress.jsonl"));
            request["stop_file"] = json!(directory.join("stop.request"));
            request["log_path"] = json!(directory.join("worker.log"));
        }
        if !job.detach {
            let pid = std::process::id();
            if let Some(token) = self.supervisor.identify(pid)
                && let Err(error) = self.runs.set_supervisor(root, &id, pid, &token)
            {
                let failed = Report::error(
                    job.operation,
                    "RUN_SUPERVISOR_REGISTRATION_FAILED",
                    error,
                    "任务尚未启动；检查运行记录锁和磁盘权限。",
                );
                self.finish_failed_run(root, &id, &failed);
                return Err(failed);
            }
        }
        Ok(Some(id))
    }

    fn finish_failed_run(&self, root: &std::path::Path, id: &str, report: &Report) {
        let serialized = serde_json::to_value(report).unwrap_or(Value::Null);
        let _ = self.runs.finish(root, id, &serialized);
    }

    pub(super) fn launch_detached_training(
        &self,
        root: &std::path::Path,
        id: &str,
        request: &mut Value,
    ) -> Report {
        let run_directory = match self.runs.run_directory(root, id) {
            Ok(directory) => directory,
            Err(error) => {
                let failed = Report::error(
                    "train",
                    "RUN_DIRECTORY_UNAVAILABLE",
                    error,
                    "检查运行记录目录权限。",
                );
                let _ = self.runs.finish(
                    root,
                    id,
                    &serde_json::to_value(&failed).unwrap_or(Value::Null),
                );
                return failed;
            }
        };
        request["interactive_progress"] = json!(true);
        request["progress_path"] = json!(run_directory.join("progress.jsonl"));
        request["stop_file"] = json!(run_directory.join("stop.request"));
        request["verbose"] = json!(true);
        let executable = match std::env::current_exe() {
            Ok(path) => path,
            Err(error) => {
                let failed = Report::error(
                    "train",
                    "BACKGROUND_LAUNCH_FAILED",
                    error.to_string(),
                    "检查 m-model 可执行文件路径。",
                );
                let _ = self.runs.finish(
                    root,
                    id,
                    &serde_json::to_value(&failed).unwrap_or(Value::Null),
                );
                return failed;
            }
        };
        let log_path = run_directory.join("worker.log");
        let launched = self.supervisor.launch(
            &executable,
            root,
            id,
            request,
            &log_path,
            &mut |pid, token| self.runs.set_supervisor(root, id, pid, token),
        );
        if let Err(error) = launched {
            let failed = Report::error(
                "train",
                "BACKGROUND_LAUNCH_FAILED",
                error,
                "检查目录权限、平台支持和运行日志；失败的后台启动不会继续训练。",
            );
            let _ = self.runs.finish(
                root,
                id,
                &serde_json::to_value(&failed).unwrap_or(Value::Null),
            );
            return failed;
        }
        Report::ok(
            "train",
            json!({
                "run_id":id,
                "request_id":request["request_id"],
                "detached":true,
                "status":"running",
                "attach":format!("m-model runs attach {id}"),
                "stop":format!("m-model runs stop {id}")
            }),
        )
    }
}
