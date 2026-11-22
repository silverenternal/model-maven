use super::*;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

impl<B: Backend> Application<B> {
    pub(super) fn live_supervisor(&self, value: &Value) -> Option<(u32, String)> {
        let pid = value["supervisor"]["pid"].as_u64()?.try_into().ok()?;
        let token = value["supervisor"]["start_token"].as_str()?.to_owned();
        self.supervisor
            .is_alive(pid, &token)
            .then_some((pid, token))
    }

    pub(super) fn annotate_status(&self, value: &mut Value) {
        let heartbeat = value.get("heartbeat").cloned().unwrap_or(Value::Null);
        let record = if value.get("record").is_some() {
            &mut value["record"]
        } else {
            value
        };
        if record["supervisor"]["kind"] == "detached"
            && matches!(record["status"].as_str(), Some("running" | "stopping"))
        {
            if let Some(at_ms) = heartbeat["at_ms"].as_u64() {
                let expected_pid = record["supervisor"]["pid"].as_u64();
                let heartbeat_matches = heartbeat["pid"].as_u64() == expected_pid
                    && heartbeat["start_token"] == record["supervisor"]["start_token"];
                let age_ms = now_ms().saturating_sub(at_ms);
                record["heartbeat_age_ms"] = json!(age_ms);
                record["heartbeat_status"] = json!(if heartbeat_matches && age_ms <= 15_000 {
                    "fresh"
                } else if heartbeat_matches {
                    "stale"
                } else {
                    "identity_mismatch"
                });
                if !heartbeat_matches || age_ms > 15_000 {
                    record["last_observed_status"] = record["status"].clone();
                    record["status"] = json!("unknown");
                    record["status_note"] = json!(
                        "后台监督心跳过期或身份不匹配；检查 worker.log 和最近 checkpoint 后再恢复。"
                    );
                }
            } else {
                record["heartbeat_status"] = json!("starting_or_unavailable");
            }
        }
        if matches!(record["status"].as_str(), Some("running" | "stopping"))
            && self.live_supervisor(record).is_none()
        {
            record["last_observed_status"] = record["status"].clone();
            record["status"] = json!("unknown");
            record["status_note"] = json!(
                "监督进程已退出或身份无法验证；请检查最新 checkpoint 和 worker.log 后再恢复。"
            );
        }
    }

    pub(super) fn select_active_run(
        &self,
        root: &std::path::Path,
        requested: Option<&str>,
    ) -> Result<(String, Value), String> {
        if let Some(id) = requested {
            let record = self.runs.show(root, id)?;
            if self.live_supervisor(&record["record"]).is_none() {
                return Err(format!("运行 {id} 当前没有身份可验证的活动后台监督进程。"));
            }
            return Ok((id.to_owned(), record));
        }
        let active = self
            .runs
            .list(root)?
            .into_iter()
            .filter(|run| {
                matches!(run["status"].as_str(), Some("running" | "stopping"))
                    && self.live_supervisor(run).is_some()
            })
            .collect::<Vec<_>>();
        match active.as_slice() {
            [run] => {
                let id = run["id"].as_str().ok_or("运行记录缺少 ID。")?.to_owned();
                Ok((id.clone(), self.runs.show(root, &id)?))
            }
            [] => Err("当前项目没有身份可验证的活动后台训练。".into()),
            _ => Err(format!(
                "当前有 {} 个活动后台训练，请明确指定运行 ID。",
                active.len()
            )),
        }
    }

    pub(super) fn handle_runs(&self, command: &RunsCommand) -> Report {
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => {
                return Report::error("runs", "PROJECT_UNAVAILABLE", error.to_string(), "");
            }
        };
        match command {
            RunsCommand::FindRequest { request_id } => {
                if !(16..=64).contains(&request_id.len())
                    || !request_id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-')
                {
                    return Report::error(
                        "runs_find_request",
                        "REQUEST_ID_INVALID",
                        "请求标识格式无效。",
                        "使用启动时的 16–64 位字母、数字或连字符标识。",
                    );
                }
                match self.runs.find_request(&root, request_id) {
                    Ok(mut result) => {
                        result["request_id"] = json!(request_id);
                        Report::ok("runs_find_request", result)
                    }
                    Err(error) => Report::error("runs_find_request", "RUN_STORE_FAILED", error, ""),
                }
            }
            RunsCommand::List {
                query,
                status,
                kept,
            } => match self.runs.list(&root) {
                Ok(mut runs) => {
                    for run in &mut runs {
                        self.annotate_status(run);
                        if matches!(run["status"].as_str(), Some("running" | "stopping"))
                            && let Some(id) = run["id"].as_str()
                            && let Ok(progress) = self.runs.progress(&root, id)
                            && let Some(latest) = progress.last()
                        {
                            run["live_progress"] = latest.clone();
                        }
                    }
                    let query = query.as_deref().map(str::to_lowercase);
                    runs.retain(|run| {
                        let status_matches = match status.as_deref() {
                            Some("active") => {
                                matches!(run["status"].as_str(), Some("running" | "stopping"))
                            }
                            Some(expected) => run["status"] == expected,
                            None => true,
                        };
                        let query_matches = query.as_ref().is_none_or(|query| {
                            run["id"]
                                .as_str()
                                .unwrap_or("")
                                .to_lowercase()
                                .contains(query)
                                || run["name"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_lowercase()
                                    .contains(query)
                        });
                        let kept_matches = kept.is_none_or(|expected| run["kept"] == expected);
                        status_matches && query_matches && kept_matches
                    });
                    let filtered = query.is_some() || status.is_some() || kept.is_some();
                    Report::ok("runs_list", json!({"runs": runs,"filtered":filtered}))
                }
                Err(error) => Report::error("runs_list", "RUN_STORE_FAILED", error, ""),
            },
            RunsCommand::Show { id } => {
                let selected = match id {
                    Some(id) => id.clone(),
                    None => match self.runs.list(&root) {
                        Ok(runs) => match crate::domain::comparison::latest_run(&runs, None)
                            .and_then(|run| run["id"].as_str())
                        {
                            Some(id) => id.to_string(),
                            None => {
                                return Report::error(
                                    "runs_show",
                                    "RUN_NOT_FOUND_OR_INVALID",
                                    "本项目还没有可查看的试跑记录。",
                                    "先运行 m-model try，再查看最近记录。",
                                );
                            }
                        },
                        Err(error) => {
                            return Report::error("runs_show", "RUN_STORE_FAILED", error, "");
                        }
                    },
                };
                match self.runs.show(&root, &selected) {
                    Ok(mut record) => {
                        self.annotate_status(&mut record);
                        Report::ok("runs_show", record)
                    }
                    Err(error) => Report::error(
                        "runs_show",
                        "RUN_NOT_FOUND_OR_INVALID",
                        error,
                        "m-model runs list",
                    ),
                }
            }
            RunsCommand::Compare { left, right } => {
                let left_record = match self.runs.show(&root, left) {
                    Ok(record) => record,
                    Err(error) => {
                        return Report::error(
                            "runs_compare",
                            "RUN_NOT_FOUND",
                            format!("左侧运行 {left} 不可读取：{error}"),
                            "运行 m-model runs list 查看可用记录。",
                        );
                    }
                };
                let right_record = match self.runs.show(&root, right) {
                    Ok(record) => record,
                    Err(error) => {
                        return Report::error(
                            "runs_compare",
                            "RUN_NOT_FOUND",
                            format!("右侧运行 {right} 不可读取：{error}"),
                            "运行 m-model runs list 查看可用记录。",
                        );
                    }
                };
                Report::ok(
                    "runs_compare",
                    crate::domain::comparison::compare_runs(&left_record, &right_record),
                )
            }
            RunsCommand::Attach { id, .. } => {
                let (id, mut record) = match id {
                    Some(id) => match self.runs.show(&root, id) {
                        Ok(record) => (id.clone(), record),
                        Err(error) => {
                            return Report::error(
                                "runs_attach",
                                "RUN_NOT_FOUND",
                                error,
                                "运行 m-model runs list 查看可用 ID。",
                            );
                        }
                    },
                    None => match self.select_active_run(&root, None) {
                        Ok(selected) => selected,
                        Err(error) => {
                            return Report::error(
                                "runs_attach",
                                "RUN_NOT_ACTIVE",
                                error,
                                "运行 m-model runs list 查看状态；存在多个活动运行时指定 ID。",
                            );
                        }
                    },
                };
                self.annotate_status(&mut record);
                let progress = match self.runs.progress(&root, &id) {
                    Ok(progress) => progress,
                    Err(error) => {
                        return Report::error(
                            "runs_attach",
                            "RUN_PROGRESS_UNAVAILABLE",
                            error,
                            "查看 m-model runs show <id> 与对应 worker.log。",
                        );
                    }
                };
                let log_tail = match self.runs.log_tail(&root, &id, 16 * 1024) {
                    Ok(log) => log,
                    Err(error) => {
                        return Report::error(
                            "runs_attach",
                            "RUN_LOG_UNAVAILABLE",
                            error,
                            "检查运行目录权限。",
                        );
                    }
                };
                Report::ok(
                    "runs_attach",
                    json!({"run_id":id,"record":record["record"],"progress":progress,"log_tail":log_tail}),
                )
            }
            RunsCommand::Stop { id, force } => {
                let (id, record) = match self.select_active_run(&root, id.as_deref()) {
                    Ok(selected) => selected,
                    Err(error) => {
                        return Report::error(
                            "runs_stop",
                            "RUN_NOT_ACTIVE",
                            error,
                            "运行 m-model runs list 查看活动运行。",
                        );
                    }
                };
                if let Err(error) = self.runs.request_stop(&root, &id) {
                    return Report::error(
                        "runs_stop",
                        "RUN_STOP_FAILED",
                        error,
                        "检查运行状态后重试。",
                    );
                }
                if *force {
                    let Some((pid, token)) = self.live_supervisor(&record["record"]) else {
                        return Report::error(
                            "runs_stop",
                            "RUN_NOT_ACTIVE",
                            "停止请求已写入，但监督进程身份已变化，未发送信号。",
                            "确认运行状态后检查最近 checkpoint。",
                        );
                    };
                    if let Err(error) = self.supervisor.interrupt(pid, &token) {
                        return Report::error(
                            "runs_stop",
                            "RUN_INTERRUPT_FAILED",
                            format!("协作停止请求已写入；立即中断失败：{error}"),
                            "worker 仍会在下一个训练步骤边界安全停止。",
                        );
                    }
                }
                Report::ok(
                    "runs_stop",
                    json!({
                        "run_id":id,
                        "status":"stopping",
                        "force":force,
                        "message": if *force { "已发出立即中断信号；有已提交 checkpoint 时可从该恢复点继续，未产生 checkpoint 的运行不可续训。" } else { "已请求停止；训练会在下一个完整步骤后保存 checkpoint 并退出。" }
                    }),
                )
            }
            RunsCommand::Storage { id } => match self.runs.storage(&root, id.as_deref()) {
                Ok(data) => Report::ok("runs_storage", data),
                Err(error) => Report::error(
                    "runs_storage",
                    "RUN_STORAGE_SCAN_FAILED",
                    error,
                    "运行 m-model runs list 核对记录并检查运行目录权限。",
                ),
            },
            RunsCommand::Rename { id, name } => match self.runs.rename(&root, id, name) {
                Ok(()) => Report::ok("runs_rename", json!({"run_id":id,"name":name})),
                Err(error) => Report::error(
                    "runs_rename",
                    "RUN_METADATA_UPDATE_FAILED",
                    error,
                    "运行 m-model runs show <id> 确认记录后重试。",
                ),
            },
            RunsCommand::Keep { id } | RunsCommand::Unkeep { id } => {
                let kept = matches!(command, RunsCommand::Keep { .. });
                match self.runs.set_kept(&root, id, kept) {
                    Ok(()) => Report::ok(
                        if kept { "runs_keep" } else { "runs_unkeep" },
                        json!({"run_id":id,"kept":kept}),
                    ),
                    Err(error) => Report::error(
                        "runs_keep",
                        "RUN_METADATA_UPDATE_FAILED",
                        error,
                        "运行 m-model runs show <id> 确认记录后重试。",
                    ),
                }
            }
            RunsCommand::Clean {
                run,
                include_logs,
                include_progress,
                include_checkpoints,
                artifact,
                dry_run,
                apply,
                release_checkpoints,
            } => {
                if *dry_run == apply.is_some() {
                    return Report::error(
                        "runs_clean",
                        "CLEAN_ACTION_REQUIRED",
                        "请选择 --dry-run 预览，或 --apply <plan-id> 执行已预览计划。",
                        "普通清理不会直接删除文件。",
                    );
                }
                if *include_checkpoints && run.is_none() {
                    return Report::error(
                        "runs_clean",
                        "CLEAN_TARGET_REQUIRED",
                        "释放权重必须使用 --run <id> 明确指定一个已结束运行。",
                        "先运行 m-model runs clean --run <id> --include-checkpoints --dry-run 查看后果。",
                    );
                }
                if artifact.is_some() && !include_checkpoints {
                    return Report::error(
                        "runs_clean",
                        "CLEAN_ARTIFACT_REQUIRES_CHECKPOINTS",
                        "--artifact 只能与 --include-checkpoints 一起使用。",
                        "重新预览时指定 --run、--include-checkpoints 和目标 artifact ID。",
                    );
                }
                if apply.is_some()
                    && (*include_logs
                        || *include_progress
                        || *include_checkpoints
                        || run.is_some()
                        || artifact.is_some())
                {
                    return Report::error(
                        "runs_clean",
                        "CLEAN_PLAN_ONLY",
                        "执行时只接受 --apply <plan-id>；目标和选项必须在预览阶段确定。",
                        "重新使用 --dry-run 生成计划，再单独执行该计划。",
                    );
                }
                if *dry_run {
                    match self.runs.plan_clean(
                        &root,
                        run.as_deref(),
                        *include_logs,
                        *include_progress,
                        *include_checkpoints,
                        artifact.as_deref(),
                    ) {
                        Ok(plan) => Report::ok("runs_clean_plan", plan),
                        Err(error) => Report::error(
                            "runs_clean_plan",
                            "CLEAN_PLAN_FAILED",
                            error,
                            "检查运行状态和 .model-maven/runs 中的文件。",
                        ),
                    }
                } else {
                    let Some(plan_id) = apply.as_deref() else {
                        unreachable!("selected clean action has an apply plan")
                    };
                    match self.runs.apply_clean(&root, plan_id, *release_checkpoints) {
                        Ok(result) => Report::ok("runs_clean_apply", result),
                        Err(error) => Report::error(
                            "runs_clean_apply",
                            "CLEAN_APPLY_FAILED",
                            error,
                            "文件变化或运行受保护时计划会停止；重新预览当前候选。",
                        ),
                    }
                }
            }
        }
    }
}
