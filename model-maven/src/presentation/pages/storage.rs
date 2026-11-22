use serde_json::Value;

/// Render a `runs storage` service result without double-counting its reported
/// deduplicated files. Byte totals remain explicitly reported measurements.
pub fn render_storage(data: &Value) -> Vec<String> {
    let runs = data["runs"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let mut lines = vec![format!("运行记录占用 · {} 条记录", runs.len())];
    lines.push(format!(
        "合计：{} · {} 个硬链接/共享文件去重",
        bytes(data["total_bytes"].as_u64().unwrap_or(0)),
        data["deduplicated_file_count"].as_u64().unwrap_or(0)
    ));
    let categories = &data["categories"];
    for (key, label) in [
        ("metadata", "运行记录与配置"),
        ("logs", "日志"),
        ("progress", "进度记录"),
        ("checkpoints", "恢复权重"),
        ("artifacts", "登记产物"),
    ] {
        lines.push(format!(
            "{label}：{}",
            bytes(categories[key].as_u64().unwrap_or(0))
        ));
    }
    for run in runs {
        let id = run["id"].as_str().unwrap_or("未知 ID");
        let name = run["name"]
            .as_str()
            .filter(|name| !name.is_empty())
            .unwrap_or("未命名");
        let status = status_label(run["status"].as_str().unwrap_or("unknown"));
        let mut protections = Vec::new();
        if run["kept"] == true {
            protections.push("已标记保留".to_owned());
        }
        if matches!(
            run["status"].as_str(),
            Some("running" | "stopping" | "unknown")
        ) {
            protections.push("运行状态受保护，不能清理".to_owned());
        }
        lines.push(format!(
            "{name} · {id} · {status} · {} · {} 个文件",
            bytes(run["total_bytes"].as_u64().unwrap_or(0)),
            run["file_count"].as_u64().unwrap_or(0)
        ));
        if !protections.is_empty() {
            lines.push(format!("  保护：{}", protections.join("；")));
        }
        if run["checkpoint_released"] == true {
            lines.push("  恢复权重已释放；此运行无法续训或导出权重。".into());
        }
        if let Some(skipped) = run["skipped_symlinks"]
            .as_array()
            .filter(|items| !items.is_empty())
        {
            lines.push(format!("  已跳过符号链接：{} 项", skipped.len()));
        }
    }
    match data["filesystem_space"]["status"].as_str() {
        Some("available") => lines.push(format!(
            "运行目录所在文件系统：剩余 {} / 总计 {}",
            bytes(
                data["filesystem_space"]["available_bytes"]
                    .as_u64()
                    .unwrap_or(0)
            ),
            bytes(
                data["filesystem_space"]["total_bytes"]
                    .as_u64()
                    .unwrap_or(0)
            )
        )),
        Some(_) => lines.push(format!(
            "运行目录所在文件系统：可用空间未知{}",
            data["filesystem_space"]["reason"]
                .as_str()
                .map(|reason| format!("（{reason}）"))
                .unwrap_or_default()
        )),
        None => lines.push("运行目录所在文件系统：可用空间未检测".into()),
    }
    if let Some(scope) = data["scope"].as_str() {
        lines.push(format!("范围：{scope}"));
    }
    lines.push("运行文件占用仅涵盖服务扫描的运行目录；文件系统剩余空间是所在文件系统的整体值。共享文件按文件身份去重。".into());
    lines
}

/// Render the exact immutable cleanup plan returned by the run service.
/// This is review-only: callers must apply the returned plan `id` unchanged.
pub fn render_clean_plan(plan: &Value) -> Vec<String> {
    let items = plan["items"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let mut lines = vec!["清理计划预览 · 尚未执行".to_owned()];
    if let Some(id) = plan["id"].as_str() {
        lines.push(format!("计划 ID：{id}"));
    }
    if let Some(run_id) = plan["run_id"].as_str() {
        lines.push(format!("限定运行：{run_id}"));
    }
    lines.push(format!(
        "候选：{} 项 · 最多可释放 {}",
        items.len(),
        bytes(plan["total_bytes"].as_u64().unwrap_or(0))
    ));
    let releases_checkpoint = items.iter().any(|item| {
        item["release_checkpoint"] == true
            || item["kind"] == "latest_checkpoint_release"
            || item["kind"] == "managed_checkpoint_release"
    });
    if releases_checkpoint {
        lines.push("重要：本计划会移除恢复权重；相应运行将失去续训和权重导出能力。".into());
    }
    if items.is_empty() {
        lines.push("没有可执行候选；不会删除任何文件。".into());
    }
    for item in items {
        let target = item["path"].as_str().unwrap_or("未知路径");
        let run_id = item["run_id"].as_str().unwrap_or("未知运行");
        lines.push(format!(
            "{} · {} · {} · {}",
            item["item_id"].as_str().unwrap_or("未编号"),
            run_id,
            target,
            bytes(item["size_bytes"].as_u64().unwrap_or(0))
        ));
        if let Some(reason) = item["reason"].as_str() {
            lines.push(format!("  后果/原因：{reason}"));
        }
    }
    if let Some(skipped) = plan["skipped"].as_array().filter(|items| !items.is_empty()) {
        lines.push(format!("保持保护/跳过：{} 项", skipped.len()));
        for item in skipped.iter().take(8) {
            lines.push(format!(
                "  {} · {} · {}",
                item["run_id"].as_str().unwrap_or("运行未知"),
                item["path"].as_str().unwrap_or("未提供路径"),
                item["reason"].as_str().unwrap_or("服务未提供原因")
            ));
        }
    }
    lines.push(
        "只有再次确认后，才能把此固定计划 ID 交给服务执行；计划不会因页面选择而扩展。".into(),
    );
    lines
}

/// Render an apply result from the service, including partial failures.
pub fn render_clean_result(result: &Value) -> Vec<String> {
    let status = result["status"].as_str().unwrap_or("unknown");
    let has_skipped = result["apply_skipped"]
        .as_array()
        .is_some_and(|items| !items.is_empty());
    let mut lines = vec![match status {
        "applied" | "completed" if has_skipped => "清理部分完成；有目标未执行".to_owned(),
        "applied" | "completed" => "清理计划已执行".to_owned(),
        "partial_release" => "清理部分完成".to_owned(),
        "rejected" | "stale" => "清理计划未执行".to_owned(),
        _ => "清理结果待确认".to_owned(),
    }];
    if let Some(id) = result["plan_id"].as_str().or_else(|| result["id"].as_str()) {
        lines.push(format!("计划 ID：{id}"));
    }
    let applied = result["applied_items"].as_array();
    if let Some(applied) = applied {
        lines.push(format!("已处理 {} 项。", applied.len()));
        for item in applied {
            if item["release_checkpoint"] == true {
                lines.push(format!(
                    "运行 {} 的恢复权重已释放；不能续训或导出权重。",
                    item["run_id"].as_str().unwrap_or("未知")
                ));
            }
        }
    }
    for key in ["apply_skipped", "apply_warnings"] {
        if let Some(entries) = result[key].as_array().filter(|entries| !entries.is_empty()) {
            lines.push(format!("{}：{} 项", key_label(key), entries.len()));
            for entry in entries.iter().take(5) {
                lines.push(format!("  {}", item_description(entry)));
            }
        }
    }
    if let Some(message) = result["message"]
        .as_str()
        .or_else(|| result["note"].as_str())
    {
        lines.push(message.into());
    }
    lines
}

fn key_label(key: &str) -> &'static str {
    match key {
        "apply_skipped" => "未执行项（目标变化或受保护）",
        _ => "诊断",
    }
}

fn item_description(item: &Value) -> String {
    let path = item["path"].as_str().unwrap_or("未提供路径");
    let reason = item["reason"]
        .as_str()
        .or_else(|| item["message"].as_str())
        .unwrap_or("服务未提供原因");
    format!("{path} · {reason}")
}

fn status_label(status: &str) -> &'static str {
    match status {
        "running" => "运行中",
        "stopping" => "正在停止",
        "completed" => "已完成",
        "failed" => "失败待处理",
        "interrupted" => "已中断",
        "unknown" => "状态未知",
        _ => "未知状态",
    }
}

fn bytes(value: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut amount = value as f64;
    let mut unit = 0;
    while amount >= 1024.0 && unit < UNITS.len() - 1 {
        amount /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value} B")
    } else {
        format!("{amount:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::{render_clean_plan, render_clean_result, render_storage};
    use serde_json::json;

    #[test]
    fn storage_explains_categories_protection_and_scan_limits() {
        let text = render_storage(&json!({
            "total_bytes":4096,
            "deduplicated_file_count":2,
            "filesystem_space":{"status":"available","available_bytes":1073741824_u64,"total_bytes":2147483648_u64},
            "categories":{"metadata":100,"logs":20,"progress":30,"checkpoints":3900,"artifacts":46},
            "scope":"仅扫描运行目录",
            "runs":[{"id":"r1","name":"baseline","status":"unknown","kept":true,"total_bytes":4096,"file_count":3,"skipped_symlinks":["x"]}]
        })).join("\n");
        assert!(text.contains("恢复权重：3.8 KiB"));
        assert!(text.contains("已标记保留"));
        assert!(text.contains("运行状态受保护"));
        assert!(text.contains("已跳过符号链接"));
        assert!(text.contains("运行目录所在文件系统：剩余 1.0 GiB / 总计 2.0 GiB"));
    }

    #[test]
    fn storage_marks_filesystem_capacity_unknown_when_probe_fails() {
        let text = render_storage(&json!({
            "filesystem_space":{"status":"unknown","reason":"permission denied"},
            "runs":[]
        }))
        .join("\n");
        assert!(text.contains("可用空间未知（permission denied）"));
    }

    #[test]
    fn checkpoint_plan_is_review_only_and_calls_out_lost_recovery() {
        let text = render_clean_plan(&json!({
            "id":"clean-fixed", "run_id":"r1", "total_bytes":50,
            "items":[{"item_id":"item-0001","run_id":"r1","path":"checkpoints/latest.pt","size_bytes":50,"release_checkpoint":true,"reason":"移除后将失去恢复权重"}],
            "skipped":[{"run_id":"r2","path":"latest.pt","reason":"运行仍受保护"}]
        })).join("\n");
        assert!(text.contains("尚未执行"));
        assert!(text.contains("clean-fixed"));
        assert!(text.contains("失去续训和权重导出能力"));
        assert!(text.contains("不会因页面选择而扩展"));
        assert!(text.contains("运行仍受保护"));
    }

    #[test]
    fn result_reports_partial_filesystem_failure_as_partial() {
        let text = render_clean_result(&json!({
            "status":"partial_release",
            "plan_id":"clean-1",
            "apply_skipped":[{"path":"checkpoints/a.pt","reason":"文件内容与预览时不同"}]
        }))
        .join("\n");
        assert!(text.contains("清理部分完成"));
        assert!(text.contains("文件内容与预览时不同"));
    }
}
