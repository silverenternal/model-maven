use crate::domain::Report;

/// Human-oriented summary of an export report. It reports only facts returned
/// by the exporter and does not imply that omitted external resources exist.
pub fn render(report: &Report) -> Vec<String> {
    let data = &report.data;
    let mut lines = if report.success() {
        vec!["导出完成".to_owned()]
    } else {
        vec!["导出未完成".to_owned()]
    };
    if let Some(run) = data["run_id"].as_str() {
        lines.push(format!("来源运行：{run}"));
    }
    if let Some(path) = data["path"].as_str() {
        lines.push(format!("导出位置：{path}"));
    }
    if let Some(steps) = data["checkpoint_completed_steps"].as_u64() {
        lines.push(format!("恢复权重：第 {steps} 步 checkpoint"));
    }
    let source_count = data["source_file_count"].as_u64().unwrap_or(0);
    let extension_count = data["extension_package_count"].as_u64().unwrap_or(0);
    let dependency_count = data["dependency_package_count"].as_u64().unwrap_or(0);
    if data["file_count"].is_number() {
        lines.push(format!(
            "已校验文件：{} 个 · {} 字节",
            data["file_count"].as_u64().unwrap_or(0),
            data["verified_bytes"].as_u64().unwrap_or(0)
        ));
        lines.push(format!(
            "包含：{source_count} 个源码文件、{extension_count} 个已登记扩展包、{dependency_count} 个扩展运行依赖包。"
        ));
    }
    match data["environment_snapshot_status"].as_str() {
        Some("captured") => lines.push(format!(
            "环境快照：已采集（{}）· {} 个 distribution",
            data["environment_snapshot_timing"]
                .as_str()
                .map_or("采集时点未知", timing_label),
            data["environment_distribution_count"].as_u64().unwrap_or(0)
        )),
        Some(status) => lines.push(format!("环境快照：{}", status_label(status))),
        None => lines.push("环境快照：报告未提供".into()),
    }
    if let Some(backup) = data["backup_path"].as_str() {
        lines.push(format!("原目标已备份到：{backup}"));
    }
    lines.push("导出目录包含源码、变体（如有）、恢复权重及运行环境说明；可查看其中 README.md 了解恢复步骤。".into());
    lines.push("需要外部准备：数据集、未登记包、私有索引配置、操作系统共享库及加速器驱动。".into());
    if report.operation == "export" && report.success() {
        lines.push("PyTorch 作为外部运行时依赖记录；导出会校验已归档文件的 SHA-256。".into());
    }
    for diagnostic in &report.diagnostics {
        lines.push(format!("{}：{}", diagnostic.code, diagnostic.message));
        if let Some(hint) = diagnostic.hint.as_deref().filter(|hint| !hint.is_empty()) {
            lines.push(format!("下一步：{hint}"));
        }
    }
    lines
}

fn timing_label(timing: &str) -> &'static str {
    match timing {
        "training_start" => "训练开始时",
        "export_time" => "导出时补采",
        _ => "采集时点未知",
    }
}

fn status_label(status: &str) -> &'static str {
    match status {
        "unavailable" => "不可用",
        "runtime_mismatch" => "运行环境不匹配",
        "captured" => "已采集",
        _ => "未确认",
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::domain::Report;
    use serde_json::json;

    #[test]
    fn export_summary_shows_origin_destination_and_external_requirements() {
        let report = Report::ok(
            "export",
            json!({
                "path":"/tmp/model-export",
                "run_id":"run-42",
                "checkpoint_completed_steps":8,
                "source_file_count":3,
                "extension_package_count":1,
                "dependency_package_count":2,
                "file_count":14,
                "verified_bytes":4096,
                "environment_snapshot_status":"captured",
                "environment_snapshot_timing":"training_start",
                "environment_distribution_count":27,
                "backup_path":"/tmp/.model-export.backup"
            }),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("来源运行：run-42"));
        assert!(text.contains("导出位置：/tmp/model-export"));
        assert!(text.contains("3 个源码文件、1 个已登记扩展包"));
        assert!(text.contains("数据集、未登记包、私有索引配置"));
        assert!(text.contains("原目标已备份"));
    }

    #[test]
    fn failed_export_preserves_service_diagnostic_and_recovery_hint() {
        let report = Report::error(
            "export",
            "EXPORT_CONFLICT",
            "目标目录已有文件",
            "指定 --overwrite 后重试。",
        );
        let text = render(&report).join("\n");
        assert!(text.contains("导出未完成"));
        assert!(text.contains("目标目录已有文件"));
        assert!(text.contains("指定 --overwrite 后重试"));
    }
}
