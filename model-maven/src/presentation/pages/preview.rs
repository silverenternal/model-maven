use crate::{domain::Report, presentation::widgets};
use serde_json::Value;

pub fn render(report: &Report) -> Vec<String> {
    let mut lines = vec!["数据预览".into()];
    lines.extend(widgets::preview_state::render(report));
    if !report.success() {
        lines.extend(render_previous_preview(&report.data["previous_preview"]));
        lines.extend(crate::presentation::pages::diagnostic::render(report, true));
        return lines;
    }

    let scope = &report.data["scope"];
    let source_kind = match scope["source_kind"].as_str().unwrap_or("unknown") {
        "dataset" => "Dataset",
        "batch" => "批次工厂",
        _ => "来源未知",
    };
    let split = scope["split"].as_str().unwrap_or("未知");
    let count = report.data["sample_count"]
        .as_u64()
        .or_else(|| scope["actual_samples"].as_u64());
    lines.push(format!(
        "范围：{source_kind} · 切分 {split} · {}",
        count.map_or_else(
            || "样本数未知".into(),
            |actual| format!("{actual} 个实际样本")
        )
    ));
    if let Some(requested) = scope["requested_samples"].as_u64()
        && let Some(actual) = scope["actual_samples"].as_u64()
    {
        lines.push(format!(
            "采样：请求最多 {requested} 个，实际取得 {actual} 个"
        ));
    }
    if let Some(seed) = scope["seed"].as_u64() {
        lines.push(format!("随机种子：{seed}"));
    }
    if let Some(adapter) = scope["data_observation"]["adapter"].as_str() {
        lines.push(format!("数据适配器：{adapter}"));
    }

    if let Some(request_id) = report.data["request_id"].as_str() {
        let generation = report.data["generation"]
            .as_u64()
            .map_or_else(|| "未知版本".into(), |value| value.to_string());
        lines.push(format!("请求：{request_id} · generation {generation}"));
    }
    if let Some(signature) = report.data["input_signature"].as_str() {
        lines.push(format!(
            "输入签名：{}…",
            signature.chars().take(12).collect::<String>()
        ));
    }

    let artifacts = report.data["artifacts"].as_array();
    if let Some(artifacts) = artifacts.filter(|items| !items.is_empty()) {
        lines.push(artifact_heading(report).into());
        lines.extend(artifacts.iter().map(render_artifact));
    } else {
        lines.push("产物清单为空；不会假定预览文件已生成。".into());
    }
    if let Some(image_count) = report.data["rendered_image_count"].as_u64() {
        lines.push(format!("已渲染图像：{image_count}"));
    }
    lines.push("\n交付：".into());
    lines.extend(widgets::preview_delivery::render(&report.data["delivery"]));
    lines
}

fn artifact_heading(report: &Report) -> &'static str {
    if report.data["published"] == true {
        "已发布产物："
    } else if matches!(
        report.data["status"].as_str(),
        Some("superseded" | "replaced")
    ) {
        "本次请求产物（未发布为当前结果）："
    } else {
        "返回的产物引用（发布状态未确认）："
    }
}

fn render_previous_preview(previous: &Value) -> Vec<String> {
    if !previous.is_object() {
        return Vec::new();
    }
    let request_id = previous["request_id"].as_str().unwrap_or("未知请求");
    let generation = previous["generation"]
        .as_u64()
        .map_or_else(|| "未知版本".into(), |value| value.to_string());
    let mut lines = vec![format!(
        "上一版预览（已过期）· 请求 {request_id} · generation {generation}"
    )];
    if let Some(artifacts) = previous["artifacts"].as_array() {
        for artifact in artifacts {
            let kind = artifact["kind"].as_str().unwrap_or("artifact");
            let path = artifact["uri"].as_str().unwrap_or("路径未知");
            lines.push(format!("  {kind} · {path}"));
        }
    }
    lines.push("上一版文件仍保留，但内容不代表这次设置；未自动打开。".into());
    lines
}

fn render_artifact(artifact: &Value) -> String {
    let kind = artifact["kind"].as_str().unwrap_or("artifact");
    let path = artifact["uri"].as_str().unwrap_or("路径未知");
    let media_type = artifact["media_type"]
        .as_str()
        .map(|value| format!(" · {value}"))
        .unwrap_or_default();
    let size = artifact["size_bytes"]
        .as_u64()
        .map(|value| format!(" · {}", byte_size(value)))
        .unwrap_or_default();
    format!("  {kind} · {path}{media_type}{size}")
}

fn byte_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::domain::Report;
    use serde_json::json;

    #[test]
    fn preview_page_shows_scope_artifacts_freshness_and_safe_delivery() {
        let report = Report::ok(
            "preview",
            json!({
                "status":"succeeded",
                "published":true,
                "cached":false,
                "request_id":"preview-1",
                "generation":3,
                "input_signature":"1234567890abcdef",
                "scope":{"source_kind":"dataset", "split":"validation", "requested_samples":12, "actual_samples":4, "seed":7},
                "sample_count":4,
                "rendered_image_count":4,
                "artifacts":[{"kind":"preview_document", "uri":".model-maven/previews/preview-1/preview.html", "media_type":"text/html", "size_bytes":2048}],
                "delivery":{"mode":"local", "path":"/tmp/preview.html", "opened":false}
            }),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("当前预览 · 本次请求已发布"));
        assert!(text.contains("Dataset · 切分 validation · 4 个实际样本"));
        assert!(text.contains("请求最多 12 个，实际取得 4 个"));
        assert!(text.contains("preview.html · text/html · 2.0 KiB"));
        assert!(text.contains("/tmp/preview.html"));
        assert!(text.contains("尚未自动打开"));
        assert!(!text.contains("1234567890abcdef"));
    }

    #[test]
    fn empty_artifact_list_is_not_presented_as_a_generated_preview() {
        let report = Report::ok(
            "preview",
            json!({"status":"succeeded", "published":true, "artifacts":[]}),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("产物清单为空"));
    }

    #[test]
    fn failed_preview_keeps_the_previous_artifact_visible_as_stale() {
        let mut report = Report::error("preview", "RENDER_FAILED", "renderer failed", "retry");
        report.data = json!({
            "previous_result_stale":true,
            "previous_preview":{
                "request_id":"preview-old",
                "generation":4,
                "artifacts":[{"kind":"preview_document", "uri":".model-maven/previews/preview-old/preview.html"}]
            }
        });
        let text = render(&report).join("\n");
        assert!(text.contains("上一版预览已过期"));
        assert!(text.contains("preview-old · generation 4"));
        assert!(text.contains(".model-maven/previews/preview-old/preview.html"));
        assert!(text.contains("内容不代表这次设置"));
    }

    #[test]
    fn superseded_preview_artifacts_are_not_labeled_as_current_publication() {
        let report = Report::ok(
            "preview",
            json!({
                "status":"superseded",
                "published":false,
                "request_id":"preview-old",
                "generation":6,
                "current":{"request_id":"preview-new", "generation":7},
                "artifacts":[{"kind":"preview_document", "uri":"history/preview-old.html"}]
            }),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("不属于当前结果"));
        assert!(text.contains("当前请求：preview-new · 第 7 版"));
        assert!(text.contains("本次请求产物（未发布为当前结果）"));
        assert!(text.contains("history/preview-old.html"));
        assert!(!text.contains("已发布产物"));
    }

    #[test]
    fn pending_or_unknown_preview_artifact_references_are_not_claimed_published() {
        let report = Report::ok(
            "preview",
            json!({
                "status":"generating",
                "published":false,
                "artifacts":[{"kind":"preview_document", "uri":"pending/result.html"}]
            }),
        );
        let text = render(&report).join("\n");
        assert!(text.contains("正在生成预览"));
        assert!(text.contains("发布状态未确认"));
        assert!(!text.contains("已发布产物"));
    }
}
