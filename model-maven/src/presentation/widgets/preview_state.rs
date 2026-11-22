use crate::domain::Report;

/// Present only preview freshness that the request lifecycle actually reports.
pub fn render(report: &Report) -> Vec<String> {
    let data = &report.data;
    let mut lines = Vec::new();

    if !report.success() {
        lines.push("预览失败 · 本次请求没有生成可确认的新结果。".into());
        if data["previous_result_stale"] == true {
            lines.push("上一版预览已过期；不能代表当前输入。".into());
        } else if data["current"].is_object() {
            lines.push("项目中仍有上一版结果；请查看其请求身份后再使用。".into());
        } else {
            lines.push("上一版结果状态未知；不会把旧产物当作当前结果。".into());
        }
        return lines;
    }

    match data["status"].as_str().unwrap_or("unknown") {
        "succeeded" if data["published"] == true => {
            if data["cached"] == true {
                lines.push("当前预览 · 复用已验证结果".into());
            } else {
                lines.push("当前预览 · 本次请求已发布".into());
            }
            if data["previous_result_stale"] == true {
                lines.push("上一版预览已过期；以下产物对应当前已提交输入。".into());
            }
        }
        "superseded" | "replaced" => {
            lines.push("本次请求已被较新的预览取代 · 不属于当前结果".into());
            if let Some(current) = data["current"].as_object() {
                let request_id = current
                    .get("request_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("未知请求");
                let generation = current
                    .get("generation")
                    .and_then(serde_json::Value::as_u64)
                    .map_or_else(|| "未知版本".into(), |value| format!("第 {value} 版"));
                lines.push(format!("当前请求：{request_id} · {generation}"));
            }
        }
        "running" | "queued" | "generating" => {
            lines.push("正在生成预览 · 尚无新结果可用".into());
        }
        "failed" => {
            lines.push("预览失败 · 当前结果保持不变".into());
        }
        "cancelled" | "canceled" => {
            lines.push("预览已取消 · 当前结果保持不变".into());
        }
        _ => {
            lines.push("预览状态未知 · 不推断当前结果或上一版是否有效".into());
        }
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::domain::Report;
    use serde_json::json;

    #[test]
    fn published_cached_and_superseded_results_have_distinct_freshness_labels() {
        let cached = Report::ok(
            "preview",
            json!({"status":"succeeded", "published":true, "cached":true}),
        );
        assert!(render(&cached)[0].contains("复用已验证结果"));

        let superseded = Report::ok(
            "preview",
            json!({
                "status":"superseded",
                "published":false,
                "current":{"request_id":"preview-new", "generation":7}
            }),
        );
        let text = render(&superseded).join("\n");
        assert!(text.contains("不属于当前结果"));
        assert!(text.contains("preview-new · 第 7 版"));
        assert!(!text.contains("当前预览 ·"));
    }

    #[test]
    fn failed_preview_never_claims_the_old_result_is_current() {
        let mut report = Report::error("preview", "RENDER_FAILED", "renderer failed", "retry");
        report.data = json!({"previous_result_stale":true});
        let text = render(&report).join("\n");
        assert!(text.contains("预览失败"));
        assert!(text.contains("上一版预览已过期"));
        assert!(!text.contains("当前预览 ·"));
    }
}
