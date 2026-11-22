use serde_json::Value;

pub fn render_call_list(calls: &[Value], page: usize, page_size: usize) -> String {
    let page_size = page_size.max(1);
    let pages = calls.len().div_ceil(page_size).max(1);
    let page = page.min(pages - 1);
    let start = page * page_size;
    let end = start.saturating_add(page_size).min(calls.len());
    let mut lines = vec![format!(
        "逐次调用 · 共 {} 次 · 第 {}/{} 页",
        calls.len(),
        page + 1,
        pages
    )];
    if calls.is_empty() {
        lines.push("当前批次没有观测到模块调用；不代表未执行的分支已验证。".into());
    }
    for (local_index, call) in calls.get(start..end).unwrap_or_default().iter().enumerate() {
        let ordinal = start + local_index + 1;
        let paths = call["paths"]
            .as_array()
            .map(|paths| {
                paths
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .unwrap_or_default();
        let parent = call["parent_call_id"]
            .as_u64()
            .map(|id| format!(" · 父调用 #{id}"))
            .unwrap_or_default();
        lines.push(format!(
            "{ordinal}. {} · {} · 本模块第 {} 次{}",
            call["module_type"].as_str().unwrap_or("未知模块"),
            paths,
            call["invocation_index"].as_u64().unwrap_or(0),
            parent
        ));
    }
    lines.push("输入序号查看该次输入/输出；n/p 翻页；all 依次查看本页；back 返回。".into());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn shared_module_calls_remain_separate_and_include_parent_context() {
        let calls = vec![
            json!({"module_type":"Linear","paths":["model.shared","model.text"],"invocation_index":1,"parent_call_id":2}),
            json!({"module_type":"Linear","paths":["model.shared","model.video"],"invocation_index":2,"parent_call_id":3}),
        ];
        let rendered = render_call_list(&calls, 0, 40);
        assert!(rendered.contains("共 2 次"));
        assert!(rendered.contains("model.text"));
        assert!(rendered.contains("model.video"));
        assert!(rendered.contains("本模块第 1 次"));
        assert!(rendered.contains("本模块第 2 次"));
        assert!(rendered.contains("父调用 #3"));
    }

    #[test]
    fn long_call_list_paginates_without_dropping_identity() {
        let calls = (0..83)
            .map(|index| json!({"module_type":"Linear","paths":[format!("model.layers.{index}")],"invocation_index":1}))
            .collect::<Vec<_>>();
        let second = render_call_list(&calls, 1, 40);
        assert!(second.contains("第 2/3 页"));
        assert!(second.contains("model.layers.40"));
        assert!(!second.contains("model.layers.39"));
    }
}
