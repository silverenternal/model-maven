use crate::presentation::pages::model_tree::ModelTreePage;
use serde_json::Value;

pub fn render_model_tree(rows: &[Value], page: &ModelTreePage) -> String {
    let visible = page.current_page(rows);
    let total = page.visible_rows(rows).len();
    let model_parameters = rows
        .iter()
        .find(|row| row["path"] == "model")
        .and_then(|row| row["parameters"].as_u64());
    let mut lines = vec![format!(
        "模型树 · {} 个匹配节点 · 当前显示 {} 个节点{}{}",
        page.match_count(rows),
        total,
        model_parameters
            .map(|count| format!(" · 模型参数总数 {count}（后端统计，未按共享别名重复相加）"))
            .unwrap_or_default(),
        if page.query().is_empty() {
            String::new()
        } else {
            format!(" · 搜索：{}", page.query())
        }
    )];
    if visible.is_empty() {
        lines.push("没有匹配的 Layer；可按路径、类型、别名或可改字段搜索。".into());
    }
    for (index, row) in visible.iter().enumerate() {
        let path = row["path"].as_str().unwrap_or("<无路径>");
        let depth = path.matches('.').count();
        let shared = row["aliases"]
            .as_array()
            .is_some_and(|aliases| aliases.len() > 1);
        let direct_parameters = row["own_parameters"]
            .as_u64()
            .map(|count| format!(" · 本层参数 {count}"))
            .unwrap_or_default();
        let marker = if row["can_insert"] == true || row["can_source_insert"] == true {
            "▾"
        } else {
            "•"
        };
        lines.push(format!(
            "{:>2}. {}{} {} · {}{}{}",
            index + 1,
            "  ".repeat(depth),
            marker,
            path,
            row["type"].as_str().unwrap_or("未知类型"),
            direct_parameters,
            if shared { " · 共享实例" } else { "" }
        ));
    }
    lines.push(format!(
        "第 {}/{} 页 · 输入序号查看详情，路径/类型/别名搜索，n/p 翻页，c <路径> 折叠/展开，r 刷新，back 返回。",
        page.page() + 1,
        page.page_count(rows)
    ));
    lines.join("\n")
}

pub fn toggle_path(page: &mut ModelTreePage, rows: &[Value], path: &str) -> Result<bool, String> {
    let Some(row) = rows.iter().find(|row| row["path"] == path) else {
        return Err(format!("找不到模型节点：{path}"));
    };
    if row["can_insert"] != true && row["can_source_insert"] != true {
        return Err(format!("{path} 没有可折叠的子容器。"));
    }
    Ok(page.toggle(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tree_shows_complete_path_shared_state_and_unique_model_total() {
        let rows = vec![
            json!({"path":"model","type":"Sequential","parameters":8,"can_insert":true}),
            json!({"path":"model.shared","type":"Linear","own_parameters":8,"aliases":["model.shared","model.head"],"instance_id":"s1"}),
        ];
        let tree = render_model_tree(&rows, &ModelTreePage::default());
        assert!(tree.contains("模型参数总数 8"));
        assert!(tree.contains("model.shared"));
        assert!(tree.contains("共享实例"));
        assert!(tree.contains("本层参数 8"));
    }

    #[test]
    fn cannot_collapse_a_leaf_as_if_it_were_a_container() {
        let rows = vec![json!({"path":"model.layer","type":"Linear"})];
        let mut page = ModelTreePage::default();
        assert!(toggle_path(&mut page, &rows, "model.layer").is_err());
        assert!(toggle_path(&mut page, &rows, "missing").is_err());
    }
}
