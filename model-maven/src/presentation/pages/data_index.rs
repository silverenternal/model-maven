use serde_json::Value;

pub fn render(data: &Value) -> Vec<String> {
    let classes = data["classes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    let mut lines = vec![format!(
        "分类数据索引 · {} 类 · {} 个样本 · {} 张图像已验证",
        classes.len(),
        data["sample_count"].as_u64().unwrap_or(0),
        data["verified_images"].as_u64().unwrap_or(0)
    )];
    lines.push(format!(
        "数据指纹：{}",
        data["dataset_fingerprint"].as_str().unwrap_or("未知")
    ));
    if let Some(counts) = data["class_counts"].as_object() {
        lines.push("各类样本：".into());
        for (label, count) in counts {
            lines.push(format!("  {label}：{}", count.as_u64().unwrap_or(0)));
        }
    }
    let excluded = data["excluded_files"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if !excluded.is_empty() {
        lines.push(format!("已排除（不删除原文件）：{} 项", excluded.len()));
        for path in excluded.iter().filter_map(Value::as_str) {
            lines.push(format!("  {path}"));
        }
    }
    if data["excluded_without_deletion"] == true {
        lines.push("排除协议已写入当前项目；分类预设批次会验证内容指纹并应用该清单。".into());
    }
    if data["exclusions_cleared"] == true {
        lines.push("已清除项目排除协议；所有复验通过的原始图像重新参与分类批次。".into());
    }
    lines
}
