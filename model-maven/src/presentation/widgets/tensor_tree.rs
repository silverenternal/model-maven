use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Default)]
pub struct TensorTreePage {
    expanded: BTreeSet<String>,
}

impl TensorTreePage {
    pub fn new(root_label: &str) -> Self {
        Self {
            expanded: [root_label.to_owned()].into_iter().collect(),
        }
    }

    pub fn render(&self, root: &Value, root_label: &str) -> String {
        let mut nodes = Vec::new();
        collect_visible(root, root_label, 0, &self.expanded, &mut nodes);
        nodes
            .iter()
            .enumerate()
            .map(|(index, (path, depth, summary, container))| {
                format!(
                    "{:>2}. {}{}{} · {}",
                    index + 1,
                    "  ".repeat(*depth),
                    if *container {
                        if self.expanded.contains(path) {
                            "▾ "
                        } else {
                            "▸ "
                        }
                    } else {
                        "• "
                    },
                    path,
                    summary
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn toggle_visible(
        &mut self,
        root: &Value,
        root_label: &str,
        visible_index: usize,
    ) -> Result<String, String> {
        let mut nodes = Vec::new();
        collect_visible(root, root_label, 0, &self.expanded, &mut nodes);
        let Some((path, _, _, is_container)) = visible_index
            .checked_sub(1)
            .and_then(|index| nodes.get(index))
        else {
            return Err("没有这个端口序号。".into());
        };
        if !is_container {
            return Err("该端口是叶节点，没有可展开的子项。".into());
        }
        let path = path.clone();
        if !self.expanded.remove(&path) {
            self.expanded.insert(path.clone());
        }
        Ok(path)
    }
}

type VisibleNode = (String, usize, String, bool);

fn collect_visible(
    value: &Value,
    path: &str,
    depth: usize,
    expanded: &BTreeSet<String>,
    nodes: &mut Vec<VisibleNode>,
) {
    let (summary, children) = node_summary(value);
    let is_container = children.is_some();
    nodes.push((path.to_owned(), depth, summary, is_container));
    if !expanded.contains(path) {
        return;
    }
    match children {
        Some(Children::Object(items)) => {
            for (key, child) in items {
                collect_visible(child, &format!("{path}.{key}"), depth + 1, expanded, nodes);
            }
        }
        Some(Children::Array(items)) => {
            for (index, child) in items.iter().enumerate() {
                collect_visible(
                    child,
                    &format!("{path}[{index}]"),
                    depth + 1,
                    expanded,
                    nodes,
                );
            }
        }
        None => {}
    }
}

enum Children<'a> {
    Object(&'a serde_json::Map<String, Value>),
    Array(&'a [Value]),
}

fn node_summary(value: &Value) -> (String, Option<Children<'_>>) {
    match value["kind"].as_str() {
        Some("tensor") => (
            format!(
                "Tensor · shape={} · dtype={} · device={}{}",
                value["shape"],
                value["dtype"],
                value["device"],
                if value["requires_grad"] == true {
                    " · requires_grad"
                } else {
                    ""
                }
            ),
            None,
        ),
        Some("dict") => {
            let items = value["items"].as_object();
            (
                format!(
                    "字典 · {} 项{}",
                    items.map_or(0, |items| items.len()),
                    truncation_suffix(value)
                ),
                items.map(Children::Object),
            )
        }
        Some("tuple") | Some("list") => {
            let items = value["items"].as_array();
            (
                format!(
                    "{} · {} 项{}",
                    value["kind"].as_str().unwrap_or("序列"),
                    items.map_or(0, |items| items.len()),
                    truncation_suffix(value)
                ),
                items.map(|items| Children::Array(items.as_slice())),
            )
        }
        Some("opaque") => (
            format!(
                "不透明对象 · {}",
                value["type"].as_str().unwrap_or("未知类型")
            ),
            None,
        ),
        Some("truncated") => ("结构已截断".into(), None),
        _ if value.is_null() => ("None".into(), None),
        _ if value.is_string() => ("字符串标量（值已隐藏）".into(), None),
        _ if value.is_boolean() => ("布尔标量（值已隐藏）".into(), None),
        _ if value.is_number() => ("数值标量（值已隐藏）".into(), None),
        _ => ("未知或不支持的节点".into(), None),
    }
}

fn truncation_suffix(value: &Value) -> String {
    if value["truncated"].as_u64().is_some_and(|count| count > 0) {
        format!(" · 另有 {} 项未展开", value["truncated"])
    } else if value["items"].get("<truncated>").is_some() {
        " · 字典内容已截断".into()
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nested_ports_expand_individually_and_show_tensor_metadata() {
        let value = json!({"kind":"dict","items":{
            "args":{"kind":"tuple","items":[{"kind":"tensor","shape":[2,4],"dtype":"torch.float32","device":"cpu","requires_grad":false}]},
            "kwargs":{"kind":"dict","items":{"mask":{"kind":"tensor","shape":[2,4],"dtype":"torch.bool","device":"cpu","requires_grad":false}}}
        }});
        let mut page = TensorTreePage::new("inputs");
        let first = page.render(&value, "inputs");
        assert!(first.contains("inputs.args"));
        assert!(first.contains("inputs.kwargs"));
        assert!(!first.contains("shape=[2,4]"));
        page.toggle_visible(&value, "inputs", 2).unwrap();
        let expanded = page.render(&value, "inputs");
        assert!(expanded.contains("inputs.args[0]"));
        assert!(expanded.contains("shape=[2,4]"));
        assert!(!expanded.contains("inputs.kwargs.mask"));
    }

    #[test]
    fn scalar_values_are_not_echoed_and_truncation_is_visible() {
        let value = json!({"kind":"dict","items":{"token":"secret-value","<truncated>":4}});
        let rendered = TensorTreePage::new("outputs").render(&value, "outputs");
        assert!(!rendered.contains("secret-value"));
        assert!(rendered.contains("已截断"));
    }
}
