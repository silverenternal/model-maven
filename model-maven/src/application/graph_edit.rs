//! Read-only, all-or-nothing graph edit batches using stable identifiers.
//! Saving the resulting draft is a separate conditional graph-save operation.
use super::*;
use serde::Deserialize;
use std::io::Write as _;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Port { node: String, port: String }
impl Port {
    fn value(&self) -> Value { json!({"node":self.node,"port":self.port}) }
}

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum Edit {
    AddNode { node: Value },
    UpdateNode { id: String, changes: Map<String, Value> },
    CopyNode { id: String, new_id: String },
    RemoveNode { id: String, #[serde(default)] cascade: bool },
    AddDefinition { definition: Value },
    UpdateDefinition { id: String, changes: Map<String, Value> },
    RemoveDefinition { id: String },
    AddGroup { group: Value },
    SetGroups { groups: Option<Vec<Value>> },
    UpdateGraph { changes: Map<String, Value> },
    AddObjective { objective: Value },
    SetParameter { definition: String, name: String, value: Value },
    RemoveParameter { definition: String, name: String },
    Connect { from: Port, to: Port },
    Disconnect { from: Port, to: Port },
    MoveBefore { node: String, anchor: String },
    MoveAfter { node: String, anchor: String },
}

fn index(items: &[Value], id: &str) -> Result<usize, String> {
    let mut found = items.iter().enumerate().filter(|(_, item)| item["id"].as_str() == Some(id));
    let first = found.next().map(|(index, _)| index).ok_or_else(|| format!("找不到 ID：{id}"))?;
    if found.next().is_some() { return Err(format!("ID 不唯一：{id}")); }
    Ok(first)
}
fn list<'a>(graph: &'a mut Value, key: &str) -> Result<&'a mut Vec<Value>, String> {
    graph.get_mut(key).and_then(Value::as_array_mut).ok_or_else(|| format!("图缺少数组：{key}"))
}
fn insert(items: &mut Vec<Value>, item: Value) -> Result<String, String> {
    let id = item.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()).ok_or("新增对象缺少 ID。")?.to_owned();
    if items.iter().any(|value| value["id"].as_str() == Some(&id)) { return Err(format!("ID 已存在：{id}")); }
    items.push(item);
    Ok(id)
}
fn patch(items: &mut [Value], id: &str, changes: Map<String, Value>, allowed: &[&str]) -> Result<(), String> {
    let position = index(items, id)?;
    if changes.is_empty() || changes.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err("修改字段为空或不受支持；稳定 ID 与节点类型不能直接改写。".into());
    }
    let object = items[position].as_object_mut().ok_or("图对象无效。")?;
    for (key, value) in changes {
        if value.is_null() { object.remove(&key); } else { object.insert(key, value); }
    }
    Ok(())
}
// Include calls to containers that transitively own this definition.
fn affected_nodes(graph: &Value, id: &str) -> usize {
    let mut affected = std::collections::HashSet::from([id.to_owned()]);
    loop {
        let mut changed = false;
        for definition in graph["definitions"].as_array().into_iter().flatten() {
            let Some(parent) = definition["id"].as_str() else { continue };
            if !affected.contains(parent) && definition["children"].as_array().is_some_and(|children|
                children.iter().any(|child| child.as_str().is_some_and(|value| affected.contains(value)))) {
                affected.insert(parent.to_owned()); changed = true;
            }
        }
        if !changed { break; }
    }
    graph["nodes"].as_array().map_or(0, |nodes| nodes.iter().filter(|node|
        node["definition"].as_str().is_some_and(|value| affected.contains(value)) ||
        node["dtype_definitions"].as_array().is_some_and(|items| items.iter().any(|item|
            item.as_str().is_some_and(|value| affected.contains(value))))).count())
}
fn temporary(graph: &Value) -> Result<(tempfile::NamedTempFile, String), String> {
    let bytes = serde_json::to_vec(graph).map_err(|error| error.to_string())?;
    if bytes.len() > 128 * 1024 { return Err("编辑结果超过 128 KiB。".into()); }
    let mut file = tempfile::NamedTempFile::new().map_err(|error| error.to_string())?;
    file.write_all(&bytes).map_err(|error| error.to_string())?;
    Ok((file, super::scaffold::sha256_bytes(&bytes)))
}
fn move_chain(graph: &mut Value, node: String, anchor: String, after: bool) -> Result<Value, String> {
    let (file, digest) = temporary(graph)?;
    let report = super::graph::reorder_plan(file.path(), &digest, &node, &anchor, after);
    if !report.success() {
        return Err(report.diagnostics.first().map(|item| item.message.clone()).unwrap_or_else(|| "无法重排节点。".into()));
    }
    *graph = report.data["graph"].clone();
    Ok(json!({"action":if after {"move_after"} else {"move_before"},"node":node,"anchor":anchor,
        "dtype_sequence_updates":report.data["dtype_sequence_updates"]}))
}
fn apply(graph: &mut Value, edit: Edit) -> Result<Value, String> {
    match edit {
        Edit::AddNode { node } => Ok(json!({"action":"add_node","id":insert(list(graph, "nodes")?, node)?})),
        Edit::UpdateNode { id, changes } => {
            patch(list(graph, "nodes")?, &id, changes, &["label", "definition", "dtype_definitions", "track_sequence", "output_name", "axis", "axes", "width", "call_style", "output_style", "inputs", "outputs", "input_shapes", "output_shapes", "input_dtypes", "output_dtypes", "semantics", "draft_rules"])?;
            Ok(json!({"action":"update_node","id":id}))
        },
        Edit::CopyNode { id, new_id } => {
            let nodes = list(graph, "nodes")?;
            let mut copy = nodes[index(nodes, &id)?].clone();
            let object = copy.as_object_mut().ok_or("节点无效。")?;
            object.insert("id".into(), json!(new_id));
            if object.get("kind").and_then(Value::as_str) == Some("output") { object.remove("output_name"); }
            let new_id = insert(nodes, copy)?;
            Ok(json!({"action":"copy_node","id":id,"new_id":new_id,"copies_edges":false,"shares_definition":true}))
        },
        Edit::RemoveNode { id, cascade } => {
            index(graph["nodes"].as_array().ok_or("图缺少节点数组。")?, &id)?;
            let connections = graph["edges"].as_array().ok_or("图缺少边数组。")?.iter().filter(|edge| edge["from"]["node"] == id || edge["to"]["node"] == id).count();
            let objectives = graph.get("objectives").and_then(Value::as_array).map_or(0, |items| items.iter().filter(|item| item["output"] == id).count());
            let groups = graph.get("groups").and_then(Value::as_array).map_or(0, |items| items.iter().filter(|group| group["nodes"].as_array().is_some_and(|nodes| nodes.iter().any(|node| node == &id))).count());
            if !cascade && connections + objectives + groups > 0 { return Err(format!("节点 {id} 有 {connections} 条连接、{objectives} 个损失和 {groups} 个分组引用；须显式 cascade。")); }
            list(graph, "edges")?.retain(|edge| edge["from"]["node"] != id && edge["to"]["node"] != id);
            if let Some(items) = graph.get_mut("objectives").and_then(Value::as_array_mut) { items.retain(|item| item["output"] != id); }
            if let Some(items) = graph.get_mut("groups").and_then(Value::as_array_mut) {
                for group in items { if let Some(nodes) = group.get_mut("nodes").and_then(Value::as_array_mut) { nodes.retain(|node| node != &id); } }
            }
            let nodes = list(graph, "nodes")?; let position = index(nodes, &id)?; nodes.remove(position);
            Ok(json!({"action":"remove_node","id":id,"removed_edges":connections,"removed_objectives":objectives,"affected_groups":groups}))
        },
        Edit::AddDefinition { definition } => Ok(json!({"action":"add_definition","id":insert(list(graph, "definitions")?, definition)?})),
        Edit::UpdateDefinition { id, changes } => {
            patch(list(graph, "definitions")?, &id, changes, &["factory", "params", "children", "children_style"])?;
            let affected = affected_nodes(graph, &id);
            Ok(json!({"action":"update_definition","id":id,"affected_nodes":affected}))
        },
        Edit::RemoveDefinition { id } => {
            if graph["nodes"].as_array().is_some_and(|nodes| nodes.iter().any(|node| node["definition"] == id || node["dtype_definitions"].as_array().is_some_and(|items| items.iter().any(|item| item == &id)))) {
                return Err(format!("组件 {id} 仍被节点引用；先移除或更改全部引用。"));
            }
            if graph["definitions"].as_array().is_some_and(|items| items.iter().any(|item| item["children"].as_array().is_some_and(|children| children.iter().any(|child| child == &id)))) {
                return Err(format!("组件 {id} 仍被容器引用；先移除全部容器引用。"));
            }
            let definitions = list(graph, "definitions")?; let position = index(definitions, &id)?; definitions.remove(position);
            Ok(json!({"action":"remove_definition","id":id}))
        },
        Edit::AddGroup { group } => {
            if graph.get("groups").is_none() { graph["groups"] = json!([]); }
            Ok(json!({"action":"add_group","id":insert(list(graph, "groups")?, group)?}))
        },
        Edit::SetGroups { groups } => {
            let count = groups.as_ref().map_or(0, Vec::len);
            if count > 512 { return Err("分组数量超过上限。".into()); }
            let object = graph.as_object_mut().ok_or("图声明无效。")?;
            if let Some(groups) = groups { object.insert("groups".into(), Value::Array(groups)); }
            else { object.remove("groups"); }
            Ok(json!({"action":"set_groups","groups":count}))
        },
        Edit::UpdateGraph { changes } => {
            let allowed = ["data", "training", "objectives", "supervision", "component_storage"];
            if changes.is_empty() || changes.keys().any(|key| !allowed.contains(&key.as_str())) {
                return Err("图设置修改为空或包含不受支持的字段。".into());
            }
            let fields = changes.keys().cloned().collect::<Vec<_>>();
            let object = graph.as_object_mut().ok_or("图声明无效。")?;
            for (key, value) in changes {
                if value.is_null() { object.remove(&key); } else { object.insert(key, value); }
            }
            Ok(json!({"action":"update_graph","fields":fields}))
        },
        Edit::AddObjective { objective } => {
            if !objective.is_object() { return Err("损失声明必须是对象。".into()); }
            if graph.get("objectives").is_none() { graph["objectives"] = json!([]); }
            let output = objective.get("output").and_then(Value::as_str).ok_or("损失声明缺少输出。")?.to_owned();
            list(graph, "objectives")?.push(objective);
            Ok(json!({"action":"add_objective","output":output}))
        },
        Edit::SetParameter { definition, name, value } => {
            if !python_identifier(&name) { return Err("参数名必须是 Python 标识符。".into()); }
            let defs = list(graph, "definitions")?; let position = index(defs, &definition)?;
            let item = defs[position].as_object_mut().ok_or("组件定义无效。")?;
            let params = item.entry("params").or_insert_with(|| json!({})).as_object_mut().ok_or("组件参数不是对象。")?;
            params.insert(name.clone(), value);
            let affected = affected_nodes(graph, &definition);
            Ok(json!({"action":"set_parameter","definition":definition,"name":name,"affected_nodes":affected}))
        },
        Edit::RemoveParameter { definition, name } => {
            let defs = list(graph, "definitions")?; let position = index(defs, &definition)?;
            let params = defs[position].get_mut("params").and_then(Value::as_object_mut).ok_or("组件没有参数。")?;
            if params.remove(&name).is_none() { return Err(format!("参数不存在：{name}")); }
            let affected = affected_nodes(graph, &definition);
            Ok(json!({"action":"remove_parameter","definition":definition,"name":name,"affected_nodes":affected}))
        },
        Edit::Connect { from, to } | Edit::Disconnect { from, to } => {
            // The variant is encoded in the original action before this arm;
            // connection removal and addition are handled in separate arms below.
            let _ = (from, to);
            Err("内部操作分派错误。".into())
        },
        Edit::MoveBefore { node, anchor } | Edit::MoveAfter { node, anchor } => {
            let _ = (node, anchor);
            Err("内部操作分派错误。".into())
        },
    }
}

pub(super) fn plan(path: &Path, expected: &str, operations: &Path, expected_operations: &str) -> Report {
    let fail = |code, message: String| Report::error("graph_edit_plan", code, message, "图和操作文件均未修改；修正后重新预览。 ");
    let bytes = match super::scaffold::read_recipe_for_update(path) { Ok(bytes) => bytes, Err(error) => return fail("GRAPH_READ_FAILED", error) };
    if super::scaffold::sha256_bytes(&bytes) != expected { return fail("GRAPH_CONFLICT", "图草稿已变化。".into()); }
    let operation_bytes = match super::scaffold::read_recipe_for_update(operations) { Ok(bytes) => bytes, Err(error) => return fail("GRAPH_OPERATIONS_READ_FAILED", error) };
    if super::scaffold::sha256_bytes(&operation_bytes) != expected_operations { return fail("GRAPH_OPERATIONS_CONFLICT", "操作文件已变化。".into()); }
    let edits: Vec<Edit> = match serde_json::from_slice(&operation_bytes) {
        Ok(edits) => edits, Err(error) => return fail("GRAPH_OPERATIONS_INVALID", error.to_string()),
    };
    if edits.is_empty() || edits.len() > 8192 { return fail("GRAPH_OPERATIONS_INVALID", "需要 1 至 8192 项操作，且操作文件不能超过大小限制。".into()); }
    let mut graph: Value = match serde_json::from_slice(&bytes) { Ok(graph) => graph, Err(error) => return fail("GRAPH_INVALID", error.to_string()) };
    if !graph.is_object() { return fail("GRAPH_INVALID", "图声明必须是对象。".into()); }
    if graph.get("definitions").is_none() { graph["definitions"] = json!([]); }
    let mut outcomes = Vec::new();
    for (index, edit) in edits.into_iter().enumerate() {
        let result = match edit {
            Edit::Connect { from, to } => {
                let edge = json!({"from":from.value(),"to":to.value()});
                match list(&mut graph, "edges") {
                    Ok(edges) if edges.contains(&edge) => Err("连接已存在。".to_owned()),
                    Ok(edges) => { edges.push(edge); Ok(json!({"action":"connect","from":from.value(),"to":to.value()})) },
                    Err(error) => Err(error),
                }
            },
            Edit::Disconnect { from, to } => {
                let edge = json!({"from":from.value(),"to":to.value()});
                match list(&mut graph, "edges") {
                    Ok(edges) => match edges.iter().position(|value| value == &edge) {
                        Some(position) => { edges.remove(position); Ok(json!({"action":"disconnect","from":from.value(),"to":to.value()})) },
                        None => Err("连接不存在。".to_owned()),
                    },
                    Err(error) => Err(error),
                }
            },
            Edit::MoveBefore { node, anchor } => move_chain(&mut graph, node, anchor, false),
            Edit::MoveAfter { node, anchor } => move_chain(&mut graph, node, anchor, true),
            other => apply(&mut graph, other),
        };
        match result { Ok(value) => outcomes.push(value), Err(error) => return fail("GRAPH_EDIT_INVALID", format!("第 {} 项操作：{error}", index + 1)) }
    }
    let (file, digest) = match temporary(&graph) { Ok(value) => value, Err(error) => return fail("GRAPH_INVALID", error) };
    let validation = super::graph::plan(file.path(), &digest);
    let repairable = validation.diagnostics.first().is_some_and(|item|
        matches!(item.code.as_str(), "GRAPH_SHAPE_CONFLICT" | "GRAPH_DTYPE_ORDER_CONFLICT" | "GRAPH_AXIS_INVALID"));
    if !validation.success() && !repairable {
        let mut result = validation;
        result.operation = "graph_edit_plan".into();
        if !result.data.is_object() { result.data = json!({}); }
        result.data["input_sha256"] = json!(expected);
        result.data["operations_sha256"] = json!(expected_operations);
        result.data["candidate_sha256"] = json!(digest);
        return result;
    }
    Report::ok("graph_edit_plan", json!({"committed":false,"input_sha256":expected,"operations_sha256":expected_operations,
        "graph_sha256":digest,"graph":graph,"outcomes":outcomes,"structural_review":validation,"runtime_verified":false,
        "note":"只读批量编辑预览；形状、轴及类型顺序问题可保留在草稿中修复。另用 graph-save 条件提交，不导入工厂或执行模型。"}))
}
