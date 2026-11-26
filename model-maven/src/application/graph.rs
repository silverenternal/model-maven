//! Read-only graph contract and topology planner. No user code is imported.
use super::*;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write as _;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Graph {
    version: u32,
    #[serde(default)]
    component_storage: Option<String>,
    #[serde(default)]
    definitions: Vec<Definition>,
    #[serde(default)]
    objectives: Vec<GraphObjective>,
    #[serde(default)]
    data: Option<GraphData>,
    #[serde(default)]
    training: Option<GraphTraining>,
    #[serde(default)]
    supervision: Option<Value>,
    #[serde(default)]
    legacy_import: Option<LegacyImport>,
    #[serde(default)]
    groups: Vec<GraphGroup>,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyImport {
    kind: String,
    sha256: String,
    content: String,
}
impl LegacyImport {
    fn valid(&self) -> bool {
        matches!(self.kind.as_str(), "scaffold" | "image_classification")
            && self.content.len() <= 128 * 1024
            && super::scaffold::sha256_bytes(self.content.as_bytes()) == self.sha256
            && serde_json::from_str::<Value>(&self.content).is_ok()
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphTraining {
    #[serde(default)]
    trial_steps: Option<u32>,
    #[serde(default)]
    optimizer: Option<GraphOptimizer>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    step: Option<String>,
    #[serde(default)]
    step_params: Map<String, Value>,
}
#[derive(serde::Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphOptimizer {
    #[serde(default, skip_serializing_if = "Option::is_none")] factory: Option<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")] params: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")] lr: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")] weight_decay: Option<f64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphGroup {
    id: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default)]
    nodes: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphData {
    factory: String,
    #[serde(default)]
    binding: Option<String>,
    #[serde(default)]
    params: Map<String, Value>,
}
#[derive(serde::Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphObjective {
    output: String,
    target: String,
    loss: String,
    weight: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Definition {
    id: String,
    factory: String,
    #[serde(default)]
    params: Map<String, Value>,
    #[serde(default)]
    children: Vec<String>,
    #[serde(default)]
    children_style: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Node {
    id: String,
    kind: String,
    #[serde(default)]
    output_name: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    definition: Option<String>,
    #[serde(default)]
    dtype_definitions: Vec<String>,
    #[serde(default)]
    track_sequence: bool,
    #[serde(default)]
    axis: Option<i32>,
    #[serde(default)]
    axes: Vec<usize>,
    #[serde(default)]
    width: Option<usize>,
    #[serde(default)]
    call_style: Option<String>,
    #[serde(default)]
    output_style: Option<String>,
    #[serde(default)]
    inputs: Vec<String>,
    #[serde(default)]
    outputs: Vec<String>,
    #[serde(default)]
    input_shapes: BTreeMap<String, Vec<Option<u64>>>,
    #[serde(default)]
    output_shapes: BTreeMap<String, Vec<Option<u64>>>,
    #[serde(default)]
    input_dtypes: BTreeMap<String, String>,
    #[serde(default)]
    output_dtypes: BTreeMap<String, String>,
    #[serde(default)]
    semantics: BTreeMap<String, String>,
    #[serde(default)]
    draft_rules: BTreeMap<String, Vec<GraphDraftRule>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphDraftRule {
    kind: String,
    mode: String,
    #[serde(default)]
    min: Option<f64>,
    #[serde(default)]
    max: Option<f64>,
    #[serde(default)]
    weight: Option<f64>,
}
fn draft_rule_valid(rule: &GraphDraftRule) -> bool {
    let numbers = [rule.min, rule.max, rule.weight];
    if numbers.into_iter().flatten().any(|v| !v.is_finite()) { return false; }
    if !matches!(rule.mode.as_str(), "observe" | "require" | "penalty") { return false; }
    if rule.mode == "penalty" {
        if rule.kind != "range" || rule.weight.is_none_or(|weight| weight <= 0.0) { return false; }
    } else if rule.weight.is_some() { return false; }
    match rule.kind.as_str() {
        "finite" => rule.min.is_none() && rule.max.is_none(),
        "range" => (rule.min.is_some() || rule.max.is_some())
            && !matches!((rule.min, rule.max), (Some(min), Some(max)) if min > max),
        _ => false,
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Port {
    node: String,
    port: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    from: Port,
    to: Port,
}
fn identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 128
        && value.chars().all(|ch| ch == '-' || unicode_ident::is_xid_continue(ch))
}
fn ports_valid(ports: &[String]) -> bool {
    let mut seen = HashSet::new();
    ports.len() <= 64 && ports.iter().all(|port| identifier(port) && seen.insert(port))
}
type Shape = Vec<Option<u64>>;
fn contracts_valid(contracts: &BTreeMap<String, Shape>, ports: &[String]) -> bool {
    contracts.len() <= ports.len() && contracts.iter().all(|(name, shape)| {
        ports.contains(name) && shape.len() <= 16 && shape.iter().all(|dim| dim.is_none_or(|value| (1..=1_000_000_000_000).contains(&value)))
    })
}
fn dtype_contracts_valid(contracts: &BTreeMap<String, String>, ports: &[String]) -> bool {
    contracts.len() <= ports.len() && contracts.iter().all(|(port, dtype)| {
        let mut bytes = dtype.bytes();
        ports.contains(port) && dtype.len() <= 64 && bytes.next().is_some_and(|byte| byte.is_ascii_alphabetic())
            && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    })
}
fn compatible(a: &Shape, b: &Shape) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(left, right)| match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => true,
    })
}
// Interpret only the bundled CSV adapter's declared per-sample dimensions.
// Returning None leaves unrecognized or malformed field declarations unknown.
fn csv_field_shape(value: &Value) -> Option<Vec<u64>> {
    if let Some(fields) = value.as_object() {
        if fields.keys().any(|name| !matches!(name.as_str(), "columns" | "shape" | "dtype")) { return None; }
        if let Some(dtype) = fields.get("dtype") {
            let dtype = dtype.as_str()?;
            let mut bytes = dtype.bytes();
            if dtype.len() > 64 || !bytes.next().is_some_and(|byte| byte.is_ascii_alphabetic())
                || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_') { return None; }
        }
    }
    let columns = if value.is_array() { value.as_array()? } else { value.get("columns")?.as_array()? };
    if columns.is_empty() || columns.len() > 4096 || columns.iter().any(|v| v.as_str().is_none_or(|s| s.is_empty() || s.chars().count() > 256)) {
        return None;
    }
    let shape = match value.get("shape") {
        None => vec![columns.len() as u64],
        Some(value) => {
            let values = value.as_array()?;
            if values.len() > 8 { return None; }
            values.iter().map(|v| v.as_u64().filter(|dim| *dim > 0)).collect::<Option<Vec<_>>>()?
        }
    };
    if shape.iter().try_fold(1u64, |product, dim| product.checked_mul(*dim))? != columns.len() as u64 { return None; }
    Some(shape)
}

fn csv_field_dtype(value: &Value) -> Option<&str> {
    csv_field_shape(value)?;
    if value.is_array() { Some("float32") } else { value.get("dtype").map_or(Some("float32"), Value::as_str) }
}

fn csv_objective_field<'a>(graph: &Graph, objective: &GraphObjective, data: &'a GraphData) -> Option<&'a Value> {
    let value = data.params.get("targets")?;
    let anonymous = match value {
        Value::Array(_) => Some(value),
        Value::Object(fields) if fields.len() == 1 => fields.get("$tensor"),
        _ => None,
    };
    if let Some(field) = anonymous {
        let same_target = graph.objectives.iter().all(|item| item.target == objective.target);
        same_target.then_some(field)
    } else {
        value.as_object()?.get(&objective.target)
    }
}

fn fail(code: &str, message: impl Into<String>) -> Report {
    Report::error("graph_plan", code, message, "检查节点、端口和连接后重新规划；尚未写入或运行模型。")
}
fn fail_at(code: &str, message: impl Into<String>, node: &str, port: Option<&str>, revision: &str) -> Report {
    let mut report = fail(code, message);
    report.data = json!({"node_id":node,"port":port,"input_sha256":revision});
    report
}

pub(super) fn plan(path: &Path, expected: &str) -> Report {
    let bytes = match super::scaffold::read_recipe_for_update(path) {
        Ok(bytes) => bytes,
        Err(error) => return fail("GRAPH_READ_FAILED", error),
    };
    let revision = super::scaffold::sha256_bytes(&bytes);
    if revision != expected { return fail("GRAPH_CONFLICT", "图文件内容已变化。") }
    let graph: Graph = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => return fail("GRAPH_INVALID", error.to_string()),
    };
    if graph.legacy_import.as_ref().is_some_and(|snapshot| !snapshot.valid()) {
        return fail("GRAPH_LEGACY_INVALID", "旧声明快照损坏或摘要不匹配；未修改原文件。")
    }
    if graph.training.as_ref().is_some_and(|training| {
        let optimizer = training.optimizer.as_ref();
        training.trial_steps.is_some_and(|steps| !(1..=1000).contains(&steps))
            || training.mode.as_deref().is_some_and(|mode| !matches!(mode, "managed-step" | "manual-step" | "external-loop"))
            || (matches!(training.mode.as_deref(), Some("manual-step" | "external-loop")) && training.step.is_none())
            || (training.mode.as_deref() == Some("manual-step") && optimizer.is_some_and(|item| item.factory.is_some()))
            || (training.mode.as_deref() == Some("external-loop") && optimizer.is_some_and(|item| item.factory.is_some() || !item.params.is_empty() || item.lr.is_some() || item.weight_decay.is_some()))
            || training.step.as_ref().is_some_and(|value| value.len() > 240 || (value != "$model:step" && !python_entrypoint(value))
                || (value == "$model:step" && training.mode.as_deref() == Some("external-loop")))
            || training.step_params.len() > 64 || !training.step_params.keys().all(|name| python_identifier(name))
            || serde_json::to_vec(&training.step_params).map_or(true, |value| value.len() > 65_536)
            || optimizer.is_some_and(|item| item.factory.as_ref().is_some_and(|value| value.len() > 240 || !python_entrypoint(value))
                || (item.factory.is_some() && (item.lr.is_some() || item.weight_decay.is_some()))
                || (item.factory.is_none() && !item.params.is_empty())
                || item.params.len() > 64 || !item.params.keys().all(|name| python_identifier(name))
                || serde_json::to_vec(&item.params).map_or(true, |value| value.len() > 65_536)
                || item.lr.is_some_and(|value| !value.is_finite() || value <= 0.0)
                || item.weight_decay.is_some_and(|value| !value.is_finite() || value < 0.0))
    }) { return fail("GRAPH_TRAINING_INVALID", "试跑步数、训练步骤、参数或优化器声明无效。"); }
    if graph.supervision.as_ref().is_some_and(|reference| super::scaffold::validate_supervision_value(reference).is_err()) {
        return fail("GRAPH_SUPERVISION_INVALID", "监督引用必须包含项目相对文件路径及 SHA-256。 ");
    }
    if graph.version != 1 || graph.component_storage.as_deref().is_some_and(|value| value != "named")
        || graph.nodes.len() > 512 || graph.edges.len() > 4096 || graph.definitions.len() > 512 || graph.objectives.len() > 64 || graph.groups.len() > 512 {
        return fail("GRAPH_INVALID", "图版本或组件存储方式不受支持，或节点、连接、定义超过限制。")
    }
    if graph.data.as_ref().is_some_and(|data|
        data.factory.len() > 240 || data.factory.chars().any(char::is_whitespace)
        || data.factory.split_once(':').is_none_or(|(module, name)| module.is_empty() || name.is_empty() || name.contains(':'))
        || data.binding.as_deref().is_some_and(|binding| !matches!(binding, "envelope" | "kwargs" | "args" | "single" | "single-raw"))
        || data.params.len() > 64 || serde_json::to_vec(&data.params).map_or(true, |raw| raw.len() > 65536)) {
        return fail("GRAPH_DATA_INVALID", "数据工厂、绑定方式或参数无效。");
    }
    let mut definitions = BTreeMap::new();
    for definition in &graph.definitions {
        if !identifier(&definition.id) || definitions.contains_key(&definition.id)
            || definition.factory.len() > 240 || definition.factory.chars().any(char::is_whitespace)
            || definition.factory.split_once(':').is_none_or(|(module, callable)| module.is_empty() || callable.is_empty() || callable.contains(':'))
            || definition.params.len() > 64 || serde_json::to_vec(&definition.params).map_or(true, |raw| raw.len() > 65536) {
            return fail("GRAPH_DEFINITION_INVALID", format!("组件定义 {} 无效或重复。", definition.id));
        }
        definitions.insert(definition.id.clone(), definition);
    }
    // Children are constructor arguments, not visual groups. Reject missing
    // references and cycles before generating or executing any user code.
    fn visit_definition<'a>(id: &'a str, definitions: &BTreeMap<String, &'a Definition>,
        states: &mut BTreeMap<&'a str, u8>, order: &mut Vec<&'a str>) -> Result<(), String> {
        match states.get(id) { Some(1) => return Err(format!("组件容器存在循环引用：{id}")), Some(2) => return Ok(()), _ => {} }
        let definition = definitions.get(id).ok_or_else(|| format!("找不到子组件：{id}"))?;
        states.insert(id, 1);
        for child in &definition.children { visit_definition(child, definitions, states, order)?; }
        states.insert(id, 2); order.push(id); Ok(())
    }
    let mut definition_states = BTreeMap::new();
    let mut definition_order = Vec::new();
    for definition in &graph.definitions {
        if definition.children.len() > 64 || definition.children_style.as_deref().is_some_and(|style| style != "list") {
            return fail("GRAPH_DEFINITION_INVALID", "组件子项不能超过 64 个；子项构造方式不受支持。");
        }
        if let Err(error) = visit_definition(&definition.id, &definitions, &mut definition_states, &mut definition_order) {
            return fail("GRAPH_DEFINITION_INVALID", error);
        }
    }
    let mut nodes = BTreeMap::new();
    let mut calls = BTreeMap::<String, Vec<String>>::new();
    let mut output_names = HashSet::new();
    for node in &graph.nodes {
        if !identifier(&node.id) || nodes.contains_key(&node.id) || !ports_valid(&node.inputs) || !ports_valid(&node.outputs)
            || node.label.as_ref().is_some_and(|label| label.len() > 256 || label.chars().any(char::is_control)) {
            return fail_at("GRAPH_NODE_INVALID", format!("节点 {} 的身份、标签或端口无效或重复。", node.id), &node.id, None, &revision);
        }
        let valid = match node.kind.as_str() {
            "input" => node.inputs.is_empty() && !node.outputs.is_empty() && node.definition.is_none(),
            "output" => node.inputs.len() == 1 && node.outputs.is_empty() && node.definition.is_none(),
            "call" => !node.inputs.is_empty() && !node.outputs.is_empty()
                && node.definition.as_ref().is_some_and(|id| definitions.contains_key(id)),
            // Cast to a component's first parameter dtype, or the runtime
            // default float dtype if no component/parameter is available.
            "cast" => node.inputs.len() == 1 && node.outputs.len() == 1
                && node.definition.as_ref().is_none_or(|id| definitions.contains_key(id))
                && (node.definition.is_none() || node.dtype_definitions.is_empty())
                && node.dtype_definitions.len() <= 64
                && node.dtype_definitions.iter().all(|id| definitions.contains_key(id))
                && node.dtype_definitions.iter().collect::<HashSet<_>>().len() == node.dtype_definitions.len(),
            "clone" => node.inputs.len() == 1 && node.outputs.len() == 1 && node.definition.is_none(),
            "mean_dynamic" => node.inputs.len() == 1 && node.outputs.len() == 1 && node.definition.is_none()
                && !node.axes.is_empty() && node.axes.len() <= 8 && node.axes.iter().all(|axis| *axis < 8)
                && node.axes.iter().collect::<HashSet<_>>().len() == node.axes.len(),
            "assert_flat" => node.inputs.len() == 2 && node.outputs.len() == 1 && node.definition.is_none()
                && node.width.is_some_and(|width| (1..=65_536).contains(&width)),
            "assert_batch" => node.inputs.len() == 2 && node.outputs.len() == 1 && node.definition.is_none(),
            "after" => node.inputs.len() == 2 && node.outputs.len() == 1 && node.definition.is_none(),
            "add" | "concat" => node.inputs.len() >= 2 && node.outputs.len() == 1 && node.definition.is_none(),
            _ => false,
        };
        if !valid || node.output_name.as_ref().is_some_and(|name| node.kind != "output" || !identifier(name))
            || (node.kind == "output" && !output_names.insert(node.output_name.as_deref().unwrap_or(&node.id)))
            || (node.axis.is_some() && node.kind != "concat") || (!node.axes.is_empty() && node.kind != "mean_dynamic")
            || (!node.dtype_definitions.is_empty() && node.kind != "cast")
            || (node.track_sequence && (node.kind != "cast" || node.definition.is_some()))
            || (node.width.is_some() && node.kind != "assert_flat")
            || node.call_style.as_deref().is_some_and(|style|
                node.kind != "call" || !matches!(style, "positional" | "keyword" | "list"))
            || node.output_style.as_deref().is_some_and(|style|
                node.kind != "call" || !matches!(style, "mapping" | "sequence"))
            || !contracts_valid(&node.input_shapes, &node.inputs)
            || !contracts_valid(&node.output_shapes, &node.outputs)
            || !dtype_contracts_valid(&node.input_dtypes, &node.inputs)
            || !dtype_contracts_valid(&node.output_dtypes, &node.outputs)
            || (!node.semantics.is_empty() && !matches!(node.kind.as_str(), "input" | "output"))
            || node.semantics.iter().any(|(port, label)| {
                let ports = if node.kind == "input" { &node.outputs } else { &node.inputs };
                !ports.contains(port) || label.trim().is_empty() || label.len() > 256 || label.chars().any(char::is_control)
            })
            || (!node.draft_rules.is_empty() && !matches!(node.kind.as_str(), "input" | "output"))
            || node.draft_rules.iter().any(|(port, rules)| {
                let ports = if node.kind == "input" { &node.outputs } else { &node.inputs };
                !ports.contains(port) || rules.is_empty() || rules.len() > 32 || rules.iter().any(|rule| !draft_rule_valid(rule))
            })
            || (!matches!(node.kind.as_str(), "input" | "call" | "cast" | "clone" | "mean_dynamic" | "assert_flat" | "assert_batch" | "after" | "add" | "concat") && !node.output_shapes.is_empty()) {
            return fail_at("GRAPH_NODE_INVALID", format!("节点 {} 的类型、端口、轴或组件定义不合法。", node.id), &node.id, None, &revision);
        }
        if node.kind == "call" {
            if let Some(id) = &node.definition { calls.entry(id.clone()).or_default().push(node.id.clone()); }
        }
        nodes.insert(node.id.clone(), node);
    }
    let mut group_ids = HashSet::new();
    let mut grouped_nodes = HashSet::new();
    for group in &graph.groups {
        if !identifier(&group.id) || !group_ids.insert(&group.id) || group.nodes.len() > 512
            || group.nodes.iter().any(|id| !nodes.contains_key(id) || !grouped_nodes.insert(id)) {
            return fail("GRAPH_GROUP_INVALID", format!("分组 {} 无效、重复，或节点不存在/属于多个分组。", group.id));
        }
    }
    let parents = graph.groups.iter().map(|group| (group.id.as_str(), group.parent.as_deref())).collect::<BTreeMap<_, _>>();
    for group in &graph.groups {
        let mut current = group.parent.as_deref();
        let mut visited = HashSet::new();
        visited.insert(group.id.as_str());
        while let Some(parent) = current {
            if !visited.insert(parent) || !parents.contains_key(parent) {
                return fail("GRAPH_GROUP_INVALID", format!("分组 {} 的父级不存在或形成循环。", group.id));
            }
            current = parents[parent];
        }
    }
    for objective in &graph.objectives {
        let valid_loss = matches!(objective.loss.as_str(), "mse" | "cross_entropy")
            || (objective.loss.len() <= 240 && objective.loss.split_once(':').is_some_and(|(module, name)|
                !module.is_empty() && !name.is_empty() && !name.contains(':') && !objective.loss.chars().any(char::is_whitespace)));
        if nodes.get(&objective.output).is_none_or(|node| node.kind != "output")
            || !identifier(&objective.target) || !valid_loss || !objective.weight.is_finite() || objective.weight < 0.0 {
            return fail("GRAPH_OBJECTIVE_INVALID", "损失声明的输出、目标、损失或权重无效。");
        }
    }
    let mut data_review = match &graph.data {
        None => json!({"status":"not_declared","issues":[],"runtime_verified":false}),
        Some(data) if data.factory == "model_maven_worker.data.presets:graph_image_batch" => {
            let mut issues = Vec::new();
            let selected_input = data.params.get("input_name").and_then(Value::as_str);
            let selected_target = data.params.get("target_name").and_then(Value::as_str);
            if selected_target.is_none_or(|name| name.is_empty() || name.chars().count() > 128 || name.chars().any(|ch| ch < '\u{20}')) {
                issues.push(json!({"kind":"image_invalid_target","name":selected_target}));
            }
            if data.params.get("root").and_then(Value::as_str).is_none() {
                issues.push(json!({"kind":"image_invalid_root"}));
            }
            let split = data.params.get("split").map(Value::as_str).unwrap_or(Some("train"));
            if !matches!(split, Some("train" | "validation" | "test")) {
                issues.push(json!({"kind":"image_invalid_split"}));
            }
            if data.params.get("batch_size").is_some_and(|value| value.as_u64().is_none_or(|size| size == 0)) {
                issues.push(json!({"kind":"image_invalid_batch_size"}));
            }
            if let Some(value) = data.params.get("split_transforms").filter(|value| !value.is_null()) {
                let valid = data.params.get("transforms").is_none_or(Value::is_null)
                    && value.as_object().is_some_and(|map| {
                        map.keys().all(|name| matches!(name.as_str(), "train" | "validation" | "test"))
                            && split.and_then(|name| map.get(name)).is_some_and(Value::is_array)
                    });
                if !valid { issues.push(json!({"kind":"image_invalid_transforms"})); }
            }
            let mut inputs = BTreeSet::new();
            for node in graph.nodes.iter().filter(|node| node.kind == "input") {
                for port in &node.outputs {
                    let name = if node.outputs.len() == 1 { node.id.clone() } else { format!("{}.{}", node.id, port) };
                    if selected_input != Some(name.as_str()) {
                        issues.push(json!({"kind":"image_missing_input","node":node.id,"name":name}));
                    }
                    inputs.insert(name);
                }
            }
            if selected_input.is_none_or(|name| !inputs.contains(name)) {
                issues.push(json!({"kind":"image_unknown_input","name":selected_input}));
            }
            for objective in &graph.objectives {
                if selected_target != Some(objective.target.as_str()) {
                    issues.push(json!({"kind":"image_missing_target","node":objective.output,"name":objective.target}));
                }
            }
            if data.binding.as_deref().is_some_and(|binding| binding != "envelope") {
                issues.push(json!({"kind":"unexpected_binding","name":data.binding}));
            }
            json!({"status":if issues.is_empty() {"mapping_consistent"} else {"needs_attention"},
                "issues":issues,"runtime_verified":false})
        }
        Some(data) if data.factory != "model_csv:batch" => json!({"status":"unknown_factory_contract","issues":[],"runtime_verified":false}),
        Some(data) => {
            let mut issues = Vec::new();
            let inputs = data.params.get("inputs").and_then(Value::as_object);
            if data.params.get("path").and_then(Value::as_str)
                .is_none_or(|path| path.is_empty() || path.chars().count() > 4096) {
                issues.push(json!({"kind":"invalid_csv_path"}));
            }
            let split = data.params.get("split").filter(|value| !value.is_null());
            let weights = data.params.get("split_weights").filter(|value| !value.is_null());
            let seed = data.params.get("split_seed").filter(|value| !value.is_null());
            let valid_split = match split {
                None => weights.is_none() && seed.is_none(),
                Some(value) => {
                    let names = ["train", "val", "test"];
                    matches!(value.as_str(), Some("train" | "val" | "test"))
                        && weights.and_then(Value::as_object).is_some_and(|values| {
                            values.len() == names.len() && names.iter().all(|name|
                                values.get(*name).and_then(Value::as_u64).is_some_and(|n| n <= 1_000_000))
                                && names.iter().filter_map(|name| values.get(*name).and_then(Value::as_u64)).sum::<u64>() > 0
                        })
                        && seed.and_then(Value::as_i64).is_some_and(|value| value >= 0)
                }
            };
            if !valid_split { issues.push(json!({"kind":"invalid_csv_split"})); }
            let batch_size = match data.params.get("batch_size") {
                None => Some(2),
                Some(value) => value.as_u64().filter(|size| (1..=256).contains(size)),
            };
            if batch_size.is_none() { issues.push(json!({"kind":"invalid_batch_size"})); }
            let mut expected_inputs = BTreeSet::new();
            for node in graph.nodes.iter().filter(|node| node.kind == "input") {
                for port in &node.outputs {
                    let name = if node.outputs.len() == 1 { node.id.clone() } else { format!("{}.{}", node.id, port) };
                    if name.len() > 128 {
                        issues.push(json!({"kind":"csv_input_name_too_long","node":node.id,"name":name}));
                    } else if inputs.is_none_or(|fields| !fields.contains_key(&name)) {
                        issues.push(json!({"kind":"missing_input","node":node.id,"name":name}));
                    } else if let Some(field) = inputs.and_then(|fields| fields.get(&name)) {
                        if let Some(shape) = csv_field_shape(field) {
                            if let (Some(batch), Some(contract)) = (batch_size, node.output_shapes.get(port)) {
                                let declared = std::iter::once(Some(batch)).chain(shape.iter().copied().map(Some)).collect::<Shape>();
                                if !compatible(contract, &declared) {
                                    issues.push(json!({"kind":"input_shape_conflict","node":node.id,"name":name,
                                        "declared_shape":contract,"csv_shape":declared}));
                                }
                            }
                            if let (Some(actual), Some(expected)) = (csv_field_dtype(field), node.output_dtypes.get(port)) {
                                if actual != expected {
                                    issues.push(json!({"kind":"input_dtype_review","node":node.id,"name":name,
                                        "declared_dtype":expected,"csv_dtype":actual}));
                                }
                            }
                        } else {
                            issues.push(json!({"kind":"invalid_input_field","node":node.id,"name":name}));
                        }
                    }
                    expected_inputs.insert(name);
                }
            }
            if let Some(inputs) = inputs {
                for name in inputs.keys().filter(|name| !expected_inputs.contains(*name)) {
                    issues.push(json!({"kind":"unexpected_input","name":name}));
                }
            }
            for objective in &graph.objectives {
                let field = csv_objective_field(&graph, objective, data);
                if field.is_none() {
                    issues.push(json!({"kind":"missing_target","node":objective.output,"name":objective.target}));
                } else if field.and_then(csv_field_shape).is_none() {
                    issues.push(json!({"kind":"invalid_target_field","node":objective.output,"name":objective.target}));
                }
            }
            if data.binding.as_deref().is_some_and(|binding| binding != "envelope") {
                issues.push(json!({"kind":"unexpected_binding","name":data.binding}));
            }
            json!({"status":if issues.is_empty() {"mapping_consistent"} else {"needs_attention"},"issues":issues,"runtime_verified":false})
        }
    };
    let mut occupied = HashSet::new();
    let mut dtype_advisories = Vec::new();
    let mut degree = nodes.keys().map(|id| (id.clone(), 0usize)).collect::<BTreeMap<_, _>>();
    let mut successors = BTreeMap::<String, Vec<String>>::new();
    for edge in &graph.edges {
        let Some(from) = nodes.get(&edge.from.node) else { return fail("GRAPH_EDGE_INVALID", format!("找不到源节点 {}。", edge.from.node)); };
        let Some(to) = nodes.get(&edge.to.node) else { return fail("GRAPH_EDGE_INVALID", format!("找不到目标节点 {}。", edge.to.node)); };
        if !from.outputs.contains(&edge.from.port) || !to.inputs.contains(&edge.to.port) {
            let (node, port) = if !to.inputs.contains(&edge.to.port) { (&edge.to.node, &edge.to.port) }
                else { (&edge.from.node, &edge.from.port) };
            return fail_at("GRAPH_EDGE_INVALID", format!("端口不存在或方向错误：{}.{} → {}.{}。", edge.from.node, edge.from.port, edge.to.node, edge.to.port), node, Some(port), &revision);
        }
        if !occupied.insert((edge.to.node.clone(), edge.to.port.clone())) {
            return fail_at("GRAPH_INPUT_CONFLICT", format!("输入端口 {}.{} 已有连接。", edge.to.node, edge.to.port), &edge.to.node, Some(&edge.to.port), &revision);
        }
        if let (Some(source), Some(target)) = (from.output_dtypes.get(&edge.from.port), to.input_dtypes.get(&edge.to.port)) {
            if source != target {
                dtype_advisories.push(json!({"from":{"node":edge.from.node,"port":edge.from.port,"dtype":source},
                    "to":{"node":edge.to.node,"port":edge.to.port,"dtype":target}}));
            }
        }
        *degree.get_mut(&edge.to.node).expect("validated target") += 1;
        successors.entry(edge.from.node.clone()).or_default().push(edge.to.node.clone());
    }
    let mut ready = degree.iter().filter(|(_, count)| **count == 0).map(|(id, _)| id.clone()).collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(id) = ready.pop_first() {
        order.push(id.clone());
        for next in successors.get(&id).into_iter().flatten() {
            let count = degree.get_mut(next).expect("validated successor");
            *count -= 1;
            if *count == 0 { ready.insert(next.clone()); }
        }
    }
    if order.len() != nodes.len() {
        let blocked = degree.iter().filter(|(_, count)| **count > 0).map(|(id, _)| id.as_str()).collect::<Vec<_>>();
        return fail_at("GRAPH_CYCLE", format!("图包含循环；无法排序的节点：{}。循环请封装为用户组件。", blocked.join(", ")),
            blocked[0], None, &revision);
    }
    for cast in graph.nodes.iter().filter(|node| node.kind == "cast" && node.track_sequence) {
        let mut current = cast.id.as_str();
        let mut actual = Vec::new();
        let mut terminated = false;
        for _ in 0..=graph.nodes.len() {
            let next = successors.get(current).map(Vec::as_slice).unwrap_or(&[]);
            if next.is_empty() { break; } // An incomplete draft remains saveable.
            if next.len() != 1 {
                return fail_at("GRAPH_DTYPE_ORDER_CONFLICT", format!("节点 {} 后的层栈有分支，请审核类型来源并关闭顺序跟踪。", cast.id), &cast.id, None, &revision);
            }
            let node = nodes.get(&next[0]).expect("validated successor");
            match node.kind.as_str() {
                "call" => actual.push(node.definition.as_deref().expect("validated call").to_owned()),
                "assert_flat" => {},
                "cast" | "concat" | "output" | "add" => { terminated = true; break; },
                _ => return fail_at("GRAPH_DTYPE_ORDER_CONFLICT", format!("节点 {} 后包含不支持的层栈节点，请审核类型来源。", cast.id), &cast.id, None, &revision),
            }
            current = &node.id;
        }
        if terminated && actual != cast.dtype_definitions {
            return fail_at("GRAPH_DTYPE_ORDER_CONFLICT", format!("节点 {} 的类型来源顺序已与实际层栈不一致；请在节点详情中重新选择。", cast.id), &cast.id, None, &revision);
        }
    }
    let missing = nodes.values().flat_map(|node| node.inputs.iter().filter_map(|port| {
        (!occupied.contains(&(node.id.clone(), port.clone()))).then(|| json!({"node":node.id,"port":port}))
    })).collect::<Vec<_>>();
    // These are conditional conclusions from manually declared contracts, not
    // observations of a factory or proof of what the runtime will produce.
    let inbound = graph.edges.iter().map(|edge| ((edge.to.node.as_str(), edge.to.port.as_str()),
        (edge.from.node.as_str(), edge.from.port.as_str()))).collect::<BTreeMap<_, _>>();
    let mut shapes = BTreeMap::<(String, String), Option<Shape>>::new();
    let mut output_shapes = BTreeMap::<String, BTreeMap<String, Option<Shape>>>::new();
    let mut objective_output_shapes = BTreeMap::<String, Shape>::new();
    for id in &order {
        let node = nodes.get(id).expect("sorted node exists");
        let mut received = Vec::new();
        for port in &node.inputs {
            let upstream = inbound.get(&(node.id.as_str(), port.as_str()));
            let actual = upstream.and_then(|(from, source_port)| {
                shapes.get(&(from.to_string(), source_port.to_string())).cloned().flatten()
            });
            let expected = node.input_shapes.get(port);
            let shape = match (actual.as_ref(), expected, upstream.is_some()) {
                (Some(actual), Some(expected), _) if !compatible(actual, expected) => {
                    return fail_at("GRAPH_SHAPE_CONFLICT", format!("节点 {} 输入 {} 的声明形状 {:?} 与上游 {:?} 冲突。", node.id, port, expected, actual), &node.id, Some(port), &revision);
                },
                (Some(actual), Some(expected), _) => Some(actual.iter().zip(expected).map(|(a, b)| a.or(*b)).collect()),
                (Some(actual), None, _) => Some(actual.clone()),
                (None, Some(expected), true) => Some(expected.clone()),
                _ => None,
            };
            received.push(shape);
        }
        if node.kind == "output" {
            if let Some(port) = node.inputs.first() {
                if inbound.contains_key(&(node.id.as_str(), port.as_str())) {
                    if let Some(shape) = received.first().cloned().flatten().or_else(|| node.input_shapes.get(port).cloned()) {
                        objective_output_shapes.insert(node.id.clone(), shape);
                    }
                }
            }
        }
        let inferred = if matches!(node.kind.as_str(), "cast" | "clone") {
            received.first().cloned().flatten()
        } else if node.kind == "mean_dynamic" {
            if let Some(mut shape) = received.first().cloned().flatten() {
                for axis in &node.axes {
                    let Some(dim) = shape.get_mut(*axis + 1) else {
                        return fail_at("GRAPH_SHAPE_CONFLICT", format!("节点 {} 的动态轴超出输入秩。", node.id), &node.id, None, &revision);
                    };
                    *dim = Some(1);
                }
                Some(shape)
            } else { None }
        } else if node.kind == "assert_flat" {
            Some(vec![received.get(1).and_then(Option::as_ref).and_then(|shape| shape.first().copied().flatten()), node.width.map(|width| width as u64)])
        } else if matches!(node.kind.as_str(), "assert_batch" | "after") {
            received.first().cloned().flatten()
        } else if matches!(node.kind.as_str(), "add" | "concat") {
            let known = received.iter().filter_map(Option::as_ref).collect::<Vec<_>>();
            if let Some(first) = known.first() {
                if known.iter().any(|shape| shape.len() != first.len()) {
                    return fail_at("GRAPH_SHAPE_CONFLICT", format!("节点 {} 的输入秩不一致。", node.id), &node.id, None, &revision);
                }
                let axis = if node.kind == "concat" {
                    match node.axis {
                        Some(value) => {
                            let rank = first.len() as i64;
                            let normalized = if value < 0 { i64::from(value) + rank } else { i64::from(value) };
                            if normalized < 0 || normalized >= rank {
                                return fail_at("GRAPH_AXIS_INVALID", format!("节点 {} 的拼接轴 {} 超出已知秩 {}。", node.id, value, rank), &node.id, None, &revision);
                            }
                            Some(normalized as usize)
                        }
                        None => None,
                    }
                } else { None };
                for dimension in 0..first.len() {
                    if axis == Some(dimension) || (node.kind == "concat" && axis.is_none()) { continue }
                    let values = known.iter().filter_map(|shape| shape[dimension]).collect::<BTreeSet<_>>();
                    if values.len() > 1 {
                        return fail_at("GRAPH_SHAPE_CONFLICT", format!("节点 {} 第 {} 维的已声明输入不一致。", node.id, dimension), &node.id, None, &revision);
                    }
                }
                if known.len() != received.len() || (node.kind == "concat" && axis.is_none()) { None }
                else {
                    let mut result = Vec::with_capacity(first.len());
                    for dimension in 0..first.len() {
                        let value = if axis == Some(dimension) {
                            let values = known.iter().map(|shape| shape[dimension]).collect::<Option<Vec<_>>>();
                            if let Some(values) = values {
                                match values.into_iter().try_fold(0u64, u64::checked_add) {
                                    Some(sum) => Some(sum),
                                    None => return fail_at("GRAPH_SHAPE_CONFLICT", format!("节点 {} 拼接维度溢出。", node.id), &node.id, None, &revision),
                                }
                            } else { None }
                        } else {
                            // Equal-shape/non-axis constraints propagate known dimensions
                            // even when another input left that dimension unspecified.
                            known.iter().find_map(|shape| shape[dimension])
                        };
                        result.push(value);
                    }
                    Some(result)
                }
            } else { None }
        } else { None };
        let mut node_shapes = BTreeMap::new();
        for port in &node.outputs {
            let shape = if matches!(node.kind.as_str(), "input" | "call") {
                node.output_shapes.get(port).cloned()
            } else if matches!(node.kind.as_str(), "cast" | "clone" | "mean_dynamic" | "assert_flat" | "assert_batch" | "after" | "add" | "concat") {
                let declared = node.output_shapes.get(port);
                match (inferred.as_ref(), declared) {
                    (Some(actual), Some(expected)) if !compatible(actual, expected) => {
                        return fail_at("GRAPH_SHAPE_CONFLICT", format!("节点 {} 输出 {} 的声明形状 {:?} 与条件推导结果 {:?} 冲突。", node.id, port, expected, actual), &node.id, Some(port), &revision);
                    },
                    (Some(actual), Some(expected)) => Some(actual.iter().zip(expected).map(|(a, b)| a.or(*b)).collect()),
                    (Some(actual), None) => Some(actual.clone()),
                    (None, Some(expected)) => Some(expected.clone()),
                    (None, None) => None,
                }
            } else { inferred.clone() };
            shapes.insert((node.id.clone(), port.clone()), shape.clone());
            node_shapes.insert(port.clone(), shape);
        }
        if !node_shapes.is_empty() { output_shapes.insert(node.id.clone(), node_shapes); }
    }
    if let Some(data) = graph.data.as_ref().filter(|data| data.factory == "model_csv:batch") {
        let batch = match data.params.get("batch_size") {
            None => Some(2),
            Some(value) => value.as_u64().filter(|size| (1..=256).contains(size)),
        };
        let mut reviews = Vec::new();
        let mut conflicts = Vec::new();
        for objective in &graph.objectives {
            let prediction = objective_output_shapes.get(&objective.output);
            let field = csv_objective_field(&graph, objective, data);
            let target = batch.zip(field.and_then(csv_field_shape))
                .map(|(batch, shape)| std::iter::once(Some(batch)).chain(shape.into_iter().map(Some)).collect::<Shape>());
            let status = match (prediction, target.as_ref(), objective.loss.as_str()) {
                (Some(prediction), Some(target), "mse") => {
                    if !compatible(prediction, target) { "conflict" }
                    else if prediction.iter().any(Option::is_none) { "unknown" } else { "consistent" }
                }
                (Some(prediction), Some(target), "cross_entropy") => {
                    // PyTorch accepts class-index targets with the class axis
                    // removed, or probability targets with the same shape.
                    let index_shape = (prediction.len() >= 2).then(|| prediction.iter().enumerate()
                        .filter_map(|(index, dim)| (index != 1).then_some(*dim)).collect::<Shape>());
                    let possible = compatible(prediction, target) || index_shape.as_ref().is_some_and(|shape| compatible(shape, target));
                    if !possible { "conflict" }
                    else if prediction.iter().any(Option::is_none) { "unknown" } else { "consistent" }
                }
                _ => "unknown",
            };
            if status == "conflict" {
                conflicts.push(json!({"kind":"target_shape_conflict","node":objective.output,"name":objective.target,
                    "loss":objective.loss,"prediction_shape":prediction,"target_shape":target}));
            }
            // This is a limited advisory based on declared shapes and familiar
            // dtype spellings, not a runtime dtype or value check.
            if objective.loss == "cross_entropy" {
                if let (Some(prediction), Some(target), Some(dtype)) = (prediction, target.as_ref(), field.and_then(csv_field_dtype)) {
                    let index_shape = (prediction.len() >= 2).then(|| prediction.iter().enumerate()
                        .filter_map(|(index, dim)| (index != 1).then_some(*dim)).collect::<Shape>());
                    let index_match = index_shape.as_ref().is_some_and(|shape| compatible(shape, target));
                    let probability_match = compatible(prediction, target);
                    let float_name = matches!(dtype, "float16" | "float32" | "float64" | "bfloat16" | "half" | "float" | "double");
                    let integer_name = matches!(dtype, "int64" | "long" | "int32" | "int" | "int16" | "short" | "int8" | "uint8" | "bool");
                    if (index_match && !probability_match && float_name) || (probability_match && !index_match && integer_name) {
                        conflicts.push(json!({"kind":"target_dtype_review","node":objective.output,"name":objective.target,
                            "dtype":dtype,"prediction_shape":prediction,"target_shape":target}));
                    }
                }
            }
            reviews.push(json!({"node":objective.output,"target":objective.target,"loss":objective.loss,"shape_status":status}));
        }
        data_review["objective_shapes"] = json!(reviews);
        if !conflicts.is_empty() {
            if let Some(issues) = data_review["issues"].as_array_mut() { issues.extend(conflicts); }
            data_review["status"] = json!("needs_attention");
        }
    }
    let mut referenced = graph.nodes.iter().flat_map(|node| node.definition.iter().chain(node.dtype_definitions.iter()).map(String::as_str)).collect::<HashSet<_>>();
    for id in definition_order.iter().rev() { if referenced.contains(id) { referenced.extend(definitions[*id].children.iter().map(String::as_str)); } }
    let unused = definitions.keys().filter(|id| !referenced.contains(id.as_str())).cloned().collect::<Vec<_>>();
    let shared = calls.into_iter().filter(|(_, nodes)| nodes.len() > 1).collect::<BTreeMap<_, _>>();
    let missing_boundaries = ["input", "output"].into_iter()
        .filter(|kind| !nodes.values().any(|node| node.kind == *kind)).collect::<Vec<_>>();
    let complete = missing.is_empty() && missing_boundaries.is_empty();
    Report::ok("graph_plan", json!({"committed":false,"input_sha256":revision,
        "topological_order":order,"unconnected_inputs":missing,"missing_boundaries":missing_boundaries,"unused_definitions":unused,
        "shared_definitions":shared,"status":if complete {"structurally_connected"} else {"incomplete"},
        "shape_status":"conditional_on_declared_contracts","output_shapes":output_shapes,"data_review":data_review,
        "dtype_advisories":dtype_advisories,
        "groups":graph.groups.iter().map(|group| json!({"id":group.id,"parent":group.parent,"nodes":group.nodes})).collect::<Vec<_>>(),"runtime_verified":false,
        "note":"形状仅基于手动声明和汇合规则；未知维度保持未知，不导入工厂、不运行模型或写入项目。"}))
}

pub(super) fn bundle_reference(bytes: &[u8]) -> Result<Option<Value>, String> {
    let graph: Graph = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    if graph.version != 1 { return Err("不支持此图声明版本。".into()); }
    if let Some(reference) = &graph.supervision { super::scaffold::validate_supervision_value(reference)?; }
    Ok(graph.supervision)
}

pub(super) fn training_profile(path: &Path, bytes: &[u8]) -> Result<Value, String> {
    let revision = super::scaffold::sha256_bytes(bytes);
    let checked = plan(path, &revision);
    if !checked.success() { return Err("图声明无效或已变化，请先检查图。".into()); }
    let graph: Graph = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let training = graph.training.ok_or("图声明未配置训练。")?;
    let mode = training.mode.as_deref().unwrap_or("managed-step");
    let entry = training.step.as_deref().unwrap_or("$model:step");
    let optimizer = if mode == "external-loop" { Value::Null } else { json!(training.optimizer) };
    Ok(json!({"mode":mode,"entry":entry,"params":training.step_params,"optimizer":optimizer}))
}


/// Preview one reorder without writing either the graph or generated source.
pub(super) fn reorder_plan(path: &Path, expected: &str, moved: &str, anchor: &str, after: bool) -> Report {
    let reject = |code, text: String| Report::error("graph_reorder_plan", code, text, "检查链、端口和声明形状；未修改图。");
    let mut existing_plan = plan(path, expected);
    // Structural validation runs before shape propagation. A current shape
    // conflict can be the reason for reordering, so allow that diagnostic and
    // check every replacement connection below.
    if !existing_plan.success() && !existing_plan.diagnostics.iter().any(|item| matches!(item.code.as_str(), "GRAPH_SHAPE_CONFLICT" | "GRAPH_DTYPE_ORDER_CONFLICT")) {
        existing_plan.operation = "graph_reorder_plan".into(); return existing_plan;
    }
    let bytes = match super::scaffold::read_recipe_for_update(path) {
        Ok(value) => value, Err(error) => return reject("GRAPH_READ_FAILED", error),
    };
    if super::scaffold::sha256_bytes(&bytes) != expected { return reject("GRAPH_CONFLICT", "图文件已变化。".into()); }
    let graph: Graph = match serde_json::from_slice(&bytes) {
        Ok(value) => value, Err(error) => return reject("GRAPH_INVALID", error.to_string()),
    };
    let nodes = graph.nodes.iter().map(|node| (node.id.as_str(), node)).collect::<BTreeMap<_, _>>();
    let mut incoming = BTreeMap::<&str, Vec<usize>>::new();
    let mut outgoing = BTreeMap::<&str, Vec<usize>>::new();
    for (index, edge) in graph.edges.iter().enumerate() {
        incoming.entry(&edge.to.node).or_default().push(index);
        outgoing.entry(&edge.from.node).or_default().push(index);
    }
    let simple = |id: &str| -> bool {
        let (Some(node), Some(ins), Some(outs)) = (nodes.get(id), incoming.get(id), outgoing.get(id)) else { return false };
        node.kind == "call" && node.inputs.len() == 1 && node.outputs.len() == 1 && ins.len() == 1 && outs.len() == 1
            && graph.edges[ins[0]].to.port == node.inputs[0] && graph.edges[outs[0]].from.port == node.outputs[0]
    };
    if moved == anchor || !simple(moved) || !simple(anchor) {
        return reject("GRAPH_REORDER_UNSUPPORTED", "请选择同一无分支链中的两个单输入输出组件。".into());
    }
    let mut first = moved;
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(first) { return reject("GRAPH_CYCLE", "组件链包含循环。".into()); }
        let previous = graph.edges[incoming[first][0]].from.node.as_str();
        if !simple(previous) { break; }
        first = previous;
    }
    let mut chain = Vec::new();
    let mut current = first;
    visited.clear();
    while simple(current) {
        if !visited.insert(current) { return reject("GRAPH_CYCLE", "组件链包含循环。".into()); }
        chain.push(current);
        current = graph.edges[outgoing[current][0]].to.node.as_str();
    }
    if !chain.contains(&anchor) { return reject("GRAPH_REORDER_UNSUPPORTED", "节点不属于同一无分支链。".into()); }
    let mut reordered = chain.clone();
    reordered.retain(|id| *id != moved);
    let Some(target) = reordered.iter().position(|id| *id == anchor) else { return reject("GRAPH_REORDER_UNSUPPORTED", "找不到目标节点。".into()); };
    reordered.insert(target + usize::from(after), moved);
    if reordered == chain { return reject("GRAPH_REORDER_UNCHANGED", "节点已经位于指定位置。".into()); }
    let entry = incoming[first][0];
    let exit = outgoing[*chain.last().expect("nonempty chain")][0];
    let mut removed = HashSet::from([entry]);
    for id in &chain { removed.insert(outgoing[*id][0]); }
    let mut replacements = Vec::new();
    let mut dtype_advisories = Vec::new();
    let mut append = |from_node: &str, from_port: &str, to_node: &str, to_port: &str| -> bool {
        let (Some(source), Some(target)) = (nodes.get(from_node), nodes.get(to_node)) else { return false };
        if !source.outputs.iter().any(|port| port == from_port) || !target.inputs.iter().any(|port| port == to_port) { return false; }
        if let (Some(a), Some(b)) = (source.output_shapes.get(from_port), target.input_shapes.get(to_port)) {
            if !compatible(a, b) { return false; }
        }
        if let (Some(a), Some(b)) = (source.output_dtypes.get(from_port), target.input_dtypes.get(to_port)) {
            if a != b {
                dtype_advisories.push(json!({"from":{"node":from_node,"port":from_port,"dtype":a},
                    "to":{"node":to_node,"port":to_port,"dtype":b}}));
            }
        }
        replacements.push(json!({"from":{"node":from_node,"port":from_port},"to":{"node":to_node,"port":to_port}}));
        true
    };
    let entry_edge = &graph.edges[entry];
    let exit_edge = &graph.edges[exit];
    let head = nodes[reordered[0]];
    let tail = nodes[*reordered.last().expect("nonempty chain")];
    let mut valid = append(&entry_edge.from.node, &entry_edge.from.port, &head.id, &head.inputs[0]);
    for pair in reordered.windows(2) {
        let left = nodes[pair[0]]; let right = nodes[pair[1]];
        valid = append(&left.id, &left.outputs[0], &right.id, &right.inputs[0]) && valid;
    }
    valid = append(&tail.id, &tail.outputs[0], &exit_edge.to.node, &exit_edge.to.port) && valid;
    if !valid { return reject("GRAPH_SHAPE_CONFLICT", "重排后的连接与已声明形状或端口冲突。".into()); }
    let mut proposal: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value, Err(error) => return reject("GRAPH_INVALID", error.to_string()),
    };
    let edges = proposal["edges"].as_array().expect("validated edge list");
    let mut updated = edges.iter().enumerate().filter(|(index, _)| !removed.contains(index)).map(|(_, edge)| edge.clone()).collect::<Vec<_>>();
    updated.extend(replacements);
    proposal["edges"] = json!(updated);
    let old_definitions = chain.iter().map(|id| nodes[id].definition.as_ref().expect("call definition").clone()).collect::<Vec<_>>();
    let new_definitions = reordered.iter().map(|id| nodes[id].definition.as_ref().expect("call definition").clone()).collect::<Vec<_>>();
    let mut dtype_updates = Vec::new();
    for cast in graph.nodes.iter().filter(|node| node.kind == "cast" && node.track_sequence) {
        let mut cursor = cast.id.as_str();
        let mut found = false;
        for _ in 0..graph.nodes.len() {
            let Some(edges) = outgoing.get(cursor).filter(|edges| edges.len() == 1) else { break };
            let next = graph.edges[edges[0]].to.node.as_str();
            if next == first { found = true; break }
            let Some(node) = nodes.get(next) else { break };
            if !matches!(node.kind.as_str(), "call" | "assert_flat") { break }
            cursor = next;
        }
        if !found { continue }
        let Some(start) = cast.dtype_definitions.windows(old_definitions.len()).position(|part| part == old_definitions) else {
            return reject("GRAPH_DTYPE_ORDER_CONFLICT", format!("节点 {} 的类型来源已不匹配，请先审核。", cast.id));
        };
        let mut sequence = cast.dtype_definitions.clone();
        sequence.splice(start..start + old_definitions.len(), new_definitions.clone());
        let Some(target) = proposal["nodes"].as_array_mut().and_then(|items| items.iter_mut().find(|item| item["id"] == cast.id)) else {
            return reject("GRAPH_INVALID", "找不到类型转换节点。".into());
        };
        target["dtype_definitions"] = json!(sequence);
        dtype_updates.push(cast.id.clone());
    }
    Report::ok("graph_reorder_plan", json!({"committed":false,"input_sha256":expected,"before":chain,"after":reordered,
        "graph":proposal,"dtype_advisories":dtype_advisories,"dtype_sequence_updates":dtype_updates,"runtime_verified":false,
        "note":"仅预览连接变更；显示分组成员不变，dtype 名称差异可能为别名，未知形状与运行行为未验证。"}))
}

// Generate a reviewable source proposal only. The input is reread and checked
// again after planning, so the topology and code always describe the same bytes.
pub(super) fn generate<B: Backend>(app: &Application<B>, path: &Path, expected: &str,
    output: Option<&str>, expected_source: Option<&str>, apply: bool, if_plan: Option<&str>) -> Report {
    let mut planned = plan(path, expected);
    if !planned.success() { planned.operation = "graph_generate".into(); return planned; }
    let fail_generate = |code, message| Report::error("graph_generate", code, message, "检查图声明并重新预览；未写入文件或执行用户代码。 ");
    if planned.data["status"] != "structurally_connected" {
        return fail_generate("GRAPH_INCOMPLETE", "图仍有未连接端口或缺失边界。".to_owned());
    }
    let bytes = match super::scaffold::read_recipe_for_update(path) {
        Ok(bytes) => bytes, Err(error) => return fail_generate("GRAPH_READ_FAILED", error),
    };
    if super::scaffold::sha256_bytes(&bytes) != expected {
        return fail_generate("GRAPH_CONFLICT", "图文件在规划期间发生变化。".to_owned());
    }
    let graph: Graph = match serde_json::from_slice(&bytes) {
        Ok(graph) => graph, Err(error) => return fail_generate("GRAPH_INVALID", error.to_string()),
    };
    if let Some(node) = graph.nodes.iter().find(|node| node.kind == "concat" && node.axis.is_none()) {
        return fail_generate("GRAPH_AXIS_REQUIRED", format!("拼接节点 {} 必须显式声明 axis。", node.id));
    }
    let nodes = graph.nodes.iter().map(|node| (node.id.as_str(), node)).collect::<BTreeMap<_, _>>();
    let connections = graph.edges.iter().map(|edge| ((edge.to.node.as_str(), edge.to.port.as_str()),
        (edge.from.node.as_str(), edge.from.port.as_str()))).collect::<BTreeMap<_, _>>();
    let mut used = graph.nodes.iter().flat_map(|node| node.definition.iter().chain(node.dtype_definitions.iter()).map(String::as_str)).collect::<HashSet<_>>();
    let definition_map = graph.definitions.iter().map(|item| (item.id.as_str(), item)).collect::<BTreeMap<_, _>>();
    let mut pending = used.iter().copied().collect::<Vec<_>>();
    while let Some(id) = pending.pop() { for child in &definition_map[id].children { if used.insert(child) { pending.push(child); } } }
    let active_definitions = graph.definitions.iter().filter(|definition| used.contains(definition.id.as_str())).collect::<Vec<_>>();
    let definitions = active_definitions.iter().enumerate().map(|(index, definition)| (definition.id.as_str(), index)).collect::<BTreeMap<_, _>>();
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(root) => root, Err(error) => return fail_generate("PROJECT_UNAVAILABLE", error.to_string()),
    };
    let mut resolved = Vec::new();
    let mut component_refs = Map::new();
    for definition in &active_definitions {
        if let Some(id) = definition.factory.strip_prefix("component:") {
            if id.split_once('@').is_none() {
                return fail_generate("GRAPH_COMPONENT_INVALID", format!("组件 {} 必须固定版本。", definition.id));
            }
            let descriptor = match app.components.resolve_layer(&root, id, &definition.params) {
                Ok(value) => value, Err(error) => return fail_generate("GRAPH_COMPONENT_INVALID", error),
            };
            let reference = match super::component_reference(&root, &descriptor) {
                Ok(value) => value, Err(error) => return fail_generate("GRAPH_COMPONENT_INVALID", error),
            };
            let factory = match descriptor["manifest"]["entrypoint"].as_str().filter(|value| python_entrypoint(value)) {
                Some(value) => value.to_owned(), None => return fail_generate("GRAPH_COMPONENT_INVALID", "注册组件入口无效。".to_owned()),
            };
            let params = match descriptor["resolved_params"].as_object() {
                Some(value) => value.clone(), None => return fail_generate("GRAPH_COMPONENT_INVALID", "组件参数解析结果无效。".to_owned()),
            };
            component_refs.insert(definition.id.clone(), reference);
            resolved.push((factory, params));
        } else {
            resolved.push((definition.factory.clone(), definition.params.clone()));
        }
    }
    let Some(order) = planned.data["topological_order"].as_array() else {
        return fail_generate("GRAPH_PLAN_INVALID", "缺少拓扑顺序。".to_owned());
    };
    let quote = |text: &str| serde_json::to_string(text).expect("string serialization");
    let checked = |expression: String, shape: Option<&Shape>, dtype: Option<&String>, location: String| -> String {
        let expression = match shape {
            None => expression,
            Some(shape) => {
                let dimensions = shape.iter().map(|dim| dim.map_or_else(|| "None".to_owned(), |value| value.to_string())).collect::<Vec<_>>().join(", ");
                format!("_check_shape({expression}, [{dimensions}], {})", quote(&location))
            }
        };
        match dtype {
            None => expression,
            Some(dtype) => format!("_check_dtype({expression}, {}, {})", quote(dtype), quote(&location)),
        }
    };
    let mut source = String::from(r#"# Editable Model-Maven graph proposal. Factories run when GraphModel is constructed.
# Calls sharing one definition share one module. Shapes and interfaces are not verified.
import importlib
import json
import torch
from torch import nn


def _make_component(reference, parameters, children=()):
    module, name = reference.split(":", 1)
    value = importlib.import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    component = value(*children, **json.loads(parameters))
    if not isinstance(component, nn.Module):
        raise TypeError(f"{reference} must construct torch.nn.Module")
    return component


def _check_shape(value, expected, location):
    # Declared contracts are checked against actual tensors only when executed.
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{location} must be a tensor for its declared shape contract")
    actual = tuple(value.shape)
    if len(actual) != len(expected) or any(size is not None and actual[index] != size for index, size in enumerate(expected)):
        raise ValueError(f"{location} expected shape {expected}, got {actual}")
    return value


class GraphModel(nn.Module):
    def __init__(self):
        super().__init__()
"#);
    if graph.nodes.iter().any(|node| !node.input_dtypes.is_empty() || !node.output_dtypes.is_empty()) {
        let dtype_helper = r#"

def _check_dtype(value, name, location):
    # Resolve the declaration in the selected runtime; do not cast implicitly.
    expected = getattr(torch, name, None)
    if not isinstance(expected, torch.dtype):
        raise ValueError(f"{location}: dtype {name!r} is unavailable in this PyTorch runtime")
    if not isinstance(value, torch.Tensor) or value.dtype != expected:
        actual = value.dtype if isinstance(value, torch.Tensor) else type(value).__name__
        raise TypeError(f"{location} expected dtype {name}, got {actual}")
    return value
"#;
        source = source.replacen("\n\nclass GraphModel", &format!("{dtype_helper}\n\nclass GraphModel"), 1);
    }
    let named_components = graph.component_storage.as_deref() == Some("named");
    let has_containers = active_definitions.iter().any(|definition| !definition.children.is_empty() || definition.children_style.is_some());
    let has_list_children = active_definitions.iter().any(|definition| definition.children_style.as_deref() == Some("list"));
    if has_list_children {
        source = source.replacen("def _make_component(reference, parameters, children=()):", "def _make_component(reference, parameters, children=(), children_style=None):", 1)
            .replacen("    component = value(*children, **json.loads(parameters))", "    component = value(children, **json.loads(parameters)) if children_style == 'list' else value(*children, **json.loads(parameters))", 1);
    }
    if !has_containers {
        // Preserve byte-for-byte generation for existing graphs. Identity
        // checks and conditional updates rely on this source remaining stable.
        source = source.replacen("def _make_component(reference, parameters, children=()):", "def _make_component(reference, parameters):", 1)
            .replacen("    component = value(*children, **json.loads(parameters))", "    component = value(**json.loads(parameters))", 1);
        source.push_str(if named_components { "        self.components = nn.ModuleDict({\n" } else { "        self.components = nn.ModuleList([\n" });
        for (definition, (factory, params)) in active_definitions.iter().zip(&resolved) {
            let params = match serde_json::to_string(params) {
                Ok(params) => params, Err(error) => return fail_generate("GRAPH_INVALID", error.to_string()),
            };
            if named_components {
                writeln!(source, "            {}: _make_component({}, {}),  # definition {}", quote(&format!("component_{}", definition.id)), quote(factory), quote(&params), definition.id).unwrap();
            } else {
                writeln!(source, "            _make_component({}, {}),  # definition {}", quote(factory), quote(&params), definition.id).unwrap();
            }
        }
    } else {
        // Stable DFS order constructs children first, while storage order remains
        // compatible with existing graph component indices and named keys.
        fn construction_order<'a>(id: &'a str, defs: &BTreeMap<&'a str, &'a Definition>, seen: &mut HashSet<&'a str>, order: &mut Vec<&'a str>) {
            if !seen.insert(id) { return; }
            for child in &defs[id].children { construction_order(child, defs, seen, order); }
            order.push(id);
        }
        let mut seen = HashSet::new();
        let mut construction = Vec::new();
        for definition in &active_definitions { construction_order(&definition.id, &definition_map, &mut seen, &mut construction); }
        source.push_str("        built = {}\n");
        for id in construction {
            let definition = definition_map[id];
            let (factory, params) = &resolved[definitions[id]];
            let params = match serde_json::to_string(params) {
                Ok(params) => params, Err(error) => return fail_generate("GRAPH_INVALID", error.to_string()),
            };
            let children = definition.children.iter().map(|id| format!("built[{}]", quote(id))).collect::<Vec<_>>().join(", ");
            if definition.children_style.as_deref() == Some("list") {
                writeln!(source, "        built[{}] = _make_component({}, {}, [{}], 'list')", quote(id), quote(factory), quote(&params), children).unwrap();
            } else {
                writeln!(source, "        built[{}] = _make_component({}, {}, [{}])", quote(id), quote(factory), quote(&params), children).unwrap();
            }
        }
        source.push_str(if named_components { "        self.components = nn.ModuleDict({\n" } else { "        self.components = nn.ModuleList([\n" });
        for definition in &active_definitions {
            if named_components {
                writeln!(source, "            {}: built[{}],  # definition {}", quote(&format!("component_{}", definition.id)), quote(&definition.id), definition.id).unwrap();
            } else {
                writeln!(source, "            built[{}],  # definition {}", quote(&definition.id), definition.id).unwrap();
            }
        }
    }
    let input_nodes = graph.nodes.iter().filter(|node| node.kind == "input").collect::<Vec<_>>();
    source.push_str(if named_components { "        })\n" } else { "        ])\n" });
    source.push_str("\n    def forward(self, *args, **kwargs):\n        # Accept either the graph's nested input mapping or named batch inputs.\n        if args:\n            if len(args) != 1 or kwargs:\n                raise TypeError('Pass one graph input mapping or named inputs, not both')\n            inputs = args[0]\n");
    if input_nodes.len() == 1 && input_nodes[0].outputs.len() == 1 {
        let node = input_nodes[0];
        writeln!(source, "            if not isinstance(inputs, dict):\n                inputs = {{{}: {{{}: inputs}}}}", quote(&node.id), quote(&node.outputs[0])).unwrap();
    }
    source.push_str("        else:\n");
    source.push_str("            inputs = {}\n            used = set()\n");
    for node in input_nodes {
        if node.outputs.len() == 1 {
            writeln!(source, "            if {} not in kwargs:\n                raise ValueError({})", quote(&node.id), quote(&format!("Missing graph input {}", node.id))).unwrap();
            writeln!(source, "            inputs[{}] = {{{}: kwargs[{}]}}", quote(&node.id), quote(&node.outputs[0]), quote(&node.id)).unwrap();
            writeln!(source, "            used.add({})", quote(&node.id)).unwrap();
        } else {
            writeln!(source, "            if {} in kwargs:", quote(&node.id)).unwrap();
            writeln!(source, "                inputs[{}] = kwargs[{}]  # mapping of named output ports", quote(&node.id), quote(&node.id)).unwrap();
            writeln!(source, "                used.add({})", quote(&node.id)).unwrap();
            source.push_str("            else:\n");
            writeln!(source, "                inputs[{}] = {{}}", quote(&node.id)).unwrap();
            for port in &node.outputs {
                let flat = format!("{}.{}", node.id, port);
                writeln!(source, "                if {} not in kwargs:\n                    raise ValueError({})", quote(&flat), quote(&format!("Missing graph input {flat}"))).unwrap();
                writeln!(source, "                inputs[{}][{}] = kwargs[{}]", quote(&node.id), quote(port), quote(&flat)).unwrap();
                writeln!(source, "                used.add({})", quote(&flat)).unwrap();
            }
        }
    }
    source.push_str("            if set(kwargs) != used:\n                raise ValueError(f'Unexpected graph input names: {sorted(set(kwargs) - used)}')\n");
    source.push_str("        values = {}\n        outputs = {}\n");
    let mut locations = Map::new();
    // The generated prefix and every node block end in a newline. Count the
    // prefix once and then only scan each appended block, rather than scanning
    // the entire growing source for every node.
    let mut next_line = source.bytes().filter(|byte| *byte == b'\n').count() + 1;
    for id in order {
        let Some(id) = id.as_str() else { return fail_generate("GRAPH_PLAN_INVALID", "拓扑节点无效。".to_owned()); };
        let Some(node) = nodes.get(id) else { return fail_generate("GRAPH_PLAN_INVALID", "拓扑节点不存在。".to_owned()); };
        locations.insert(id.to_owned(), json!(next_line));
        let block_start = source.len();
        writeln!(source, "        # node {} ({})", node.id, node.kind).unwrap();
        let incoming = node.inputs.iter().map(|port| {
            let (from_node, from_port) = connections[&(node.id.as_str(), port.as_str())];
            let value = format!("values[({}, {})]", quote(from_node), quote(from_port));
            checked(value, node.input_shapes.get(port), node.input_dtypes.get(port), format!("{}.{} input", node.id, port))
        }).collect::<Vec<_>>();
        match node.kind.as_str() {
            "input" => for port in &node.outputs {
                let value = format!("inputs[{}][{}]", quote(&node.id), quote(port));
                let value = checked(value, node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
            },
            "call" => {
                let index = definitions[node.definition.as_deref().expect("validated definition")];
                let key = if named_components { quote(&format!("component_{}", node.definition.as_deref().expect("validated definition"))) } else { index.to_string() };
                if node.call_style.as_deref() == Some("list") {
                    writeln!(source, "        result = self.components[{key}]([{}])", incoming.join(", ")).unwrap();
                } else if node.call_style.as_deref() == Some("positional") {
                    writeln!(source, "        result = self.components[{key}]({})", incoming.join(", ")).unwrap();
                } else {
                    let kwargs = node.inputs.iter().zip(&incoming).map(|(port, value)| format!("{}: {value}", quote(port))).collect::<Vec<_>>().join(", ");
                    writeln!(source, "        result = self.components[{key}](**{{{kwargs}}})").unwrap();
                }
                if node.outputs.len() == 1 && node.output_style.is_none() {
                    let port = &node.outputs[0];
                    let value = checked("result".to_owned(), node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                    writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
                } else {
                    if node.output_style.as_deref() == Some("sequence") {
                        writeln!(source, "        if not isinstance(result, (tuple, list)) or len(result) != {}:", node.outputs.len()).unwrap();
                        writeln!(source, "            raise ValueError(\"Component output sequence does not match node {}\")", node.id).unwrap();
                    } else {
                        writeln!(source, "        if not isinstance(result, dict) or set(result) != {{{}}}:", node.outputs.iter().map(|port| quote(port)).collect::<Vec<_>>().join(", ")).unwrap();
                        writeln!(source, "            raise ValueError(\"Component output ports do not match node {}\")", node.id).unwrap();
                    }
                    for (index, port) in node.outputs.iter().enumerate() {
                        let lookup = if node.output_style.as_deref() == Some("sequence") { index.to_string() } else { quote(port) };
                        let value = checked(format!("result[{lookup}]"), node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                        writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
                    }
                }
            },
            "cast" => {
                let dtype = if !node.dtype_definitions.is_empty() {
                    let components = node.dtype_definitions.iter().map(|definition| {
                        let index = definitions[definition.as_str()];
                        let key = if named_components { quote(&format!("component_{definition}")) } else { index.to_string() };
                        format!("self.components[{key}]")
                    }).collect::<Vec<_>>().join(", ");
                    writeln!(source, "        parameter = next((p for component in [{components}] for p in component.parameters()), None)").unwrap();
                    "parameter.dtype if parameter is not None else torch.get_default_dtype()"
                } else if let Some(definition) = node.definition.as_deref() {
                    let index = definitions[definition];
                    let key = if named_components { quote(&format!("component_{definition}")) } else { index.to_string() };
                    writeln!(source, "        parameter = next(self.components[{key}].parameters(), None)").unwrap();
                    "parameter.dtype if parameter is not None else torch.get_default_dtype()"
                } else { "torch.get_default_dtype()" };
                let port = &node.outputs[0];
                let value = format!("{}.to(dtype={dtype})", incoming[0]);
                let value = checked(value, node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
            },
            "clone" => {
                let port = &node.outputs[0];
                let value = checked(format!("{}.clone()", incoming[0]), node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
            },
            "mean_dynamic" => {
                writeln!(source, "        value = {}", incoming[0]).unwrap();
                source.push_str("        if not (value.is_floating_point() or value.is_complex()):\n            value = value.to(dtype=torch.get_default_dtype())\n");
                let dimensions = node.axes.iter().map(|axis| axis + 1).collect::<Vec<_>>();
                let port = &node.outputs[0];
                let value = format!("value.mean(dim=tuple({dimensions:?}), keepdim=True)");
                let value = checked(value, node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
            },
            "assert_flat" => {
                writeln!(source, "        value = {}\n        reference = {}", incoming[0], incoming[1]).unwrap();
                writeln!(source, "        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != reference.shape[0] or value.shape[1] != {}:", node.width.expect("validated width")).unwrap();
                writeln!(source, "            raise ValueError({})", quote(&format!("Node {} expected a flat tensor with the original batch size and width {}", node.id, node.width.expect("validated width")))).unwrap();
                let port = &node.outputs[0];
                let value = checked("value".to_owned(), node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
            },
            "assert_batch" => {
                writeln!(source, "        value = {}\n        reference = {}", incoming[0], incoming[1]).unwrap();
                writeln!(source, "        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != reference.shape[0]:").unwrap();
                writeln!(source, "            raise ValueError({})", quote(&format!("Node {} expected a rank-two tensor with the original batch size", node.id))).unwrap();
                let port = &node.outputs[0];
                let value = checked("value".to_owned(), node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
            },
            "after" => {
                let port = &node.outputs[0];
                let value = checked(incoming[0].clone(), node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
            },
            "add" => {
                writeln!(source, "        parts = [{}]", incoming.join(", ")).unwrap();
                source.push_str("        if any(not isinstance(part, torch.Tensor) or part.shape != parts[0].shape for part in parts):\n            raise ValueError(\"Add requires tensors of equal shapes\")\n        result = parts[0]\n        for part in parts[1:]:\n            result = result + part\n");
                let port = &node.outputs[0];
                let value = checked("result".to_owned(), node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
            },
            "concat" => {
                let port = &node.outputs[0];
                let value = format!("torch.cat([{}], dim={})", incoming.join(", "), node.axis.expect("checked axis"));
                let value = checked(value, node.output_shapes.get(port), node.output_dtypes.get(port), format!("{}.{} output", node.id, port));
                writeln!(source, "        values[({}, {})] = {value}", quote(&node.id), quote(port)).unwrap();
            },
            "output" => writeln!(source, "        outputs[{}] = {}", quote(node.output_name.as_deref().unwrap_or(&node.id)), incoming[0]).unwrap(),
            _ => return fail_generate("GRAPH_PLAN_INVALID", "拓扑节点类型无效。".to_owned()),
        }
        next_line += source.as_bytes()[block_start..].iter().filter(|byte| **byte == b'\n').count();
    }
    source.push_str("        return outputs\n\n\ndef model():\n    return GraphModel()\n");
    let objective_values = graph.objectives.iter().map(|objective| {
        let node = nodes[objective.output.as_str()];
        json!({"output":node.output_name.as_deref().unwrap_or(&node.id),"target":objective.target,
            "loss":objective.loss,"weight":objective.weight})
    }).collect::<Vec<_>>();
    let objectives = serde_json::to_string(&objective_values).expect("validated objectives serialize");
    writeln!(source, "\n\nGRAPH_OBJECTIVES = json.loads({})", quote(&objectives)).unwrap();
    let references = serde_json::to_string(&component_refs).expect("resolved references serialize");
    writeln!(source, "GRAPH_COMPONENT_REFS = json.loads({})  # provenance at generation time", quote(&references)).unwrap();
    source.push_str(include_str!("templates/graph_step.py"));
    let source_sha256 = super::scaffold::sha256_bytes(source.as_bytes());
    let module = match output {
        Some(file) => match file.strip_suffix(".py").filter(|name| python_identifier(name) && name.is_ascii() && name.len() <= 128) {
            Some(name) => Some(name),
            None => return fail_generate("GRAPH_OUTPUT_INVALID", "输出须为项目根目录内的 ASCII Python 模块名，例如 my_graph.py。".to_owned()),
        },
        None => None,
    };
    if let Some(revision) = expected_source {
        let Some(file) = output else { return fail_generate("GRAPH_OUTPUT_REQUIRED", "更新源码必须指定输出文件。".into()); };
        let target = root.join(file);
        let metadata = match std::fs::symlink_metadata(&target) {
            Ok(value) if value.is_file() && value.len() <= 16 * 1024 * 1024 => value,
            _ => return fail_generate("GRAPH_SOURCE_INVALID", "待更新源码必须是项目内不超过 16 MiB 的普通文件。".into()),
        };
        let _ = metadata;
        match std::fs::read(&target) {
            Ok(contents) if contents.len() <= 16 * 1024 * 1024 && super::scaffold::sha256_bytes(&contents) == revision => (),
            _ => return fail_generate("GRAPH_SOURCE_CONFLICT", "待更新源码已变化或不可读取；请重新审核。".into()),
        }
    }
    // Prefixes describe this graph only. They are not an exhaustive checkpoint
    // migration plan and do not prove which graph produced an older checkpoint.
    let component_prefixes = active_definitions.iter().enumerate().map(|(index, definition)| {
        json!({"definition":definition.id,"legacy":format!("components.{index}."),
            "named":format!("components.component_{}.", definition.id)})
    }).collect::<Vec<_>>();
    let proposal = json!({"version":1,"project_root":root,"input_sha256":expected,"source_sha256":source_sha256,
        "output":output,"previous_source_sha256":expected_source,"model":module.map(|name| format!("{name}:model")),"node_lines":locations,"components":component_refs,
        "component_storage":if named_components {"named"} else {"legacy"},"component_prefixes":component_prefixes,
        "migration_note":"当前图的前缀建议，不证明旧检查点对应此图；迁移仍需完整状态键、形状、类型和共享关系审核。"});
    let plan_revision = match crate::domain::input::content_revision(&proposal) {
        Ok(revision) => revision, Err(error) => return fail_generate("GRAPH_PLAN_INVALID", error),
    };
    if !apply {
        return Report::ok("graph_generate", json!({"committed":false,"input_sha256":expected,"source_sha256":source_sha256,
            "source":source,"node_lines":locations,"plan":proposal,"plan_revision":plan_revision,
            "shape_status":"conditional_on_declared_contracts","runtime_verified":false,
            "note":"可审阅的 PyTorch 源码草案；工厂仅在实例化时执行，尚未写入或验证。"}));
    }
    if if_plan != Some(plan_revision.as_str()) {
        return fail_generate("GRAPH_PLAN_CONFLICT", "源码、输出位置或项目已变化；请重新预览。".to_owned());
    }
    let Some(file) = output else { return fail_generate("GRAPH_OUTPUT_REQUIRED", "保存时必须指定输出文件。".to_owned()); };
    if let Some(previous) = expected_source {
        let _lock = match app.source_edits.lock(&root) {
            Ok(lock) => lock, Err(error) => return fail_generate("GRAPH_LOCK_FAILED", error),
        };
        if super::scaffold::read_recipe_for_update(path).map_or(true, |value| super::scaffold::sha256_bytes(&value) != expected) {
            return fail_generate("GRAPH_CONFLICT", "图在提交源码时已变化。".into());
        }
        let target = root.join(file);
        let current = match std::fs::symlink_metadata(&target) {
            Ok(metadata) if metadata.is_file() && metadata.len() <= 16 * 1024 * 1024 => std::fs::read(&target).ok(),
            _ => None,
        };
        if current.as_ref().is_none_or(|bytes| bytes.len() > 16 * 1024 * 1024 || super::scaffold::sha256_bytes(bytes) != previous) {
            return fail_generate("GRAPH_SOURCE_CONFLICT", "源码在提交时已变化；未覆盖。".into());
        }
        let unchanged = previous == source_sha256;
        let receipt = if unchanged { Value::Null } else {
            match app.source_edits.commit_batch(&root, &[json!({"path":file,"before_sha256":previous,"contents":source})]) {
                Ok(receipt) => receipt,
                Err(error) => return fail_generate("GRAPH_SOURCE_WRITE_FAILED", error),
            }
        };
        return Report::ok("graph_generate", json!({"committed":true,"updated":!unchanged,"unchanged":unchanged,
            "previous_source_sha256":previous,"input_sha256":expected,"source_sha256":source_sha256,
            "plan_revision":plan_revision,"output":file,"model":module.map(|name| format!("{name}:model")),
            "source_receipt":receipt,"runtime_verified":false,"note":"条件更新源码；未修改项目配置或运行模型。"}));
    }
    match app.project_source_writer.write_new(&root, Path::new(file), source.as_bytes()) {
        Ok(_) => Report::ok("graph_generate", json!({"committed":true,"input_sha256":expected,"source_sha256":source_sha256,
            "plan_revision":plan_revision,"output":file,"model":module.map(|name| format!("{name}:model")),
            "runtime_verified":false,"note":"已创建新源码；未修改项目配置或运行模型。"})),
        Err((code, error)) => fail_generate(code, error),
    }
}


/// Run draft-generated source in a fresh worker against a captured input.
pub(super) fn draft_probe<B: Backend>(app: &Application<B>, cli: &Cli, path: &Path, expected: &str,
    snapshot: &Path, input_digest: &str, batch: Option<&str>, model_device: Option<&str>, expected_config: &str) -> Report {
    let fail = |code, message: String| Report::error("graph_draft_probe", code, message,
        "草稿、源码和快照均未修改；检查图、输入及运行环境后重试。");
    let digest_valid = |value: &str| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !digest_valid(input_digest) || (expected_config != "absent" && !digest_valid(expected_config)) {
        return fail("GRAPH_DRAFT_IDENTITY_INVALID", "输入或项目配置摘要无效。".into());
    }
    if model_device.is_some_and(|value| value.trim() != value || value.is_empty() || value.len() > 128 || value.chars().any(char::is_control)) {
        return fail("PROBE_DEVICE_INVALID", "预览模型设备标识无效。".into());
    }
    if cli.variant.is_some() || cli.capture_input_to.is_some() || cli.replay_input_from.is_some() {
        return fail("GRAPH_DRAFT_OPTIONS_UNSUPPORTED", "草稿预览不加载变体，也不接受全局捕获或重放选项。".into());
    }
    let python = match require_python(cli, "graph_draft_probe") { Ok(python) => python, Err(report) => return report };
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(root) => root, Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
    };
    let config = match app.project_config.load(&root) { Ok(value) => value, Err(error) => return fail("PROJECT_CONFIG_INVALID", error) };
    let revision = match config.as_ref().map(crate::domain::input::content_revision).transpose() {
        Ok(value) => value.unwrap_or_else(|| "absent".to_owned()),
        Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
    };
    if revision != expected_config { return fail("PROJECT_CONFIG_CONFLICT", "项目配置已变化。".into()); }
    let proposal = generate(app, path, expected, None, None, false, None);
    if !proposal.success() { let mut report = proposal; report.operation = "graph_draft_probe".into(); return report; }
    let Some(source) = proposal.data["source"].as_str() else { return fail("GRAPH_DRAFT_SOURCE_INVALID", "未取得草稿源码。".into()); };
    let Some(source_digest) = proposal.data["source_sha256"].as_str() else { return fail("GRAPH_DRAFT_SOURCE_INVALID", "未取得草稿源码摘要。".into()); };
    let Some(snapshot_path) = snapshot.to_str() else { return fail("GRAPH_DRAFT_INPUT_INVALID", "快照路径无法表示为 UTF-8。".into()); };
    let mut runtime = json!({"python_executable":python,"timeout_seconds":cli.timeout.unwrap_or(30)});
    if let Some(config) = &config { super::run::apply_project_runtime_defaults(config, &mut runtime); }
    if let Err(mut report) = super::run::resolve_worker_directory("probe", &root, &mut runtime, false) { report.operation = "graph_draft_probe".into(); return report; }
    if let Err(mut report) = super::run::resolve_worker_import_roots("probe", &root, &mut runtime, false) { report.operation = "graph_draft_probe".into(); return report; }
    let request = json!({"protocol_version":PROTOCOL_VERSION,"operation":"graph_draft_probe",
        "required_features":["graph.draft_replay_v1"],"project_root":root,"generated_source":source,
        "generated_source_sha256":source_digest,"draft_graph_sha256":expected,"replay_input_from":snapshot_path,
        "expected_input_content":input_digest,"batch":batch,"batch_params":{},"training":false,"seed":cli.seed,
        "probe_device":model_device,"runtime":runtime});
    let mut result = app.backend.execute(&request);
    if !result.success() && result.data["draft_graph_sha256"] == expected
        && result.data["draft_source_sha256"] == source_digest {
        let line = result.data["draft_forward_line"].as_u64();
        let end_line = source.lines().position(|line| line == "        return outputs").map(|index| index as u64 + 1);
        if let (Some(line), Some(end_line), Some(locations)) = (line, end_line, proposal.data["node_lines"].as_object()) {
            if line < end_line {
                if let Some((node, _)) = locations.iter().filter_map(|(node, start)| start.as_u64().filter(|start| *start <= line && *start < end_line).map(|start| (node, start))).max_by_key(|(_, start)| *start) {
                    result.data["possible_node_id"] = json!(node);
                    result.data["location_scope"] = json!("generated_forward_stack_frame_not_root_cause");
                }
            }
        }
    }
    if result.success() && (result.data["draft_graph_sha256"] != expected || result.data["draft_source_sha256"] != source_digest
        || result.data["preview_scope"] != "fresh-worker-graph-draft" || result.data["weights_loaded"] != false
        || result.data["batch_content"]["algorithm"] != "sha256-forward-input-v1"
        || result.data["batch_content"]["digest"] != input_digest || result.data["batch_snapshot"]["status"] != "replayed"
        || result.data["batch_snapshot"]["format"] != "model-maven-forward-input-snapshot-v1"
        || result.data["forward_execution"] != "executed") {
        return fail("GRAPH_DRAFT_EVIDENCE_INVALID", "草稿前向或输入重放回执不完整。".into());
    }
    result.operation = "graph_draft_probe".into();
    result
}

/// Bind a reviewed, already-generated graph source and optionally its data in
/// one versioned project-config update. No model or data factory is executed.
pub(super) fn bind_project<B: Backend>(app: &Application<B>, graph_path: &Path, expected_graph: &str,
    source: &str, expected_config: &str, include_data: bool, apply: bool, if_plan: Option<&str>) -> Report {
    use fs2::FileExt as _;
    use std::io::Read as _;
    let fail = |code, message: String| Report::error("graph_bind", code, message,
        "重新审核图、源码和项目配置；未执行模型或数据工厂。 ");
    let root = match std::env::current_dir().and_then(|value| value.canonicalize()) {
        Ok(value) => value, Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
    };
    let graph_meta = std::fs::symlink_metadata(graph_path);
    let graph_canonical = graph_path.canonicalize();
    if !graph_meta.is_ok_and(|metadata| metadata.is_file())
        || !graph_canonical.as_ref().is_ok_and(|path| path.parent() == Some(root.as_path())) {
        return fail("GRAPH_PATH_INVALID", "图必须是项目根目录中的普通文件。".into());
    }
    let generated = generate(app, graph_path, expected_graph, Some(source), None, false, None);
    if !generated.success() { let mut report = generated; report.operation = "graph_bind".into(); return report; }
    let Some(expected_source) = generated.data["source_sha256"].as_str() else {
        return fail("GRAPH_SOURCE_INVALID", "无法取得生成源码摘要。".into());
    };
    let Some(model) = generated.data["plan"]["model"].as_str() else {
        return fail("GRAPH_SOURCE_INVALID", "无法确定源码模块入口。".into());
    };
    let read_source = || -> Result<Vec<u8>, String> {
        let path = root.join(source);
        let meta = std::fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if !meta.is_file() || meta.len() > 4 * 1024 * 1024 { return Err("源码须为不超过 4 MiB 的普通文件。".into()); }
        let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
        let mut bytes = Vec::new();
        file.take(4 * 1024 * 1024 + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
        if bytes.len() > 4 * 1024 * 1024 { return Err("源码超过 4 MiB。".into()); }
        Ok(bytes)
    };
    let bytes = match read_source() { Ok(value) => value, Err(error) => return fail("GRAPH_SOURCE_INVALID", error) };
    if super::scaffold::sha256_bytes(&bytes) != expected_source {
        return fail("GRAPH_SOURCE_CONFLICT", "当前源码与该图的生成结果不一致。".into());
    }
    let graph_bytes = match super::scaffold::read_recipe_for_update(graph_path) {
        Ok(value) if super::scaffold::sha256_bytes(&value) == expected_graph => value,
        Ok(_) => return fail("GRAPH_CONFLICT", "图声明已变化。".into()),
        Err(error) => return fail("GRAPH_READ_FAILED", error),
    };
    let graph: Graph = match serde_json::from_slice(&graph_bytes) {
        Ok(value) => value, Err(error) => return fail("GRAPH_INVALID", error.to_string()),
    };
    if include_data && graph.data.is_none() { return fail("GRAPH_DATA_REQUIRED", "图中没有数据声明。".into()); }
    let current = match app.project_config.load(&root) {
        Ok(value) => value,
        Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
    };
    let revision = match &current {
        Some(value) => match crate::domain::input::content_revision(value) {
            Ok(value) => value, Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
        },
        None => "absent".to_owned(),
    };
    if revision != expected_config { return fail("PROJECT_CONFIG_CONFLICT", "项目配置已变化。".into()); }
    let mut candidate = current.clone().unwrap_or_else(|| json!({"schema_version":1,"model":model,
        "batch":null,"batch_binding":null,"model_injections":{}}));
    candidate["model"] = json!(model);
    candidate["model_params"] = json!({});
    candidate["model_injections"] = json!({});
    if include_data {
        let data = graph.data.as_ref().expect("checked data");
        candidate["batch"] = json!(data.factory);
        candidate["batch_binding"] = json!(data.binding);
        candidate["batch_params"] = json!(data.params);
    }
    let candidate_revision = match crate::domain::input::content_revision(&candidate) {
        Ok(value) => value, Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
    };
    let plan = json!({"version":1,"project_root":root,"graph_sha256":expected_graph,"source":source,
        "source_sha256":expected_source,"from_revision":revision,"candidate_revision":candidate_revision,
        "model":model,"include_data":include_data,"batch":if include_data {graph.data.as_ref().map(|data| data.factory.as_str())} else {None},
        "creates_project":current.is_none(),"clears_model_params_and_injections":current.is_some(),
        // These declarations remain in the graph and must not be mistaken for
        // configured or enabled project runtime settings.
        "graph_training_review": {
            "declared":graph.training.is_some(),
            "mode":graph.training.as_ref().and_then(|training| training.mode.as_deref()),
            "step":graph.training.as_ref().and_then(|training| training.step.as_deref()),
            "trial_steps":graph.training.as_ref().and_then(|training| training.trial_steps),
            "objectives":graph.objectives,
            "project_binding":"not_applied"
        },
        "graph_supervision_review": {
            "reference":graph.supervision,
            "semantic_port_count":graph.nodes.iter().map(|node| node.semantics.len()).sum::<usize>(),
            "draft_rule_count":graph.nodes.iter().flat_map(|node| node.draft_rules.values()).map(Vec::len).sum::<usize>(),
            "runtime_binding":"not_applied"
        },
        "unchanged":current.as_ref() == Some(&candidate)});
    let plan_revision = match crate::domain::input::content_revision(&plan) {
        Ok(value) => value, Err(error) => return fail("GRAPH_PLAN_INVALID", error),
    };
    if !apply {
        return Report::ok("graph_bind", json!({"committed":false,"plan":plan,"plan_revision":plan_revision,
            "runtime_verified":false,"note":"单次配置更新；图、源码、训练声明及监督引用不被改写，工厂未执行。"}));
    }
    if if_plan != Some(plan_revision.as_str()) { return fail("GRAPH_PLAN_CONFLICT", "绑定计划已变化。".into()); }
    let _source_lock = match app.source_edits.lock(&root) {
        Ok(lock) => lock, Err(error) => return fail("GRAPH_LOCK_FAILED", error),
    };
    let mut options = std::fs::OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt as _; options.mode(0o600).custom_flags(libc::O_NOFOLLOW); }
    let graph_lock = match options.open(root.join(".model-maven-graph.lock")) {
        Ok(lock) => lock, Err(error) => return fail("GRAPH_LOCK_FAILED", error.to_string()),
    };
    if graph_lock.try_lock_exclusive().is_err() { return fail("GRAPH_LOCK_FAILED", "另一个图保存正在进行。".into()); }
    if super::scaffold::read_recipe_for_update(graph_path).map_or(true, |bytes| super::scaffold::sha256_bytes(&bytes) != expected_graph)
        || read_source().map_or(true, |bytes| super::scaffold::sha256_bytes(&bytes) != expected_source) {
        return fail("GRAPH_BIND_CONFLICT", "图或源码在审核期间变化。".into());
    }
    if current.as_ref() == Some(&candidate) {
        if app.project_config.load(&root).ok().flatten().as_ref() != Some(&candidate) {
            return fail("PROJECT_CONFIG_CONFLICT", "项目配置在审核期间变化。".into());
        }
        return Report::ok("graph_bind", json!({"committed":true,"unchanged":true,"plan_revision":plan_revision,"config_revision":revision,"runtime_verified":false}));
    }
    if current.is_none() {
        return match app.project_config.create(&root, &candidate) {
            Ok(_) => Report::ok("graph_bind", json!({"committed":true,"created":true,"unchanged":false,
                "plan_revision":plan_revision,"config_revision":candidate_revision,"model":model,
                "include_data":include_data,"runtime_verified":false})),
            Err(error) => fail("PROJECT_CONFIG_WRITE_FAILED", error),
        };
    }
    match app.project_config.update_versioned(&root, &revision, &candidate) {
        Ok(update) => Report::ok("graph_bind", json!({"committed":true,"unchanged":false,"plan_revision":plan_revision,
            "config_revision":update.revision,"model":model,"include_data":include_data,"runtime_verified":false})),
        Err(ProjectConfigUpdateError::Conflict { .. }) => fail("PROJECT_CONFIG_CONFLICT", "项目配置在提交时变化。".into()),
        Err(error) => fail("PROJECT_CONFIG_WRITE_FAILED", error.message()),
    }
}

/// Read-only overview of the three separate graph delivery stages. This does
/// not reserve files or authorize any later writes; each stage must replan.
pub(super) fn delivery_plan<B: Backend>(app: &Application<B>, draft: &Path, expected_draft: &str,
    output: &str, expected_base: &str, source: &str) -> Report {
    let fail = |code, message: String| Report::error("graph_delivery_plan", code, message,
        "重新核对草稿及项目状态；本次未写入任何文件。 ");
    let save_review = save(app, draft, expected_draft, output, expected_base, false, None);
    if !save_review.success() { let mut report = save_review; report.operation = "graph_delivery_plan".into(); return report; }
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(root) => root, Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
    };
    let valid_source = source.strip_suffix(".py").is_some_and(|name| name.is_ascii() && name.len() <= 128 && python_identifier(name));
    if !valid_source { return fail("GRAPH_OUTPUT_INVALID", "源码须为项目根目录内的 ASCII Python 模块文件。".into()); }
    let source_path = root.join(source);
    let source_revision = match std::fs::symlink_metadata(&source_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Ok(meta) if meta.is_file() && meta.len() <= 16 * 1024 * 1024 => match std::fs::read(&source_path) {
            Ok(bytes) if bytes.len() <= 16 * 1024 * 1024 => Some(super::scaffold::sha256_bytes(&bytes)),
            _ => return fail("GRAPH_SOURCE_INVALID", "无法读取源码或源码超过上限。".into()),
        },
        _ => return fail("GRAPH_SOURCE_INVALID", "源码须为普通文件且不超过 16 MiB。".into()),
    };
    let mut source_review = generate(app, draft, expected_draft, Some(source), source_revision.as_deref(), false, None);
    // The caller can explicitly request a full source preview separately.
    if let Some(data) = source_review.data.as_object_mut() { data.remove("source"); }
    let config = match app.project_config.load(&root) {
        Ok(config) => config, Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
    };
    let config_revision = match &config {
        Some(config) => match crate::domain::input::content_revision(config) {
            Ok(value) => value, Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
        },
        None => "absent".to_owned(),
    };
    let graph_bytes = match super::scaffold::read_recipe_for_update(draft) {
        Ok(bytes) if super::scaffold::sha256_bytes(&bytes) == expected_draft => bytes,
        Ok(_) => return fail("GRAPH_CONFLICT", "审核期间草稿已变化。".into()),
        Err(error) => return fail("GRAPH_READ_FAILED", error),
    };
    let graph: Graph = match serde_json::from_slice(&graph_bytes) {
        Ok(graph) => graph, Err(error) => return fail("GRAPH_INVALID", error.to_string()),
    };
    let graph_data = graph.data.as_ref();
    let matches_data = match (graph_data, config.as_ref()) {
        (Some(data), Some(current)) => current["batch"] == data.factory && current["batch_binding"] == json!(data.binding)
            && current["batch_params"] == json!(data.params),
        _ => false,
    };
    let overview = json!({"draft_sha256":expected_draft,"graph_output":output,"graph_base":expected_base,
        "save_review":save_review.data,"source_output":source,
        "source_base":source_revision.as_deref().unwrap_or("absent"),"source_review":source_review,
        "project_config_revision":config_revision,"project_model":config.as_ref().and_then(|value| value["model"].as_str()),
        "proposed_model":format!("{}:model", source.strip_suffix(".py").expect("validated source")),
        "graph_data_factory":graph_data.map(|data| data.factory.as_str()),"graph_data_matches_project":matches_data,
        "losses":graph.objectives,"training_declared":graph.training.is_some(),"supervision_reference":graph.supervision,
        "draft_rule_count":graph.nodes.iter().flat_map(|node| node.draft_rules.values()).map(Vec::len).sum::<usize>()});
    let revision = match crate::domain::input::content_revision(&overview) {
        Ok(value) => value, Err(error) => return fail("GRAPH_PLAN_INVALID", error),
    };
    Report::ok("graph_delivery_plan", json!({"committed":false,"plan_revision":revision,"plan":overview,
        "note":"只读联合审核；不锁定或写入文件。保存图、生成源码及绑定项目仍是独立条件提交；训练与监督不会自动启用。"}))
}

/// Persist a graph draft under a cooperative lock and a reviewed revision.
/// Semantic diagnostics are reported but do not prevent saving an incomplete
/// or invalid draft; generation remains guarded by the separate graph planner.
pub(super) fn save<B: Backend>(app: &Application<B>, draft: &Path, expected_draft: &str,
    output: &str, expected_base: &str, apply: bool, if_plan: Option<&str>) -> Report {
    use fs2::FileExt as _;
    let fail_save = |code, message| Report::error("graph_save", code, message, "检查草稿、目标版本与计划后重试；不会覆盖未知内容。 ");
    let valid_name = output.strip_suffix(".json").is_some_and(|name| {
        !name.is_empty() && name.len() <= 128 && name.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    });
    if !valid_name { return fail_save("GRAPH_OUTPUT_INVALID", "图文件须为项目根目录下的简单 JSON 文件名。".to_owned()); }
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(root) => root, Err(error) => return fail_save("PROJECT_UNAVAILABLE", error.to_string()),
    };
    let bytes = match super::scaffold::read_recipe_for_update(draft) {
        Ok(bytes) => bytes, Err(error) => return fail_save("GRAPH_READ_FAILED", error),
    };
    if super::scaffold::sha256_bytes(&bytes) != expected_draft {
        return fail_save("GRAPH_CONFLICT", "草稿内容已变化。".to_owned());
    }
    let parsed: Graph = match serde_json::from_slice(&bytes) {
        Ok(graph) => graph, Err(error) => return fail_save("GRAPH_INVALID", error.to_string()),
    };
    if parsed.version != 1 { return fail_save("GRAPH_INVALID", "不支持此图声明版本。".to_owned()); }
    if parsed.legacy_import.as_ref().is_some_and(|snapshot| !snapshot.valid()) {
        return fail_save("GRAPH_LEGACY_INVALID", "旧声明快照损坏或摘要不匹配。".to_owned());
    }
    let target = root.join(output);
    let read_base = || -> Result<Option<Vec<u8>>, String> {
        match std::fs::symlink_metadata(&target) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.to_string()),
            Ok(_) => super::scaffold::read_recipe_for_update(&target).map(Some),
        }
    };
    let base = match read_base() { Ok(value) => value, Err(error) => return fail_save("GRAPH_OUTPUT_INVALID", error) };
    if let Some(previous) = &base {
        let existing = serde_json::from_slice::<Graph>(previous);
        if existing.as_ref().map_or(true, |graph| graph.version != 1) {
            return fail_save("GRAPH_OUTPUT_INVALID", "已有文件不是受支持的图声明，不能作为图草稿覆盖。".to_owned());
        }
    }
    let base_revision = base.as_ref().map(|bytes| super::scaffold::sha256_bytes(bytes)).unwrap_or_else(|| "absent".to_owned());
    if base_revision != expected_base { return fail_save("GRAPH_BASE_CONFLICT", "图文件已变化；请重新打开并合并。".to_owned()); }
    let mut structural = plan(draft, expected_draft);
    if structural.diagnostics.iter().any(|item| item.code == "GRAPH_CONFLICT" || item.code == "GRAPH_READ_FAILED") {
        structural.operation = "graph_save".into(); return structural;
    }
    let proposal = json!({"version":1,"project_root":root,"output":output,"base_revision":base_revision,
        "draft_sha256":expected_draft,"unchanged":base.as_deref() == Some(bytes.as_slice()),"structural_review":structural});
    let plan_revision = match crate::domain::input::content_revision(&proposal) {
        Ok(revision) => revision, Err(error) => return fail_save("GRAPH_PLAN_INVALID", error),
    };
    if !apply {
        return Report::ok("graph_save", json!({"committed":false,"plan_revision":plan_revision,"plan":proposal,
            "note":"仅检查草稿与保存目标；图结构诊断不代表运行验证。"}));
    }
    if if_plan != Some(plan_revision.as_str()) { return fail_save("GRAPH_PLAN_CONFLICT", "保存计划已变化，请重新预览。".to_owned()); }
    let mut options = std::fs::OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)] {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let lock = match options.open(root.join(".model-maven-graph.lock")) {
        Ok(lock) => lock, Err(error) => return fail_save("GRAPH_LOCK_FAILED", error.to_string()),
    };
    if lock.try_lock_exclusive().is_err() { return fail_save("GRAPH_LOCK_FAILED", "另一个图保存正在进行。".to_owned()); }
    let latest = match read_base() { Ok(value) => value, Err(error) => return fail_save("GRAPH_OUTPUT_INVALID", error) };
    if latest != base { return fail_save("GRAPH_BASE_CONFLICT", "图文件在保存期间变化。".to_owned()); }
    let result = match base {
        None => app.project_source_writer.write_new(&root, Path::new(output), &bytes).map(|_| ()).map_err(|(_, error)| error),
        Some(previous) if previous == bytes => Ok(()),
        Some(previous) => super::scaffold::replace_recipe(&target, &previous, &bytes),
    };
    if let Err(error) = result { return fail_save("GRAPH_WRITE_FAILED", error); }
    drop(lock);
    Report::ok("graph_save", json!({"committed":true,"output":output,"base_revision":base_revision,
        "graph_sha256":expected_draft,"plan_revision":plan_revision,"runtime_verified":false,
        "note":"图草稿已保存；未运行模型、生成源码或修改项目配置。"}))
}

// Staged delivery intentionally does not claim a cross-file atomic commit.
// The private journal preserves reviewed inputs before the first write; a
// resume checks actual pre/post identities instead of trusting a progress flag.
pub(super) fn delivery_start<B: Backend>(app: &Application<B>, draft: &Path, expected: &str,
    output: &str, base: &str, source: &str, reviewed: &str) -> Report {
    use super::graph_delivery_journal::{DeliveryJournal, DeliveryJournalStore};
    let failure = |code, message: String| Report::error("graph_delivery", code, message,
        "未开始新提交；检查已有交付记录与项目文件。 ");
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(root) => root, Err(error) => return failure("PROJECT_UNAVAILABLE", error.to_string()),
    };
    let store = match DeliveryJournalStore::lock(&root) {
        Ok(store) => store, Err(error) => return failure("GRAPH_DELIVERY_LOCK", error),
    };
    match store.load() {
        Ok(None) => (),
        Ok(Some(_)) => return failure("GRAPH_DELIVERY_PENDING", "已有未完成交付，请先恢复。".into()),
        Err(error) => return failure("GRAPH_DELIVERY_JOURNAL", error),
    }
    let review = delivery_plan(app, draft, expected, output, base, source);
    if !review.success() { let mut result = review; result.operation = "graph_delivery".into(); return result; }
    if review.data["plan_revision"] != reviewed {
        return failure("GRAPH_PLAN_CONFLICT", "联合审核已过期，请重新审核。".into());
    }
    let plan = &review.data["plan"];
    if plan["source_review"]["status"] != "success" {
        return failure("GRAPH_SOURCE_INVALID", "源码生成预览未成功；请先修正图。".into());
    }
    let Some(source_after) = plan["source_review"]["data"]["source_sha256"].as_str() else {
        return failure("GRAPH_PLAN_INVALID", "缺少源码目标摘要。".into());
    };
    let Some(source_before) = plan["source_base"].as_str() else { return failure("GRAPH_PLAN_INVALID", "缺少源码基线。".into()); };
    let Some(config_before) = plan["project_config_revision"].as_str() else { return failure("GRAPH_PLAN_INVALID", "缺少项目配置基线。".into()); };
    let bytes = match super::scaffold::read_recipe_for_update(draft) {
        Ok(bytes) if super::scaffold::sha256_bytes(&bytes) == expected => bytes,
        Ok(_) => return failure("GRAPH_CONFLICT", "图草稿在审核后变化。".into()),
        Err(error) => return failure("GRAPH_READ_FAILED", error),
    };
    let graph: Graph = match serde_json::from_slice(&bytes) {
        Ok(graph) => graph, Err(error) => return failure("GRAPH_INVALID", error.to_string()),
    };
    let current = match app.project_config.load(&root) {
        Ok(config) => config, Err(error) => return failure("PROJECT_CONFIG_INVALID", error),
    };
    let revision = match current.as_ref() {
        Some(value) => match crate::domain::input::content_revision(value) { Ok(value) => value, Err(error) => return failure("PROJECT_CONFIG_INVALID", error) },
        None => "absent".to_owned(),
    };
    if revision != config_before { return failure("PROJECT_CONFIG_CONFLICT", "项目配置在审核后变化。".into()); }
    let model = format!("{}:model", source.strip_suffix(".py").unwrap_or(""));
    let mut candidate = current.unwrap_or_else(|| json!({"schema_version":1,"model":model,
        "batch":null,"batch_binding":null,"model_injections":{}}));
    candidate["model"] = json!(model); candidate["model_params"] = json!({}); candidate["model_injections"] = json!({});
    if let Some(data) = &graph.data {
        candidate["batch"] = json!(data.factory); candidate["batch_binding"] = json!(data.binding); candidate["batch_params"] = json!(data.params);
    }
    let config_after = match crate::domain::input::content_revision(&candidate) {
        Ok(value) => value, Err(error) => return failure("PROJECT_CONFIG_INVALID", error),
    };
    let contents = match String::from_utf8(bytes) { Ok(value) => value, Err(error) => return failure("GRAPH_INVALID", error.to_string()) };
    let journal = DeliveryJournal { version: 1, project_root: root.clone(), graph_file: output.to_owned(),
        graph_before: base.to_owned(), graph_after: expected.to_owned(), graph_contents: contents,
        source_file: source.to_owned(), source_before: source_before.to_owned(), source_after: source_after.to_owned(),
        config_before: config_before.to_owned(), config_after, include_data: graph.data.is_some(), reviewed_plan: reviewed.to_owned() };
    if let Err(error) = store.create(&journal) { return failure("GRAPH_DELIVERY_JOURNAL", error); }
    delivery_continue(app, &root, &store, &journal)
}

pub(super) fn delivery_resume<B: Backend>(app: &Application<B>, expected_plan: Option<&str>) -> Report {
    use super::graph_delivery_journal::DeliveryJournalStore;
    let failure = |code, message: String| Report::error("graph_delivery", code, message, "交付记录及已有文件保持原样。 ");
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(root) => root, Err(error) => return failure("PROJECT_UNAVAILABLE", error.to_string()),
    };
    let store = match DeliveryJournalStore::lock(&root) { Ok(store) => store, Err(error) => return failure("GRAPH_DELIVERY_LOCK", error) };
    let journal = match store.load() {
        Ok(Some(journal)) => journal, Ok(None) => return failure("GRAPH_DELIVERY_MISSING", "没有待恢复的图交付。".into()),
        Err(error) => return failure("GRAPH_DELIVERY_JOURNAL", error),
    };
    if expected_plan.is_some_and(|expected| expected != journal.reviewed_plan) {
        return failure("GRAPH_DELIVERY_CONFLICT", "待恢复记录已变化，请重新审核。".into());
    }
    delivery_continue(app, &root, &store, &journal)
}

fn delivery_file_revision(path: &Path, limit: u64) -> Result<String, String> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok("absent".into()),
        Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() && meta.len() <= limit => {
            let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
            if bytes.len() as u64 > limit { return Err("文件超过大小限制。".into()); }
            Ok(super::scaffold::sha256_bytes(&bytes))
        },
        _ => Err("目标不是受支持的普通文件。".into()),
    }
}

pub(super) fn delivery_archive(expected_plan: &str) -> Report {
    use super::graph_delivery_journal::DeliveryJournalStore;
    let fail = |code, message: String| Report::error("graph_delivery_archive", code, message,
        "没有回滚图、源码或项目配置；请核查交付记录。 ");
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(value) => value, Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
    };
    let store = match DeliveryJournalStore::lock(&root) { Ok(value) => value, Err(error) => return fail("GRAPH_DELIVERY_LOCK", error) };
    let journal = match store.load() {
        Ok(Some(value)) => value, Ok(None) => return fail("GRAPH_DELIVERY_MISSING", "没有待归档的交付记录。".into()),
        Err(error) => return fail("GRAPH_DELIVERY_JOURNAL", error),
    };
    if journal.reviewed_plan != expected_plan { return fail("GRAPH_DELIVERY_CONFLICT", "交付记录已变化，请重新审核。".into()); }
    match store.archive(&journal) {
        Ok(path) => Report::ok("graph_delivery_archive", json!({"archived":true,"plan_revision":expected_plan,"archive":path,
            "note":"记录已归档；已提交文件和项目配置没有回滚。"})),
        Err(error) => fail("GRAPH_DELIVERY_JOURNAL", error),
    }
}

pub(super) fn delivery_status<B: Backend>(app: &Application<B>) -> Report {
    use super::graph_delivery_journal::DeliveryJournalStore;
    let failure = |message: String| Report::error("graph_delivery_status", "GRAPH_DELIVERY_STATUS", message,
        "未修改图、源码或项目配置；保留交付记录。 ");
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(value) => value, Err(error) => return failure(error.to_string()),
    };
    match std::fs::symlink_metadata(root.join(".model-maven")) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Report::ok("graph_delivery_status", json!({"pending":false})),
        Ok(meta) if meta.is_dir() && !meta.file_type().is_symlink() => (),
        _ => return failure("项目元数据目录不可读取或不是普通目录。".into()),
    }
    let store = match DeliveryJournalStore::lock(&root) { Ok(value) => value, Err(error) => return failure(error) };
    let journal = match store.load() { Ok(Some(value)) => value, Ok(None) => return Report::ok("graph_delivery_status", json!({"pending":false})), Err(error) => return failure(error) };
    let classify = |actual: Result<String, String>, before: &str, after: &str| -> Value {
        match actual {
            Ok(value) if value == after => json!({"state":"target","revision":value}),
            Ok(value) if value == before => json!({"state":"before","revision":value}),
            Ok(value) => json!({"state":"conflict","revision":value}),
            Err(error) => json!({"state":"unknown","reason":error}),
        }
    };
    let config = match app.project_config.load(&root) {
        Ok(Some(value)) => crate::domain::input::content_revision(&value),
        Ok(None) => Ok("absent".into()), Err(error) => Err(error),
    };
    Report::ok("graph_delivery_status", json!({"pending":true,"plan_revision":journal.reviewed_plan,
        "graph":{"file":journal.graph_file,"comparison":classify(delivery_file_revision(&root.join(&journal.graph_file), 128 * 1024), &journal.graph_before, &journal.graph_after)},
        "source":{"file":journal.source_file,"comparison":classify(delivery_file_revision(&root.join(&journal.source_file), 16 * 1024 * 1024), &journal.source_before, &journal.source_after)},
        "project":classify(config, &journal.config_before, &journal.config_after),"include_data":journal.include_data,
        "note":"只比较当前状态；不会锁定文件供后续恢复，执行时仍会重新核对。"}))
}

fn delivery_continue<B: Backend>(app: &Application<B>, root: &Path, store: &super::graph_delivery_journal::DeliveryJournalStore,
    journal: &super::graph_delivery_journal::DeliveryJournal) -> Report {
    use std::io::Write as _;
    let failure = |stage: &str, message: String| {
        let mut report = Report::error("graph_delivery", "GRAPH_DELIVERY_PARTIAL", message,
            "交付可能部分完成；记录已保留。核查冲突后使用 graph-delivery-resume。 ");
        report.data = json!({"committed":false,"stage":stage,"journal_retained":true,"plan_revision":journal.reviewed_plan}); report
    };
    let graph_path = root.join(&journal.graph_file);
    let source_path = root.join(&journal.source_file);
    let actual = match delivery_file_revision(&graph_path, 128 * 1024) { Ok(value) => value, Err(error) => return failure("graph", error) };
    if actual != journal.graph_after {
        if actual != journal.graph_before { return failure("graph", "图文件已被外部修改。".into()); }
        let mut draft = match tempfile::NamedTempFile::new() { Ok(value) => value, Err(error) => return failure("graph", error.to_string()) };
        if let Err(error) = draft.write_all(journal.graph_contents.as_bytes()) { return failure("graph", error.to_string()); }
        let preview = save(app, draft.path(), &journal.graph_after, &journal.graph_file, &journal.graph_before, false, None);
        let Some(plan) = preview.data["plan_revision"].as_str().filter(|_| preview.success()) else { return failure("graph", format!("图保存预览失败：{:?}", preview.diagnostics)); };
        let committed = save(app, draft.path(), &journal.graph_after, &journal.graph_file, &journal.graph_before, true, Some(plan));
        if !committed.success() { return failure("graph", format!("图保存失败：{:?}", committed.diagnostics)); }
    }
    let actual = match delivery_file_revision(&source_path, 16 * 1024 * 1024) { Ok(value) => value, Err(error) => return failure("source", error) };
    if actual != journal.source_after {
        if actual != journal.source_before { return failure("source", "源码已被外部修改。".into()); }
        let previous = (actual != "absent").then_some(actual.as_str());
        let preview = generate(app, &graph_path, &journal.graph_after, Some(&journal.source_file), previous, false, None);
        if !preview.success() || preview.data["source_sha256"] != journal.source_after { return failure("source", "生成源码已变化或无法预览。".into()); }
        let Some(plan) = preview.data["plan_revision"].as_str() else { return failure("source", "缺少源码计划。".into()); };
        let committed = generate(app, &graph_path, &journal.graph_after, Some(&journal.source_file), previous, true, Some(plan));
        if !committed.success() { return failure("source", format!("源码提交失败：{:?}", committed.diagnostics)); }
    }
    let revision = match app.project_config.load(root) {
        Ok(Some(config)) => match crate::domain::input::content_revision(&config) { Ok(value) => value, Err(error) => return failure("project", error) },
        Ok(None) => "absent".into(), Err(error) => return failure("project", error),
    };
    if revision != journal.config_after {
        if revision != journal.config_before { return failure("project", "项目配置已被外部修改。".into()); }
        let preview = bind_project(app, &graph_path, &journal.graph_after, &journal.source_file, &journal.config_before, journal.include_data, false, None);
        if !preview.success() || preview.data["plan"]["candidate_revision"] != journal.config_after {
            return failure("project", "项目绑定预览已变化或无法生成。".into());
        }
        let Some(plan) = preview.data["plan_revision"].as_str() else { return failure("project", "缺少项目绑定计划。".into()); };
        let committed = bind_project(app, &graph_path, &journal.graph_after, &journal.source_file, &journal.config_before, journal.include_data, true, Some(plan));
        if !committed.success() { return failure("project", format!("项目绑定失败：{:?}", committed.diagnostics)); }
    }
    if delivery_file_revision(&graph_path, 128 * 1024).ok().as_deref() != Some(&journal.graph_after)
        || delivery_file_revision(&source_path, 16 * 1024 * 1024).ok().as_deref() != Some(&journal.source_after)
        || app.project_config.load(root).ok().flatten().and_then(|value| crate::domain::input::content_revision(&value).ok()).as_deref() != Some(&journal.config_after) {
        return failure("verify", "最终状态已变化；交付结果未确认。".into());
    }
    if let Err(error) = store.finish(journal) { return failure("journal", error); }
    Report::ok("graph_delivery", json!({"committed":true,"plan_revision":journal.reviewed_plan,
        "graph_sha256":journal.graph_after,"source_sha256":journal.source_after,"config_revision":journal.config_after,
        "include_data":journal.include_data,"runtime_verified":false,
        "note":"图、源码及项目绑定已完成分阶段条件提交；不是单个原子事务。训练和监督未自动启用。"}))
}
