//! Validated, partial tensor identity observations shared by terminal modes.
use super::{Shape, Wire};
use model_maven_tui::diagnosis::Origin;

#[derive(Clone)]
pub(super) struct Input {
    pub wire: Option<Wire>,
    pub actual: Option<(Shape, Origin)>,
    pub expected: Option<(Shape, Origin)>,
    pub label: Option<String>,
    pub metadata: Option<String>,
    pub matches: Vec<ObservedOutput>,
}

#[derive(Clone)]
pub(super) struct ObservedOutput {
    pub alias: String,
    pub ambiguous_alias: bool,
    pub invocation: Option<u64>,
    pub invocation_id: String,
    pub output_path: String,
}

/// Match only explicit CLI evidence within the same recorded forward. An
/// object match is not a claim that the earlier module produced the value.
pub(super) fn attach_observed_matches(
    inputs: &mut [Input],
    call: &serde_json::Value,
    report: &serde_json::Value,
) {
    let data = &report["data"];
    if data["tensor_identity_trace"]["kind"] != "same_tensor_object_and_version" {
        return;
    }
    let Some(forward) = data["forward_id"].as_str().filter(|id| !id.is_empty()) else {
        return;
    };
    let Some(current_order) = call["call_id"].as_u64() else {
        return;
    };
    if call["forward_id"].as_str() != Some(forward) {
        return;
    }
    let Some(calls) = data["calls"].as_array() else {
        return;
    };
    let Some(relations) = call["input_tensor_matches"].as_array() else {
        return;
    };
    if calls.len() > 4096 || relations.len() > 128 {
        return;
    }
    let Some(current_id) = call["invocation_id"].as_str().filter(|id| !id.is_empty()) else {
        return;
    };
    let same_consumer = |candidate: &serde_json::Value| {
        candidate["invocation_id"].as_str() == Some(current_id)
            && candidate["forward_id"].as_str() == Some(forward)
            && candidate["call_id"].as_u64() == Some(current_order)
            && candidate["inputs"] == call["inputs"]
            && candidate["input_tensor_matches"] == call["input_tensor_matches"]
    };
    let mut current_calls = calls
        .iter()
        .filter(|candidate| candidate["invocation_id"].as_str() == Some(current_id));
    match current_calls.next() {
        Some(candidate) if same_consumer(candidate) && current_calls.next().is_none() => {}
        None if same_consumer(&data["failure_location"]["last_observed_module_call"]) => {}
        _ => return,
    }

    fn contains_tensor(
        value: &serde_json::Value,
        prefix: &str,
        target: &str,
        depth: usize,
    ) -> bool {
        if depth > 8 {
            return false;
        }
        if target != prefix
            && !target
                .strip_prefix(prefix)
                .is_some_and(|tail| tail.starts_with('.') || tail.starts_with('['))
        {
            return false;
        }
        match value["kind"].as_str() {
            Some("tensor") => prefix == target,
            Some("dict") => value["items"].as_object().is_some_and(|items| {
                items.iter().take(32).any(|(key, item)| {
                    contains_tensor(item, &format!("{prefix}.{key}"), target, depth + 1)
                })
            }),
            Some("tuple" | "list") => value["items"].as_array().is_some_and(|items| {
                items.iter().take(32).enumerate().any(|(index, item)| {
                    contains_tensor(item, &format!("{prefix}[{index}]"), target, depth + 1)
                })
            }),
            _ => false,
        }
    }

    for input in inputs {
        let Some(path) = input.label.as_deref() else {
            continue;
        };
        if !input
            .metadata
            .as_deref()
            .is_some_and(|value| value.starts_with("tensor · "))
        {
            continue;
        }
        let mut candidates = relations
            .iter()
            .filter(|item| item["input_path"].as_str() == Some(path));
        let Some(relation) = candidates.next() else {
            continue;
        };
        if candidates.next().is_some() {
            continue;
        }
        let Some(outputs) = relation["prior_outputs"]
            .as_array()
            .filter(|items| items.len() <= 8)
        else {
            continue;
        };
        let mut seen = std::collections::BTreeSet::new();
        for output in outputs {
            let Some(id) = output["invocation_id"]
                .as_str()
                .filter(|id| !id.is_empty() && id.len() <= 256)
            else {
                continue;
            };
            let Some(output_path) = output["output_path"]
                .as_str()
                .filter(|path| path.len() <= 512)
            else {
                continue;
            };
            if !seen.insert((id, output_path)) {
                continue;
            }
            let mut matching = calls
                .iter()
                .filter(|source| source["invocation_id"].as_str() == Some(id));
            let Some(source) = matching.next() else {
                continue;
            };
            if matching.next().is_some()
                || source["forward_id"].as_str() != Some(forward)
                || !source["call_id"]
                    .as_u64()
                    .is_some_and(|order| order < current_order)
                || !contains_tensor(&source["outputs"], "outputs", output_path, 0)
            {
                continue;
            }
            let Some(aliases) = source["alias_paths"]
                .as_array()
                .filter(|aliases| !aliases.is_empty() && aliases.len() <= 32)
            else {
                continue;
            };
            let Some(alias) = aliases[0]
                .as_str()
                .filter(|path| !path.is_empty() && path.len() <= 4096)
            else {
                continue;
            };
            if !aliases
                .iter()
                .all(|path| path.as_str().is_some_and(|path| !path.is_empty()))
            {
                continue;
            }
            input.matches.push(ObservedOutput {
                alias: alias.to_owned(),
                ambiguous_alias: aliases.len() > 1,
                invocation: source["invocation_index"].as_u64(),
                invocation_id: id.to_owned(),
                output_path: output_path.to_owned(),
            });
        }
    }
}

pub(super) fn observed_inputs(tree: &serde_json::Value, expected: Option<&Shape>) -> Vec<Input> {
    const LIMIT: usize = 4096;
    fn visit(
        value: &serde_json::Value,
        path: &str,
        depth: usize,
        out: &mut Vec<Input>,
        expected: Option<&Shape>,
    ) {
        if out.len() >= LIMIT {
            return;
        }
        let leaf = |out: &mut Vec<Input>, shape: Option<Shape>, metadata: String| {
            let expected = if matches!(path, "inputs.args[0]" | "inputs.kwargs.input")
                && metadata.starts_with("tensor · ")
            {
                expected.cloned().map(|shape| (shape, Origin::Declaration))
            } else {
                None
            };
            out.push(Input {
                wire: None,
                actual: shape.map(|shape| (shape, Origin::Observation)),
                expected,
                label: Some(path.to_owned()),
                metadata: Some(metadata),
                matches: Vec::new(),
            });
        };
        if depth > 16 {
            leaf(out, None, "truncated · nesting limit".into());
            return;
        }
        match value["kind"].as_str() {
            Some("tensor") => {
                let dtype = value["dtype"].as_str().unwrap_or("?");
                leaf(
                    out,
                    Shape::parse(&value["shape"]),
                    format!("tensor · {dtype}"),
                );
            }
            Some("dict") if value["items"].is_object() => {
                let items = value["items"].as_object().unwrap();
                if items.is_empty() {
                    leaf(out, None, "empty dict".into());
                }
                for (key, item) in items {
                    if key == "<truncated>" && item.as_u64().is_some() {
                        leaf(
                            out,
                            None,
                            format!("truncated · {} omitted", item.as_u64().unwrap()),
                        );
                    } else {
                        visit(item, &format!("{path}.{key}"), depth + 1, out, expected);
                    }
                    if out.len() >= LIMIT {
                        break;
                    }
                }
            }
            Some(kind @ ("tuple" | "list")) if value["items"].is_array() => {
                let items = value["items"].as_array().unwrap();
                if items.is_empty() {
                    leaf(out, None, format!("empty {kind}"));
                }
                for (index, item) in items.iter().enumerate() {
                    visit(item, &format!("{path}[{index}]"), depth + 1, out, expected);
                    if out.len() >= LIMIT {
                        break;
                    }
                }
                if let Some(count) = value["truncated"].as_u64().filter(|count| *count > 0) {
                    if out.len() < LIMIT {
                        leaf(out, None, format!("truncated · {count} omitted"));
                    }
                }
            }
            Some(kind) => leaf(out, None, kind.to_owned()),
            None => leaf(
                out,
                None,
                match value {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "bool",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::String(_) => "string",
                    _ => "unrecognized",
                }
                .into(),
            ),
        }
    }
    let mut inputs = Vec::new();
    if !tree.is_null() {
        visit(tree, "inputs", 0, &mut inputs, expected);
    }
    if inputs.len() >= LIMIT {
        inputs.push(Input {
            wire: None,
            actual: None,
            expected: None,
            label: Some("inputs".into()),
            metadata: Some("truncated · display limit".into()),
            matches: Vec::new(),
        });
    }
    inputs
}

#[derive(Clone)]
pub(super) struct ObservedConsumer {
    pub alias: String,
    pub ambiguous_alias: bool,
    pub invocation: Option<u64>,
    pub invocation_id: String,
    pub input_index: usize,
    pub input_path: String,
    pub output_path: String,
}

/// Validate later input observations using the same rules as earlier matches.
pub(super) fn observed_consumers(
    id: &str,
    report: &serde_json::Value,
) -> (Vec<ObservedConsumer>, bool) {
    let mut entries: Vec<ObservedConsumer> = Vec::new();
    let mut limited = false;
    if report["data"]["calls"]
        .as_array()
        .is_some_and(|calls| calls.len() > 4096)
    {
        return (entries, true);
    }
    if let Some(calls) = report["data"]["calls"]
        .as_array()
        .filter(|calls| calls.len() <= 4096)
    {
        let last = &report["data"]["failure_location"]["last_observed_module_call"];
        let mut checked = 0usize;
        let last_only = last["invocation_id"]
            .as_str()
            .filter(|id| {
                !calls
                    .iter()
                    .any(|call| call["invocation_id"].as_str() == Some(id))
            })
            .map(|_| last);
        'calls: for call in calls.iter().chain(last_only) {
            if call["input_tensor_matches"]
                .as_array()
                .is_some_and(|relations| {
                    relations.len() > 128
                        || relations.iter().any(|relation| {
                            relation["prior_outputs"]
                                .as_array()
                                .is_some_and(|outputs| outputs.len() > 8)
                        })
                })
            {
                limited = true;
            }
            let Some(relations) = call["input_tensor_matches"]
                .as_array()
                .filter(|r| r.len() <= 128)
            else {
                continue;
            };
            if !relations.iter().any(|relation| {
                relation["prior_outputs"].as_array().is_some_and(|outputs| {
                    outputs.len() <= 8
                        && outputs
                            .iter()
                            .any(|output| output["invocation_id"].as_str() == Some(id))
                })
            }) {
                continue;
            }
            let Some(aliases) = call["alias_paths"].as_array().filter(|a| {
                !a.is_empty()
                    && a.len() <= 32
                    && a.iter()
                        .all(|v| v.as_str().is_some_and(|s| !s.is_empty() && s.len() <= 4096))
            }) else {
                continue;
            };
            let Some(call_id) = call["invocation_id"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 256)
            else {
                continue;
            };
            // The failure call can also occur in calls. Deduplicate it by invocation and input/output identity below.
            let inputs = observed_inputs(&call["inputs"], None);
            if inputs.iter().any(|input| {
                input
                    .metadata
                    .as_deref()
                    .is_some_and(|metadata| metadata.starts_with("truncated"))
            }) {
                limited = true;
            }
            for (input_index, mut input) in inputs.into_iter().enumerate() {
                let Some(input_path) = input.label.as_deref() else {
                    continue;
                };
                if !relations.iter().any(|relation| {
                    relation["input_path"].as_str() == Some(input_path)
                        && relation["prior_outputs"].as_array().is_some_and(|outputs| {
                            outputs.len() <= 8
                                && outputs
                                    .iter()
                                    .any(|output| output["invocation_id"].as_str() == Some(id))
                        })
                }) {
                    continue;
                }
                if checked >= 128 {
                    limited = true;
                    break 'calls;
                }
                checked += 1;
                attach_observed_matches(std::slice::from_mut(&mut input), call, report);
                for matched in input
                    .matches
                    .iter()
                    .filter(|matched| matched.invocation_id == id)
                {
                    let Some(input_path) = &input.label else {
                        continue;
                    };
                    if entries.iter().any(|entry| {
                        entry.invocation_id == call_id
                            && entry.input_path == *input_path
                            && entry.output_path == matched.output_path
                    }) {
                        continue;
                    }
                    if entries.len() == 128 {
                        limited = true;
                        break 'calls;
                    }
                    entries.push(ObservedConsumer {
                        alias: aliases[0].as_str().unwrap().to_owned(),
                        ambiguous_alias: aliases.len() > 1,
                        invocation: call["invocation_index"].as_u64(),
                        invocation_id: call_id.to_owned(),
                        input_index,
                        input_path: input_path.clone(),
                        output_path: matched.output_path.clone(),
                    });
                }
            }
        }
    }
    (entries, limited)
}
