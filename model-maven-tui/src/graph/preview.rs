//! Render a compact, read-only view of an actual declared graph.
use super::{Edit, GraphDocument, Inlet, LayoutCache, ModelView, Operator, OperatorId, Outlet, Port, Segment, Wire};
use serde_json::{Value, json};
use std::{borrow::Cow, collections::HashMap, io::{self, Read}};

pub(super) fn load(path: &std::path::Path) -> io::Result<ModelView<'static>> {
    let file = std::fs::File::open(path)?;
    if file.metadata()?.len() > 2_500_000 { return Err(io::Error::other("preview request too large")); }
    let request: Value = serde_json::from_reader(file).map_err(io::Error::other)?;
    from_graph(&request["document"])
}

#[cfg(unix)]
pub(crate) fn interactive(path: &std::path::Path, locale: crate::terminal::locale::Locale) -> io::Result<()> {
    let mut view = load(path)?;
    super::interaction::show_live_preview(&mut view, path, locale)
}

pub(crate) fn run() -> io::Result<()> {
    let mut raw = String::new();
    io::stdin().take(2_500_001).read_to_string(&mut raw)?;
    if raw.len() > 2_500_000 { return Err(io::Error::other("preview request too large")); }
    let request: Value = serde_json::from_str(&raw).map_err(io::Error::other)?;
    let mut view = from_graph(&request["document"])?;
    if view.layers.is_empty() { println!("{{\"art\":\"\",\"nodes\":0,\"visible\":0}}"); return Ok(()); }
    if let Some(selected) = request["selected"].as_u64().and_then(|value| usize::try_from(value).ok()) {
        if selected < view.layers.len() { view.selected = selected; }
    }
    let width = request["width"].as_u64().unwrap_or(34).clamp(20, 100) as usize;
    let height = request["height"].as_u64().unwrap_or(18).clamp(12, 40) as usize;
    let scene = view.scene(width, height, false, None);
    let mut art = String::new();
    scene.write_art(&mut art, view.selected, false, false);
    println!("{}", json!({"art":art,"nodes":view.layers.len(),"visible":scene.page_size}));
    Ok(())
}


pub(super) fn from_graph(graph: &Value) -> io::Result<ModelView<'static>> {
    let nodes = graph["nodes"].as_array().ok_or_else(|| io::Error::other("missing graph nodes"))?;
    let edges = graph["edges"].as_array().ok_or_else(|| io::Error::other("missing graph edges"))?;
    if nodes.len() > 512 || edges.len() > 4096 { return Err(io::Error::other("graph exceeds supported declaration limits")); }
    let definitions: HashMap<_, _> = graph["definitions"].as_array().into_iter().flatten()
        .filter_map(|value| Some((value["id"].as_str()?, value))).collect();
    let mut ids = HashMap::new();
    let mut inputs = HashMap::new();
    let mut outputs = HashMap::new();
    let mut edits = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let name = node["id"].as_str().ok_or_else(|| io::Error::other("node lacks ID"))?;
        if name.is_empty() || name.len() > 128 || name.chars().any(char::is_control) { return Err(io::Error::other("invalid node ID")); }
        let id = OperatorId(index as u64 + 1);
        if ids.insert(name, id).is_some() { return Err(io::Error::other("duplicate node ID")); }
        let def = node["definition"].as_str().and_then(|id| definitions.get(id));
        let kind = def.and_then(|d| d["factory"].as_str()).and_then(|name| name.rsplit([':', '.']).next())
            .or_else(|| node["kind"].as_str()).unwrap_or("?").to_owned();
        edits.push(Edit::AddOperator(id, Operator { path: name.to_owned(), kind, parent: None, attributes: Cow::Owned(json!({})) }));
        for (key, input) in [("inputs", true), ("outputs", false)] {
            for (slot, port) in node[key].as_array().into_iter().flatten().enumerate() {
                let name = port.as_str().ok_or_else(|| io::Error::other("invalid port"))?;
                if slot > 255 { return Err(io::Error::other("too many ports")); }
                let port = if input { Port::Input(Inlet {operator:id, slot:slot as u32}) }
                    else { Port::Output(Outlet {operator:id, slot:slot as u32}) };
                let table = if input { &mut inputs } else { &mut outputs };
                if table.insert((id, name.to_owned()), port).is_some() { return Err(io::Error::other("duplicate port")); }
                edits.push(Edit::AddPort(port, None));
            }
        }
    }
    for edge in edges {
        let from = &edge["from"]; let to = &edge["to"];
        let source = from["node"].as_str().and_then(|id| ids.get(id)).and_then(|id| from["port"].as_str().and_then(|name| outputs.get(&(*id, name.to_owned()))));
        let target = to["node"].as_str().and_then(|id| ids.get(id)).and_then(|id| to["port"].as_str().and_then(|name| inputs.get(&(*id, name.to_owned()))));
        match (source, target) {
            (Some(Port::Output(source)), Some(Port::Input(target))) => edits.push(Edit::Connect(Wire { from:*source, to:*target })),
            _ => return Err(io::Error::other("edge references an unknown port")),
        }
    }
    let mut document = GraphDocument::default();
    document.apply(Segment {base_revision:0, edits}).map_err(io::Error::other)?;
    Ok(ModelView::from_document(document, false, LayoutCache::default()))
}
