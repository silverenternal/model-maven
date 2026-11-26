//! Source scopes are materialized only when the user opens the corresponding node.
use super::{
    Edit, GraphDocument, LayoutCache, ModelView, Operator, OperatorId, Port, Segment, Wire,
};
use crate::terminal::locale::Locale;
use model_maven_tui::graph_document::{Inlet, Outlet, Shape};
use serde::Deserialize;
use serde_json::Value;
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::Path,
    process::{Command, Stdio},
};

const GENERATOR: &str = "models/generator.py";
const LATENT_L: &str = "configs/gen/latent_sota_L.yaml";

fn port_attribute<'a>(
    attributes: &'a serde_json::Map<String, Value>,
    port: Port,
    keys: [&str; 2],
) -> Option<&'a Value> {
    let (side, slot) = match port {
        Port::Input(inlet) => (0, inlet.slot),
        Port::Output(outlet) => (1, outlet.slot),
    };
    attributes.get(keys[side])?.get(slot as usize)
}

pub(super) fn axis_label(document: &GraphDocument<'_>, port: Port) -> Option<String> {
    let attributes = document.operator(port.operator())?.attributes.as_object()?;
    let labels = port_attribute(attributes, port, ["input_axes", "output_axes"])?
        .as_array()?
        .iter()
        .map(|axis| {
            axis.as_str()
                .map_or_else(|| axis.to_string(), str::to_owned)
        })
        .collect::<Vec<_>>();
    Some(labels.join("×"))
}

pub(super) fn port_location<'a>(
    document: &'a GraphDocument<'_>,
    port: Port,
) -> Option<Cow<'a, str>> {
    let attributes = document.operator(port.operator())?.attributes.as_object()?;
    if let Some(line) =
        port_attribute(attributes, port, ["input_lines", "output_lines"]).and_then(Value::as_u64)
    {
        return Some(Cow::Owned(format!("{GENERATOR}:{line}")));
    }
    attributes.get("source")?.as_str().map(Cow::Borrowed)
}

pub(super) fn connection_label(from: &str, to: &str, wire: Wire, width: usize) -> String {
    use crate::terminal::text::display_width;
    let format = |from, to| format!("{from}[{}] → {to}[{}]", wire.from.slot, wire.to.slot);
    let full = format(from, to);
    if display_width(&full) <= width {
        return full;
    }
    let Some((dot, _)) = from
        .match_indices('.')
        .take_while(|&(dot, _)| to.starts_with(&from[..=dot]))
        .last()
    else {
        return full;
    };
    let short_from = format!("…{}", &from[dot + 1..]);
    let qualified_target = format(&short_from, to);
    if display_width(&qualified_target) <= width {
        return qualified_target;
    }
    let short_to = format!("…{}", &to[dot + 1..]);
    format(&short_from, &short_to)
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
struct Manifest {
    root: Node,
    scopes: BTreeMap<String, Scope>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
struct Node {
    name: String,
    kind: String,
    line: usize,
    end: usize,
    scope: Option<String>,
    attributes: serde_json::Map<String, Value>,
}

#[derive(Default, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
struct Scope {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    #[serde(default)]
    inputs: Vec<Boundary>,
    #[serde(default)]
    outputs: Vec<Boundary>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
struct Boundary {
    slot: u32,
    node: String,
    port: u32,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
struct Edge {
    source: String,
    target: String,
    outlet: u32,
}

pub(super) struct Project {
    scopes: BTreeMap<String, Scope>,
    pending: BTreeMap<OperatorId, String>,
    input_boundaries: BTreeMap<Inlet, Inlet>,
    output_boundaries: BTreeMap<Outlet, Outlet>,
}

#[cfg(any(unix, windows))]
pub(super) struct SearchMatch {
    pub path: String,
    pub kind: String,
    pub line: usize,
    pub id: OperatorId,
    pub parents: Vec<OperatorId>,
    pub port: Option<(Port, Option<Wire>, String)>,
}

impl Scope {
    /// Predict the exact sockets and internal edges materialized by `expand`.
    #[cfg(any(unix, windows))]
    fn searchable_ports(&self, path: &str) -> BTreeMap<Port, Vec<Wire>> {
        let id = |name: &str| operator_id(&format!("{path}.{name}"));
        let mut ports = BTreeMap::<Port, Vec<Wire>>::new();
        for boundary in &self.inputs {
            ports
                .entry(Port::Input(Inlet {
                    operator: id(&boundary.node),
                    slot: boundary.port,
                }))
                .or_default();
        }
        for boundary in &self.outputs {
            ports
                .entry(Port::Output(Outlet {
                    operator: id(&boundary.node),
                    slot: boundary.port,
                }))
                .or_default();
        }
        let mut next_input = BTreeMap::<OperatorId, u32>::new();
        for edge in &self.edges {
            let from = Outlet {
                operator: id(&edge.source),
                slot: edge.outlet,
            };
            let target = id(&edge.target);
            let slot = next_input.entry(target).or_default();
            while ports.contains_key(&Port::Input(Inlet {
                operator: target,
                slot: *slot,
            })) {
                *slot += 1;
            }
            let to = Inlet {
                operator: target,
                slot: *slot,
            };
            *slot += 1;
            let wire = Wire { from, to };
            ports.entry(Port::Output(from)).or_default().push(wire);
            ports.entry(Port::Input(to)).or_default().push(wire);
        }
        ports
    }
}

pub(crate) fn show(root: &Path, preset: Option<&Path>, locale: Locale) -> io::Result<()> {
    let source = read_input(&root.join(GENERATOR), "--drifting-source", locale)?;
    let default = root.join(LATENT_L);
    let (path, option) = preset.map_or((default.as_path(), "--drifting-source"), |path| {
        (path, "--drifting-config")
    });
    let config = read_input(path, option, locale)?;
    let manifest = import(&source, &config).map_err(|error| {
        let label = locale.text("Preset", "预设", "預設");
        io::Error::new(error.kind(), format!("{error}\n{label} {}", path.display()))
    })?;
    show_manifest(manifest, locale)
}

pub(crate) fn show_demo(locale: Locale) -> io::Result<()> {
    show_manifest(bundled_manifest()?, locale)
}

fn bundled_manifest() -> io::Result<Manifest> {
    serde_json::from_str(include_str!(
        "../../fixtures/drifting/latent_sota_L.graph.json"
    ))
    .map_err(io::Error::other)
}

fn read_input(path: &Path, option: &str, locale: Locale) -> io::Result<String> {
    fs::read_to_string(path).map_err(|error| {
        let message = locale.text(
            "Cannot read Drifting file for",
            "Drifting 导入无法读取文件，参数为",
            "Drifting 匯入無法讀取檔案，參數為",
        );
        let hint = locale.text(
            "Pass the actual Drifting checkout directory to --drifting-source. From the TUI repository, these commands import the included source or open the bundled preview.",
            "请把实际的 Drifting 源码目录传给 --drifting-source。在 TUI 仓库目录可运行下方命令，导入附带的源码或打开内置预览。",
            "請將實際的 Drifting 原始碼目錄傳給 --drifting-source。在 TUI 儲存庫目錄可執行下方指令，匯入附帶的原始碼或開啟內建預覽。",
        );
        io::Error::new(
            error.kind(),
            format!(
                "{message} {option}\n  {} ({error})\n{hint}\n  cargo run --release --locked -- --drifting-source ./fixtures/drifting\n  cargo run --release --locked -- --drifting-demo",
                path.display(),
            ),
        )
    })
}

fn import(source: &str, config: &str) -> io::Result<Manifest> {
    let mut parser = Command::new("python3")
        .args(["-I", "-S", "-B", "-c", include_str!("drifting_source.py")])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("Cannot start python3 for Drifting source import ({error})"),
            )
        })?;
    let request = serde_json::to_writer(
        parser.stdin.take().expect("piped parser input"),
        &(source, config),
    );
    let output = parser.wait_with_output()?;
    if !output.status.success() {
        return Err(io::Error::other(
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        ));
    }
    request.map_err(io::Error::other)?;
    serde_json::from_slice(&output.stdout).map_err(io::Error::other)
}

fn show_manifest(manifest: Manifest, locale: Locale) -> io::Result<()> {
    super::show_view(manifest.view()?, locale)
}

impl Node {
    fn operator(&self, path: String, parent: Option<OperatorId>) -> Operator<'static> {
        let mut attributes = self.attributes.clone();
        let source = if self.line == self.end {
            format!("{GENERATOR}:{}", self.line)
        } else {
            format!("{GENERATOR}:{}–{}", self.line, self.end)
        };
        attributes.insert("source".into(), source.into());
        Operator {
            path,
            kind: self.kind.clone(),
            parent,
            attributes: Cow::Owned(Value::Object(attributes)),
        }
    }

    fn shape(&self, port: Port) -> Option<Shape> {
        if let Some(axes) = port_attribute(&self.attributes, port, ["input_axes", "output_axes"])
            .and_then(Value::as_array)
        {
            return Some(Shape::Dimensions(axes.iter().map(Value::as_u64).collect()));
        }
        let key = match port {
            Port::Input(Inlet { slot: 0, .. }) => "input_features",
            Port::Output(Outlet { slot: 0, .. }) => "output_features",
            _ => return None,
        };
        self.attributes
            .get(key)
            .and_then(Value::as_u64)
            .map(Shape::Features)
    }
}

impl Manifest {
    fn view(self) -> io::Result<ModelView<'static>> {
        let mut document = GraphDocument::default();
        let root = operator_id(&self.root.name);
        document
            .apply(Segment {
                base_revision: 0,
                edits: vec![Edit::AddOperator(
                    root,
                    self.root.operator(self.root.name.clone(), None),
                )],
            })
            .map_err(io::Error::other)?;
        let mut view = ModelView::from_document(document, false, LayoutCache::default());
        let mut project = Project {
            pending: self
                .root
                .scope
                .into_iter()
                .map(|name| (root, name))
                .collect(),
            scopes: self.scopes,
            input_boundaries: BTreeMap::new(),
            output_boundaries: BTreeMap::new(),
        };
        project.expand(&mut view, root)?;
        let backbone = operator_id(&format!("{}.backbone", self.root.name));
        if let Some(&index) = view.indices.get(&backbone) {
            view.selected = index;
        }
        view.source_project = Some(project);
        Ok(view)
    }
}

impl Project {
    #[cfg(any(unix, windows))]
    pub(super) fn search(&self, view: &ModelView<'_>, query: &str) -> Vec<SearchMatch> {
        fn visit(
            project: &Project,
            template: &str,
            path: &str,
            query: &str,
            parents: &mut Vec<OperatorId>,
            active: &mut Vec<String>,
            found: &mut Vec<SearchMatch>,
        ) {
            if active.iter().any(|item| item == template) {
                return;
            }
            let Some(scope) = project.scopes.get(template) else {
                return;
            };
            let ports = (query.contains("in[") || query.contains("out["))
                .then(|| scope.searchable_ports(path));
            let paths: BTreeMap<_, _> = ports.as_ref().map_or_else(BTreeMap::new, |_| {
                scope
                    .nodes
                    .iter()
                    .map(|node| {
                        let child = format!("{path}.{}", node.name);
                        (operator_id(&child), child)
                    })
                    .collect()
            });
            active.push(template.to_owned());
            for node in &scope.nodes {
                let child_path = format!("{path}.{}", node.name);
                let id = operator_id(&child_path);
                if child_path.to_lowercase().contains(query)
                    || node.kind.to_lowercase().contains(query)
                {
                    found.push(SearchMatch {
                        path: child_path.clone(),
                        kind: node.kind.clone(),
                        line: node.line,
                        id,
                        parents: parents.clone(),
                        port: None,
                    });
                }
                if let Some(ports) = &ports {
                    for (&port, wires) in ports.iter().filter(|(port, _)| port.operator() == id) {
                        let (direction, slot) = match port {
                            Port::Input(input) => ("in", input.slot),
                            Port::Output(output) => ("out", output.slot),
                        };
                        let label = format!("{child_path}.{direction}[{slot}]");
                        if !label.to_lowercase().contains(query) {
                            continue;
                        }
                        let choices: Vec<_> = if wires.is_empty() {
                            vec![None]
                        } else {
                            wires.iter().copied().map(Some).collect()
                        };
                        for wire in choices {
                            let label = wire.map_or_else(
                                || label.clone(),
                                |wire| {
                                    let (other, direction, slot, arrow) = match port {
                                        Port::Input(_) => {
                                            (wire.from.operator, "out", wire.from.slot, "←")
                                        }
                                        Port::Output(_) => {
                                            (wire.to.operator, "in", wire.to.slot, "→")
                                        }
                                    };
                                    let other = paths.get(&other).map_or("?", String::as_str);
                                    let relative =
                                        other.strip_prefix(path).and_then(|s| s.strip_prefix('.'));
                                    let other = relative.map_or_else(
                                        || other.to_owned(),
                                        |name| format!("…{name}"),
                                    );
                                    format!("{label} {arrow} {other}.{direction}[{slot}]")
                                },
                            );
                            found.push(SearchMatch {
                                path: child_path.clone(),
                                kind: node.kind.clone(),
                                line: node.line,
                                id,
                                parents: parents.clone(),
                                port: Some((port, wire, label)),
                            });
                        }
                    }
                }
                if let Some(scope) = &node.scope {
                    parents.push(id);
                    visit(project, scope, &child_path, query, parents, active, found);
                    parents.pop();
                }
            }
            active.pop();
        }
        let mut found = Vec::new();
        for (&id, template) in &self.pending {
            if let Some(operator) = view.document.operator(id) {
                visit(
                    self,
                    template,
                    &operator.path,
                    query,
                    &mut vec![id],
                    &mut Vec::new(),
                    &mut found,
                );
            }
        }
        found
    }

    #[cfg(any(unix, windows))]
    pub(super) fn reveal(view: &mut ModelView<'_>, parents: &[OperatorId]) -> io::Result<()> {
        let Some(mut project) = view.source_project.take() else {
            return Err(io::Error::other("Source project is no longer available"));
        };
        let result = parents.iter().try_for_each(|&id| project.expand(view, id));
        view.source_project = Some(project);
        result
    }

    pub(super) fn boundary_wires(&self, document: &GraphDocument<'_>, port: Port) -> Vec<Wire> {
        match port {
            Port::Input(mut inlet) => {
                while let Some(&parent) = self.input_boundaries.get(&inlet) {
                    if let Some(from) = document.source(parent) {
                        return vec![Wire { from, to: parent }];
                    }
                    inlet = parent;
                }
                Vec::new()
            }
            Port::Output(mut outlet) => {
                let mut wires = Vec::new();
                while let Some(&parent) = self.output_boundaries.get(&outlet) {
                    wires.extend(
                        document
                            .consumers(parent)
                            .map(|to| Wire { from: parent, to }),
                    );
                    outlet = parent;
                }
                wires
            }
        }
    }

    pub(super) fn expand(&mut self, view: &mut ModelView<'_>, id: OperatorId) -> io::Result<()> {
        let Some(template) = self.pending.get(&id) else {
            return Ok(());
        };
        let scope = self
            .scopes
            .get(template)
            .ok_or_else(|| io::Error::other(format!("Missing source scope {template}")))?;
        let parent = &view
            .document
            .operator(id)
            .ok_or_else(|| io::Error::other("Missing source parent"))?
            .path;
        let mut edits = Vec::new();
        let mut pending = Vec::new();
        let mut children = BTreeMap::new();
        if !scope.inputs.is_empty() || !scope.outputs.is_empty() {
            edits.push(Edit::SetAttribute(
                id,
                "source_boundaries".into(),
                Some(Value::Bool(true)),
            ));
        }
        let mut ports = BTreeMap::new();
        for node in &scope.nodes {
            let path = format!("{parent}.{}", node.name);
            let child = operator_id(&path);
            if children.insert(node.name.as_str(), (child, node)).is_some() {
                return Err(io::Error::other("Duplicate node in source scope"));
            }
            edits.push(Edit::AddOperator(child, node.operator(path, Some(id))));
            if let Some(name) = &node.scope {
                pending.push((child, name.clone()));
            }
        }
        let resolve = |name: &str| {
            children
                .get(name)
                .copied()
                .ok_or_else(|| io::Error::other(format!("Unmapped source endpoint {name}")))
        };
        let outer_ports: BTreeSet<_> = view.document.ports_for(id).map(|(port, _)| port).collect();
        let mut input_boundaries = BTreeMap::new();
        let mut output_boundaries = BTreeMap::new();
        for alias in &scope.inputs {
            let (operator, node) = resolve(&alias.node)?;
            let inner = Inlet {
                operator,
                slot: alias.port,
            };
            let outer = Inlet {
                operator: id,
                slot: alias.slot,
            };
            if input_boundaries.insert(inner, outer).is_some()
                || !outer_ports.contains(&Port::Input(outer))
            {
                return Err(io::Error::other(format!(
                    "Invalid input boundary in {parent} for {}",
                    alias.node
                )));
            }
            let port = Port::Input(inner);
            ports.insert(port, node.shape(port));
        }
        let mut bound_outputs = BTreeSet::new();
        for alias in &scope.outputs {
            let (operator, node) = resolve(&alias.node)?;
            let inner = Outlet {
                operator,
                slot: alias.port,
            };
            let outer = Outlet {
                operator: id,
                slot: alias.slot,
            };
            if output_boundaries.insert(inner, outer).is_some()
                || !bound_outputs.insert(outer)
                || !outer_ports.contains(&Port::Output(outer))
            {
                return Err(io::Error::other(format!(
                    "Invalid output boundary in {parent} for {}",
                    alias.node
                )));
            }
            let port = Port::Output(inner);
            ports.insert(port, node.shape(port));
        }
        let mut wires = Vec::new();
        let mut inputs = BTreeMap::<_, u32>::new();
        for edge in &scope.edges {
            let (producer, source) = resolve(&edge.source)?;
            let (operator, target) = resolve(&edge.target)?;
            let from = Outlet {
                operator: producer,
                slot: edge.outlet,
            };
            let slot = inputs.entry(operator).or_default();
            while ports.contains_key(&Port::Input(Inlet {
                operator,
                slot: *slot,
            })) {
                *slot += 1;
            }
            let to = Inlet {
                operator,
                slot: *slot,
            };
            *slot += 1;
            let output = Port::Output(from);
            let input = Port::Input(to);
            ports.entry(output).or_insert_with(|| source.shape(output));
            ports.insert(input, target.shape(input));
            wires.push(Wire { from, to });
        }
        edits.extend(
            ports
                .into_iter()
                .map(|(port, shape)| Edit::AddPort(port, shape)),
        );
        edits.extend(wires.into_iter().map(Edit::Connect));
        view.apply_segment_in_order(
            Segment {
                base_revision: view.document.revision(),
                edits,
            },
            true,
        )
        .map_err(io::Error::other)?;
        self.pending.remove(&id);
        self.pending.extend(pending);
        self.input_boundaries.extend(input_boundaries);
        self.output_boundaries.extend(output_boundaries);
        Ok(())
    }
}

// Source identity stays stable when disconnected scopes are opened in different orders.
fn operator_id(path: &str) -> OperatorId {
    OperatorId(path.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(unix, windows))]
    #[test]
    fn capture_drifting_visual_review_samples_when_requested() {
        let Some(root) = std::env::var_os("MODEL_MAVEN_PRESET_CAPTURE_DIR") else {
            return;
        };
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        let write = |view: &ModelView<'_>, name: &str| {
            let scene = view.scene(120, 35, true, None);
            let png = super::super::pixels::png_for_test(&scene, view.camera, view.selected);
            std::fs::write(root.join(format!("{name}.png")), png).unwrap();
        };
        let capture_presets = |mut view: ModelView<'_>, prefix: &str| {
            let selected = view.layers[view.selected].id;
            let selected_wire = view.selected_wire;
            write(&view, &format!("{prefix}-spatial"));
            view.toggle_flat_flow();
            assert_eq!(view.layers[view.selected].id, selected);
            assert_eq!(view.selected_wire, selected_wire);
            write(&view, &format!("{prefix}-flow-2d"));
            view.toggle_flat_flow();
            view.set_explosion(1);
            assert_eq!(view.layers[view.selected].id, selected);
            assert_eq!(view.selected_wire, selected_wire);
            write(&view, &format!("{prefix}-exploded-1"));
            view.set_explosion(2);
            assert_eq!(view.layers[view.selected].id, selected);
            assert_eq!(view.selected_wire, selected_wire);
            write(&view, &format!("{prefix}-exploded-2"));
        };

        capture_presets(bundled_manifest().unwrap().view().unwrap(), "drifting");
        let source = include_str!("../../fixtures/drifting/models/generator.py");
        let config = include_str!("../../fixtures/drifting/configs/gen/latent_sota_B.yaml");
        capture_presets(import(source, config).unwrap().view().unwrap(), "drift-b");

        let capture_blocks = |mut view: ModelView<'_>, prefix: &str| {
            let mut search = super::super::search::Search::new();
            for byte in "DitGen.backbone.blocks.block_07".bytes() {
                search.input(byte);
            }
            let target = search
                .selected(&view)
                .expect("block_07 is available as a lazy source match");
            super::super::search::open(&mut view, target);
            assert_eq!(
                view.layers[view.selected].path,
                "DitGen.backbone.blocks.block_07"
            );
            assert!(
                view.focus
                    .is_some_and(|focus| view.children[focus].len() >= 12)
            );
            for (width, height) in [(40, 23), (80, 23), (100, 29), (120, 35)] {
                for level in 1..=2 {
                    for yaw in [0.0, 0.8, 1.8, 2.7] {
                        view.set_explosion(0);
                        view.set_explosion(level);
                        view.camera = super::super::Camera::isometric();
                        view.camera.orbit(yaw, 0.16);
                        let scene = view.scene(width, height, false, None);
                        let captions: std::collections::BTreeSet<_> = scene
                            .captions
                            .iter()
                            .flatten()
                            .map(|(_, label)| label.as_str())
                            .collect();
                        let visible_modules: std::collections::BTreeSet<_> = scene
                            .solids
                            .iter()
                            .map(|solid| {
                                view.layers[solid.layer]
                                    .path
                                    .rsplit('.')
                                    .next()
                                    .unwrap_or_default()
                            })
                            .collect();
                        for module in &visible_modules {
                            assert!(
                                captions.contains(module),
                                "visible module {module} must have a label at {width}×{} exploded x{level}, yaw {yaw}; captions={captions:?}",
                                height + 1
                            );
                        }
                        assert!(
                            captions.contains("block_07"),
                            "selected block_07 label must remain visible at {width}×{} exploded x{level}, yaw {yaw}",
                            height + 1
                        );
                    }
                }
            }
            view.set_explosion(0);
            capture_presets(view, prefix);
        };
        capture_blocks(
            bundled_manifest().unwrap().view().unwrap(),
            "drifting-blocks",
        );

        let mut residual = bundled_manifest().unwrap().view().unwrap();
        let mut search = super::super::search::Search::new();
        for byte in "DitGen.backbone.blocks.block_07.attention_residual".bytes() {
            search.input(byte);
        }
        let target = search
            .selected(&residual)
            .expect("attention residual is available as a lazy source match");
        super::super::search::open(&mut residual, target);
        assert_eq!(
            residual.layers[residual.selected].path,
            "DitGen.backbone.blocks.block_07.attention_residual"
        );
        capture_presets(residual, "drifting-residual");

        let mut attention = bundled_manifest().unwrap().view().unwrap();
        let mut search = super::super::search::Search::new();
        for byte in "block_07.attention.qkv".bytes() {
            search.input(byte);
        }
        let target = search
            .selected(&attention)
            .expect("QKV is available as a lazy source match");
        super::super::search::open(&mut attention, target);
        capture_presets(attention, "qkv");
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn drifting_root_and_attention_crossing_counts_are_comparable_across_presets() {
        let mut view = bundled_manifest().unwrap().view().unwrap();
        let crossings = |scene: &super::super::Scene| {
            scene
                .handoffs
                .iter()
                .map(super::super::Handoff::crossing_marks)
                .sum::<usize>()
                / 2
        };
        let route_geometry = |scene: &super::super::Scene| {
            scene
                .handoffs
                .iter()
                .map(|route| (route.wire, route.layers, route.from, route.to, route.bend))
                .collect::<Vec<_>>()
        };
        let compare = |view: &mut ModelView<'_>, scope: &str| {
            let spatial = view.scene(120, 36, false, None);
            let spatial_count = crossings(&spatial);
            let spatial_geometry = route_geometry(&spatial);
            assert_eq!(
                route_geometry(&view.scene(120, 36, false, None)),
                spatial_geometry,
                "{scope} spatial routes must be stable across redraws"
            );
            let selected = view.layers[view.selected].id;
            let selected_wire = view.selected_wire;
            view.toggle_flat_flow();
            let flow = view.scene(120, 36, false, None);
            let flow_count = crossings(&flow);
            let flow_geometry = route_geometry(&flow);
            assert_eq!(
                route_geometry(&view.scene(120, 36, false, None)),
                flow_geometry,
                "{scope} flow routes must be stable across redraws"
            );
            assert_eq!(
                flow.handoffs
                    .iter()
                    .map(|route| route.wire)
                    .collect::<Vec<_>>(),
                spatial
                    .handoffs
                    .iter()
                    .map(|route| route.wire)
                    .collect::<Vec<_>>(),
                "{scope} must keep every graph edge"
            );
            view.toggle_flat_flow();
            assert_eq!(view.layers[view.selected].id, selected);
            assert_eq!(view.selected_wire, selected_wire);
            (spatial_count, flow_count)
        };

        let root_counts = compare(&mut view, "root");
        assert_eq!(root_counts, (1, 2));
        let mut project = view.source_project.take().unwrap();
        for path in [
            "DitGen.backbone",
            "DitGen.backbone.blocks",
            "DitGen.backbone.blocks.block_07",
            "DitGen.backbone.blocks.block_07.attention",
        ] {
            project
                .expand(&mut view, operator_id(path))
                .expect("Drift-L scope expands");
        }
        view.source_project = Some(project);
        view.focus = Some(view.indices[&operator_id("DitGen.backbone.blocks.block_07")]);
        view.selected = view.indices[&operator_id("DitGen.backbone.blocks.block_07.attention.qkv")];
        let attention_counts = compare(&mut view, "block_07 attention");
        assert_eq!(attention_counts, (1, 4));
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn expanding_a_nested_scope_preserves_visible_root_order_and_selection() {
        let mut view = bundled_manifest().unwrap().view().unwrap();
        let root = view.roots[0];
        let root_children: BTreeSet<_> = view.children[root]
            .iter()
            .map(|&index| view.layers[index].id)
            .collect();
        let selected = view.layers[view.selected].id;
        let selected_wire = view.document.wires().next().unwrap();
        view.selected_wire = Some(selected_wire);
        view.toggle_flat_flow();

        let visible = |view: &ModelView<'_>| {
            let scene = view.scene(120, 36, false, None);
            let nodes = scene
                .solids
                .iter()
                .filter_map(|solid| {
                    let id = solid.key.operator;
                    root_children.contains(&id).then_some((id, solid.center))
                })
                .collect::<Vec<_>>();
            let routes = scene
                .handoffs
                .iter()
                .map(|route| (route.wire, route.from, route.to, route.bend))
                .collect::<Vec<_>>();
            (nodes, routes)
        };
        let before = visible(&view);
        let before_root_layout = view.scope_layouts[root]
            .as_ref()
            .expect("root graph layout")
            .layout
            .stages(0..view.scope_layouts[root].as_ref().unwrap().layout.count())
            .as_ptr();

        let backbone = operator_id("DitGen.backbone");
        view.zoom(view.indices[&backbone]);
        assert_eq!(
            view.focus.map(|index| view.layers[index].id),
            Some(backbone)
        );
        assert!(view.zoom_out());

        assert_eq!(view.layers[view.selected].id, selected);
        assert_eq!(view.selected_wire, Some(selected_wire));
        let after_root_layout = view.scope_layouts[root]
            .as_ref()
            .expect("root graph layout")
            .layout
            .stages(0..view.scope_layouts[root].as_ref().unwrap().layout.count())
            .as_ptr();
        assert_eq!(after_root_layout, before_root_layout);
        assert_eq!(visible(&view), before);
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn search_reveals_only_selected_lazy_source_path_and_preserves_history() {
        use super::super::search::{Search, Target};
        let mut view = bundled_manifest().unwrap().view().unwrap();
        let before = view.bookmark();
        let count = view.layers.len();
        let mut search = Search::new();
        for byte in "block_07.attention.qkv".bytes() {
            search.input(byte);
        }
        let target = search.selected(&view).expect("unloaded source match");
        assert!(matches!(target, Target::Source(_, _)));
        assert_eq!(view.layers.len(), count);
        let path = "DitGen.backbone.blocks.block_07.attention.qkv";
        assert!(view.document.operator(operator_id(path)).is_none());
        super::super::search::open(&mut view, target);
        assert_eq!(view.layers[view.selected].path, path);
        assert!(
            view.document
                .operator(operator_id("DitGen.backbone.blocks.block_08.attention.qkv"))
                .is_none()
        );
        assert!(view.zoom_out());
        assert_eq!(view.bookmark().selected, before.selected);
        assert_eq!(view.bookmark().focus, before.focus);
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn lazy_port_search_matches_materialized_slots_fanout_and_history() {
        use super::super::search::{Search, Target};
        let mut view = bundled_manifest().unwrap().view().unwrap();
        let before = view.bookmark();
        let count = view.layers.len();
        let mut search = Search::new();
        for byte in "block_07.attention.reshape_qkv.out[0]".bytes() {
            search.input(byte);
        }
        let target = search.selected(&view).expect("unloaded source port");
        let narrow = search.render(&view, 40, 24, Locale::English);
        for name in ["q.in[0]", "k.in[0]", "v.in[0]"] {
            assert!(narrow.contains(name), "{narrow}");
        }
        let Target::SourcePort(port, Some(expected), _) = &target else {
            panic!("expected a source port with an explicit internal wire")
        };
        let port = *port;
        let expected = *expected;
        assert_eq!(view.layers.len(), count);
        super::super::search::open(&mut view, target);
        assert_eq!(view.document.source(expected.to), Some(expected.from));
        assert!(
            view.document
                .ports_for(port.operator())
                .any(|(item, _)| item == port)
        );
        assert_eq!(view.selected_wire, Some(expected));
        assert!(
            view.document
                .operator(operator_id("DitGen.backbone.blocks.block_08.attention.qkv"))
                .is_none()
        );
        let outputs: Vec<_> = view.document.consumers(expected.from).collect();
        assert_eq!(outputs.len(), 3);
        assert!(view.zoom_out());
        assert!(view.bookmark() == before);
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn planned_scope_sockets_match_expansion_in_drifting_attention() {
        let manifest = bundled_manifest().unwrap();
        let path = "DitGen.backbone.blocks.block_07.attention";
        let expected = manifest.scopes["attention"].searchable_ports(path);
        let mut view = manifest.view().unwrap();
        let parents = [
            "DitGen.backbone",
            "DitGen.backbone.blocks",
            "DitGen.backbone.blocks.block_07",
            path,
        ]
        .map(operator_id);
        Project::reveal(&mut view, &parents).unwrap();
        let node_ids: BTreeSet<_> = expected.keys().map(|port| port.operator()).collect();
        let actual: BTreeSet<_> = node_ids
            .iter()
            .flat_map(|id| view.document.ports_for(*id).map(|(port, _)| port))
            .collect();
        assert_eq!(actual, expected.keys().copied().collect());
        for (port, wires) in expected {
            let actual: BTreeSet<_> = match port {
                Port::Input(to) => view
                    .document
                    .source(to)
                    .map(|from| Wire { from, to })
                    .into_iter()
                    .collect(),
                Port::Output(from) => view
                    .document
                    .consumers(from)
                    .map(|to| Wire { from, to })
                    .collect(),
            };
            assert_eq!(actual, wires.into_iter().collect(), "{port:?}");
        }
    }

    #[test]
    fn long_source_connections_keep_both_port_numbers() {
        let parent = "DitGen.backbone.blocks.block_07.attention";
        let from = format!("{parent}.head_axes.k_transposed");
        let to = format!("{parent}.logits");
        let edge = Wire {
            from: Outlet {
                operator: operator_id(&from),
                slot: 0,
            },
            to: Inlet {
                operator: operator_id(&to),
                slot: 1,
            },
        };
        assert_eq!(
            connection_label(&from, &to, edge, 150),
            format!("{from}[0] → {to}[1]")
        );
        assert_eq!(
            connection_label(&from, &to, edge, 120),
            format!("…head_axes.k_transposed[0] → {to}[1]")
        );
        assert_eq!(
            connection_label(&from, &to, edge, 50),
            "…head_axes.k_transposed[0] → …logits[1]"
        );
    }

    #[test]
    #[ignore = "requires Python 3.9+ to parse the official source fixtures"]
    fn official_drifting_presets_expand_one_attention_scope_at_a_time() {
        let source = include_str!("../../fixtures/drifting/models/generator.py");
        for (config, depth, width, heads) in [
            (
                include_str!("../../fixtures/drifting/configs/gen/latent_sota_L.yaml"),
                24,
                1024,
                16,
            ),
            (
                include_str!("../../fixtures/drifting/configs/gen/latent_sota_B.yaml"),
                12,
                768,
                12,
            ),
        ] {
            let manifest = import(source, config).unwrap();
            if depth == 24 {
                assert!(
                    manifest == bundled_manifest().unwrap(),
                    "bundled Drift-L graph must match the source and preset"
                );
            }
            let mut view = manifest.view().unwrap();
            let backbone = operator_id("DitGen.backbone");
            let stack = operator_id("DitGen.backbone.blocks");
            let first = operator_id("DitGen.backbone.blocks.block_00");
            let last = operator_id(&format!("DitGen.backbone.blocks.block_{:02}", depth - 1));
            let attention = operator_id("DitGen.backbone.blocks.block_00.attention");
            let operation = |name: &str| {
                operator_id(&format!("DitGen.backbone.blocks.block_00.attention.{name}"))
            };
            assert_eq!(view.document.operators().count(), 6);
            for (parent, child) in [
                (backbone, stack),
                (stack, first),
                (first, attention),
                (attention, operation("q")),
            ] {
                assert!(view.document.operator(child).is_none());
                view.zoom(view.indices[&parent]);
                assert!(view.document.operator(child).is_some());
                #[cfg(any(unix, windows))]
                if parent == stack {
                    view.explosion = 1;
                    let scene = view.scene(120, 36, false, None);
                    let mut instances = scene
                        .solids
                        .iter()
                        .filter_map(|solid| {
                            view.layers
                                .get(solid.layer)
                                .map(|layer| (layer.id, solid.center))
                        })
                        .filter(|(id, _)| {
                            view.document
                                .operator(*id)
                                .is_some_and(|operator| operator.parent == Some(stack))
                        })
                        .collect::<Vec<_>>();
                    instances.sort_by_key(|(id, _)| *id);
                    instances.dedup_by_key(|(id, _)| *id);
                    assert!(
                        instances.len() >= 2,
                        "exploded scope shows separate block instances"
                    );
                    assert!(instances.iter().any(|(id, _)| *id == first));
                    assert!(
                        instances
                            .iter()
                            .any(|(_, a)| instances.iter().any(|(_, b)| (a[1] - b[1]).abs() > 1.0))
                    );
                    assert!(
                        scene
                            .handoffs
                            .iter()
                            .any(|route| route.wire.from.operator == first
                                || route.wire.to.operator == first)
                    );
                    let mut matrix =
                        super::super::matrix::Matrix::new(&view, &scene, view.selected_wire)
                            .expect("loaded block scope has matrix entries");
                    assert!(
                        matrix
                            .render(&view, 120, 36, Locale::English)
                            .contains("loaded scope")
                    );
                    let selected = matrix.selected_wire();
                    matrix.toggle_groups(true, true);
                    if selected.is_some() {
                        assert_eq!(matrix.selected_wire(), selected);
                    }
                    assert!(
                        matrix
                            .render(&view, 120, 36, Locale::English)
                            .contains("Pairs")
                    );
                    view.explosion = 0;
                }
            }
            assert_eq!(view.children[view.indices[&stack]].len(), depth);
            assert!(view.document.operator(last).is_some());
            assert!(
                view.document
                    .operator(operator_id("DitGen.backbone.blocks.block_01.attention"))
                    .is_none()
            );
            assert_eq!(
                view.document.operator(attention).unwrap().attributes["heads"],
                heads
            );
            let qkv = operation("qkv");
            assert_eq!(
                view.document.operator(qkv).unwrap().attributes["width"],
                width * 3
            );
            let input = |operator, slot| Inlet { operator, slot };
            let output = |operator, slot| Outlet { operator, slot };
            let reshape = operation("reshape_qkv");
            assert_eq!(
                view.document.source(input(reshape, 0)),
                Some(output(qkv, 0))
            );
            let packed =
                Shape::Dimensions(vec![None, None, Some(3), Some(heads), Some(width / heads)]);
            for arm in ["q", "k", "v"] {
                let split = input(operation(arm), 0);
                assert_eq!(view.document.source(split), Some(output(reshape, 0)));
                assert_eq!(
                    view.document.operator(split.operator).unwrap().attributes["source"],
                    "models/generator.py:184"
                );
                assert_eq!(view.document.shape(Port::Input(split)), Some(&packed));
            }
            assert_eq!(
                view.document.operator(operation("logits")).unwrap().kind,
                "QKᵀ matmul"
            );
            let revision = view.document.revision();
            let index = view.indices[&operation("q")];
            assert!(view.zoom_out());
            view.zoom(view.indices[&attention]);
            assert_eq!(view.document.revision(), revision);
            assert_eq!(view.indices[&operation("q")], index);
            let rope = operation("rope");
            let head_axes = operation("head_axes");
            assert!(
                view.document
                    .operator(operation("rope.q_rotated"))
                    .is_none()
            );
            view.zoom(view.indices[&rope]);
            let project = view.source_project.as_ref().unwrap();
            for (arm, slot, line) in [("q", 0, 131), ("k", 1, 132)] {
                let rotated = operation(&format!("rope.{arm}_rotated"));
                let inbound = project.boundary_wires(
                    &view.document,
                    Port::Input(Inlet {
                        operator: rotated,
                        slot: 0,
                    }),
                );
                assert_eq!(
                    inbound,
                    vec![Wire {
                        from: Outlet {
                            operator: operation(&format!("{arm}_norm")),
                            slot: 0
                        },
                        to: Inlet {
                            operator: rope,
                            slot
                        },
                    }]
                );
                let outbound = project.boundary_wires(
                    &view.document,
                    Port::Output(Outlet {
                        operator: rotated,
                        slot: 0,
                    }),
                );
                assert_eq!(
                    outbound,
                    vec![Wire {
                        from: Outlet {
                            operator: rope,
                            slot
                        },
                        to: Inlet {
                            operator: head_axes,
                            slot
                        },
                    }]
                );
                assert_eq!(
                    view.document.operator(rotated).unwrap().attributes["source"],
                    format!("models/generator.py:{line}")
                );
            }
            view.zoom_out();
            let key_transpose = operation("head_axes.k_transposed");
            assert!(view.document.operator(key_transpose).is_none());
            view.zoom(view.indices[&head_axes]);
            let key_heads = operation("head_axes.k_heads");
            assert_eq!(
                view.document.source(input(key_transpose, 0)),
                Some(output(key_heads, 0))
            );
            let rotated_key = Port::Output(output(key_transpose, 0));
            assert_eq!(
                view.document.shape(rotated_key),
                Some(&Shape::Dimensions(vec![
                    None,
                    Some(heads),
                    Some(width / heads),
                    None
                ]))
            );
            assert_eq!(
                axis_label(&view.document, rotated_key),
                Some(format!("B×{heads}×{}×N", width / heads))
            );
            assert_eq!(
                view.source_project
                    .as_ref()
                    .unwrap()
                    .boundary_wires(&view.document, rotated_key),
                vec![Wire {
                    from: output(head_axes, 1),
                    to: input(operation("logits"), 1),
                }]
            );
            assert_eq!(
                view.document.operator(key_transpose).unwrap().attributes["source"],
                "models/generator.py:215"
            );
        }
    }

    #[test]
    #[ignore = "requires Python 3.9+ to parse the official source fixtures"]
    fn drifting_noise_coordinate_scopes_keep_label_and_accumulator_ports_separate() {
        let mut view = import(
            include_str!("../../fixtures/drifting/models/generator.py"),
            include_str!("../../fixtures/drifting/configs/gen/latent_sota_L.yaml"),
        )
        .unwrap()
        .view()
        .unwrap();
        let conditioning = operator_id("DitGen.conditioning");
        let node = |name: &str| operator_id(&format!("DitGen.conditioning.{name}"));
        let group = node("noise_embeds");
        let first = node("noise_embeds.coord_00");
        let second = node("noise_embeds.coord_01");
        let last = node("noise_embeds.coord_31");
        let sample = operator_id("DitGen.noise_labels");
        let input = |operator, slot| Inlet { operator, slot };
        let output = |operator, slot| Outlet { operator, slot };
        assert!(view.document.operator(group).is_none());
        view.zoom(view.indices[&conditioning]);
        assert_eq!(
            view.document.source(input(group, 0)),
            Some(output(node("class_embed"), 0))
        );
        assert!(view.document.operator(first).is_none());
        view.zoom(view.indices[&group]);
        assert_eq!(view.children[view.indices[&group]].len(), 32);
        assert_eq!(
            view.document.source(input(second, 0)),
            Some(output(first, 0))
        );
        let labels = Wire {
            from: output(sample, 0),
            to: input(conditioning, 0),
        };
        for (coordinate, name, next) in [
            (
                first,
                "coord_00",
                Wire {
                    from: output(first, 0),
                    to: input(second, 0),
                },
            ),
            (
                last,
                "coord_31",
                Wire {
                    from: output(group, 0),
                    to: input(node("merge"), 0),
                },
            ),
        ] {
            assert_eq!(
                view.source_project
                    .as_ref()
                    .unwrap()
                    .boundary_wires(&view.document, Port::Input(input(coordinate, 1)),),
                vec![labels]
            );
            let label = node(&format!("noise_embeds.{name}.label"));
            assert!(view.document.operator(label).is_none());
            view.zoom(view.indices[&coordinate]);
            let project = view.source_project.as_ref().unwrap();
            assert_eq!(
                project.boundary_wires(&view.document, Port::Input(input(label, 0))),
                vec![labels]
            );
            assert_eq!(
                view.document.operator(label).unwrap().attributes["source"],
                "models/generator.py:585"
            );
            let sum = node(&format!("noise_embeds.{name}.add"));
            let embedding = node(&format!("noise_embeds.{name}.embedding"));
            assert_eq!(
                view.document.source(input(embedding, 0)),
                Some(output(label, 0))
            );
            assert_eq!(
                view.document.source(input(sum, 1)),
                Some(output(embedding, 0))
            );
            assert_eq!(
                project.boundary_wires(&view.document, Port::Output(output(sum, 0))),
                vec![next]
            );
            assert!(view.zoom_out());
        }
        assert!(
            view.document
                .operator(node("noise_embeds.coord_01.label"))
                .is_none()
        );
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn nested_boundaries_project_the_original_connection_and_restore_the_local_view() {
        let node = |name: &str, scope: Option<&str>| Node {
            name: name.into(),
            kind: name.into(),
            line: 10,
            end: 10,
            scope: scope.map(str::to_owned),
            attributes: serde_json::Map::new(),
        };
        let edge = |source: &str, target: &str| Edge {
            source: source.into(),
            target: target.into(),
            outlet: 0,
        };
        let boundary = |node: &str| Boundary {
            slot: 0,
            node: node.into(),
            port: 0,
        };
        let manifest = Manifest {
            root: node("DitGen", Some("generator")),
            scopes: BTreeMap::from([
                (
                    "generator".into(),
                    Scope {
                        nodes: vec![
                            node("input", None),
                            node("backbone", Some("backbone")),
                            node("output", None),
                        ],
                        edges: vec![edge("input", "backbone"), edge("backbone", "output")],
                        ..Scope::default()
                    },
                ),
                (
                    "backbone".into(),
                    Scope {
                        nodes: vec![node("stack", Some("stack"))],
                        inputs: vec![boundary("stack")],
                        outputs: vec![boundary("stack")],
                        ..Scope::default()
                    },
                ),
                (
                    "stack".into(),
                    Scope {
                        nodes: vec![node("first", None), node("last", None)],
                        edges: vec![edge("first", "last")],
                        inputs: vec![boundary("first")],
                        outputs: vec![boundary("last")],
                    },
                ),
            ]),
        };
        let mut view = manifest.view().unwrap();
        let backbone = operator_id("DitGen.backbone");
        let input = operator_id("DitGen.input");
        let stack = operator_id("DitGen.backbone.stack");
        let first = operator_id("DitGen.backbone.stack.first");
        let last = operator_id("DitGen.backbone.stack.last");
        let incoming = Wire {
            from: Outlet {
                operator: input,
                slot: 0,
            },
            to: Inlet {
                operator: backbone,
                slot: 0,
            },
        };
        let original_indices = view.indices.clone();
        let root = operator_id("DitGen");
        let overview_members = view.scope_layouts[view.indices[&root]]
            .as_ref()
            .unwrap()
            .members
            .as_ptr();
        assert!(view.document.operator(first).is_none());
        view.selected_wire = Some(incoming);
        view.zoom(view.indices[&backbone]);
        assert_eq!(view.layers[view.selected].id, stack);
        assert!(view.document.operator(first).is_none());
        view.zoom(view.indices[&stack]);
        for (operator, index) in original_indices {
            assert_eq!(view.indices[&operator], index);
        }
        assert_eq!(
            view.scope_layouts[view.indices[&root]]
                .as_ref()
                .unwrap()
                .members
                .as_ptr(),
            overview_members
        );
        assert_eq!(view.layers[view.selected].id, first);
        assert_eq!(view.selected_wire, Some(incoming));
        assert_eq!(view.document.source(incoming.to), Some(incoming.from));
        assert_eq!(
            view.document.source(Inlet {
                operator: first,
                slot: 0
            }),
            None
        );
        view.viewport.zoom(1.12, [12.0, 4.0]);
        let viewport = view.viewport;
        let scene = view.scene(80, 25, false, None);
        assert!(
            scene
                .handoffs
                .iter()
                .any(|route| route.wire == incoming && route.displayed.to.operator == first)
        );
        let external_output = scene
            .handoffs
            .iter()
            .find(|route| route.displayed.from.operator == last)
            .unwrap();
        assert_eq!(external_output.wire.from.operator, backbone);
        assert_eq!(
            external_output.wire.to.operator,
            operator_id("DitGen.output")
        );
        view.follow_wire(&scene, false, None);
        assert!(view.focus.is_none());
        assert_eq!(view.layers[view.selected].id, input);
        assert!(view.zoom_out());
        assert_eq!(view.layers[view.focus.unwrap()].id, stack);
        assert_eq!(view.layers[view.selected].id, first);
        assert_eq!(view.selected_wire, Some(incoming));
        assert!(view.viewport == viewport);
    }

    #[test]
    fn source_scopes_load_once_and_keep_their_operator_and_port_identity() {
        let node = |name: &str, scope: Option<&str>| Node {
            name: name.into(),
            kind: "Source node".into(),
            line: 22,
            end: 25,
            scope: scope.map(str::to_owned),
            attributes: serde_json::Map::new(),
        };
        let manifest = Manifest {
            root: node("DitGen", Some("generator")),
            scopes: BTreeMap::from([
                (
                    "generator".into(),
                    Scope {
                        nodes: vec![node("backbone", Some("backbone"))],
                        edges: vec![],
                        ..Scope::default()
                    },
                ),
                (
                    "backbone".into(),
                    Scope {
                        nodes: vec![node("attention", Some("attention"))],
                        edges: vec![],
                        ..Scope::default()
                    },
                ),
                (
                    "attention".into(),
                    Scope {
                        nodes: vec![node("rope", None), node("matmul", None)],
                        edges: [0, 1]
                            .map(|outlet| Edge {
                                source: "rope".into(),
                                target: "matmul".into(),
                                outlet,
                            })
                            .into(),
                        ..Scope::default()
                    },
                ),
            ]),
        };
        let mut view = manifest.view().unwrap();
        let backbone = operator_id("DitGen.backbone");
        let attention = operator_id("DitGen.backbone.attention");
        assert!(view.document.operator(attention).is_none());
        view.zoom(view.indices[&backbone]);
        assert_eq!(
            view.document.operator(attention).unwrap().attributes["source"],
            "models/generator.py:22–25"
        );
        let matmul = operator_id("DitGen.backbone.attention.matmul");
        assert!(view.document.operator(matmul).is_none());
        view.zoom(view.indices[&attention]);
        let rope = operator_id("DitGen.backbone.attention.rope");
        for slot in [0, 1] {
            assert_eq!(
                view.document.source(Inlet {
                    operator: matmul,
                    slot
                }),
                Some(Outlet {
                    operator: rope,
                    slot
                })
            );
        }
        let revision = view.document.revision();
        assert!(view.zoom_out());
        view.zoom(view.indices[&attention]);
        assert_eq!(view.document.revision(), revision);
    }
}
