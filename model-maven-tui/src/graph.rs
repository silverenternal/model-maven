//! Interactive, schematic 3D view of model modules.
use crate::{
    terminal::{layout::Background, locale::Locale, text},
    workbench::Style,
};
use model_maven_tui::graph_document::{
    ChangeSet, Edit, EditError, GraphDocument, Inlet, Operator, OperatorId, Outlet, Port, Segment,
    Wire,
};
use serde_json::{Value, json};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Write as _,
};

mod evidence;
pub(crate) mod preview;
pub(crate) struct IdentityMatch {
    pub input: String,
    pub alias: String,
    pub ambiguous_alias: bool,
    pub invocation: Option<u64>,
    pub output: String,
    pub call_number: Option<usize>,
}

pub(crate) struct IdentityConsumer {
    pub input: String,
    pub alias: String,
    pub ambiguous_alias: bool,
    pub invocation: Option<u64>,
    pub output: String,
    pub call_number: Option<usize>,
}

fn identity_call_number(report: &Value, id: &str) -> Option<usize> {
    let calls = report["data"]["calls"].as_array()?;
    let mut matches = calls
        .iter()
        .enumerate()
        .filter(|(_, call)| call["invocation_id"].as_str() == Some(id));
    let index = matches.next()?.0;
    matches.next().is_none().then_some(index + 1)
}

pub(crate) fn line_identity_consumers(
    call: &Value,
    report: &Value,
) -> (Vec<IdentityConsumer>, bool) {
    let Some(id) = call["invocation_id"].as_str().filter(|id| !id.is_empty()) else {
        return (Vec::new(), false);
    };
    let Some(index) = identity_call_number(report, id).and_then(|number| number.checked_sub(1))
    else {
        return (Vec::new(), false);
    };
    if report["data"]["calls"]
        .as_array()
        .and_then(|calls| calls.get(index))
        != Some(call)
    {
        return (Vec::new(), false);
    }
    let (consumers, limited) = evidence::observed_consumers(id, report);
    (
        consumers
            .into_iter()
            .map(|entry| IdentityConsumer {
                input: entry.input_path,
                alias: entry.alias,
                ambiguous_alias: entry.ambiguous_alias,
                invocation: entry.invocation,
                output: entry.output_path,
                call_number: identity_call_number(report, &entry.invocation_id),
            })
            .collect(),
        limited,
    )
}

/// Return only matches validated by the same rules used by the graphical view.
/// The bound limits work and output in line-oriented terminals.
pub(crate) fn line_identity_matches(call: &Value, report: &Value) -> (Vec<IdentityMatch>, bool) {
    let mut inputs = evidence::observed_inputs(&call["inputs"], None);
    let mut limited = inputs.len() > 128
        || inputs.iter().any(|input| {
            input
                .metadata
                .as_deref()
                .is_some_and(|metadata| metadata.starts_with("truncated"))
        });
    inputs.truncate(128);
    evidence::attach_observed_matches(&mut inputs, call, report);
    let mut matches = Vec::new();
    for input in inputs {
        for observed in input.matches {
            if matches.len() == 128 {
                limited = true;
                return (matches, limited);
            }
            matches.push(IdentityMatch {
                input: input.label.clone().unwrap_or_default(),
                alias: observed.alias,
                ambiguous_alias: observed.ambiguous_alias,
                invocation: observed.invocation,
                output: observed.output_path,
                call_number: identity_call_number(report, &observed.invocation_id),
            });
        }
    }
    (matches, limited)
}
mod appearance;
mod large;
use appearance::{Appearance, LinkInk, SliceStyle};
mod solid;
use solid::{Camera, Canvas, Cuboid, SliceKey};
mod flow_layout;
use flow_layout::FlowLayout;
#[cfg(unix)]
mod candidates;
#[cfg(unix)]
pub(crate) mod diff;
#[cfg(unix)]
mod failure;
#[cfg(unix)]
mod matrix;
#[cfg(unix)]
mod search;
mod tensor;
use tensor::{Alignment, Continuation, Handoff, Interface, Shape};
mod source;
pub(crate) use source::{show as show_source, show_demo as show_drifting_demo};

const DEMO_CONNECTIONS: &[[usize; 2]] = &[[0, 1], [0, 2], [1, 4], [2, 4], [4, 5]];

#[cfg(unix)]
mod axes;
#[cfg(unix)]
mod editor;
#[cfg(unix)]
mod pixels;

struct Layer<'a> {
    id: OperatorId,
    path: String,
    kind: String,
    data: Cow<'a, Value>,
    appearance: SliceStyle,
    hop_range: f64,
}

// Scene Y uses two units per terminal row.
const HOP_PERIOD: f64 = 2.6;

fn parameter_count(data: &Value) -> Option<u64> {
    data["parameter_count"]
        .as_u64()
        .or_else(|| data["parameters"].as_u64())
}

fn hop_range(data: &Value) -> f64 {
    parameter_count(data).map_or(1.10, |count| {
        0.35 + 1.85 / (1.0 + (count as f64 / 65_536.0).sqrt())
    })
}

fn hop_progress(time: f64, operator: OperatorId) -> f64 {
    let phase = (time / HOP_PERIOD + (operator.0 % 997) as f64 * 0.381_966).fract();
    // Ease into and out of the hop, then briefly rest on the baseline.
    let arc = (phase * 1.5).min(1.0);
    (arc * std::f64::consts::PI).sin().powi(2)
}

#[cfg(unix)]
const SCENE_HEADER: usize = 3;
#[cfg(unix)]
const GRAPH_TOOLBAR: &str = "[-][+][<][>][^][v][R][E]";
#[cfg(unix)]
const GRAPH_TOOLBAR_MIN_WIDTH: usize = 36;
#[cfg(unix)]
const GRAPH_MODE_TOOLBAR_MIN_WIDTH: usize = 92;
#[cfg(unix)]
const GRAPH_MODE_TOOLBAR_WIDTH: usize = 12;

struct Scene {
    appearance: Appearance,
    width: usize,
    short: bool,
    tooltip_rows: usize,
    interactive_motion: bool,
    canvas: Canvas,
    labels: Vec<Vec<char>>,
    captions: Vec<Vec<(usize, String)>>,
    solids: Vec<Cuboid>,
    #[cfg(unix)]
    links: Vec<Curve>,
    #[cfg(unix)]
    handoffs: Vec<Handoff>,
    #[cfg(unix)]
    observed_routes: Vec<ObservedRoute>,
    #[cfg(unix)]
    candidate: Option<Handoff>,
    #[cfg(unix)]
    edit_mark: Option<PortMark>,
    focus_mark: Option<PortMark>,
    #[cfg(unix)]
    groups: Vec<Group>,
    #[cfg(unix)]
    axes: Option<axes::Drawing>,
    outline_cursor: usize,
    page_size: usize,
    page_start: usize,
    lane_page: bool,
    vertical_flow: bool,
    #[cfg(unix)]
    previous_page: usize,
    item_count: usize,
}

impl Scene {
    fn new(appearance: Appearance, width: usize, height: usize, tooltips: bool) -> Self {
        let width = width.clamp(20, 180);
        let height = height.clamp(12, 80);
        let short = height < 20;
        let art_height = height.saturating_sub(if short { 9 } else { 10 }).max(3);
        let tooltip_rows = usize::from(tooltips && art_height >= 9) * 2;
        let rows = art_height - tooltip_rows;
        Self {
            appearance,
            width,
            short,
            tooltip_rows,
            interactive_motion: false,
            canvas: Canvas::new(width, rows - 1),
            labels: vec![vec![' '; width]; rows],
            captions: vec![Vec::new(); rows],
            solids: Vec::new(),
            #[cfg(unix)]
            links: Vec::new(),
            #[cfg(unix)]
            handoffs: Vec::new(),
            #[cfg(unix)]
            observed_routes: Vec::new(),
            #[cfg(unix)]
            candidate: None,
            #[cfg(unix)]
            edit_mark: None,
            focus_mark: None,
            #[cfg(unix)]
            groups: Vec::new(),
            #[cfg(unix)]
            axes: None,
            outline_cursor: 0,
            page_size: 1,
            page_start: 0,
            lane_page: false,
            vertical_flow: false,
            #[cfg(unix)]
            previous_page: 0,
            item_count: 0,
        }
    }

    #[cfg(unix)]
    fn for_axes(
        appearance: Appearance,
        width: usize,
        height: usize,
        comparison: axes::Comparison,
        pixels: bool,
        locale: Locale,
    ) -> Self {
        let mut scene = Self::new(appearance, width, height, false);
        let drawing = comparison.draw(scene.width, scene.labels.len() - 1);
        for (index, &row) in drawing.rows.iter().enumerate() {
            let label = offline::axis_label(index, comparison.candidate, locale, scene.width < 32);
            let text = if comparison.missing(index) {
                format!("{label} ?")
            } else {
                label.into()
            };
            scene.captions[row].push((2, text));
        }
        if !pixels {
            for stroke in &drawing.strokes {
                let [start, end] = stroke.segment;
                scene
                    .canvas
                    .connector(start, end, (!stroke.guide, stroke.ink));
            }
        }
        scene.axes = Some(drawing);
        scene
    }

    #[cfg(unix)]
    fn render_axes(&self, locale: Locale, color: bool, pixels: bool) -> String {
        let mut out = String::new();
        let title = locale.text("Tensor cross-sections", "张量截面对齐", "張量截面對齊");
        let hint = locale.text(
            "Shared last-axis scale · ? unavailable",
            "末轴共用比例 · ? 缺少证据",
            "末軸共用比例 · ? 缺少證據",
        );
        let divider = self.appearance.divider().to_string();
        for line in [title, hint, &divider.repeat(self.width.min(100))] {
            push(&mut out, line, self.width);
        }
        self.write_art(&mut out, 0, color, pixels);
        push(
            &mut out,
            locale.text("+/- zoom", "+/- 缩放", "+/- 縮放"),
            self.width,
        );
        out
    }

    fn write_art(&self, out: &mut String, active: usize, color: bool, pixels: bool) {
        for (row, underneath) in self.labels.iter().enumerate() {
            if pixels {
                let mut column = 0;
                for (start, caption) in &self.captions[row] {
                    if *start >= column {
                        out.extend(std::iter::repeat_n(' ', start - column));
                        out.push_str(caption);
                        column = *start + text::display_width(caption) + 1;
                        // The next caption is given at least one separating cell.
                        out.push(' ');
                    }
                }
                if column > 0 {
                    out.pop();
                }
                out.push('\n');
            } else {
                self.canvas.write_row(
                    out,
                    row,
                    underneath,
                    &self.captions[row],
                    active,
                    color.then_some(self.appearance),
                );
            }
        }
    }

    #[cfg(unix)]
    fn handoffs_available(&self) -> bool {
        !self.handoffs.is_empty()
    }

    fn art_extent(&self) -> [f64; 2] {
        [self.width as f64, ((self.labels.len() - 1) * 2) as f64]
    }

    #[cfg(unix)]
    fn art_row_at(&self, column: usize, row: usize) -> Option<usize> {
        let row = row.checked_sub(SCENE_HEADER + self.tooltip_rows)?;
        (column < self.width && row < self.labels.len() - 1).then_some(row)
    }

    #[cfg(unix)]
    fn layer_at(&self, column: usize, row: usize) -> Option<usize> {
        let row = self.art_row_at(column, row)?;
        self.canvas.layer_at(column, row).or_else(|| {
            // A one-row gap between slices still belongs to the stack.
            let above = self.canvas.layer_at(column, row.checked_sub(1)?)?;
            (self.canvas.layer_at(column, row + 1) == Some(above)).then_some(above)
        })
    }

    #[cfg(unix)]
    fn handoffs_at(&self, column: usize, row: usize) -> Vec<Wire> {
        let Some(art_row) = self.art_row_at(column, row) else {
            return Vec::new();
        };
        let point = [column as f64 + 0.5, art_row as f64 * 2.0 + 1.0];
        let layer = self.layer_at(column, row);
        let mut sockets = Vec::new();
        let mut paths = Vec::new();
        for handoff in &self.handoffs {
            if let Some(distance) = handoff.socket_distance_at(point, layer) {
                sockets.push((distance, handoff.wire));
            } else if layer.is_none() {
                if let Some(distance) = handoff
                    .distance_at(point)
                    .into_iter()
                    .chain(handoff.nearby_socket_distance_at(point))
                    .min_by(f64::total_cmp)
                {
                    paths.push((distance, handoff.wire));
                }
            }
        }
        let mut hits = if sockets.is_empty() { paths } else { sockets };
        hits.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut seen = BTreeSet::new();
        hits.into_iter()
            .map(|(_, wire)| wire)
            .filter(|&wire| seen.insert(wire))
            .collect()
    }

    #[cfg(unix)]
    fn displayed_wire(&self, wire: Wire, active: usize, cell: Option<(usize, usize)>) -> Wire {
        let routes = self.handoffs.iter().filter(|handoff| handoff.wire == wire);
        let near = cell.and_then(|(column, row)| {
            let row = self.art_row_at(column, row)?;
            let point = [column as f64 + 0.5, row as f64 * 2.0 + 1.0];
            let layer = self.layer_at(column, row + SCENE_HEADER + self.tooltip_rows);
            routes
                .clone()
                .filter_map(|handoff| {
                    handoff
                        .socket_distance_at(point, layer)
                        .map(|distance| (0, distance, handoff))
                        .or_else(|| {
                            handoff
                                .distance_at(point)
                                .map(|distance| (1, distance, handoff))
                        })
                        .or_else(|| {
                            layer
                                .is_none()
                                .then(|| handoff.nearby_socket_distance_at(point))
                                .flatten()
                                .map(|distance| (2, distance, handoff))
                        })
                })
                .min_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)))
                .map(|(_, _, handoff)| handoff)
        });
        near.or_else(|| {
            routes
                .clone()
                .find(|handoff| handoff.layers.contains(&active))
        })
        .or_else(|| routes.into_iter().next())
        .map_or(wire, |handoff| handoff.displayed)
    }

    fn refine(&mut self, camera: Camera, budget: usize) -> bool {
        let end = self
            .outline_cursor
            .saturating_add(budget)
            .min(self.solids.len());
        for &solid in &self.solids[self.outline_cursor..end] {
            self.canvas.outline(camera, solid);
        }
        let changed = self.outline_cursor < end;
        self.outline_cursor = end;
        changed
    }
}

struct ModelView<'a> {
    document: GraphDocument<'a>,
    layers: Vec<Layer<'a>>,
    indices: HashMap<OperatorId, usize>,
    wires: HashMap<[usize; 2], Vec<Wire>>,
    /// Explicit negative evidence for exact port pairs in this snapshot.
    disconnected: BTreeSet<Wire>,
    scope_crossings: Vec<Vec<[usize; 2]>>,
    children: Vec<Vec<usize>>,
    parents: Vec<Option<usize>>,
    roots: Vec<usize>,
    scope_layouts: Vec<Option<CachedFlow>>,
    unconnected_scopes: Vec<Option<FlowLayout>>,
    overview_layout: Option<CachedFlow>,
    hierarchy: bool,
    example: bool,
    offline: bool,
    edit_handoff: bool,
    source_project: Option<source::Project>,
    source_error: Option<String>,
    selected: usize,
    focus_pulse: f64,
    hover: Option<usize>,
    hover_handoff: Option<Wire>,
    selected_wire: Option<Wire>,
    selected_port: Option<Port>,
    /// A selected matrix port pair, without an assertion that a wire exists.
    selected_pair: Option<Wire>,
    focus: Option<usize>,
    trail: Vec<Bookmark>,
    #[cfg(unix)]
    last_search: Option<search::Search>,
    camera: Camera,
    viewport: Viewport,
    spatial_presets: [Option<SpatialPresetView>; 3],
    module_layout: ModuleLayout,
    layout_edit: bool,
    mouse_capture: bool,
    style: Style,
    background: Background,
    edit_overlay: Option<EditOverlay>,
    live_evidence: Option<LiveEvidence>,
    flat_flow: bool,
    explosion: u8,
    saved_camera: Option<Camera>,
    saved_viewport: Option<Viewport>,
    flat_viewport: Option<Viewport>,
    #[cfg(unix)]
    live_axes_zoom: Option<u8>,
}

struct LiveEvidence {
    path: String,
    ambiguous_registration: bool,
    last_observed: bool,
    from_calls: bool,
    reused: bool,
    input: Option<Shape>,
    invocation: Option<u64>,
    invocation_id: Option<String>,
    expected: Option<Shape>,
    raw_report: String,
    #[cfg(unix)]
    route_cache: std::cell::OnceCell<Vec<(String, String, String)>>,
    #[cfg(unix)]
    inputs: Vec<failure::Input>,
    #[cfg(unix)]
    selected_input: Option<usize>,
    #[cfg(unix)]
    show_input: bool,
    #[cfg(unix)]
    graph_target_unresolved: bool,
    #[cfg(unix)]
    projection: Option<ObservedProjection>,
    #[cfg(unix)]
    diagnostics: Vec<(String, String)>,
}

#[cfg(unix)]
struct ObservedProjection {
    source: Option<String>,
    counts: HashMap<OperatorId, usize>,
    hidden: bool,
    limited: bool,
    unmapped: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EditOverlay {
    Draft {
        existing: Wire,
        candidate: Option<Wire>,
    },
    Confirmed {
        target: Inlet,
        expanded: bool,
    },
    Evidence {
        current: Wire,
        observed: Option<Wire>,
        ink: LinkInk,
    },
}

#[derive(Clone, Copy)]
struct PortMark {
    point: [f64; 2],
    radius: f64,
    ink: LinkInk,
}

impl PortMark {
    fn points(self) -> [[f64; 2]; 5] {
        let [x, y] = self.point;
        let r = self.radius;
        [[x - r, y], [x, y - r], [x + r, y], [x, y + r], [x - r, y]]
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Bookmark {
    focus: Option<OperatorId>,
    selected: OperatorId,
    wire: Option<Wire>,
    port: Option<Port>,
    pair: Option<Wire>,
    camera: Camera,
    viewport: Viewport,
    explosion: u8,
    spatial_presets: [Option<SpatialPresetView>; 3],
    flat_viewport: Option<Viewport>,
    hierarchy: bool,
}

#[derive(Clone, Copy, PartialEq)]
struct SpatialPresetView {
    camera: Camera,
    viewport: Viewport,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GraphScope {
    focus: Option<OperatorId>,
    hierarchy: bool,
}

#[derive(Clone, Default)]
struct ModuleLayout {
    offsets: HashMap<(GraphScope, OperatorId), [f64; 2]>,
    #[cfg(unix)]
    generation: u64,
}

impl ModuleLayout {
    fn offset(&self, scope: GraphScope, operator: OperatorId) -> [f64; 2] {
        self.offsets
            .get(&(scope, operator))
            .copied()
            .unwrap_or_default()
    }

    #[cfg(unix)]
    fn set_offset(&mut self, scope: GraphScope, operator: OperatorId, offset: [f64; 2]) {
        if self.offset(scope, operator) == offset {
            return;
        }
        if offset == [0.0; 2] {
            self.offsets.remove(&(scope, operator));
        } else {
            self.offsets.insert((scope, operator), offset);
        }
        self.generation = self.generation.wrapping_add(1);
    }

    #[cfg(unix)]
    fn reset_scope(&mut self, scope: GraphScope) {
        let previous = self.offsets.len();
        self.offsets.retain(|&(saved, _), _| saved != scope);
        if self.offsets.len() != previous {
            self.generation = self.generation.wrapping_add(1);
        }
    }
}

struct CachedFlow {
    members: Vec<usize>,
    edges: Vec<[usize; 2]>,
    layout: FlowLayout,
}

impl CachedFlow {
    fn build(members: &[usize], edges: Vec<[usize; 2]>, previous: Option<Self>) -> Self {
        if let Some(previous) =
            previous.filter(|entry| entry.members == members && entry.edges == edges)
        {
            return previous;
        }
        let linear = edges.len() + 1 == members.len()
            && edges
                .iter()
                .zip(members.windows(2))
                .all(|(edge, pair)| edge == pair);
        let layout = if linear {
            FlowLayout::chain(members)
        } else {
            FlowLayout::directed(members, &edges)
        };
        Self {
            members: members.to_vec(),
            edges,
            layout,
        }
    }
}

#[derive(Default)]
struct LayoutCache {
    scopes: HashMap<OperatorId, CachedFlow>,
    overview: Option<CachedFlow>,
    materialized_order: HashMap<OperatorId, usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Viewport {
    scale: f64,
    offset: [f64; 2],
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            scale: 1.0,
            offset: [0.0; 2],
        }
    }
}

impl Viewport {
    fn point(self, [x, y]: [f64; 2]) -> [f64; 2] {
        [
            x * self.scale + self.offset[0],
            y * self.scale + self.offset[1],
        ]
    }

    #[cfg(unix)]
    fn zoom(&mut self, factor: f64, anchor: [f64; 2]) {
        let next = (self.scale * factor).clamp(0.65, 2.5);
        let ratio = next / self.scale;
        for (offset, anchor) in self.offset.iter_mut().zip(anchor) {
            *offset = anchor + (*offset - anchor) * ratio;
        }
        self.scale = next;
    }
}

struct RouteNode {
    layer: usize,
    tier: usize,
    top: [f64; 2],
    bottom: [f64; 2],
    left: [f64; 2],
    right: [f64; 2],
    solid: Cuboid,
    slices: std::ops::Range<usize>,
    half_slot: f64,
}

impl RouteNode {
    fn ordered_port(
        &self,
        solids: &[Cuboid],
        camera: Camera,
        vertical: bool,
        direction: f64,
        ordinal: usize,
        count: usize,
    ) -> [f64; 2] {
        if count == 1 {
            return if vertical && direction < 0.0 {
                self.top
            } else if vertical {
                self.bottom
            } else if direction < 0.0 {
                self.left
            } else {
                self.right
            };
        }
        let stack = &solids[self.slices.clone()];
        let position = (ordinal as f64 + 0.5) * stack.len() as f64 / count as f64;
        let slice = stack[stack.len() - 1 - position as usize];
        let axis = usize::from(vertical);
        if count <= stack.len() {
            slice.port(camera, axis, direction)
        } else {
            let [min, max] = slice.bounds(camera);
            let cross = 1 - axis;
            let at = min[cross] + (max[cross] - min[cross]) * (0.15 + position.fract() * 0.7);
            slice.port_at(camera, axis, direction, Some(at))
        }
    }
}

#[derive(Clone, Copy)]
struct Curve {
    points: [[f64; 2]; 4],
    #[cfg_attr(not(unix), allow(dead_code))]
    layers: [usize; 2],
}

/// A visual association with partial object-identity evidence, never a Wire.
#[cfg(unix)]
struct ObservedRoute {
    points: Vec<[f64; 2]>,
    target: usize,
    external: bool,
}

#[cfg(unix)]
impl ObservedRoute {
    fn draw(&self, mut segment: impl FnMut([f64; 2], [f64; 2])) {
        for pair in self.points.windows(2) {
            let [a, b] = [pair[0], pair[1]];
            let length = (b[0] - a[0]).hypot(b[1] - a[1]);
            if length < 0.01 {
                continue;
            }
            let steps = (length / 0.65).ceil().clamp(1.0, 512.0) as usize;
            for i in (0..steps).step_by(2) {
                let point = |t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
                segment(
                    point(i as f64 / steps as f64),
                    point((i + 1).min(steps) as f64 / steps as f64),
                );
            }
        }
    }
}

struct Group {
    left: f64,
    right: f64,
    top: f64,
}

impl Group {
    fn points(&self) -> [[f64; 2]; 4] {
        let [a, b, y] = [self.left, self.right, self.top];
        [[a, y + 1.35], [a, y], [b, y], [b, y + 1.35]]
    }
}

impl Curve {
    fn point(self, t: f64) -> [f64; 2] {
        let u = 1.0 - t;
        let weights = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
        std::array::from_fn(|axis| {
            self.points
                .iter()
                .zip(weights)
                .map(|(point, weight)| point[axis] * weight)
                .sum()
        })
    }

    fn segments(self, max_step: f64) -> impl Iterator<Item = ([f64; 2], [f64; 2])> {
        let length: f64 = self
            .points
            .windows(2)
            .map(|pair| (pair[0][0] - pair[1][0]).hypot(pair[0][1] - pair[1][1]))
            .sum();
        let steps = (length / max_step).ceil().clamp(4.0, 512.0) as usize;
        let mut previous = self.points[0];
        (1..=steps).map(move |step| {
            let next = self.point(step as f64 / steps as f64);
            (std::mem::replace(&mut previous, next), next)
        })
    }
}

fn route_hierarchy(
    nodes: &[RouteNode],
    parents: &[Option<usize>],
    focus: Option<usize>,
    gutter: f64,
) -> Vec<Curve> {
    let mut routes = Vec::new();
    for (lane, parent) in nodes.iter().enumerate() {
        let mut children = nodes
            .iter()
            .filter(|child| parents[child.layer] == Some(parent.layer))
            .peekable();
        let start = parent.right;
        let hub = [gutter.max(start[0] + 1.0), start[1]];
        if focus == Some(parent.layer) && children.peek().is_some() {
            let dx = hub[0] - start[0];
            routes.push(Curve {
                points: [
                    start,
                    [start[0] + dx * 0.3, start[1]],
                    [hub[0] - dx * 0.3, hub[1]],
                    hub,
                ],
                layers: [parent.layer; 2],
            });
        }
        for child in children {
            let points = if focus == Some(parent.layer) {
                let sibling_before = |node: &&RouteNode| {
                    parents[node.layer] == Some(parent.layer)
                        && node.tier == child.tier
                        && node.left[0] < child.left[0]
                };
                let from_below = child.left[1] <= hub[1];
                let edge = |node: &RouteNode| if from_below { node.bottom } else { node.top };
                let margin = if from_below { 1.6 } else { -1.6 };
                let clearance = nodes
                    .iter()
                    .filter(sibling_before)
                    .map(|node| edge(node)[1] + margin)
                    .reduce(|a, b| if from_below { a.max(b) } else { a.min(b) });
                if let Some(clearance) = clearance {
                    let end = edge(child);
                    let bend = if from_below {
                        clearance.max(end[1] + margin)
                    } else {
                        clearance.min(end[1] + margin)
                    };
                    [
                        hub,
                        [hub[0] + (end[0] - hub[0]) * 0.45, bend],
                        [end[0], bend],
                        end,
                    ]
                } else {
                    let end = child.left;
                    let reach = ((end[0] - hub[0]).abs() * 0.48).clamp(1.5, 22.0);
                    [hub, [hub[0] + reach, hub[1]], [end[0] - reach, end[1]], end]
                }
            } else {
                let a = parent.top;
                let b = child.top;
                let dx = b[0] - a[0];
                let rise = (dx.abs() * 0.13).clamp(2.5, 8.5) + (lane % 3) as f64 * 0.75;
                let crest = (a[1].min(b[1]) - rise).max(0.1);
                [a, [a[0] + dx * 0.15, crest], [b[0] - dx * 0.15, crest], b]
            };
            routes.push(Curve {
                points,
                layers: [parent.layer, child.layer],
            });
        }
    }
    routes
}

impl Layer<'_> {
    fn dimensions(&self, fields: &[&str]) -> String {
        fields
            .iter()
            .filter_map(|key| {
                let value = self.data.get(*key)?;
                if let Some(dims) = value
                    .as_array()
                    .filter(|_| matches!(*key, "shape" | "weight_shape"))
                {
                    let mut dimensions = dims
                        .iter()
                        .take(8)
                        .map(|dim| {
                            dim.as_i64()
                                .map(|n| n.to_string())
                                .unwrap_or_else(|| "?".to_owned())
                        })
                        .collect::<Vec<_>>();
                    if dims.len() > 8 {
                        dimensions.push("…".to_owned());
                    }
                    Some(format!("{key}: [{}]", dimensions.join(",")))
                } else if matches!(*key, "parameter_count" | "parameters") {
                    value.as_u64().map(|count| format!("{key}: {count}"))
                } else if value.is_number() {
                    Some(format!("{key}: {value}"))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("  ·  ")
    }

    fn hover_heading(&self, prefix: &str, width: usize, locale: Locale) -> String {
        let stack = self.data["stack"]
            .as_u64()
            .map_or_else(|| "?".into(), |count| count.to_string());
        let params =
            parameter_count(&self.data).map_or_else(|| "?".into(), |count| count.to_string());
        let stack_label = locale.text("stack", "堆叠", "堆疊");
        let params_label = locale.text("params", "参数量", "參數量");
        let counts = format!("{stack_label} {stack} · {params_label} {params}");
        let name_width =
            width.saturating_sub(text::display_width(prefix) + text::display_width(&counts) + 3);
        if name_width > 1 {
            let path = if text::display_width(&self.path) <= name_width {
                &self.path
            } else {
                self.path.rsplit('.').next().unwrap_or(&self.path)
            };
            let name = text::truncate(path, name_width);
            format!("{prefix}{name} · {counts}")
        } else {
            format!("{prefix}{counts}")
        }
    }

    fn geometry(
        &self,
        slot: f64,
        rows: usize,
        zoomed: bool,
        emphasize_stack: bool,
    ) -> ([f64; 3], usize) {
        let shape = self
            .data
            .get("shape")
            .or_else(|| self.data.get("weight_shape"))
            .and_then(Value::as_array);
        let dimension = |key: &str, axis: usize| {
            self.data[key]
                .as_u64()
                .or_else(|| shape?.get(axis)?.as_u64())
                .filter(|value| *value > 0)
        };
        let scale = |value: u64| (value as f64).log2() + 1.0;
        let aspect = match (dimension("width", 1), dimension("length", 0)) {
            (Some(width), Some(length)) => (scale(width) / scale(length)).clamp(0.45, 1.75),
            _ => 1.0,
        };
        let stack = self.data["stack"]
            .as_u64()
            .unwrap_or(self.appearance.count as u64)
            .clamp(1, (rows + 1).min(if zoomed { 9 } else { 5 }) as u64)
            as usize;
        let size = (slot * 0.40)
            .min(rows as f64 * if zoomed { 1.13 } else { 0.81 })
            .min(if zoomed { 40.0 } else { 11.0 });
        let width = size * (aspect / 1.4).min(1.0);
        let depth =
            dimension("depth", 2).map_or(0.46, |value| (scale(value) / 14.0).clamp(0.28, 0.85));
        let [width_scale, depth_scale] = self.appearance.footprint;
        let width = width * width_scale;
        let depth = size * depth * depth_scale;
        let height = if stack == 1 {
            (size * 0.08).clamp(0.12, 0.52)
        } else {
            let room = (rows as f64 * 0.69).max(0.68);
            let magnification = if zoomed {
                size * 0.18
            } else if emphasize_stack {
                size * 0.36
            } else {
                0.0
            };
            (0.80 + stack as f64 * 0.58 + magnification).min(room)
        };
        let fit = (slot * 0.47 / width.hypot(depth)).min(1.0);
        ([width * fit, height, depth * fit], stack)
    }
}

fn wire(from: usize, to: usize, slot: u32) -> Wire {
    Wire {
        from: Outlet {
            operator: OperatorId(from as u64),
            slot: 0,
        },
        to: Inlet {
            operator: OperatorId(to as u64),
            slot,
        },
    }
}

fn materialize_sequential_aliases<'a>(mut records: Vec<Cow<'a, Value>>) -> Vec<Cow<'a, Value>> {
    // The CLI stores one record per instance, with all registered paths in
    // `aliases`. Materialize only paths explicitly named by a Sequential
    // declaration, and only when exactly one reported instance owns the alias.
    let mut paths = HashMap::<String, usize>::new();
    let mut path_indices = HashMap::<String, usize>::new();
    let mut originals: Vec<usize> = (0..records.len()).collect();
    let mut aliases = HashMap::<String, Option<usize>>::new();
    for (index, record) in records.iter().enumerate() {
        if let Some(path) = record["path"].as_str() {
            *paths.entry(path.to_owned()).or_default() += 1;
            path_indices.insert(path.to_owned(), index);
        }
        if record["instance_id"].as_str().is_none_or(str::is_empty) {
            continue;
        }
        let Some(names) = record["aliases"].as_array() else {
            continue;
        };
        let Some(path) = record["path"].as_str() else {
            continue;
        };
        if !names.iter().any(|alias| alias.as_str() == Some(path)) {
            continue;
        }
        let mut seen = std::collections::HashSet::new();
        for alias in names
            .iter()
            .filter_map(Value::as_str)
            .filter(|alias| seen.insert(*alias))
        {
            aliases
                .entry(alias.to_owned())
                .and_modify(|owner| *owner = None)
                .or_insert(Some(index));
        }
    }
    let mut cursor = 0;
    while cursor < records.len() {
        let parent_index = cursor;
        let parent = records[cursor]["path"].as_str().map(str::to_owned);
        let names = records[cursor]["sequential_children"].as_array().cloned();
        cursor += 1;
        let (Some(parent), Some(names)) = (parent, names) else {
            continue;
        };
        let mut seen = std::collections::HashSet::new();
        if !names.iter().all(|name| {
            name.as_str().is_some_and(|name| {
                !name.is_empty() && !name.contains('.') && seen.insert(name.to_owned())
            })
        }) {
            continue;
        }
        let mut pending = Vec::new();
        let mut conflict = false;
        for name in names.iter().filter_map(Value::as_str) {
            let path = if parent.is_empty() {
                name.to_owned()
            } else {
                format!("{parent}.{name}")
            };
            if paths.get(&path) == Some(&1) {
                if let Some(alias_owner) = aliases.get(&path) {
                    let owner = originals[path_indices[&path]];
                    if *alias_owner != Some(owner) {
                        conflict = true;
                        break;
                    }
                }
                continue;
            }
            if paths.contains_key(&path) {
                pending.clear();
                break;
            }
            let Some(Some(owner)) = aliases.get(&path) else {
                pending.clear();
                break;
            };
            pending.push((path, *owner));
        }
        if conflict {
            if let Some(parent) = records[parent_index].to_mut().as_object_mut() {
                parent.insert("_tui_order_conflict".into(), Value::Bool(true));
            }
            continue;
        }
        if records.len() + pending.len() > 4096 {
            continue;
        }
        for (path, owner) in pending {
            let mut record = records[owner].as_ref().clone();
            let Some(object) = record.as_object_mut() else {
                continue;
            };
            object.insert("path".into(), Value::String(path.clone()));
            object.insert("_tui_alias_projection".into(), Value::Bool(true));
            records.push(Cow::Owned(record));
            originals.push(owner);
            path_indices.insert(path.clone(), records.len() - 1);
            paths.insert(path, 1);
        }
    }
    records
}

fn report_segment<'a>(records: Vec<Cow<'a, Value>>, example: bool) -> Segment<'a> {
    let records = if example {
        records
    } else {
        materialize_sequential_aliases(records)
    };
    let paths: Vec<_> = records
        .iter()
        .map(|record| record["path"].as_str().unwrap_or("—"))
        .collect();
    let lookup: HashMap<_, _> = paths
        .iter()
        .copied()
        .enumerate()
        .map(|(index, path)| (path, index))
        .collect();
    let mut path_counts = HashMap::new();
    for path in &paths {
        *path_counts.entry(*path).or_insert(0_usize) += 1;
    }
    let mut children = vec![Vec::new(); paths.len()];
    let parents: Vec<_> = paths
        .iter()
        .enumerate()
        .map(|(index, &path)| {
            let mut prefix = path;
            let parent = loop {
                let Some((next, _)) = prefix.rsplit_once('.') else {
                    break lookup.get("").copied().filter(|&root| root != index);
                };
                if let Some(&parent) = lookup.get(next) {
                    break Some(parent);
                }
                prefix = next;
            };
            if let Some(parent) = parent {
                children[parent].push(index);
            }
            parent.map(|id| OperatorId(id as u64))
        })
        .collect();
    let mut wires = Vec::new();
    if example {
        wires.extend(
            DEMO_CONNECTIONS
                .iter()
                .map(|&[from, to]| wire(from, to, u32::from(from == 2 && to == 4))),
        );
    } else {
        for (parent, children) in children.iter().enumerate() {
            if !matches!(
                records[parent]["type"].as_str(),
                Some(
                    "Sequential" | "torch.nn.Sequential" | "torch.nn.modules.container.Sequential"
                )
            ) {
                continue;
            }
            if records[parent]["_tui_order_conflict"] == true {
                continue;
            }
            let ordered = if records[parent].get("sequential_children").is_some() {
                records[parent]["sequential_children"]
                    .as_array()
                    .and_then(|names| {
                        let mut seen = std::collections::HashSet::new();
                        let direct: std::collections::HashSet<_> =
                            children.iter().copied().collect();
                        if names.len() != children.len()
                            || path_counts.get(paths[parent]) != Some(&1)
                        {
                            return None;
                        }
                        names
                            .iter()
                            .map(|name| {
                                let name = name.as_str()?;
                                if name.is_empty() || name.contains('.') || !seen.insert(name) {
                                    return None;
                                }
                                let path = if paths[parent].is_empty() {
                                    name.to_owned()
                                } else {
                                    format!("{}.{}", paths[parent], name)
                                };
                                if path_counts.get(path.as_str()) != Some(&1) {
                                    return None;
                                }
                                let index = *lookup.get(path.as_str())?;
                                direct.contains(&index).then_some(index)
                            })
                            .collect::<Option<Vec<_>>>()
                    })
            } else if children.iter().enumerate().all(|(position, &index)| {
                let path = paths[index];
                let (prefix, name) = path.rsplit_once('.').unwrap_or(("", path));
                prefix == paths[parent] && name == position.to_string()
            }) {
                Some(children.clone())
            } else {
                None
            };
            if let Some(ordered) = ordered {
                wires.extend(ordered.windows(2).map(|pair| wire(pair[0], pair[1], 0)));
            }
        }
    }

    let mut edits = Vec::new();
    for (index, (record, parent)) in records.into_iter().zip(parents).enumerate() {
        let id = OperatorId(index as u64);
        let kind = record["type"].as_str().unwrap_or("—").to_owned();
        let interface = Interface::from_record(&record, &kind);
        edits.push(Edit::AddPort(
            Port::Input(Inlet {
                operator: id,
                slot: 0,
            }),
            interface.input.clone(),
        ));
        if example && index == 4 {
            edits.push(Edit::AddPort(
                Port::Input(Inlet {
                    operator: id,
                    slot: 1,
                }),
                interface.input,
            ));
        }
        edits.push(Edit::AddPort(
            Port::Output(Outlet {
                operator: id,
                slot: 0,
            }),
            interface.output,
        ));
        edits.push(Edit::AddOperator(
            id,
            Operator {
                path: record["path"].as_str().unwrap_or("—").to_owned(),
                kind,
                parent,
                attributes: record,
            },
        ));
    }
    edits.extend(wires.into_iter().map(Edit::Connect));
    Segment {
        base_revision: 0,
        edits,
    }
}

impl<'a> Layer<'a> {
    fn from_operator(id: OperatorId, operator: &Operator<'a>) -> Self {
        Self {
            id,
            path: operator.path.clone(),
            kind: operator.kind.clone(),
            data: operator.attributes.clone(),
            appearance: SliceStyle::for_module(&operator.kind),
            hop_range: hop_range(&operator.attributes),
        }
    }
}

impl<'a> ModelView<'a> {
    fn observed_match_count(&self, id: OperatorId) -> usize {
        #[cfg(unix)]
        {
            self.live_evidence
                .as_ref()
                .and_then(|evidence| evidence.projection.as_ref())
                .filter(|projection| !projection.hidden)
                .and_then(|projection| projection.counts.get(&id))
                .copied()
                .unwrap_or(0)
        }
        #[cfg(not(unix))]
        {
            let _ = id;
            0
        }
    }

    fn focus_path(&mut self, path: &str) -> bool {
        let mut matches = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| layer.path == path);
        let Some((target, _)) = matches.next() else {
            return false;
        };
        if matches.next().is_some() {
            return false;
        }
        let mut parents = Vec::new();
        let mut parent = self.parents[target];
        while let Some(index) = parent {
            parents.push(index);
            parent = self.parents[index];
        }
        for index in parents.into_iter().rev() {
            self.selected = index;
            self.zoom(index);
        }
        self.selected = target;
        self.hover = None;
        true
    }

    fn new(modules: Option<&'a [Value]>) -> Self {
        let example = modules.is_none_or(|items| items.is_empty());
        let records = if example {
            [
                json!({"path":"encoder.embedding", "type":"Embedding", "shape":[32000, 256], "parameter_count":8_192_000, "input_shape":[2,32], "output_shape":[2,32,256]}),
                json!({"path":"encoder.attention", "type":"Attention", "width":256, "heads":8, "parameter_count":263_168, "input_shape":[2,32,256], "output_shape":[2,32,256]}),
                json!({"path":"encoder.blocks", "type":"TransformerBlock", "width":256, "stack":4, "parameter_count":3_159_040, "input_shape":[2,32,256], "output_shape":[2,32,256]}),
                json!({"path":"encoder", "type":"Encoder", "stack":4, "parameter_count":11_614_208}),
                json!({"path":"merge", "type":"Add + Linear", "weight_shape":[128, 256], "parameter_count":32_896, "input_shape":[2,32,256], "output_shape":[2,32,128]}),
                json!({"path":"classifier", "type":"Linear", "weight_shape":[10, 128], "parameter_count":1_290, "input_shape":[2,32,128], "output_shape":[2,32,10]}),
            ].into_iter().map(Cow::Owned).collect::<Vec<_>>()
        } else {
            modules
                .unwrap_or_default()
                .iter()
                .map(Cow::Borrowed)
                .collect()
        };
        let mut view =
            Self::from_document(GraphDocument::default(), example, LayoutCache::default());
        view.apply_segment(report_segment(records, example))
            .expect("inspection projection creates a valid graph");
        view
    }

    fn from_document(document: GraphDocument<'a>, example: bool, mut cached: LayoutCache) -> Self {
        let mut layers: Vec<_> = document
            .operators()
            .map(|(id, operator)| Layer::from_operator(id, operator))
            .collect();
        if !cached.materialized_order.is_empty() {
            // An expanded source scope appends render indices. Existing scopes
            // can then keep their layouts even if new IDs sort before theirs.
            layers.sort_by_key(|layer| {
                cached
                    .materialized_order
                    .get(&layer.id)
                    .copied()
                    .unwrap_or(usize::MAX)
            });
        }
        let indices: HashMap<_, _> = layers
            .iter()
            .enumerate()
            .map(|(index, layer)| (layer.id, index))
            .collect();
        let mut children = vec![Vec::new(); layers.len()];
        let mut parents = vec![None; layers.len()];
        let mut roots = Vec::new();
        for (index, layer) in layers.iter().enumerate() {
            if let Some(parent) = document
                .operator(layer.id)
                .and_then(|operator| operator.parent)
            {
                let parent = indices[&parent];
                children[parent].push(index);
                parents[index] = Some(parent);
            } else {
                roots.push(index);
            }
        }
        let mut wires: HashMap<_, Vec<_>> = HashMap::new();
        for wire in document.wires() {
            let peers = [indices[&wire.from.operator], indices[&wire.to.operator]];
            wires.entry(peers).or_default().push(wire);
        }
        let pairs: BTreeSet<_> = wires
            .keys()
            .copied()
            .filter(|[from, to]| from != to)
            .collect();
        let mut scopes = vec![Vec::new(); layers.len()];
        let mut scope_crossings = vec![Vec::new(); layers.len()];
        let mut cross_scope = false;
        for &[from, to] in &pairs {
            match (parents[from], parents[to]) {
                (Some(a), Some(b)) if a == b => scopes[a].push([from, to]),
                _ => {
                    cross_scope = true;
                    if parents[from] != parents[to] {
                        scope_crossings[from].push([from, to]);
                        scope_crossings[to].push([from, to]);
                    }
                }
            }
        }
        let scope_layouts: Vec<_> = scopes
            .into_iter()
            .zip(&children)
            .zip(&layers)
            .map(|((edges, members), layer)| {
                let has_boundary = members
                    .iter()
                    .any(|&member| !scope_crossings[member].is_empty());
                (!edges.is_empty()
                    || has_boundary
                    || (!members.is_empty()
                        && layer.data["source_boundaries"].as_bool() == Some(true)))
                .then(|| CachedFlow::build(members, edges, cached.scopes.remove(&layer.id)))
            })
            .collect();
        let unconnected_scopes = children
            .iter()
            .zip(&scope_layouts)
            .map(|(members, recorded)| {
                (recorded.is_none() && !members.is_empty())
                    .then(|| FlowLayout::unconnected(members))
            })
            .collect();
        let overview_layout = (example || cross_scope)
            .then(|| {
                let involved: BTreeSet<_> = pairs.iter().flatten().copied().collect();
                let members: Vec<_> = children
                    .iter()
                    .enumerate()
                    .filter_map(|(index, children)| {
                        (children.is_empty() || involved.contains(&index)).then_some(index)
                    })
                    .collect();
                (!members.is_empty()).then(|| {
                    CachedFlow::build(&members, pairs.into_iter().collect(), cached.overview)
                })
            })
            .flatten();
        let selected = if let Some(graph) = &overview_layout {
            graph.layout.layer(0)
        } else {
            match roots.as_slice() {
                [root] if scope_layouts[*root].is_some() => children[*root][0],
                _ if layers.len() <= 12 => 0,
                _ => roots[0],
            }
        };
        Self {
            document,
            layers,
            indices,
            wires,
            disconnected: BTreeSet::new(),
            scope_crossings,
            children,
            parents,
            roots,
            scope_layouts,
            unconnected_scopes,
            overview_layout,
            hierarchy: false,
            example,
            offline: false,
            edit_handoff: false,
            source_project: None,
            source_error: None,
            selected,
            focus_pulse: 0.0,
            hover: None,
            hover_handoff: None,
            selected_wire: None,
            selected_port: None,
            selected_pair: None,
            focus: None,
            trail: Vec::new(),
            #[cfg(unix)]
            last_search: None,
            camera: Camera::default(),
            viewport: Viewport::default(),
            spatial_presets: [None; 3],
            module_layout: ModuleLayout::default(),
            layout_edit: false,
            mouse_capture: true,
            style: Style::Default,
            background: Background::configured().unwrap_or_default(),
            edit_overlay: None,
            live_evidence: None,
            flat_flow: false,
            explosion: 0,
            saved_camera: None,
            saved_viewport: None,
            flat_viewport: None,
            #[cfg(unix)]
            live_axes_zoom: None,
        }
    }

    fn apply_segment(&mut self, segment: Segment<'a>) -> Result<ChangeSet, EditError> {
        self.apply_segment_in_order(segment, false)
    }

    fn apply_segment_in_order(
        &mut self,
        segment: Segment<'a>,
        append_source_layers: bool,
    ) -> Result<ChangeSet, EditError> {
        let change = self.document.apply(segment)?;
        if !change.topology_changed {
            for &id in &change.operators {
                self.layers[self.indices[&id]] =
                    Layer::from_operator(id, self.document.operator(id).unwrap());
            }
            return Ok(change);
        }

        // Restore UI state by operator ID, never by a compact render index.
        let get_id = |index| self.layers.get(index).map(|layer: &Layer<'_>| layer.id);
        let selected = get_id(self.selected);
        let hover = self.hover.and_then(get_id);
        let focus = self.focus.and_then(get_id);
        let previous = LayoutCache {
            scopes: std::mem::take(&mut self.scope_layouts)
                .into_iter()
                .zip(&self.layers)
                .filter_map(|(entry, layer)| entry.map(|entry| (layer.id, entry)))
                .collect(),
            overview: self.overview_layout.take(),
            materialized_order: if append_source_layers {
                std::mem::take(&mut self.indices)
            } else {
                HashMap::new()
            },
        };
        let mut replacement =
            Self::from_document(std::mem::take(&mut self.document), self.example, previous);
        let get_index = |id| replacement.indices.get(&id).copied();
        replacement.selected = selected.and_then(get_index).unwrap_or(replacement.selected);
        replacement.hover = hover.and_then(get_index);
        replacement.focus = focus.and_then(get_index);
        replacement.trail = std::mem::take(&mut self.trail);
        #[cfg(unix)]
        {
            replacement.last_search = self.last_search.take();
        }
        replacement.hover_handoff = self
            .hover_handoff
            .filter(|wire| replacement.document.source(wire.to) == Some(wire.from));
        replacement.selected_wire = self
            .selected_wire
            .filter(|wire| replacement.document.source(wire.to) == Some(wire.from));
        replacement.selected_pair = self.selected_pair.filter(|pair| {
            replacement
                .document
                .ports_for(pair.from.operator)
                .any(|(port, _)| port == Port::Output(pair.from))
                && replacement
                    .document
                    .ports_for(pair.to.operator)
                    .any(|(port, _)| port == Port::Input(pair.to))
        });
        replacement.hierarchy = self.hierarchy;
        replacement.camera = self.camera;
        replacement.viewport = self.viewport;
        replacement.spatial_presets = self.spatial_presets;
        replacement.module_layout = std::mem::take(&mut self.module_layout);
        replacement.layout_edit = self.layout_edit;
        replacement.mouse_capture = self.mouse_capture;
        replacement
            .module_layout
            .offsets
            .retain(|&(scope, operator), _| {
                replacement.indices.contains_key(&operator)
                    && scope
                        .focus
                        .is_none_or(|focus| replacement.indices.contains_key(&focus))
            });
        replacement.style = self.style;
        replacement.background = self.background;
        replacement.offline = self.offline;
        replacement.source_project = self.source_project.take();
        replacement.source_error = self.source_error.take();
        replacement.live_evidence = self.live_evidence.take();
        replacement.flat_flow = self.flat_flow;
        replacement.explosion = self.explosion;
        replacement.saved_camera = self.saved_camera;
        replacement.saved_viewport = self.saved_viewport;
        replacement.flat_viewport = self.flat_viewport;
        #[cfg(unix)]
        {
            replacement.live_axes_zoom = self.live_axes_zoom;
        }
        *self = replacement;
        Ok(change)
    }

    fn flow_layout(&self) -> Option<&FlowLayout> {
        if self.hierarchy {
            return None;
        }
        self.available_flow().or_else(|| {
            (self.flat_flow || self.explosion > 0)
                .then(|| self.unconnected_flow())
                .flatten()
        })
    }

    fn unconnected_flow(&self) -> Option<&FlowLayout> {
        let container = match (self.focus, self.roots.as_slice()) {
            (Some(focus), _) => focus,
            (None, [root]) => *root,
            _ => return None,
        };
        self.unconnected_scopes[container].as_ref()
    }

    fn toggle_flat_flow(&mut self) {
        if self.flat_flow {
            self.flat_flow = false;
            self.camera = self.saved_camera.take().unwrap_or_default();
            self.flat_viewport = Some(self.viewport);
            self.viewport = self.saved_viewport.take().unwrap_or_default();
        } else if self.flow_layout().is_some()
            || (!self.hierarchy && self.unconnected_flow().is_some())
        {
            self.saved_camera = Some(self.camera);
            self.saved_viewport = Some(self.viewport);
            self.viewport = self.flat_viewport.take().unwrap_or_default();
            self.camera = Camera {
                yaw: 0.0,
                pitch: 0.0,
            };
            self.flat_flow = true;
        }
        self.hover = None;
        self.hover_handoff = None;
    }

    fn select_graph_preset(&mut self, preset: u8) -> bool {
        match preset {
            1 | 3 | 4 => {
                if self.flat_flow {
                    self.toggle_flat_flow();
                }
                self.set_explosion(match preset {
                    3 => 1,
                    4 => 2,
                    _ => 0,
                });
                true
            }
            2 => {
                if self.flow_layout().is_none()
                    && !self.flat_flow
                    && (self.hierarchy || self.unconnected_flow().is_none())
                {
                    return false;
                }
                if !self.flat_flow {
                    self.toggle_flat_flow();
                }
                self.flat_flow
            }
            _ => false,
        }
    }

    fn set_explosion(&mut self, level: u8) {
        if self.flat_flow || level > 2 || level == self.explosion {
            return;
        }
        self.spatial_presets[usize::from(self.explosion)] = Some(SpatialPresetView {
            camera: self.camera,
            viewport: self.viewport,
        });
        self.explosion = level;
        let state = self.spatial_presets[usize::from(level)].unwrap_or(SpatialPresetView {
            camera: if level == 0 {
                Camera::default()
            } else {
                Camera::isometric()
            },
            viewport: Viewport::default(),
        });
        self.camera = state.camera;
        self.viewport = state.viewport;
        self.hover = None;
        self.hover_handoff = None;
    }

    fn cycle_explosion(&mut self) {
        self.set_explosion((self.explosion + 1) % 3);
    }

    fn reset_spatial_view(&mut self) {
        let state = SpatialPresetView {
            camera: if self.explosion == 0 {
                Camera::default()
            } else {
                Camera::isometric()
            },
            viewport: Viewport::default(),
        };
        self.spatial_presets[usize::from(self.explosion)] = Some(state);
        if self.flat_flow {
            self.camera = Camera {
                yaw: 0.0,
                pitch: 0.0,
            };
            self.saved_camera = Some(state.camera);
            self.saved_viewport = Some(state.viewport);
        } else {
            self.camera = state.camera;
        }
        self.viewport = state.viewport;
        self.hover = None;
        self.hover_handoff = None;
    }

    fn graph_scope(&self) -> GraphScope {
        GraphScope {
            focus: self.focus.map(|index| self.layers[index].id),
            hierarchy: self.hierarchy,
        }
    }

    fn available_flow(&self) -> Option<&FlowLayout> {
        let container = match (self.focus, self.roots.as_slice()) {
            (None, _) if self.overview_layout.is_some() => {
                return self.overview_layout.as_ref().map(|entry| &entry.layout);
            }
            (Some(focus), _) => focus,
            (None, [root]) => *root,
            _ => return None,
        };
        self.scope_layouts[container]
            .as_ref()
            .map(|entry| &entry.layout)
    }

    #[cfg(unix)]
    fn toggle_hierarchy(&mut self) {
        if self.available_flow().is_none() {
            return;
        }
        self.hierarchy = !self.hierarchy;
        if let Some(flow) = self
            .flow_layout()
            .filter(|flow| !flow.contains(self.selected))
        {
            self.selected = flow.layer(0);
        }
        self.hover = None;
        self.viewport = Viewport::default();
    }

    fn navigate(&mut self, delta: isize) {
        if self.layers.is_empty() {
            return;
        }
        let current = self.hover.take().unwrap_or(self.selected);
        if let Some(flow) = self.flow_layout() {
            self.selected = flow.next_layer(current, delta);
        } else if let Some(focus) = self.focus {
            let children = &self.children[focus];
            let position = self.position(current);
            let next = position.saturating_add_signed(delta).min(children.len());
            self.selected = if next == 0 { focus } else { children[next - 1] };
        } else if self.layers.len() <= 12 {
            self.selected = current
                .saturating_add_signed(delta)
                .min(self.layers.len() - 1);
        } else {
            let next = self
                .position(current)
                .saturating_add_signed(delta)
                .min(self.roots.len() - 1);
            self.selected = self.roots[next];
        }
        if self.selected != current {
            self.selected_port = None;
        }
    }

    fn position(&self, index: usize) -> usize {
        if let Some(focus) = self.focus {
            self.children[focus]
                .binary_search(&index)
                .map_or(0, |offset| offset + 1)
        } else if self.layers.len() <= 12 {
            index
        } else {
            self.roots.binary_search(&index).unwrap_or(0)
        }
    }

    #[cfg(unix)]
    fn page(&mut self, scene: &Scene, forward: bool) {
        let step = scene.page_size;
        if step >= scene.item_count {
            return;
        }
        let first = scene.page_start;
        let next = if forward {
            first + step
        } else {
            scene.previous_page
        };
        if next == first || next >= scene.item_count {
            return;
        }
        self.selected = if let Some(flow) = self.flow_layout() {
            flow.layer(next)
        } else if let Some(focus) = self.focus {
            self.children[focus][next]
        } else if self.layers.len() <= 12 {
            next
        } else {
            self.roots[next]
        };
        self.hover = None;
        self.selected_port = None;
        self.viewport = Viewport::default();
    }

    fn zoom(&mut self, layer: usize) {
        if layer >= self.layers.len() {
            return;
        }
        if self.focus == Some(layer) {
            self.zoom_out();
            return;
        }
        let layer = if let Some(mut project) = self.source_project.take() {
            let id = self.layers[layer].id;
            let result = project.expand(self, id);
            self.source_project = Some(project);
            self.source_error = result.err().map(|error| error.to_string());
            if self.source_error.is_some() {
                return;
            }
            self.indices[&id]
        } else {
            layer
        };
        self.remember_view();
        self.selected_port = None;
        self.focus = Some(layer);
        let boundary_target = if let Some(project) = &self.source_project {
            let target = self.selected_wire.and_then(|wire| {
                let exposes_wire = |&child: &usize| {
                    self.document
                        .ports_for(self.layers[child].id)
                        .any(|(port, _)| {
                            project.boundary_wires(&self.document, port).contains(&wire)
                        })
                };
                if let Some(flow) = self.flow_layout() {
                    (0..flow.count())
                        .map(|index| flow.layer(index))
                        .find(exposes_wire)
                } else {
                    self.children[layer].iter().copied().find(exposes_wire)
                }
            });
            if target.is_none() {
                self.selected_wire = None;
            }
            self.hover_handoff = None;
            target
        } else {
            None
        };
        self.viewport = Viewport::default();
        self.selected = boundary_target
            .unwrap_or_else(|| self.flow_layout().map_or(layer, |graph| graph.layer(0)));
        self.hover = None;
    }

    fn remember_view(&mut self) {
        self.trail.push(self.bookmark());
    }

    fn bookmark(&self) -> Bookmark {
        Bookmark {
            focus: self.focus.map(|index| self.layers[index].id),
            selected: self.layers[self.selected].id,
            wire: self.selected_wire,
            port: self.selected_port,
            pair: self.selected_pair,
            camera: self.saved_camera.unwrap_or(self.camera),
            viewport: self.saved_viewport.unwrap_or(self.viewport),
            explosion: self.explosion,
            spatial_presets: self.spatial_presets,
            flat_viewport: if self.flat_flow {
                Some(self.viewport)
            } else {
                self.flat_viewport
            },
            hierarchy: self.hierarchy,
        }
    }

    #[cfg(unix)]
    fn follow_wire(&mut self, scene: &Scene, downstream: bool, cell: Option<(usize, usize)>) {
        if let Some(pair) = self.selected_pair {
            let port = if downstream {
                Port::Input(pair.to)
            } else {
                Port::Output(pair.from)
            };
            let Some(&target) = self.indices.get(&port.operator()) else {
                return;
            };
            if !self
                .document
                .ports_for(port.operator())
                .any(|(item, _)| item == port)
            {
                return;
            }
            self.remember_view();
            self.focus = self.parents[target].filter(|&parent| {
                self.source_project.is_none() || self.roots.as_slice() != [parent]
            });
            self.selected = target;
            self.selected_wire = None;
            self.selected_pair = None;
            self.selected_port = Some(port);
            self.viewport = Viewport::default();
            self.hover = None;
            self.hover_handoff = None;
            return;
        }
        let Some(wire) = self.selected_wire else {
            return;
        };
        let displayed = scene.displayed_wire(wire, self.selected, cell);
        let id = if downstream {
            displayed.to.operator
        } else {
            displayed.from.operator
        };
        let Some(&target) = self.indices.get(&id) else {
            return;
        };
        if self.selected == target {
            return;
        }
        self.focus_wire_endpoint(target, wire);
    }

    fn follow_observed_source(&mut self, wire: Wire) -> bool {
        if self.document.source(wire.to) != Some(wire.from) {
            return false;
        }
        let Some(&target) = self.indices.get(&wire.from.operator) else {
            return false;
        };
        self.focus_wire_endpoint(target, wire);
        true
    }

    fn focus_wire_endpoint(&mut self, target: usize, wire: Wire) {
        self.remember_view();
        if self
            .focus
            .is_some_and(|focus| focus != target && self.parents[target] != Some(focus))
        {
            self.focus = self.parents[target].filter(|&parent| {
                self.source_project.is_none() || self.roots.as_slice() != [parent]
            });
        }
        self.selected = target;
        self.selected_wire = Some(wire);
        self.selected_port = None;
        self.viewport = Viewport::default();
        self.hover = None;
        self.hover_handoff = None;
    }

    fn zoom_out(&mut self) -> bool {
        let Some(previous) = self.trail.pop() else {
            return false;
        };
        self.restore_view(previous);
        true
    }

    fn restore_view(&mut self, previous: Bookmark) {
        self.explosion = previous.explosion;
        self.spatial_presets = previous.spatial_presets;
        self.selected = self
            .indices
            .get(&previous.selected)
            .copied()
            .unwrap_or(self.selected);
        self.focus = previous.focus.and_then(|id| self.indices.get(&id).copied());
        if self.flat_flow {
            self.saved_viewport = Some(previous.viewport);
            self.viewport = previous.flat_viewport.unwrap_or_default();
        } else {
            self.viewport = previous.viewport;
            self.flat_viewport = previous.flat_viewport;
        }
        if self.flat_flow {
            self.saved_camera = Some(previous.camera);
        } else {
            self.camera = previous.camera;
        }
        self.hierarchy = previous.hierarchy;
        self.selected_wire = previous
            .wire
            .filter(|wire| self.document.source(wire.to) == Some(wire.from));
        self.selected_port = previous.port.filter(|port| {
            port.operator() == self.layers[self.selected].id
                && self
                    .document
                    .ports_for(port.operator())
                    .any(|(item, _)| item == *port)
        });
        self.selected_pair = previous.pair.filter(|pair| {
            self.document
                .ports_for(pair.from.operator)
                .any(|(port, _)| port == Port::Output(pair.from))
                && self
                    .document
                    .ports_for(pair.to.operator)
                    .any(|(port, _)| port == Port::Input(pair.to))
        });
        self.hover = None;
        self.hover_handoff = None;
    }

    fn appearance(&self) -> Appearance {
        Appearance {
            style: self.style,
            background: self.background,
        }
    }

    #[cfg(unix)]
    pub(super) fn toolbar_text(&self, width: usize) -> String {
        if !self.mouse_capture || width < GRAPH_TOOLBAR_MIN_WIDTH {
            return String::new();
        }
        if width < GRAPH_MODE_TOOLBAR_MIN_WIDTH {
            return GRAPH_TOOLBAR.to_owned();
        }
        let flow_available = self.flow_layout().is_some()
            || self.flat_flow
            || (!self.hierarchy && self.unconnected_flow().is_some());
        let active = if self.flat_flow {
            2
        } else {
            match self.explosion {
                1 => 3,
                2 => 4,
                _ => 1,
            }
        };
        let preset = |number: u8, available: bool| {
            let label = if available {
                number.to_string()
            } else {
                "·".to_owned()
            };
            if available && active == number {
                format!("{{{label}}}")
            } else {
                format!("[{label}]")
            }
        };
        format!(
            "{GRAPH_TOOLBAR}{}{}{}{}",
            preset(1, true),
            preset(2, flow_available),
            preset(3, true),
            preset(4, true),
        )
    }

    #[cfg(unix)]
    fn live_axis_view(&self) -> Option<axes::Comparison> {
        let evidence = self.live_evidence.as_ref()?;
        Some(axes::Comparison::from_shapes(
            [evidence.input.as_ref(), evidence.expected.as_ref()],
            None,
            self.live_axes_zoom?,
        ))
    }

    fn scene(&self, width: usize, height: usize, pixels: bool, animation: Option<f64>) -> Scene {
        let mut scene = Scene::new(self.appearance(), width, height, true);
        let width = scene.width;
        let extent = scene.art_extent();
        let Scene {
            canvas,
            labels,
            captions,
            ..
        } = &mut scene;
        let rows = labels.len();
        let scope = self.graph_scope();
        let mut placements = Vec::new();
        let flow = self.flow_layout();
        let exploded_scope =
            !self.flat_flow && self.explosion > 0 && (self.focus.is_some() || width < 60);
        let vertical = flow.is_some() && ((self.flat_flow && width < 60) || exploded_scope);
        let columns = if vertical {
            let spacing = if exploded_scope {
                4 + usize::from(self.explosion)
            } else {
                3
            };
            (rows.saturating_sub(1) / spacing).clamp(1, 4)
        } else {
            (width / 16).clamp(1, 7)
        };
        let lane_limit = if vertical {
            flow.map_or(usize::MAX, |graph| {
                ((width as f64 * graph.vertical_lane_span() * 0.9 / 2.0) as usize + 1).clamp(1, 32)
            })
        } else if flow.is_some() && self.available_flow().is_none() {
            (rows.saturating_sub(3) / 2).clamp(1, 32)
        } else {
            usize::MAX
        };
        let flow_page = flow.map(|graph| graph.page(self.selected, columns, lane_limit));
        let branched = flow.is_some_and(FlowLayout::branched);
        let mut lane_page = false;
        let (page_size, page_start, item_count);
        if self.layers.is_empty() {
            (page_size, page_start, item_count) = (0, 0, 0);
        } else if let Some((graph, page)) = flow.zip(flow_page.as_ref()) {
            item_count = graph.count();
            page_start = page.start;
            page_size = page.len();
            let stages = graph.stages(page.clone());
            let lane_span = graph.vertical_lane_span();
            let split_rank = graph.rank_size(stages[0].rank) > lane_limit;
            lane_page = split_rank;
            let lane = |stage: flow_layout::Stage| {
                if split_rank {
                    if stages.len() == 1 {
                        0.0
                    } else {
                        let position = stages
                            .iter()
                            .position(|item| item.layer == stage.layer)
                            .expect("visible stage belongs to the current page");
                        position as f64 / (stages.len() - 1) as f64 - 0.5
                    }
                } else {
                    graph.lane(stage)
                }
            };
            // A rank can contain many parallel branches. Size each visible
            // branch against its nearest same-rank lane so their projections
            // remain distinguishable as the screen gets narrower.
            let mut rank_slots = BTreeMap::new();
            if vertical && branched {
                for rank in stages.chunk_by(|left, right| left.rank == right.rank) {
                    let closest = rank
                        .windows(2)
                        .map(|pair| (lane(pair[1]) - lane(pair[0])).abs())
                        .filter(|gap| *gap > 1e-9)
                        .reduce(f64::min);
                    if let Some(gap) = closest {
                        rank_slots.insert(
                            rank[0].rank,
                            ((gap * width as f64 * lane_span * 0.9) as usize).max(1),
                        );
                    }
                }
            }
            let first_rank = stages[0].rank;
            let slot = width as f64 / columns.min(graph.rank_count()) as f64;
            let lane_travel = (rows as f64 * 0.82).min(25.0);
            for &stage in stages {
                let column = stage.rank - first_rank;
                if vertical {
                    let rank_height = (rows.saturating_sub(3) * 2) as f64
                        / columns.min(graph.rank_count()) as f64;
                    let node_height = ((rank_height / 2.0).floor() as usize)
                        .clamp(3, rows.saturating_sub(2).max(3));
                    let x = width as f64 * 0.5 + lane(stage) * width as f64 * lane_span;
                    let y = (column as f64 + 0.5) * rank_height;
                    placements.push((
                        stage.layer,
                        [x, y],
                        if branched {
                            (width / 2).min(*rank_slots.get(&stage.rank).unwrap_or(&(width / 2)))
                        } else {
                            width * 2 / 3
                        },
                        node_height,
                        ((y + 2.0) / 2.0) as usize,
                        false,
                    ));
                    continue;
                }
                let y = (rows - 2) as f64 + lane(stage) * lane_travel;
                placements.push((
                    stage.layer,
                    [(column as f64 + 0.5) * slot, y],
                    slot as usize,
                    if branched {
                        (rows / 2).max(2)
                    } else {
                        rows - 1
                    },
                    rows - 1,
                    false,
                ));
            }
        } else if let Some(focus) = self.focus {
            let children = &self.children[focus];
            item_count = children.len();
            let main_width = if children.is_empty() {
                width
            } else {
                width * 55 / 100
            };
            placements.push((
                focus,
                [main_width as f64 / 2.0, (rows - 2) as f64],
                main_width,
                rows - 1,
                rows - 1,
                true,
            ));
            if !children.is_empty() {
                let columns = ((width - main_width) / 18).clamp(1, 3);
                let tiers = if rows >= 12 { 2 } else { 1 };
                page_size = columns * tiers;
                let offset = self.position(self.selected).saturating_sub(1);
                let start = offset / page_size * page_size;
                page_start = start;
                let slot = (width - main_width) / columns;
                let tier_height = rows / tiers;
                for (offset, &layer) in children[start..].iter().take(page_size).enumerate() {
                    let tier = offset / columns;
                    let x = (main_width + offset % columns * slot) as f64 + slot as f64 * 0.5;
                    let y = (tier * 2 * tier_height + tier_height.saturating_sub(2)) as f64;
                    let label_row = ((tier + 1) * tier_height - 1).min(rows - 1);
                    placements.push((layer, [x, y], slot, tier_height - 1, label_row, false));
                }
            } else {
                page_size = 1;
                page_start = 0;
            }
        } else {
            let small: Vec<_> = if self.layers.len() <= 12 {
                (0..self.layers.len()).collect()
            } else {
                Vec::new()
            };
            let sequence = if self.layers.len() <= 12 {
                &small
            } else {
                &self.roots
            };
            let visible = (width / 20).clamp(1, 7).min(sequence.len());
            page_size = visible;
            item_count = sequence.len();
            let start = self.position(self.selected) / visible * visible;
            page_start = start;
            for (offset, &layer) in sequence[start..].iter().take(visible).enumerate() {
                let x = (offset as f64 + 0.5) * width as f64 / visible as f64;
                placements.push((
                    layer,
                    [x, (rows - 2) as f64],
                    width / visible,
                    rows - 1,
                    rows - 1,
                    false,
                ));
            }
        }
        let mut solids = Vec::new();
        let mut nodes = Vec::new();
        let mut flow_labels = Vec::new();
        let contextual_labels = if flow.is_some() {
            let label_inputs: Vec<_> = placements
                .iter()
                .map(|&(layer, _, slot, ..)| {
                    let observed_marker =
                        usize::from(self.observed_match_count(self.layers[layer].id) > 0);
                    (
                        self.layers[layer].path.as_str(),
                        ((slot as f64 * self.viewport.scale) as usize)
                            .saturating_sub(3)
                            .saturating_sub(observed_marker)
                            .clamp(1, width),
                    )
                })
                .collect();
            let (paths, budgets): (Vec<_>, Vec<_>) = label_inputs.into_iter().unzip();
            contextual_flow_labels(&paths, &budgets)
        } else {
            Vec::new()
        };
        let mut flow_bottom = f64::NEG_INFINITY;
        let mut slice_labels = Vec::new();
        let overview_headroom = pixels
            && flow.is_none()
            && self.focus.is_none()
            && rows >= 9
            && placements.iter().any(|&(layer, ..)| {
                self.parents[layer]
                    .is_some_and(|parent| placements.iter().any(|item| item.0 == parent))
            });
        let motion_room = (rows.saturating_sub(2) as f64 / 7.0).clamp(0.4, 1.0);
        for (placement_index, &(index, center, slot, height, label_row, zoomed)) in
            placements.iter().enumerate()
        {
            let layer = &self.layers[index];
            let offset = self.module_layout.offset(scope, layer.id);
            let displacement = [offset[0] * extent[0], offset[1] * extent[1]];
            let center = [
                center[0] + displacement[0],
                center[1] + displacement[1] + if overview_headroom { 1.75 } else { 0.0 },
            ];
            let center = self.viewport.point(center);
            // Move the complete stack around its rest position; leave the
            // camera, hit row boundaries and text labels independent of time.
            let bob = animation.filter(|_| !self.flat_flow).map_or(0.0, |time| {
                -layer.hop_range * motion_room * self.viewport.scale * hop_progress(time, layer.id)
            });
            let shape_slot = if flow.is_some() {
                slot as f64 * 0.58
            } else {
                slot as f64
            };
            let narrow_explosion_focus =
                width <= 40 && self.explosion > 0 && index == self.selected && !zoomed;
            let (mut half, mut stack) = layer.geometry(
                shape_slot,
                height.saturating_sub(1).max(1),
                zoomed,
                narrow_explosion_focus,
            );
            if self.flat_flow && flow.is_some() {
                half[1] = 1.15;
                half[2] = 0.05;
                stack = 1;
            } else if vertical && branched && slot >= 2 && stack == 1 {
                // A very small single-plate module can otherwise fall between
                // every braille sample and disappear from pointer hit-testing.
                half[0] = half[0].max(0.55);
                half[1] = half[1].max(0.40);
                half[2] = half[2].max(0.20);
            }
            let appearance = layer.appearance.styled(self.style);
            let half = half.map(|size| size * self.viewport.scale);
            let [x, base_y, z] = half;
            let y = if !self.flat_flow && stack > 1 {
                // Make the two exploded levels visibly distinct while the
                // available rank height still bounds each stack.
                (base_y * (1.0 + 0.75 * f64::from(self.explosion)))
                    .min((height as f64 * 0.8).max(base_y))
            } else {
                base_y
            };
            let step = 2.0 * y / stack as f64;
            let plate_half_height = base_y / stack as f64 * appearance.fill;
            // All plates share the same projection center and focal distance.
            let perspective = Cuboid::layer_perspective(half);
            let mid_offset = if flow.is_some() && stack > 1 {
                self.camera.projected_elevation(
                    step * (stack as f64 * 0.5).floor() + step * 0.5 - y,
                    perspective,
                )
            } else {
                0.0
            };
            let moving_center = [center[0], center[1] + bob - mid_offset];
            let start = solids.len();
            let mut bottom = f64::NEG_INFINITY;
            for plate in 0..stack {
                let solid = Cuboid {
                    center: moving_center,
                    half: if stack == 1 {
                        half
                    } else {
                        [x, plate_half_height, z]
                    },
                    elevation: if stack == 1 {
                        0.0
                    } else {
                        -y + step * (plate as f64 + 0.5)
                    },
                    layer: index,
                    key: SliceKey {
                        operator: layer.id,
                        ordinal: plate,
                    },
                    perspective,
                    material: appearance.material,
                };
                let bounds = solid.bounds(self.camera);
                if width >= 60 && self.explosion > 0 && index == self.selected && stack > 1 {
                    let label = format!("s{}", plate + 1);
                    let label_width = text::display_width(&label);
                    let row = ((bounds[0][1] + bounds[1][1]) * 0.25).round();
                    if row >= 0.0 && row < rows as f64 {
                        let right = bounds[1][0].ceil().max(0.0) as usize + 1;
                        let left = (bounds[0][0].floor().max(0.0) as usize)
                            .saturating_sub(label_width + 1);
                        let column = if right + label_width <= width {
                            Some(right)
                        } else if bounds[0][0] >= (label_width + 1) as f64 {
                            Some(left)
                        } else {
                            None
                        };
                        if let Some(column) = column {
                            slice_labels.push((row as usize, column, label));
                        }
                    }
                }
                bottom = bottom.max(bounds[1][1]);
                canvas.cuboid(self.camera, solid);
                solids.push(solid);
            }
            let top = solids[start + stack - 1];
            let mid = solids[start + stack / 2];
            nodes.push(RouteNode {
                layer: index,
                tier: label_row,
                top: top.port(self.camera, 1, -1.0),
                bottom: solids[start].port(self.camera, 1, 1.0),
                left: mid.port(self.camera, 0, -1.0),
                right: mid.port(self.camera, 0, 1.0),
                solid: mid,
                slices: start..solids.len(),
                half_slot: slot as f64 * self.viewport.scale * 0.5,
            });
            let x = center[0];
            let y = self
                .viewport
                .point([0.0, label_row as f64 * 2.0 + displacement[1]])[1];
            if flow.is_some() || pixels {
                let bottom_at_rest = bottom - bob;
                let row = ((bottom_at_rest + if branched { 0.3 } else { 2.0 }) * 0.5).ceil();
                if flow.is_some() || (row >= 0.0 && row < rows as f64) {
                    let name = contextual_labels
                        .get(placement_index)
                        .map(String::as_str)
                        .unwrap_or_else(|| {
                            self.layers[index]
                                .path
                                .rsplit('.')
                                .next()
                                .filter(|name| !name.is_empty())
                                .unwrap_or(&self.layers[index].kind)
                        });
                    let name = text::plain_text(name).replace('\n', " ");
                    let width_budget = ((slot as f64 * self.viewport.scale) as usize)
                        .saturating_sub(3)
                        .clamp(1, width);
                    let name = if self.observed_match_count(layer.id) > 0 {
                        if width_budget == 1 {
                            "?".to_owned()
                        } else {
                            format!("?{name}")
                        }
                    } else {
                        name
                    };
                    if flow.is_some() && !vertical && !branched && displacement[1] == 0.0 {
                        flow_labels.push((x, name, width_budget));
                        flow_bottom = flow_bottom.max(bottom_at_rest);
                    } else if let Some(row) = captions.get_mut(row as usize) {
                        center_caption(row, x, &name, width_budget, width);
                    }
                }
            } else if x >= 0.0 && x < width as f64 && y >= 0.0 && y < (rows * 2) as f64 {
                let label = if self.observed_match_count(layer.id) > 0 {
                    format!("?{}", index + 1)
                } else {
                    (index + 1).to_string()
                };
                let column = (x as usize).saturating_sub(label.len() / 2);
                for (cell, character) in labels[y as usize / 2][column..]
                    .iter_mut()
                    .zip(label.chars())
                {
                    *cell = character;
                }
            }
        }
        let group_label_row = ((flow_bottom + 2.0) * 0.5).ceil();
        if let Some(row) = captions.get_mut((group_label_row as usize).min(rows - 1)) {
            for (x, name, budget) in flow_labels {
                center_caption(row, x, &name, budget, width);
            }
        }
        for (row_index, column, label) in slice_labels {
            let Some(row) = captions.get_mut(row_index) else {
                continue;
            };
            let end = column + text::display_width(&label);
            if !row.iter().any(|(start, caption)| {
                let existing_end = start + text::display_width(caption);
                column < existing_end && *start < end
            }) {
                row.push((column, label));
            }
        }
        let gutter = self.viewport.point([(width * 55 / 100) as f64, 0.0])[0];
        let sockets = if flow.is_some() {
            self.project_sockets(&nodes, &solids, vertical)
        } else {
            BTreeMap::new()
        };
        let (links, handoffs) = if let Some((graph, page)) = flow.zip(flow_page) {
            let node_indices: Vec<_> = nodes.iter().map(|node| node.layer).collect();
            let mut handoffs =
                self.tensor_routes(graph, page, &node_indices, &sockets, width, rows, vertical);
            for handoff in &mut handoffs {
                handoff.orthogonal = self.flat_flow;
                handoff.vertical = vertical;
            }
            if self.flat_flow {
                let mut groups: BTreeMap<[usize; 2], Vec<usize>> = BTreeMap::new();
                for (index, handoff) in handoffs.iter().enumerate() {
                    groups.entry(handoff.layers).or_default().push(index);
                }
                for indices in groups.values_mut().filter(|indices| indices.len() > 1) {
                    indices.sort_by_key(|&index| handoffs[index].wire);
                    let count = indices.len();
                    for (position, &index) in indices.iter().enumerate() {
                        let handoff = &mut handoffs[index];
                        let fraction = (position + 1) as f64 / (count + 1) as f64;
                        let axis = usize::from(vertical);
                        handoff.bend = Some(
                            handoff.from[axis] + (handoff.to[axis] - handoff.from[axis]) * fraction,
                        );
                    }
                }
            }
            Handoff::separate_crossings(&mut handoffs);
            (Vec::new(), handoffs)
        } else {
            (
                route_hierarchy(&nodes, &self.parents, self.focus, gutter),
                Vec::new(),
            )
        };
        let edit_mark = self.edit_overlay.and_then(|overlay| {
            let (target, radius, ink) = match overlay {
                EditOverlay::Draft { existing, .. } => (existing.to, 0.8, LinkInk::Draft),
                EditOverlay::Confirmed { target, expanded } => (
                    target,
                    if expanded { 1.55 } else { 1.2 },
                    LinkInk::Confirmed,
                ),
                EditOverlay::Evidence { current, ink, .. } => (
                    current.to,
                    if ink == LinkInk::Confirmed { 1.2 } else { 0.8 },
                    ink,
                ),
            };
            let point = *sockets.get(&Port::Input(target))?;
            Some(PortMark { point, radius, ink })
        });
        // A restrained breathing diamond gives the selected module a clear
        // focus anchor in spatial view. It is a paint-only mark: it never
        // changes the scene's surface samples or pointer hit targets.
        let focus_mark = animation
            .filter(|_| !self.flat_flow && self.edit_overlay.is_none())
            .and_then(|time| {
                let node = nodes.iter().find(|node| node.layer == self.selected)?;
                let bounds = node.solid.bounds(self.camera);
                let point = [
                    (bounds[0][0] + bounds[1][0]) * 0.5,
                    (bounds[0][1] + bounds[1][1]) * 0.5,
                ];
                let cycle = std::f64::consts::TAU * time / 2.8;
                let breath = 0.5 - 0.5 * cycle.cos();
                let diameter = (bounds[1][0] - bounds[0][0]).max(bounds[1][1] - bounds[0][1]);
                let radius =
                    (diameter * 0.5 + 0.65) * (1.12 + 0.28 * breath + 0.18 * self.focus_pulse);
                Some(PortMark {
                    point,
                    radius,
                    ink: if breath >= 0.55 || self.focus_pulse >= 0.18 {
                        LinkInk::FocusBright
                    } else {
                        LinkInk::Focus
                    },
                })
            });
        let candidate = self.edit_overlay.and_then(|overlay| {
            let wire = match overlay {
                EditOverlay::Draft { candidate, .. } => candidate?,
                EditOverlay::Evidence {
                    observed, current, ..
                } => observed.filter(|wire| *wire != current)?,
                EditOverlay::Confirmed { .. } => return None,
            };
            let from = *self.indices.get(&wire.from.operator)?;
            let to = *self.indices.get(&wire.to.operator)?;
            let target = edit_mark?.point;
            let source = sockets.get(&Port::Output(wire.from)).copied();
            let mut candidate = Handoff::new(
                wire,
                [from, to],
                [
                    source.unwrap_or(if vertical {
                        [target[0], 0.65]
                    } else {
                        [0.65, target[1]]
                    }),
                    target,
                ],
                None,
                None,
                source.is_none().then_some(Continuation::Incoming),
            );
            candidate.orthogonal = self.flat_flow;
            candidate.vertical = vertical;
            Some(candidate)
        });
        #[cfg(unix)]
        let observed_routes = if flow.is_some() {
            self.observed_routes(&nodes, vertical)
        } else {
            Vec::new()
        };
        #[cfg(unix)]
        for route in observed_routes.iter().filter(|route| route.external) {
            let [x, y] = route.points[0];
            if x >= 0.0
                && x < width as f64
                && y >= 0.0
                && let Some(row) = captions.get_mut((y * 0.5) as usize)
            {
                row.push(((x as usize).min(width.saturating_sub(1)), "?".into()));
            }
        }
        let groups = if flow.is_some() && rows >= 9 && !vertical {
            self.flow_groups(&nodes, width, &mut captions[0])
        } else {
            Vec::new()
        };
        if !pixels {
            let active = self.hover.unwrap_or(self.selected);
            let chosen_wire = self.selected_wire.or(self.hover_handoff);
            for group in &groups {
                for pair in group.points().windows(2) {
                    canvas.connector(pair[0], pair[1], (false, LinkInk::Structure));
                }
            }
            for &curve in &links {
                for (from, to) in curve.segments(1.5) {
                    canvas.connector(
                        from,
                        to,
                        (curve.layers.contains(&active), LinkInk::Structure),
                    );
                }
            }
            #[cfg(unix)]
            for route in &observed_routes {
                route.draw(|from, to| {
                    canvas.connector(from, to, (route.target == active, LinkInk::Unknown))
                });
            }
            for handoff in &handoffs {
                let focused = chosen_wire.map_or_else(
                    || handoff.layers.contains(&active),
                    |wire| wire == handoff.wire,
                );
                handoff.draw(chosen_wire, |from, to| {
                    canvas.connector(from, to, (focused, handoff.alignment.into()))
                });
            }
            if let Some(candidate) = &candidate {
                candidate.draw(None, |from, to| {
                    canvas.connector(from, to, (true, LinkInk::Draft))
                });
            }
            if let Some(mark) = focus_mark {
                canvas.mark(&mark.points(), mark.ink);
            }
            if let Some(mark) = edit_mark {
                canvas.mark(&mark.points(), mark.ink);
            }
        }
        for row in captions {
            row.sort_by_key(|(column, _)| *column);
            let mut until = 0;
            row.retain(|(column, caption)| {
                if *column < until {
                    return false;
                }
                until = column + text::display_width(caption) + 1;
                true
            });
        }
        // Refine the active geometry before its neighbors. The visible page
        // caps both the coarse raster and the deferred work for large reports.
        solids.sort_by_key(|solid| solid.layer != self.selected);
        scene.interactive_motion = animation.is_some();
        scene.solids = solids;
        scene.page_size = page_size;
        scene.page_start = page_start;
        scene.lane_page = lane_page;
        scene.item_count = item_count;
        #[cfg(unix)]
        {
            scene.links = links;
            scene.handoffs = handoffs;
            scene.observed_routes = observed_routes;
            scene.candidate = candidate;
            scene.edit_mark = edit_mark;
            scene.groups = groups;
            scene.previous_page = flow.map_or(page_start.saturating_sub(page_size), |graph| {
                graph.previous_page(page_start, columns, lane_limit)
            });
        }
        scene.focus_mark = focus_mark;
        scene.vertical_flow = vertical;
        scene
    }

    #[cfg(unix)]
    fn observed_routes(&self, nodes: &[RouteNode], vertical: bool) -> Vec<ObservedRoute> {
        let Some(projection) = self
            .live_evidence
            .as_ref()
            .and_then(|evidence| evidence.projection.as_ref())
            .filter(|projection| !projection.hidden)
        else {
            return Vec::new();
        };
        let source = projection
            .source
            .as_deref()
            .and_then(|name| {
                let mut matching = self
                    .layers
                    .iter()
                    .enumerate()
                    .filter(|(_, layer)| layer.path == name);
                let (index, _) = matching.next()?;
                matching.next().is_none().then_some(index)
            })
            .and_then(|index| nodes.iter().find(|node| node.layer == index));
        nodes
            .iter()
            .filter_map(|node| {
                if !projection.counts.contains_key(&self.layers[node.layer].id)
                    || source.is_some_and(|source| source.layer == node.layer)
                {
                    return None;
                }
                let to = if vertical { node.top } else { node.left };
                let from = source.map_or_else(
                    || if vertical { [to[0], 0.5] } else { [0.5, to[1]] },
                    |source| {
                        if vertical {
                            source.bottom
                        } else {
                            source.right
                        }
                    },
                );
                let points = if source.is_none() {
                    vec![from, to]
                } else if vertical {
                    let bend = (from[1] + to[1]) * 0.5;
                    vec![from, [from[0], bend], [to[0], bend], to]
                } else {
                    let bend = (from[0] + to[0]) * 0.5;
                    vec![from, [bend, from[1]], [bend, to[1]], to]
                };
                Some(ObservedRoute {
                    points,
                    target: node.layer,
                    external: source.is_none(),
                })
            })
            .collect()
    }

    fn flow_groups(
        &self,
        nodes: &[RouteNode],
        width: usize,
        captions: &mut Vec<(usize, String)>,
    ) -> Vec<Group> {
        let mut groups = Vec::new();
        for siblings in nodes.chunk_by(|a, b| self.parents[a.layer] == self.parents[b.layer]) {
            let parent = self.parents[siblings[0].layer]
                .filter(|&index| !self.layers[index].path.is_empty());
            if let Some(parent) = parent.filter(|_| siblings.len() > 1) {
                let top = self.viewport.point([0.0, 2.0])[1];
                let (left, right) = siblings.iter().fold(
                    (f64::INFINITY, f64::NEG_INFINITY),
                    |(left, right), node| {
                        let x = node.solid.center[0];
                        (
                            left.min(x - node.half_slot + 1.0),
                            right.max(x + node.half_slot - 1.0),
                        )
                    },
                );
                if (1.3..=4.5).contains(&top) && right > 0.0 && left < width as f64 {
                    groups.push(Group { left, right, top });
                    let name = self.layers[parent]
                        .path
                        .rsplit('.')
                        .next()
                        .unwrap_or_default();
                    let name = text::plain_text(name).replace('\n', " ");
                    center_caption(
                        captions,
                        (left + right) * 0.5,
                        &name,
                        (right - left) as usize,
                        width,
                    );
                }
            }
        }
        groups
    }

    fn project_sockets(
        &self,
        nodes: &[RouteNode],
        solids: &[Cuboid],
        vertical: bool,
    ) -> BTreeMap<Port, [f64; 2]> {
        let mut sockets = BTreeMap::new();
        let mut ports = Vec::new();
        for node in nodes {
            ports.clear();
            ports.extend(
                self.document
                    .ports_for(self.layers[node.layer].id)
                    .map(|(port, _)| port),
            );
            let split = ports.partition_point(|port| matches!(port, Port::Input(_)));
            let (inputs, outputs) = ports.split_at(split);
            for (side, direction) in [(inputs, -1.0), (outputs, 1.0)] {
                for (ordinal, &port) in side.iter().enumerate() {
                    sockets.insert(
                        port,
                        node.ordered_port(
                            solids,
                            self.camera,
                            vertical,
                            direction,
                            ordinal,
                            side.len(),
                        ),
                    );
                }
            }
        }
        sockets
    }

    fn tensor_routes(
        &self,
        graph: &FlowLayout,
        page: std::ops::Range<usize>,
        nodes: &[usize],
        sockets: &BTreeMap<Port, [f64; 2]>,
        width: usize,
        rows: usize,
        vertical: bool,
    ) -> Vec<Handoff> {
        let mut routes = Vec::new();
        let mut edges = graph.edges(page);
        let crossings: BTreeSet<_> = nodes
            .iter()
            .flat_map(|&node| &self.scope_crossings[node])
            .copied()
            .filter(|[from, to]| !graph.contains(*from) || !graph.contains(*to))
            .collect();
        edges.extend(crossings);
        for layers in edges {
            for &wire in &self.wires[&layers] {
                let output_port = sockets.get(&Port::Output(wire.from)).copied();
                let input_port = sockets.get(&Port::Input(wire.to)).copied();
                let ([a, b], continuation) = match (output_port, input_port) {
                    (Some(a), Some(b)) => ([a, b], None),
                    (None, Some(b)) => (
                        [if vertical { [b[0], 0.65] } else { [0.65, b[1]] }, b],
                        Some(Continuation::Incoming),
                    ),
                    (Some(a), None) => (
                        [
                            a,
                            if vertical {
                                [a[0], (rows as f64 * 2.0) - 2.0]
                            } else {
                                [width as f64 - 0.65, a[1]]
                            },
                        ],
                        Some(Continuation::Outgoing),
                    ),
                    (None, None) => continue,
                };
                let output = self.document.shape(Port::Output(wire.from));
                let input = self.document.shape(Port::Input(wire.to));
                routes.push(Handoff::new(
                    wire,
                    layers,
                    [a, b],
                    output,
                    input,
                    continuation,
                ));
            }
        }
        if let Some(project) = &self.source_project {
            for (&port, &point) in sockets {
                for wire in project.boundary_wires(&self.document, port) {
                    let [x, y] = point;
                    let (displayed, endpoints, continuation) = match port {
                        Port::Input(to) => (
                            Wire { to, ..wire },
                            [if vertical { [x, y - 5.0] } else { [x - 5.0, y] }, point],
                            Continuation::Incoming,
                        ),
                        Port::Output(from) => (
                            Wire { from, ..wire },
                            [point, if vertical { [x, y + 5.0] } else { [x + 5.0, y] }],
                            Continuation::Outgoing,
                        ),
                    };
                    let mut handoff = Handoff::new(
                        wire,
                        [
                            self.indices[&displayed.from.operator],
                            self.indices[&displayed.to.operator],
                        ],
                        endpoints,
                        self.document.shape(Port::Output(displayed.from)),
                        self.document.shape(Port::Input(displayed.to)),
                        Some(continuation),
                    );
                    handoff.displayed = displayed;
                    routes.push(handoff);
                }
            }
        }
        routes
    }

    fn render(&self, scene: &Scene, locale: Locale, color: bool, pixels: bool) -> String {
        #[cfg(unix)]
        if scene.axes.is_some() {
            let mut output = scene.render_axes(locale, color, pixels);
            if let Some(evidence) = self
                .live_evidence
                .as_ref()
                .filter(|_| self.live_axes_zoom.is_some())
            {
                push(
                    &mut output,
                    &format!(
                        "{}  {}",
                        if evidence.reused {
                            locale.text(
                                "Reused · no new forward · input",
                                "复用 · 本次未前向 · 输入",
                                "重用 · 本次未前向 · 輸入",
                            )
                        } else {
                            locale.text("Observed input", "观测输入", "觀測輸入")
                        },
                        shape_label(evidence.input.as_ref())
                    ),
                    scene.width,
                );
                push(
                    &mut output,
                    &format!(
                        "{}  {}",
                        locale.text("Inspected snapshot", "检查快照", "檢查快照"),
                        shape_label(evidence.expected.as_ref())
                    ),
                    scene.width,
                );
                push(
                    &mut output,
                    locale.text("Esc · Back", "Esc · 返回", "Esc · 返回"),
                    scene.width,
                );
            }
            return output;
        }
        let width = scene.width;
        let active = if (self.selected_port.is_some() || self.selected_pair.is_some())
            && self.selected_wire.is_none()
        {
            self.selected
        } else {
            self.hover.unwrap_or(self.selected)
        };
        let flow = self.flow_layout();
        let unconnected_layout = flow.is_some() && self.available_flow().is_none();
        #[cfg(unix)]
        let projection = self
            .live_evidence
            .as_ref()
            .and_then(|evidence| evidence.projection.as_ref())
            .filter(|projection| !projection.hidden);
        #[cfg(unix)]
        let projection_visible = projection.is_some();
        #[cfg(not(unix))]
        let projection_visible = false;
        let mut out = String::new();
        let mode = if self.flat_flow {
            locale.text("2D FLOW", "二维流图", "二維流圖")
        } else if self.explosion > 0 {
            if self.explosion == 1 {
                locale.text("EXPLODE x1", "爆炸图 x1", "爆炸圖 x1")
            } else {
                locale.text("EXPLODE x2", "爆炸图 x2", "爆炸圖 x2")
            }
        } else {
            locale.text("SPATIAL 3D", "空间图 3D", "空間圖 3D")
        };
        let title = format!(
            "{} {mode}  /  {}",
            if self.edit_handoff {
                "E:edit S:src"
            } else {
                ""
            },
            locale.text(
                "MODEL-MAVEN STRUCTURE",
                "MODEL-MAVEN 模型结构",
                "MODEL-MAVEN 模型結構"
            )
        );
        #[cfg(unix)]
        let title = if !self.toolbar_text(width).is_empty() {
            let toolbar = self.toolbar_text(width);
            let budget = width - toolbar.len();
            let title = text::truncate(&title, budget);
            format!(
                "{title}{}{toolbar}",
                " ".repeat(budget.saturating_sub(text::display_width(&title))),
            )
        } else {
            title
        };
        push(&mut out, &title, width);
        let source = if unconnected_layout {
            locale.text("no links recorded", "未记录连线", "未記錄連線")
        } else if self.source_project.is_some() {
            locale.text(
                "SOURCE · static Drifting graph",
                "源码 · Drifting 静态图",
                "原始碼 · Drifting 靜態圖",
            )
        } else if self.offline {
            if self.layers.is_empty() {
                locale.text(
                    "OFFLINE · matching snapshot unavailable",
                    "离线 · 缺少匹配的结构快照",
                    "離線 · 缺少相符的結構快照",
                )
            } else if flow.is_some() && self.layers[active].id.0 >= large::FIRST {
                locale.text(
                    "OFFLINE · generated tensor flow",
                    "离线 · 生成样例连线",
                    "離線 · 生成樣例連線",
                )
            } else if flow.is_some() {
                locale.text(
                    "OFFLINE · observed tensor flow",
                    "离线 · 样例观测连线",
                    "離線 · 樣例觀測連線",
                )
            } else {
                locale.text(
                    "OFFLINE · structure snapshot",
                    "离线 · 结构快照",
                    "離線 · 結構快照",
                )
            }
        } else if flow.is_some() {
            if self.example {
                locale.text("DEMO · tensor flow", "演示 · 张量流", "展示 · 張量流")
            } else {
                locale.text(
                    "Sequential · declared flow",
                    "Sequential · 声明顺序",
                    "Sequential · 宣告順序",
                )
            }
        } else if self.example {
            locale.text(
                "DEMO · illustrative layers",
                "演示数据 · 示意层",
                "展示資料 · 示意層",
            )
        } else {
            locale.text(
                "INSPECT · reported modules",
                "检查报告 · 模块列表",
                "檢查報告 · 模組清單",
            )
        };
        let source = if projection_visible && unconnected_layout {
            locale
                .text("no links · ? partial", "无连线 · ? 部分", "無連線 · ? 部分")
                .to_owned()
        } else if projection_visible {
            format!("? {} · {source}", locale.text("partial", "部分", "部分"))
        } else if self.example || self.offline {
            format!("{source} v{}", self.document.revision())
        } else {
            source.to_owned()
        };
        let context = if let Some(focus) = self.focus.filter(|_| flow.is_none()) {
            let count = self.children[focus].len();
            let local = locale.text("Zoom", "局部", "局部");
            let visible_children = scene
                .solids
                .iter()
                .map(|solid| solid.layer)
                .filter(|&layer| layer != focus);
            let first = visible_children
                .clone()
                .min()
                .map(|layer| self.position(layer));
            let last = visible_children.max().map(|layer| self.position(layer));
            let range = match (first, last) {
                (Some(first), Some(last)) => format!("{first}–{last}/{count}"),
                _ => "0".to_owned(),
            };
            format!(
                "{local} · {range}  |  {}  ·  {source}",
                self.layers[focus].path
            )
        } else {
            let first = if scene.item_count == 0 {
                0
            } else {
                scene.page_start + 1
            };
            let last = (scene.page_start + scene.page_size).min(scene.item_count);
            if (scene.lane_page || (scene.vertical_flow && self.explosion > 0))
                && !unconnected_layout
            {
                format!("{first}–{last}/{}  |  {source}", scene.item_count)
            } else {
                format!("{source}  |  {first}–{last}/{}", scene.item_count)
            }
        };
        let mode = if pixels {
            locale.text("PIXELS", "像素", "像素")
        } else {
            locale.text("CELLS", "字符", "字元")
        };
        let context = if self.live_evidence.as_ref().is_some_and(|evidence| {
            evidence.ambiguous_registration
                && self
                    .layers
                    .get(active)
                    .is_some_and(|layer| layer.path == evidence.path)
        }) {
            format!(
                "{} · {context}",
                locale.text(
                    "Shared call · path unknown",
                    "共享调用 · 路径未知",
                    "共享呼叫 · 路徑未知"
                )
            )
        } else {
            context
        };
        let zoom = (self.viewport.scale * 100.0).round() as u32;
        // Keep an orientation readout beside the zoom on wider 3D canvases.
        // It gives mouse orbit a stable reference without spending scarce
        // narrow-screen space or suggesting a world-axis convention.
        let orientation = if !self.flat_flow && width >= 72 {
            let yaw = self.camera.yaw.to_degrees().rem_euclid(360.0).round() as u16;
            let pitch = self.camera.pitch.to_degrees().round() as i16;
            format!("  ·  CAM {yaw:03}°/{pitch:+03}°")
        } else {
            String::new()
        };
        let context = if unconnected_layout && self.flat_flow {
            format!("2D · {context}")
        } else if unconnected_layout && self.explosion > 0 {
            format!("x{}/2 · {context}", self.explosion)
        } else if self.flat_flow && width < 34 {
            format!("2D {zoom}% {context}")
        } else if self.flat_flow {
            format!("{zoom}%  ·  2D  ·  {context}  ·  {mode}")
        } else if self.explosion > 0 && width < 34 {
            format!("x{}/2 {zoom}% {context}", self.explosion)
        } else if self.explosion > 0 {
            format!(
                "{zoom}% · {} {}/2 · {context} · {mode}{orientation}",
                locale.text("exploded slices", "展开切片", "展開切片"),
                self.explosion
            )
        } else {
            format!("{zoom}%  ·  {context}  ·  {mode}{orientation}")
        };
        push(&mut out, &context, width);
        let divider = scene.appearance.divider().to_string();
        push(&mut out, &divider.repeat(width.min(100)), width);
        let Some(layer) = self.layers.get(active) else {
            push(
                &mut out,
                locale.text(
                    "Empty graph · Esc back",
                    "图为空 · Esc返回",
                    "圖為空 · Esc返回",
                ),
                width,
            );
            return out;
        };
        let mut details = layer.dimensions(&[
            "shape",
            "weight_shape",
            "length",
            "width",
            "depth",
            "stack",
            "parameter_count",
            "parameters",
            "heads",
        ]);
        let source_location = layer.data["source"].as_str();
        let inlet = Port::Input(Inlet {
            operator: layer.id,
            slot: 0,
        });
        let outlet = Port::Output(Outlet {
            operator: layer.id,
            slot: 0,
        });
        let port_label = |port| {
            self.source_project
                .as_ref()
                .and_then(|_| source::axis_label(&self.document, port))
                .unwrap_or_else(|| shape_label(self.document.shape(port)))
        };
        let tensor_details = (self.document.shape(inlet).is_some()
            || self.document.shape(outlet).is_some())
        .then(|| {
            let input = locale.text("IN", "输入", "輸入");
            let output = locale.text("OUT", "输出", "輸出");
            format!(
                "{input} {}  →  {output} {}",
                port_label(inlet),
                port_label(outlet)
            )
        });
        if parameter_count(&layer.data).is_none()
            && (!details.is_empty() || tensor_details.is_some() || source_location.is_some())
        {
            let unknown = locale.text("Parameter count unknown", "参数量未知", "參數量未知");
            if details.is_empty() {
                details = unknown.to_owned();
            } else {
                details = format!("{unknown} · {details}");
            }
        }
        let layer_hover = (self.hover.is_some() && self.hover_handoff.is_none()).then(|| {
            tensor_details.clone().unwrap_or_else(|| {
                let dims = layer.dimensions(&["shape", "weight_shape", "length", "width", "depth"]);
                if dims.is_empty() {
                    locale
                        .text("Dimensions unavailable", "维度未提供", "維度未提供")
                        .into()
                } else {
                    dims
                }
            })
        });
        let describe_handoff = |wire: Wire| {
            #[cfg(unix)]
            let displayed = scene.displayed_wire(
                wire,
                if self.selected_wire == Some(wire) {
                    self.selected
                } else {
                    active
                },
                None,
            );
            #[cfg(not(unix))]
            let displayed = wire;
            let producer = &self.layers[self.indices[&displayed.from.operator]];
            let consumer = &self.layers[self.indices[&displayed.to.operator]];
            let output = self.document.shape(Port::Output(displayed.from));
            let input = self.document.shape(Port::Input(displayed.to));
            let result = match Alignment::compare(output, input) {
                Alignment::Match => locale.text("shape match", "形状对齐", "形狀對齊"),
                Alignment::Features => locale.text("last axis match", "末轴对齐", "末軸對齊"),
                Alignment::Mismatch => locale.text("shape mismatch", "形状错配", "形狀不符"),
                Alignment::Unknown => locale.text("shape unresolved", "形状未知", "形狀未知"),
            };
            let boundary = if displayed != wire {
                locale.text(" · scope boundary", " · 跨层端口", " · 跨層埠")
            } else {
                ""
            };
            let labels = [
                if self.source_project.is_some() {
                    source::connection_label(
                        &producer.path,
                        &consumer.path,
                        displayed,
                        width.saturating_sub(2),
                    )
                } else if self.example && displayed.from.slot == 0 && displayed.to.slot == 0 {
                    format!("{} → {}", producer.path, consumer.path)
                } else {
                    format!(
                        "{}[{}] → {}[{}]",
                        producer.path, displayed.from.slot, consumer.path, displayed.to.slot
                    )
                },
                format!(
                    "{} → {}  ·  {result}{boundary}",
                    port_label(Port::Output(displayed.from)),
                    port_label(Port::Input(displayed.to)),
                ),
            ];
            let locations = source::port_location(&self.document, Port::Output(displayed.from))
                .zip(source::port_location(
                    &self.document,
                    Port::Input(displayed.to),
                ))
                .map(|(from, to)| format!("{from} → {to}"));
            (labels, locations)
        };
        let handoff_details = self
            .selected_wire
            .or(self
                .hover_handoff
                .filter(|_| self.selected_port.is_none() && self.selected_pair.is_none()))
            .map(describe_handoff);
        let port_focus = self
            .selected_port
            .filter(|port| self.selected_wire.is_none() && port.operator() == layer.id)
            .map(|port| {
                let (direction, slot, connected) = match port {
                    Port::Input(input) => ("in", input.slot, self.document.source(input).is_some()),
                    Port::Output(output) => (
                        "out",
                        output.slot,
                        self.document.consumers(output).next().is_some(),
                    ),
                };
                let connected = connected
                    || self.source_project.as_ref().is_some_and(|project| {
                        !project.boundary_wires(&self.document, port).is_empty()
                    });
                let status = if connected {
                    locale.text("Recorded wire available", "有已记录连线", "有已記錄連線")
                } else {
                    locale.text(
                        "No recorded wire · status unknown",
                        "无已记录连线 · 状态未知",
                        "無已記錄連線 · 狀態未知",
                    )
                };
                (
                    format!("{}.{direction}[{slot}]", layer.path),
                    format!("{status} · {}", port_label(port)),
                )
            });
        let pair_focus = self
            .selected_pair
            .filter(|_| self.selected_wire.is_none())
            .and_then(|pair| {
                let from = self.document.operator(pair.from.operator)?;
                let to = self.document.operator(pair.to.operator)?;
                let status = if self.document.source(pair.to) == Some(pair.from) {
                    locale.text("Recorded wire", "已记录连线", "已記錄連線")
                } else if self.disconnected.contains(&pair) {
                    locale.text("Declared no wire", "已声明无连线", "已宣告無連線")
                } else {
                    locale.text("Connection unknown", "连线状态未知", "連線狀態未知")
                };
                Some((
                    format!(
                        "{} · {status}",
                        locale.text("Port pair", "端口对", "連接埠對")
                    ),
                    format!(
                        "{} {}.out[{}]\n{} {}.in[{}]",
                        locale.text("Output", "输出", "輸出"),
                        from.path,
                        pair.from.slot,
                        locale.text("Input", "输入", "輸入"),
                        to.path,
                        pair.to.slot
                    ),
                ))
            });
        let hover_details = self.hover_handoff.map(describe_handoff);
        if scene.tooltip_rows > 0 {
            if let Some(shape) = &layer_hover {
                push(
                    &mut out,
                    &layer.hover_heading("  ╭─ ", width, locale),
                    width,
                );
                push(&mut out, &format!("  ╰─ {shape}"), width);
            } else if let Some(([link, shapes], _)) =
                hover_details.as_ref().or(handoff_details.as_ref())
            {
                push(&mut out, &format!("  ╭─ {link}"), width);
                push(&mut out, &format!("  ╰─ {shapes}"), width);
            } else {
                out.push_str("\n\n");
            }
        }
        scene.write_art(&mut out, active, color, pixels);
        let missing = locale.text(
            "Dimensions and parameter count unavailable",
            "维度与参数量未提供",
            "維度與參數量未提供",
        );
        let provenance = if unconnected_layout {
            locale.text(
                "Layout only · links unknown",
                "仅布局 · 连线未知",
                "僅佈局 · 連線未知",
            )
        } else if flow.is_some() {
            locale.text(
                "× mismatch · dashed unknown · H tree",
                "× 错配 · 虚线未知 · H 层级",
                "× 不符 · 虛線未知 · H 層級",
            )
        } else if self.available_flow().is_some() {
            locale.text(
                "Module tree · H tensor flow",
                "模块层级 · H 张量流",
                "模組層級 · H 張量流",
            )
        } else {
            locale.text(
                "Module tree · forward links unavailable",
                "模块层级 · 报告未提供前向连线",
                "模組層級 · 報告未提供前向連線",
            )
        };
        let detail_width = width.saturating_sub(2);
        if !scene.short {
            push(&mut out, &divider.repeat(width), width);
        }
        let focus = scene.appearance.focus();
        let compact_hover = layer_hover.filter(|_| {
            scene.tooltip_rows == 0
                && self.selected_wire.is_none()
                && port_focus.is_none()
                && pair_focus.is_none()
        });
        push(
            &mut out,
            &if compact_hover.is_some() {
                layer.hover_heading(&format!("{focus} "), width, locale)
            } else {
                pair_focus
                    .as_ref()
                    .map(|(label, _)| format!("{focus} {label}"))
                    .unwrap_or_else(|| {
                        handoff_details.as_ref().map_or_else(
                            || {
                                port_focus.as_ref().map_or_else(
                                    || format!("{focus} {}   ·   {}", layer.path, layer.kind),
                                    |(label, _)| format!("{focus} {label}"),
                                )
                            },
                            |([link, _], _)| format!("{focus} {link}"),
                        )
                    })
            },
            width,
        );
        let live_details = self
            .live_evidence
            .as_ref()
            .filter(|evidence| {
                evidence.path == layer.path
                    && self.hover_handoff.is_none()
                    && self.selected_wire.is_none()
                    && port_focus.is_none()
                    && pair_focus.is_none()
            })
            .map(|evidence| {
                let invocation = evidence
                    .invocation
                    .map_or_else(|| "?".to_owned(), |index| format!("#{index}"));
                let expected = (layer.kind == "Linear")
                    .then(|| layer.data["editable_params"]["in_features"].as_u64())
                    .flatten();
                format!(
                    "{} {invocation} · {} {}\n{} {}",
                    if evidence.reused {
                        locale.text(
                            "Reused · no new forward",
                            "复用 · 本次未前向",
                            "重用 · 本次未前向",
                        )
                    } else {
                        locale.text("Observed call", "观测调用", "觀測呼叫")
                    },
                    locale.text("input 0", "输入 0", "輸入 0"),
                    shape_label(evidence.input.as_ref()),
                    locale.text("Snapshot input features", "快照输入维度", "快照輸入維度"),
                    expected.map_or_else(|| "?".to_owned(), |value| value.to_string())
                )
            });
        #[cfg(unix)]
        let observed_details = projection.and_then(|projection| {
            projection.counts.get(&layer.id).map(|count| {
                let source = projection
                    .source
                    .as_deref()
                    .map(text::plain_text)
                    .unwrap_or_else(|| {
                        locale
                            .text("shared call", "共享调用", "共享呼叫")
                            .to_owned()
                    });
                format!(
                    "? {count} {} {source}\n{}",
                    locale.text("matches from", "项匹配来自", "項匹配來自"),
                    if scene
                        .observed_routes
                        .iter()
                        .any(|route| route.target == active && route.external)
                    {
                        locale.text(
                            "Partial · origin not shown · unproven",
                            "部分匹配 · 来源未显示 · 未证实",
                            "部分匹配 · 來源未顯示 · 未證實",
                        )
                    } else {
                        locale.text(
                            "Partial object match · not proven flow",
                            "部分对象匹配 · 不证明数据流",
                            "部分物件匹配 · 不證明資料流",
                        )
                    }
                )
            })
        });
        #[cfg(not(unix))]
        let observed_details: Option<String> = None;
        let shared_details = (self.selected_wire.is_none() && self.hover_handoff.is_none())
            .then_some(&layer.data["instance_id"])
            .and_then(Value::as_str)
            .and_then(|instance| {
                let aliases = layer.data["aliases"].as_array()?;
                let unique: std::collections::HashSet<_> =
                    aliases.iter().filter_map(Value::as_str).collect();
                (unique.len() > 1 && unique.contains(layer.path.as_str())).then(|| {
                    format!(
                        "{} {} · {} {}",
                        locale.text("Shared instance · CLI", "共享实例 · CLI", "共享實例 · CLI"),
                        text::plain_text(instance),
                        locale.text("registered paths", "注册路径", "註冊路徑"),
                        unique.len(),
                    )
                })
            });
        let detail_source = pair_focus
            .map(|(_, detail)| detail)
            .or_else(|| {
                port_focus
                    .map(|(_, detail)| detail)
                    .or(observed_details)
                    .or(live_details)
            })
            .or(shared_details)
            .or(compact_hover)
            .unwrap_or_else(|| {
                handoff_details.map_or_else(
                    || {
                        if let Some(location) = source_location {
                            let summary = match (details.is_empty(), tensor_details.as_deref()) {
                                (true, Some(interface)) => interface.to_owned(),
                                (false, Some(interface)) => format!("{details}  ·  {interface}"),
                                _ => details,
                            };
                            return format!("{location}\n{summary}");
                        }
                        match (tensor_details.as_deref(), details.is_empty()) {
                            (Some(interface), true) => interface.to_owned(),
                            (Some(interface), false) => format!("{interface}\n{details}"),
                            (None, true) => missing.to_owned(),
                            (None, false) => details,
                        }
                    },
                    |([_, shapes], locations)| match locations {
                        Some(source) => format!("{shapes}\n{source}"),
                        None => shapes,
                    },
                )
            });
        #[cfg(unix)]
        let unresolved = self
            .live_evidence
            .as_ref()
            .is_some_and(|evidence| evidence.graph_target_unresolved);
        #[cfg(not(unix))]
        let unresolved = false;
        let details = if unresolved {
            locale.text(
                "Call location unknown · graph selection unchanged",
                "调用位置未知 · 图中选择未改变",
                "呼叫位置未知 · 圖中選擇未改變",
            )
        } else {
            self.source_error.as_deref().unwrap_or(&detail_source)
        };
        let mut fields = text::wrap(details, detail_width).into_iter();
        push(
            &mut out,
            &format!("  {}", fields.next().unwrap_or_default()),
            width,
        );
        if !scene.short {
            push(
                &mut out,
                &format!("  {}", fields.next().unwrap_or_default()),
                width,
            );
        }
        let provenance = if self
            .live_evidence
            .as_ref()
            .is_some_and(|evidence| evidence.path == layer.path)
        {
            format!(
                "{} · {provenance}",
                locale.text("a Compare axes", "a 对照末轴", "a 對照末軸")
            )
        } else if self.example {
            format!(
                "{} · {provenance}",
                locale.text("e Editor", "e 编辑", "e 編輯")
            )
        } else {
            provenance.to_owned()
        };
        let flow_preset_available = self.flow_layout().is_some()
            || self.flat_flow
            || (!self.hierarchy && self.unconnected_flow().is_some());
        let preset_keys = if flow_preset_available {
            if width < 65 {
                locale.text(
                    "1 3D · 2D · 3 x1 · 4 x2",
                    "1三维 · 2二维 · 3x1 · 4x2",
                    "1三維 · 2二維 · 3x1 · 4x2",
                )
            } else {
                locale.text(
                    "1 3D · 2 Flow · 3 x1 · 4 x2",
                    "1 三维 · 2 流图 · 3 爆炸1 · 4 爆炸2",
                    "1 三維 · 2 流圖 · 3 爆炸1 · 4 爆炸2",
                )
            }
        } else {
            locale.text(
                "1 3D · 3 x1 · 4 x2",
                "1三维 · 3x1 · 4x2",
                "1三維 · 3x1 · 4x2",
            )
        };
        let provenance = if flow_preset_available && width >= 65 {
            format!("{provenance} · {preset_keys}")
        } else {
            format!("{preset_keys} · {provenance}")
        };
        let provenance = if let Some(evidence) = &self.live_evidence {
            format!(
                "F {} · {provenance}",
                if evidence.last_observed {
                    locale.text("Failure path", "故障路径", "故障路徑")
                } else {
                    locale.text("Call path", "调用路径", "呼叫路徑")
                }
            )
        } else {
            provenance
        };
        let provenance = if self
            .live_evidence
            .as_ref()
            .is_some_and(|evidence| evidence.from_calls)
        {
            format!(
                "{} · {provenance}",
                locale.text("l Call list", "l 调用列表", "l 呼叫清單")
            )
        } else {
            provenance
        };
        let provenance = if !self.flat_flow {
            format!(
                "{} · {provenance}",
                if width < 65 {
                    locale.text("x slices", "x切片", "x切片")
                } else {
                    locale.text("x slices", "x切片", "x切片")
                }
            )
        } else {
            provenance
        };
        let provenance = if width >= 65 {
            format!("/ {} · {provenance}", locale.text("Search", "搜索", "搜尋"))
        } else {
            format!("/ {provenance}")
        };
        let provenance = if self.selected_pair.is_some() {
            format!(
                "{} · {provenance}",
                locale.text(
                    "u/d ports · v evidence",
                    "u/d 两端 · v 证据",
                    "u/d 兩端 · v 證據"
                )
            )
        } else {
            provenance
        };
        #[cfg(unix)]
        let module_details_available = self.selected_pair.is_none()
            && self.selected_port.is_none()
            && self.selected_wire.is_none()
            && scene.axes.is_none();
        #[cfg(not(unix))]
        let module_details_available = false;
        #[cfg(unix)]
        let provenance = if let Some(projection) = self
            .live_evidence
            .as_ref()
            .and_then(|evidence| evidence.projection.as_ref())
        {
            let coverage = if projection.limited || projection.unmapped > 0 {
                locale.text(" · limited", " · 受限", " · 受限")
            } else {
                ""
            };
            format!(
                "o {}{coverage} · {provenance}",
                if projection.hidden {
                    locale.text("show matches", "显示匹配", "顯示匹配")
                } else {
                    locale.text("hide matches", "隐藏匹配", "隱藏匹配")
                }
            )
        } else {
            provenance
        };
        if scene.interactive_motion {
            let motion = locale.text(
                "Space · Toggle motion",
                "空格 · 开关动画",
                "空白鍵 · 開關動畫",
            );
            #[cfg(unix)]
            let projection_controls = self
                .live_evidence
                .as_ref()
                .is_some_and(|evidence| evidence.projection.is_some());
            #[cfg(not(unix))]
            let projection_controls = false;
            let line = if self.selected_pair.is_some() || projection_controls {
                format!("{provenance}   │   {motion}")
            } else if width < 65 {
                if flow_preset_available {
                    locale
                        .text(
                            "Space · 1 3D · 2 2D · 3 x1 · 4 x2",
                            "空格动画 · 1三维 · 2二维 · 3x1 · 4x2",
                            "空格動畫 · 1三維 · 2二維 · 3x1 · 4x2",
                        )
                        .to_owned()
                } else {
                    locale
                        .text(
                            "Space · 1 3D · 3 x1 · 4 x2",
                            "空格动画 · 1三维 · 3x1 · 4x2",
                            "空格動畫 · 1三維 · 3x1 · 4x2",
                        )
                        .to_owned()
                }
            } else {
                format!("{motion}   │   {provenance}")
            };
            push(&mut out, &line, width);
        } else {
            push(&mut out, &provenance, width);
        }
        if scene.item_count > scene.page_size {
            push(
                &mut out,
                locale.text(
                    "PgUp / PgDn or [ ]  Pages",
                    "PgUp / PgDn 或 [ ] 翻页",
                    "PgUp / PgDn 或 [ ] 翻頁",
                ),
                width,
            );
        }
        let controls = if self.flat_flow {
            locale.text(
                "Drag pan · +/- Zoom · Enter · z 2D/3D · Esc",
                "拖空白平移 · +/- 缩放 · Enter 展开 · z 二维/三维 · Esc",
                "拖空白平移 · +/- 縮放 · Enter 展開 · z 二維/三維 · Esc",
            )
        } else if self.source_project.is_some() {
            locale.text(
                "Edit mode: drag module · Drag canvas rotates · Enter expand · w/u/d trace · r Reset",
                "编辑模式拖模块 · 拖空白旋转 · Enter 展开 · w/u/d 追线 · r 重置",
                "編輯模式拖模組 · 拖空白旋轉 · Enter 展開 · w/u/d 追線 · r 重置",
            )
        } else {
            locale.text(
                "2× click focus · Drag canvas rotates · +/- Zoom · Edit mode: drag module · r Reset",
                "双击聚焦 · 拖空白旋转 · 滚动/+−缩放 · 编辑模式拖模块 · r 重置",
                "雙擊聚焦 · 拖空白旋轉 · 捲動/+−縮放 · 編輯模式拖模組 · r 重置",
            )
        };
        let modes = format!(
            "L {} · T {}",
            if self.layout_edit {
                locale.text(
                    "Layout on [nodes]",
                    "布局开 [模块优先]",
                    "布局開 [模組優先]",
                )
            } else {
                locale.text("Layout off", "布局关", "布局關")
            },
            if self.mouse_capture {
                locale.text("Mouse on", "鼠标开", "滑鼠開")
            } else {
                locale.text("Mouse off", "鼠标关", "滑鼠關")
            }
        );
        let modes = if module_details_available {
            format!("{modes} · {}", locale.text("v detail", "v 详情", "v 詳情"))
        } else {
            modes
        };
        #[cfg(unix)]
        let wire_list = if scene.handoffs_available() {
            format!(" · W {}", locale.text("Wires", "连线", "連線"))
        } else {
            String::new()
        };
        #[cfg(not(unix))]
        let wire_list = "";
        push(&mut out, &format!("{modes}{wire_list} · {controls}"), width);
        #[cfg(unix)]
        if self.offline && scene.handoffs_available() && !scene.short {
            push(
                &mut out,
                locale.text(
                    "w Cycle · W Wire list · u Source · d Destination",
                    "w 切换 · W 连线列表 · u 源端口 · d 目标端口",
                    "w 切換 · W 連線清單 · u 來源埠 · d 目標埠",
                ),
                width,
            );
        }
        out
    }
}

fn push(out: &mut String, value: &str, width: usize) {
    let single_line = text::plain_text(value).replace('\n', " ");
    let _ = writeln!(out, "{}", text::truncate(&single_line, width));
}

fn shape_label(shape: Option<&Shape>) -> String {
    shape.map_or_else(|| "?".into(), Shape::label)
}

fn contextual_flow_labels(paths: &[&str], budgets: &[usize]) -> Vec<String> {
    let parts: Vec<Vec<_>> = paths.iter().map(|path| path.split('.').collect()).collect();
    let mut same_leaf = BTreeMap::<&str, Vec<usize>>::new();
    for (index, path) in parts.iter().enumerate() {
        same_leaf
            .entry(path.last().copied().unwrap_or_default())
            .or_default()
            .push(index);
    }
    let mut labels: Vec<_> = parts
        .iter()
        .map(|path| path.last().copied().unwrap_or_default().to_owned())
        .collect();
    for (leaf, indices) in same_leaf {
        if indices.len() < 2 {
            continue;
        }
        let contexts: Vec<_> = indices
            .iter()
            .map(|&index| {
                (1..parts[index].len()).find_map(|depth| {
                    let ancestors =
                        &parts[index][parts[index].len() - depth - 1..parts[index].len() - 1];
                    indices
                        .iter()
                        .filter(|&&other| other != index)
                        .all(|&other| {
                            parts[other]
                                .len()
                                .checked_sub(depth + 1)
                                .map(|start| &parts[other][start..parts[other].len() - 1])
                                != Some(ancestors)
                        })
                        .then(|| ancestors.join("."))
                })
            })
            .collect();
        for (&index, context) in indices.iter().zip(&contexts) {
            let fallback = indices.iter().position(|&item| item == index).unwrap() + 1;
            let full = context.as_ref().map_or_else(
                || format!("{leaf} · #{fallback}"),
                |parent| format!("{leaf} · {parent}"),
            );
            let budget = budgets.get(index).copied().unwrap_or(usize::MAX);
            if text::display_width(&full) <= budget {
                labels[index] = full;
                continue;
            }
            let peers: Vec<_> = contexts
                .iter()
                .flatten()
                .filter(|other| Some(*other) != context.as_ref())
                .map(String::as_str)
                .collect();
            let marker = context
                .as_deref()
                .and_then(|context| shortest_unique_fragment(context, &peers))
                .unwrap_or_else(|| fallback.to_string());
            let marker = format!("#{marker}");
            let leaf_budget = budget.saturating_sub(text::display_width(&marker));
            labels[index] = format!("{}{marker}", text::truncate(leaf, leaf_budget));
        }
    }
    labels
}

fn shortest_unique_fragment(value: &str, others: &[&str]) -> Option<String> {
    let characters: Vec<_> = value.chars().collect();
    for length in 1..=characters.len() {
        for start in 0..=characters.len() - length {
            let fragment: String = characters[start..start + length].iter().collect();
            if fragment.chars().all(char::is_alphanumeric)
                && others.iter().all(|other| !other.contains(&fragment))
            {
                return Some(fragment);
            }
        }
    }
    None
}

fn center_caption(
    row: &mut Vec<(usize, String)>,
    x: f64,
    value: &str,
    budget: usize,
    width: usize,
) {
    if x < 0.0 || x >= width as f64 {
        return;
    }
    let caption = text::truncate(value, budget.min(width));
    let caption_width = text::display_width(&caption);
    let column = (x as usize)
        .saturating_sub(caption_width / 2)
        .min(width - caption_width);
    row.push((column, caption));
}

#[cfg(unix)]
mod interaction;
#[cfg(unix)]
pub(crate) use interaction::show_editor_demo;

mod offline;

pub(crate) enum GraphEdit {
    Variant(String),
    Source(String),
}

pub(crate) fn show(
    modules: Option<&Vec<Value>>,
    locale: Locale,
) -> std::io::Result<Option<GraphEdit>> {
    match modules.filter(|modules| !modules.is_empty()) {
        Some(modules) => {
            let mut view = ModelView::new(Some(modules));
            #[cfg(unix)]
            return interaction::show_view_for_edit(&mut view, locale);
            #[cfg(not(unix))]
            {
                show_view_static(&view, locale)?;
                Ok(None)
            }
        }
        None => {
            show_drifting_demo(locale)?;
            Ok(None)
        }
    }
}

pub(crate) fn show_focused(
    modules: &Vec<Value>,
    path: &str,
    call: &Value,
    raw_report: &str,
    last_observed: bool,
    from_calls: bool,
    input_query: Option<&str>,
    input_selection: Option<usize>,
    previous: Option<&FocusedViewState>,
    allow_edit: bool,
    locale: Locale,
) -> std::io::Result<FocusedViewState> {
    let mut view = ModelView::new(Some(modules));
    view.focus_path(path);
    let expected = view
        .layers
        .get(view.selected)
        .filter(|layer| layer.path == path && layer.kind == "Linear")
        .and_then(|layer| layer.data["editable_params"]["in_features"].as_u64())
        .map(Shape::Features);
    #[cfg(unix)]
    let mut inputs = failure::observed_inputs(&call["inputs"], expected.as_ref());
    #[cfg(unix)]
    if let Ok(report) = serde_json::from_str::<Value>(raw_report) {
        failure::attach_observed_matches(&mut inputs, call, &report);
    }
    #[cfg(unix)]
    let matched_input = input_selection
        .filter(|&index| index < inputs.len())
        .or_else(|| input_query.and_then(|query| failure::unique_input_match(&inputs, query)));
    #[cfg(not(unix))]
    let _ = (input_query, input_selection);
    view.live_evidence = Some(LiveEvidence {
        path: path.to_owned(),
        ambiguous_registration: crate::cli::ambiguous_registration(call),
        last_observed,
        from_calls,
        reused: serde_json::from_str::<Value>(raw_report)
            .ok()
            .is_some_and(|report| crate::cli::reused_evidence(&report["data"])),
        input: [
            &call["inputs"]["items"]["args"]["items"][0],
            &call["inputs"]["items"]["kwargs"]["items"]["input"],
        ]
        .into_iter()
        .find(|value| value["kind"] == "tensor")
        .and_then(|value| Shape::parse(&value["shape"])),
        invocation: call["invocation_index"].as_u64(),
        invocation_id: call["invocation_id"].as_str().map(str::to_owned),
        expected,
        raw_report: raw_report.to_owned(),
        #[cfg(unix)]
        route_cache: std::cell::OnceCell::new(),
        #[cfg(unix)]
        inputs,
        #[cfg(unix)]
        selected_input: matched_input.or_else(|| previous.and_then(|state| state.selected_input)),
        #[cfg(unix)]
        show_input: matched_input.is_some()
            || (input_query.is_none() && previous.is_some_and(|state| state.show_input)),
        #[cfg(unix)]
        graph_target_unresolved: false,
        #[cfg(unix)]
        projection: None,
        #[cfg(unix)]
        diagnostics: serde_json::from_str::<Value>(raw_report)
            .ok()
            .and_then(|report| report["diagnostics"].as_array().cloned())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|diagnostic| {
                Some((
                    diagnostic["code"].as_str()?.to_owned(),
                    diagnostic["message"].as_str().unwrap_or("").to_owned(),
                ))
            })
            .collect(),
    });
    if let Some(previous) = previous {
        view.flat_flow = previous.flat_flow;
        view.explosion = previous.explosion;
        view.spatial_presets = previous.spatial_presets;
        view.module_layout = previous.module_layout.clone();
        view.layout_edit = previous.layout_edit;
        view.style = previous.style;
        view.background = previous.background;
        #[cfg(unix)]
        {
            view.live_axes_zoom = previous.live_axes_zoom;
        }
        view.restore_view(previous.bookmark);
        view.trail = previous.trail.clone();
    }
    #[cfg(unix)]
    let edit = if allow_edit {
        interaction::show_view_for_edit(&mut view, locale)?
    } else {
        interaction::show_view_borrowed(&mut view, locale)?;
        None
    };
    #[cfg(not(unix))]
    let edit = {
        let _ = allow_edit;
        show_view_static(&view, locale)?;
        None
    };
    Ok(FocusedViewState {
        edit,
        bookmark: view.bookmark(),
        trail: view.trail.clone(),
        flat_flow: view.flat_flow,
        explosion: view.explosion,
        spatial_presets: view.spatial_presets,
        module_layout: view.module_layout.clone(),
        layout_edit: view.layout_edit,
        style: view.style,
        background: view.background,
        #[cfg(unix)]
        live_axes_zoom: view.live_axes_zoom,
        #[cfg(unix)]
        selected_input: view
            .live_evidence
            .as_ref()
            .and_then(|evidence| evidence.selected_input),
        #[cfg(unix)]
        show_input: view
            .live_evidence
            .as_ref()
            .is_some_and(|evidence| evidence.show_input),
    })
}

#[cfg(unix)]
pub(crate) struct InputCandidate {
    pub index: usize,
    pub path: String,
    pub metadata: String,
    pub shape: String,
}

#[cfg(unix)]
pub(crate) fn matching_call_inputs(call: &Value, query: &str) -> (Vec<InputCandidate>, bool) {
    let inputs = failure::observed_inputs(&call["inputs"], None);
    let (indices, incomplete) = failure::matching_input_indices(&inputs, query);
    let candidates = indices
        .into_iter()
        .map(|index| {
            let input = &inputs[index];
            InputCandidate {
                index,
                path: input.label.clone().unwrap_or_else(|| "?".into()),
                metadata: input.metadata.clone().unwrap_or_else(|| "?".into()),
                shape: input
                    .actual
                    .as_ref()
                    .map_or_else(|| "?".into(), |(shape, _)| shape.label()),
            }
        })
        .collect();
    (candidates, incomplete)
}

pub(crate) struct FocusedViewState {
    pub(crate) edit: Option<GraphEdit>,
    bookmark: Bookmark,
    trail: Vec<Bookmark>,
    flat_flow: bool,
    explosion: u8,
    spatial_presets: [Option<SpatialPresetView>; 3],
    module_layout: ModuleLayout,
    layout_edit: bool,
    style: Style,
    background: Background,
    #[cfg(unix)]
    live_axes_zoom: Option<u8>,
    #[cfg(unix)]
    selected_input: Option<usize>,
    #[cfg(unix)]
    show_input: bool,
}

fn show_view(view: ModelView<'_>, locale: Locale) -> std::io::Result<()> {
    #[cfg(unix)]
    return interaction::show_view(view, locale);
    #[cfg(not(unix))]
    show_view_static(&view, locale)
}

#[cfg(not(unix))]
pub(crate) fn show_editor_demo(locale: Locale) -> std::io::Result<()> {
    show_view_static(&ModelView::new(None), locale)
}

pub(crate) fn show_diagnosis(
    locale: Locale,
    case: model_maven_tui::diagnosis::Case,
    line_mode: bool,
) -> std::io::Result<()> {
    use crate::welcome::{self, ProjectState};
    let fullscreen = !line_mode && {
        #[cfg(unix)]
        {
            crate::terminal::menu::supported()
        }
        #[cfg(not(unix))]
        {
            false
        }
    };
    #[cfg(unix)]
    let _interrupt = fullscreen.then(crate::terminal::interrupt::Guard::install);
    let project = std::path::Path::new("offline-demo");
    let state = ProjectState::Configured;
    let enter = |handoff: welcome::Handoff| {
        if fullscreen {
            handoff.welcome(project, state, locale)
        } else {
            welcome::show_line(project, state, locale)
        }
    };
    let Some(mut handoff) = enter(welcome::Handoff::default())? else {
        return Ok(());
    };
    let mut offline = offline::Offline::new(case);
    let mut view = offline.view();
    loop {
        #[cfg(unix)]
        let exit = if fullscreen {
            interaction::run(&mut view, locale, None, Some(&mut offline), &mut handoff)?
        } else {
            offline.run_line(&mut view, locale)?
        };
        #[cfg(not(unix))]
        let exit = offline.run_line(&mut view, locale)?;
        match exit {
            offline::Exit::Quit => return Ok(()),
            offline::Exit::Edit => return Ok(()),
            offline::Exit::EditSource => return Ok(()),
            offline::Exit::Welcome => match enter(handoff)? {
                Some(screen) => handoff = screen,
                None => return Ok(()),
            },
        }
    }
}

fn show_view_static(view: &ModelView<'_>, locale: Locale) -> std::io::Result<()> {
    use std::io::{self, Write};
    let mut scene = view.scene(80, 24, false, None);
    scene.refine(view.camera, usize::MAX);
    let frame = view.render(&scene, locale, false, false);
    let mut output = io::stdout().lock();
    write!(output, "{frame}")?;
    output.flush()?;
    io::stdin().read_line(&mut String::new())?;
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn flow_labels_disambiguate_repeated_modules_with_their_block_instance() {
        let paths = [
            "model.blocks.block_06.attention.qkv",
            "model.blocks.block_07.attention.qkv",
            "model.blocks.block_07.mlp.qkv",
            "model.embedding",
        ];
        let labels = contextual_flow_labels(&paths, &[40; 4]);
        assert_eq!(
            labels,
            [
                "qkv · block_06.attention",
                "qkv · block_07.attention",
                "qkv · mlp",
                "embedding"
            ]
        );
        assert_eq!(
            contextual_flow_labels(&paths, &[6; 4]),
            ["qkv#6", "qkv#7", "qkv#m", "embedding"]
        );

        let same_path =
            contextual_flow_labels(&["model.shared.linear", "model.shared.linear"], &[40; 2]);
        assert_eq!(same_path, ["linear · #1", "linear · #2"]);
    }

    #[test]
    fn flow_scene_displays_context_for_repeated_module_names() {
        let mut view = ModelView::new(None);
        assert!(view.layers.len() >= 2);
        view.layers[0].path = "model.block_06.attention.qkv".into();
        view.layers[1].path = "model.block_07.attention.qkv".into();

        let scene = view.scene(100, 30, false, None);
        let captions = scene
            .captions
            .iter()
            .flat_map(|row| row.iter().map(|(_, caption)| caption.as_str()))
            .collect::<Vec<_>>();
        assert!(captions.contains(&"qkv#6"), "{captions:?}");
        assert!(captions.contains(&"qkv#7"), "{captions:?}");
    }

    #[test]
    fn graph_header_identifies_the_active_visual_preset() {
        let mut view = ModelView::new(None);
        for locale in [Locale::English, Locale::Simplified, Locale::Traditional] {
            for width in [34, 40] {
                view.flat_flow = false;
                view.explosion = 0;
                let spatial = view.scene(width, 24, false, None);
                assert!(
                    view.render(&spatial, locale, false, false)
                        .lines()
                        .next()
                        .unwrap()
                        .contains(locale.text("SPATIAL 3D", "空间图 3D", "空間圖 3D"))
                );

                view.toggle_flat_flow();
                let flow = view.scene(width, 24, false, None);
                assert!(
                    view.render(&flow, locale, false, false)
                        .lines()
                        .next()
                        .unwrap()
                        .contains(locale.text("2D FLOW", "二维流图", "二維流圖"))
                );

                view.toggle_flat_flow();
                view.explosion = 1;
                let exploded = view.scene(width, 24, false, None);
                assert!(
                    view.render(&exploded, locale, false, false)
                        .lines()
                        .next()
                        .unwrap()
                        .contains(locale.text("EXPLODE x1", "爆炸图 x1", "爆炸圖 x1"))
                );

                view.explosion = 2;
                let exploded = view.scene(width, 24, false, None);
                assert!(
                    view.render(&exploded, locale, false, false)
                        .lines()
                        .next()
                        .unwrap()
                        .contains(locale.text("EXPLODE x2", "爆炸图 x2", "爆炸圖 x2"))
                );
            }
        }
    }

    #[test]
    fn spatial_presets_restore_their_own_camera_zoom_and_navigation_bookmarks() {
        let mut view = ModelView::new(None);

        view.camera.orbit(0.25, 0.12);
        view.viewport.zoom(1.2, [18.0, 7.0]);
        let spatial = SpatialPresetView {
            camera: view.camera,
            viewport: view.viewport,
        };

        view.set_explosion(1);
        assert_eq!(view.camera, Camera::isometric());
        assert_eq!(view.viewport, Viewport::default());
        view.camera.orbit(0.45, -0.10);
        view.viewport.zoom(1.4, [24.0, 9.0]);
        let exploded_1 = SpatialPresetView {
            camera: view.camera,
            viewport: view.viewport,
        };

        view.set_explosion(2);
        assert_eq!(view.camera, Camera::isometric());
        assert_eq!(view.viewport, Viewport::default());
        view.camera.orbit(-0.32, 0.18);
        view.viewport.zoom(1.6, [28.0, 11.0]);
        let exploded_2 = SpatialPresetView {
            camera: view.camera,
            viewport: view.viewport,
        };

        view.remember_view();
        view.set_explosion(1);
        assert_eq!(view.camera, exploded_1.camera);
        assert_eq!(view.viewport, exploded_1.viewport);
        view.camera.orbit(0.11, 0.05);
        view.zoom_out();
        assert_eq!(view.explosion, 2);
        assert_eq!(view.camera, exploded_2.camera);
        assert_eq!(view.viewport, exploded_2.viewport);

        view.set_explosion(0);
        assert_eq!(view.camera, spatial.camera);
        assert_eq!(view.viewport, spatial.viewport);
        view.set_explosion(1);
        assert_eq!(view.camera, exploded_1.camera);
        assert_eq!(view.viewport, exploded_1.viewport);
        view.reset_spatial_view();
        view.set_explosion(0);
        view.set_explosion(1);
        assert_eq!(view.camera, Camera::isometric());
        assert_eq!(view.viewport, Viewport::default());

        view.toggle_flat_flow();
        let flat_viewport = view.viewport;
        view.set_explosion(2);
        assert_eq!(view.explosion, 1);
        assert_eq!(view.viewport, flat_viewport);
    }

    #[test]
    fn spatial_status_shows_camera_orientation_without_cluttering_narrow_or_2d_views() {
        let mut view = ModelView::new(None);
        let wide = view.scene(80, 24, false, None);
        let frame = view.render(&wide, Locale::English, false, false);
        assert!(
            frame.lines().any(|line| line.contains("CAM 032°/+30°")),
            "{frame}"
        );

        view.camera.orbit(0.2, -0.1);
        let rotated = view.scene(80, 24, false, None);
        let frame = view.render(&rotated, Locale::English, false, false);
        assert!(
            frame.lines().any(|line| line.contains("CAM 044°/+24°")),
            "{frame}"
        );

        let narrow = view.scene(40, 24, false, Some(0.0));
        let frame = view.render(&narrow, Locale::English, false, false);
        assert!(!frame.lines().any(|line| line.contains("CAM ")));

        view.toggle_flat_flow();
        let flow = view.scene(80, 24, false, None);
        let frame = view.render(&flow, Locale::English, false, false);
        assert!(!frame.lines().any(|line| line.contains("CAM ")));

        view.toggle_flat_flow();
        view.set_explosion(1);
        assert_eq!(view.camera, Camera::isometric());
        let exploded = view.scene(100, 24, false, None);
        let frame = view.render(&exploded, Locale::English, false, false);
        assert!(
            frame.lines().any(|line| line.contains("CAM 045°/+35°")),
            "{frame}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn exploded_slices_keep_module_identity_and_connectors_attached() {
        let mut view = ModelView::new(None);
        let original = view.scene(100, 30, false, None);
        let selected = view.layers[view.selected].id;
        let first_wire = original.handoffs.first().expect("example has a wire").wire;
        let original_slices: Vec<_> = original
            .solids
            .iter()
            .filter(|solid| solid.layer == view.selected)
            .collect();
        assert!(original_slices.len() > 1);
        let original_extent = original_slices
            .iter()
            .map(|solid| solid.elevation)
            .fold(f64::NEG_INFINITY, f64::max)
            - original_slices
                .iter()
                .map(|solid| solid.elevation)
                .fold(f64::INFINITY, f64::min);
        view.explosion = 2;
        let exploded = view.scene(100, 30, false, None);
        let expanded_slices: Vec<_> = exploded
            .solids
            .iter()
            .filter(|solid| solid.layer == view.selected)
            .collect();
        assert_eq!(expanded_slices.len(), original_slices.len());
        let expanded_extent = expanded_slices
            .iter()
            .map(|solid| solid.elevation)
            .fold(f64::NEG_INFINITY, f64::max)
            - expanded_slices
                .iter()
                .map(|solid| solid.elevation)
                .fold(f64::INFINITY, f64::min);
        assert!(expanded_extent > original_extent);
        let expanded_frame = view.render(&exploded, Locale::English, false, false);
        for ordinal in 1..=original_slices.len() {
            assert!(
                expanded_frame.contains(&format!("s{ordinal}")),
                "wide exploded view should label slice s{ordinal}: {expanded_frame}"
            );
        }
        assert!(
            expanded_slices
                .iter()
                .all(|solid| solid.key.operator == selected)
        );
        assert!(
            exploded
                .handoffs
                .iter()
                .any(|route| route.wire == first_wire)
        );
        assert_eq!(view.layers[view.selected].id, selected);
        view.camera.orbit(0.2, 0.1);
        let rotated = view.scene(100, 30, false, None);
        assert!(
            rotated
                .handoffs
                .iter()
                .any(|route| route.wire == first_wire)
        );
        view.explosion = 0;
        assert_eq!(view.layers[view.selected].id, selected);

        view.explosion = 0;
        for (width, height) in [(40, 23), (100, 29), (120, 35)] {
            for level in [1, 2] {
                view.set_explosion(0);
                view.set_explosion(level);
                for yaw in [0.0, 0.8, 1.8, 2.7] {
                    view.camera = Camera::isometric();
                    view.camera.orbit(yaw, 0.16);
                    let scene = view.scene(width, height, false, None);
                    let selected_slices: BTreeSet<_> = scene
                        .solids
                        .iter()
                        .filter(|solid| solid.layer == view.selected)
                        .map(|solid| (solid.key.operator, solid.key.ordinal))
                        .collect();
                    let visible_slice_cells = solid::visible_slice_cells_for_test(
                        &scene.solids,
                        view.camera,
                        scene.width,
                        scene.labels.len().saturating_sub(1),
                    );
                    assert_eq!(selected_slices.len(), original_slices.len());
                    let frame = view.render(&scene, Locale::English, false, false);
                    if width < 60 {
                        assert!(scene.vertical_flow);
                        assert!(
                            scene.canvas.link_dot_count_for_test() > 0,
                            "narrow exploded view must retain visible Braille connection strokes: {frame}"
                        );
                        assert!(
                            frame.lines().nth(1).is_some_and(
                                |status| status.contains(&format!("/{}", scene.item_count))
                            ),
                            "narrow exploded views must keep the current page range visible: {frame}"
                        );
                    }
                    assert!(
                        selected_slices.iter().all(|slice| {
                            visible_slice_cells.get(slice).copied().unwrap_or_default() >= 2
                        }),
                        "each selected slice needs at least two visible Braille cells at {width}×{} exploded x{level}, yaw {yaw}; visible cells={visible_slice_cells:?}\n{frame}",
                        height + 1,
                    );
                }
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn exploded_vertical_flow_uses_available_rank_height_for_stacked_modules() {
        let root = OperatorId(100);
        let modules = [OperatorId(101), OperatorId(102), OperatorId(103)];
        let mut edits = Vec::new();
        edits.push(Edit::AddOperator(
            root,
            Operator {
                path: "model".into(),
                kind: "Sequential".into(),
                parent: None,
                attributes: Cow::Owned(serde_json::json!({})),
            },
        ));
        for (index, id) in modules.iter().copied().enumerate() {
            edits.push(Edit::AddOperator(
                id,
                Operator {
                    path: format!("model.block_{index:02}.attention"),
                    kind: "Linear".into(),
                    parent: Some(root),
                    attributes: Cow::Owned(serde_json::json!({"stack":3,"width":256,"length":256})),
                },
            ));
            edits.push(Edit::AddPort(
                Port::Input(Inlet {
                    operator: id,
                    slot: 0,
                }),
                None,
            ));
            edits.push(Edit::AddPort(
                Port::Output(Outlet {
                    operator: id,
                    slot: 0,
                }),
                None,
            ));
        }
        for pair in modules.windows(2) {
            edits.push(Edit::Connect(Wire {
                from: Outlet {
                    operator: pair[0],
                    slot: 0,
                },
                to: Inlet {
                    operator: pair[1],
                    slot: 0,
                },
            }));
        }
        let mut document = GraphDocument::default();
        document
            .apply(Segment {
                base_revision: 0,
                edits,
            })
            .unwrap();
        let mut view = ModelView::from_document(document, false, LayoutCache::default());
        view.focus = Some(view.indices[&root]);
        view.selected = view.indices[&modules[0]];
        for (width, height) in [(40, 24), (100, 30), (120, 36)] {
            for id in modules {
                view.selected = view.indices[&id];
                view.explosion = 1;
                let first = view.scene(width, height, true, None);
                let first_slices: Vec<_> = first
                    .solids
                    .iter()
                    .filter(|solid| solid.key.operator == id)
                    .collect();
                let first_extent = first_slices
                    .iter()
                    .map(|solid| solid.bounds(view.camera)[1][1])
                    .fold(f64::NEG_INFINITY, f64::max)
                    - first_slices
                        .iter()
                        .map(|solid| solid.bounds(view.camera)[0][1])
                        .fold(f64::INFINITY, f64::min);
                view.explosion = 2;
                let scene = view.scene(width, height, true, None);
                let slices: Vec<_> = scene
                    .solids
                    .iter()
                    .filter(|solid| solid.key.operator == id)
                    .collect();
                assert_eq!(
                    slices.len(),
                    3,
                    "{id:?} keeps all slices at {width}x{height}"
                );
                let bounds = slices
                    .iter()
                    .map(|solid| solid.bounds(view.camera))
                    .collect::<Vec<_>>();
                let extent = bounds
                    .iter()
                    .map(|bound| bound[1][1])
                    .fold(f64::NEG_INFINITY, f64::max)
                    - bounds
                        .iter()
                        .map(|bound| bound[0][1])
                        .fold(f64::INFINITY, f64::min);
                if width >= 100 {
                    assert!(
                        extent > 3.0,
                        "{id:?} exploded extent at {width}x{height} is {extent}"
                    );
                    assert!(
                        extent > first_extent * 1.15,
                        "second explosion level {extent} should be visibly larger than first {first_extent} at {width}x{height}"
                    );
                }
                let layer = view.indices[&id];
                assert!(
                    (0..scene.width).any(|column| {
                        (SCENE_HEADER + scene.tooltip_rows
                            ..SCENE_HEADER + scene.tooltip_rows + scene.labels.len() - 1)
                            .any(|row| scene.layer_at(column, row) == Some(layer))
                    }),
                    "{id:?} remains individually hit-testable at {width}x{height}"
                );
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn exploded_fanout_keeps_parallel_branches_distinct_and_preserves_selection() {
        let root = OperatorId(1);
        let source = OperatorId(2);
        let sink = OperatorId(30);
        let branches: Vec<_> = (3..19).map(OperatorId).collect();
        let mut edits = Vec::new();
        for id in std::iter::once(root)
            .chain(std::iter::once(source))
            .chain(branches.iter().copied())
            .chain(std::iter::once(sink))
        {
            edits.push(Edit::AddOperator(
                id,
                Operator {
                    path: format!("model.branch_{}", id.0),
                    kind: "Linear".into(),
                    parent: (id != root).then_some(root),
                    attributes: Cow::Owned(serde_json::json!({"stack":1,"shape":[48,48]})),
                },
            ));
            if id != root {
                edits.push(Edit::AddPort(
                    Port::Output(Outlet {
                        operator: id,
                        slot: 0,
                    }),
                    None,
                ));
                let inputs = if id == sink { branches.len() } else { 1 };
                for slot in 0..inputs {
                    edits.push(Edit::AddPort(
                        Port::Input(Inlet {
                            operator: id,
                            slot: slot as u32,
                        }),
                        None,
                    ));
                }
            }
        }
        for (slot, &branch) in branches.iter().enumerate() {
            edits.push(Edit::Connect(Wire {
                from: Outlet {
                    operator: source,
                    slot: 0,
                },
                to: Inlet {
                    operator: branch,
                    slot: 0,
                },
            }));
            edits.push(Edit::Connect(Wire {
                from: Outlet {
                    operator: branch,
                    slot: 0,
                },
                to: Inlet {
                    operator: sink,
                    slot: slot as u32,
                },
            }));
        }
        let mut document = GraphDocument::default();
        document
            .apply(Segment {
                base_revision: 0,
                edits,
            })
            .unwrap();
        let mut view = ModelView::from_document(document, false, LayoutCache::default());
        view.focus = Some(view.indices[&root]);
        view.selected = view.indices[&branches[0]];
        let selected = Wire {
            from: Outlet {
                operator: source,
                slot: 0,
            },
            to: Inlet {
                operator: branches[0],
                slot: 0,
            },
        };
        view.selected_wire = Some(selected);
        view.selected_port = Some(Port::Input(selected.to));
        for width in [40, 100] {
            let mut first_positions = None;
            for level in [1, 2] {
                view.explosion = level;
                let scene = view.scene(width, 24, false, None);
                let graph = view.flow_layout().unwrap();
                let branch_stages: Vec<_> = graph
                    .stages(0..graph.count())
                    .iter()
                    .filter(|stage| branches.contains(&view.layers[stage.layer].id))
                    .collect();
                assert_eq!(branch_stages.len(), branches.len());
                assert!(
                    branch_stages
                        .iter()
                        .all(|stage| stage.rank == branch_stages[0].rank)
                );
                let mut bounds: Vec<_> = scene
                    .solids
                    .iter()
                    .filter(|solid| branches.contains(&solid.key.operator))
                    .map(|solid| (solid.key.operator, solid.bounds(view.camera)))
                    .collect();
                assert_eq!(
                    bounds.len(),
                    branches.len(),
                    "all branches should be visible"
                );
                bounds.sort_by(|left, right| left.1[0][0].total_cmp(&right.1[0][0]));
                assert!(
                    bounds
                        .iter()
                        .all(|(_, bound)| bound[0][0] >= 0.0 && bound[1][0] < width as f64)
                );
                let positions: BTreeMap<_, _> = scene
                    .solids
                    .iter()
                    .filter(|solid| branches.contains(&solid.key.operator))
                    .map(|solid| (solid.key.operator, solid.center[0]))
                    .collect();
                if let Some(first) = &first_positions {
                    assert_eq!(
                        &positions, first,
                        "changing expansion should not move lanes horizontally"
                    );
                } else {
                    first_positions = Some(positions);
                }
                for pair in bounds.windows(2) {
                    assert!(
                        pair[0].1[1][0] < pair[1].1[0][0],
                        "overlapping branches at width {width}, level {level}"
                    );
                }
                if width == 40 {
                    assert!(graph.vertical_lane_span() > 0.55);
                    assert!(
                        scene
                            .solids
                            .iter()
                            .filter(|solid| branches.contains(&solid.key.operator))
                            .all(|solid| solid.half[0] >= 0.55)
                    );
                    let mut old_bounds: Vec<_> = scene
                        .solids
                        .iter()
                        .filter(|solid| branches.contains(&solid.key.operator))
                        .map(|solid| {
                            let mut old = *solid;
                            old.half = view.layers[solid.layer]
                                .geometry((width / 2) as f64 * 0.58, 2, false, false)
                                .0;
                            old.bounds(view.camera)
                        })
                        .collect();
                    old_bounds.sort_by(|a, b| a[0][0].total_cmp(&b[0][0]));
                    assert!(
                        old_bounds
                            .windows(2)
                            .any(|pair| pair[0][1][0] >= pair[1][0][0]),
                        "the former fixed-width layout should overlap in this fixture"
                    );
                }
                assert!(scene.handoffs.iter().any(|route| route.wire == selected));
                let mut visible_hits = BTreeSet::new();
                let mut targets = BTreeMap::new();
                for row in SCENE_HEADER + scene.tooltip_rows
                    ..SCENE_HEADER + scene.tooltip_rows + scene.labels.len() - 1
                {
                    for column in 0..scene.width {
                        if let Some(index) = scene.layer_at(column, row) {
                            let id = view.layers[index].id;
                            visible_hits.insert(id);
                            let overlaps_wire = !scene.handoffs_at(column, row).is_empty();
                            let target = targets.entry(id).or_insert((column, row, overlaps_wire));
                            if overlaps_wire {
                                *target = (column, row, true);
                            }
                        }
                    }
                }
                assert!(
                    branches.iter().all(|id| visible_hits.contains(id)),
                    "all branches must be individually hit-testable at width {width}, level {level}: {visible_hits:?}"
                );
                assert_eq!(view.selected_wire, Some(selected));
                assert_eq!(view.selected_port, Some(Port::Input(selected.to)));
                let header = view.render(&scene, Locale::English, false, false);
                assert!(header.contains(&format!("exploded slices {level}/2")));
                assert!(
                    branches.iter().any(|id| targets[id].2),
                    "fixture must include module/wire ambiguity"
                );
                view.layout_edit = true;
                for id in &branches {
                    let (column, row, _) = targets[id];
                    interaction::click_for_test(&mut view, &scene, column, row);
                    assert_eq!(view.layers[view.selected].id, *id);
                }
                view.layout_edit = false;
                let (column, row, _) = branches
                    .iter()
                    .map(|id| targets[id])
                    .find(|target| target.2)
                    .unwrap();
                let wires = scene.handoffs_at(column, row);
                view.selected_wire = None;
                interaction::click_for_test(&mut view, &scene, column, row);
                assert!(
                    view.selected_wire.is_some_and(|wire| wires.contains(&wire)),
                    "normal mode must retain wire priority"
                );
                view.selected = view.indices[&branches[0]];
                view.selected_wire = Some(selected);
                view.selected_port = Some(Port::Input(selected.to));
            }
        }
        view.toggle_flat_flow();
        let flat = view.scene(40, 24, false, None);
        let mut flat_bounds: Vec<_> = flat
            .solids
            .iter()
            .filter(|solid| branches.contains(&solid.key.operator))
            .map(|solid| solid.bounds(view.camera))
            .collect();
        assert_eq!(flat_bounds.len(), branches.len());
        flat_bounds.sort_by(|a, b| a[0][0].total_cmp(&b[0][0]));
        assert!(
            flat_bounds
                .windows(2)
                .all(|pair| pair[0][1][0] < pair[1][0][0])
        );
        let mut flat_hits = BTreeSet::new();
        for row in SCENE_HEADER + flat.tooltip_rows
            ..SCENE_HEADER + flat.tooltip_rows + flat.labels.len() - 1
        {
            for column in 0..flat.width {
                if let Some(index) = flat.layer_at(column, row) {
                    flat_hits.insert(view.layers[index].id);
                }
            }
        }
        assert!(
            branches.iter().all(|id| flat_hits.contains(id)),
            "flat branches must be individually hit-testable: {flat_hits:?}"
        );
        assert_eq!(view.selected_wire, Some(selected));
        view.toggle_flat_flow();
        view.camera.orbit(0.35, 0.15);
        let rotated = view.scene(100, 30, false, None);
        assert!(rotated.handoffs.iter().any(|route| route.wire == selected));
        view.explosion = 0;
        assert_eq!(view.selected_wire, Some(selected));
        assert_eq!(view.selected_port, Some(Port::Input(selected.to)));
        let edits = branches
            .iter()
            .map(|&id| Edit::SetAttribute(id, "stack".into(), Some(json!(3))))
            .collect();
        view.apply_segment(Segment {
            base_revision: view.document.revision(),
            edits,
        })
        .unwrap();
        view.focus = Some(view.indices[&root]);
        view.selected = view.indices[&branches[0]];
        view.camera = Camera::default();
        for level in [1, 2] {
            for pixels in [false, true] {
                view.explosion = level;
                let stacked = view.scene(40, 24, pixels, None);
                let mut stacked_hits = BTreeSet::new();
                for row in SCENE_HEADER + stacked.tooltip_rows
                    ..SCENE_HEADER + stacked.tooltip_rows + stacked.labels.len() - 1
                {
                    for column in 0..stacked.width {
                        if let Some(index) = stacked.layer_at(column, row) {
                            stacked_hits.insert(view.layers[index].id);
                        }
                    }
                }
                assert!(
                    branches.iter().all(|id| stacked_hits.contains(id)),
                    "stacked branches must remain individually hit-testable at level {level}, pixels {pixels}: {stacked_hits:?}"
                );
            }
        }
        // Grow the same rank beyond the narrow-screen lane capacity. Both
        // halves must retain their own endpoints and cross-page connections.
        let extra: Vec<_> = (40..56).map(OperatorId).collect();
        let mut edits = Vec::new();
        for (offset, &id) in extra.iter().enumerate() {
            edits.push(Edit::AddOperator(
                id,
                Operator {
                    path: format!("model.branch_{}", id.0),
                    kind: "Linear".into(),
                    parent: Some(root),
                    attributes: Cow::Owned(json!({"stack":3,"shape":[48,48]})),
                },
            ));
            edits.push(Edit::AddPort(
                Port::Input(Inlet {
                    operator: id,
                    slot: 0,
                }),
                None,
            ));
            edits.push(Edit::AddPort(
                Port::Output(Outlet {
                    operator: id,
                    slot: 0,
                }),
                None,
            ));
            edits.push(Edit::AddPort(
                Port::Input(Inlet {
                    operator: sink,
                    slot: (offset + branches.len()) as u32,
                }),
                None,
            ));
            edits.push(Edit::Connect(Wire {
                from: Outlet {
                    operator: source,
                    slot: 0,
                },
                to: Inlet {
                    operator: id,
                    slot: 0,
                },
            }));
            edits.push(Edit::Connect(Wire {
                from: Outlet {
                    operator: id,
                    slot: 0,
                },
                to: Inlet {
                    operator: sink,
                    slot: (offset + branches.len()) as u32,
                },
            }));
        }
        view.apply_segment(Segment {
            base_revision: view.document.revision(),
            edits,
        })
        .unwrap();
        view.focus = Some(view.indices[&root]);
        view.explosion = 2;
        let graph = view.flow_layout().unwrap();
        let first_branch = graph
            .stages(0..graph.count())
            .iter()
            .position(|stage| {
                let id = view.layers[stage.layer].id;
                branches.contains(&id) || extra.contains(&id)
            })
            .unwrap();
        let first_layer = graph.layer(first_branch);
        let ids = |scene: &Scene, view: &ModelView<'_>| -> BTreeSet<OperatorId> {
            scene
                .solids
                .iter()
                .map(|solid| view.layers[solid.layer].id)
                .filter(|id| branches.contains(id) || extra.contains(id))
                .collect()
        };
        for (flat, level) in [(false, 1), (false, 2), (true, 0)] {
            if view.flat_flow != flat {
                view.toggle_flat_flow();
            }
            view.explosion = level;
            for pixels in [false, true] {
                view.selected = first_layer;
                let first = view.scene(40, 24, pixels, None);
                assert_eq!(first.page_start, first_branch);
                assert_eq!(first.page_size, 16);
                let first_frame = view.render(&first, Locale::English, false, pixels);
                assert!(
                    first_frame.contains("2–17/34"),
                    "page range must be visible: {first_frame}"
                );
                let first_ids = ids(&first, &view);
                assert_eq!(first_ids.len(), 16);
                view.page(&first, true);
                let second = view.scene(40, 24, pixels, None);
                assert_eq!(second.page_start, first.page_start + first.page_size);
                assert_eq!(second.page_size, 16);
                let second_frame = view.render(&second, Locale::English, false, pixels);
                assert!(
                    second_frame.contains("18–33/34"),
                    "page range must be visible: {second_frame}"
                );
                let second_ids = ids(&second, &view);
                assert_eq!(second_ids.len(), 16);
                assert!(first_ids.is_disjoint(&second_ids));
                assert_eq!(first_ids.union(&second_ids).count(), 32);
                view.layout_edit = true;
                for (scene, ids) in [(&first, &first_ids), (&second, &second_ids)] {
                    if pixels {
                        let covered =
                            pixels::visible_layers_for_test(scene, view.camera, view.selected);
                        assert!(
                            ids.iter().all(|id| covered.contains(&view.indices[id])),
                            "all page branches need visible pixel coverage"
                        );
                    }
                    for &id in ids {
                        assert!(
                            scene
                                .handoffs
                                .iter()
                                .any(|route| route.wire.from.operator == source
                                    && route.wire.to.operator == id
                                    && route.continuation == Some(Continuation::Incoming))
                        );
                        assert!(
                            scene
                                .handoffs
                                .iter()
                                .any(|route| route.wire.from.operator == id
                                    && route.wire.to.operator == sink
                                    && route.continuation == Some(Continuation::Outgoing))
                        );
                        let target = (SCENE_HEADER + scene.tooltip_rows
                            ..SCENE_HEADER + scene.tooltip_rows + scene.labels.len() - 1)
                            .find_map(|row| {
                                (0..scene.width)
                                    .find(|&column| {
                                        scene
                                            .layer_at(column, row)
                                            .is_some_and(|index| view.layers[index].id == id)
                                    })
                                    .map(|column| (column, row))
                            })
                            .unwrap_or_else(|| panic!("missing branch hit target: {id:?}"));
                        interaction::click_for_test(&mut view, scene, target.0, target.1);
                        assert_eq!(view.layers[view.selected].id, id);
                    }
                }
                view.layout_edit = false;
                let retained = Wire {
                    from: Outlet {
                        operator: source,
                        slot: 0,
                    },
                    to: Inlet {
                        operator: *second_ids.first().unwrap(),
                        slot: 0,
                    },
                };
                view.selected_wire = Some(retained);
                view.page(&second, false);
                let back = view.scene(40, 24, pixels, None);
                assert_eq!(back.page_start, first.page_start);
                assert_eq!(view.selected_wire, Some(retained));
                view.page(&back, true);
                let returned = view.scene(40, 24, pixels, None);
                assert!(returned.handoffs.iter().any(|route| route.wire == retained));
                assert_eq!(view.selected_wire, Some(retained));
            }
        }
        if view.flat_flow {
            view.toggle_flat_flow();
        }
        view.explosion = 2;
        view.selected = first_layer;
        let mut narrow_ids = BTreeSet::new();
        for page_number in 0..4 {
            let scene = view.scene(20, 24, false, None);
            assert_eq!(scene.page_size, 8);
            let frame = view.render(&scene, Locale::English, false, false);
            let expected_range = format!(
                "{}–{}/34",
                scene.page_start + 1,
                scene.page_start + scene.page_size
            );
            assert!(
                frame.contains(&expected_range) && frame.contains("x2/2"),
                "compact status must expose page and expansion: {frame}"
            );
            let page_ids = ids(&scene, &view);
            assert_eq!(page_ids.len(), 8);
            for &id in &page_ids {
                assert!(
                    (SCENE_HEADER + scene.tooltip_rows
                        ..SCENE_HEADER + scene.tooltip_rows + scene.labels.len() - 1)
                        .any(|row| (0..scene.width).any(|column| scene
                            .layer_at(column, row)
                            .is_some_and(|index| view.layers[index].id == id))),
                    "missing 20-column branch hit target on page {page_number}: {id:?}"
                );
            }
            assert!(narrow_ids.is_disjoint(&page_ids));
            narrow_ids.extend(page_ids);
            if page_number < 3 {
                view.page(&scene, true);
            }
        }
        assert_eq!(narrow_ids.len(), 32);
        let last = OperatorId(60);
        let last_input = Inlet {
            operator: last,
            slot: 0,
        };
        let last_output = Outlet {
            operator: last,
            slot: 0,
        };
        view.apply_segment(Segment {
            base_revision: view.document.revision(),
            edits: vec![
                Edit::AddOperator(
                    last,
                    Operator {
                        path: "model.last_branch".into(),
                        kind: "Linear".into(),
                        parent: Some(root),
                        attributes: Cow::Owned(json!({"stack":3})),
                    },
                ),
                Edit::AddPort(Port::Input(last_input), None),
                Edit::AddPort(Port::Output(last_output), None),
                Edit::AddPort(
                    Port::Input(Inlet {
                        operator: sink,
                        slot: 32,
                    }),
                    None,
                ),
                Edit::Connect(Wire {
                    from: Outlet {
                        operator: source,
                        slot: 0,
                    },
                    to: last_input,
                }),
                Edit::Connect(Wire {
                    from: last_output,
                    to: Inlet {
                        operator: sink,
                        slot: 32,
                    },
                }),
            ],
        })
        .unwrap();
        if view.flat_flow {
            view.toggle_flat_flow();
        }
        view.focus = Some(view.indices[&root]);
        let graph = view.flow_layout().unwrap();
        let branch_rank = graph
            .stages(0..graph.count())
            .iter()
            .find(|stage| view.layers[stage.layer].id == last)
            .unwrap()
            .rank;
        let last_index = graph
            .stages(0..graph.count())
            .iter()
            .rposition(|stage| stage.rank == branch_rank)
            .unwrap();
        view.selected = graph.layer(last_index);
        view.explosion = 2;
        let tail = view.scene(40, 24, false, None);
        assert_eq!(tail.page_size, 1);
        assert!(
            view.render(&tail, Locale::English, false, false)
                .contains("34–34/35")
        );
        assert!(
            tail.handoffs
                .iter()
                .any(|route| route.continuation == Some(Continuation::Incoming))
        );
        assert!(
            tail.handoffs
                .iter()
                .any(|route| route.continuation == Some(Continuation::Outgoing))
        );
        view.page(&tail, false);
        assert_eq!(view.scene(40, 24, false, None).page_size, 16);
    }

    #[cfg(unix)]
    #[test]
    fn flat_flow_preserves_wires_selection_and_the_spatial_camera() {
        let mut view = ModelView::new(None);
        assert!(view.flow_layout().is_some());
        view.camera.orbit(0.31, -0.12);
        let camera = view.camera;
        let selected = view.layers[view.selected].id;
        let spatial = view.scene(100, 30, false, None);
        let spatial_wires: Vec<_> = spatial.handoffs.iter().map(|route| route.wire).collect();
        assert!(!spatial_wires.is_empty());
        view.viewport = Viewport {
            scale: 1.2,
            offset: [4.0, 2.0],
        };
        let spatial_viewport = view.viewport;
        view.toggle_flat_flow();
        assert!(view.viewport == Viewport::default());
        let flat = view.scene(100, 30, false, None);
        let flat_wires: Vec<_> = flat.handoffs.iter().map(|route| route.wire).collect();
        assert_eq!(flat_wires.len(), spatial_wires.len());
        assert!(flat_wires.iter().all(|wire| spatial_wires.contains(wire)));
        assert_eq!(view.layers[view.selected].id, selected);
        assert!(flat.solids.iter().all(|solid| solid.half[2] < 0.1));
        assert!(
            view.render(&flat, Locale::English, false, false)
                .contains("z 2D/3D"),
            "{}",
            view.render(&flat, Locale::English, false, false)
        );
        view.viewport = Viewport {
            scale: 1.4,
            offset: [-2.0, 3.0],
        };
        let flat_viewport = view.viewport;
        view.toggle_flat_flow();
        assert!(view.camera == camera);
        assert!(view.viewport == spatial_viewport);
        assert_eq!(view.layers[view.selected].id, selected);
        view.toggle_flat_flow();
        assert!(view.viewport == flat_viewport);
        let bookmark = view.bookmark();
        view.viewport = Viewport::default();
        view.restore_view(bookmark);
        assert!(view.viewport == flat_viewport);
        view.toggle_flat_flow();
        assert!(view.viewport == spatial_viewport);
    }

    #[cfg(unix)]
    #[test]
    fn numeric_graph_presets_select_exact_mode_and_preserve_graph_identity() {
        let mut view = ModelView::new(None);
        let selected = view.layers[view.selected].id;
        let selected_wire = view.document.wires().next().unwrap();
        view.selected_wire = Some(selected_wire);

        assert!(view.select_graph_preset(4));
        assert_eq!(view.explosion, 2);
        assert!(!view.flat_flow);
        assert!(view.select_graph_preset(2));
        assert!(view.flat_flow);
        assert_eq!(view.layers[view.selected].id, selected);
        assert_eq!(view.selected_wire, Some(selected_wire));

        assert!(view.select_graph_preset(1));
        assert!(!view.flat_flow);
        assert_eq!(view.explosion, 0);
        assert_eq!(view.layers[view.selected].id, selected);
        assert_eq!(view.selected_wire, Some(selected_wire));
        assert!(view.select_graph_preset(3));
        assert_eq!(view.explosion, 1);
        assert!(view.select_graph_preset(4));
        assert_eq!(view.explosion, 2);
        assert!(!view.select_graph_preset(0));

        let wide = view.scene(100, 30, false, None);
        let wide_frame = view.render(&wide, Locale::English, false, false);
        assert!(wide_frame.contains("1 3D · 2 Flow · 3 x1 · 4 x2"));
        let narrow = view.scene(40, 24, false, None);
        let narrow_frame = view.render(&narrow, Locale::Simplified, false, false);
        assert!(
            narrow_frame.contains("1三维 · 2二维 · 3x1 · 4x2"),
            "{narrow_frame}"
        );

        let mut hierarchy = ModelView::new(None);
        hierarchy.hierarchy = true;
        assert!(!hierarchy.select_graph_preset(2));
        assert!(!hierarchy.flat_flow);
        let hierarchy_scene = hierarchy.scene(40, 24, false, Some(0.0));
        let hierarchy_frame = hierarchy.render(&hierarchy_scene, Locale::Simplified, false, false);
        assert!(
            hierarchy_frame.contains("1三维 · 3x1 · 4x2"),
            "{hierarchy_frame}"
        );
        assert!(
            !hierarchy_frame.contains("1三维 · 2二维"),
            "{hierarchy_frame}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn flat_flow_separates_parallel_port_routes_stably() {
        let offline = offline::Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts);
        let mut view = offline.view();
        let spatial = view.scene(80, 30, false, None);
        let spatial_wires: Vec<_> = spatial.handoffs.iter().map(|route| route.wire).collect();
        let spatial_crossings = spatial
            .handoffs
            .iter()
            .map(Handoff::crossing_marks)
            .sum::<usize>()
            / 2;
        view.toggle_flat_flow();
        let first = view.scene(80, 30, false, None);
        let flow_crossings = first
            .handoffs
            .iter()
            .map(Handoff::crossing_marks)
            .sum::<usize>()
            / 2;
        assert_eq!((spatial_crossings, flow_crossings), (2, 3));
        assert_eq!(
            first
                .handoffs
                .iter()
                .map(|route| route.wire)
                .collect::<Vec<_>>(),
            spatial_wires
        );
        let mut pairs: BTreeMap<[usize; 2], Vec<_>> = BTreeMap::new();
        for route in &first.handoffs {
            pairs.entry(route.layers).or_default().push(route);
        }
        let parallel = pairs
            .values()
            .find(|routes| routes.len() >= 2)
            .expect("crossed ports expose parallel module edges");
        assert_ne!(parallel[0].wire, parallel[1].wire);
        assert_ne!(parallel[0].bend, parallel[1].bend);
        let saved: Vec<_> = parallel
            .iter()
            .map(|route| (route.wire, route.bend))
            .collect();
        let second = view.scene(80, 30, false, None);
        for (wire, bend) in saved {
            assert_eq!(
                second
                    .handoffs
                    .iter()
                    .find(|route| route.wire == wire)
                    .unwrap()
                    .bend,
                bend
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn narrow_flat_flow_arranges_ranks_vertically_and_routes_ports() {
        let mut view = ModelView::new(None);
        view.toggle_flat_flow();
        let scene = view.scene(40, 24, false, None);
        assert!(scene.page_size >= 2);
        let route = scene
            .handoffs
            .iter()
            .find(|route| route.continuation.is_none())
            .expect("two ranks are visible");
        assert!(route.vertical);
        assert!(route.to[1] > route.from[1]);
        let middle = [route.from[0], (route.from[1] * 0.75 + route.to[1] * 0.25)];
        assert!(route.distance_at(middle).is_some());
        view.page(&scene, true);
        let next = view.scene(40, 24, false, None);
        assert!(next.page_start > scene.page_start);
        view.page(&next, false);
        let restored = view.scene(40, 24, false, None);
        assert_eq!(restored.page_start, scene.page_start);
    }

    #[test]
    fn inspected_structure_can_focus_observed_nested_module_and_return() {
        let modules = vec![
            json!({"path": "model", "type": "Model"}),
            json!({"path": "model.encoder", "type": "Encoder"}),
            json!({"path": "model.encoder.proj", "type": "Linear"}),
            json!({"path": "model.encoder.other", "type": "Linear"}),
        ];
        let mut view = ModelView::new(Some(&modules));
        assert!(view.focus_path("model.encoder.proj"));
        assert_eq!(view.layers[view.selected].path, "model.encoder.proj");
        assert_eq!(view.layers[view.focus.unwrap()].path, "model.encoder");
        assert!(detailed(&view, 80).contains("model.encoder.proj"));
        assert!(view.zoom_out());
        assert_eq!(view.layers[view.selected].path, "model.encoder");
        assert!(!view.focus_path("model.missing"));
    }

    #[test]
    fn located_module_shows_observation_and_snapshot_with_separate_sources() {
        let modules = vec![
            json!({"path": "model.head", "type": "Linear", "editable_params": {"in_features": 32}}),
        ];
        let mut view = ModelView::new(Some(&modules));
        assert!(view.focus_path("model.head"));
        view.live_evidence = Some(LiveEvidence {
            path: "model.head".into(),
            ambiguous_registration: false,
            last_observed: true,
            from_calls: false,
            reused: false,
            #[cfg(unix)]
            selected_input: None,
            #[cfg(unix)]
            show_input: false,
            #[cfg(unix)]
            graph_target_unresolved: false,
            #[cfg(unix)]
            projection: None,
            input: Shape::parse(&json!([2, 48])),
            invocation: Some(2),
            invocation_id: None,
            expected: Some(Shape::Features(32)),
            raw_report: String::new(),
            #[cfg(unix)]
            route_cache: std::cell::OnceCell::new(),
            #[cfg(unix)]
            inputs: Vec::new(),
            #[cfg(unix)]
            diagnostics: Vec::new(),
        });
        let frame = detailed(&view, 80);
        assert!(frame.contains("Observed call #2 · input 0 2×48"), "{frame}");
        assert!(frame.contains("Snapshot input features 32"), "{frame}");
        view.live_evidence.as_mut().unwrap().ambiguous_registration = true;
        let shared = detailed(&view, 40);
        assert!(shared.contains("Shared call · path unknown"), "{shared}");
        view.live_evidence.as_mut().unwrap().reused = true;
        let cached_frame = detailed(&view, 80);
        assert!(
            cached_frame.contains("Reused · no new forward"),
            "{cached_frame}"
        );
        #[cfg(unix)]
        {
            let original = view.bookmark();
            view.live_axes_zoom = Some(8);
            let comparison = view.live_axis_view().unwrap();
            let scene = Scene::for_axes(
                view.appearance(),
                80,
                21,
                comparison,
                false,
                Locale::English,
            );
            let axes = view.render(&scene, Locale::English, false, false);
            assert!(
                axes.contains("Reused · no new forward · input  2×48"),
                "{axes}"
            );
            assert!(axes.contains("Inspected snapshot  …×32"), "{axes}");
            view.live_axes_zoom = None;
            assert!(view.bookmark() == original);
        }
    }

    fn detailed(view: &ModelView<'_>, width: usize) -> String {
        let mut scene = view.scene(width, 24, false, None);
        scene.refine(view.camera, usize::MAX);
        view.render(&scene, Locale::English, false, false)
    }

    #[test]
    fn sliced_modules_change_with_camera_and_keep_layer_metadata() {
        let mut view = ModelView::new(None);
        view.hierarchy = true;
        let first = detailed(&view, 80);
        assert!(first.contains("DEMO"));
        assert!(first.contains("encoder.embedding"));
        assert!(first.contains("32000"));
        assert!(
            first
                .chars()
                .any(|ch| ('\u{2801}'..='\u{28ff}').contains(&ch))
        );
        let scene = view.scene(80, 24, false, None);
        let slices: Vec<_> = scene
            .solids
            .iter()
            .filter(|solid| solid.layer == 0)
            .collect();
        assert_eq!(slices.len(), 4);
        assert!(slices.windows(2).all(|pair| {
            pair[0].elevation + pair[0].half[1] < pair[1].elevation - pair[1].half[1]
        }));
        assert!(first.lines().count() <= 24);
        assert!(first.lines().all(|line| text::display_width(line) <= 80));
        view.camera.orbit(0.7, -0.4);
        assert_ne!(first, detailed(&view, 80));
        view.hover = Some(1);
        let second = detailed(&view, 80);
        assert!(second.contains("╭─ encoder.attention"));
        assert!(second.contains("heads: 8"));
    }

    #[cfg(unix)]
    #[test]
    fn layer_hover_shows_interface_and_counts_while_a_wire_is_selected() {
        let mut view = ModelView::new(None);
        view.selected_wire = Some(wire(1, 4, 0));
        view.hover = Some(2);
        for pixels in [false, true] {
            for height in [17, 24] {
                let scene = view.scene(40, height, pixels, None);
                let hovered = view.render(&scene, Locale::English, false, pixels);
                if scene.tooltip_rows > 0 {
                    assert!(hovered.contains("stack 4 · params 3159040"), "{hovered}");
                    assert!(
                        hovered.contains("IN 2×32×256  →  OUT 2×32×256"),
                        "{hovered}"
                    );
                }
                assert!(hovered.contains("encoder.attention → merge"), "{hovered}");
                assert!(hovered.lines().all(|line| text::display_width(line) <= 40));
            }
        }
        assert_eq!(view.selected_wire, Some(wire(1, 4, 0)));
        view.hover = None;
        let scene = view.scene(40, 24, false, None);
        let wire_details = view.render(&scene, Locale::English, false, false);
        assert!(wire_details.contains("encoder.attention → merge"));
    }

    #[cfg(unix)]
    #[test]
    fn hover_preview_never_replaces_the_pinned_wire_evidence() {
        let mut view = offline::Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts).view();
        let target = view
            .layers
            .iter()
            .find(|layer| layer.path == "net.merge")
            .unwrap()
            .id;
        let first = view
            .document
            .wires()
            .find(|wire| wire.to.operator == target && wire.to.slot == 0)
            .unwrap();
        let second = view
            .document
            .wires()
            .find(|wire| wire.to.operator == target && wire.to.slot == 1)
            .unwrap();
        view.selected_wire = Some(first);
        view.hover_handoff = Some(second);
        for pixels in [false, true] {
            let scene = view.scene(80, 30, pixels, None);
            let frame = view.render(&scene, Locale::English, false, pixels);
            assert!(
                frame.contains("model.encoder[0] → net.merge[1]"),
                "hover preview: {frame}"
            );
            assert!(
                frame.contains("model.encoder[1] → net.merge[0]"),
                "pinned evidence: {frame}"
            );
            assert!(
                frame.rfind("model.encoder[1] → net.merge[0]").unwrap()
                    > frame.find("model.encoder[0] → net.merge[1]").unwrap()
            );
            assert_eq!(view.selected_wire, Some(first));
        }
    }

    #[cfg(unix)]
    #[test]
    fn curves_follow_floating_stacks_without_moving_captions() {
        let mut view = ModelView::new(None);
        view.hierarchy = true;
        let first = view.scene(120, 30, true, Some(0.0));
        let next = view.scene(120, 30, true, Some(0.7));
        assert_eq!(first.captions, next.captions);
        assert!(
            first
                .captions
                .iter()
                .flatten()
                .any(|(_, label)| label.contains("embedding"))
        );
        for (before, after) in first.solids.iter().zip(&next.solids) {
            assert_eq!(before.center[0], after.center[0]);
            assert!(
                (before.center[1] - after.center[1]).abs() <= view.layers[before.layer].hop_range
            );
            assert_eq!(before.elevation, after.elevation);
        }
        let relation = first.links[0].layers;
        assert_eq!(relation, next.links[0].layers);
        for (end, layer) in [(0, relation[0]), (3, relation[1])] {
            let get_y = |scene: &Scene| {
                scene
                    .solids
                    .iter()
                    .find(|solid| solid.layer == layer)
                    .unwrap()
                    .center[1]
            };
            let delta = next.links[0].points[end][1] - first.links[0].points[end][1];
            assert!(delta.abs() > 0.01);
            assert!((delta - (get_y(&next) - get_y(&first))).abs() < 1e-8);
        }
    }

    #[test]
    fn spatial_focus_mark_breathes_while_flat_flow_stays_static() {
        let mut view = ModelView::new(None);
        let smallest = view.scene(100, 30, false, Some(0.0));
        let largest = view.scene(100, 30, true, Some(1.4));

        let smallest = smallest
            .focus_mark
            .expect("selected spatial module is marked");
        let largest = largest
            .focus_mark
            .expect("pixel scene also marks selection");
        assert!(smallest.ink == LinkInk::Focus);
        assert!(largest.ink == LinkInk::FocusBright);
        assert!(largest.radius > smallest.radius * 1.15);

        view.focus_pulse = 1.0;
        let pulse = view
            .scene(100, 30, false, Some(0.0))
            .focus_mark
            .expect("a new focus briefly brightens the selected module");
        assert!(pulse.ink == LinkInk::FocusBright);
        assert!(pulse.radius > smallest.radius * 1.15);

        view.flat_flow = true;
        assert!(view.scene(100, 30, false, Some(1.4)).focus_mark.is_none());
    }

    #[test]
    fn capture_focus_breathe_review_samples_when_requested() {
        let Some(root) = std::env::var_os("MODEL_MAVEN_FOCUS_CAPTURE_DIR") else {
            return;
        };
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        let view = ModelView::new(None);
        let make_scene = |factor: f64, ink: LinkInk| {
            let mut scene = view.scene(100, 30, true, None);
            let solid = scene
                .solids
                .iter()
                .find(|solid| solid.layer == view.selected)
                .unwrap();
            let bounds = solid.bounds(view.camera);
            let point = [
                (bounds[0][0] + bounds[1][0]) * 0.5,
                (bounds[0][1] + bounds[1][1]) * 0.5,
            ];
            let diameter = (bounds[1][0] - bounds[0][0]).max(bounds[1][1] - bounds[0][1]);
            let mark = PortMark {
                point,
                radius: (diameter * 0.5 + 0.65) * factor,
                ink,
            };
            scene.canvas.mark(&mark.points(), mark.ink);
            scene.focus_mark = Some(mark);
            scene
        };
        let write_frame = |name: &str, scene: &Scene| {
            let cells = view.render(scene, Locale::English, false, false);
            let prefix = if name == "feedback" {
                "focus-feedback"
            } else {
                "focus-breathe"
            };
            std::fs::write(root.join(format!("{prefix}-{name}-cells.txt")), &cells).unwrap();
            let png = pixels::png_for_test(scene, view.camera, view.selected);
            std::fs::write(root.join(format!("{prefix}-{name}.png")), &png).unwrap();
            (cells, png)
        };
        let rest = make_scene(1.12, LinkInk::Focus);
        let peak = make_scene(1.4, LinkInk::FocusBright);
        let rest = write_frame("rest", &rest);
        let peak = write_frame("peak", &peak);
        assert_ne!(rest.0, peak.0, "Braille focus outline should breathe");
        assert_ne!(rest.1, peak.1, "pixel focus outline should breathe");

        let mut feedback_view = ModelView::new(None);
        feedback_view.focus_pulse = 1.0;
        let feedback = feedback_view.scene(100, 30, true, Some(0.0));
        let feedback_mark = feedback.focus_mark.unwrap();
        let stable_mark = view.scene(100, 30, true, Some(0.0)).focus_mark.unwrap();
        assert!(feedback_mark.radius > stable_mark.radius);
        let feedback = write_frame("feedback", &feedback);
        assert_ne!(
            rest.0, feedback.0,
            "selection feedback should enlarge the Braille mark"
        );
        assert_ne!(
            rest.1, feedback.1,
            "selection feedback should enlarge the pixel mark"
        );
    }

    #[test]
    fn parameter_counts_set_hop_height_in_both_renderers() {
        let modules = [
            json!({"path":"light", "parameter_count":1_000, "parameters":100_000_000}),
            json!({"path":"medium", "parameters":100_000}),
            json!({"path":"heavy", "parameter_count":10_000_000}),
            json!({"path":"unknown", "parameter_count":-1, "weight_shape":[1024, 1024]}),
        ];
        let mut view = ModelView::new(Some(&modules));
        for pixels in [false, true] {
            let rest = view.scene(90, 30, pixels, None);
            let mut highest = [0.0_f64; 4];
            for tick in 0..48 {
                let time = f64::from(tick) * HOP_PERIOD / 48.0;
                let frame = view.scene(90, 30, pixels, Some(time));
                for (baseline, solid) in rest.solids.iter().zip(frame.solids) {
                    assert_eq!(baseline.center[0], solid.center[0]);
                    let lift = baseline.center[1] - solid.center[1];
                    assert!(lift >= 0.0);
                    let peak = &mut highest[solid.layer];
                    *peak = peak.max(lift);
                }
            }
            let [light, medium, heavy, unknown] = highest;
            assert!(light > medium && medium > heavy);
            assert!(light / heavy > 3.5);
            assert!((1.08..=1.10 + 1e-8).contains(&unknown), "{unknown}");
        }
        view.selected = 3;
        view.hover = Some(3);
        let scene = view.scene(90, 30, false, None);
        let details = view.render(&scene, Locale::English, false, false);
        assert!(details.contains("stack ? · params ?"));
        assert!(details.contains("Parameter count unknown"));
        assert!(details.contains("weight_shape: [1024,1024]"));
        assert!(!details.contains("parameter_count: -1"));
    }

    #[cfg(unix)]
    #[test]
    fn tensor_ports_keep_their_position_on_each_hopping_layer() {
        let view = ModelView::new(None);
        let rest = view.scene(80, 24, true, None);
        let frame = view.scene(80, 24, true, Some(1.4));
        for (before, after) in rest.handoffs.iter().zip(&frame.handoffs) {
            for (&layer, (a, b)) in before
                .layers
                .iter()
                .zip([(before.from, after.from), (before.to, after.to)])
            {
                let layer_y = |scene: &Scene| {
                    scene
                        .solids
                        .iter()
                        .find(|solid| solid.layer == layer)
                        .unwrap()
                        .center[1]
                };
                assert!((a[0] - b[0]).abs() < 1e-8);
                assert!(((b[1] - a[1]) - (layer_y(&frame) - layer_y(&rest))).abs() < 1e-8);
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn editing_one_of_two_merge_inputs_updates_only_its_wire() {
        let mut view = ModelView::new(None);
        let changed = wire(2, 4, 1);
        let neighbor = wire(1, 4, 0);
        let before = view.scene(100, 28, true, None);
        let original_layout = view.flow_layout().unwrap() as *const FlowLayout;
        let baseline = view.document.revision();
        view.hover_handoff = Some(changed);
        let change = view
            .apply_segment(Segment {
                base_revision: baseline,
                edits: vec![Edit::SetShape(
                    Port::Input(changed.to),
                    Some(Shape::Dimensions(vec![Some(2), Some(32), Some(128)])),
                )],
            })
            .unwrap();
        assert_eq!(change.inlets, BTreeSet::from([changed.to]));
        assert_eq!(
            original_layout,
            view.flow_layout().unwrap() as *const FlowLayout
        );
        let after = view.scene(100, 28, true, None);
        assert_eq!(before.captions, after.captions);
        for edge in &after.handoffs {
            let old = before
                .handoffs
                .iter()
                .find(|old| old.wire == edge.wire)
                .unwrap();
            assert_eq!([old.from, old.to], [edge.from, edge.to]);
            let expected = if edge.wire == changed {
                Alignment::Mismatch
            } else {
                Alignment::Match
            };
            assert_eq!(edge.alignment, expected);
        }
        let frame = view.render(&after, Locale::English, true, true);
        assert!(frame.contains("encoder.blocks[0] → merge[1]"));
        assert!(frame.contains("2×32×256 → 2×32×128"));
        assert!(after.handoffs.iter().any(|edge| edge.wire == neighbor));
    }

    #[cfg(unix)]
    #[test]
    fn parallel_wires_keep_distinct_sockets_when_the_model_rotates() {
        let mut view = ModelView::new(None);
        let first = wire(2, 4, 1);
        let second = Wire {
            from: Outlet {
                operator: first.from.operator,
                slot: 1,
            },
            to: Inlet {
                operator: first.to.operator,
                slot: 2,
            },
        };
        view.apply_segment(Segment {
            base_revision: view.document.revision(),
            edits: vec![
                Edit::AddPort(Port::Output(second.from), Some(Shape::Features(32))),
                Edit::AddPort(Port::Input(second.to), Some(Shape::Features(32))),
                Edit::Connect(second),
            ],
        })
        .unwrap();
        for delta in [0.0, 0.7, 1.5] {
            view.camera.orbit(delta, 0.06);
            let scene = view.scene(100, 30, false, None);
            let route = |wire| {
                scene
                    .handoffs
                    .iter()
                    .find(|route| route.wire == wire)
                    .unwrap()
            };
            assert_ne!(route(first).from, route(second).from);
            assert_ne!(route(first).to, route(second).to);
        }
        let neighbor = wire(1, 4, 0);
        let inspect = |scene: &Scene, wire: Wire| {
            let route = scene
                .handoffs
                .iter()
                .find(|route| route.wire == wire)
                .unwrap();
            [route.from, route.to]
        };
        let before = view.scene(100, 30, true, None);
        let reassigned = Wire {
            from: neighbor.from,
            ..second
        };
        view.apply_segment(Segment {
            base_revision: view.document.revision(),
            edits: vec![Edit::Disconnect(second), Edit::Connect(reassigned)],
        })
        .unwrap();
        let after = view.scene(100, 30, true, None);
        for unaffected in [first, neighbor] {
            assert_eq!(inspect(&before, unaffected), inspect(&after, unaffected));
        }
        assert_eq!(inspect(&before, second)[1], inspect(&after, reassigned)[1]);
        assert_eq!(inspect(&after, neighbor)[0], inspect(&after, reassigned)[0]);
    }

    #[test]
    fn topology_commit_keeps_selected_operator_when_render_indices_shift() {
        let mut view = ModelView::new(None);
        let source = OperatorId(0);
        let selected = OperatorId(5);
        view.selected = view.indices[&selected];
        view.camera.orbit(0.4, 0.1);
        let camera = view.camera;
        let old_index = view.selected;
        view.apply_segment(Segment {
            base_revision: view.document.revision(),
            edits: vec![
                Edit::Disconnect(wire(0, 1, 0)),
                Edit::Disconnect(wire(0, 2, 0)),
                Edit::RemovePort(Port::Input(Inlet {
                    operator: source,
                    slot: 0,
                })),
                Edit::RemovePort(Port::Output(Outlet {
                    operator: source,
                    slot: 0,
                })),
                Edit::RemoveOperator(source),
            ],
        })
        .unwrap();
        assert_ne!(old_index, view.selected);
        assert_eq!(view.layers[view.selected].id, selected);
        assert!(view.camera == camera);
        let scene = view.scene(100, 28, false, None);
        assert!(
            scene
                .solids
                .iter()
                .any(|slice| slice.key.operator == selected)
        );
        assert!(
            !scene
                .solids
                .iter()
                .any(|slice| slice.key.operator == source)
        );
    }

    #[test]
    fn large_demo_reuses_layout_of_an_unaffected_scope() {
        let mut records = vec![
            json!({"path": "short", "type": "Sequential"}),
            json!({"path": "long", "type": "Sequential"}),
        ];
        for (path, count) in [("short", 4), ("long", 2_048)] {
            records.extend((0..count).map(|index| {
                json!({
                    "path": format!("{path}.{index}"),
                    "type": "Linear",
                    "weight_shape": [32, 32],
                })
            }));
        }
        let mut view = ModelView::new(Some(&records));
        view.zoom(1);
        let long_layout = |view: &ModelView<'_>| {
            let layout = &view.scope_layouts[1].as_ref().unwrap().layout;
            layout.stages(0..1).as_ptr()
        };
        let unchanged_layout = long_layout(&view);
        let last = OperatorId((records.len() - 1) as u64);
        let shape_start = std::time::Instant::now();
        let change = view
            .apply_segment(Segment {
                base_revision: view.document.revision(),
                edits: vec![
                    Edit::SetShape(
                        Port::Input(Inlet {
                            operator: last,
                            slot: 0,
                        }),
                        Some(Shape::Features(48)),
                    ),
                    Edit::SetAttribute(last, "parameter_count".into(), Some(json!(800))),
                ],
            })
            .unwrap();
        assert!(!change.topology_changed);
        assert_eq!(change.operators, BTreeSet::from([last]));
        assert_eq!(
            change.inlets,
            BTreeSet::from([Inlet {
                operator: last,
                slot: 0
            }])
        );
        assert_eq!(long_layout(&view), unchanged_layout);
        let changed = view.indices[&last];
        view.selected = changed;
        let mut frame = view.scene(100, 32, false, None);
        frame.refine(view.camera, usize::MAX);
        assert!(
            view.render(&frame, Locale::English, false, false)
                .contains("parameter_count: 800")
        );
        let shape_time = shape_start.elapsed();
        assert!(frame.solids.len() <= 7 * 9);

        let rewire_start = std::time::Instant::now();
        view.apply_segment(Segment {
            base_revision: view.document.revision(),
            edits: vec![
                Edit::Disconnect(wire(3, 4, 0)),
                Edit::Connect(wire(2, 4, 0)),
            ],
        })
        .unwrap();
        assert_eq!(long_layout(&view), unchanged_layout);
        assert_eq!(view.selected, changed);
        let mut frame = view.scene(100, 32, false, None);
        frame.refine(view.camera, usize::MAX);
        let _ = view.render(&frame, Locale::English, false, false);
        eprintln!(
            "2,052 demo operators — shape and attribute to frame {shape_time:?}, rewire to frame {:?}",
            rewire_start.elapsed()
        );
    }

    #[test]
    fn report_without_dimensions_keeps_metadata_unknown() {
        let modules = vec![json!({"path":"net.head", "type":"Linear"})];
        let view = ModelView::new(Some(&modules));
        let frame = detailed(&view, 40);
        assert!(frame.contains("INSPECT"));
        assert!(frame.contains("net.head"));
        assert!(frame.contains("Dimensions and parameter count"));
        assert!(!frame.contains("32000"));
        assert!(frame.lines().count() <= 24);
        assert!(frame.lines().all(|line| text::display_width(line) <= 40));
    }

    #[cfg(unix)]
    #[test]
    fn shared_sequential_aliases_keep_registered_positions_and_instance_identity() {
        let modules = vec![
            json!({"path":"model", "type":"Sequential", "sequential_children":["first", "middle", "again"]}),
            json!({"path":"model.first", "type":"Linear", "instance_id":"module-2", "aliases":["model.first", "model.again"]}),
            json!({"path":"model.middle", "type":"ReLU", "instance_id":"module-3", "aliases":["model.middle"]}),
        ];
        let mut view = ModelView::new(Some(&modules));
        let first = view
            .layers
            .iter()
            .position(|layer| layer.path == "model.first")
            .unwrap();
        let middle = view
            .layers
            .iter()
            .position(|layer| layer.path == "model.middle")
            .unwrap();
        let again = view
            .layers
            .iter()
            .position(|layer| layer.path == "model.again")
            .unwrap();
        assert_eq!(
            view.layers[first].data["instance_id"],
            view.layers[again].data["instance_id"]
        );
        assert_eq!(
            view.document.source(Inlet {
                operator: view.layers[middle].id,
                slot: 0
            }),
            Some(Outlet {
                operator: view.layers[first].id,
                slot: 0
            })
        );
        assert_eq!(
            view.document.source(Inlet {
                operator: view.layers[again].id,
                slot: 0
            }),
            Some(Outlet {
                operator: view.layers[middle].id,
                slot: 0
            })
        );
        view.selected = again;
        let scene = view.scene(100, 24, false, None);
        let frame = view.render(&scene, Locale::English, false, false);
        assert!(frame.contains("Shared instance · CLI module-2"), "{frame}");

        let mut conflict = modules.clone();
        conflict.push(json!({"path":"model.other", "type":"Linear", "instance_id":"module-4", "aliases":["model.other", "model.again"]}));
        let view = ModelView::new(Some(&conflict));
        assert!(!view.layers.iter().any(|layer| layer.path == "model.again"));
        assert!(view.document.wires().next().is_none());
        let mut conflicting_existing_path = modules.clone();
        conflicting_existing_path.push(json!({"path":"model.again", "type":"Linear", "instance_id":"module-9", "aliases":["model.again"]}));
        let view = ModelView::new(Some(&conflicting_existing_path));
        assert!(view.document.wires().next().is_none());
        let mut missing_identity = modules;
        missing_identity[1]
            .as_object_mut()
            .unwrap()
            .remove("instance_id");
        assert!(
            !ModelView::new(Some(&missing_identity))
                .layers
                .iter()
                .any(|layer| layer.path == "model.again")
        );
    }

    #[cfg(unix)]
    #[test]
    fn shared_sequential_nested_aliases_expand_without_mixing_scopes() {
        let modules = vec![
            json!({"path":"model", "type":"Sequential", "sequential_children":["left", "right"]}),
            json!({"path":"model.left", "type":"Sequential", "instance_id":"module-2", "aliases":["model.left", "model.right"], "sequential_children":["0", "1"]}),
            json!({"path":"model.left.0", "type":"Linear", "instance_id":"module-3", "aliases":["model.left.0", "model.right.0"]}),
            json!({"path":"model.left.1", "type":"ReLU", "instance_id":"module-4", "aliases":["model.left.1", "model.right.1"]}),
        ];
        let view = ModelView::new(Some(&modules));
        let id = |path: &str| {
            view.layers
                .iter()
                .find(|layer| layer.path == path)
                .unwrap()
                .id
        };
        for scope in ["left", "right"] {
            let from = id(&format!("model.{scope}.0"));
            let to = id(&format!("model.{scope}.1"));
            assert_eq!(
                view.document.source(Inlet {
                    operator: to,
                    slot: 0
                }),
                Some(Outlet {
                    operator: from,
                    slot: 0
                })
            );
        }
        let from = id("model.left");
        let to = id("model.right");
        assert_eq!(
            view.document.source(Inlet {
                operator: to,
                slot: 0
            }),
            Some(Outlet {
                operator: from,
                slot: 0
            })
        );
    }

    #[cfg(unix)]
    #[test]
    fn named_sequential_uses_only_complete_explicit_child_order() {
        let mut modules = vec![
            json!({"path":"model", "type":"Sequential", "sequential_children":["first", "second"]}),
            json!({"path":"model.second", "type":"Linear"}),
            json!({"path":"model.first", "type":"Linear"}),
        ];
        let view = ModelView::new(Some(&modules));
        assert_eq!(
            view.document.source(Inlet {
                operator: OperatorId(1),
                slot: 0
            }),
            Some(Outlet {
                operator: OperatorId(2),
                slot: 0
            })
        );
        assert_eq!(view.scene(100, 24, false, None).handoffs.len(), 1);
        for invalid in [
            json!(["first"]),
            json!(["first", "first"]),
            json!(["first", "missing"]),
            json!(["first", 7]),
            json!(null),
        ] {
            modules[0]["sequential_children"] = invalid;
            let view = ModelView::new(Some(&modules));
            assert!(view.document.wires().next().is_none());
            assert!(view.flow_layout().is_none());
        }
        modules[0]
            .as_object_mut()
            .unwrap()
            .remove("sequential_children");
        assert!(ModelView::new(Some(&modules)).flow_layout().is_none());
        let duplicate = vec![
            json!({"path":"model", "type":"Sequential", "sequential_children":["a", "b"]}),
            json!({"path":"model.a", "type":"Linear"}),
            json!({"path":"model.a", "type":"Linear"}),
            json!({"path":"model.b", "type":"Linear"}),
        ];
        assert!(
            ModelView::new(Some(&duplicate))
                .document
                .wires()
                .next()
                .is_none()
        );
        let numeric_conflict = vec![
            json!({"path":"model", "type":"Sequential", "sequential_children":["1", "missing"]}),
            json!({"path":"model.0", "type":"Linear"}),
            json!({"path":"model.1", "type":"Linear"}),
        ];
        assert!(
            ModelView::new(Some(&numeric_conflict))
                .document
                .wires()
                .next()
                .is_none()
        );
        let mut custom_named_sequential = numeric_conflict;
        custom_named_sequential[0]["sequential_children"] = Value::Null;
        assert!(
            ModelView::new(Some(&custom_named_sequential))
                .document
                .wires()
                .next()
                .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn sequential_contract_marks_mismatch_and_ignores_undeclared_flow() {
        let mut modules = vec![
            json!({"path":"", "type":"Sequential"}),
            json!({"path":"0", "type":"Linear", "weight_shape":[48,64]}),
            json!({"path":"1", "type":"Linear", "weight_shape":[16,32]}),
            json!({"path":"2", "type":"Linear", "weight_shape":[8,16]}),
        ];
        let mut view = ModelView::new(Some(&modules));
        let scene = view.scene(100, 24, false, None);
        assert_eq!(scene.handoffs.len(), 2);
        assert_eq!(scene.handoffs[0].alignment, Alignment::Mismatch);
        assert_eq!(scene.handoffs[1].alignment, Alignment::Features);
        let mismatch = &scene.handoffs[0];
        let column = ((mismatch.from[0] + mismatch.to[0]) * 0.5) as usize;
        let row = ((mismatch.from[1] + mismatch.to[1]) * 0.25) as usize;
        view.hover_handoff = scene
            .handoffs_at(column, row + SCENE_HEADER + scene.tooltip_rows)
            .first()
            .copied();
        assert_eq!(view.hover_handoff, Some(wire(1, 2, 0)));
        let frame = view.render(&scene, Locale::English, false, false);
        assert!(frame.contains("…×48 → …×32  ·  shape mismatch"));
        assert!(
            scene
                .captions
                .iter()
                .flatten()
                .all(|(_, caption)| !caption.contains('×') && !caption.contains('≠'))
        );

        modules[0]["type"] = json!("ModuleList");
        let view = ModelView::new(Some(&modules));
        assert!(view.scene(100, 24, false, None).handoffs.is_empty());
        let module_list_details = detailed(&view, 100);
        assert!(
            module_list_details.contains("forward links unavailable"),
            "{module_list_details}"
        );
        modules[0]["type"] = json!("Sequential");
        modules.swap(1, 2);
        assert!(ModelView::new(Some(&modules)).flow_layout().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn demo_carries_tensor_handoffs_across_page_boundaries() {
        let mut view = ModelView::new(None);
        let first = view.scene(40, 24, false, None);
        let solid = |layer| {
            first
                .solids
                .iter()
                .find(|solid| solid.layer == layer)
                .unwrap()
        };
        assert_eq!(solid(1).center[0], solid(2).center[0]);
        assert!((solid(1).center[1] - solid(2).center[1]).abs() > 2.0);
        assert_eq!(first.page_size, 3);
        let branch = |layer| {
            first
                .handoffs
                .iter()
                .find(|edge| edge.layers == [0, layer])
                .unwrap()
        };
        assert_eq!(branch(1).wire.from, branch(2).wire.from);
        assert_eq!(branch(1).from, branch(2).from);
        assert_ne!(branch(1).to, branch(2).to);
        for layers in [[0, 1], [0, 2]] {
            assert!(first.handoffs.iter().any(|edge| edge.layers == layers
                && edge.continuation.is_none()
                && edge.alignment == Alignment::Match));
        }
        for layers in [[1, 4], [2, 4]] {
            assert!(
                first.handoffs.iter().any(|edge| edge.layers == layers
                    && edge.continuation == Some(Continuation::Outgoing))
            );
        }
        view.page(&first, true);
        let next = view.scene(40, 24, false, None);
        let incoming = |layers| {
            next.handoffs
                .iter()
                .find(|edge| {
                    edge.layers == layers && edge.continuation == Some(Continuation::Incoming)
                })
                .unwrap()
        };
        assert_ne!(incoming([1, 4]).to[1], incoming([2, 4]).to[1]);
        view.page(&next, false);
        assert_eq!(view.selected, 0);
    }

    #[test]
    fn large_reports_render_one_branch_page_and_refine_in_batches() {
        let count_layers = |scene: &Scene| {
            scene
                .solids
                .iter()
                .map(|solid| solid.layer)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        };
        let mut modules = vec![json!({"path":"model", "type":"Network"})];
        modules.extend(
            (0..2000).map(|index| json!({"path":format!("model.layer{index}"), "type":"Linear"})),
        );
        modules.push(json!({"path":"model.layer25.weight", "type":"Parameter"}));
        let mut view = ModelView::new(Some(&modules));
        assert_eq!(count_layers(&view.scene(80, 24, false, None)), 1);

        view.zoom(0);
        let mut page = view.scene(80, 24, false, None);
        assert!((4..=7).contains(&count_layers(&page)));
        assert!(page.solids.len() <= 7 * 9);
        assert!(page.refine(view.camera, 1));
        assert_eq!(page.outline_cursor, 1);
        assert!(
            view.render(&page, Locale::English, false, false)
                .contains("/2000")
        );

        view.navigate(26);
        assert_eq!(view.selected, 26);
        assert!(
            view.scene(80, 24, false, None)
                .solids
                .iter()
                .any(|solid| solid.layer == 26)
        );
        view.zoom(26);
        assert_eq!(count_layers(&view.scene(80, 24, false, None)), 2);
        assert!(view.zoom_out());
        assert_eq!((view.focus, view.selected), (Some(0), 26));
    }

    #[cfg(unix)]
    #[test]
    fn following_a_cross_group_socket_restores_scope_port_and_camera() {
        use model_maven_tui::diagnosis::Case;
        let fixture = offline::Offline::new(Case::Large);
        let mut view = fixture.view();
        assert!(view.zoom_out());
        let producer = OperatorId(large::FIRST + large::PER_GROUP - 1);
        let consumer = OperatorId(producer.0 + 1);
        let wire = Wire {
            from: Outlet {
                operator: producer,
                slot: 0,
            },
            to: Inlet {
                operator: consumer,
                slot: 0,
            },
        };
        let source = view.indices[&producer];
        let target = view.indices[&consumer];
        let source_group = view.parents[source].unwrap();
        let target_group = view.parents[target].unwrap();
        view.selected = target_group;
        view.zoom(target_group);
        view.toggle_hierarchy();
        assert_eq!(view.selected, target);
        view.selected_wire = Some(wire);
        view.camera.orbit(0.3, -0.1);
        view.viewport.zoom(1.1, [20.0, 6.0]);
        let camera = view.camera;
        let viewport = view.viewport;
        let before = view.scene(60, 24, false, None);
        assert!(before.handoffs.iter().any(|edge| {
            edge.wire == wire && edge.continuation == Some(Continuation::Incoming)
        }));
        view.follow_wire(&before, false, None);
        assert_eq!((view.focus, view.selected), (Some(source_group), source));
        let upstream = view.scene(60, 24, false, None);
        assert!(upstream.handoffs.iter().any(|edge| {
            edge.wire == wire && edge.continuation == Some(Continuation::Outgoing)
        }));
        assert!(view.zoom_out());
        let after = view.scene(60, 24, false, None);
        assert_eq!((view.focus, view.selected), (Some(target_group), target));
        assert_eq!(view.selected_wire, Some(wire));
        assert!(view.camera == camera);
        assert!(view.viewport == viewport);
        assert_eq!(after.page_start, before.page_start);
    }

    #[cfg(unix)]
    #[test]
    fn changing_style_and_background_preserves_socket_geometry() {
        let mut view = offline::Offline::new(model_maven_tui::diagnosis::Case::Mismatch).view();
        view.camera.orbit(0.38, 0.12);
        view.viewport.zoom(1.12, [25.0, 7.0]);
        let selected = view.selected_wire.unwrap();
        let anchor = |scene: &Scene| {
            let route = scene
                .handoffs
                .iter()
                .find(|route| route.wire == selected)
                .unwrap();
            (route.from, route.to)
        };
        let hit_cells = |scene: &Scene| {
            let top = SCENE_HEADER + scene.tooltip_rows;
            (top..top + scene.labels.len() - 1)
                .flat_map(|row| (0..scene.width).map(move |column| (column, row)))
                .filter(|&(column, row)| scene.handoffs_at(column, row).contains(&selected))
                .collect::<Vec<_>>()
        };
        for pixels in [false, true] {
            let baseline = view.scene(100, 30, pixels, None);
            let baseline_hits = hit_cells(&baseline);
            assert!(!baseline_hits.is_empty());
            let mut frames = BTreeSet::new();
            for style in [Style::Default, Style::Compact, Style::Story] {
                view.style = style;
                for background in [Background::Dark, Background::Light] {
                    view.background = background;
                    let mut scene = view.scene(100, 30, pixels, None);
                    assert_eq!(anchor(&scene), anchor(&baseline));
                    assert_eq!(hit_cells(&scene), baseline_hits);
                    assert_eq!(scene.solids.len(), baseline.solids.len());
                    for (solid, original) in scene.solids.iter().zip(&baseline.solids) {
                        assert_eq!(solid.center, original.center);
                        assert_eq!(solid.half, original.half);
                    }
                    assert_eq!(view.selected_wire, Some(selected));
                    assert_eq!(view.document.revision(), 1);
                    if pixels {
                        let mut display = pixels::Display::default();
                        display
                            .prepare(&scene, view.camera, view.selected, Some(selected), true)
                            .unwrap();
                        let mut image = Vec::new();
                        display.place(&mut image, &scene).unwrap();
                        display.clear(&mut Vec::new()).unwrap();
                        frames.insert(image);
                    } else {
                        scene.refine(view.camera, usize::MAX);
                        frames.insert(
                            view.render(&scene, Locale::English, true, false)
                                .into_bytes(),
                        );
                    }
                }
            }
            assert_eq!(frames.len(), 6);
        }
    }

    #[cfg(unix)]
    #[test]
    fn thousand_operator_graph_keeps_layout_and_rendering_local() {
        let mut edits = Vec::new();
        large::append(&mut edits);
        let mut document = GraphDocument::default();
        document
            .apply(Segment {
                base_revision: 0,
                edits,
            })
            .unwrap();
        let mut view = ModelView::from_document(document, false, LayoutCache::default());
        assert_eq!(
            view.overview_layout.as_ref().unwrap().layout.count() as u64,
            large::GROUPS * large::PER_GROUP
        );
        for (width, height, pixels) in [
            (40, 24, false),
            (40, 24, true),
            (100, 30, false),
            (100, 30, true),
            (200, 60, false),
            (200, 60, true),
        ] {
            let mut samples = Vec::new();
            let mut detail = Vec::new();
            let mut encoding = Vec::new();
            let mut display = pixels.then(pixels::Display::default);
            let mut peak_bytes = 0;
            for step in 0..100 {
                let started = std::time::Instant::now();
                view.navigate(if step % 2 == 0 { 1 } else { -1 });
                view.camera.orbit(0.025, 0.0);
                let mut scene = view.scene(width, height, pixels, None);
                assert!(scene.solids.len() <= 7 * 9);
                if !pixels {
                    let refine_started = std::time::Instant::now();
                    scene.refine(view.camera, 3);
                    assert!(scene.outline_cursor <= 3);
                    scene.refine(view.camera, usize::MAX);
                    detail.push(refine_started.elapsed());
                }
                let frame = view.render(&scene, Locale::English, !pixels, pixels);
                assert_eq!(scene.item_count, 1024);
                if let Some(display) = &mut display {
                    let mut output = Vec::new();
                    let raster_started = std::time::Instant::now();
                    display.rasterize(&scene, view.camera, view.selected, None, false);
                    detail.push(raster_started.elapsed());
                    let encode_started = std::time::Instant::now();
                    display.encode_raster().unwrap();
                    encoding.push(encode_started.elapsed());
                    display.place(&mut output, &scene).unwrap();
                    peak_bytes = peak_bytes.max(output.len() + frame.len());
                } else {
                    peak_bytes = peak_bytes.max(frame.len());
                }
                samples.push(started.elapsed());
            }
            if let Some(display) = &mut display {
                display.clear(&mut Vec::new()).unwrap();
            }
            samples.sort_unstable();
            detail.sort_unstable();
            encoding.sort_unstable();
            eprintln!(
                "{width}×{height} frame, pixels={pixels}: total p50 {:?}, p95 {:?}, {} p50 {:?}, p95 {:?}, max {peak_bytes} bytes",
                samples[49],
                samples[94],
                if pixels { "raster" } else { "refine" },
                detail[49],
                detail[94],
            );
            if pixels {
                eprintln!(
                    "{width}×{height} pixel compression + encoding p50 {:?}, p95 {:?}",
                    encoding[49], encoding[94]
                );
            }
        }
    }
}
