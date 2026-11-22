use super::editor::{Editor, MIN_HEIGHT, ROWS};
use super::offline::{self, Exit, Offline, Shortcut};
use super::{Camera, EditOverlay, ModelView, Scene, pixels, show_view_static};
use crate::terminal::{
    interrupt,
    layout::{self, Background},
    locale::Locale,
    menu,
};
use std::{
    collections::{HashMap, VecDeque},
    io::{self, IsTerminal, Write},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Eq, PartialEq)]
struct Point {
    column: usize,
    row: usize,
}

#[derive(Clone, Copy)]
enum Mouse {
    Press(Point),
    Drag(Point),
    Release(Point),
    Move(Point),
    Zoom(Point, bool),
    Page(Point, bool),
}

enum Input {
    Key(u8),
    Escape,
    Arrow(u8),
    Page(bool),
    Focus(bool),
    Background(Background),
    Pointer(Mouse),
    Ignored,
}

impl From<Shortcut> for Input {
    fn from(shortcut: Shortcut) -> Self {
        match shortcut {
            Shortcut::Previous => Self::Arrow(b'A'),
            Shortcut::Next => Self::Arrow(b'B'),
            Shortcut::SwitchPane => Self::Key(b'\t'),
            Shortcut::Expand => Self::Key(b'\n'),
        }
    }
}

#[derive(Default)]
struct InputReader {
    pending: VecDeque<Input>,
}

impl InputReader {
    fn read(&mut self, terminal: &menu::Terminal, timeout: i32) -> io::Result<Option<Input>> {
        if let Some(event) = self.pending.pop_front() {
            return Ok(Some(event));
        }
        let Some(lead) = terminal.read(timeout)? else {
            return Ok(None);
        };
        if lead != 27 {
            return Ok(Some(Input::Key(lead)));
        }
        let deadline = Instant::now() + Duration::from_millis(40);
        let read = || {
            terminal.read(
                deadline
                    .saturating_duration_since(Instant::now())
                    .as_millis() as i32,
            )
        };
        let event = match read()? {
            None => Input::Escape,
            Some(b']') => {
                let mut reply = Vec::with_capacity(24);
                let mut terminated = false;
                for _ in 0..32 {
                    match read()? {
                        Some(7) => {
                            terminated = true;
                            break;
                        }
                        Some(27) => {
                            terminated = read()? == Some(b'\\');
                            break;
                        }
                        Some(byte) => reply.push(byte),
                        None => break,
                    }
                }
                let background = terminated.then(|| Background::parse_osc(&reply)).flatten();
                background.map_or(Input::Ignored, Input::Background)
            }
            Some(b'[' | b'O') => match read()? {
                Some(key @ (b'A'..=b'D')) => Input::Arrow(key),
                Some(b'Z') => Input::Key(b'\t'),
                Some(b'I') => Input::Focus(true),
                Some(b'O') => Input::Focus(false),
                Some(b'<') => read_mouse(read)?.map_or(Input::Ignored, Input::Pointer),
                Some(key @ (b'5' | b'6')) => {
                    let mut page = false;
                    for _ in 0..12 {
                        match read()? {
                            Some(b'~') => {
                                page = true;
                                break;
                            }
                            Some(b'0'..=b'9' | b';') => {}
                            _ => break,
                        }
                    }
                    if page {
                        Input::Page(key == b'6')
                    } else {
                        Input::Ignored
                    }
                }
                _ => Input::Ignored,
            },
            Some(key) => {
                self.pending.push_back(Input::Key(key));
                Input::Escape
            }
        };
        Ok(Some(event))
    }
}

impl Mouse {
    fn point(&self) -> Point {
        match *self {
            Self::Press(point)
            | Self::Drag(point)
            | Self::Release(point)
            | Self::Move(point)
            | Self::Zoom(point, _)
            | Self::Page(point, _) => point,
        }
    }
}

fn read_mouse(mut read: impl FnMut() -> io::Result<Option<u8>>) -> io::Result<Option<Mouse>> {
    let mut report = String::new();
    for _ in 0..24 {
        match read()? {
            Some(final_byte @ (b'M' | b'm')) => return Ok(parse_mouse(&report, final_byte)),
            Some(byte @ (b'0'..=b'9' | b';')) => report.push(char::from(byte)),
            _ => break,
        }
    }
    Ok(None)
}

fn parse_mouse(report: &str, final_byte: u8) -> Option<Mouse> {
    let mut fields = report.split(';');
    let flags = fields.next()?.parse::<u8>().ok()?;
    let point = Point {
        column: fields.next()?.parse::<usize>().ok()?.checked_sub(1)?,
        row: fields.next()?.parse::<usize>().ok()?.checked_sub(1)?,
    };
    if fields.next().is_some() {
        return None;
    }
    // Ignore Shift, Alt and Ctrl, retaining the button and motion bits.
    match (flags & !28, final_byte) {
        (0, b'M') => Some(Mouse::Press(point)),
        (32, b'M') => Some(Mouse::Drag(point)),
        (0, b'm') => Some(Mouse::Release(point)),
        (35, b'M') => Some(Mouse::Move(point)),
        (64 | 65, b'M') => Some(Mouse::Zoom(point, flags & 1 == 0)),
        (66 | 67, b'M') => Some(Mouse::Page(point, flags & 1 != 0)),
        _ => None,
    }
}

#[derive(Default)]
struct Pointer {
    press: Option<Press>,
    previous_click: Option<LayerClick>,
    wire_cell: Option<Point>,
    last_wire_click: Option<Point>,
}

#[derive(Default)]
struct Toolbar {
    pressed: Option<usize>,
    cancelled: bool,
    hovered: Option<usize>,
}
impl Toolbar {
    fn hit(point: Point, width: usize) -> Option<usize> {
        if point.row != 0 || width < super::GRAPH_TOOLBAR_MIN_WIDTH {
            return None;
        }
        let toolbar_width = if width >= super::GRAPH_MODE_TOOLBAR_MIN_WIDTH {
            super::GRAPH_TOOLBAR.len() + super::GRAPH_MODE_TOOLBAR_WIDTH
        } else {
            super::GRAPH_TOOLBAR.len()
        };
        let offset = point.column.checked_sub(width - toolbar_width)?;
        (offset < toolbar_width).then_some(offset / 3)
    }
    fn track(&mut self, mouse: Mouse, width: usize) -> bool {
        let hovered = match mouse {
            Mouse::Move(point)
            | Mouse::Press(point)
            | Mouse::Drag(point)
            | Mouse::Release(point) => Self::hit(point, width),
            Mouse::Zoom(_, _) | Mouse::Page(_, _) => self.hovered,
        };
        let changed = self.hovered != hovered;
        self.hovered = hovered;
        changed
    }
    fn decorate(&self, frame: &mut String, view: &ModelView<'_>, width: usize, locale: Locale) {
        let Some(index) = self.hovered else { return };
        let Some(title) = frame.lines().next() else {
            return;
        };
        if !title.ends_with(&view.toolbar_text(width)) {
            return;
        }
        let hint = match index {
            0 => locale.text("Zoom out", "缩小", "縮小"),
            1 => locale.text("Zoom in", "放大", "放大"),
            2 => locale.text("Orbit left", "向左旋转", "向左旋轉"),
            3 => locale.text("Orbit right", "向右旋转", "向右旋轉"),
            4 => locale.text("Orbit up", "向上旋转", "向上旋轉"),
            5 => locale.text("Orbit down", "向下旋转", "向下旋轉"),
            6 => locale.text("Reset view", "重置视图", "重設視圖"),
            7 => locale.text("Focus selected", "定位所选层", "定位所選層"),
            8 => locale.text("Spatial 3D", "空间 3D", "空間 3D"),
            9 if view.flow_layout().is_some()
                || view.flat_flow
                || (!view.hierarchy && view.unconnected_flow().is_some()) =>
            {
                locale.text("2D flow", "二维流图", "二維流圖")
            }
            9 => locale.text("2D flow unavailable", "暂无二维流图", "暫無二維流圖"),
            10 => locale.text("Exploded x1", "爆炸图 x1", "爆炸圖 x1"),
            11 => locale.text("Exploded x2", "爆炸图 x2", "爆炸圖 x2"),
            _ => return,
        };
        let mut offset = 0;
        for line in frame.split_inclusive('\n') {
            let content = line.strip_suffix('\n').unwrap_or(line);
            if content.contains("Mouse on")
                || content.contains("鼠标开")
                || content.contains("滑鼠開")
            {
                let hint = crate::terminal::text::truncate(hint, width);
                let mut decorated = hint.to_owned();
                decorated.push_str(
                    &" ".repeat(width.saturating_sub(crate::terminal::text::display_width(&hint))),
                );
                frame.replace_range(offset..offset + content.len(), &decorated);
                return;
            }
            offset += line.len();
        }
    }
    fn input(&mut self, view: &mut ModelView<'_>, scene: &Scene, mouse: Mouse) -> Option<bool> {
        match mouse {
            Mouse::Press(point) => {
                self.pressed = Self::hit(point, scene.width);
                self.cancelled = false;
                self.pressed.map(|_| false)
            }
            Mouse::Drag(point) if self.pressed.is_some() => {
                if Self::hit(point, scene.width) != self.pressed {
                    self.cancelled = true;
                }
                Some(false)
            }
            Mouse::Release(point) if self.pressed.is_some() => {
                let pressed = self.pressed.take().unwrap();
                if self.cancelled || Self::hit(point, scene.width) != Some(pressed) {
                    return Some(false);
                }
                let zoom = matches!(pressed, 0 | 1);
                match pressed {
                    0 | 1 => view.viewport.zoom(
                        if pressed == 0 { 1.0 / 1.10 } else { 1.10 },
                        [scene.width as f64 / 2.0, scene.labels.len() as f64],
                    ),
                    2..=5 if view.flat_flow => match pressed {
                        2 => view.viewport.offset[0] += 2.0,
                        3 => view.viewport.offset[0] -= 2.0,
                        4 => view.viewport.offset[1] += 2.0,
                        _ => view.viewport.offset[1] -= 2.0,
                    },
                    2..=5 => match pressed {
                        2 => view.camera.orbit(-0.2, 0.0),
                        3 => view.camera.orbit(0.2, 0.0),
                        4 => view.camera.orbit(0.0, -0.2),
                        _ => view.camera.orbit(0.0, 0.2),
                    },
                    6 => {
                        if view.flat_flow {
                            view.camera = Camera {
                                yaw: 0.0,
                                pitch: 0.0,
                            };
                            view.saved_camera = Some(Camera::default());
                        } else {
                            view.camera = Camera::default();
                        }
                        view.viewport = Default::default();
                        view.module_layout.reset_scope(view.graph_scope());
                    }
                    7 => view.zoom(view.selected),
                    8 => {
                        view.select_graph_preset(1);
                    }
                    9 => {
                        view.select_graph_preset(2);
                    }
                    10 => {
                        view.select_graph_preset(3);
                    }
                    11 => {
                        view.select_graph_preset(4);
                    }
                    _ => unreachable!(),
                }
                view.hover = None;
                view.hover_handoff = None;
                Some(zoom)
            }
            _ => None,
        }
    }
}

struct Press {
    start: Point,
    point: Point,
    layer: Option<usize>,
    wire: bool,
    camera: Camera,
    viewport: super::Viewport,
    module: Option<ModuleDrag>,
    moved: bool,
}

struct ModuleDrag {
    layer: usize,
    anchor: Point,
    center: [f64; 2],
    offset: [f64; 2],
}

struct LayerClick {
    layer: usize,
    point: Point,
    time: Instant,
    wire: Option<super::Wire>,
}

impl Pointer {
    fn input(&mut self, view: &mut ModelView<'_>, scene: &Scene, mouse: Mouse, now: Instant) {
        let hit = |point: Point| scene.layer_at(point.column, point.row);
        let wires_at = |point: Point| scene.handoffs_at(point.column, point.row);
        view.hover_handoff = None;
        match mouse {
            Mouse::Press(point) => {
                if scene.art_row_at(point.column, point.row).is_some() {
                    let wire_hit = !wires_at(point).is_empty();
                    // Layout editing gives the visible module priority over a
                    // socket in the same cell; normal mode keeps wire priority.
                    let layer = hit(point).filter(|_| view.layout_edit || !wire_hit);
                    let wire = wire_hit && layer.is_none();
                    let module = layer.filter(|_| view.layout_edit).and_then(|layer| {
                        let solid = scene.solids.iter().find(|solid| solid.layer == layer)?;
                        Some(ModuleDrag {
                            layer,
                            anchor: point,
                            center: solid.center,
                            offset: view
                                .module_layout
                                .offset(view.graph_scope(), view.layers[layer].id),
                        })
                    });
                    self.press = Some(Press {
                        start: point,
                        point,
                        layer,
                        wire,
                        camera: view.camera,
                        viewport: view.viewport,
                        module,
                        moved: false,
                    });
                    view.hover = None;
                }
            }
            Mouse::Drag(point) => self.drag(view, scene, point),
            Mouse::Release(point) => {
                if self
                    .press
                    .as_ref()
                    .is_some_and(|press| press.point != point)
                {
                    self.drag(view, scene, point);
                }
                if self.dragging() {
                    self.press = None;
                    view.hover = None;
                    return;
                }
                view.hover = hit(point);
                if let Some(press) = self.press.take() {
                    if press.point == point && press.layer.is_none() {
                        view.selected_port = None;
                        let candidates = wires_at(point);
                        let current = (self.last_wire_click == Some(point))
                            .then_some(view.selected_wire)
                            .flatten();
                        view.selected_wire = cycle_wire(candidates.into_iter(), current);
                        self.wire_cell = view.selected_wire.map(|_| point);
                        self.last_wire_click = self.wire_cell;
                        if let Some(wire) = view.selected_wire {
                            view.hover =
                                Self::boundary_layer(view, scene, wire, point).or(view.hover);
                        }
                    }
                    if let Some(layer) = press.layer.filter(|&layer| view.hover == Some(layer)) {
                        view.selected = layer;
                        view.selected_port = None;
                        let operator = view.layers[layer].id;
                        let wire = view.selected_wire.take().filter(|&wire| {
                            let displayed = scene.displayed_wire(wire, layer, None);
                            [
                                wire.from.operator,
                                wire.to.operator,
                                displayed.from.operator,
                                displayed.to.operator,
                            ]
                            .contains(&operator)
                        });
                        self.wire_cell = None;
                        self.last_wire_click = None;
                        if let Some(previous) = self.previous_click.take().filter(|click| {
                            click.layer == layer
                                && now.duration_since(click.time) <= Duration::from_millis(420)
                                && click.point.column.abs_diff(point.column) <= 1
                                && click.point.row.abs_diff(point.row) <= 1
                        }) {
                            view.selected_wire = wire
                                .or(previous.wire)
                                .filter(|wire| view.document.source(wire.to) == Some(wire.from));
                            view.zoom(layer);
                        } else {
                            self.previous_click = Some(LayerClick {
                                layer,
                                point,
                                time: now,
                                wire,
                            });
                        }
                        return;
                    }
                }
                self.previous_click = None;
            }
            Mouse::Move(point) => {
                self.press = None;
                view.hover = hit(point);
                if self.last_wire_click != Some(point) {
                    self.last_wire_click = None;
                }
                let candidates = wires_at(point);
                view.hover_handoff = view
                    .selected_wire
                    .filter(|wire| candidates.contains(wire))
                    .or_else(|| candidates.first().copied());
                if let Some(wire) = view.hover_handoff {
                    view.hover = Self::boundary_layer(view, scene, wire, point).or(view.hover);
                }
                self.wire_cell = view.hover_handoff.map(|_| point);
            }
            Mouse::Zoom(point, inward) => {
                if let Some(row) = scene.art_row_at(point.column, point.row) {
                    let factor = if inward { 1.10 } else { 1.0 / 1.10 };
                    view.viewport
                        .zoom(factor, [point.column as f64 + 0.5, row as f64 * 2.0 + 1.0]);
                    view.hover = None;
                    *self = Self::default();
                }
            }
            Mouse::Page(_, forward) => view.page(scene, forward),
        }
    }

    fn drag(&mut self, view: &mut ModelView<'_>, scene: &Scene, point: Point) {
        let Some(press) = self.press.as_mut().filter(|press| press.point != point) else {
            return;
        };
        if !press.moved
            && press.start.column.abs_diff(point.column) <= 1
            && press.start.row.abs_diff(point.row) <= 1
        {
            return;
        }
        if let Some(module) = &press.module {
            let movement = [
                point.column as f64 - module.anchor.column as f64,
                2.0 * (point.row as f64 - module.anchor.row as f64),
            ];
            let extent = scene.art_extent();
            let offset = std::array::from_fn(|axis| {
                let center = (module.center[axis] + movement[axis]).clamp(1.0, extent[axis] - 1.0);
                module.offset[axis]
                    + (center - module.center[axis]) / (extent[axis] * view.viewport.scale)
            });
            view.module_layout
                .set_offset(view.graph_scope(), view.layers[module.layer].id, offset);
            view.hover = Some(module.layer);
        } else if press.layer.is_none() && !press.wire {
            let dx = point.column as f64 - press.point.column as f64;
            let dy = point.row as f64 - press.point.row as f64;
            if view.flat_flow {
                view.viewport.offset[0] += dx;
                view.viewport.offset[1] += dy * 2.0;
            } else {
                view.camera.orbit(dx * 0.065, -dy * 0.10);
            }
            view.hover = None;
        }
        press.point = point;
        press.moved = true;
        self.previous_click = None;
        self.wire_cell = None;
        self.last_wire_click = None;
    }

    fn dragging(&self) -> bool {
        self.press.as_ref().is_some_and(|press| press.moved)
    }

    fn cancel(&mut self, view: &mut ModelView<'_>) {
        if let Some(press) = self.press.take().filter(|press| press.moved) {
            if let Some(module) = press.module {
                view.module_layout.set_offset(
                    view.graph_scope(),
                    view.layers[module.layer].id,
                    module.offset,
                );
            } else if press.layer.is_none() && !press.wire {
                view.camera = press.camera;
                view.viewport = press.viewport;
            }
        }
        *self = Self::default();
        view.hover = None;
        view.hover_handoff = None;
    }

    fn boundary_layer(
        view: &ModelView<'_>,
        scene: &Scene,
        wire: super::Wire,
        point: Point,
    ) -> Option<usize> {
        let displayed = scene.displayed_wire(wire, view.selected, Some((point.column, point.row)));
        if displayed.to != wire.to {
            view.indices.get(&displayed.to.operator).copied()
        } else if displayed.from != wire.from {
            view.indices.get(&displayed.from.operator).copied()
        } else {
            None
        }
    }

    fn advance_wire(&self, view: &mut ModelView<'_>, scene: &Scene) {
        view.selected_port = None;
        let nearby = self
            .wire_cell
            .map(|point| scene.handoffs_at(point.column, point.row))
            .filter(|wires| {
                wires.len() > 1
                    && view
                        .hover_handoff
                        .or(view.selected_wire)
                        .is_some_and(|wire| wires.contains(&wire))
            });
        if let Some(wires) = nearby {
            view.selected_wire =
                cycle_wire(wires.into_iter(), view.hover_handoff.or(view.selected_wire));
            view.hover_handoff = view.selected_wire;
        } else {
            let mut seen = std::collections::BTreeSet::new();
            view.selected_wire = cycle_wire(
                scene
                    .handoffs
                    .iter()
                    .map(|handoff| handoff.wire)
                    .filter(|&wire| seen.insert(wire)),
                view.selected_wire,
            );
            view.hover_handoff = None;
        }
    }
}

#[cfg(test)]
pub(super) fn click_for_test(view: &mut ModelView<'_>, scene: &Scene, column: usize, row: usize) {
    let point = Point { column, row };
    let now = Instant::now();
    let mut pointer = Pointer::default();
    pointer.input(view, scene, Mouse::Press(point), now);
    pointer.input(view, scene, Mouse::Release(point), now);
}

fn cycle_wire(
    mut candidates: impl Iterator<Item = super::Wire>,
    current: Option<super::Wire>,
) -> Option<super::Wire> {
    let first = candidates.next()?;
    if current == Some(first) {
        candidates.next().or(Some(first))
    } else {
        current
            .and_then(|wire| candidates.find(|&candidate| candidate == wire))
            .and_then(|_| candidates.next())
            .or(Some(first))
    }
}

struct MouseMode;

impl Drop for MouseMode {
    fn drop(&mut self) {
        let mut output = io::stderr().lock();
        let _ = output.write_all(b"\x1b[?1003l\x1b[?1006l\x1b[?1004l");
        let _ = output.flush();
    }
}

const FRAME_INTERVAL: Duration = Duration::from_millis(33);
const CELL_FRAME_INTERVAL: Duration = Duration::from_millis(75);
const INTERACTION_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const ZOOM_REFINE_DELAY: Duration = Duration::from_millis(64);

struct ConfirmationClock {
    settle_at: Option<Instant>,
    expires_at: Instant,
}

fn changed_rows<'a>(previous: &'a str, frame: &'a str) -> impl Iterator<Item = (usize, &'a str)> {
    previous
        .lines()
        .zip(frame.lines())
        .enumerate()
        .filter_map(|(row, (before, after))| (before != after).then_some((row, after)))
}

fn write_rows<'a>(
    output: &mut impl Write,
    rows: impl Iterator<Item = (usize, &'a str)>,
) -> io::Result<()> {
    for (row, line) in rows {
        write!(output, "\x1b[{};1H\x1b[2K{line}", row + 1)?;
    }
    Ok(())
}

fn repaint_moving_rows(output: &mut impl Write, previous: &str, frame: &str) -> io::Result<()> {
    let mut changed = changed_rows(previous, frame).peekable();
    if changed.peek().is_none() {
        return Ok(());
    }
    output.write_all(b"\x1b[?2026h")?;
    write_rows(output, changed)?;
    output.write_all(b"\x1b[?2026l")?;
    output.flush()
}

#[derive(Default)]
struct Motion {
    visible_phase: f64,
    started: Option<(Instant, f64)>,
    focus_feedback: Option<Instant>,
}

impl Motion {
    fn set_running(&mut self, now: Instant, running: bool) {
        if running {
            self.started.get_or_insert((now, self.visible_phase));
        } else {
            self.started = None;
        }
    }

    fn frame_phase(&mut self, now: Instant, advance: bool) -> f64 {
        if let Some((start, phase)) = self.started.filter(|_| advance) {
            self.visible_phase = phase + now.duration_since(start).as_secs_f64();
        }
        // Hover, press and other non-animated redraws share the displayed pose.
        self.visible_phase
    }

    fn trigger_focus_feedback(&mut self, now: Instant, enabled: bool) {
        self.focus_feedback = enabled.then_some(now);
    }

    fn focus_feedback(&self, now: Instant) -> f64 {
        let Some(started) = self.focus_feedback else {
            return 0.0;
        };
        let progress = now.duration_since(started).as_secs_f64() / 0.8;
        if !(0.0..1.0).contains(&progress) {
            return 0.0;
        }
        (std::f64::consts::PI * progress).sin().powi(2)
    }

    fn focus_feedback_active(&self, now: Instant) -> bool {
        self.focus_feedback
            // Keep one frame scheduled after the 800 ms pulse reaches zero.
            .is_some_and(|started| now.duration_since(started) < Duration::from_millis(900))
    }
}

pub(crate) fn show_editor_demo(locale: Locale) -> io::Result<()> {
    show_graph(ModelView::new(None), locale, Some(Editor::default()))
}

pub(super) fn show_view(view: ModelView<'_>, locale: Locale) -> io::Result<()> {
    show_graph(view, locale, None)
}

pub(super) fn show_view_borrowed(view: &mut ModelView<'_>, locale: Locale) -> io::Result<()> {
    if !menu::supported() {
        return show_view_static(view, locale);
    }
    let _interrupt = interrupt::Guard::install();
    run(
        view,
        locale,
        None,
        None,
        &mut crate::welcome::Handoff::default(),
    )?;
    Ok(())
}

pub(super) fn show_view_for_edit(
    view: &mut ModelView<'_>,
    locale: Locale,
) -> io::Result<Option<super::GraphEdit>> {
    if !menu::supported() {
        show_view_static(view, locale)?;
        return Ok(None);
    }
    view.edit_handoff = true;
    let _interrupt = interrupt::Guard::install();
    let exit = run(
        view,
        locale,
        None,
        None,
        &mut crate::welcome::Handoff::default(),
    )?;
    Ok(match exit {
        Exit::Edit => view
            .layers
            .get(view.selected)
            .map(|layer| super::GraphEdit::Variant(layer.path.clone())),
        Exit::EditSource => view
            .layers
            .get(view.selected)
            .map(|layer| super::GraphEdit::Source(layer.path.clone())),
        _ => None,
    })
}

fn show_graph(mut view: ModelView<'_>, locale: Locale, editor: Option<Editor>) -> io::Result<()> {
    if !menu::supported() {
        return show_view_static(&view, locale);
    }
    let _interrupt = interrupt::Guard::install();
    run(
        &mut view,
        locale,
        editor,
        None,
        &mut crate::welcome::Handoff::default(),
    )?;
    Ok(())
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Panes {
    height: usize,
    graph_rows: usize,
    details: bool,
    full_detail: bool,
    editor: bool,
}

impl Panes {
    fn new(
        height: usize,
        offline: Option<&Offline>,
        has_network: bool,
        editor_open: bool,
        details_focused: bool,
    ) -> Self {
        let shared_rows = height.saturating_sub(offline::ROWS);
        let editor = editor_open && height >= MIN_HEIGHT;
        let details = offline.is_some() && (shared_rows >= 12 || details_focused);
        let full_detail = offline.is_some_and(Offline::full_detail_open);
        let evidence_only =
            offline.is_some_and(|details| !has_network && details.axis_view().is_none());
        let graph_rows = if full_detail || evidence_only {
            0
        } else if editor {
            height - ROWS - 1
        } else if details && shared_rows >= 12 {
            shared_rows
        } else if details {
            0
        } else {
            height
        };
        Self {
            height,
            graph_rows,
            details,
            full_detail,
            editor,
        }
    }
}

enum SidePanel<'a> {
    Details(&'a mut Offline, &'a mut bool),
    Editor(&'a mut Editor),
    None,
}

fn route_pointer(
    pointer: &mut Pointer,
    view: &mut ModelView<'_>,
    scene: &Scene,
    mouse: Mouse,
    panel: SidePanel<'_>,
    panes: Panes,
) -> bool {
    if pointer.press.is_some() && matches!(mouse, Mouse::Drag(_) | Mouse::Release(_)) {
        pointer.input(view, scene, mouse, Instant::now());
        return false;
    }
    let in_panel = mouse.point().row >= panes.graph_rows;
    if in_panel {
        *pointer = Pointer::default();
        view.hover = None;
        view.hover_handoff = None;
    }
    let focus_changed = match panel {
        SidePanel::Details(details, _) if scene.axes.is_some() => {
            return match mouse {
                Mouse::Zoom(point, inward)
                    if !in_panel && scene.art_row_at(point.column, point.row).is_some() =>
                {
                    details.zoom_axes(inward)
                }
                _ => false,
            };
        }
        SidePanel::Details(details, focused) => {
            if in_panel {
                match mouse {
                    Mouse::Press(_) if !*focused && !details.modal_open() => {
                        details.select_graph_call(view);
                        *focused = true;
                        return true;
                    }
                    Mouse::Page(_, forward) => {
                        details.navigate(forward, view);
                        return true;
                    }
                    Mouse::Zoom(_, inward) => {
                        details.navigate(!inward, view);
                        return true;
                    }
                    _ => return false,
                }
            }
            if details.modal_open() {
                return false;
            }
            let changed = matches!(mouse, Mouse::Press(_)) && *focused;
            if changed {
                *focused = false;
            }
            changed
        }
        SidePanel::Editor(editor) => {
            if in_panel {
                if matches!(mouse, Mouse::Press(_)) && !editor.focused {
                    editor.enter(view);
                    return true;
                }
                return false;
            }
            let changed = matches!(mouse, Mouse::Press(_)) && editor.focused;
            if changed {
                editor.focused = false;
            }
            changed
        }
        SidePanel::None => false,
    };
    if !in_panel {
        pointer.input(view, scene, mouse, Instant::now());
    }
    focus_changed
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum MotionClass {
    Drag,
    Zoom,
    Hover,
}

fn motion_class(mouse: Mouse, graph_bottom: usize) -> Option<MotionClass> {
    if mouse.point().row >= graph_bottom {
        return None;
    }
    match mouse {
        Mouse::Drag(_) => Some(MotionClass::Drag),
        Mouse::Zoom(..) => Some(MotionClass::Zoom),
        Mouse::Move(_) => Some(MotionClass::Hover),
        _ => None,
    }
}

pub(super) fn select_matrix_wire(
    view: &mut ModelView<'_>,
    wire: super::Wire,
    displayed: super::Wire,
) {
    view.selected_pair = None;
    let target = [displayed.to.operator, displayed.from.operator]
        .into_iter()
        .filter_map(|id| view.indices.get(&id).copied())
        .find(|&index| view.flow_layout().is_some_and(|flow| flow.contains(index)));
    if let Some(index) = target.filter(|&index| index != view.selected) {
        view.focus_wire_endpoint(index, wire);
    } else {
        view.selected_wire = Some(wire);
        view.selected_port = None;
    }
}

fn select_matrix_pair(view: &mut ModelView<'_>, pair: super::Wire) {
    let Some(&target) = view.indices.get(&pair.to.operator) else {
        return;
    };
    if target != view.selected {
        view.remember_view();
        view.selected = target;
        view.viewport = Default::default();
    }
    view.selected_wire = None;
    view.selected_port = None;
    view.selected_pair = Some(pair);
    view.hover = None;
    view.hover_handoff = None;
}

fn open_matrix(
    view: &ModelView<'_>,
    scene: &Scene,
    previous: &HashMap<super::GraphScope, super::matrix::MatrixState>,
) -> Option<super::matrix::Matrix> {
    let mut matrix = super::matrix::Matrix::new(view, scene, view.selected_wire)?;
    if let Some(old) = previous.get(&view.graph_scope()) {
        matrix.restore(
            old,
            (view.selected_wire, view.selected_port, view.selected_pair),
        );
    }
    Some(matrix)
}

/// Locate the uniquely identified registration without turning object matches into wires.
fn focus_failure_target(view: &mut ModelView<'_>, path: &super::failure::Path) -> bool {
    let target = path.graph_target().and_then(|target| {
        let mut matches = view
            .layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| layer.path == target);
        let (index, _) = matches.next()?;
        matches.next().is_none().then_some(index)
    });
    if let Some(index) = target {
        let scope = view.parents[index];
        let relocating = view.selected != index || view.focus != scope;
        if relocating
            || view.selected_wire.is_some()
            || view.selected_port.is_some()
            || view.selected_pair.is_some()
        {
            view.remember_view();
        }
        if relocating {
            view.focus = scope;
            view.selected = index;
            view.viewport = Default::default();
        }
        view.selected_wire = None;
        view.selected_port = None;
        view.selected_pair = None;
        view.hover = None;
        view.hover_handoff = None;
    }
    if let Some(evidence) = &mut view.live_evidence {
        evidence.graph_target_unresolved = target.is_none();
    }
    target.is_some()
}

fn project_failure_matches(
    view: &mut ModelView<'_>,
    path: &super::failure::Path,
    stack: &[super::failure::Path],
) {
    let projection = path.graph_projection(view).or_else(|| {
        stack
            .iter()
            .rev()
            .find_map(|previous| previous.graph_projection(view))
    });
    if let Some(projection) = projection
        && let Some(evidence) = &mut view.live_evidence
    {
        evidence.projection = Some(projection);
    }
}

fn open_failure_path(
    view: &mut ModelView<'_>,
    offline: Option<&mut Offline>,
    graph_focused: bool,
    existing: &mut Option<super::failure::Path>,
) -> bool {
    let next = if let Some(details) = offline {
        if graph_focused {
            details.select_graph_call(view);
        }
        Some(details.failure_path())
    } else {
        super::failure::Path::live(view)
    };
    let Some(next) = next else {
        return false;
    };
    if existing.as_ref().is_none_or(|previous| {
        previous.alias != next.alias
            || previous.invocation != next.invocation
            || previous.invocation_id != next.invocation_id
            || previous.report != next.report
    }) {
        *existing = Some(next);
    }
    if let Some(evidence) = &mut view.live_evidence {
        evidence.show_input = true;
    }
    true
}

fn open_repair_diff(
    view: &ModelView<'_>,
    offline: Option<&Offline>,
    locale: Locale,
    existing: &mut Option<super::diff::Diff>,
) -> bool {
    let Some(mut next) = offline.and_then(|details| details.repair_diff(view, locale)) else {
        return false;
    };
    if let Some(previous) = existing.as_ref() {
        next.preserve_side(previous);
    }
    *existing = Some(next);
    true
}

fn make_scene(
    view: &ModelView<'_>,
    offline: Option<&Offline>,
    width: usize,
    panes: Panes,
    pixels: bool,
    seconds: f64,
    locale: Locale,
) -> Scene {
    if let Some(comparison) = offline
        .and_then(Offline::axis_view)
        .or_else(|| view.live_axis_view())
    {
        return Scene::for_axes(
            view.appearance(),
            width,
            if view.live_axes_zoom.is_some() {
                panes.graph_rows.saturating_sub(3)
            } else {
                panes.graph_rows
            },
            comparison,
            pixels,
            locale,
        );
    }
    let mut scene = view.scene(width, panes.graph_rows, pixels, Some(seconds));
    if panes.editor && !pixels {
        scene.refine(view.camera, usize::MAX);
    }
    scene
}

#[derive(Clone, Copy)]
enum RenderMode {
    Cells { color: bool },
    Pixels,
}

#[derive(Clone, Copy)]
enum DetailTarget {
    Pair(super::Wire),
    Module(super::OperatorId),
}

struct PairDetails {
    target: DetailTarget,
    offset: usize,
}

impl PairDetails {
    fn wrapped(
        &self,
        view: &ModelView<'_>,
        width: usize,
        locale: Locale,
    ) -> Vec<(String, usize, usize)> {
        let path = |id| {
            view.document
                .operator(id)
                .map_or("?", |operator| operator.path.as_str())
        };
        let body = match self.target {
            DetailTarget::Pair(pair) => {
                let state = if view.document.source(pair.to) == Some(pair.from) {
                    locale.text("Recorded wire", "已记录连线", "已記錄連線")
                } else if view.disconnected.contains(&pair) {
                    locale.text("Declared no wire", "已声明无连线", "已宣告無連線")
                } else {
                    locale.text("Connection unknown", "连线状态未知", "連線狀態未知")
                };
                format!(
                    "{}\n\n{}\n{}.out[{}]\n\n{}\n{}.in[{}]",
                    state,
                    locale.text("Output", "输出", "輸出"),
                    path(pair.from.operator),
                    pair.from.slot,
                    locale.text("Input", "输入", "輸入"),
                    path(pair.to.operator),
                    pair.to.slot
                )
            }
            DetailTarget::Module(id) => {
                let Some(layer) = view
                    .indices
                    .get(&id)
                    .and_then(|&index| view.layers.get(index))
                else {
                    return vec![(
                        locale
                            .text("Module unavailable", "模块不可用", "模組無法取得")
                            .to_owned(),
                        0,
                        0,
                    )];
                };
                let breadcrumb = layer
                    .path
                    .split('.')
                    .map(str::trim)
                    .collect::<Vec<_>>()
                    .join(" › ");
                let mut body = format!(
                    "{}\n{}\n\n{}\n{}\n\n{}\n{}",
                    locale.text("Hierarchy", "层级位置", "層級位置"),
                    breadcrumb,
                    locale.text("Exact path", "完整路径", "完整路徑"),
                    layer.path,
                    locale.text("Type", "类型", "類型"),
                    layer.kind
                );
                let texture = super::appearance::SliceStyle::texture_cue(&layer.kind, locale);
                if let Some(texture) = texture {
                    body.push_str(&format!(
                        "\n\n{}\n{texture}",
                        locale.text("Surface texture cue", "表面纹理提示", "表面紋理提示")
                    ));
                }
                if let Some(instance) = layer.data["instance_id"].as_str() {
                    body.push_str(&format!(
                        "\n\n{}\n{}",
                        locale.text(
                            "Instance ID · reported",
                            "实例标识 · 已报告",
                            "實例識別 · 已回報"
                        ),
                        instance
                    ));
                }
                if let Some(count) = super::parameter_count(&layer.data) {
                    body.push_str(&format!(
                        "\n\n{}\n{count}",
                        locale.text(
                            "Parameters · reported",
                            "参数量 · 已报告",
                            "參數量 · 已回報"
                        )
                    ));
                }
                if let Some(source) = layer.data["source"].as_str() {
                    body.push_str(&format!(
                        "\n\n{}\n{}",
                        locale.text("Source · reported", "来源 · 已报告", "來源 · 已回報"),
                        source
                    ));
                }
                if let Some(aliases) = layer.data["aliases"]
                    .as_array()
                    .filter(|aliases| !aliases.is_empty())
                {
                    body.push_str(&format!(
                        "\n\n{}",
                        locale.text(
                            "Registered aliases · reported",
                            "注册别名 · 已报告",
                            "註冊別名 · 已回報"
                        )
                    ));
                    for alias in aliases
                        .iter()
                        .take(64)
                        .filter_map(serde_json::Value::as_str)
                    {
                        body.push('\n');
                        body.push_str(alias);
                    }
                    if aliases.len() > 64 {
                        body.push_str(&format!(
                            "\n{} {}",
                            locale.text(
                                "Additional aliases omitted:",
                                "其余别名已省略：",
                                "其餘別名已省略："
                            ),
                            aliases.len() - 64
                        ));
                    }
                }
                body
            }
        };
        let body = crate::terminal::text::plain_text(&body);
        let mut output = Vec::new();
        for (paragraph_index, paragraph) in body.split('\n').enumerate() {
            if paragraph.is_empty() {
                output.push((String::new(), paragraph_index, 0));
                continue;
            }
            let mut start = 0;
            let mut columns = 0;
            for (index, ch) in paragraph.char_indices() {
                let cell_width = crate::terminal::text::display_width(&ch.to_string());
                if columns + cell_width > width.max(1) && index > start {
                    output.push((paragraph[start..index].to_owned(), paragraph_index, start));
                    start = index;
                    columns = 0;
                }
                columns += cell_width;
            }
            if start < paragraph.len() {
                output.push((paragraph[start..].to_owned(), paragraph_index, start));
            }
        }
        output
    }

    fn lines(&self, view: &ModelView<'_>, width: usize, locale: Locale) -> Vec<String> {
        self.wrapped(view, width, locale)
            .into_iter()
            .map(|(line, _, _)| line)
            .collect()
    }

    fn resize(
        &mut self,
        view: &ModelView<'_>,
        old_size: (usize, usize),
        new_size: (usize, usize),
        locale: Locale,
    ) {
        let old = self.wrapped(view, old_size.0, locale);
        let old_offset = self.offset.min(
            old.len()
                .saturating_sub(old_size.1.saturating_sub(3).max(1)),
        );
        let Some((_, paragraph, byte)) = old.get(old_offset) else {
            self.offset = 0;
            return;
        };
        let new = self.wrapped(view, new_size.0, locale);
        let anchor = new
            .iter()
            .enumerate()
            .filter(|(_, (_, p, b))| p == paragraph && b <= byte)
            .map(|(index, _)| index)
            .next_back()
            .unwrap_or(0);
        self.offset = anchor.min(
            new.len()
                .saturating_sub(new_size.1.saturating_sub(3).max(1)),
        );
    }

    fn scroll(
        &mut self,
        view: &ModelView<'_>,
        width: usize,
        height: usize,
        locale: Locale,
        forward: bool,
        page: bool,
    ) {
        let available = height.saturating_sub(3).max(1);
        let max = self
            .lines(view, width, locale)
            .len()
            .saturating_sub(available);
        let step = if page { available } else { 1 };
        self.offset = if forward {
            self.offset.saturating_add(step).min(max)
        } else {
            self.offset.min(max).saturating_sub(step)
        };
    }

    fn render(&self, view: &ModelView<'_>, width: usize, height: usize, locale: Locale) -> String {
        let mut output = String::new();
        super::push(
            &mut output,
            match self.target {
                DetailTarget::Pair(_) => {
                    locale.text("Port pair evidence", "端口对证据", "連接埠對證據")
                }
                DetailTarget::Module(_) => locale.text("Module details", "模块详情", "模組詳情"),
            },
            width,
        );
        let lines = self.lines(view, width, locale);
        let available = height.saturating_sub(3).max(1);
        let offset = self.offset.min(lines.len().saturating_sub(available));
        for line in lines.iter().skip(offset).take(available) {
            super::push(&mut output, line, width);
        }
        super::push(
            &mut output,
            &format!(
                "{} {}/{}",
                locale.text("Lines", "行", "行"),
                offset + 1,
                lines.len()
            ),
            width,
        );
        super::push(
            &mut output,
            locale.text(
                "↑↓/Pg scroll · v/Esc back",
                "↑↓/Pg 滚动 · v/Esc 返回",
                "↑↓/Pg 捲動 · v/Esc 返回",
            ),
            width,
        );
        output
    }
}

fn frame(
    view: &ModelView<'_>,
    scene: &Scene,
    matrix: Option<&super::matrix::Matrix>,
    search: Option<&super::search::Search>,
    failure: Option<&super::failure::Path>,
    diff: Option<&super::diff::Diff>,
    candidates: Option<&super::candidates::Candidates>,
    pair_details: Option<&PairDetails>,
    editor: Option<&Editor>,
    offline: Option<(&Offline, bool)>,
    panes: Panes,
    locale: Locale,
    mode: RenderMode,
) -> String {
    if let Some(details) = pair_details {
        return details.render(view, scene.width, panes.height, locale);
    }
    if let Some(candidates) = candidates {
        return candidates.render(view, scene.width, panes.height, locale);
    }
    if let Some(diff) = diff {
        return diff.render(scene.width, panes.height, locale);
    }
    if let Some(failure) = failure {
        return failure.render(
            view,
            scene.width,
            panes.height,
            locale,
            offline.map(|(details, _)| details),
        );
    }
    if let Some(search) = search {
        return search.render(view, scene.width, panes.height, locale);
    }
    if let Some(matrix) = matrix {
        return matrix.render(view, scene.width, panes.height, locale);
    }
    let (color, pixels) = match mode {
        RenderMode::Cells { color } => (color, false),
        RenderMode::Pixels => (true, true),
    };
    let mut frame = if panes.graph_rows > 0 {
        view.render(scene, locale, color, pixels)
    } else {
        String::new()
    };
    if let Some((details, focused)) = offline {
        let graph_rows = panes.graph_rows;
        if graph_rows == 0 {
            return details
                .render_full_detail(locale, scene.width, panes.height)
                .unwrap_or_else(|| details.render(locale, scene.width, focused, false));
        }
        let graph_end = frame
            .split_inclusive('\n')
            .take(graph_rows)
            .map(str::len)
            .sum();
        frame.truncate(graph_end);
        let padding = graph_rows.saturating_sub(frame.lines().count());
        frame.extend(std::iter::repeat_n('\n', padding));
        if let Some(title_end) = frame.find('\n').filter(|_| !focused) {
            let hint = details.graph_hint(view, locale);
            let toolbar = view.toolbar_text(scene.width);
            let hint = if scene.width >= super::GRAPH_MODE_TOOLBAR_MIN_WIDTH {
                let hint = details.graph_wide_hint(view, locale);
                if toolbar.is_empty() {
                    crate::terminal::text::truncate(&hint, scene.width).to_owned()
                } else {
                    let budget = scene.width.saturating_sub(toolbar.len());
                    let hint = crate::terminal::text::truncate(&hint, budget);
                    let gap = budget.saturating_sub(crate::terminal::text::display_width(&hint));
                    format!("{hint}{}{toolbar}", " ".repeat(gap))
                }
            } else {
                crate::terminal::text::truncate(&hint, scene.width).to_owned()
            };
            frame.replace_range(..title_end, &hint);
        }
        if panes.details {
            frame.push_str(&details.render(locale, scene.width, focused, scene.axes.is_some()));
        }
        return frame;
    }
    if let Some(editor) = editor.filter(|_| panes.editor) {
        let padding = panes.graph_rows.saturating_sub(frame.lines().count());
        frame.extend(std::iter::repeat_n('\n', padding));
        frame.push_str(&editor.render(view, locale, scene.width));
    } else if editor.is_some() {
        let hint = locale.text(
            "Need 18 rows for editor",
            "编辑区需 18 行",
            "編輯區需 18 行",
        );
        if let Some(title_end) = frame.find('\n') {
            frame.replace_range(
                ..title_end,
                &crate::terminal::text::truncate(hint, scene.width),
            );
        }
    }
    frame
}

pub(super) fn run(
    view: &mut ModelView<'_>,
    locale: Locale,
    mut editor: Option<Editor>,
    mut offline: Option<&mut Offline>,
    handoff: &mut crate::welcome::Handoff,
) -> io::Result<Exit> {
    let terminal = menu::Terminal::open()?;
    handoff.enter_screen()?;
    let _mouse = MouseMode;
    let color = layout::color_enabled(io::stderr().is_terminal());
    let query_background = color
        && Background::override_from_env().is_none()
        && (matches!(
            std::env::var("TERM").as_deref(),
            Ok("xterm-ghostty" | "xterm-kitty")
        ) || std::env::var("TERM_PROGRAM").as_deref() == Ok("ghostty"));
    let mut display = pixels::supported(color).then(pixels::Display::default);
    let mut output = io::stderr().lock();
    output.write_all(b"\x1b[?1006h\x1b[?1003h\x1b[?1004h")?;
    if query_background {
        output.write_all(b"\x1b]11;?\x1b\\")?;
    }

    let pixel_mode = display.is_some();
    let render_mode = if pixel_mode {
        RenderMode::Pixels
    } else {
        RenderMode::Cells { color }
    };
    let mut size = (layout::viewport_width(), menu::rows());
    // Live inspection has no offline details panel; graph shortcuts must be
    // available immediately instead of focusing a panel that cannot receive Tab.
    let mut details_focused = offline.is_some();
    let mut panes = Panes::new(
        size.1,
        offline.as_deref(),
        !view.layers.is_empty(),
        editor.is_some(),
        details_focused,
    );
    let mut scene = make_scene(
        view,
        offline.as_deref(),
        size.0,
        panes,
        pixel_mode,
        0.0,
        locale,
    );
    let mut dirty = true;
    let mut panel_dirty = false;
    let mut pointer = Pointer::default();
    let mut toolbar = Toolbar::default();
    let mut matrix: Option<super::matrix::Matrix> = None;
    let mut previous_matrix = HashMap::new();
    let mut matrix_press: Option<(usize, usize)> = None;
    let mut search: Option<super::search::Search> = None;
    let mut failure: Option<super::failure::Path> = view
        .live_evidence
        .as_ref()
        .filter(|evidence| evidence.show_input)
        .and_then(|_| super::failure::Path::live(view));
    let mut failure_visible = failure.is_some();
    let mut failure_stack: Vec<super::failure::Path> = Vec::new();
    let mut diff: Option<super::diff::Diff> = None;
    let mut diff_visible = false;
    let mut candidates: Option<super::candidates::Candidates> = None;
    let mut pair_details: Option<PairDetails> = None;
    let mut candidate_press: Option<usize> = None;
    let mut consumer_press: Option<usize> = None;
    let mut input = InputReader::default();
    let mut refine_after = Instant::now() + Duration::from_millis(16);
    let mut zoom_refine_after: Option<Instant> = None;
    let mut next_frame = Instant::now();
    let mut chrome = String::new();
    let mut chrome_size = None;
    let mut motion_enabled = handoff.motion_enabled();
    let mut focused = true;
    let mut confirmation: Option<ConfirmationClock> = None;
    let mut motion = Motion::default();
    let mut projection_pending = false;
    loop {
        if interrupt::take() {
            return Ok(Exit::Quit);
        }
        let now = Instant::now();
        view.focus_pulse = if motion_enabled {
            motion.focus_feedback(now)
        } else {
            0.0
        };
        let details_has_focus =
            details_focused || offline.as_deref().is_some_and(Offline::modal_open);
        if zoom_refine_after.is_some_and(|deadline| now >= deadline) {
            zoom_refine_after = None;
            if pixel_mode && panes.graph_rows > 0 && !pointer.dragging() {
                dirty = true;
                next_frame = now;
            }
        }
        let focus_feedback_active = motion_enabled
            && motion.focus_feedback_active(now)
            && !view.flat_flow
            && view.edit_overlay.is_none();
        let animate = motion_enabled
            && focused
            && matrix.is_none()
            && search.is_none()
            && !failure_visible
            && !diff_visible
            && candidates.is_none()
            && pair_details.is_none()
            && !view.flat_flow
            && pointer.press.is_none()
            && zoom_refine_after.is_none()
            && view.hover.is_none()
            && view.hover_handoff.is_none()
            && !(offline.is_some() && details_has_focus)
            && !panes.editor;
        motion.set_running(now, animate);
        let tick = (animate || focus_feedback_active) && now >= next_frame;
        let mut marker_changed = false;
        if confirmation
            .as_ref()
            .is_some_and(|clock| now >= clock.expires_at)
        {
            confirmation = None;
            view.edit_overlay = None;
            marker_changed = true;
        } else if let Some(clock) = confirmation
            .as_mut()
            .filter(|clock| clock.settle_at.is_some_and(|deadline| now >= deadline))
        {
            clock.settle_at = None;
            if let Some(EditOverlay::Confirmed { expanded, .. }) = &mut view.edit_overlay {
                *expanded = false;
                marker_changed = true;
            }
        }
        if marker_changed {
            dirty |= pixel_mode;
            next_frame = now;
        }
        let pixel_frame_due = pixel_mode && (dirty || tick) && now >= next_frame;
        if (!pixel_mode || pixel_frame_due) && (projection_pending || tick || marker_changed) {
            // Until an update is emitted, input uses the last displayed geometry.
            scene = make_scene(
                view,
                offline.as_deref(),
                size.0,
                panes,
                pixel_mode,
                motion.frame_phase(now, tick && animate),
                locale,
            );
            projection_pending = false;
            if !pixel_mode && (tick || marker_changed) {
                scene.refine(view.camera, usize::MAX);
            }
        }
        // A short workbench can show the receipt before there is room for the graph.
        if confirmation.is_none()
            && !projection_pending
            && panes.graph_rows > 0
            && scene.edit_mark.is_some()
            && matches!(
                view.edit_overlay,
                Some(EditOverlay::Confirmed { expanded: true, .. })
            )
        {
            let start = Instant::now();
            let duration = Duration::from_millis(view.appearance().feedback_ms());
            confirmation = Some(ConfirmationClock {
                settle_at: Some(start + duration / 3),
                expires_at: start + duration,
            });
        }
        let can_refine = panes.graph_rows > 0
            && !pixel_mode
            && !pointer.dragging()
            && zoom_refine_after.is_none()
            && scene.outline_cursor < scene.solids.len();
        let refined = if can_refine && now >= refine_after {
            refine_after = Instant::now() + Duration::from_millis(16);
            scene.refine(view.camera, 3)
        } else {
            false
        };
        if let Some(display) = &mut display {
            if pixel_frame_due {
                let interacting = pointer.dragging() || zoom_refine_after.is_some();
                let active = view.hover.unwrap_or(view.selected);
                let graph_visible = panes.graph_rows > 0
                    && matrix.is_none()
                    && search.is_none()
                    && !failure_visible
                    && !diff_visible;
                let graph_visible = graph_visible && candidates.is_none() && pair_details.is_none();
                if graph_visible {
                    display.prepare(
                        &scene,
                        view.camera,
                        active,
                        view.selected_wire.or(view.hover_handoff),
                        !interacting,
                    )?;
                }
                let frame = frame(
                    view,
                    &scene,
                    matrix.as_ref(),
                    search.as_ref(),
                    failure.as_ref().filter(|_| failure_visible),
                    diff.as_ref().filter(|_| diff_visible),
                    candidates.as_ref(),
                    pair_details.as_ref(),
                    editor.as_ref(),
                    offline.as_deref().map(|d| (d, details_focused)),
                    panes,
                    locale,
                    render_mode,
                );
                let mut frame = frame;
                toolbar.decorate(&mut frame, view, scene.width, locale);
                output.write_all(b"\x1b[?2026h")?;
                if frame != chrome || chrome_size != Some(size) {
                    if interacting
                        && chrome_size == Some(size)
                        && chrome.lines().count() == frame.lines().count()
                    {
                        // Keep unchanged text and the previous placement visible until the new image is ready.
                        write_rows(&mut output, changed_rows(&chrome, &frame))?;
                    } else {
                        let screen = frame.strip_suffix('\n').unwrap_or(&frame);
                        write!(output, "\x1b[H\x1b[2J{screen}")?;
                    }
                    chrome = frame;
                    chrome_size = Some(size);
                }
                if graph_visible {
                    display.place(&mut output, &scene)?;
                } else {
                    display.clear(&mut output)?;
                }
                output.write_all(b"\x1b[?2026l")?;
                output.flush()?;
                let interval = if interacting {
                    INTERACTION_FRAME_INTERVAL
                } else {
                    FRAME_INTERVAL
                };
                next_frame = (now + interval).max(Instant::now() + Duration::from_millis(2));
                dirty = false;
                panel_dirty = false;
            }
        } else if dirty || tick || marker_changed || refined {
            let frame = frame(
                view,
                &scene,
                matrix.as_ref(),
                search.as_ref(),
                failure.as_ref().filter(|_| failure_visible),
                diff.as_ref().filter(|_| diff_visible),
                candidates.as_ref(),
                pair_details.as_ref(),
                editor.as_ref(),
                offline.as_deref().map(|d| (d, details_focused)),
                panes,
                locale,
                render_mode,
            );
            let mut frame = frame;
            toolbar.decorate(&mut frame, view, scene.width, locale);
            if dirty {
                // A newline on the last screen row scrolls the header off the alternate screen.
                let screen = frame.strip_suffix('\n').unwrap_or(&frame);
                write!(output, "\x1b[?2026h\x1b[H\x1b[2J{screen}\x1b[?2026l")?;
                output.flush()?;
            } else {
                repaint_moving_rows(&mut output, &chrome, &frame)?;
            }
            chrome = frame;
            if tick {
                next_frame = now + CELL_FRAME_INTERVAL;
            }
            dirty = false;
            panel_dirty = false;
        }
        if panel_dirty && !dirty {
            let frame = frame(
                view,
                &scene,
                matrix.as_ref(),
                search.as_ref(),
                failure.as_ref().filter(|_| failure_visible),
                diff.as_ref().filter(|_| diff_visible),
                candidates.as_ref(),
                pair_details.as_ref(),
                editor.as_ref(),
                offline.as_deref().map(|d| (d, details_focused)),
                panes,
                locale,
                render_mode,
            );
            let mut frame = frame;
            toolbar.decorate(&mut frame, view, scene.width, locale);
            repaint_moving_rows(&mut output, &chrome, &frame)?;
            chrome = frame;
            panel_dirty = false;
        }
        let old = (
            view.selected,
            view.focus,
            view.camera,
            view.viewport,
            view.hierarchy,
        );
        let old_appearance = (view.style, view.background);
        let old_flat_flow = view.flat_flow;
        let old_explosion = view.explosion;
        let old_matrix_open = matrix.is_some();
        let old_search = search.clone();
        let old_failure_visible = failure_visible;
        let old_diff_visible = diff_visible;
        let old_candidates_open = candidates.is_some();
        let old_pair_details_open = pair_details.is_some();
        let old_hover = view.hover;
        let old_handoff = view.hover_handoff;
        let old_selected_wire = view.selected_wire;
        let old_selected_port = view.selected_port;
        let old_selected_pair = view.selected_pair;
        let old_overlay = view.edit_overlay;
        let old_revision = view.document.revision();
        let old_layout = view.module_layout.generation;
        let old_input_modes = (view.layout_edit, view.mouse_capture);
        let old_source_error = view.source_error.clone();
        let old_panel = editor.is_some();
        let was_dragging = pointer.dragging();
        if old.0 != view.selected
            || old.1 != view.focus
            || old_selected_wire != view.selected_wire
            || old_selected_port != view.selected_port
        {
            motion.trigger_focus_feedback(
                Instant::now(),
                motion_enabled && !view.flat_flow && view.edit_overlay.is_none(),
            );
        }
        let timeout = if animate || focus_feedback_active || (pixel_mode && dirty) {
            next_frame
                .saturating_duration_since(Instant::now())
                .as_millis()
                .max(1) as i32
        } else if can_refine {
            16
        } else {
            100
        };
        let confirmation_due = confirmation
            .as_ref()
            .map(|clock| clock.settle_at.unwrap_or(clock.expires_at));
        let timeout = [confirmation_due, zoom_refine_after]
            .into_iter()
            .flatten()
            .fold(timeout, |timeout, deadline| {
                timeout.min(
                    deadline
                        .saturating_duration_since(Instant::now())
                        .as_millis()
                        .max(1) as i32,
                )
            });
        let mut viewport_zoom = false;
        let event = input.read(&terminal, timeout)?.map(|event| match event {
            Input::Key(key)
                if search.is_none()
                    && matrix.is_none()
                    && candidates.is_none()
                    && pair_details.is_none()
                    && !failure_visible
                    && !diff_visible =>
            {
                offline
                    .as_deref()
                    .and_then(|details| details.shortcut(key, details_has_focus))
                    .map_or(Input::Key(key), Input::from)
            }
            event => event,
        });
        if toolbar.pressed.is_some()
            && matches!(
                &event,
                Some(Input::Key(_) | Input::Arrow(_) | Input::Page(_))
            )
        {
            toolbar.pressed = None;
        }
        if candidates.is_some()
            && matches!(
                &event,
                Some(Input::Key(_) | Input::Arrow(_) | Input::Page(_))
            )
        {
            candidate_press = None;
        }
        if matrix.is_some()
            && matches!(
                &event,
                Some(Input::Key(_) | Input::Arrow(_) | Input::Page(_))
            )
        {
            matrix_press = None;
        }
        if matches!(
            &event,
            Some(Input::Key(_) | Input::Arrow(_) | Input::Page(_))
        ) {
            consumer_press = None;
        }
        match event {
            Some(Input::Escape | Input::Key(b'v' | b'V' | b'b' | b'B'))
                if pair_details.is_some() =>
            {
                pair_details = None;
            }
            Some(Input::Arrow(key @ (b'A' | b'B'))) if pair_details.is_some() => {
                pair_details.as_mut().unwrap().scroll(
                    view,
                    size.0,
                    size.1,
                    locale,
                    key == b'B',
                    false,
                );
                dirty = true;
            }
            Some(Input::Page(forward)) if pair_details.is_some() => {
                pair_details
                    .as_mut()
                    .unwrap()
                    .scroll(view, size.0, size.1, locale, forward, true);
                dirty = true;
            }
            Some(Input::Pointer(Mouse::Zoom(_, inward))) if pair_details.is_some() => {
                pair_details
                    .as_mut()
                    .unwrap()
                    .scroll(view, size.0, size.1, locale, !inward, false);
                dirty = true;
            }
            Some(Input::Key(b'q' | b'Q' | 4)) if pair_details.is_some() => return Ok(Exit::Quit),
            Some(Input::Key(_) | Input::Arrow(_) | Input::Pointer(_)) if pair_details.is_some() => {
            }
            Some(Input::Escape) if toolbar.pressed.is_some() => toolbar.pressed = None,
            Some(Input::Escape) if pointer.press.is_some() => pointer.cancel(view),
            Some(Input::Escape | Input::Key(b'W')) if candidates.is_some() => candidates = None,
            Some(Input::Arrow(key @ (b'A' | b'B'))) if candidates.is_some() => {
                candidates.as_mut().unwrap().movement(key == b'B');
                dirty = true;
            }
            Some(Input::Page(forward)) if candidates.is_some() => {
                candidates.as_mut().unwrap().page(forward, panes.height);
                dirty = true;
            }
            Some(Input::Key(b's' | b'S')) if candidates.is_some() => {
                candidates.as_mut().unwrap().toggle_scope(view, &scene);
                candidate_press = None;
                dirty = true;
            }
            Some(Input::Key(b'\r' | b'\n')) if candidates.is_some() => {
                let (wire, displayed) = candidates.as_ref().unwrap().selected();
                select_matrix_wire(view, wire, displayed);
                view.hover = None;
                view.hover_handoff = None;
                candidates = None;
            }
            Some(Input::Pointer(Mouse::Press(point))) if candidates.is_some() => {
                candidate_press = (point.column < scene.width)
                    .then(|| {
                        candidates
                            .as_ref()
                            .unwrap()
                            .index_at(point.row, panes.height)
                    })
                    .flatten();
            }
            Some(Input::Pointer(Mouse::Drag(point) | Mouse::Move(point)))
                if candidates.is_some() =>
            {
                if candidate_press.is_some()
                    && (point.column >= scene.width
                        || candidates
                            .as_ref()
                            .unwrap()
                            .index_at(point.row, panes.height)
                            != candidate_press)
                {
                    candidate_press = None;
                }
            }
            Some(Input::Pointer(Mouse::Release(point))) if candidates.is_some() => {
                let pressed = candidate_press.take();
                let released = (point.column < scene.width)
                    .then(|| {
                        candidates
                            .as_ref()
                            .unwrap()
                            .index_at(point.row, panes.height)
                    })
                    .flatten();
                if let Some(index) = pressed.filter(|index| Some(*index) == released) {
                    candidates.as_mut().unwrap().select(index);
                    let (wire, displayed) = candidates.as_ref().unwrap().selected();
                    select_matrix_wire(view, wire, displayed);
                    view.hover = None;
                    view.hover_handoff = None;
                    candidates = None;
                }
            }
            Some(Input::Pointer(Mouse::Zoom(_, inward))) if candidates.is_some() => {
                candidate_press = None;
                candidates.as_mut().unwrap().movement(!inward);
                dirty = true;
            }
            Some(Input::Key(_) | Input::Arrow(_) | Input::Page(_) | Input::Pointer(_))
                if candidates.is_some() => {}
            Some(Input::Escape | Input::Key(b'v' | b'V'))
                if diff_visible && diff.as_ref().is_some_and(|diff| diff.detail_open()) =>
            {
                diff.as_mut().unwrap().toggle_details();
                dirty = true;
            }
            Some(Input::Arrow(key @ (b'A' | b'B')))
                if diff_visible && diff.as_ref().is_some_and(|diff| diff.detail_open()) =>
            {
                diff.as_mut()
                    .unwrap()
                    .scroll(size.0, size.1, key == b'B', false);
                dirty = true;
            }
            Some(Input::Page(forward))
                if diff_visible && diff.as_ref().is_some_and(|diff| diff.detail_open()) =>
            {
                diff.as_mut().unwrap().scroll(size.0, size.1, forward, true);
                dirty = true;
            }
            Some(Input::Key(_) | Input::Arrow(_) | Input::Pointer(_))
                if diff_visible && diff.as_ref().is_some_and(|diff| diff.detail_open()) => {}
            Some(Input::Key(key @ (b'n' | b'p'))) if diff_visible => {
                diff.as_mut()
                    .unwrap()
                    .scroll(size.0, size.1, key == b'n', true);
                dirty = true;
            }
            Some(Input::Arrow(key @ (b'A' | b'B'))) if diff_visible => {
                diff.as_mut()
                    .unwrap()
                    .scroll(size.0, size.1, key == b'B', false);
                dirty = true;
            }
            Some(Input::Arrow(key @ (b'C' | b'D'))) if diff_visible => {
                diff.as_mut()
                    .unwrap()
                    .scroll_horizontal(size.0, key == b'C');
                dirty = true;
            }
            Some(Input::Key(key @ (b'[' | b']'))) if diff_visible => {
                diff.as_mut()
                    .unwrap()
                    .scroll_horizontal(size.0, key == b']');
                dirty = true;
            }
            Some(Input::Page(forward)) if diff_visible => {
                diff.as_mut().unwrap().scroll(size.0, size.1, forward, true);
                dirty = true;
            }
            Some(Input::Escape | Input::Key(b'D')) if diff_visible => diff_visible = false,
            Some(Input::Key(b'\t')) if diff_visible => {
                diff.as_mut().unwrap().change_side();
                dirty = true;
            }
            Some(Input::Key(b'v' | b'V')) if diff_visible => {
                diff.as_mut().unwrap().toggle_details();
                dirty = true;
            }
            Some(Input::Key(b'g' | b'G')) if diff_visible => {
                diff.as_mut().unwrap().toggle_diagram();
                dirty = true;
            }
            Some(Input::Key(_) | Input::Arrow(_) | Input::Page(_) | Input::Pointer(_))
                if diff_visible => {}
            Some(Input::Escape | Input::Key(b'v' | b'V'))
                if failure_visible && failure.as_ref().is_some_and(|path| path.report_open()) =>
            {
                failure
                    .as_mut()
                    .unwrap()
                    .toggle_report(view, offline.as_deref());
                dirty = true;
            }
            Some(Input::Arrow(key @ (b'A' | b'B')))
                if failure_visible && failure.as_ref().is_some_and(|path| path.report_open()) =>
            {
                failure.as_mut().unwrap().scroll_report(
                    view,
                    offline.as_deref(),
                    size.0,
                    size.1,
                    locale,
                    key == b'B',
                    false,
                );
                dirty = true;
            }
            Some(Input::Page(forward))
                if failure_visible && failure.as_ref().is_some_and(|path| path.report_open()) =>
            {
                failure.as_mut().unwrap().scroll_report(
                    view,
                    offline.as_deref(),
                    size.0,
                    size.1,
                    locale,
                    forward,
                    true,
                );
                dirty = true;
            }
            Some(Input::Escape | Input::Key(b'b' | b'B' | b'c' | b'C'))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                failure.as_mut().unwrap().close_consumers();
                dirty = true;
            }
            Some(Input::Arrow(key @ (b'A' | b'B')))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                failure
                    .as_mut()
                    .unwrap()
                    .move_consumer(key == b'B', false, size.1);
                dirty = true;
            }
            Some(Input::Page(forward))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                failure
                    .as_mut()
                    .unwrap()
                    .move_consumer(forward, true, size.1);
                dirty = true;
            }
            Some(Input::Key(b'\r' | b'\n'))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                if failure_stack.len() < 128
                    && let Some(next) = failure.as_ref().and_then(|path| path.open_consumer(view))
                {
                    failure_stack.push(failure.replace(next).unwrap());
                }
                dirty = true;
            }
            Some(Input::Key(b'g' | b'G'))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                failure.as_mut().unwrap().toggle_consumer_layout();
                dirty = true;
            }
            Some(Input::Pointer(Mouse::Press(point)))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                consumer_press =
                    failure
                        .as_ref()
                        .unwrap()
                        .consumer_at(point.column, point.row, size.0, size.1);
            }
            Some(Input::Pointer(Mouse::Drag(point) | Mouse::Move(point)))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                if consumer_press.is_some()
                    && failure.as_ref().unwrap().consumer_at(
                        point.column,
                        point.row,
                        size.0,
                        size.1,
                    ) != consumer_press
                {
                    consumer_press = None;
                }
            }
            Some(Input::Pointer(Mouse::Release(point)))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                let pressed = consumer_press.take();
                let released =
                    failure
                        .as_ref()
                        .unwrap()
                        .consumer_at(point.column, point.row, size.0, size.1);
                if let Some(index) = pressed.filter(|index| Some(*index) == released) {
                    failure.as_mut().unwrap().select_consumer(index);
                    dirty = true;
                }
            }
            Some(Input::Pointer(Mouse::Zoom(_, inward)))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                consumer_press = None;
                failure
                    .as_mut()
                    .unwrap()
                    .move_consumer(!inward, false, size.1);
                dirty = true;
            }
            Some(Input::Pointer(Mouse::Page(_, forward)))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) =>
            {
                consumer_press = None;
                failure
                    .as_mut()
                    .unwrap()
                    .move_consumer(forward, true, size.1);
                dirty = true;
            }
            Some(Input::Key(key))
                if key != b'F'
                    && failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) => {}
            Some(Input::Arrow(_) | Input::Pointer(_))
                if failure_visible
                    && failure.as_ref().is_some_and(|path| path.consumers_open()) => {}
            Some(Input::Key(b'c' | b'C')) if failure_visible && offline.is_none() => {
                failure.as_mut().unwrap().show_consumers(view);
                dirty = true;
            }
            Some(Input::Key(b'v' | b'V')) if failure_visible => {
                failure
                    .as_mut()
                    .unwrap()
                    .toggle_report(view, offline.as_deref());
                dirty = true;
            }
            Some(Input::Key(b'l'))
                if failure_visible
                    && failure_stack.is_empty()
                    && offline.is_none()
                    && !failure.as_ref().is_some_and(|path| path.report_open())
                    && view
                        .live_evidence
                        .as_ref()
                        .is_some_and(|evidence| evidence.from_calls) =>
            {
                if let (Some(evidence), Some(path)) = (&mut view.live_evidence, &failure) {
                    evidence.selected_input = Some(path.selected_input());
                    evidence.show_input = true;
                }
                return Ok(Exit::Quit);
            }
            Some(Input::Key(_) | Input::Arrow(_) | Input::Pointer(_))
                if failure_visible && failure.as_ref().is_some_and(|path| path.report_open()) => {}
            Some(Input::Escape | Input::Key(b'F')) if failure_visible => {
                if offline.is_none()
                    && let Some(path) = &failure
                {
                    project_failure_matches(view, path, &failure_stack);
                    projection_pending = true;
                    focus_failure_target(view, path);
                }
                failure_visible = false;
                if let Some(evidence) = &mut view.live_evidence {
                    evidence.show_input = false;
                }
            }
            Some(Input::Arrow(key @ (b'A' | b'B'))) if failure_visible => {
                failure.as_mut().unwrap().movement(key == b'B');
                if failure_stack.is_empty()
                    && let Some(evidence) = &mut view.live_evidence
                {
                    evidence.selected_input = Some(failure.as_ref().unwrap().selected_input());
                }
                dirty = true;
            }
            Some(Input::Arrow(key @ (b'C' | b'D'))) if failure_visible => {
                failure.as_mut().unwrap().move_match(key == b'C');
                dirty = true;
            }
            Some(Input::Key(b'b' | b'B')) if failure_visible => {
                if let Some(previous) = failure_stack.pop() {
                    failure = Some(previous);
                } else {
                    if offline.is_none()
                        && let Some(path) = &failure
                    {
                        project_failure_matches(view, path, &failure_stack);
                        projection_pending = true;
                        focus_failure_target(view, path);
                    }
                    failure_visible = false;
                    if let Some(evidence) = &mut view.live_evidence {
                        evidence.show_input = false;
                    }
                }
                dirty = true;
            }
            Some(Input::Key(b'\r' | b'\n')) if failure_visible => {
                if let Some(next) = failure.as_ref().and_then(|path| path.follow_match(view)) {
                    if failure_stack.len() >= 128 {
                        failure.as_mut().unwrap().mark_navigation_limit();
                    } else {
                        failure_stack.push(failure.replace(next).unwrap());
                    }
                    dirty = true;
                    continue;
                }
                if let Some(wire) = failure.as_ref().and_then(|path| path.wire()) {
                    if view.document.source(wire.to) == Some(wire.from) {
                        select_matrix_wire(view, wire, wire);
                        details_focused = false;
                    }
                } else if offline.is_none()
                    && let Some(path) = &failure
                {
                    project_failure_matches(view, path, &failure_stack);
                    projection_pending = true;
                    focus_failure_target(view, path);
                }
                failure_visible = false;
                if let Some(evidence) = &mut view.live_evidence {
                    evidence.show_input = false;
                }
            }
            Some(Input::Key(_) | Input::Arrow(_) | Input::Page(_) | Input::Pointer(_))
                if failure_visible => {}
            Some(Input::Escape) if search.is_some() => search = None,
            Some(Input::Arrow(key @ (b'A' | b'B'))) if search.is_some() => {
                search.as_mut().unwrap().movement(view, key == b'B');
            }
            Some(Input::Key(b'\r' | b'\n')) if search.is_some() => {
                if let Some(target) = search.as_ref().and_then(|search| search.selected(view)) {
                    use super::search::{Action, Target};
                    match target {
                        Target::EvidenceInput(index) => {
                            if let Some(mut path) = super::failure::Path::live(view) {
                                view.last_search = search.clone();
                                failure_stack.clear();
                                path.select_input(index);
                                if let Some(evidence) = &mut view.live_evidence {
                                    evidence.selected_input = Some(index);
                                    evidence.show_input = true;
                                }
                                failure = Some(path);
                                failure_visible = true;
                                search = None;
                                pointer = Pointer::default();
                            }
                        }
                        Target::Diagnostic(code) => {
                            if let Some(mut path) = super::failure::Path::live(view) {
                                view.last_search = search.clone();
                                failure_stack.clear();
                                path.locate_report(view, &code, size.0, size.1, locale);
                                failure = Some(path);
                                failure_visible = true;
                                search = None;
                                pointer = Pointer::default();
                            }
                        }
                        Target::Action(action)
                            if search.as_ref().unwrap().available(view, action) =>
                        {
                            view.last_search = search.clone();
                            let graph_focused = !details_focused;
                            if let Some(details) = offline.as_deref_mut() {
                                details.close_graph_help();
                            }
                            details_focused = false;
                            match action {
                                Action::Flow => view.toggle_flat_flow(),
                                Action::Explosion if view.flat_flow => {
                                    view.select_graph_preset(3);
                                }
                                Action::Explosion => view.cycle_explosion(),
                                Action::Matrix => {
                                    matrix = open_matrix(view, &scene, &previous_matrix);
                                    details_focused = false;
                                }
                                Action::FailurePath => {
                                    failure_stack.clear();
                                    failure_visible = open_failure_path(
                                        view,
                                        offline.as_deref_mut(),
                                        graph_focused,
                                        &mut failure,
                                    );
                                }
                                Action::RepairDiff => {
                                    diff_visible = open_repair_diff(
                                        view,
                                        offline.as_deref(),
                                        locale,
                                        &mut diff,
                                    );
                                }
                                Action::Reset => {
                                    view.reset_spatial_view();
                                    view.module_layout.reset_scope(view.graph_scope());
                                }
                            }
                            search = None;
                            pointer = Pointer::default();
                        }
                        Target::Action(_) => {}
                        _ => {
                            view.last_search = search.clone();
                            if let Some(details) = offline.as_deref_mut() {
                                if details.graph_help_open() {
                                    details.close_graph_help();
                                    details_focused = false;
                                }
                            }
                            super::search::open(view, target);
                            search = None;
                            pointer = Pointer::default();
                        }
                    }
                }
            }
            Some(Input::Key(4)) if search.is_some() => search = None,
            Some(Input::Key(key)) if search.is_some() => search.as_mut().unwrap().input(key),
            Some(Input::Arrow(_) | Input::Page(_) | Input::Pointer(_)) if search.is_some() => {}
            Some(Input::Key(b'v' | b'V'))
                if matrix.is_none()
                    && !failure_visible
                    && !diff_visible
                    && candidates.is_none()
                    && !details_has_focus
                    && editor.is_none()
                    && view.selected_pair.is_some() =>
            {
                pair_details = view.selected_pair.map(|pair| PairDetails {
                    target: DetailTarget::Pair(pair),
                    offset: 0,
                });
                pointer = Pointer::default();
            }
            Some(Input::Key(b'v' | b'V'))
                if matrix.is_none()
                    && search.is_none()
                    && !failure_visible
                    && !diff_visible
                    && candidates.is_none()
                    && !details_has_focus
                    && editor.is_none()
                    && !offline.as_deref().is_some_and(Offline::modal_open)
                    && scene.axes.is_none()
                    && view.selected_pair.is_none()
                    && view.selected_port.is_none()
                    && view.selected_wire.is_none() =>
            {
                pair_details = view.layers.get(view.selected).map(|layer| PairDetails {
                    target: DetailTarget::Module(layer.id),
                    offset: 0,
                });
                pointer = Pointer::default();
            }
            Some(Input::Key(b'q' | b'Q' | 4)) => return Ok(Exit::Quit),
            Some(Input::Key(b'l'))
                if offline.is_none()
                    && view
                        .live_evidence
                        .as_ref()
                        .is_some_and(|evidence| evidence.from_calls) =>
            {
                return Ok(Exit::Quit);
            }
            Some(Input::Escape | Input::Key(b'v' | b'V'))
                if matrix
                    .as_ref()
                    .is_some_and(|matrix| matrix.axis_details_open()) =>
            {
                matrix.as_mut().unwrap().toggle_axis_details();
                dirty = true;
            }
            Some(Input::Arrow(key @ (b'A' | b'B')))
                if matrix
                    .as_ref()
                    .is_some_and(|matrix| matrix.axis_details_open()) =>
            {
                matrix.as_mut().unwrap().scroll_axis_details(
                    view,
                    scene.width,
                    panes.height,
                    locale,
                    key == b'B',
                    false,
                );
                dirty = true;
            }
            Some(Input::Page(forward))
                if matrix
                    .as_ref()
                    .is_some_and(|matrix| matrix.axis_details_open()) =>
            {
                matrix.as_mut().unwrap().scroll_axis_details(
                    view,
                    scene.width,
                    panes.height,
                    locale,
                    forward,
                    true,
                );
                dirty = true;
            }
            Some(Input::Pointer(Mouse::Zoom(_, inward)))
                if matrix
                    .as_ref()
                    .is_some_and(|matrix| matrix.axis_details_open()) =>
            {
                matrix.as_mut().unwrap().scroll_axis_details(
                    view,
                    scene.width,
                    panes.height,
                    locale,
                    !inward,
                    false,
                );
                dirty = true;
            }
            Some(Input::Key(_) | Input::Arrow(_) | Input::Pointer(_))
                if matrix
                    .as_ref()
                    .is_some_and(|matrix| matrix.axis_details_open()) => {}
            Some(Input::Key(b'v' | b'V'))
                if matrix.as_ref().is_some_and(|matrix| matrix.axis_open()) =>
            {
                matrix.as_mut().unwrap().toggle_axis_details();
                matrix_press = None;
                dirty = true;
            }
            Some(Input::Escape | Input::Key(b'a' | b'A'))
                if matrix.as_ref().is_some_and(|matrix| matrix.axis_open()) =>
            {
                matrix.as_mut().unwrap().toggle_axis();
                matrix_press = None;
                dirty = true;
            }
            Some(Input::Key(b'a' | b'A')) if matrix.is_some() => {
                matrix.as_mut().unwrap().toggle_axis();
                matrix_press = None;
                dirty = true;
            }
            Some(Input::Key(b'\t')) if matrix.as_ref().is_some_and(|matrix| matrix.axis_open()) => {
                matrix.as_mut().unwrap().switch_axis();
                matrix_press = None;
                dirty = true;
            }
            Some(Input::Key(b'\r' | b'\n'))
                if matrix.as_ref().is_some_and(|matrix| matrix.axis_open()) =>
            {
                matrix.as_mut().unwrap().confirm_axis();
                dirty = true;
            }
            Some(Input::Escape | Input::Key(b'm' | b'M' | b'b' | b'B')) if matrix.is_some() => {
                let (scope, state) = matrix.take().unwrap().snapshot((
                    view.selected_wire,
                    view.selected_port,
                    view.selected_pair,
                ));
                previous_matrix.insert(scope, state);
            }
            Some(Input::Arrow(key)) if matrix.is_some() => {
                let matrix = matrix.as_mut().unwrap();
                match key {
                    b'A' => matrix.movement(-1, 0),
                    b'B' => matrix.movement(1, 0),
                    b'C' => matrix.movement(0, 1),
                    b'D' => matrix.movement(0, -1),
                    _ => {}
                }
                dirty = true;
            }
            Some(Input::Page(forward)) if matrix.is_some() => {
                matrix
                    .as_mut()
                    .unwrap()
                    .page(scene.width, panes.height, forward);
                dirty = true;
            }
            Some(Input::Key(_)) if matrix.as_ref().is_some_and(|matrix| matrix.axis_open()) => {}
            Some(Input::Key(key @ (b'n' | b'N' | b'p' | b'P'))) if matrix.is_some() => {
                matrix
                    .as_mut()
                    .unwrap()
                    .jump_mismatch(matches!(key, b'n' | b'N'));
                dirty = true;
            }
            Some(Input::Key(b'\r' | b'\n')) if matrix.is_some() => {
                if let Some((wire, displayed)) =
                    matrix.as_ref().and_then(|matrix| matrix.selected_wire())
                {
                    select_matrix_wire(view, wire, displayed);
                } else if let Some(pair) = matrix.as_ref().and_then(|matrix| matrix.selected_pair())
                {
                    select_matrix_pair(view, pair);
                }
                dirty = true;
            }
            Some(Input::Key(b'w' | b'W')) if matrix.is_some() => {
                let matrix = matrix.as_mut().unwrap();
                matrix.cycle();
                if let Some((wire, displayed)) = matrix.selected_wire() {
                    select_matrix_wire(view, wire, displayed);
                }
                dirty = true;
            }
            Some(Input::Key(key @ (b'r' | b'R' | b'c' | b'C' | b'g' | b'G')))
                if matrix.is_some() =>
            {
                matrix.as_mut().unwrap().toggle_groups(
                    matches!(key, b'r' | b'R' | b'g' | b'G'),
                    matches!(key, b'c' | b'C' | b'g' | b'G'),
                );
                dirty = true;
            }
            Some(Input::Pointer(Mouse::Press(point))) if matrix.is_some() => {
                matrix_press = matrix.as_ref().unwrap().cell_at(
                    point.column,
                    point.row,
                    scene.width,
                    panes.height,
                );
            }
            Some(Input::Pointer(Mouse::Drag(point) | Mouse::Move(point))) if matrix.is_some() => {
                if matrix_press.is_some()
                    && matrix.as_ref().unwrap().cell_at(
                        point.column,
                        point.row,
                        scene.width,
                        panes.height,
                    ) != matrix_press
                {
                    matrix_press = None;
                }
            }
            Some(Input::Pointer(Mouse::Release(point))) if matrix.is_some() => {
                let pressed = matrix_press.take();
                let matrix = matrix.as_mut().unwrap();
                if pressed.is_some()
                    && matrix.cell_at(point.column, point.row, scene.width, panes.height) == pressed
                    && matrix.click(point.column, point.row, scene.width, panes.height)
                {
                    if let Some((wire, displayed)) = matrix.selected_wire() {
                        select_matrix_wire(view, wire, displayed);
                    } else if let Some(pair) = matrix.selected_pair() {
                        select_matrix_pair(view, pair);
                    }
                    dirty = true;
                }
            }
            Some(Input::Pointer(Mouse::Zoom(_, inward))) if matrix.is_some() => {
                matrix_press = None;
                matrix
                    .as_mut()
                    .unwrap()
                    .movement(if inward { -1 } else { 1 }, 0);
                dirty = true;
            }
            Some(Input::Key(_) | Input::Page(_) | Input::Pointer(_)) if matrix.is_some() => {}
            Some(Input::Key(b'D'))
                if search.is_none()
                    && !failure_visible
                    && editor.is_none()
                    && !offline.as_deref().is_some_and(Offline::modal_open)
                    && scene.axes.is_none() =>
            {
                if open_repair_diff(view, offline.as_deref(), locale, &mut diff) {
                    diff_visible = true;
                    pointer = Pointer::default();
                }
            }
            Some(Input::Key(b'F'))
                if search.is_none()
                    && editor.is_none()
                    && !offline.as_deref().is_some_and(Offline::modal_open)
                    && scene.axes.is_none() =>
            {
                if !failure_stack.is_empty() && offline.is_none() {
                    failure_visible = true;
                    if let Some(evidence) = &mut view.live_evidence {
                        evidence.show_input = true;
                    }
                    pointer = Pointer::default();
                } else if open_failure_path(
                    view,
                    offline.as_deref_mut(),
                    !details_has_focus,
                    &mut failure,
                ) {
                    failure_visible = true;
                    pointer = Pointer::default();
                }
            }
            Some(Input::Key(b'/')) if matrix.is_none() => {
                let graph_available = editor.is_none()
                    && scene.axes.is_none()
                    && !offline
                        .as_deref()
                        .is_some_and(|details| details.modal_open() && !details.graph_help_open());
                let matrix_available = graph_available
                    && super::matrix::Matrix::new(view, &scene, view.selected_wire).is_some();
                let failure_available =
                    offline.is_some() || super::failure::Path::live(view).is_some();
                let diff_available = offline
                    .as_deref()
                    .and_then(|details| details.repair_diff(view, locale))
                    .is_some();
                let resume_query = view.last_search.is_some();
                let mut restored = view
                    .last_search
                    .clone()
                    .unwrap_or_else(super::search::Search::new)
                    .with_capabilities(graph_available, matrix_available)
                    .with_evidence(failure_available, diff_available);
                restored.refresh_selection(view);
                if resume_query {
                    restored.resume();
                }
                search = Some(restored);
                pointer = Pointer::default();
            }
            Some(Input::Key(b'M'))
                if !details_has_focus && editor.is_none() && scene.axes.is_none() =>
            {
                matrix = open_matrix(view, &scene, &previous_matrix);
                pointer = Pointer::default();
            }
            Some(Input::Key(b'a' | b'A'))
                if offline.is_none()
                    && view.live_evidence.as_ref().is_some_and(|evidence| {
                        view.layers[view.selected].path == evidence.path
                    }) =>
            {
                view.live_axes_zoom = if view.live_axes_zoom.is_some() {
                    None
                } else {
                    Some(8)
                };
            }
            Some(Input::Escape | Input::Key(b'b' | b'B')) if view.live_axes_zoom.is_some() => {
                view.live_axes_zoom = None;
            }
            Some(Input::Key(key @ (b'+' | b'=' | b'-'))) if view.live_axes_zoom.is_some() => {
                let zoom = view.live_axes_zoom.as_mut().unwrap();
                *zoom = if key == b'-' {
                    zoom.saturating_sub(1).max(5)
                } else {
                    zoom.saturating_add(1).min(10)
                };
            }
            Some(Input::Pointer(Mouse::Zoom(_, inward))) if view.live_axes_zoom.is_some() => {
                let zoom = view.live_axes_zoom.as_mut().unwrap();
                *zoom = if inward {
                    zoom.saturating_add(1).min(10)
                } else {
                    zoom.saturating_sub(1).max(5)
                };
            }
            Some(Input::Key(_) | Input::Arrow(_) | Input::Page(_) | Input::Pointer(_))
                if view.live_axes_zoom.is_some() => {}
            Some(Input::Key(key))
                if offline
                    .as_deref()
                    .is_some_and(|details| details.return_to_welcome(key)) =>
            {
                break;
            }
            Some(Input::Key(b'\t')) if offline.is_some() => {
                if let Some(details) = offline
                    .as_deref_mut()
                    .filter(|details| !view.layers.is_empty() && !details.modal_open())
                {
                    details_focused = !details_focused;
                    if details_focused {
                        details.select_graph_call(view);
                    }
                    panel_dirty = true;
                }
            }
            Some(Input::Key(b'L'))
                if (offline.is_none() || !details_has_focus)
                    && !(panes.editor && editor.as_ref().is_some_and(|panel| panel.focused)) =>
            {
                pointer.cancel(view);
                view.layout_edit = !view.layout_edit;
            }
            Some(Input::Key(b'T'))
                if (offline.is_none() || !details_has_focus)
                    && !(panes.editor && editor.as_ref().is_some_and(|panel| panel.focused)) =>
            {
                pointer.cancel(view);
                view.mouse_capture = !view.mouse_capture;
                toolbar.pressed = None;
                output.write_all(if view.mouse_capture {
                    b"\x1b[?1006h\x1b[?1003h"
                } else {
                    b"\x1b[?1003l\x1b[?1006l"
                })?;
                output.flush()?;
            }
            Some(Input::Key(key))
                if details_has_focus
                    && offline.as_deref_mut().is_some_and(|details| {
                        details.open_full_detail(key) || details.input(key, view)
                    }) =>
            {
                panel_dirty = true;
            }
            Some(Input::Key(key))
                if !details_has_focus
                    && offline
                        .as_deref_mut()
                        .is_some_and(|details| details.open_graph_panel(key)) =>
            {
                pointer.press = None;
                panel_dirty = true;
            }
            Some(Input::Key(key)) if offline.is_some() && details_has_focus && key != b' ' => {}
            Some(Input::Key(key @ (b'e' | b'E' | b's' | b'S')))
                if view.edit_handoff
                    && !view.example
                    && editor.is_none()
                    && offline.is_none()
                    && !details_has_focus
                    && matrix.is_none()
                    && search.is_none()
                    && !failure_visible
                    && !diff_visible
                    && candidates.is_none()
                    && pair_details.is_none()
                    && view.selected_port.is_none()
                    && view.selected_pair.is_none()
                    && view.selected_wire.is_none()
                    && view.layers.get(view.selected).is_some() =>
            {
                return Ok(if key == b's' || key == b'S' {
                    Exit::EditSource
                } else {
                    Exit::Edit
                });
            }
            Some(Input::Key(b'e' | b'E')) if view.example => {
                editor = if editor.is_some() {
                    None
                } else {
                    let mut panel = Editor::default();
                    panel.enter(view);
                    Some(panel)
                };
                pointer = Pointer::default();
            }
            Some(Input::Key(b'\t')) if panes.editor => {
                let editor = editor.as_mut().unwrap();
                if editor.focused {
                    editor.focused = false;
                } else {
                    editor.enter(view);
                }
                panel_dirty = true;
            }
            Some(Input::Key(key))
                if panes.editor && editor.as_ref().is_some_and(|panel| panel.focused) =>
            {
                editor.as_mut().unwrap().input(key, view);
                panel_dirty = true;
            }
            Some(Input::Key(b'o' | b'O'))
                if view
                    .live_evidence
                    .as_ref()
                    .and_then(|evidence| evidence.projection.as_ref())
                    .is_some() =>
            {
                if let Some(projection) = view
                    .live_evidence
                    .as_mut()
                    .and_then(|evidence| evidence.projection.as_mut())
                {
                    projection.hidden = !projection.hidden;
                }
                projection_pending = true;
                dirty = true;
            }
            Some(Input::Key(key @ (b'1'..=b'4')))
                if !details_has_focus
                    && !(panes.editor && editor.as_ref().is_some_and(|panel| panel.focused)) =>
            {
                if view.select_graph_preset(key - b'0') {
                    pointer = Pointer::default();
                }
            }
            Some(Input::Key(b'z' | b'Z'))
                if view.flow_layout().is_some()
                    || view.flat_flow
                    || (!view.hierarchy && view.unconnected_flow().is_some()) =>
            {
                view.toggle_flat_flow();
                pointer = Pointer::default();
            }
            Some(Input::Key(b'x' | b'X')) if !view.flat_flow => {
                view.cycle_explosion();
                pointer = Pointer::default();
            }
            Some(Input::Key(b'W')) if pointer.press.is_none() => {
                candidates = super::candidates::Candidates::new(
                    view,
                    &scene,
                    pointer.wire_cell.map(|point| (point.column, point.row)),
                );
                view.hover_handoff = None;
                view.hover = None;
            }
            Some(Input::Key(b'\r' | b'\n')) => view.zoom(view.hover.unwrap_or(view.selected)),
            Some(Input::Key(b'w')) => pointer.advance_wire(view, &scene),
            Some(Input::Key(key @ (b'u' | b'U' | b'd' | b'D'))) => {
                view.follow_wire(
                    &scene,
                    matches!(key, b'd' | b'D'),
                    pointer.wire_cell.map(|point| (point.column, point.row)),
                );
            }
            Some(Input::Key(key @ (b'[' | b']'))) if scene.page_size < scene.item_count => {
                view.page(&scene, key == b']');
            }
            Some(Input::Key(key @ (b'+' | b'=' | b'-'))) => {
                viewport_zoom = true;
                let factor = if key == b'-' { 1.0 / 1.10 } else { 1.10 };
                view.viewport.zoom(
                    factor,
                    [scene.width as f64 / 2.0, scene.labels.len() as f64],
                );
                view.hover = None;
            }
            Some(Input::Key(b'r' | b'R')) => {
                view.reset_spatial_view();
                view.module_layout.reset_scope(view.graph_scope());
                pointer = Pointer::default();
            }
            Some(Input::Key(b' ')) => {
                motion_enabled = handoff.toggle_motion();
                if !motion_enabled {
                    confirmation = None;
                    if let Some(EditOverlay::Confirmed { expanded, .. }) = &mut view.edit_overlay {
                        *expanded = false;
                    }
                }
            }
            Some(Input::Key(b'\t' | b'h' | b'H')) => {
                view.toggle_hierarchy();
                pointer = Pointer::default();
            }
            Some(Input::Escape | Input::Key(b'b' | b'B')) => {
                if details_has_focus
                    && offline
                        .as_deref_mut()
                        .is_some_and(|details| details.cancel(view))
                {
                    panel_dirty = true;
                } else if let Some(panel) = editor
                    .as_mut()
                    .filter(|panel| panes.editor && panel.focused)
                {
                    if !panel.cancel() {
                        editor = None;
                    }
                    panel_dirty = true;
                } else if !view.zoom_out() {
                    match offline {
                        Some(_) if !panes.details => details_focused = true,
                        Some(_) => break,
                        None => return Ok(Exit::Quit),
                    }
                }
                pointer = Pointer::default();
            }
            Some(Input::Arrow(b'A' | b'B') | Input::Page(_)) => {
                let forward = matches!(event, Some(Input::Arrow(b'B') | Input::Page(true)));
                let by_page = matches!(event, Some(Input::Page(_)));
                if let Some(details) = offline.as_deref_mut().filter(|_| details_has_focus) {
                    if !details.scroll_full_detail(forward, by_page, locale, size.0, size.1) {
                        details.navigate(forward, view);
                    }
                    panel_dirty = true;
                } else if !(panes.editor && editor.as_ref().is_some_and(|panel| panel.focused)) {
                    if by_page {
                        view.page(&scene, forward);
                    } else {
                        view.navigate(if forward { 1 } else { -1 });
                    }
                }
            }
            Some(Input::Arrow(key @ (b'C' | b'D')))
                if !(details_has_focus && offline.is_some()
                    || panes.editor && editor.as_ref().is_some_and(|panel| panel.focused)) =>
            {
                if view.flat_flow {
                    view.viewport.offset[0] += if key == b'C' { -2.0 } else { 2.0 };
                } else {
                    view.camera.orbit(if key == b'C' { 0.2 } else { -0.2 }, 0.0);
                }
            }
            Some(Input::Focus(in_focus)) => {
                if in_focus && !focused && query_background {
                    output.write_all(b"\x1b]11;?\x1b\\")?;
                    output.flush()?;
                }
                focused = in_focus;
                if !focused {
                    toolbar.pressed = None;
                    candidate_press = None;
                    consumer_press = None;
                    matrix_press = None;
                    pointer.cancel(view);
                }
            }
            Some(Input::Background(background)) if query_background => {
                view.background = background;
            }
            Some(Input::Pointer(_)) if !view.mouse_capture => {}
            Some(Input::Pointer(mut mouse)) => {
                if toolbar.track(mouse, scene.width) {
                    dirty = true;
                }
                let handled = if panes.graph_rows > 0
                    && scene.axes.is_none()
                    && !offline.as_deref().is_some_and(Offline::modal_open)
                {
                    toolbar.input(view, &scene, mouse)
                } else {
                    None
                };
                if let Some(zoom) = handled {
                    viewport_zoom = zoom;
                    details_focused = false;
                    if let Some(panel) = editor.as_mut() {
                        panel.focused = false;
                    }
                    pointer = Pointer::default();
                    dirty = true;
                } else {
                    let mut dispatch = |mouse| {
                        if let Some(details) = offline.as_deref_mut().filter(|_| panes.full_detail)
                        {
                            let (forward, by_page) = match mouse {
                                Mouse::Page(_, forward) => (forward, true),
                                Mouse::Zoom(_, inward) => (!inward, false),
                                _ => return false,
                            };
                            return details
                                .scroll_full_detail(forward, by_page, locale, size.0, size.1);
                        }
                        let panel = if let Some(details) =
                            offline.as_deref_mut().filter(|_| panes.details)
                        {
                            SidePanel::Details(details, &mut details_focused)
                        } else if let Some(editor) = editor.as_mut().filter(|_| panes.editor) {
                            SidePanel::Editor(editor)
                        } else {
                            SidePanel::None
                        };
                        route_pointer(&mut pointer, view, &scene, mouse, panel, panes)
                    };
                    let graph_bottom = panes.graph_rows;
                    let class = motion_class(mouse, graph_bottom);
                    viewport_zoom = class == Some(MotionClass::Zoom);
                    let burst_started = Instant::now();
                    for step in 0..16 {
                        if class != Some(MotionClass::Hover) {
                            panel_dirty |= dispatch(mouse);
                        }
                        if class.is_none()
                            || step == 15
                            || burst_started.elapsed() >= Duration::from_millis(2)
                        {
                            break;
                        }
                        match input.read(&terminal, 0)? {
                            Some(Input::Pointer(next))
                                if motion_class(next, graph_bottom) == class =>
                            {
                                mouse = next;
                            }
                            Some(barrier) => {
                                input.pending.push_front(barrier);
                                break;
                            }
                            None => break,
                        }
                    }
                    if class == Some(MotionClass::Hover) {
                        panel_dirty |= dispatch(mouse);
                    }
                }
            }
            _ => {}
        }
        let next_size = (layout::viewport_width(), menu::rows());
        let next_panes = Panes::new(
            next_size.1,
            offline.as_deref(),
            !view.layers.is_empty(),
            editor.is_some(),
            details_focused || offline.as_deref().is_some_and(Offline::modal_open),
        );
        let panes_changed = panes != next_panes;
        if size != next_size {
            if let Some(details) = &mut pair_details {
                details.resize(view, size, next_size, locale);
            }
        }
        if size != next_size || panes_changed {
            matrix_press = None;
            candidate_press = None;
            consumer_press = None;
        }
        if !old_failure_visible
            && !failure_visible
            && (old.0 != view.selected || old.1 != view.focus)
            && let Some(evidence) = &mut view.live_evidence
        {
            evidence.graph_target_unresolved = false;
        }
        if old_selected_pair.is_some()
            && old_selected_pair == view.selected_pair
            && (old.0 != view.selected
                || old.1 != view.focus
                || old_selected_wire != view.selected_wire
                || old_selected_port != view.selected_port)
        {
            view.selected_pair = None;
        }
        let committed = old_revision != view.document.revision();
        if committed {
            let target = offline
                .as_deref()
                .and_then(Offline::confirmed_target)
                .or_else(|| editor.as_ref().and_then(Editor::confirmed_target));
            view.edit_overlay = target.map(|target| EditOverlay::Confirmed {
                target,
                expanded: motion_enabled,
            });
            confirmation = None;
        } else if let Some(draft) = offline
            .as_deref()
            .and_then(|details| details.graph_overlay(view))
        {
            view.edit_overlay = Some(draft);
            confirmation = None;
        } else if matches!(
            view.edit_overlay,
            Some(EditOverlay::Draft { .. } | EditOverlay::Evidence { .. })
        ) || view.selected_wire != old_selected_wire
            || (old_panel && editor.is_none())
        {
            view.edit_overlay = None;
            confirmation = None;
        }
        let overlay_changed = view.edit_overlay != old_overlay;
        let axes_changed = scene.axes.as_ref().map(|drawing| drawing.comparison)
            != offline
                .as_deref()
                .and_then(Offline::axis_view)
                .or_else(|| view.live_axis_view());
        // Navigation needs the new layout before the next input. Zoom and drag
        // can keep batching against the frame that's currently visible.
        let search_changed = old_search != search;
        let publish_now = committed
            || search_changed
            || old_selected_pair != view.selected_pair
            || old_selected_port != view.selected_port
            || old_failure_visible != failure_visible
            || old_diff_visible != diff_visible
            || old_candidates_open != candidates.is_some()
            || old_pair_details_open != pair_details.is_some()
            || overlay_changed
            || old_matrix_open != matrix.is_some()
            || old_flat_flow != view.flat_flow
            || old_explosion != view.explosion
            || axes_changed
            || panes_changed
            || old_panel != editor.is_some()
            || size != next_size
            || (old.0, old.1, old.4) != (view.selected, view.focus, view.hierarchy);
        let zoomed = viewport_zoom && view.viewport != old.3;
        let now = Instant::now();
        if publish_now || (was_dragging && !pointer.dragging()) {
            zoom_refine_after = None;
        } else if zoomed {
            if zoom_refine_after.is_none() {
                next_frame = now;
            }
            zoom_refine_after = Some(now + ZOOM_REFINE_DELAY);
            motion.set_running(now, false);
        }
        if axes_changed
            || panes_changed
            || size != next_size
            || committed
            || (old.1, old.4) != (view.focus, view.hierarchy)
        {
            pointer = Pointer::default();
            view.hover = None;
        }
        let projection_changed = search_changed
            || old_selected_pair != view.selected_pair
            || old_selected_port != view.selected_port
            || old_failure_visible != failure_visible
            || old_diff_visible != diff_visible
            || old_candidates_open != candidates.is_some()
            || old_pair_details_open != pair_details.is_some()
            || size != next_size
            || panes_changed
            || old_matrix_open != matrix.is_some()
            || old_flat_flow != view.flat_flow
            || old_explosion != view.explosion
            || old_panel != editor.is_some()
            || committed
            || old_layout != view.module_layout.generation
            || old
                != (
                    view.selected,
                    view.focus,
                    view.camera,
                    view.viewport,
                    view.hierarchy,
                );
        let appearance_changed = old_appearance != (view.style, view.background);
        let scene_changed =
            projection_changed || appearance_changed || overlay_changed || axes_changed;
        let ink_changed = !pixel_mode
            && (old_hover != view.hover
                || old_handoff != view.hover_handoff
                || old_selected_wire != view.selected_wire);
        if scene_changed || ink_changed {
            size = next_size;
            panes = next_panes;
            if projection_changed && !committed && old_candidates_open == candidates.is_some() {
                view.hover_handoff = None;
                pointer.wire_cell = None;
                pointer.last_wire_click = None;
            }
            projection_pending = true;
            if publish_now || (was_dragging && !pointer.dragging()) {
                // Publish this revision before reading the next confirmation.
                next_frame = now;
            }
        }
        if scene_changed || (was_dragging && !pointer.dragging()) {
            refine_after = Instant::now() + Duration::from_millis(16);
        }
        dirty |= scene_changed
            || old_input_modes != (view.layout_edit, view.mouse_capture)
            || view.source_error != old_source_error
            || old_hover != view.hover
            || old_handoff != view.hover_handoff
            || old_selected_wire != view.selected_wire
            || (was_dragging && pixel_mode && !pointer.dragging());
    }
    view.hover = None;
    view.hover_handoff = None;
    if confirmation.is_some() {
        view.edit_overlay = None;
    }
    if let Some(display) = &mut display {
        // The welcome frame completes this update after the old graphics are removed.
        output.write_all(b"\x1b[?2026h")?;
        display.clear(&mut output)?;
        output.flush()?;
    }
    Ok(Exit::Welcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edit, Handoff, SCENE_HEADER, Segment, wire};

    #[test]
    fn module_details_preserve_unicode_identity_and_reading_position() {
        let path = format!("model.{}output", "编码器 segment_".repeat(12));
        let aliases: Vec<_> = (0..66).map(|index| format!("alias.{index}")).collect();
        let modules = vec![
            serde_json::json!({"path":path,"type":"Linear", "instance_id":"instance-7", "parameter_count":128,
            "source":"model.py:19", "aliases":aliases}),
        ];
        let mut view = ModelView::new(Some(&modules));
        view.camera.orbit(0.3, 0.1);
        let camera = view.camera;
        let viewport = view.viewport;
        let id = view.layers[view.selected].id;
        let mut details = PairDetails {
            target: DetailTarget::Module(id),
            offset: 0,
        };
        let lines = details.lines(&view, 30, Locale::English);
        let joined = lines.join("");
        assert!(joined.contains(&path), "{joined}");
        assert!(
            joined.contains(&path.replace('.', " › ")),
            "hierarchy breadcrumb missing: {joined}"
        );
        for value in [
            "Linear",
            "Surface texture cue",
            "Medium Braille grain",
            "instance-7",
            "128",
            "model.py:19",
            "alias.63",
            "Additional aliases omitted: 2",
        ] {
            assert!(joined.contains(value), "missing {value}: {joined}");
        }
        assert!(!joined.contains("alias.64"));
        for locale in [Locale::English, Locale::Simplified, Locale::Traditional] {
            let localized = details.lines(&view, 30, locale).join("");
            assert!(
                localized.contains(locale.text("Hierarchy", "层级位置", "層級位置")),
                "{localized}"
            );
            assert!(
                localized.contains(locale.text(
                    "Surface texture cue",
                    "表面纹理提示",
                    "表面紋理提示"
                )),
                "{localized}"
            );
            assert!(
                localized.contains(locale.text(
                    "Medium Braille grain",
                    "中等盲文颗粒",
                    "中等盲文顆粒"
                )),
                "{localized}"
            );
        }
        for _ in 0..4 {
            details.scroll(&view, 30, 8, Locale::English, true, false);
        }
        let old = details.wrapped(&view, 30, Locale::English)[details.offset].clone();
        details.resize(&view, (30, 8), (50, 8), Locale::English);
        let new = details.wrapped(&view, 50, Locale::English)[details.offset].clone();
        assert_eq!(old.1, new.1);
        assert!(new.2 <= old.2);
        for locale in [Locale::English, Locale::Simplified, Locale::Traditional] {
            let frame = details.render(&view, 30, 8, locale);
            assert!(frame.lines().count() <= 8, "{frame}");
            assert!(
                frame
                    .lines()
                    .all(|line| crate::terminal::text::display_width(line) <= 30)
            );
        }
        assert!(view.camera == camera);
        assert!(view.viewport == viewport);
        assert_eq!(view.layers[view.selected].id, id);
    }

    #[test]
    fn pair_evidence_wraps_and_scrolls_full_unicode_paths() {
        let first = format!("model.{}source", "编码器segment_".repeat(10));
        let second = format!("model.{}target", "解码器segment_".repeat(10));
        let modules = vec![
            serde_json::json!({"path":"model","type":"Sequential"}),
            serde_json::json!({"path":first,"type":"Linear"}),
            serde_json::json!({"path":second,"type":"Linear"}),
        ];
        let view = ModelView::new(Some(&modules));
        let first_id = view
            .layers
            .iter()
            .find(|layer| layer.path == first)
            .unwrap()
            .id;
        let second_id = view
            .layers
            .iter()
            .find(|layer| layer.path == second)
            .unwrap()
            .id;
        let pair = super::super::Wire {
            from: super::super::Outlet {
                operator: first_id,
                slot: 0,
            },
            to: super::super::Inlet {
                operator: second_id,
                slot: 0,
            },
        };
        let mut details = PairDetails {
            target: DetailTarget::Pair(pair),
            offset: 0,
        };
        let lines = details.lines(&view, 40, Locale::English);
        let all = lines.join("");
        assert!(all.contains(&first) && all.contains(&second), "{all}");
        let mut seen = String::new();
        for _ in 0..=lines.len() {
            let page = details.render(&view, 40, 10, Locale::English);
            assert!(page.lines().count() <= 10);
            assert!(
                page.lines()
                    .all(|line| crate::terminal::text::display_width(line) <= 40)
            );
            seen.push_str(&page);
            details.scroll(&view, 40, 10, Locale::English, true, true);
        }
        assert!(lines.iter().all(|line| seen.contains(line)), "{seen}");
        assert!(seen.contains("source.out[0]"));
        assert!(seen.contains("target.in[0]"));
        details.scroll(&view, 40, 10, Locale::English, false, true);
        assert!(details.offset < lines.len().saturating_sub(7));
        details.offset = 5;
        let before = details.wrapped(&view, 40, Locale::English)[details.offset].clone();
        details.resize(&view, (40, 10), (70, 10), Locale::English);
        let after = details.wrapped(&view, 70, Locale::English)[details.offset].clone();
        assert_eq!(before.1, after.1);
        assert!(after.2 <= before.2);
        assert!(after.2 + after.0.len() > before.2);
    }

    #[test]
    fn selecting_a_matrix_pair_never_invents_a_wire_and_survives_view_history() {
        let offline = Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts);
        let mut view = offline.view();
        let pair = *view.disconnected.iter().next().unwrap();
        let before = view.document.revision();
        select_matrix_pair(&mut view, pair);
        assert_eq!(view.selected_pair, Some(pair));
        assert_eq!(view.selected_wire, None);
        assert_eq!(view.document.revision(), before);
        let scene = view.scene(100, 30, false, None);
        let screen = view.render(&scene, Locale::English, false, false);
        assert!(screen.contains("Port pair · Declared no wire"), "{screen}");
        let matrix = open_matrix(&view, &scene, &HashMap::new()).unwrap();
        assert_eq!(matrix.selected_pair(), Some(pair));
        let bookmark = view.bookmark();
        view.follow_wire(&scene, false, None);
        assert_eq!(
            view.selected_port,
            Some(super::super::Port::Output(pair.from))
        );
        assert_eq!(view.selected_wire, None);
        assert!(view.zoom_out());
        assert_eq!(view.selected_pair, Some(pair));
        let scene = view.scene(100, 30, false, None);
        view.follow_wire(&scene, true, None);
        assert_eq!(view.selected_port, Some(super::super::Port::Input(pair.to)));
        assert_eq!(view.selected_wire, None);
        assert!(view.zoom_out());
        assert_eq!(view.selected_pair, Some(pair));
        view.selected_pair = None;
        view.restore_view(bookmark);
        assert_eq!(view.selected_pair, Some(pair));
        assert_eq!(view.selected_wire, None);
    }

    #[test]
    fn toolbar_captures_release_and_cancels_drag_without_moving_view() {
        let mut view = ModelView::new(None);
        let scene = view.scene(80, 24, false, None);
        let mut toolbar = Toolbar::default();
        let plus = Point { column: 60, row: 0 };
        let outside = Point { column: 20, row: 8 };
        let initial = view.viewport;
        assert_eq!(
            toolbar.input(&mut view, &scene, Mouse::Press(plus)),
            Some(false)
        );
        assert_eq!(
            toolbar.input(&mut view, &scene, Mouse::Drag(outside)),
            Some(false)
        );
        assert_eq!(
            toolbar.input(&mut view, &scene, Mouse::Release(plus)),
            Some(false)
        );
        assert!(view.viewport == initial);
        assert_eq!(
            toolbar.input(&mut view, &scene, Mouse::Press(plus)),
            Some(false)
        );
        assert_eq!(
            toolbar.input(&mut view, &scene, Mouse::Release(plus)),
            Some(true)
        );
        assert!(view.viewport.scale > initial.scale);
        let left = Point { column: 63, row: 0 };
        let original_camera = view.camera;
        toolbar.input(&mut view, &scene, Mouse::Press(left));
        toolbar.input(&mut view, &scene, Mouse::Release(left));
        assert!(view.camera != original_camera);
        let reset = Point { column: 75, row: 0 };
        toolbar.input(&mut view, &scene, Mouse::Press(reset));
        toolbar.input(&mut view, &scene, Mouse::Release(reset));
        assert!(view.camera == Camera::default() && view.viewport == initial);
        let expand = Point { column: 78, row: 0 };
        toolbar.input(&mut view, &scene, Mouse::Press(expand));
        toolbar.input(&mut view, &scene, Mouse::Release(expand));
        assert!(view.focus.is_some());
    }

    #[test]
    fn toolbar_hover_explains_mouse_controls_in_the_active_locale() {
        let width = 40;
        let mut view = ModelView::new(None);
        view.mouse_capture = true;
        let budget = width - super::super::GRAPH_TOOLBAR.len();
        let labels = [
            ("Zoom out", "缩小", "縮小"),
            ("Zoom in", "放大", "放大"),
            ("Orbit left", "向左旋转", "向左旋轉"),
            ("Orbit right", "向右旋转", "向右旋轉"),
            ("Orbit up", "向上旋转", "向上旋轉"),
            ("Orbit down", "向下旋转", "向下旋轉"),
            ("Reset view", "重置视图", "重設視圖"),
            ("Focus selected", "定位所选层", "定位所選層"),
        ];
        let locales = [
            (Locale::English, 0),
            (Locale::Simplified, 1),
            (Locale::Traditional, 2),
        ];
        for (index, label) in labels.iter().enumerate() {
            let mut toolbar = Toolbar::default();
            let point = Point {
                column: width - super::super::GRAPH_TOOLBAR.len() + index * 3 + 1,
                row: 0,
            };
            assert!(toolbar.track(Mouse::Move(point), width));
            assert!(!toolbar.track(Mouse::Move(point), width));
            for (locale, language) in locales {
                let mode_line = match locale {
                    Locale::English => "L Layout off · T Mouse on · controls",
                    Locale::Simplified => "L 布局关 · T 鼠标开 · 操作",
                    Locale::Traditional => "L 佈局關 · T 滑鼠開 · 操作",
                };
                let mut frame = format!(
                    "{:budget$}{}\n{mode_line}\n",
                    "MODEL VIEW",
                    super::super::GRAPH_TOOLBAR
                );
                let original_title = frame.lines().next().unwrap().to_owned();
                let line_count = frame.lines().count();
                toolbar.decorate(&mut frame, &view, width, locale);
                assert_eq!(frame.lines().next(), Some(original_title.as_str()));
                assert_eq!(frame.lines().count(), line_count);
                assert!(
                    frame.contains([label.0, label.1, label.2][language]),
                    "{frame}"
                );
            }
        }

        let mut toolbar = Toolbar::default();
        let point = Point { column: 0, row: 5 };
        assert!(toolbar.track(Mouse::Move(Point { column: 20, row: 0 }), width));
        assert!(toolbar.track(Mouse::Drag(point), width));
        let mut frame = format!(
            "{:budget$}{}\nL Layout off · T Mouse on · controls\n",
            "MODEL VIEW",
            super::super::GRAPH_TOOLBAR
        );
        toolbar.decorate(&mut frame, &view, width, Locale::English);
        assert!(
            frame.ends_with("L Layout off · T Mouse on · controls\n"),
            "{frame}"
        );
    }

    #[test]
    fn preset_toolbar_hover_explains_modes_in_all_locales() {
        let width = 100;
        let labels = [
            ("Spatial 3D", "空间 3D", "空間 3D"),
            ("2D flow", "二维流图", "二維流圖"),
            ("Exploded x1", "爆炸图 x1", "爆炸圖 x1"),
            ("Exploded x2", "爆炸图 x2", "爆炸圖 x2"),
        ];
        let locales = [
            (Locale::English, 0),
            (Locale::Simplified, 1),
            (Locale::Traditional, 2),
        ];
        let mut view = ModelView::new(None);
        view.mouse_capture = true;
        for (index, label) in labels.iter().enumerate() {
            let mut toolbar = Toolbar::default();
            let column = width - view.toolbar_text(width).len() + (index + 8) * 3 + 1;
            assert!(toolbar.track(Mouse::Move(Point { column, row: 0 }), width));
            for (locale, language) in locales {
                let toolbar_text = view.toolbar_text(width);
                let title = format!("{:>width$}", format!("MODEL VIEW{toolbar_text}"));
                let mut frame = format!("{title}\nL Layout off · T Mouse on · controls\n");
                toolbar.decorate(&mut frame, &view, width, locale);
                assert!(
                    frame.contains([label.0, label.1, label.2][language]),
                    "{frame}"
                );
            }
        }

        let modules = [serde_json::json!({"path":"model","type":"Linear"})];
        let mut hierarchy = ModelView::new(Some(&modules));
        hierarchy.hierarchy = true;
        hierarchy.mouse_capture = true;
        let toolbar_text = hierarchy.toolbar_text(width);
        assert!(toolbar_text.contains("[·]"));
        let column = width - toolbar_text.len() + 9 * 3 + 1;
        let mut toolbar = Toolbar::default();
        assert!(toolbar.track(Mouse::Move(Point { column, row: 0 }), width));
        for (locale, language) in locales {
            let title = format!("{:>width$}", format!("MODEL VIEW{toolbar_text}"));
            let mut frame = format!("{title}\nL Layout off · T Mouse on · controls\n");
            toolbar.decorate(&mut frame, &hierarchy, width, locale);
            let expected = ["2D flow unavailable", "暂无二维流图", "暫無二維流圖"][language];
            assert!(frame.contains(expected), "{frame}");
        }
    }

    #[test]
    fn wide_toolbar_mouse_buttons_select_each_graph_preset() {
        let width = 100;
        let mut view = ModelView::new(None);
        view.mouse_capture = true;
        let selected = view.layers[view.selected].id;
        let selected_wire = view.document.wires().next().unwrap();
        view.selected_wire = Some(selected_wire);
        let scene = view.scene(width, 30, false, None);
        assert!(view.toolbar_text(width).ends_with("{1}[2][3][4]"));
        assert_eq!(view.toolbar_text(80), super::super::GRAPH_TOOLBAR);

        let mut toolbar = Toolbar::default();
        let click = |index: usize, toolbar: &mut Toolbar, view: &mut ModelView<'_>| {
            let column = width - view.toolbar_text(width).len() + index * 3 + 1;
            let point = Point { column, row: 0 };
            assert_eq!(
                toolbar.input(view, &scene, Mouse::Press(point)),
                Some(false)
            );
            assert_eq!(
                toolbar.input(view, &scene, Mouse::Release(point)),
                Some(false)
            );
        };

        click(11, &mut toolbar, &mut view);
        assert_eq!(view.explosion, 2);
        assert_eq!(view.layers[view.selected].id, selected);
        assert_eq!(view.selected_wire, Some(selected_wire));
        assert!(view.toolbar_text(width).ends_with("[1][2][3]{4}"));
        click(10, &mut toolbar, &mut view);
        assert_eq!(view.explosion, 1);
        assert_eq!(view.layers[view.selected].id, selected);
        assert_eq!(view.selected_wire, Some(selected_wire));
        assert!(view.toolbar_text(width).ends_with("[1][2]{3}[4]"));
        click(8, &mut toolbar, &mut view);
        assert_eq!(view.explosion, 0);
        assert!(!view.flat_flow);
        assert_eq!(view.layers[view.selected].id, selected);
        assert_eq!(view.selected_wire, Some(selected_wire));
        assert!(view.toolbar_text(width).ends_with("{1}[2][3][4]"));
        click(9, &mut toolbar, &mut view);
        assert!(view.flat_flow);
        assert_eq!(view.layers[view.selected].id, selected);
        assert_eq!(view.selected_wire, Some(selected_wire));
        assert!(view.toolbar_text(width).ends_with("[1]{2}[3][4]"));
        assert_eq!(view.layers[view.selected].id, selected);

        let modules = [serde_json::json!({"path":"model","type":"Linear"})];
        let mut hierarchy = ModelView::new(Some(&modules));
        hierarchy.hierarchy = true;
        hierarchy.mouse_capture = true;
        let hierarchy_scene = hierarchy.scene(width, 30, false, None);
        let unavailable_toolbar = hierarchy.toolbar_text(width);
        assert!(unavailable_toolbar.ends_with("{1}[·][3][4]"));
        let mut toolbar = Toolbar::default();
        let disabled = Point {
            column: width - unavailable_toolbar.len() + 9 * 3 + 1,
            row: 0,
        };
        assert_eq!(
            toolbar.input(&mut hierarchy, &hierarchy_scene, Mouse::Press(disabled)),
            Some(false)
        );
        assert_eq!(
            toolbar.input(&mut hierarchy, &hierarchy_scene, Mouse::Release(disabled)),
            Some(false)
        );
        assert!(!hierarchy.flat_flow);
        assert_eq!(hierarchy.toolbar_text(width), unavailable_toolbar);
    }

    #[test]
    fn matrix_selection_moves_to_an_offscreen_wire_and_can_return() {
        let mut modules = vec![serde_json::json!({"path": "model", "type": "Sequential"})];
        modules.extend(
            (0..20).map(
                |index| serde_json::json!({"path": format!("model.{index}"), "type": "Linear"}),
            ),
        );
        let mut view = ModelView::new(Some(&modules));
        let original = view.bookmark();
        let scene = view.scene(40, 24, false, None);
        let remote = view
            .document
            .wires()
            .find(|wire| !scene.handoffs.iter().any(|route| route.wire == *wire))
            .unwrap();
        select_matrix_wire(&mut view, remote, remote);
        assert_eq!(view.selected_wire, Some(remote));
        let next = view.scene(40, 24, false, None);
        assert!(next.handoffs.iter().any(|route| route.wire == remote));
        assert!(view.zoom_out());
        assert!(view.bookmark() == original);
    }

    #[test]
    fn module_drag_keeps_capture_when_first_motion_crosses_the_editor_boundary() {
        let mut view = ModelView::new(None);
        view.layout_edit = true;
        let mut editor = Editor::default();
        let panes = Panes::new(24, None, true, true, false);
        let scene = make_scene(&view, None, 80, panes, false, 0.0, Locale::English);
        let start = Point { column: 30, row: 8 };
        let end = Point {
            column: 40,
            row: 21,
        };
        let mut pointer = Pointer::default();
        let original = view.camera;
        let layer = scene.layer_at(start.column, start.row).unwrap();
        let operator = view.layers[layer].id;
        route_pointer(
            &mut pointer,
            &mut view,
            &scene,
            Mouse::Press(start),
            SidePanel::Editor(&mut editor),
            panes,
        );
        assert!(!editor.focused);
        route_pointer(
            &mut pointer,
            &mut view,
            &scene,
            Mouse::Drag(end),
            SidePanel::Editor(&mut editor),
            panes,
        );
        assert!(pointer.dragging());
        assert!(view.camera == original);
        assert_ne!(
            view.module_layout.offset(view.graph_scope(), operator),
            [0.0; 2]
        );
        route_pointer(
            &mut pointer,
            &mut view,
            &scene,
            Mouse::Release(end),
            SidePanel::Editor(&mut editor),
            panes,
        );
        assert!(pointer.press.is_none());
        assert!(!editor.focused);
    }

    #[test]
    fn hover_drag_and_double_click_have_separate_actions() {
        let mut view = ModelView::new(None);
        view.hierarchy = true;
        let scene = view.scene(80, 24, false, None);
        let mut pointer = Pointer::default();
        let initial = view.camera;
        let now = Instant::now();
        let send = |pointer: &mut Pointer, view: &mut ModelView<'_>, data, suffix| {
            pointer.input(view, &scene, parse_mouse(data, suffix).unwrap(), now);
        };
        send(&mut pointer, &mut view, "35;30;10", b'M');
        assert_eq!(view.hover, Some(1));
        assert!(view.camera == initial);
        let empty = Point {
            column: 0,
            row: SCENE_HEADER + scene.tooltip_rows,
        };
        assert!(scene.layer_at(empty.column, empty.row).is_none());
        pointer.input(&mut view, &scene, Mouse::Press(empty), now);
        send(&mut pointer, &mut view, "32;12;11", b'M');
        assert!(view.camera.yaw != initial.yaw && view.camera.pitch != initial.pitch);
        send(&mut pointer, &mut view, "0;12;11", b'm');
        assert!(pointer.press.is_none());
        assert!(view.focus.is_none());

        for _ in 0..2 {
            send(&mut pointer, &mut view, "0;30;10", b'M');
            send(&mut pointer, &mut view, "0;30;10", b'm');
        }
        assert_eq!(view.focus, Some(1));
        assert!(view.zoom_out());
        assert!(view.focus.is_none());
    }

    #[test]
    fn module_drag_requires_layout_mode_and_small_jitter_remains_a_click() {
        let mut view = ModelView::new(None);
        let scene = view.scene(80, 24, false, None);
        let top = SCENE_HEADER + scene.tooltip_rows;
        let start = (top..top + scene.labels.len() - 1)
            .flat_map(|row| (0..scene.width - 8).map(move |column| Point { row, column }))
            .find(|point| {
                scene.layer_at(point.column, point.row).is_some()
                    && scene.layer_at(point.column + 1, point.row)
                        == scene.layer_at(point.column, point.row)
                    && scene.handoffs_at(point.column, point.row).is_empty()
                    && scene.handoffs_at(point.column + 1, point.row).is_empty()
            })
            .unwrap();
        let layer = scene.layer_at(start.column, start.row).unwrap();
        let id = view.layers[layer].id;
        let scope = view.graph_scope();
        let camera = view.camera;
        let mut pointer = Pointer::default();
        let now = Instant::now();
        let end = Point {
            column: start.column + 5,
            row: start.row,
        };
        pointer.input(&mut view, &scene, Mouse::Press(start), now);
        pointer.input(&mut view, &scene, Mouse::Drag(end), now);
        pointer.input(&mut view, &scene, Mouse::Release(end), now);
        assert_eq!(view.module_layout.offset(scope, id), [0.0; 2]);
        assert!(view.camera == camera);
        assert!(pointer.press.is_none());
        let jitter = Point {
            column: start.column + 1,
            row: start.row,
        };
        pointer.input(&mut view, &scene, Mouse::Press(start), now);
        pointer.input(&mut view, &scene, Mouse::Drag(jitter), now);
        assert!(!pointer.dragging());
        pointer.input(&mut view, &scene, Mouse::Release(jitter), now);
        assert_eq!(view.selected, layer);
        assert_eq!(view.module_layout.offset(scope, id), [0.0; 2]);
    }

    #[test]
    fn cancelling_drag_restores_layout_and_canvas_without_changing_selection() {
        let mut view = ModelView::new(None);
        view.layout_edit = true;
        let scene = view.scene(80, 24, false, None);
        let start = (SCENE_HEADER + scene.tooltip_rows
            ..SCENE_HEADER + scene.tooltip_rows + scene.labels.len() - 1)
            .flat_map(|row| (0..scene.width - 8).map(move |column| Point { row, column }))
            .find(|point| {
                scene.layer_at(point.column, point.row).is_some()
                    && scene.handoffs_at(point.column, point.row).is_empty()
            })
            .unwrap();
        let layer = scene.layer_at(start.column, start.row).unwrap();
        let id = view.layers[layer].id;
        let scope = view.graph_scope();
        let before = view.bookmark();
        let now = Instant::now();
        let mut pointer = Pointer::default();
        pointer.input(&mut view, &scene, Mouse::Press(start), now);
        pointer.input(
            &mut view,
            &scene,
            Mouse::Drag(Point {
                column: start.column + 5,
                row: start.row,
            }),
            now,
        );
        assert_ne!(view.module_layout.offset(scope, id), [0.0; 2]);
        pointer.cancel(&mut view);
        assert_eq!(view.module_layout.offset(scope, id), [0.0; 2]);
        assert!(view.bookmark() == before);
        assert!(pointer.press.is_none());
        let empty = Point {
            column: 0,
            row: SCENE_HEADER + scene.tooltip_rows,
        };
        assert!(scene.layer_at(empty.column, empty.row).is_none());
        pointer.input(&mut view, &scene, Mouse::Press(empty), now);
        pointer.input(
            &mut view,
            &scene,
            Mouse::Drag(Point {
                column: 8,
                row: empty.row + 2,
            }),
            now,
        );
        assert!(view.camera != before.camera);
        pointer.cancel(&mut view);
        assert!(view.bookmark() == before);
    }

    #[test]
    fn hovering_then_dragging_after_cancel_uses_restored_geometry() {
        for pixels in [false, true] {
            let mut view = ModelView::new(None);
            view.layout_edit = true;
            let scene = view.scene(100, 30, pixels, None);
            let start = (SCENE_HEADER + scene.tooltip_rows
                ..SCENE_HEADER + scene.tooltip_rows + scene.labels.len() - 1)
                .flat_map(|row| (0..scene.width - 8).map(move |column| Point { row, column }))
                .find(|point| {
                    scene.layer_at(point.column, point.row).is_some()
                        && scene.handoffs_at(point.column, point.row).is_empty()
                })
                .unwrap();
            let layer = scene.layer_at(start.column, start.row).unwrap();
            let id = view.layers[layer].id;
            let scope = view.graph_scope();
            let end = Point {
                column: start.column + 5,
                row: start.row,
            };
            let now = Instant::now();
            let mut pointer = Pointer::default();
            pointer.input(&mut view, &scene, Mouse::Press(start), now);
            pointer.input(&mut view, &scene, Mouse::Drag(end), now);
            pointer.cancel(&mut view);
            let restored = view.scene(100, 30, pixels, None);
            pointer.input(&mut view, &restored, Mouse::Move(start), now);
            assert_eq!(view.hover, Some(layer));
            let hovered = view.scene(100, 30, pixels, None);
            assert_eq!(hovered.layer_at(start.column, start.row), Some(layer));
            pointer.input(&mut view, &hovered, Mouse::Press(start), now);
            pointer.input(&mut view, &hovered, Mouse::Drag(end), now);
            assert_ne!(view.module_layout.offset(scope, id), [0.0; 2]);
        }
    }

    #[test]
    fn dragging_a_module_moves_only_its_endpoints_and_keeps_the_document() {
        for pixels in [false, true] {
            let mut view = ModelView::new(None);
            view.layout_edit = true;
            let affected = wire(1, 4, 0);
            let untouched = wire(2, 4, 1);
            view.selected_wire = Some(untouched);
            let module = view.indices[&affected.from.operator];
            let original = view.bookmark();
            let revision = view.document.revision();
            let scope = view.graph_scope();
            let before = view.scene(100, 30, pixels, None);
            let top = SCENE_HEADER + before.tooltip_rows;
            let start = (top..top + before.labels.len() - 1)
                .flat_map(|row| (0..before.width).map(move |column| Point { row, column }))
                .find(|point| {
                    before.layer_at(point.column, point.row) == Some(module)
                        && before.handoffs_at(point.column, point.row).is_empty()
                })
                .unwrap();
            let middle = Point {
                column: start.column + 5,
                row: start.row - 1,
            };
            let end = Point {
                column: start.column + 9,
                row: start.row - 2,
            };
            let mut pointer = Pointer::default();
            let now = Instant::now();
            pointer.input(&mut view, &before, Mouse::Press(start), now);
            pointer.input(&mut view, &before, Mouse::Drag(middle), now);
            let moving = view.scene(100, 30, pixels, None);
            assert_eq!(moving.layer_at(middle.column, middle.row), Some(module));
            pointer.input(&mut view, &moving, Mouse::Release(end), now);
            let after = view.scene(100, 30, pixels, None);
            assert_eq!(after.layer_at(end.column, end.row), Some(module));
            for prior in &before.handoffs {
                let next = after
                    .handoffs
                    .iter()
                    .find(|edge| edge.wire == prior.wire)
                    .unwrap();
                for (operator, a, b) in [
                    (prior.wire.from.operator, prior.from, next.from),
                    (prior.wire.to.operator, prior.to, next.to),
                ] {
                    let shift = if operator == affected.from.operator {
                        [9.0, -4.0]
                    } else {
                        [0.0; 2]
                    };
                    assert!((0..2).all(|axis| (b[axis] - a[axis] - shift[axis]).abs() < 1e-9));
                }
            }
            assert!(view.bookmark() == original);
            assert_eq!(view.document.revision(), revision);
            assert_eq!(view.document.source(affected.to), Some(affected.from));
            let saved = view.module_layout.offset(scope, affected.from.operator);
            view.zoom(module);
            assert_eq!(
                view.module_layout
                    .offset(view.graph_scope(), affected.from.operator),
                [0.0; 2]
            );
            assert!(view.zoom_out());
            assert_eq!(
                view.module_layout
                    .offset(view.graph_scope(), affected.from.operator),
                saved
            );

            view.apply_segment(Segment {
                base_revision: revision,
                edits: vec![Edit::Disconnect(untouched)],
            })
            .unwrap();
            assert_eq!(
                view.module_layout
                    .offset(view.graph_scope(), affected.from.operator),
                saved
            );
            view.module_layout.reset_scope(view.graph_scope());
            assert_eq!(
                view.module_layout
                    .offset(view.graph_scope(), affected.from.operator),
                [0.0; 2]
            );
        }
    }

    #[test]
    fn hovering_a_wire_holds_the_displayed_motion_phase_until_release() {
        for pixels in [false, true] {
            let mut view = ModelView::new(None);
            let mut motion = Motion::default();
            let start = Instant::now();
            motion.set_running(start, true);
            let shown_at = start + Duration::from_millis(610);
            let phase = motion.frame_phase(shown_at, true);
            let visible = view.scene(100, 30, pixels, Some(phase));
            let top = SCENE_HEADER + visible.tooltip_rows;
            let (point, wire) = (top..top + visible.labels.len() - 1)
                .flat_map(|row| (0..visible.width).map(move |column| Point { row, column }))
                .find_map(|point| {
                    if visible.layer_at(point.column, point.row).is_some() {
                        return None;
                    }
                    match visible.handoffs_at(point.column, point.row).as_slice() {
                        [wire] => Some((point, *wire)),
                        _ => None,
                    }
                })
                .expect("visible wire between two layers");
            let over = shown_at + Duration::from_millis(27);
            let mut pointer = Pointer::default();
            pointer.input(&mut view, &visible, Mouse::Move(point), over);
            assert_eq!(view.hover_handoff, Some(wire));
            motion.set_running(over, false);
            let later = over + Duration::from_secs(3);
            let still = motion.frame_phase(later, false);
            assert_eq!(still, phase);
            let paused = view.scene(100, 30, pixels, Some(still));
            for (before, after) in visible.handoffs.iter().zip(&paused.handoffs) {
                assert_eq!(
                    (before.wire, before.from, before.to),
                    (after.wire, after.from, after.to)
                );
            }
            pointer.input(&mut view, &paused, Mouse::Press(point), later);
            pointer.input(&mut view, &paused, Mouse::Release(point), later);
            assert_eq!(view.selected_wire, Some(wire));
            motion.set_running(later, true);
            let resumed = motion.frame_phase(later + Duration::from_millis(33), true);
            assert!((resumed - phase - 0.033).abs() < 1e-12);
        }
    }

    #[test]
    fn focus_feedback_is_a_short_smooth_pulse_and_respects_reduced_motion() {
        let start = Instant::now();
        let mut motion = Motion::default();
        motion.trigger_focus_feedback(start, true);
        assert_eq!(motion.focus_feedback(start), 0.0);
        assert!(motion.focus_feedback(start + Duration::from_millis(200)) > 0.4);
        assert!((motion.focus_feedback(start + Duration::from_millis(400)) - 1.0).abs() < 1e-12);
        assert!(motion.focus_feedback(start + Duration::from_millis(700)) > 0.1);
        assert_eq!(
            motion.focus_feedback(start + Duration::from_millis(800)),
            0.0
        );
        assert!(motion.focus_feedback_active(start + Duration::from_millis(850)));
        assert!(!motion.focus_feedback_active(start + Duration::from_millis(900)));

        motion.trigger_focus_feedback(start, false);
        assert_eq!(
            motion.focus_feedback(start + Duration::from_millis(400)),
            0.0
        );
    }

    #[test]
    fn wheel_zoom_keeps_the_cross_group_input_under_its_anchor() {
        for pixels in [false, true] {
            let mut view = Offline::new(model_maven_tui::diagnosis::Case::Large).view();
            view.camera.orbit(0.23, 0.11);
            let wire = view.selected_wire.unwrap();
            let endpoint = |scene: &Scene| {
                scene
                    .handoffs
                    .iter()
                    .find(|edge| edge.wire == wire)
                    .unwrap()
                    .to
            };
            let before = view.scene(100, 23, pixels, Some(0.45));
            let to = endpoint(&before);
            let point = Point {
                column: to[0].floor() as usize,
                row: SCENE_HEADER + before.tooltip_rows + (to[1] / 2.0).floor() as usize,
            };
            let row = before.art_row_at(point.column, point.row).unwrap();
            let anchor = [point.column as f64 + 0.5, row as f64 * 2.0 + 1.0];
            let mut pointer = Pointer::default();
            pointer.input(&mut view, &before, Mouse::Zoom(point, true), Instant::now());
            let mut after = view.scene(100, 23, pixels, Some(0.45));
            for (axis, actual) in endpoint(&after).into_iter().enumerate() {
                let expected = anchor[axis] + (to[axis] - anchor[axis]) * 1.10;
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "axis {axis}, {actual} vs {expected}"
                );
            }
            let projected = endpoint(&after);
            after.refine(view.camera, usize::MAX);
            assert_eq!(endpoint(&after), projected);
            assert_eq!(view.selected_wire, Some(wire));
            assert_eq!(view.document.revision(), 1);
        }
    }

    #[test]
    fn shared_terminal_cell_can_select_either_wire_without_moving_the_camera() {
        let mut view = ModelView::new(None);
        let first = wire(1, 4, 0);
        let second = wire(2, 4, 1);
        let initial = (view.camera, view.viewport, view.document.revision());
        for pixels in [false, true] {
            let mut scene = view.scene(100, 30, pixels, None);
            let top = SCENE_HEADER + scene.tooltip_rows;
            let point = Point {
                column: 19,
                row: top + 2,
            };
            assert_eq!(scene.layer_at(point.column, point.row), None);
            let edge = |wire, y, layers| {
                Handoff::new(
                    wire,
                    layers,
                    [[14.5, 5.0 - y], [24.5, 5.0 + y]],
                    None,
                    None,
                    None,
                )
            };
            scene.handoffs = vec![edge(second, -4.0, [2, 4]), edge(first, 4.0, [1, 4])];
            let picked = scene.handoffs_at(point.column, point.row);
            assert_eq!(picked, [first, second]);
            let point_outside_canvas = top + scene.labels.len() - 1;
            assert!(scene.handoffs_at(19, point_outside_canvas).is_empty());

            view.selected_wire = None;
            let mut pointer = Pointer::default();
            let now = Instant::now();
            let click = |pointer: &mut Pointer, view: &mut ModelView<'_>| {
                pointer.input(view, &scene, Mouse::Press(point), now);
                pointer.input(view, &scene, Mouse::Release(point), now);
            };
            click(&mut pointer, &mut view);
            assert_eq!(view.selected_wire, Some(first));
            click(&mut pointer, &mut view);
            assert_eq!(view.selected_wire, Some(second));
            assert!(
                view.render(&scene, Locale::English, false, pixels)
                    .contains("encoder.blocks[0] → merge[1]")
            );
            pointer.advance_wire(&mut view, &scene);
            assert_eq!(view.selected_wire, Some(first));
            assert!(
                view.render(&scene, Locale::English, false, pixels)
                    .contains("encoder.attention → merge")
            );
            assert!(view.camera == initial.0 && view.viewport == initial.1);
            assert_eq!(view.document.revision(), initial.2);
        }
    }

    #[test]
    fn visible_socket_takes_priority_over_its_layer_body() {
        use model_maven_tui::diagnosis::Case;
        for pixels in [false, true] {
            let mut view = Offline::new(Case::CrossedPorts).view();
            let (scene, point, wire, layer) = [0.0, 0.25, 0.50, 0.75]
                .into_iter()
                .find_map(|rotation| {
                    view.camera.orbit(rotation, 0.08);
                    let scene = view.scene(80, 23, pixels, None);
                    let top = SCENE_HEADER + scene.tooltip_rows;
                    let hit = scene.handoffs.iter().find_map(|edge| {
                        let [x, y] = edge.to;
                        let point = Point {
                            column: x.floor() as usize,
                            row: top + (y / 2.0).floor() as usize,
                        };
                        let layer = edge.layers[1];
                        (edge.continuation.is_none()
                            && x >= 0.0
                            && y >= 0.0
                            && scene.layer_at(point.column, point.row) == Some(layer))
                        .then_some((point, edge.wire, layer))
                    });
                    hit.map(|(point, wire, layer)| (scene, point, wire, layer))
                })
                .expect("a visible input socket should share a cell with a layer");
            view.selected_wire = None;
            let camera = view.camera;
            let mut pointer = Pointer::default();
            let now = Instant::now();
            pointer.input(&mut view, &scene, Mouse::Move(point), now);
            assert_eq!(view.hover_handoff, Some(wire));
            pointer.input(&mut view, &scene, Mouse::Press(point), now);
            pointer.input(&mut view, &scene, Mouse::Release(point), now);
            assert_eq!(view.selected_wire, Some(wire));
            assert_eq!(view.hover, Some(layer));
            assert!(view.camera == camera);
            assert!(view.focus.is_none());
        }
    }

    #[test]
    fn paired_ports_remain_addressable_through_paging_zoom_and_rewire() {
        use model_maven_tui::diagnosis::Case;
        for pixels in [false, true] {
            let mut offline = Offline::new(Case::CrossedPorts);
            let mut view = offline.view();
            let pairs = view
                .document
                .wires()
                .filter(|wire| {
                    view.document.operator(wire.to.operator).unwrap().path == "net.merge"
                })
                .collect::<Vec<_>>();
            let [edited, retained] = pairs.try_into().unwrap();
            assert_eq!([edited.from.slot, edited.to.slot], [1, 0]);
            assert_eq!([retained.from.slot, retained.to.slot], [0, 1]);

            let cells = |scene: &Scene| {
                let top = SCENE_HEADER + scene.tooltip_rows;
                let width = scene.width;
                (top..top + scene.labels.len() - 1)
                    .flat_map(move |row| (0..width).map(move |column| Point { column, row }))
            };
            let overview = view.scene(80, 23, pixels, Some(0.0));
            for (wire, other) in [(edited, retained), (retained, edited)] {
                let path = overview
                    .handoffs
                    .iter()
                    .find(|edge| edge.wire == wire)
                    .unwrap();
                let strokes = |focus| {
                    let mut count = 0;
                    path.draw(Some(focus), |_, _| count += 1);
                    count
                };
                assert!(strokes(wire) > strokes(other));
            }
            let crossing = cells(&overview)
                .find(|point| {
                    let hit = overview.handoffs_at(point.column, point.row);
                    hit.len() == 2
                        && hit.contains(&edited)
                        && hit.contains(&retained)
                        && overview.layer_at(point.column, point.row).is_none()
                })
                .expect("the actual graph exposes both wires in a single cell");
            let hits = overview.handoffs_at(crossing.column, crossing.row);
            assert_eq!((crossing.column + 1, crossing.row + 1), (27, 8));
            let mut pointer = Pointer::default();
            let now = Instant::now();
            let click = |pointer: &mut Pointer, view: &mut ModelView<'_>, scene: &Scene, point| {
                pointer.input(view, scene, Mouse::Press(point), now);
                pointer.input(view, scene, Mouse::Release(point), now);
            };
            click(&mut pointer, &mut view, &overview, crossing);
            assert_eq!(view.selected_wire, Some(hits[0]));
            click(&mut pointer, &mut view, &overview, crossing);
            assert_eq!(view.selected_wire, Some(hits[1]));
            pointer.advance_wire(&mut view, &overview);
            assert_eq!(view.selected_wire, Some(hits[0]));

            view.follow_wire(&overview, false, None);
            assert_eq!(view.layers[view.selected].id, edited.from.operator);
            view.camera.orbit(0.45, 0.17);
            let camera = view.camera;
            let outgoing = view.scene(30, 23, pixels, Some(0.0));
            let check_page = |scene: &Scene, direction| {
                for wire in [edited, retained] {
                    assert!(
                        scene
                            .handoffs
                            .iter()
                            .any(|edge| edge.wire == wire && edge.continuation == Some(direction))
                    );
                }
            };
            check_page(&outgoing, super::super::Continuation::Outgoing);
            view.page(&outgoing, true);
            let incoming = view.scene(30, 23, pixels, Some(0.0));
            check_page(&incoming, super::super::Continuation::Incoming);
            view.follow_wire(&incoming, true, None);
            let merge = view.indices[&edited.to.operator];
            assert_eq!(view.selected, merge);
            view.viewport.zoom(1.10, [48.0, 10.0]);
            let viewport = view.viewport;
            let full = view.scene(80, 23, pixels, Some(0.0));
            let module = cells(&full)
                .find(|point| full.layer_at(point.column, point.row) == Some(merge))
                .unwrap();
            click(&mut pointer, &mut view, &full, module);
            assert!(view.selected_wire.is_none());
            click(&mut pointer, &mut view, &full, module);
            assert_eq!(view.focus, Some(merge));
            assert_eq!(view.selected_wire, Some(hits[0]));
            assert!(view.zoom_out());
            assert!(view.camera == camera && view.viewport == viewport);
            assert_eq!(view.selected, merge);
            assert_eq!(view.selected_wire, Some(hits[0]));

            for key in b"eyney" {
                assert!(offline.input(*key, &mut view));
            }
            assert_eq!(view.document.revision(), 3);
            assert_ne!(view.document.source(edited.to), Some(edited.from));
            assert_eq!(view.document.source(retained.to), Some(retained.from));
            let now = view.scene(80, 23, pixels, Some(0.0));
            assert!(now.handoffs.iter().any(|edge| edge.wire == retained));
        }
    }
}
#[test]
fn recorded_call_focus_selects_the_unique_module_without_adding_wires() {
    let mut modules = vec![
        serde_json::json!({"path":"model", "type":"Model"}),
        serde_json::json!({"path":"model.branches", "type":"ModuleList"}),
    ];
    modules.extend((0..32).map(
        |index| serde_json::json!({"path": format!("model.branches.{index}"), "type":"Linear"}),
    ));
    let mut view = ModelView::new(Some(&modules));
    let path = super::failure::Path::new(
        "model.branches.31".into(),
        Some(1),
        None,
        super::failure::Stop::Call,
        Vec::new(),
    );
    let before = view.document.revision();
    assert!(focus_failure_target(&mut view, &path));
    assert_eq!(view.layers[view.selected].path, "model.branches.31");
    assert_eq!(view.layers[view.focus.unwrap()].path, "model.branches");
    assert_eq!(view.document.revision(), before);
    assert!(view.document.wires().next().is_none());
    for flat in [false, true] {
        if flat {
            view.toggle_flat_flow();
            assert!(view.flat_flow);
        }
        let scene = view.scene(40, 24, false, None);
        assert!(
            scene
                .solids
                .iter()
                .any(|solid| solid.layer == view.selected)
        );
        if flat {
            assert!(scene.handoffs.is_empty());
            let frame = view.render(&scene, Locale::English, false, false);
            assert!(frame.contains("no links recorded"), "{frame}");
        }
    }
    view.toggle_flat_flow();
    view.explosion = 1;
    let exploded = view.scene(40, 24, false, None);
    assert!(
        exploded
            .solids
            .iter()
            .any(|solid| solid.layer == view.selected)
    );
    assert!(exploded.handoffs.is_empty());
    assert!(
        view.render(&exploded, Locale::English, false, false)
            .contains("no links recorded")
    );
    let selected = view.selected;
    view.live_evidence = Some(super::LiveEvidence {
        path: "model.branches.31".into(),
        ambiguous_registration: false,
        last_observed: false,
        from_calls: false,
        reused: false,
        input: None,
        invocation: Some(1),
        invocation_id: Some("f1:1".into()),
        expected: None,
        raw_report: String::new(),
        #[cfg(unix)]
        route_cache: std::cell::OnceCell::new(),
        inputs: Vec::new(),
        selected_input: None,
        show_input: false,
        graph_target_unresolved: false,
        #[cfg(unix)]
        projection: None,
        diagnostics: Vec::new(),
    });
    let unknown = super::failure::Path::new(
        "model.missing".into(),
        None,
        None,
        super::failure::Stop::Call,
        Vec::new(),
    );
    assert!(!focus_failure_target(&mut view, &unknown));
    assert_eq!(view.selected, selected);
    let scene = view.scene(80, 24, false, None);
    assert!(
        view.render(&scene, Locale::English, false, false)
            .contains("Call location unknown")
    );
    assert!(focus_failure_target(&mut view, &path));
    assert!(!view.live_evidence.as_ref().unwrap().graph_target_unresolved);
    view.live_evidence.as_mut().unwrap().projection = Some(super::ObservedProjection {
        source: Some("model.src".into()),
        counts: [(view.layers[selected].id, 1)].into_iter().collect(),
        hidden: false,
        limited: false,
        unmapped: 0,
    });
    view.explosion = 0;
    view.toggle_flat_flow();
    for pixels in [false, true] {
        let scene = view.scene(40, 24, pixels, None);
        assert!(
            scene
                .captions
                .iter()
                .flatten()
                .any(|(_, caption)| caption.contains('?'))
        );
        let frame = view.render(&scene, Locale::English, false, pixels);
        assert!(frame.contains("? 1 matches from model.src"), "{frame}");
        assert!(frame.contains("no links · ? partial"), "{frame}");
        assert!(scene.handoffs.is_empty());
        assert_eq!(scene.observed_routes.len(), 1);
        assert!(scene.observed_routes[0].external);
        let mut segments = 0;
        scene.observed_routes[0].draw(|_, _| segments += 1);
        assert!(segments > 0);
    }
    let visible_pixels =
        pixels::pixels_for_test(&view.scene(40, 24, true, None), view.camera, view.selected);
    let wide = view.scene(100, 24, false, None);
    assert!(wide.page_size < wide.item_count);
    let mut bounds: Vec<_> = wide
        .solids
        .iter()
        .map(|solid| solid.bounds(view.camera))
        .collect();
    bounds.sort_by(|a, b| a[0][1].total_cmp(&b[0][1]));
    assert!(
        bounds
            .windows(2)
            .all(|pair| pair[0][1][1] <= pair[1][0][1] + 1e-9)
    );
    view.live_evidence
        .as_mut()
        .unwrap()
        .projection
        .as_mut()
        .unwrap()
        .hidden = true;
    let hidden = view.scene(40, 24, false, None);
    assert!(hidden.observed_routes.is_empty());
    let hidden_pixels =
        pixels::pixels_for_test(&view.scene(40, 24, true, None), view.camera, view.selected);
    assert_ne!(visible_pixels, hidden_pixels);
    assert!(
        !hidden
            .captions
            .iter()
            .flatten()
            .any(|(_, caption)| caption.contains('?'))
    );
    assert!(
        !view
            .render(&hidden, Locale::English, false, false)
            .contains("? 1 matches from")
    );
    assert_eq!(view.document.revision(), before);
    assert!(view.document.wires().next().is_none());
    view.live_evidence
        .as_mut()
        .unwrap()
        .projection
        .as_mut()
        .unwrap()
        .hidden = false;
    view.toggle_flat_flow();
    view.explosion = 1;
    let exploded = view.scene(40, 24, true, None);
    assert_eq!(exploded.observed_routes.len(), 1);
    assert!(exploded.handoffs.is_empty());
    view.live_evidence
        .as_mut()
        .unwrap()
        .projection
        .as_mut()
        .unwrap()
        .source = Some("model.branches.30".into());
    let internal = view.scene(40, 24, true, None);
    assert_eq!(internal.observed_routes.len(), 1);
    assert!(!internal.observed_routes[0].external);
    view.live_evidence
        .as_mut()
        .unwrap()
        .projection
        .as_mut()
        .unwrap()
        .source = Some("model.branches.31".into());
    let same_module = view.scene(40, 24, true, None);
    assert!(same_module.observed_routes.is_empty());
    assert!(same_module.handoffs.is_empty());
    let counts = view.children[view.focus.unwrap()]
        .iter()
        .map(|&index| (view.layers[index].id, 1))
        .collect();
    let projection = view
        .live_evidence
        .as_mut()
        .unwrap()
        .projection
        .as_mut()
        .unwrap();
    projection.source = Some("model.src".into());
    projection.counts = counts;
    let last_page = view.scene(40, 24, false, None);
    let last_targets: std::collections::BTreeSet<_> = last_page
        .observed_routes
        .iter()
        .map(|route| route.target)
        .collect();
    assert_eq!(last_targets.len(), last_page.page_size);
    assert!(last_page.observed_routes.iter().all(|route| route.external));
    view.page(&last_page, false);
    let previous_page = view.scene(40, 24, true, None);
    let previous_targets: std::collections::BTreeSet<_> = previous_page
        .observed_routes
        .iter()
        .map(|route| route.target)
        .collect();
    assert_eq!(previous_targets.len(), previous_page.page_size);
    assert!(previous_targets.is_disjoint(&last_targets));
    assert!(previous_page.handoffs.is_empty());
}
