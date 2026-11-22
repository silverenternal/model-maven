use super::appearance::{Appearance, LinkInk, Material};
use model_maven_tui::graph_document::OperatorId;
use std::fmt::Write as _;

// Keep the nearest and farthest points within roughly 5% of parallel projection.
const PERSPECTIVE_STRENGTH: f64 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Camera {
    pub yaw: f64,
    pub pitch: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.56,
            pitch: 0.52,
        }
    }
}

impl Camera {
    pub fn isometric() -> Self {
        Self {
            yaw: std::f64::consts::FRAC_PI_4,
            pitch: (1.0 / 2.0_f64.sqrt()).atan(),
        }
    }

    pub fn projected_elevation(self, elevation: f64, perspective: f64) -> f64 {
        self.project([0.0, elevation, 0.0], [0.0; 2], perspective).y
    }

    pub fn orbit(&mut self, yaw: f64, pitch: f64) {
        self.yaw = (self.yaw + yaw).rem_euclid(std::f64::consts::TAU);
        self.pitch = (self.pitch + pitch).clamp(-1.35, 1.35);
    }

    fn project(self, [x, y, z]: [f64; 3], [cx, cy]: [f64; 2], perspective: f64) -> Vertex {
        let (sy, cyaw) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let depth = x * sy + z * cyaw;
        let view_depth = depth * cp + y * sp;
        let scale = (1.0 - view_depth * perspective).recip();
        Vertex {
            x: cx + (x * cyaw - z * sy) * scale,
            y: cy + (-y * cp + depth * sp) * scale,
            z: view_depth,
            scale,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Vertex {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub scale: f64,
}

impl Vertex {
    pub fn edge(self, other: Self, x: f64, y: f64) -> f64 {
        (other.x - self.x) * (y - self.y) - (other.y - self.y) * (x - self.x)
    }
}

#[derive(Clone, Copy)]
struct Sample {
    depth: f64,
    layer: usize,
    shade: usize,
    #[cfg(test)]
    slice: (OperatorId, usize),
}

#[derive(Clone, Copy)]
pub(super) struct SliceKey {
    pub operator: OperatorId,
    pub ordinal: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Cuboid {
    pub center: [f64; 2],
    pub half: [f64; 3],
    pub elevation: f64,
    pub layer: usize,
    pub key: SliceKey,
    pub perspective: f64,
    pub material: Material,
}

impl Cuboid {
    pub fn layer_perspective([x, y, z]: [f64; 3]) -> f64 {
        PERSPECTIVE_STRENGTH / x.hypot(y).hypot(z).max(1.0)
    }

    fn corners(self) -> [[f64; 3]; 8] {
        let [x, y, z] = self.half;
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ]
        .map(|[x, y, z]| [x, y + self.elevation, z])
    }

    pub fn vertices(self, camera: Camera) -> [Vertex; 8] {
        self.corners()
            .map(|corner| camera.project(corner, self.center, self.perspective))
    }

    pub fn grain(self, camera: Camera, mut draw: impl FnMut(Vertex, bool)) {
        let [_, _, _, _, shows_top, shows_bottom] = self.visible_faces(camera);
        if !shows_top && !shows_bottom {
            return;
        }
        let [x, y, z] = self.half;
        let vertical = self.elevation + if shows_top { y } else { -y };
        let spacing = self.material.grain_spacing;
        let columns = ((x * 2.0 / spacing) as usize).clamp(1, 28);
        let rows = ((z * 2.0 / spacing) as usize).clamp(1, 18);
        let id = self.key.operator.0;
        let seed = ((id ^ (id >> 32)) as u32).wrapping_mul(73)
            ^ (self.key.ordinal as u32).wrapping_mul(88_493);
        for row in 0..rows {
            for column in 0..columns {
                let value = (column as u32 * 1_031 + row as u32 * 1_543)
                    .wrapping_add(seed.rotate_left(7))
                    .wrapping_mul(2_654_435_761)
                    >> 28;
                if value > 8 {
                    continue;
                }
                let u = (column as f64 + 0.5) / columns as f64;
                let v = (row as f64 + 0.5) / rows as f64;
                let point = [(2.0 * u - 1.0) * x, vertical, (2.0 * v - 1.0) * z];
                draw(
                    camera.project(point, self.center, self.perspective),
                    value < u32::from(self.material.grain_density),
                );
            }
        }
    }

    pub fn bounds(self, camera: Camera) -> [[f64; 2]; 2] {
        let vertices = self.vertices(camera);
        vertices
            .iter()
            .fold([[vertices[0].x, vertices[0].y]; 2], |[min, max], vertex| {
                [
                    [min[0].min(vertex.x), min[1].min(vertex.y)],
                    [max[0].max(vertex.x), max[1].max(vertex.y)],
                ]
            })
    }

    pub fn visible_faces(self, camera: Camera) -> [bool; 6] {
        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = camera.pitch.sin_cos();
        let (x, y, z) = (yaw_sin * pitch_cos, pitch_sin, yaw_cos * pitch_cos);
        let [half_x, half_y, half_z] = self.half.map(|value| value * self.perspective);
        let elevation = self.elevation * self.perspective;
        [
            -z - half_z,
            z - half_z,
            -x - half_x,
            x - half_x,
            y - half_y - elevation,
            -y - half_y + elevation,
        ]
        .map(|facing| facing > 1e-6)
    }

    pub fn port(self, camera: Camera, axis: usize, direction: f64) -> [f64; 2] {
        self.port_at(camera, axis, direction, None)
    }

    pub fn port_at(
        self,
        camera: Camera,
        axis: usize,
        direction: f64,
        cross_coordinate: Option<f64>,
    ) -> [f64; 2] {
        let vertices = self.vertices(camera);
        let visible = self.visible_faces(camera);
        let center = camera.project([0.0, self.elevation, 0.0], self.center, self.perspective);
        let mut center = [center.x, center.y];
        let cross_axis = 1 - axis;
        let across = |vertex: &Vertex| if cross_axis == 0 { vertex.x } else { vertex.y };
        let min = vertices.iter().map(across).fold(f64::INFINITY, f64::min);
        let max = vertices
            .iter()
            .map(across)
            .fold(f64::NEG_INFINITY, f64::max);
        center[cross_axis] = cross_coordinate
            .unwrap_or(center[cross_axis])
            .clamp(min + (max - min) * 0.15, max - (max - min) * 0.15);
        let mut distance: f64 = 0.0;
        for (a, b, left, right) in EDGES {
            if visible[left] == visible[right] {
                continue;
            }
            let (a, b) = (vertices[a], vertices[b]);
            let a = [a.x, a.y];
            let b = [b.x, b.y];
            let t = (center[cross_axis] - a[cross_axis]) / (b[cross_axis] - a[cross_axis]);
            if (0.0..=1.0).contains(&t) {
                let edge = a[axis] + (b[axis] - a[axis]) * t;
                distance = distance.max((edge - center[axis]) * direction);
            }
        }
        let mut port = center;
        port[axis] += distance * direction;
        port
    }
}

// Back, front, left, right, top, bottom. Shade order follows light from above.
pub(super) const FACES: [([usize; 4], usize); 6] = [
    ([0, 1, 2, 3], 2),
    ([4, 5, 6, 7], 2),
    ([0, 3, 7, 4], 1),
    ([1, 2, 6, 5], 1),
    ([3, 2, 6, 7], 3),
    ([0, 1, 5, 4], 0),
];

// Endpoints and two adjoining faces per edge.
pub(super) const EDGES: [(usize, usize, usize, usize); 12] = [
    (0, 1, 0, 5),
    (1, 2, 0, 3),
    (2, 3, 0, 4),
    (3, 0, 0, 2),
    (4, 5, 1, 5),
    (5, 6, 1, 3),
    (6, 7, 1, 4),
    (7, 4, 1, 2),
    (0, 4, 2, 5),
    (1, 5, 3, 5),
    (2, 6, 3, 4),
    (3, 7, 2, 4),
];

const BRAILLE_DOTS: [[u8; 2]; 4] = [[1, 8], [2, 16], [4, 32], [64, 128]];

#[derive(Clone, Copy)]
enum Stroke {
    Surface { layer: usize, seam: bool },
    Link(bool, LinkInk),
    Mark(LinkInk),
}

pub(super) struct Canvas {
    width: usize,
    rows: usize,
    samples: Vec<Option<Sample>>,
    outlines: Vec<u8>,
    seams: Vec<u8>,
    grain: Vec<u8>,
    links: Vec<u8>,
    link_inks: Vec<(bool, LinkInk)>,
    marks: Vec<(u8, LinkInk)>,
}

impl Canvas {
    pub fn new(width: usize, rows: usize) -> Self {
        Self {
            width,
            rows,
            samples: vec![None; width * rows * 8],
            outlines: vec![0; width * rows],
            seams: vec![0; width * rows],
            grain: vec![0; width * rows],
            links: vec![0; width * rows],
            link_inks: vec![(false, LinkInk::Structure); width * rows],
            marks: vec![(0, LinkInk::Structure); width * rows],
        }
    }

    pub fn cuboid(&mut self, camera: Camera, solid: Cuboid) {
        let vertices = solid.vertices(camera);
        let layer = solid.layer;
        let slice = (solid.key.operator, solid.key.ordinal);
        for ([a, b, c, d], shade) in FACES {
            self.triangle([vertices[a], vertices[b], vertices[c]], layer, shade, slice);
            self.triangle([vertices[a], vertices[c], vertices[d]], layer, shade, slice);
        }
    }

    #[cfg(test)]
    fn visible_slice_cells(&self) -> std::collections::BTreeMap<(OperatorId, usize), usize> {
        let mut cells = std::collections::BTreeMap::<
            (OperatorId, usize),
            std::collections::BTreeSet<(usize, usize)>,
        >::new();
        for (index, sample) in self.samples.iter().enumerate() {
            let Some(sample) = sample else { continue };
            let row = index / (self.width * 2);
            let column = index % (self.width * 2);
            cells
                .entry(sample.slice)
                .or_default()
                .insert((column / 2, row / 4));
        }
        cells
            .into_iter()
            .map(|(slice, cells)| (slice, cells.len()))
            .collect()
    }

    pub fn outline(&mut self, camera: Camera, solid: Cuboid) {
        let vertices = solid.vertices(camera);
        let visible = solid.visible_faces(camera);
        for (a, b, left, right) in EDGES {
            if visible[left] || visible[right] {
                self.stroke(
                    vertices[a],
                    vertices[b],
                    Stroke::Surface {
                        layer: solid.layer,
                        seam: visible[left] && visible[right],
                    },
                );
            }
        }
        solid.grain(camera, |point, bright| {
            if !bright || point.x < 0.0 || point.y < 0.0 {
                return;
            }
            let column = (point.x * 2.0) as usize;
            let row = (point.y * 2.0) as usize;
            if column < self.width * 2
                && row < self.rows * 4
                && self.samples[row * self.width * 2 + column].is_some_and(|sample| {
                    sample.layer == solid.layer && (sample.depth - point.z).abs() < 0.45
                })
            {
                self.grain[(row / 4) * self.width + column / 2] |=
                    BRAILLE_DOTS[row % 4][column % 2];
            }
        });
    }

    pub fn connector(&mut self, from: [f64; 2], to: [f64; 2], ink: (bool, LinkInk)) {
        self.stroke(
            Self::vertex(from),
            Self::vertex(to),
            Stroke::Link(ink.0, ink.1),
        );
    }

    pub fn mark(&mut self, points: &[[f64; 2]; 5], ink: LinkInk) {
        for pair in points.windows(2) {
            self.stroke(
                Self::vertex(pair[0]),
                Self::vertex(pair[1]),
                Stroke::Mark(ink),
            );
        }
    }

    fn vertex([x, y]: [f64; 2]) -> Vertex {
        Vertex {
            x,
            y,
            z: 0.0,
            scale: 1.0,
        }
    }

    fn stroke(&mut self, a: Vertex, b: Vertex, stroke: Stroke) {
        let dots = |vertex: Vertex| {
            [
                i64::from((vertex.x * 2.0).floor() as i32),
                i64::from((vertex.y * 2.0).floor() as i32),
            ]
        };
        let (a, b) = if dots(a) <= dots(b) { (a, b) } else { (b, a) };
        let [mut x, mut y] = dots(a);
        let [end_x, end_y] = dots(b);
        let [width, height] = [(self.width * 2) as i64, (self.rows * 4) as i64];
        if end_x < 0 || x >= width || y.max(end_y) < 0 || y.min(end_y) >= height {
            return;
        }
        let dx = (end_x - x).abs();
        let dy = -(end_y - y).abs();
        let step_x = (end_x - x).signum();
        let step_y = (end_y - y).signum();
        let length = dx.max(-dy).max(1);
        // Fast-forward along the major axis. The original Bresenham phase and
        // depth are preserved at the first visible step.
        let (first, last) = if dx >= -dy {
            (0.max(-x), length.min(width - 1 - x))
        } else if step_y > 0 {
            (0.max(-y), length.min(height - 1 - y))
        } else {
            (0.max(y - height + 1), length.min(y))
        };
        let mut error = dx + dy;
        if first > 0 {
            let advance = |distance: i64| {
                ((i128::from(first) * i128::from(distance) + i128::from(length / 2))
                    / i128::from(length)) as i64
            };
            let x_steps = advance(dx);
            let y_steps = advance(-dy);
            let error_delta =
                i128::from(x_steps) * i128::from(dy) + i128::from(y_steps) * i128::from(dx);
            error += error_delta as i64;
            x += step_x * x_steps;
            y += step_y * y_steps;
        }
        for step in first..=last {
            if x >= 0 && y >= 0 && x < width && y < height {
                let (column, row) = (x as usize, y as usize);
                let surface = self.samples[row * self.width * 2 + column];
                let cell = (row / 4) * self.width + column / 2;
                let dot = BRAILLE_DOTS[row % 4][column % 2];
                match stroke {
                    Stroke::Mark(ink) => {
                        let (mask, role) = &mut self.marks[cell];
                        *mask |= dot;
                        *role = ink;
                    }
                    Stroke::Link(focused, ink) if surface.is_none() => {
                        self.link_inks[cell] = self.link_inks[cell].max((focused, ink));
                        self.links[cell] |= dot;
                    }
                    Stroke::Surface { layer, seam } => {
                        let t = step as f64 / length as f64;
                        let wa = (1.0 - t) * a.scale;
                        let wb = t * b.scale;
                        let depth = (a.z * wa + b.z * wb) / (wa + wb);
                        if surface.is_none_or(|surface| {
                            layer == surface.layer && depth + 0.45 >= surface.depth
                        }) {
                            let buffer = if seam {
                                &mut self.seams
                            } else {
                                &mut self.outlines
                            };
                            buffer[cell] |= dot;
                        }
                    }
                    Stroke::Link(..) => {}
                }
            }
            if x == end_x && y == end_y {
                break;
            }
            let twice = 2 * error;
            if twice >= dy {
                error += dy;
                x += step_x;
            }
            if twice <= dx {
                error += dx;
                y += step_y;
            }
        }
    }

    fn triangle(
        &mut self,
        [a, b, c]: [Vertex; 3],
        layer: usize,
        shade: usize,
        _slice: (OperatorId, usize),
    ) {
        let area = a.edge(b, c.x, c.y);
        if area.abs() < 0.001 {
            return;
        }
        let left = (2.0 * a.x.min(b.x).min(c.x)).floor().max(0.0) as usize;
        let right = ((2.0 * a.x.max(b.x).max(c.x)).ceil() as usize).min(self.width * 2);
        let top = (2.0 * a.y.min(b.y).min(c.y)).floor().max(0.0) as usize;
        let bottom = ((2.0 * a.y.max(b.y).max(c.y)).ceil() as usize).min(self.rows * 4);
        for row in top..bottom {
            for column in left..right {
                let x = (column as f64 + 0.5) * 0.5;
                let y = (row as f64 + 0.5) * 0.5;
                let wa = b.edge(c, x, y) / area;
                let wb = c.edge(a, x, y) / area;
                let wc = 1.0 - wa - wb;
                if wa.min(wb).min(wc) < -1e-6 {
                    continue;
                }
                let (wa, wb, wc) = (wa * a.scale, wb * b.scale, wc * c.scale);
                let depth = (a.z * wa + b.z * wb + c.z * wc) / (wa + wb + wc);
                let cell = &mut self.samples[row * self.width * 2 + column];
                if cell.is_none_or(|sample| depth > sample.depth) {
                    *cell = Some(Sample {
                        depth,
                        layer,
                        shade,
                        #[cfg(test)]
                        slice: _slice,
                    });
                }
            }
        }
    }

    #[cfg(unix)]
    pub fn layer_at(&self, column: usize, row: usize) -> Option<usize> {
        self.cell(column, row)
            .into_iter()
            .flatten()
            .max_by(|left, right| left.depth.total_cmp(&right.depth))
            .map(|sample| sample.layer)
    }

    #[cfg(test)]
    pub(super) fn link_dot_count_for_test(&self) -> u32 {
        self.links.iter().map(|mask| mask.count_ones()).sum()
    }

    fn cell(&self, column: usize, row: usize) -> [Option<Sample>; 8] {
        if column >= self.width || row >= self.rows {
            return [None; 8];
        }
        let offset = row * 4 * self.width * 2 + column * 2;
        std::array::from_fn(|i| self.samples[offset + (i / 2) * self.width * 2 + i % 2])
    }

    pub fn write_row(
        &self,
        out: &mut String,
        row: usize,
        underneath: &[char],
        captions: &[(usize, String)],
        selected: usize,
        color: Option<Appearance>,
    ) {
        let mut style = (None, None);
        let mut captions = captions.iter().peekable();
        let mut covered_until = 0;
        for (column, &fallback) in underneath.iter().enumerate() {
            if column < covered_until {
                continue;
            }
            if captions.peek().is_some_and(|(start, _)| *start == column) {
                let (_, caption) = captions.next().unwrap();
                if style != (None, None) {
                    out.push_str("\x1b[0m");
                    style = (None, None);
                }
                out.push_str(caption);
                covered_until = column + crate::terminal::text::display_width(caption);
                continue;
            }
            let samples = self.cell(column, row);
            let cell = row * self.width + column;
            let outline = self.outlines.get(cell).copied().unwrap_or_default();
            let seam = self.seams.get(cell).copied().unwrap_or_default();
            let link = self.links.get(cell).copied().unwrap_or_default();
            let grain = self.grain.get(cell).copied().unwrap_or_default();
            let coverage = samples.iter().enumerate().fold(0_u8, |dots, (i, sample)| {
                if sample.is_some() {
                    dots | BRAILLE_DOTS[i / 2][i % 2]
                } else {
                    dots
                }
            });
            let fill = dominant(&samples);
            let dots = outline | seam;
            let (mark, mark_ink) = self.marks.get(cell).copied().unwrap_or_default();
            let braille = |mask| char::from_u32(0x2800 + u32::from(mask)).unwrap_or(' ');
            let Some(color) = color else {
                out.push(match fill {
                    _ if mark != 0 => braille(mark),
                    None if dots | link != 0 => braille(dots | link),
                    Some(_) if outline != 0 && coverage != 0xff => braille(outline),
                    Some(_) if coverage.count_ones() <= 5 => braille(coverage | grain),
                    Some(_) if grain != 0 => braille(grain | (coverage & 0x95)),
                    Some(sample) if sample.layer == selected => ['░', '▒', '▓', '█'][sample.shade],
                    Some(sample) => ['·', '░', '▒', '▓'][sample.shade],
                    None => fallback,
                });
                continue;
            };
            let paint = |sample: Sample| color.face(sample.shade, sample.layer == selected);
            let (character, next) = match fill {
                _ if mark != 0 => (
                    braille(mark),
                    (Some(color.link(mark_ink, true)), fill.map(paint)),
                ),
                None if dots != 0 => (braille(dots), (None, None)),
                None if link != 0 => {
                    let (focused, ink) = self.link_inks[cell];
                    (braille(link), (Some(color.link(ink, focused)), None))
                }
                None => (fallback, (None, None)),
                Some(sample) if coverage.count_ones() <= 5 => {
                    let silhouette = (if outline == 0 { coverage } else { outline }) | grain;
                    (braille(silhouette), (Some(paint(sample)), None))
                }
                Some(sample) if dots | grain != 0 => {
                    let edge = if color.background == crate::terminal::layout::Background::Light {
                        if sample.shade >= 2 { 16 } else { 253 }
                    } else if sample.shade == 3 {
                        24
                    } else if outline != 0 {
                        152
                    } else {
                        109
                    };
                    (braille(dots | grain), (Some(edge), Some(paint(sample))))
                }
                Some(sample) => {
                    let upper = dominant(&samples[..4]).map_or(paint(sample), paint);
                    let lower = dominant(&samples[4..]).map_or(paint(sample), paint);
                    if upper == lower {
                        ('█', (Some(upper), None))
                    } else {
                        ('▀', (Some(upper), Some(lower)))
                    }
                }
            };
            if next != style {
                out.push_str("\x1b[0m");
                if let Some(foreground) = next.0 {
                    let _ = write!(out, "\x1b[38;5;{foreground}m");
                }
                if let Some(background) = next.1 {
                    let _ = write!(out, "\x1b[48;5;{background}m");
                }
                style = next;
            }
            out.push(character);
        }
        if style != (None, None) {
            out.push_str("\x1b[0m");
        }
        out.push('\n');
    }
}

fn dominant(samples: &[Option<Sample>]) -> Option<Sample> {
    let mut counts = [0; 4];
    for sample in samples.iter().flatten() {
        counts[sample.shade] += 1;
    }
    samples.iter().flatten().copied().max_by(|a, b| {
        counts[a.shade]
            .cmp(&counts[b.shade])
            .then_with(|| a.depth.total_cmp(&b.depth))
    })
}

#[cfg(test)]
pub(super) fn visible_slice_cells_for_test(
    solids: &[Cuboid],
    camera: Camera,
    width: usize,
    rows: usize,
) -> std::collections::BTreeMap<(OperatorId, usize), usize> {
    let mut canvas = Canvas::new(width, rows);
    for &solid in solids {
        canvas.cuboid(camera, solid);
    }
    canvas.visible_slice_cells()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isometric_camera_projects_the_three_axes_with_equal_length() {
        let camera = Camera::isometric();
        let projected = [
            camera.project([1.0, 0.0, 0.0], [0.0; 2], 0.0),
            camera.project([0.0, 1.0, 0.0], [0.0; 2], 0.0),
            camera.project([0.0, 0.0, 1.0], [0.0; 2], 0.0),
        ];
        let lengths = projected.map(|vertex| vertex.x.hypot(vertex.y));
        assert!(
            lengths
                .iter()
                .all(|length| (*length - lengths[0]).abs() < 1e-12),
            "projected axis lengths: {lengths:?}"
        );
    }

    #[test]
    fn module_families_have_distinct_monochrome_surface_grain_without_geometry_changes() {
        use crate::{graph::appearance::SliceStyle, workbench::Style};
        use std::collections::BTreeSet;

        let camera = Camera::default();
        let kinds = [
            "Embedding",
            "Attention",
            "Linear",
            "Add",
            "TransformerBlock",
        ];
        let mut samples = String::from("NO-COLOR MODULE SURFACE GRAIN · fixed 30×20 canvas\n");
        for (style_index, style) in [Style::Default, Style::Compact, Style::Story]
            .into_iter()
            .enumerate()
        {
            let mut signatures = BTreeSet::new();
            let mut reference_hits = None;
            for kind in kinds {
                let mut canvas = Canvas::new(30, 20);
                let material = SliceStyle::for_module(kind).styled(style).material;
                let solid = Cuboid {
                    center: [15.0, 10.0],
                    half: [7.0, 4.0, 5.0],
                    elevation: 0.0,
                    layer: 0,
                    key: SliceKey {
                        operator: OperatorId(17),
                        ordinal: 0,
                    },
                    perspective: Cuboid::layer_perspective([7.0, 4.0, 5.0]),
                    material,
                };
                canvas.cuboid(camera, solid);
                canvas.outline(camera, solid);

                let hits: Vec<_> = (0..canvas.rows)
                    .flat_map(|row| (0..canvas.width).map(move |column| (column, row)))
                    .map(|(column, row)| canvas.layer_at(column, row).is_some())
                    .collect();
                if let Some(reference) = &reference_hits {
                    assert_eq!(&hits, reference, "{kind} changed the hit silhouette");
                } else {
                    reference_hits = Some(hits);
                }

                let mut frame = String::new();
                for row in 0..canvas.rows {
                    canvas.write_row(&mut frame, row, &vec![' '; canvas.width], &[], 0, None);
                }
                if style == Style::Default {
                    use std::fmt::Write as _;
                    let _ = writeln!(samples, "\n{kind}\n{frame}");
                }
                signatures.insert(frame);
            }
            assert_eq!(
                signatures.len(),
                kinds.len(),
                "surface grain must distinguish module families for style {style_index}"
            );
        }
        if let Some(directory) = std::env::var_os("MODEL_MAVEN_GRAIN_CAPTURE_DIR") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join("module-grain-cells.txt"), samples).unwrap();
        }
    }

    #[test]
    fn clipped_strokes_match_a_crop_of_the_full_depth_tested_line() {
        const OFFSET_COL: usize = 32;
        const OFFSET_ROW: usize = 16;
        for endpoints in [
            [[-20.5, -9.5], [42.0, 19.0]],
            [[-20.5, 19.0], [42.0, -9.5]],
            [[-9.0, -22.0], [30.0, 35.0]],
            [[2.0, 35.0], [12.0, -22.0]],
            [[5.0, -30.0], [5.0, 30.0]],
            [[-30.0, 10.0], [30.0, 10.0]],
            [[5.0, 5.0], [5.0, 5.0]],
        ] {
            for order in [[0, 1], [1, 0]] {
                let mut visible = Canvas::new(16, 12);
                let mut full = Canvas::new(100, 64);
                for y in 0..visible.rows * 4 {
                    for x in 0..visible.width * 2 {
                        let sample = Some(Sample {
                            layer: usize::from(x % 7 != 0),
                            depth: ((x + y * 2) % 7) as f64 * 0.16,
                            shade: 1,
                            #[cfg(test)]
                            slice: (OperatorId(0), 0),
                        });
                        visible.samples[y * visible.width * 2 + x] = sample;
                        full.samples[(y + OFFSET_ROW * 4) * full.width * 2 + x + OFFSET_COL * 2] =
                            sample;
                    }
                }
                let point = |index: usize| {
                    let [x, y] = endpoints[index];
                    Vertex {
                        x,
                        y,
                        z: 0.18 + 0.64 * index as f64,
                        scale: 0.85 + 0.25 * index as f64,
                    }
                };
                let translate = |vertex: Vertex| Vertex {
                    x: vertex.x + OFFSET_COL as f64,
                    y: vertex.y + (OFFSET_ROW * 2) as f64,
                    ..vertex
                };
                let seam = Stroke::Surface {
                    layer: 1,
                    seam: true,
                };
                visible.stroke(point(order[0]), point(order[1]), seam);
                full.stroke(translate(point(order[0])), translate(point(order[1])), seam);
                for row in 0..visible.rows {
                    let from = (row + OFFSET_ROW) * full.width + OFFSET_COL;
                    assert_eq!(
                        &visible.seams[row * visible.width..(row + 1) * visible.width],
                        &full.seams[from..from + visible.width],
                        "{endpoints:?}, order {order:?}, row {row}",
                    );
                }
            }
        }
    }

    #[test]
    fn port_marker_remains_visible_on_an_opaque_slice_in_both_color_modes() {
        let mut canvas = Canvas::new(4, 1);
        canvas.samples.fill(Some(Sample {
            depth: 1.0,
            layer: 0,
            shade: 3,
            #[cfg(test)]
            slice: (OperatorId(0), 0),
        }));
        let points = [[0.9, 1.0], [1.5, 0.4], [2.1, 1.0], [1.5, 1.6], [0.9, 1.0]];
        for edge in points.windows(2) {
            canvas.connector(edge[0], edge[1], (true, LinkInk::Confirmed));
        }
        let render = |canvas: &Canvas, color| {
            let mut row = String::new();
            canvas.write_row(&mut row, 0, &[' '; 4], &[], 0, color);
            row
        };
        assert_eq!(render(&canvas, None), "████\n");
        canvas.mark(&points, LinkInk::Confirmed);
        let monochrome = render(&canvas, None);
        assert!(monochrome.chars().any(|ch| ('⠁'..='⣿').contains(&ch)));
        assert!(monochrome.ends_with("█\n"));
        let palette = Appearance {
            style: crate::workbench::Style::Default,
            background: crate::terminal::layout::Background::Dark,
        };
        let colored = render(&canvas, Some(palette));
        assert!(colored.contains("\x1b[38;5;157m"));
        assert_eq!(crate::terminal::text::plain_text(&colored), monochrome);
    }

    #[test]
    fn perspective_separates_near_and_far_faces_without_exaggerating_zoom() {
        let camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
        };
        for size in [1.0, 10.0, 40.0] {
            let half = [size, size, size];
            let center = [50.0, 35.0];
            let solid = Cuboid {
                center,
                half,
                elevation: 0.0,
                layer: 0,
                key: SliceKey {
                    operator: OperatorId(0),
                    ordinal: 0,
                },
                perspective: Cuboid::layer_perspective(half),
                material: crate::graph::appearance::SliceStyle::DEFAULT.material,
            };
            let vertices = solid.vertices(camera);
            let far = vertices[1].x - vertices[0].x;
            let near = vertices[5].x - vertices[4].x;
            assert!((0.95..1.0).contains(&(far / (2.0 * size))));
            assert!((1.0..1.06).contains(&(near / (2.0 * size))));
            let projected_center = camera.project([0.0; 3], center, solid.perspective);
            assert_eq!([projected_center.x, projected_center.y], center);
        }
    }

    #[test]
    fn focus_mark_is_paint_only_and_does_not_change_hit_targets() {
        let camera = Camera::default();
        let solid = Cuboid {
            center: [16.0, 10.0],
            half: [5.0, 4.0, 3.0],
            elevation: 0.0,
            layer: 7,
            key: SliceKey {
                operator: OperatorId(7),
                ordinal: 0,
            },
            perspective: Cuboid::layer_perspective([5.0, 4.0, 3.0]),
            material: crate::graph::appearance::SliceStyle::DEFAULT.material,
        };
        let mut canvas = Canvas::new(32, 20);
        canvas.cuboid(camera, solid);
        let before: Vec<_> = (0..20)
            .flat_map(|row| (0..32).map(move |column| (column, row)))
            .map(|(column, row)| canvas.layer_at(column, row))
            .collect();

        let bounds = solid.bounds(camera);
        let center = [
            (bounds[0][0] + bounds[1][0]) * 0.5,
            (bounds[0][1] + bounds[1][1]) * 0.5,
        ];
        canvas.mark(
            &crate::graph::PortMark {
                point: center,
                radius: 4.5,
                ink: LinkInk::Focus,
            }
            .points(),
            LinkInk::Focus,
        );
        assert!(
            canvas
                .marks
                .iter()
                .any(|(mask, ink)| *mask != 0 && *ink == LinkInk::Focus)
        );
        let after: Vec<_> = (0..20)
            .flat_map(|row| (0..32).map(move |column| (column, row)))
            .map(|(column, row)| canvas.layer_at(column, row))
            .collect();

        assert_eq!(before, after);
    }
}
