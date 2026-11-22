//! A rotating projection of the Penrose tribar, rasterized into Braille cells.

use std::f64::consts::TAU;
#[cfg(unix)]
use std::fmt::Write as _;

const DOTS: [[u8; 2]; 4] = [[1, 8], [2, 16], [4, 32], [64, 128]];
const BAYER: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
const DENSITY: [u8; 3] = [16, 10, 6];
const OPENING_ANGLE: f64 = TAU / 30.0;
const BREATH_DEPTH: f64 = 0.10;

#[cfg(unix)]
const SECONDS_PER_ARC: f64 = 2.2;
#[cfg(unix)]
const ACCELERATION_FRACTION: f64 = 0.382;
#[cfg(unix)]
const BREATH_SECONDS: f64 = 5.5;

#[cfg(unix)]
fn rotation_angle(seconds: f64) -> f64 {
    const SPEED_VARIATION: f64 = 0.50;

    let arcs = (seconds / SECONDS_PER_ARC) % 3.0;
    let phase = arcs.fract();
    let slow_down = 1.0 - ACCELERATION_FRACTION;
    let pi = std::f64::consts::PI;
    // Integrate two half-cosine speed ramps. The speed and acceleration
    // match at the split and where one 120-degree arc meets the next.
    let offset = if phase < ACCELERATION_FRACTION {
        -ACCELERATION_FRACTION * (pi * phase / ACCELERATION_FRACTION).sin()
    } else {
        slow_down * (pi * (phase - ACCELERATION_FRACTION) / slow_down).sin()
    };
    (TAU / 3.0) * (arcs + SPEED_VARIATION * offset / pi)
}

#[derive(Clone, Copy)]
struct Pose {
    angle: f64,
    zoom: f64,
    glow: f64,
}

impl Pose {
    fn steady(angle: f64) -> Self {
        Self {
            angle,
            zoom: 1.0,
            glow: 0.0,
        }
    }

    #[cfg(unix)]
    fn animated(seconds: f64) -> Self {
        let wave = (TAU * (seconds % BREATH_SECONDS) / BREATH_SECONDS).cos();
        Self {
            angle: OPENING_ANGLE + rotation_angle(seconds),
            zoom: 1.0 + BREATH_DEPTH * wave,
            glow: (1.0 + wave) * 0.5,
        }
    }
}

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

fn turn(point: Point, (sin, cos): (f64, f64)) -> Point {
    Point {
        x: point.x * cos - point.y * sin,
        y: point.x * sin + point.y * cos,
    }
}

/// The lower beam turns up the left side and ends at the inside of the
/// opposite beam. The two rotated copies repeat that overlap cyclically.
fn faces(angle: f64) -> [[Point; 6]; 3] {
    let lattice = |x, y: f64| Point {
        x: x * 0.5,
        y: (y - 4.0) * (3.0_f64.sqrt() * 0.5),
    };
    let face = [
        lattice(-1.0, 2.0),
        lattice(0.0, 3.0),
        lattice(-2.5, 5.5),
        lattice(6.5, 5.5),
        lattice(5.5, 6.5),
        lattice(-5.5, 6.5),
    ];
    std::array::from_fn(|index| {
        let rotation = (angle + index as f64 * TAU / 3.0).sin_cos();
        face.map(|point| turn(point, rotation))
    })
}

fn contains(polygon: &[Point], point: Point) -> bool {
    let mut inside = false;
    for (start, end) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
        if (start.y > point.y) != (end.y > point.y)
            && point.x < start.x + (point.y - start.y) * (end.x - start.x) / (end.y - start.y)
        {
            inside = !inside;
        }
    }
    inside
}

fn ink_centroid(polygons: &[[Point; 6]; 3]) -> Point {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut weighted_area = 0.0;
    for (polygon, density) in polygons.iter().zip(DENSITY) {
        for (a, b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
            let weight = f64::from(density) * (a.x * b.y - b.x * a.y);
            x += (a.x + b.x) * weight;
            y += (a.y + b.y) * weight;
            weighted_area += weight;
        }
    }
    Point {
        x: x / (3.0 * weighted_area),
        y: y / (3.0 * weighted_area),
    }
}

fn optical_shift(polygons: &[[Point; 6]; 3], scale: f64) -> Point {
    let ink = ink_centroid(polygons);
    // Counter only part of the brightness imbalance. The silhouette should
    // still stay close to its geometric pivot at every angle.
    let gain = (scale * 0.55).min(1.25 / ink.x.hypot(ink.y));
    Point {
        x: ink.x * gain,
        y: ink.y * gain,
    }
}

struct Segment {
    start: Point,
    delta: Point,
    inverse_length_squared: f64,
}

impl Segment {
    fn new(start: Point, end: Point) -> Self {
        let delta = Point {
            x: end.x - start.x,
            y: end.y - start.y,
        };
        Self {
            start,
            delta,
            inverse_length_squared: (delta.x * delta.x + delta.y * delta.y).recip(),
        }
    }

    fn within(&self, point: Point, radius: f64) -> bool {
        let x = point.x - self.start.x;
        let y = point.y - self.start.y;
        let along = (x * self.delta.x + y * self.delta.y) * self.inverse_length_squared;
        let along = along.clamp(0.0, 1.0);
        let dx = x - along * self.delta.x;
        let dy = y - along * self.delta.y;
        dx * dx + dy * dy < radius * radius
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Tone {
    #[default]
    Base,
    Rim,
    Shine,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Cell {
    dots: u8,
    face: u8,
    tone: Tone,
}

fn viewport_margin(dot_aspect: f64) -> f64 {
    // Leave room for the 1.25-dot optical shift and the quarter-dot samples.
    1.5 * dot_aspect.max(1.0)
}

fn viewport_radius(width: usize, height: usize, dot_aspect: f64) -> f64 {
    (width as f64 * dot_aspect).min(2.0 * height as f64) - viewport_margin(dot_aspect)
}

/// The earlier layout reserved room for a ±4% breath. Grow the neutral
/// radius by 30%, then reserve enough rows for the stronger peak.
#[cfg(unix)]
pub(super) fn enlarged_size(previous_height: usize, dot_aspect: f64) -> (usize, f64) {
    let previous_margin = 2.0 * dot_aspect.max(1.0);
    let baseline_radius = (2.0 * previous_height as f64 - previous_margin).max(0.0) * 1.30 / 1.04;
    let peak_with_margin = baseline_radius * (1.0 + BREATH_DEPTH) + viewport_margin(dot_aspect);
    let rows = (peak_with_margin / 2.0).ceil() as usize;
    (rows, baseline_radius)
}

impl Cell {
    fn glyph(self) -> char {
        if self.dots == 0 {
            ' '
        } else {
            char::from_u32(0x2800 + u32::from(self.dots)).expect("valid Braille pattern")
        }
    }
}

fn rasterize(
    frame: &mut [Cell],
    previous: Option<&[Cell]>,
    width: usize,
    dot_aspect: f64,
    pose: Pose,
    color: bool,
) {
    let height = frame.len() / width;
    let polygons = faces(pose.angle);
    let joins = polygons.map(|face| Segment::new(face[0], face[1]));
    let rim = polygons.map(|face| {
        [
            Segment::new(face[3], face[4]),
            Segment::new(face[4], face[5]),
        ]
    });
    // Project in physical dot units. A Braille dot can be taller or wider
    // than its horizontal neighbor, depending on the terminal font grid.
    let half_width = width as f64 * dot_aspect;
    let half_height = (height * 2) as f64;
    let scale =
        viewport_radius(width, height, dot_aspect) * pose.zoom / (3.5 * (1.0 + BREATH_DEPTH));
    let shift = optical_shift(&polygons, scale);
    let center_x = half_width - shift.x;
    let center_y = half_height - shift.y;
    let offset_y = 0.25 / scale;
    let offset_x = offset_y * dot_aspect;
    let ridge_width = 0.78 / scale;
    let light_width = pose.glow * 5.6 / scale;
    let shine_width = pose.glow.powi(2) * 2.5 / scale;
    for (index, cell) in frame.iter_mut().enumerate() {
        let col = index % width;
        let row = index / width;
        let old_dots = previous.map_or(0, |cells| cells[index].dots);
        let mut dots = 0;
        let mut hits = [0; 3];
        for (dy, pattern) in DOTS.iter().enumerate() {
            for (dx, bit) in pattern.iter().enumerate() {
                let point = Point {
                    x: (((col * 2 + dx) as f64 + 0.5) * dot_aspect - center_x) / scale,
                    y: ((row * 4 + dy) as f64 + 0.5 - center_y) / scale,
                };
                let mut coverage = [0; 3];
                for y in [point.y - offset_y, point.y + offset_y] {
                    for x in [point.x - offset_x, point.x + offset_x] {
                        if let Some(face) = polygons
                            .iter()
                            .position(|polygon| contains(polygon, Point { x, y }))
                        {
                            coverage[face] += 1;
                        }
                    }
                }
                // Keep interior dither tied to screen dots. Supersampling and
                // a two-sample hold keep a grazing edge from flashing on/off.
                let rank = BAYER[dy][(col * 2 + dx) % 4];
                let mut ink = 0;
                for (face, samples) in coverage.into_iter().enumerate().filter(|(_, n)| *n != 0) {
                    let mut density = DENSITY[face];
                    let previous_face = (face + 2) % 3;
                    if joins[face].within(point, ridge_width) {
                        // Each join touches its owner and the next face.
                        density = if face < 2 { (density + 4).min(16) } else { 3 };
                    } else if joins[previous_face].within(point, ridge_width) {
                        density = if face == 0 { 16 } else { density - 3 };
                    } else if face == 2
                        && rim[face].iter().any(|edge| edge.within(point, ridge_width))
                    {
                        density = 12;
                    } else if !color
                        && rank >= density
                        && rank < density + 4
                        && rim[face].iter().any(|edge| edge.within(point, light_width))
                    {
                        // Preserve the patterned faces in monochrome. Build
                        // the reflection from a few extra dots near the rim.
                        density += 4;
                    }
                    if rank < density {
                        ink += samples;
                        if color {
                            hits[face] += samples;
                        }
                    }
                }
                if ink >= 3 || (ink == 2 && (previous.is_none() || old_dots & bit != 0)) {
                    dots |= bit;
                }
            }
        }
        let face = if color && dots != 0 {
            hits.iter()
                .enumerate()
                .max_by_key(|(_, hits)| *hits)
                .unwrap()
                .0 as u8
        } else {
            0
        };
        // Let the reflected light grow along the silhouette. The interiors
        // keep their tone and dot pattern, even when the light is strongest.
        let tone = if color && dots != 0 && pose.glow > 0.0 {
            let center = Point {
                x: ((col * 2 + 1) as f64 * dot_aspect - center_x) / scale,
                y: ((row * 4 + 2) as f64 - center_y) / scale,
            };
            let edges = &rim[usize::from(face)];
            if edges.iter().any(|edge| edge.within(center, shine_width)) {
                Tone::Shine
            } else if edges.iter().any(|edge| edge.within(center, light_width)) {
                Tone::Rim
            } else {
                Tone::Base
            }
        } else {
            Tone::Base
        };
        *cell = Cell { dots, face, tone };
    }
}

#[cfg(unix)]
#[derive(Clone, Copy)]
pub(super) struct ArtPalette {
    pub(super) faces: [&'static str; 3],
    pub(super) rim: [&'static str; 3],
    pub(super) shine: [&'static str; 3],
}

#[cfg(unix)]
impl ArtPalette {
    fn ink(self, cell: Cell) -> &'static str {
        let colors = match cell.tone {
            Tone::Base => self.faces,
            Tone::Rim => self.rim,
            Tone::Shine => self.shine,
        };
        colors[usize::from(cell.face)]
    }
}

#[cfg(unix)]
pub(super) struct Animation {
    width: usize,
    dot_aspect: f64,
    size_fraction: f64,
    palette: Option<ArtPalette>,
    previous: Vec<Cell>,
    next: Vec<Cell>,
    patch: String,
}

#[cfg(unix)]
impl Animation {
    pub(super) fn new(
        width: usize,
        height: usize,
        dot_aspect: f64,
        baseline_radius: f64,
        palette: Option<ArtPalette>,
    ) -> Self {
        let blank = vec![Cell::default(); width * height];
        Self {
            width,
            dot_aspect,
            size_fraction: (baseline_radius * (1.0 + BREATH_DEPTH)
                / viewport_radius(width, height, dot_aspect))
            .min(1.0),
            palette,
            previous: blank.clone(),
            next: blank,
            patch: String::new(),
        }
    }

    pub(super) fn frame(&mut self, seconds: f64, top: usize, left: usize) -> &str {
        let mut pose = Pose::animated(seconds);
        pose.zoom *= self.size_fraction;
        rasterize(
            &mut self.next,
            Some(&self.previous),
            self.width,
            self.dot_aspect,
            pose,
            self.palette.is_some(),
        );
        if let Some(palette) = self.palette {
            for cell in &mut self.next {
                let face = usize::from(cell.face);
                let ink = palette.ink(*cell);
                if ink == palette.faces[face] {
                    cell.tone = Tone::Base;
                } else if ink == palette.rim[face] {
                    cell.tone = Tone::Rim;
                }
            }
        }
        self.encode_dirty(top, left);
        std::mem::swap(&mut self.previous, &mut self.next);
        &self.patch
    }

    /// Adjacent dirty cells share one cursor move. Spaces erase previous ink.
    fn encode_dirty(&mut self, top: usize, left: usize) {
        self.patch.clear();
        let mut shade = None;
        for (row, (before, after)) in self
            .previous
            .chunks_exact(self.width)
            .zip(self.next.chunks_exact(self.width))
            .enumerate()
        {
            let mut col = 0;
            while col < self.width {
                if before[col] == after[col] {
                    col += 1;
                    continue;
                }
                let _ = write!(self.patch, "\x1b[{};{}H", top + row + 1, left + col + 1);
                while col < self.width && before[col] != after[col] {
                    let cell = after[col];
                    if let Some(palette) = self.palette {
                        let foreground = palette.ink(cell);
                        if cell.dots != 0 && shade != Some(foreground) {
                            shade = Some(foreground);
                            self.patch.push_str(foreground);
                        }
                    }
                    self.patch.push(cell.glyph());
                    col += 1;
                }
            }
        }
        if shade.is_some() {
            self.patch.push_str("\x1b[0m");
        }
    }
}

pub(super) fn still(dot_aspect: f64) -> String {
    let (width, height) = (18, 8);
    let mut cells = vec![Cell::default(); width * height];
    rasterize(
        &mut cells,
        None,
        width,
        dot_aspect,
        Pose::steady(OPENING_ANGLE),
        false,
    );
    let mut picture = String::new();
    for row in cells.chunks_exact(width) {
        let end = row
            .iter()
            .rposition(|cell| cell.dots != 0)
            .map_or(0, |i| i + 1);
        picture.extend(row[..end].iter().map(|cell| cell.glyph()));
        picture.push('\n');
    }
    format!("{}\n", picture.trim_matches('\n'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rotation_keeps_three_faces_and_an_open_center() {
        let (width, height) = (28, 12);
        let mut cells = vec![Cell::default(); width * height];
        for step in 0..36 {
            rasterize(
                &mut cells,
                None,
                width,
                1.0,
                Pose::steady(f64::from(step) * TAU / 36.0),
                true,
            );
            let center = cells[height / 2 * width + width / 2];
            assert_eq!(center.dots & (DOTS[0][0] | DOTS[0][1]), 0);
            for face in 0..3 {
                assert!(
                    cells
                        .iter()
                        .filter(|cell| cell.dots != 0 && cell.face == face)
                        .count()
                        > 15
                );
            }
            for row in cells.chunks_exact(width) {
                assert_eq!(row[0], Cell::default());
                assert_eq!(row[width - 1], Cell::default());
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn dirty_cells_coalesce_and_erased_dots_write_a_space() {
        let blank = Cell::default();
        let ink = Cell { dots: 1, ..blank };
        let mut render = Animation::new(3, 1, 1.0, 1.0, None);
        render.previous.copy_from_slice(&[ink, blank, ink]);
        render.next.copy_from_slice(&[ink, ink, blank]);
        render.encode_dirty(2, 4);
        assert_eq!(render.patch, "\x1b[3;6H⠁ ");
        render.previous.copy_from_slice(&render.next);
        render.encode_dirty(2, 4);
        assert!(render.patch.is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn each_third_turn_ends_at_the_same_nonzero_speed() {
        let delta = 0.001;
        let initial = rotation_angle(delta) / delta;
        let peak_time = SECONDS_PER_ARC * ACCELERATION_FRACTION;
        let peak = (rotation_angle(peak_time + delta) - rotation_angle(peak_time)) / delta;
        assert!(initial > 0.0 && peak > 2.0 * initial);
        for arc in 1..=3 {
            let boundary = SECONDS_PER_ARC * f64::from(arc);
            let bearing = TAU * f64::from(arc) / 3.0;
            let arriving = (bearing - rotation_angle(boundary - delta)) / delta;
            let departing = (rotation_angle(boundary + delta) + if arc == 3 { TAU } else { 0.0 }
                - bearing)
                / delta;
            assert!((initial - arriving).abs() < 1e-6);
            assert!((initial - departing).abs() < 1e-6);
        }
    }

    #[test]
    fn ordered_dither_gives_three_distinct_densities() {
        for start in [0, 2] {
            for (threshold, expected) in DENSITY.into_iter().zip([8, 5, 3]) {
                let lit = BAYER
                    .iter()
                    .flat_map(|row| &row[start..start + 2])
                    .filter(|&&rank| rank < threshold)
                    .count();
                assert_eq!(lit, expected);
            }
        }
    }

    #[test]
    fn optical_correction_reduces_centroid_travel_without_moving_far() {
        for degrees in [0, 12, 90, 180, 300] {
            let polygons = faces(f64::from(degrees) * TAU / 360.0);
            for scale in [5.0, 8.0] {
                let ink = ink_centroid(&polygons);
                let shift = optical_shift(&polygons, scale);
                assert!(shift.x.hypot(shift.y) <= 1.25 + 1e-10);
                let remaining = (ink.x * scale - shift.x).hypot(ink.y * scale - shift.y);
                assert!(remaining < ink.x.hypot(ink.y) * scale * 0.55);
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn breathing_changes_size_and_adds_light_to_the_rim_in_each_mode() {
        let (width, height) = (32, 15);
        let expanded = Pose::animated(0.0);
        let contracted = Pose {
            angle: expanded.angle,
            ..Pose::animated(BREATH_SECONDS / 2.0)
        };
        assert_eq!((contracted.zoom, expanded.zoom), (0.90, 1.10));
        assert_eq!((contracted.glow, expanded.glow), (0.0, 1.0));
        assert_eq!(Pose::animated(BREATH_SECONDS).zoom, expanded.zoom);
        let blank = vec![Cell::default(); width * height];
        let mut small = blank.clone();
        let mut large = blank.clone();
        let mut mono_peak = blank.clone();
        let mut mono_unlit = blank;
        rasterize(&mut small, None, width, 1.0, contracted, true);
        rasterize(&mut large, None, width, 1.0, expanded, true);
        rasterize(&mut mono_peak, None, width, 1.0, expanded, false);
        rasterize(
            &mut mono_unlit,
            None,
            width,
            1.0,
            Pose {
                glow: 0.0,
                ..expanded
            },
            false,
        );
        let ink = |frame: &[Cell]| frame.iter().map(|cell| cell.dots.count_ones()).sum::<u32>();
        assert!(ink(&large) > ink(&small));
        assert_eq!(ink(&large), ink(&mono_unlit));
        assert!(ink(&mono_peak) > ink(&mono_unlit));
        for tone in [Tone::Base, Tone::Rim, Tone::Shine] {
            assert!(large.iter().any(|cell| cell.dots != 0 && cell.tone == tone));
        }
        assert!(
            small
                .iter()
                .chain(&mono_peak)
                .all(|cell| cell.tone == Tone::Base)
        );
    }

    #[test]
    #[cfg(unix)]
    fn reserves_room_for_the_larger_neutral_radius_and_its_breath_peak() {
        for (row_count, dot_aspect) in [(15, 1.0), (10, 18.0 / 23.0), (15, 1.35)] {
            let (rows, target_radius) = enlarged_size(row_count, dot_aspect);
            let before = (2.0 * row_count as f64 - 2.0 * dot_aspect.max(1.0)) / 1.04;
            assert!((target_radius / before - 1.30).abs() < 1e-12);
            assert!(
                2.0 * rows as f64 - viewport_margin(dot_aspect)
                    >= target_radius * (1.0 + BREATH_DEPTH)
            );
        }
    }

    #[test]
    fn half_covered_dots_keep_their_previous_state() {
        let (width, height) = (22, 10);
        let mut still = vec![Cell::default(); width * height];
        let blank = still.clone();
        let mut arriving = blank.clone();
        let mut held = blank.clone();
        let pose = Pose::steady(0.31);

        rasterize(&mut still, None, width, 1.0, pose, false);
        rasterize(&mut arriving, Some(&blank), width, 1.0, pose, false);
        rasterize(&mut held, Some(&still), width, 1.0, pose, false);

        assert_ne!(arriving, still);
        assert_eq!(held, still);
        assert!(
            arriving
                .iter()
                .zip(still)
                .all(|(entering, steady)| entering.dots & !steady.dots == 0)
        );
    }

    #[test]
    fn preserves_physical_proportions_with_different_terminal_cells() {
        let (width, height) = (44, 15);
        let mut cells = vec![Cell::default(); width * height];
        for (angle, expected) in [(0.0, 1.155), (TAU / 12.0, 0.866)] {
            for dot_aspect in [18.0 / 23.0, 1.0, 1.35] {
                rasterize(
                    &mut cells,
                    None,
                    width,
                    dot_aspect,
                    Pose::steady(angle),
                    false,
                );
                let (mut min_x, mut min_y, mut max_x, mut max_y) = (usize::MAX, usize::MAX, 0, 0);
                for (index, cell) in cells.iter().enumerate() {
                    let (col, row) = (index % width, index / width);
                    for (dy, pattern) in DOTS.iter().enumerate() {
                        for (dx, bit) in pattern.iter().enumerate() {
                            if cell.dots & bit == 0 {
                                continue;
                            }
                            let (x, y) = (col * 2 + dx, row * 4 + dy);
                            min_x = min_x.min(x);
                            min_y = min_y.min(y);
                            max_x = max_x.max(x);
                            max_y = max_y.max(y);
                        }
                    }
                }
                let screen_ratio =
                    (max_x - min_x + 1) as f64 * dot_aspect / (max_y - min_y + 1) as f64;
                assert!(
                    (screen_ratio - expected).abs() < 0.08,
                    "{angle}, {dot_aspect}: {screen_ratio}"
                );
            }
        }
    }
}
