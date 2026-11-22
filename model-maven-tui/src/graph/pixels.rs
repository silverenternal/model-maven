//! Pixel canvas for terminals that support the Kitty graphics protocol.
use super::{
    Camera, SCENE_HEADER, Scene, Wire,
    appearance::{LinkInk, xterm_rgb},
    solid::{EDGES, FACES, Vertex},
};
use crate::terminal::layout::Background;
use base64::{Engine, engine::general_purpose::STANDARD};
use flate2::{Compression, write::ZlibEncoder};
use std::{
    env,
    io::{self, Write},
};

use tiny_skia::{Color, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Stroke, Transform};

pub(super) fn supported(color: bool) -> bool {
    if !color || env::var_os("TMUX").is_some() || env::var_os("STY").is_some() {
        return false;
    }
    match env::var("MODEL_MAVEN_GRAPHICS").as_deref() {
        Ok("cells") => false,
        Ok("pixels") => true,
        _ => {
            matches!(
                env::var("TERM").as_deref(),
                Ok("xterm-ghostty" | "xterm-kitty")
            ) || env::var("TERM_PROGRAM").as_deref() == Ok("ghostty")
        }
    }
}

#[derive(Clone, Copy)]
struct Sample {
    depth: f32,
    rgba: [u8; 4],
}

impl Default for Sample {
    fn default() -> Self {
        Self {
            depth: f32::NEG_INFINITY,
            rgba: [0; 4],
        }
    }
}

#[derive(Default)]
struct Raster {
    width: usize,
    height: usize,
    scale: f64,
    samples: Vec<Sample>,
    routes: Option<Pixmap>,
    rgba: Vec<u8>,
}

#[cfg(test)]
pub(super) fn visible_layers_for_test(
    scene: &Scene,
    camera: Camera,
    active: usize,
) -> std::collections::BTreeSet<usize> {
    let mut raster = Raster::default();
    raster.render(scene, camera, active, None, true);
    scene
        .solids
        .iter()
        .filter_map(|solid| {
            let [min, max] = solid.bounds(camera);
            let rows = raster.bounds(min[1], max[1], raster.height);
            let columns = raster.bounds(min[0], max[0], raster.width);
            rows.into_iter()
                .any(|row| {
                    columns
                        .clone()
                        .any(|column| raster.samples[row * raster.width + column].rgba[3] != 0)
                })
                .then_some(solid.layer)
        })
        .collect()
}

#[cfg(test)]
pub(super) fn pixels_for_test(scene: &Scene, camera: Camera, active: usize) -> Vec<u8> {
    let mut raster = Raster::default();
    raster.render(scene, camera, active, None, true);
    raster.rgba
}

#[cfg(test)]
pub(super) fn png_for_test(scene: &Scene, camera: Camera, active: usize) -> Vec<u8> {
    let mut raster = Raster::default();
    raster.render(scene, camera, active, None, true);
    let output_width = raster.width / 2;
    let mut scanlines = Vec::with_capacity((output_width * 4 + 1) * (raster.height / 2));
    for row in raster.rgba.chunks_exact(output_width * 4) {
        scanlines.push(0);
        scanlines.extend_from_slice(row);
    }
    let mut compressed = ZlibEncoder::new(Vec::new(), Compression::default());
    compressed.write_all(&scanlines).unwrap();
    let compressed = compressed.finish().unwrap();
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&(output_width as u32).to_be_bytes());
    header.extend_from_slice(&((raster.rgba.len() / (output_width * 4)) as u32).to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    png_chunk(&mut png, b"IHDR", &header);
    png_chunk(&mut png, b"IDAT", &compressed);
    png_chunk(&mut png, b"IEND", &[]);
    png
}

#[cfg(test)]
fn png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let crc = kind.iter().chain(data).fold(!0_u32, |crc, byte| {
        let mut value = crc ^ u32::from(*byte);
        for _ in 0..8 {
            value = (value >> 1) ^ (0xedb8_8320 & (0_u32.wrapping_sub(value & 1)));
        }
        value
    });
    png.extend_from_slice(&(!crc).to_be_bytes());
}

impl Raster {
    fn reset(&mut self, columns: usize, rows: usize, fine: bool) {
        // Two samples per axis in the final texture. Bound work independently
        // of font size and scale resolution down for very large viewports.
        let area = (columns * rows * 2) as f64;
        let limit = (300_000.0 / area).sqrt() as usize;
        let pixels_per_cell = limit.clamp(4, if fine { 9 } else { 6 });
        let samples_per_cell = pixels_per_cell * 2;
        self.width = columns * samples_per_cell;
        self.height = rows * samples_per_cell * 2;
        self.scale = samples_per_cell as f64;
        self.samples
            .resize(self.width * self.height, Sample::default());
        self.samples.fill(Sample::default());
    }

    fn bounds(&self, low: f64, high: f64, axis_limit: usize) -> std::ops::Range<usize> {
        let start = (low * self.scale).floor() as usize;
        let end = ((high * self.scale).ceil() as usize).min(axis_limit);
        start..end
    }

    fn triangle(&mut self, [a, b, c]: [Vertex; 3], rgb: [u8; 3], light: [f64; 2]) {
        let area = a.edge(b, c.x, c.y);
        if area.abs() < 0.001 {
            return;
        }
        let xs = self.bounds(a.x.min(b.x).min(c.x), a.x.max(b.x).max(c.x), self.width);
        let top = a.y.min(b.y).min(c.y);
        let bottom = a.y.max(b.y).max(c.y);
        let ys = self.bounds(top, bottom, self.height);
        for row in ys {
            let y = (row as f64 + 0.5) / self.scale;
            let shade = 1.04 - 0.15 * ((y - light[0]) / light[1]).clamp(0.0, 1.0);
            let [r, g, b_color] = rgb.map(|value| (f64::from(value) * shade) as u8);
            for col in xs.clone() {
                let x = (col as f64 + 0.5) / self.scale;
                let wa = b.edge(c, x, y) / area;
                let wb = c.edge(a, x, y) / area;
                let wc = 1.0 - wa - wb;
                if wa.min(wb).min(wc) < -1e-6 {
                    continue;
                }
                let (wa, wb, wc) = (wa * a.scale, wb * b.scale, wc * c.scale);
                let depth = ((a.z * wa + b.z * wb + c.z * wc) / (wa + wb + wc)) as f32;
                let sample = &mut self.samples[row * self.width + col];
                if depth > sample.depth {
                    *sample = Sample {
                        depth,
                        rgba: [r, g, b_color, 255],
                    };
                }
            }
        }
    }

    fn stroke(&mut self, a: Vertex, b: Vertex, width: f64, rgba: [u8; 4]) {
        let r = width * 0.5;
        let ys = self.bounds(a.y.min(b.y) - r, a.y.max(b.y) + r, self.height);
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let length2 = dx * dx + dy * dy;
        if length2 <= 1e-9 {
            return;
        }
        for row in ys {
            let y = (row as f64 + 0.5) / self.scale;
            // Limit the scanline to the part of the segment that can cover it.
            let (x0, x1) = if dy.abs() < 1e-9 {
                (a.x, b.x)
            } else {
                let at_y = |y: f64| a.x + dx * ((y - a.y) / dy).clamp(0.0, 1.0);
                (at_y(y - r), at_y(y + r))
            };
            for col in self.bounds(x0.min(x1) - r, x0.max(x1) + r, self.width) {
                let x = (col as f64 + 0.5) / self.scale;
                let t = (((x - a.x) * dx + (y - a.y) * dy) / length2).clamp(0.0, 1.0);
                let ox = x - (a.x + t * dx);
                let oy = y - (a.y + t * dy);
                if ox * ox + oy * oy > r * r {
                    continue;
                }
                let sample = &mut self.samples[row * self.width + col];
                let (wa, wb) = ((1.0 - t) * a.scale, t * b.scale);
                let depth = ((wa * a.z + wb * b.z) / (wa + wb)) as f32;
                if depth + 0.18 >= sample.depth {
                    sample.rgba = rgba;
                }
            }
        }
    }

    fn particle(&mut self, point: Vertex, size: f64, [r, g, b]: [u8; 3]) {
        // Stamp square pixels at projected surface points. Depth testing keeps
        // the texture on the exposed slices as the camera or stack moves.
        let extent = size.max(0.65 / self.scale);
        let xs = self.bounds(point.x - extent, point.x + extent, self.width);
        for row in self.bounds(point.y - extent, point.y + extent, self.height) {
            for column in xs.clone() {
                let sample = &mut self.samples[row * self.width + column];
                if (sample.depth - point.z as f32).abs() <= 0.20 {
                    sample.rgba = [r, g, b, 255];
                }
            }
        }
    }

    fn render(
        &mut self,
        scene: &Scene,
        camera: Camera,
        active: usize,
        hovered: Option<Wire>,
        fine: bool,
    ) {
        self.reset(scene.width, scene.labels.len() - 1, fine);
        self.draw_routes(scene, active, hovered);
        let hovered = hovered.and_then(|wire| {
            scene
                .handoffs
                .iter()
                .find(|edge| edge.wire == wire)
                .map(|edge| edge.layers)
        });
        let light = scene.appearance.background == Background::Light;

        let projected: Vec<_> = scene
            .solids
            .iter()
            .map(|solid| {
                let mut pigment = solid.material.rgb;
                let selected = hovered.map_or(solid.layer == active, |layers| {
                    layers.contains(&solid.layer)
                });
                if hovered.is_some_and(|[_, to]| to == solid.layer) {
                    pigment = [171, 162, 227];
                }
                if light {
                    pigment = tint(pigment, if selected { 0.68 } else { 0.82 }, 0);
                } else if selected {
                    pigment = pigment.map(|channel| channel.saturating_add(22));
                }
                (
                    solid,
                    pigment,
                    selected,
                    solid.vertices(camera),
                    solid.visible_faces(camera),
                )
            })
            .collect();
        for &(_, pigment, _, vertices, visible) in &projected {
            for (([a, b, c, d], shade), shown) in FACES.into_iter().zip(visible) {
                if shown {
                    let tone = tint(pigment, [0.34, 0.50, 0.65, 0.79][shade], 0);
                    let ys = [a, b, c, d].map(|vertex| vertices[vertex].y);
                    let top = ys.into_iter().fold(f64::INFINITY, f64::min);
                    let bottom = ys.into_iter().fold(f64::NEG_INFINITY, f64::max);
                    let light = [top, (bottom - top).max(0.001)];
                    self.triangle([vertices[a], vertices[b], vertices[c]], tone, light);
                    self.triangle([vertices[a], vertices[c], vertices[d]], tone, light);
                }
            }
        }
        for &(solid, pigment, selected, vertices, visible) in &projected {
            if fine || selected {
                solid.grain(camera, |point, bright| {
                    let rgb = if bright {
                        tint(pigment, 1.05, 24)
                    } else {
                        tint(pigment, 0.57, 2)
                    };
                    self.particle(point, solid.material.grain_size, rgb);
                });
            }
            for (a, b, left, right) in EDGES {
                let adjoining = u8::from(visible[left]) + u8::from(visible[right]);
                if adjoining == 0 {
                    continue;
                }
                let [r, g, b_ink] = tint(pigment, if selected { 1.04 } else { 0.92 }, 12);
                self.stroke(
                    vertices[a],
                    vertices[b],
                    if adjoining == 2 { 0.10 } else { 0.115 },
                    [r, g, b_ink, 255],
                );
            }
        }
        self.resolve();
    }

    fn draw_routes(&mut self, scene: &Scene, active: usize, hovered: Option<Wire>) {
        let (width, height) = (self.width as u32 / 2, self.height as u32 / 2);
        if self
            .routes
            .as_ref()
            .is_none_or(|pixmap| pixmap.width() != width || pixmap.height() != height)
        {
            self.routes = Pixmap::new(width, height);
        }
        let Some(pixmap) = self.routes.as_mut() else {
            return;
        };
        pixmap.fill(Color::TRANSPARENT);
        let scale = (self.scale * 0.5) as f32;
        let transform = Transform::from_scale(scale, scale);
        let mut stroke = |path: PathBuilder, [r, g, b, a]: [u8; 4], width: f32| {
            let Some(path) = path.finish() else { return };
            let mut paint = Paint::default();
            paint.set_color_rgba8(r, g, b, a);
            let style = Stroke {
                width,
                line_cap: LineCap::Round,
                line_join: LineJoin::Round,
                ..Stroke::default()
            };
            pixmap.stroke_path(&path, &paint, &style, transform, None);
        };
        let ink = |role, focused, alpha| {
            let [r, g, b] = xterm_rgb(scene.appearance.link(role, focused));
            [r, g, b, alpha]
        };
        if let Some(drawing) = &scene.axes {
            for edge in &drawing.strokes {
                let [[x, y], [end_x, end_y]] = edge.segment;
                let mut path = PathBuilder::new();
                path.move_to(x as f32, y as f32);
                path.line_to(end_x as f32, end_y as f32);
                let (opacity, width) = if edge.guide {
                    (145, 0.14)
                } else {
                    (255, scene.appearance.focused_stroke())
                };
                stroke(path, ink(edge.ink, !edge.guide, opacity), width);
            }
            return;
        }
        for group in &scene.groups {
            let [start, rest @ ..] = group.points();
            let mut path = PathBuilder::new();
            path.move_to(start[0] as f32, start[1] as f32);
            for point in rest {
                path.line_to(point[0] as f32, point[1] as f32);
            }
            stroke(path, ink(LinkInk::Structure, false, 170), 0.13);
        }
        for link in &scene.links {
            let highlighted = link.layers.contains(&active);
            let [a, b, c, d] = link.points;
            let mut path = PathBuilder::new();
            path.move_to(a[0] as f32, a[1] as f32);
            path.cubic_to(
                b[0] as f32,
                b[1] as f32,
                c[0] as f32,
                c[1] as f32,
                d[0] as f32,
                d[1] as f32,
            );
            let tone = ink(
                LinkInk::Structure,
                highlighted,
                if highlighted { 240 } else { 205 },
            );
            stroke(
                path,
                tone,
                if highlighted {
                    scene.appearance.focused_stroke()
                } else {
                    0.16
                },
            );
        }
        for route in &scene.observed_routes {
            let mut path = PathBuilder::new();
            route.draw(|from, to| {
                path.move_to(from[0] as f32, from[1] as f32);
                path.line_to(to[0] as f32, to[1] as f32);
            });
            let focused = route.target == active;
            stroke(
                path,
                ink(LinkInk::Unknown, focused, 190),
                if focused {
                    scene.appearance.focused_stroke()
                } else {
                    0.16
                },
            );
        }
        let wire_focused = |handoff: &&super::Handoff| {
            hovered.map_or_else(
                || handoff.layers.contains(&active),
                |wire| wire == handoff.wire,
            )
        };
        let wires = scene.handoffs.iter();
        for handoff in wires
            .clone()
            .filter(|edge| !wire_focused(edge))
            .chain(wires.filter(wire_focused))
        {
            let focused = wire_focused(&handoff);
            let role = LinkInk::from(handoff.alignment);
            let a = if focused || role == LinkInk::Mismatch {
                255
            } else {
                225
            };
            let mut path = PathBuilder::new();
            let mut last = None;
            handoff.draw(hovered, |from, to| {
                if last != Some(from) {
                    path.move_to(from[0] as f32, from[1] as f32);
                }
                path.line_to(to[0] as f32, to[1] as f32);
                last = Some(to);
            });
            stroke(
                path,
                ink(role, focused, a),
                if focused {
                    scene.appearance.focused_stroke()
                } else {
                    0.19
                },
            );
        }
        if let Some(candidate) = &scene.candidate {
            let mut path = PathBuilder::new();
            candidate.draw(None, |from, to| {
                path.move_to(from[0] as f32, from[1] as f32);
                path.line_to(to[0] as f32, to[1] as f32);
            });
            stroke(
                path,
                ink(LinkInk::Draft, true, 245),
                scene.appearance.focused_stroke(),
            );
        }
        if let Some(mark) = scene.focus_mark {
            let [first, rest @ ..] = mark.points();
            let mut path = PathBuilder::new();
            path.move_to(first[0] as f32, first[1] as f32);
            for [x, y] in rest {
                path.line_to(x as f32, y as f32);
            }
            stroke(
                path,
                ink(mark.ink, true, 210),
                scene.appearance.focused_stroke() * 0.72,
            );
        }
        if let Some(mark) = scene.edit_mark {
            let [first, rest @ ..] = mark.points();
            let mut path = PathBuilder::new();
            path.move_to(first[0] as f32, first[1] as f32);
            for [x, y] in rest {
                path.line_to(x as f32, y as f32);
            }
            stroke(
                path,
                ink(mark.ink, true, 255),
                scene.appearance.focused_stroke() + 0.06,
            );
        }
    }

    fn resolve(&mut self) {
        self.rgba.clear();
        self.rgba.reserve(self.width * self.height);
        for pair in self.samples.chunks_exact(self.width * 2) {
            let (top, bottom) = pair.split_at(self.width);
            for (top, bottom) in top.chunks_exact(2).zip(bottom.chunks_exact(2)) {
                let rgba = top[0].rgba;
                if rgba == top[1].rgba && rgba == bottom[0].rgba && rgba == bottom[1].rgba {
                    self.rgba.extend_from_slice(&rgba);
                    continue;
                }
                let mut rgb = [0_u32; 3];
                let mut alpha = 0;
                for sample in [top[0], top[1], bottom[0], bottom[1]] {
                    let opacity = u32::from(sample.rgba[3]);
                    alpha += opacity;
                    for (component, channel) in rgb.iter_mut().zip(sample.rgba) {
                        *component += u32::from(channel) * opacity;
                    }
                }
                let [r, g, b] = rgb.map(|value| (value / alpha.max(1)) as u8);
                self.rgba
                    .extend_from_slice(&[r, g, b, ((alpha + 2) / 4) as u8]);
            }
        }
        if let Some(routes) = &self.routes {
            for (pixel, under) in self.rgba.chunks_exact_mut(4).zip(routes.pixels()) {
                let foreground = u32::from(pixel[3]);
                if foreground == 255 || under.alpha() == 0 {
                    continue;
                }
                let remainder = 255 - foreground;
                let alpha = foreground * 255 + u32::from(under.alpha()) * remainder;
                for (channel, base) in
                    pixel
                        .iter_mut()
                        .zip([under.red(), under.green(), under.blue()])
                {
                    let color = u32::from(*channel) * foreground + u32::from(base) * remainder;
                    *channel = ((color * 255 + alpha / 2) / alpha).min(255) as u8;
                }
                pixel[3] = ((alpha + 127) / 255) as u8;
            }
        }
    }
}

fn tint(rgb: [u8; 3], gain: f64, lift: u8) -> [u8; 3] {
    rgb.map(|channel| ((f64::from(channel) * gain) as u8).saturating_add(lift))
}

#[derive(Default)]
pub(super) struct Display {
    raster: Raster,
    next_id: u32,
    visible_id: Option<u32>,
    encoded: String,
}

impl Display {
    pub fn prepare(
        &mut self,
        scene: &Scene,
        camera: Camera,
        active: usize,
        hovered: Option<Wire>,
        fine: bool,
    ) -> io::Result<()> {
        self.rasterize(scene, camera, active, hovered, fine);
        self.encode_raster()
    }

    pub(super) fn rasterize(
        &mut self,
        scene: &Scene,
        camera: Camera,
        active: usize,
        hovered: Option<Wire>,
        fine: bool,
    ) {
        self.raster.render(scene, camera, active, hovered, fine);
    }

    pub(super) fn encode_raster(&mut self) -> io::Result<()> {
        let mut compressed = ZlibEncoder::new(Vec::new(), Compression::new(3));
        compressed.write_all(&self.raster.rgba)?;
        self.encoded.clear();
        STANDARD.encode_string(compressed.finish()?, &mut self.encoded);
        Ok(())
    }

    pub fn place(&mut self, output: &mut impl Write, scene: &Scene) -> io::Result<()> {
        // Use independent IDs so a partly written frame never replaces the
        // visible image. Both IDs are owned by this alternate-screen view.
        const FIRST_ID: u32 = 1_916_018_118;
        let id = FIRST_ID + self.next_id;
        let top = SCENE_HEADER + scene.tooltip_rows + 1;
        let columns = scene.width;
        let rows = scene.labels.len() - 1;
        let width = self.raster.width / 2;
        let height = self.raster.height / 2;
        write!(output, "\x1b[{top};1H")?;
        let mut chunks = self.encoded.as_bytes().chunks(4096).peekable();
        let mut first = true;
        while let Some(chunk) = chunks.next() {
            let more = u8::from(chunks.peek().is_some());
            if first {
                write!(
                    output,
                    "\x1b_Ga=T,t=d,f=32,o=z,s={width},v={height},c={columns},r={rows},i={id},q=2,C=1,z=-1,m={more};"
                )?;
                first = false;
            } else {
                write!(output, "\x1b_Gm={more};")?;
            }
            output.write_all(chunk)?;
            output.write_all(b"\x1b\\")?;
        }
        if let Some(previous) = self.visible_id.replace(id) {
            Self::delete(output, previous)?;
        }
        self.next_id ^= 1;
        Ok(())
    }

    fn delete(output: &mut impl Write, id: u32) -> io::Result<()> {
        write!(output, "\x1b_Ga=d,d=I,i={id},q=2;\x1b\\")
    }

    pub fn clear(&mut self, output: &mut impl Write) -> io::Result<()> {
        if let Some(id) = self.visible_id {
            Self::delete(output, id)?;
            self.visible_id = None;
        }
        Ok(())
    }
}

impl Drop for Display {
    fn drop(&mut self) {
        if let Some(id) = self.visible_id {
            let mut output = io::stderr().lock();
            let _ = Self::delete(&mut output, id);
            let _ = output.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EditOverlay, ModelView, Shape, axes::Comparison, wire};
    use crate::terminal::locale::Locale;
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    #[test]
    fn pixel_frames_are_antialiased_chunked_and_replace_the_old_image() {
        let view = ModelView::new(None);
        let scene = view.scene(80, 24, true, Some(0.0));
        let mut display = Display::default();
        display.prepare(&scene, view.camera, 0, None, true).unwrap();
        let mut output = Vec::new();
        display.place(&mut output, &scene).unwrap();
        let first = String::from_utf8(output).unwrap();
        let mut encoded = String::new();
        for chunk in first.split("\x1b_G").skip(1) {
            let (_, payload) = chunk.split_once(';').unwrap();
            let payload = payload.strip_suffix("\x1b\\").unwrap();
            assert!(payload.len() <= 4096);
            encoded.push_str(payload);
        }
        let compressed = STANDARD.decode(encoded).unwrap();
        let mut image = Vec::new();
        ZlibDecoder::new(compressed.as_slice())
            .read_to_end(&mut image)
            .unwrap();
        assert_eq!(image, display.raster.rgba);
        assert!(image.len() > scene.width * scene.labels.len() * 100);
        assert!(
            image
                .chunks_exact(4)
                .any(|pixel| (1..=254).contains(&pixel[3]))
        );
        assert!(image.chunks_exact(4).any(|pixel| pixel[3] == 255));

        let old = display.visible_id.unwrap();
        let mut output = Vec::new();
        display.place(&mut output, &scene).unwrap();
        assert_ne!(display.visible_id.take(), Some(old));
        assert!(
            String::from_utf8(output)
                .unwrap()
                .ends_with(&format!("\x1b_Ga=d,d=I,i={old},q=2;\x1b\\"))
        );
    }

    #[test]
    fn axis_placement_keeps_the_baseline_and_widths_in_the_rgba_image() {
        let view = ModelView::new(None);
        let observed = Shape::Dimensions(vec![Some(2), Some(48)]);
        let expected = Shape::Features(32);
        let scene = Scene::for_axes(
            view.appearance(),
            80,
            24,
            Comparison::from_shapes([Some(&observed), Some(&expected)], None, 8),
            true,
            Locale::English,
        );
        let mut raster = Raster::default();
        raster.render(&scene, Camera::default(), 0, None, true);
        let width = raster.width / 2;
        let split = raster.height / 4;
        let scan = |rows: std::ops::Range<usize>| {
            let columns = rows.flat_map(|row| {
                let at = row * width * 4;
                raster.rgba[at..at + width * 4]
                    .chunks_exact(4)
                    .enumerate()
                    .filter_map(|(x, pixel)| (pixel[3] > 200).then_some(x))
            });
            columns.fold((usize::MAX, 0), |(left, right), x| {
                (left.min(x), right.max(x))
            })
        };
        let (a_start, a_end) = scan(0..split);
        let (b_start, b_end) = scan(split..raster.height / 2);
        assert!(a_start < a_end && b_start < b_end);
        assert!(a_start.abs_diff(b_start) <= 1);
        let ratio = (a_end - a_start) as f64 / (b_end - b_start) as f64;
        assert!((ratio - 1.5).abs() < 0.02, "rendered ratio is {ratio}");
    }

    #[test]
    fn commit_emphasis_changes_only_pixels_next_to_the_edited_input() {
        let mut view = ModelView::new(None);
        let edited = wire(2, 4, 1);
        let other = wire(1, 4, 0);
        let mut raster = Raster::default();
        let base = view.scene(100, 30, true, None);
        raster.render(&base, view.camera, view.selected, Some(other), true);
        let original = raster.rgba.clone();
        view.edit_overlay = Some(EditOverlay::Confirmed {
            target: edited.to,
            expanded: true,
        });
        let expanded = view.scene(100, 30, true, None);
        let mark = expanded.edit_mark.unwrap();
        let target = expanded
            .handoffs
            .iter()
            .find(|route| route.wire == edited)
            .unwrap()
            .to;
        assert_eq!(mark.point, target);
        raster.render(&expanded, view.camera, view.selected, Some(other), true);
        let width = raster.width / 2;
        let scale = raster.scale / 2.0;
        let mut changed = 0;
        for (pixel, (before, after)) in original
            .chunks_exact(4)
            .zip(raster.rgba.chunks_exact(4))
            .enumerate()
        {
            if before != after {
                changed += 1;
                let x = (pixel % width) as f64 / scale;
                let y = (pixel / width) as f64 / scale;
                let margin = mark.radius + 0.4;
                assert!((x - target[0]).abs() <= margin && (y - target[1]).abs() <= margin);
            }
        }
        assert!(changed > 0);
        view.edit_overlay = Some(EditOverlay::Confirmed {
            target: edited.to,
            expanded: false,
        });
        let settled = view.scene(100, 30, true, None).edit_mark.unwrap();
        assert_eq!(settled.point, target);
        assert!(settled.radius < mark.radius);
    }
}
