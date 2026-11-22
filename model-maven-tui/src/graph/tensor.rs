//! Tensor interfaces and the display geometry of a declared handoff.
pub(super) use model_maven_tui::graph_document::Shape;
use model_maven_tui::graph_document::Wire;
use serde_json::Value;

#[derive(Default)]
pub(super) struct Interface {
    pub input: Option<Shape>,
    pub output: Option<Shape>,
}

impl Interface {
    pub fn from_record(record: &Value, kind: &str) -> Self {
        let mut input = Shape::parse(&record["input_shape"]);
        let mut output = Shape::parse(&record["output_shape"]);
        // Linear preserves every leading axis. A weight matrix declares only
        // the final axis, and its order is [out_features, in_features].
        if matches!(
            kind,
            "Linear" | "torch.nn.Linear" | "torch.nn.modules.linear.Linear"
        ) {
            let matrix = record["weight_shape"]
                .as_array()
                .filter(|axes| axes.len() == 2);
            input = input.or_else(|| {
                record["in_features"]
                    .as_u64()
                    .or_else(|| matrix?[1].as_u64())
                    .map(Shape::Features)
            });
            output = output.or_else(|| {
                record["out_features"]
                    .as_u64()
                    .or_else(|| matrix?[0].as_u64())
                    .map(Shape::Features)
            });
        }
        Self { input, output }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Alignment {
    Match,
    Features,
    Mismatch,
    Unknown,
}

impl Alignment {
    pub fn compare(output: Option<&Shape>, input: Option<&Shape>) -> Self {
        match (output, input) {
            (Some(Shape::Dimensions(a)), Some(Shape::Dimensions(b))) => {
                if a.len() != b.len()
                    || a.iter()
                        .zip(b)
                        .any(|(&a, &b)| a.zip(b).is_some_and(|(a, b)| a != b))
                {
                    Self::Mismatch
                } else if a.iter().chain(b).all(Option::is_some) {
                    Self::Match
                } else {
                    Self::Unknown
                }
            }
            (Some(Shape::Dimensions(axes)), Some(Shape::Features(_)))
            | (Some(Shape::Features(_)), Some(Shape::Dimensions(axes)))
                if axes.is_empty() =>
            {
                Self::Mismatch
            }
            (Some(a), Some(b)) => match a.last().zip(b.last()) {
                Some((a, b)) if a == b => Self::Features,
                Some(_) => Self::Mismatch,
                None => Self::Unknown,
            },
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone)]
pub(super) struct Handoff {
    pub wire: Wire,
    #[cfg_attr(not(unix), allow(dead_code))]
    pub displayed: Wire,
    pub layers: [usize; 2],
    pub alignment: Alignment,
    pub from: [f64; 2],
    pub to: [f64; 2],
    pub continuation: Option<Continuation>,
    pub orthogonal: bool,
    pub bend: Option<f64>,
    pub vertical: bool,
    crossings: Vec<Crossing>,
}

#[derive(Clone, Copy)]
struct Crossing {
    at: f64,
    half_gap: f64,
    other: Wire,
    below: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Continuation {
    Incoming,
    Outgoing,
}

impl Handoff {
    pub fn new(
        wire: Wire,
        layers: [usize; 2],
        endpoints: [[f64; 2]; 2],
        output: Option<&Shape>,
        input: Option<&Shape>,
        continuation: Option<Continuation>,
    ) -> Self {
        Self {
            wire,
            displayed: wire,
            layers,
            alignment: Alignment::compare(output, input),
            from: endpoints[0],
            to: endpoints[1],
            continuation,
            orthogonal: false,
            bend: None,
            vertical: false,
            crossings: Vec::new(),
        }
    }

    #[cfg(test)]
    pub fn crossing_marks(&self) -> usize {
        self.crossings.len()
    }

    pub fn separate_crossings(routes: &mut [Self]) {
        const STEPS: usize = 12;
        if routes.len() < 2 {
            return;
        }
        let samples: Vec<_> = routes
            .iter()
            .map(|route| {
                std::array::from_fn::<_, { STEPS + 1 }, _>(|i| {
                    route.sample(i as f64 / STEPS as f64)
                })
            })
            .collect();
        for i in 0..routes.len() {
            let (first, remaining) = routes.split_at_mut(i + 1);
            let a = &mut first[i];
            for (offset, b) in remaining.iter_mut().enumerate() {
                if a.wire == b.wire || !Self::bounds_overlap([a.from, a.to], [b.from, b.to]) {
                    continue;
                }
                for (a_step, starts) in samples[i].windows(2).enumerate() {
                    for (b_step, ends) in samples[i + offset + 1].windows(2).enumerate() {
                        let Some([a_part, b_part]) =
                            Self::intersection([starts[0], starts[1]], [ends[0], ends[1]])
                        else {
                            continue;
                        };
                        let a_at = (a_step as f64 + a_part) / STEPS as f64;
                        let b_at = (b_step as f64 + b_part) / STEPS as f64;
                        // A shared socket is a junction. Reserve gaps for crossings
                        // with room between the intersection and both sockets.
                        let a_clearance = a_at.min(1.0 - a_at) * a.length();
                        let b_clearance = b_at.min(1.0 - b_at) * b.length();
                        if a_clearance < 1.6 || b_clearance < 1.6 {
                            continue;
                        }
                        let a_below = a.wire > b.wire;
                        a.add_crossing(a_at, b.wire, a_below);
                        b.add_crossing(b_at, a.wire, !a_below);
                    }
                }
            }
        }
    }

    fn add_crossing(&mut self, at: f64, other: Wire, below: bool) {
        if self
            .crossings
            .iter()
            .any(|entry| entry.other == other && (entry.at - at).abs() < 0.001)
        {
            return;
        }
        let half_gap = 0.85 / self.length();
        self.crossings.push(Crossing {
            at,
            half_gap,
            other,
            below,
        });
    }

    fn bounds_overlap(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> bool {
        (0..2).all(|axis| {
            a[0][axis].min(a[1][axis]) <= b[0][axis].max(b[1][axis])
                && b[0][axis].min(b[1][axis]) <= a[0][axis].max(a[1][axis])
        })
    }

    fn intersection(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> Option<[f64; 2]> {
        if !Self::bounds_overlap(a, b) {
            return None;
        }
        let delta = |from: [f64; 2], to: [f64; 2]| [to[0] - from[0], to[1] - from[1]];
        let cross = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
        let [u, v] = [delta(a[0], a[1]), delta(b[0], b[1])];
        let determinant = cross(u, v);
        if determinant.abs() <= 0.15 * u[0].hypot(u[1]) * v[0].hypot(v[1]) {
            return None;
        }
        let offset = delta(a[0], b[0]);
        let parts = [
            cross(offset, v) / determinant,
            cross(offset, u) / determinant,
        ];
        parts
            .into_iter()
            .all(|part| (0.0..=1.0).contains(&part))
            .then_some(parts)
    }

    #[cfg(unix)]
    pub fn socket_distance_at(&self, point: [f64; 2], layer: Option<usize>) -> Option<f64> {
        let [x, y] = point;
        [
            (self.from, self.layers[0], Continuation::Incoming),
            (self.to, self.layers[1], Continuation::Outgoing),
        ]
        .into_iter()
        .filter(|&(_, owner, offscreen)| {
            self.continuation != Some(offscreen) && layer.is_none_or(|visible| visible == owner)
        })
        .filter(|&([px, py], _, _)| {
            px.floor() == x.floor() && (py / 2.0).floor() == (y / 2.0).floor()
        })
        .map(|([px, py], _, _)| (px - x).hypot(py - y))
        .min_by(f64::total_cmp)
    }

    #[cfg(unix)]
    pub fn nearby_socket_distance_at(&self, point: [f64; 2]) -> Option<f64> {
        [
            (self.from, Continuation::Incoming),
            (self.to, Continuation::Outgoing),
        ]
        .into_iter()
        .filter(|&(_, offscreen)| self.continuation != Some(offscreen))
        .map(|([x, y], _)| (x - point[0]).hypot((y - point[1]) / 2.0))
        .filter(|&distance| distance <= 1.35)
        .min_by(f64::total_cmp)
    }

    #[cfg(unix)]
    pub fn distance_at(&self, point: [f64; 2]) -> Option<f64> {
        let [[ax, ay], [bx, by]] = [self.from, self.to];
        let [x, y] = point;
        const RADIUS: f64 = 1.25;
        if x < ax.min(bx) - RADIUS
            || x > ax.max(bx) + RADIUS
            || y < ay.min(by) - RADIUS
            || y > ay.max(by) + RADIUS
        {
            return None;
        }
        let mut distance: f64 = RADIUS.powi(2);
        let mut hit = false;
        for (a, b, _) in self.segments(self.length()) {
            let [dx, dy] = [b[0] - a[0], b[1] - a[1]];
            if dx * dx + dy * dy < f64::EPSILON {
                continue;
            }
            let [px, py] = [point[0] - a[0], point[1] - a[1]];
            let fraction = ((px * dx + py * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
            let squared = (px - fraction * dx).powi(2) + (py - fraction * dy).powi(2);
            if squared <= distance {
                distance = squared;
                hit = true;
            }
        }
        hit.then(|| distance.sqrt())
    }

    fn length(&self) -> f64 {
        if self.orthogonal {
            (self.to[0] - self.from[0]).abs() + (self.to[1] - self.from[1]).abs()
        } else {
            (self.to[0] - self.from[0]).hypot(self.to[1] - self.from[1])
        }
    }

    fn sample(&self, t: f64) -> [f64; 2] {
        let [a, b] = [self.from, self.to];
        if self.orthogonal {
            let axis = usize::from(self.vertical);
            let cross = 1 - axis;
            let middle = self.bend.unwrap_or((a[axis] + b[axis]) * 0.5);
            let mut point = a;
            return if t < 0.25 {
                point[axis] = a[axis] + (middle - a[axis]) * t * 4.0;
                point
            } else if t < 0.75 {
                point[axis] = middle;
                point[cross] = a[cross] + (b[cross] - a[cross]) * (t - 0.25) * 2.0;
                point
            } else {
                point[axis] = middle + (b[axis] - middle) * (t - 0.75) * 4.0;
                point[cross] = b[cross];
                point
            };
        }
        let blend = t * t * (3.0 - 2.0 * t);
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * blend]
    }

    fn segments(&self, length: f64) -> impl Iterator<Item = ([f64; 2], [f64; 2], f64)> + '_ {
        let steps = ((length * 3.0) as usize).clamp(8, 180);
        let steps = if self.orthogonal {
            (steps / 4) * 4
        } else {
            steps
        };
        (0..steps).filter(move |_| length > 0.12).map(move |step| {
            let start = step as f64 / steps as f64;
            let end = (step + 1) as f64 / steps as f64;
            (self.sample(start), self.sample(end), (start + end) * 0.5)
        })
    }

    pub fn draw(&self, selected: Option<Wire>, mut stroke: impl FnMut([f64; 2], [f64; 2])) {
        let sample = |t| self.sample(t);
        let length = self.length();
        if length <= 0.12 {
            return;
        }
        let hidden = |t: f64| {
            selected != Some(self.wire)
                && self.crossings.iter().any(|crossing| {
                    (crossing.below || selected == Some(crossing.other))
                        && (crossing.at - t).abs() <= crossing.half_gap
                })
        };
        for (from, to, midpoint) in self.segments(length) {
            if hidden(midpoint) {
                continue;
            }
            match self.alignment {
                Alignment::Mismatch if (0.37..0.63).contains(&midpoint) => continue,
                Alignment::Unknown if ((midpoint * length) as usize) % 2 == 1 => continue,
                _ => stroke(from, to),
            }
        }
        // A pinned connection gets a larger arrowhead so its direction remains
        // visible in monochrome Braille, where link colors are unavailable.
        let size = if selected == Some(self.wire) {
            0.85_f64
        } else {
            0.50_f64
        }
        .min(length * 0.18);
        if self.alignment == Alignment::Mismatch {
            if let Some(t) = [0.50, 0.44, 0.56].into_iter().find(|&t| !hidden(t)) {
                let at = sample(t);
                for side in [-1.0, 1.0] {
                    stroke(
                        [at[0] - size, at[1] - size * side],
                        [at[0] + size, at[1] + size * side],
                    );
                }
            }
        } else if let Some(t) = if self.orthogonal {
            [0.85, 0.60, 0.32]
        } else {
            [0.60, 0.75, 0.32]
        }
        .into_iter()
        .find(|&t| {
            let before = sample(t - 0.08);
            let tip = sample(t);
            !hidden(t)
                && !hidden(t - 0.06)
                && (tip[0] - before[0]).hypot(tip[1] - before[1]) > f64::EPSILON
        }) {
            Self::arrow(&mut stroke, sample(t - 0.08), sample(t), size);
        }
        if let Some(continuation) = self.continuation {
            let offset = 1.25_f64.min(length * 0.24);
            let t = match continuation {
                Continuation::Incoming => offset / length,
                Continuation::Outgoing => 1.0 - offset / length,
            };
            Self::arrow(&mut stroke, sample(t - 0.02), sample(t), size);
        }
    }

    fn arrow(
        stroke: &mut impl FnMut([f64; 2], [f64; 2]),
        before: [f64; 2],
        tip: [f64; 2],
        size: f64,
    ) {
        let [dx, dy] = [tip[0] - before[0], tip[1] - before[1]];
        let length = dx.hypot(dy);
        if length <= f64::EPSILON {
            return;
        }
        let [dx, dy] = [dx * size / length, dy * size / length];
        for side in [-1.0, 1.0] {
            stroke(
                [
                    tip[0] - dx - side * dy * 0.68,
                    tip[1] - dy + side * dx * 0.68,
                ],
                tip,
            );
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn orthogonal_routes_share_their_geometry_with_hit_testing() {
        let shape = Shape::Features(32);
        for endpoints in [
            [[2.0, 2.0], [18.0, 10.0]],
            [[18.0, 10.0], [2.0, 2.0]],
            [[4.0, 3.0], [4.0, 12.0]],
            [[2.0, 3.0], [18.0, 3.0]],
        ] {
            let mut route = Handoff::new(
                super::super::wire(0, 1, 0),
                [0, 1],
                endpoints,
                Some(&shape),
                Some(&shape),
                None,
            );
            route.orthogonal = true;
            let segments: Vec<_> = route.segments(route.length()).collect();
            assert_eq!(segments.first().unwrap().0, endpoints[0]);
            assert_eq!(segments.last().unwrap().1, endpoints[1]);
            for (a, b, _) in segments {
                assert!((a[0] - b[0]).abs() < 1e-9 || (a[1] - b[1]).abs() < 1e-9);
                if a != b {
                    let point = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
                    assert!(route.distance_at(point).unwrap() < 1e-8);
                }
            }
            let mut has_arrow = false;
            route.draw(None, |a, b| {
                assert!(a.into_iter().chain(b).all(f64::is_finite));
                has_arrow |= (a[0] - b[0]).abs() > 1e-9 && (a[1] - b[1]).abs() > 1e-9;
            });
            assert!(has_arrow);
        }
    }

    #[cfg(unix)]
    #[test]
    fn rotated_handoffs_draw_and_hit_in_both_directions() {
        let shape = Shape::Features(32);
        for to in [[1.0, 8.0], [8.0, 17.0]] {
            let from = [8.0, 8.0];
            let handoff = Handoff::new(
                super::super::wire(0, 1, 0),
                [0, 1],
                [from, to],
                Some(&shape),
                Some(&shape),
                None,
            );
            let mut strokes = Vec::new();
            handoff.draw(None, |a, b| strokes.push((a, b)));
            assert_eq!(strokes.first().unwrap().0, from);
            assert!(strokes.iter().any(|(_, b)| *b == to));
            assert!(handoff.distance_at(handoff.sample(0.5)).unwrap() < 0.01);
            assert!(handoff.distance_at([20.0, 20.0]).is_none());
        }
    }

    #[cfg(unix)]
    #[test]
    fn nearby_socket_hit_expands_the_target_without_inventing_offscreen_ports() {
        let shape = Shape::Features(32);
        let mut route = Handoff::new(
            super::super::wire(0, 1, 0),
            [0, 1],
            [[10.0, 10.0], [20.0, 10.0]],
            Some(&shape),
            Some(&shape),
            None,
        );
        let adjacent = [9.0, 11.0];
        assert!(route.socket_distance_at(adjacent, None).is_none());
        assert!(route.distance_at(adjacent).is_none());
        assert!(route.nearby_socket_distance_at(adjacent).is_some());
        route.continuation = Some(Continuation::Incoming);
        assert!(route.nearby_socket_distance_at(adjacent).is_none());
    }

    #[test]
    fn matching_tensor_handoff_draws_one_centerline() {
        let shape = Shape::Features(256);
        let handoff = Handoff::new(
            super::super::wire(0, 1, 0),
            [0, 1],
            [[0.0, 3.0], [20.0, 3.0]],
            Some(&shape),
            Some(&shape),
            None,
        );
        let mut interior = Vec::new();
        handoff.draw(None, |a, b| {
            if a[0] >= 2.0 && b[0] <= 8.0 {
                interior.push((a, b));
            }
        });
        assert!(!interior.is_empty());
        assert!(
            interior
                .iter()
                .all(|(from, to)| from[1] == 3.0 && to[1] == 3.0)
        );
        assert!(interior.windows(2).all(|pair| pair[0].1 == pair[1].0));
    }

    #[test]
    fn pinned_handoff_has_a_more_visible_direction_marker() {
        let wire = super::super::wire(0, 1, 0);
        let shape = Shape::Features(32);
        let handoff = Handoff::new(
            wire,
            [0, 1],
            [[2.0, 4.0], [22.0, 4.0]],
            Some(&shape),
            Some(&shape),
            None,
        );
        let marker_height = |selected| {
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            handoff.draw(selected, |from, to| {
                min_y = min_y.min(from[1]).min(to[1]);
                max_y = max_y.max(from[1]).max(to[1]);
            });
            max_y - min_y
        };
        let ordinary = marker_height(None);
        let pinned = marker_height(Some(wire));
        assert!(
            pinned > ordinary * 1.5,
            "pinned direction marker should be clearly larger: {ordinary} -> {pinned}"
        );
        assert_eq!(handoff.distance_at([12.0, 4.0]), Some(0.0));
    }

    #[test]
    fn pinned_direction_marker_covers_more_monochrome_braille_dots() {
        use super::super::{appearance::LinkInk, solid::Canvas};

        let wire = super::super::wire(0, 1, 0);
        let shape = Shape::Features(32);
        let handoff = Handoff::new(
            wire,
            [0, 1],
            [[2.0, 4.0], [22.0, 4.0]],
            Some(&shape),
            Some(&shape),
            None,
        );
        let dots = |selected| {
            let mut canvas = Canvas::new(30, 12);
            handoff.draw(selected, |from, to| {
                canvas.connector(from, to, (selected == Some(wire), LinkInk::Structure));
            });
            canvas.link_dot_count_for_test()
        };
        let ordinary = dots(None);
        let pinned = dots(Some(wire));
        assert!(
            pinned > ordinary,
            "selected Braille line should carry a larger direction marker: {ordinary} -> {pinned} dots"
        );
    }

    #[test]
    fn an_intersection_leaves_a_gap_on_only_the_lower_wire() {
        let wire = super::super::wire(0, 1, 0);
        let other = Wire {
            from: super::super::Outlet {
                slot: 1,
                ..wire.from
            },
            to: super::super::Inlet { slot: 1, ..wire.to },
        };
        let shape = Shape::Features(48);
        let route = |wire, endpoints| {
            Handoff::new(wire, [0, 1], endpoints, Some(&shape), Some(&shape), None)
        };
        for endpoints in [
            [[4.0, 16.0], [32.0, 4.0]],
            [[32.0, 4.0], [4.0, 16.0]],
            [[18.0, 4.0], [18.0, 16.0]],
        ] {
            let mut routes = [
                route(wire, [[4.0, 4.0], [32.0, 16.0]]),
                route(other, endpoints),
            ];
            Handoff::separate_crossings(&mut routes);
            assert!(routes.iter().all(|path| path.crossings.len() == 1));
            let drawn_near_intersection = |path: &Handoff, selected| {
                let mut painted = false;
                path.draw(selected, |start, end| {
                    let middle = [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5];
                    painted |= (middle[0] - 18.0).hypot(middle[1] - 10.0) < 0.30;
                });
                painted
            };
            assert!(drawn_near_intersection(&routes[0], None));
            assert!(!drawn_near_intersection(&routes[1], None));
            assert!(!drawn_near_intersection(&routes[0], Some(other)));
            assert!(drawn_near_intersection(&routes[1], Some(other)));
            #[cfg(unix)]
            assert!(
                routes
                    .iter()
                    .all(|path| path.distance_at([18.0, 10.0]).is_some_and(|d| d < 1e-8))
            );
        }

        let mut junction = [
            route(wire, [[4.0, 4.0], [32.0, 16.0]]),
            route(
                Wire {
                    from: wire.from,
                    ..other
                },
                [[4.0, 4.0], [32.0, 4.0]],
            ),
        ];
        Handoff::separate_crossings(&mut junction);
        assert!(junction.iter().all(|path| path.crossings.is_empty()));
    }
}
