//! Orthographic comparison of the measured last axes at two tensor ports.
use super::{LinkInk, Shape};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Comparison {
    extents: [Option<u64>; 2],
    open: [bool; 2],
    pub candidate: Option<usize>,
    zoom: u8,
}

pub(super) struct Drawing {
    pub comparison: Comparison,
    pub rows: [usize; 2],
    pub strokes: Vec<Stroke>,
}

pub(super) struct Stroke {
    pub segment: [[f64; 2]; 2],
    pub ink: LinkInk,
    pub guide: bool,
}

impl Comparison {
    pub fn from_shapes(shapes: [Option<&Shape>; 2], candidate: Option<usize>, zoom: u8) -> Self {
        Self {
            extents: shapes.map(|shape| shape.and_then(Shape::last)),
            open: shapes.map(|shape| !matches!(shape, Some(Shape::Dimensions(axes)) if axes.iter().all(Option::is_some))),
            candidate,
            zoom,
        }
    }

    pub fn missing(self, index: usize) -> bool {
        self.extents[index].is_none()
    }

    pub fn draw(self, columns: usize, art_rows: usize) -> Drawing {
        let height = (art_rows * 2) as f64;
        let y = [height * 0.28, height * 0.70];
        let left = if columns < 32 { 10.0 } else { 15.0 };
        let span = (columns as f64 - left - 4.0).min(90.0) * f64::from(self.zoom) / 10.0;
        let largest = self.extents.into_iter().flatten().max().unwrap_or(0);
        let end = self
            .extents
            .map(|value| value.map(|size| left + span * (size as f64 / largest.max(1) as f64)));

        let mut strokes = Vec::new();
        let mut draw = |points: &[[f64; 2]], ink| {
            strokes.extend(points.windows(2).map(|pair| Stroke {
                segment: [pair[0], pair[1]],
                ink,
                guide: ink == LinkInk::Structure,
            }));
        };
        let start = (y[0] - 1.8).max(0.5);
        let stop = (y[1] + 2.8).min(height - 0.5);
        let far_guide = match end {
            [Some(a), Some(b)] if (a - b).abs() > 0.6 => Some(a.max(b)),
            _ => None,
        };
        for x in [Some(left), far_guide].into_iter().flatten() {
            for step in 0..((stop - start) / 1.5).ceil() as usize {
                let at = start + step as f64 * 1.5;
                draw(&[[x, at], [x, (at + 0.37).min(stop)]], LinkInk::Structure);
            }
        }

        let thickness = (height * 0.24).min(2.1);
        for (index, y) in y.into_iter().enumerate() {
            let Some(right) = end[index] else {
                draw(
                    &[
                        [left + 1.4, y - thickness * 0.6],
                        [left, y],
                        [left + 1.4, y + thickness * 0.6],
                    ],
                    LinkInk::Unknown,
                );
                for offset in [2.1, 3.3, 4.5] {
                    draw(
                        &[[left + offset, y], [left + offset + 0.35, y]],
                        LinkInk::Unknown,
                    );
                }
                continue;
            };
            let ink = match (self.extents[index], self.extents[1 - index]) {
                (Some(a), Some(b)) if a == b => LinkInk::Match,
                (Some(a), Some(b)) if a > b => LinkInk::Mismatch,
                _ => LinkInk::Features,
            };
            let top = y - thickness * 0.43;
            let bottom = top + thickness;
            draw(
                &[[left, top], [left, bottom], [right, bottom], [right, top]],
                ink,
            );
            if self.open[index] {
                let corner = ((right - left) * 0.16).min(2.8);
                draw(&[[left, top], [left + corner, top]], ink);
                draw(&[[right - corner, top], [right, top]], ink);
            } else {
                draw(&[[left, top], [right, top]], ink);
            }
        }
        Drawing {
            comparison: self,
            rows: y.map(|y| (y / 2.0) as usize),
            strokes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shared_ruler_preserves_extent_and_leaves_unknown_edges_open() {
        let known = Shape::Dimensions(vec![Some(2), Some(48)]);
        let width_only = Shape::Features(32);
        let incomplete = Shape::Dimensions(vec![None, Some(48)]);
        let missing = Shape::Dimensions(vec![Some(2), None]);
        for columns in [40, 100] {
            let mut previous_width = 0.0;
            for zoom in [5, 8, 10] {
                let contours = |pair| {
                    let drawing = Comparison::from_shapes(pair, None, zoom).draw(columns, 7);
                    let horizontal = |row| {
                        drawing
                            .strokes
                            .iter()
                            .filter_map(|stroke| {
                                let [[x, y], [end, ey]] = stroke.segment;
                                (!stroke.guide && y == ey && usize::from(y > 7.0) == row)
                                    .then_some((x, end))
                            })
                            .collect::<Vec<_>>()
                    };
                    [horizontal(0), horizontal(1)]
                };
                let length = |contour: &[(f64, f64)]| contour[0].1 - contour[0].0;
                let [observed, expected] = contours([Some(&known), Some(&width_only)]);
                assert_eq!(observed[0].0, expected[0].0);
                assert_eq!(observed.len(), 2);
                assert_eq!(expected.len(), 3);
                assert!(
                    expected[1].1 < expected[2].0,
                    "unknown axes leave an open far edge"
                );
                assert!((length(&observed) / length(&expected) - 1.5).abs() < 1e-9);
                assert!(length(&expected) > previous_width);
                previous_width = length(&expected);
                let [observed, expected] = contours([Some(&width_only); 2]);
                assert_eq!(observed[0], expected[0]);
                let [observed, _] = contours([Some(&incomplete), Some(&width_only)]);
                assert!(observed[1].1 < observed[2].0);
                let [unknown, expected] = contours([Some(&missing), Some(&width_only)]);
                assert!(unknown.iter().all(|(from, to)| to - from < 0.5));
                assert!(length(&expected) > 8.0);
            }
        }
    }
}
