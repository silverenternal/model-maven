//! Translate rust-sugiyama's ranks and lanes into a paged terminal scene.
use rust_sugiyama::{
    configure::{Config, CrossingMinimization},
    from_vertices_and_edges,
};
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

#[derive(Clone, Copy)]
pub(super) struct Stage {
    pub layer: usize,
    pub rank: usize,
    pub lane: f64,
}

pub(super) struct FlowLayout {
    stages: Vec<Stage>,
    positions: HashMap<usize, usize>,
    adjacent: Vec<Vec<[usize; 2]>>,
    lanes: [f64; 2],
    rank_count: usize,
    ranks: Vec<Range<usize>>,
    max_parallel: usize,
}

impl FlowLayout {
    /// Place modules with no recorded relationships on one neutral rank.
    pub fn unconnected(layers: &[usize]) -> Self {
        Self::collect(
            layers
                .iter()
                .enumerate()
                .map(|(lane, &layer)| Stage {
                    layer,
                    rank: 0,
                    lane: lane as f64,
                })
                .collect(),
            &[],
        )
    }

    pub fn chain(layers: &[usize]) -> Self {
        let edges: Vec<_> = layers.windows(2).map(|pair| [pair[0], pair[1]]).collect();
        // Very long declared Sequential containers already define an exact
        // single-lane order. Keep their initial layout linear in model size.
        if layers.len() > 128 {
            return Self::collect(
                layers
                    .iter()
                    .enumerate()
                    .map(|(rank, &layer)| Stage {
                        layer,
                        rank,
                        lane: 0.0,
                    })
                    .collect(),
                &edges,
            );
        }
        Self::directed(layers, &edges)
    }

    pub fn directed(layers: &[usize], edges: &[[usize; 2]]) -> Self {
        if edges.is_empty() {
            return Self::collect(
                layers
                    .iter()
                    .enumerate()
                    .map(|(rank, &layer)| Stage {
                        layer,
                        rank,
                        lane: 0.0,
                    })
                    .collect(),
                edges,
            );
        }
        let local: HashMap<_, _> = layers
            .iter()
            .enumerate()
            .map(|(id, &layer)| (layer, id as u32))
            .collect();
        let vertices: Vec<_> = (0..layers.len() as u32)
            .map(|id| (id, (1.0, 1.0)))
            .collect();
        let arcs: Vec<_> = edges
            .iter()
            .map(|[from, to]| (local[from], local[to]))
            .collect();
        let config = Config {
            dummy_vertices: false,
            // Median ordering reduced crossings in Drift-L attention while keeping its wire set.
            c_minimization: CrossingMinimization::Median,
            ..Config::default()
        };
        let mut stages = Vec::with_capacity(layers.len());
        let mut rank_offset = 0;
        for (mut placed, _, _) in from_vertices_and_edges(&vertices, &arcs, &config) {
            placed.sort_by(|a, b| {
                a.1.1
                    .total_cmp(&b.1.1)
                    .then_with(|| a.1.0.total_cmp(&b.1.0))
                    .then_with(|| a.0.cmp(&b.0))
            });
            let mut ranks = placed.chunk_by(|a, b| (a.1.1 - b.1.1).abs() < 1e-6);
            for (rank, column) in ranks.by_ref().enumerate() {
                for &(id, (lane, _)) in column {
                    stages.push(Stage {
                        layer: layers[id],
                        rank: rank + rank_offset,
                        lane,
                    });
                }
            }
            rank_offset = stages.last().map_or(0, |stage| stage.rank + 1);
        }
        Self::collect(stages, edges)
    }

    fn collect(stages: Vec<Stage>, edges: &[[usize; 2]]) -> Self {
        let positions: HashMap<_, _> = stages
            .iter()
            .enumerate()
            .map(|(index, stage)| (stage.layer, index))
            .collect();
        let mut adjacent = vec![Vec::new(); stages.len()];
        for edge in edges {
            for layer in edge {
                adjacent[positions[layer]].push(*edge);
            }
        }
        let lanes = stages
            .iter()
            .fold([f64::INFINITY, f64::NEG_INFINITY], |[min, max], stage| {
                [min.min(stage.lane), max.max(stage.lane)]
            });
        let rank_count = stages.last().map_or(0, |stage| stage.rank + 1);
        let mut ranks = Vec::with_capacity(rank_count);
        let mut start = 0;
        while start < stages.len() {
            let end =
                start + stages[start..].partition_point(|stage| stage.rank == stages[start].rank);
            ranks.push(start..end);
            start = end;
        }
        let max_parallel = stages
            .chunk_by(|a, b| a.rank == b.rank)
            .map(<[_]>::len)
            .max()
            .unwrap_or(1);
        Self {
            stages,
            positions,
            adjacent,
            lanes,
            rank_count,
            ranks,
            max_parallel,
        }
    }

    pub fn count(&self) -> usize {
        self.stages.len()
    }

    pub fn layer(&self, index: usize) -> usize {
        self.stages[index].layer
    }

    pub fn contains(&self, layer: usize) -> bool {
        self.positions.contains_key(&layer)
    }

    pub fn next_layer(&self, current: usize, delta: isize) -> usize {
        let index = self
            .positions
            .get(&current)
            .map_or(0, |&index| index.saturating_add_signed(delta));
        self.layer(index.min(self.count() - 1))
    }

    pub fn branched(&self) -> bool {
        self.lanes[1] - self.lanes[0] > 1e-6
    }

    pub fn vertical_lane_span(&self) -> f64 {
        // Keep simple branch diagrams centered. Dense ranks use more of the
        // available width, with margins for projected shapes and sockets.
        (0.55 + self.max_parallel.saturating_sub(2) as f64 * 0.025).min(0.84)
    }

    pub fn lane(&self, stage: Stage) -> f64 {
        if self.branched() {
            (stage.lane - self.lanes[0]) / (self.lanes[1] - self.lanes[0]) - 0.5
        } else {
            0.0
        }
    }

    pub fn page(&self, selected: usize, columns: usize, lane_limit: usize) -> Range<usize> {
        let index = self.positions.get(&selected).copied().unwrap_or(0);
        let rank = self.stages[index].rank;
        let rank_range = &self.ranks[rank];
        if rank_range.len() > lane_limit {
            let start = rank_range.start + (index - rank_range.start) / lane_limit * lane_limit;
            return start..(start + lane_limit).min(rank_range.end);
        }
        let first_rank = rank / columns * columns;
        let end_rank = (first_rank + columns).min(self.rank_count);
        let mut start = self.ranks[first_rank].start;
        let mut end = self.ranks[end_rank - 1].end;
        for other in first_rank..end_rank {
            if self.ranks[other].len() > lane_limit {
                if other < rank {
                    start = self.ranks[other].end;
                } else if other > rank {
                    end = self.ranks[other].start;
                    break;
                }
            }
        }
        start..end
    }

    pub fn previous_page(&self, page_start: usize, columns: usize, lane_limit: usize) -> usize {
        if page_start == 0 {
            return 0;
        }
        self.page(self.layer(page_start - 1), columns, lane_limit)
            .start
    }

    pub fn rank_size(&self, rank: usize) -> usize {
        self.ranks[rank].len()
    }

    pub fn stages(&self, page: Range<usize>) -> &[Stage] {
        &self.stages[page]
    }

    pub fn rank_count(&self) -> usize {
        self.rank_count
    }

    pub fn edges(&self, page: Range<usize>) -> Vec<[usize; 2]> {
        let mut seen = HashSet::new();
        self.adjacent[page]
            .iter()
            .flatten()
            .copied()
            .filter(|edge| seen.insert(*edge))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_and_lane_pages_cover_every_stage_and_reverse_to_previous_page() {
        let mut stages = Vec::new();
        for (rank, count) in [1, 3, 40, 2, 1, 37, 1].into_iter().enumerate() {
            for lane in 0..count {
                stages.push(Stage {
                    layer: stages.len(),
                    rank,
                    lane: lane as f64,
                });
            }
        }
        let graph = FlowLayout::collect(stages, &[]);
        for columns in [1, 2, 4, 7] {
            for limit in [8, 16, 32, usize::MAX] {
                let mut start = 0;
                let mut previous = 0;
                while start < graph.count() {
                    let page = graph.page(graph.layer(start), columns, limit);
                    assert_eq!(page.start, start, "columns {columns}, limit {limit}");
                    assert!(page.end > page.start && page.end <= graph.count());
                    assert_eq!(graph.previous_page(start, columns, limit), previous);
                    for index in page.clone() {
                        assert_eq!(graph.page(graph.layer(index), columns, limit), page);
                    }
                    previous = start;
                    start = page.end;
                }
                assert_eq!(start, graph.count());
            }
        }
    }
}
