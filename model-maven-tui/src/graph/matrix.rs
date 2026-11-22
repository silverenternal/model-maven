use super::{
    Alignment, GraphScope, Handoff, Inlet, ModelView, OperatorId, Outlet, Port, Scene, Wire, push,
};
use crate::terminal::locale::Locale;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Matrix {
    scope: GraphScope,
    routes: Vec<Handoff>,
    cells: BTreeMap<(Outlet, Inlet), Vec<usize>>,
    grouped_cells: BTreeMap<(OutputGroup, InputGroup), Vec<usize>>,
    disconnected: BTreeSet<Wire>,
    grouped_disconnected: BTreeMap<(OutputGroup, InputGroup), usize>,
    rows: Vec<Outlet>,
    columns: Vec<Inlet>,
    row_starts: Vec<usize>,
    column_starts: Vec<usize>,
    row_group: Vec<usize>,
    column_group: Vec<usize>,
    rows_grouped: bool,
    columns_grouped: bool,
    row: usize,
    column: usize,
    choice: usize,
    axis: Option<AxisIndex>,
    mismatch_status: Option<MismatchStatus>,
}

enum MismatchStatus {
    Empty,
    Found { position: usize, total: usize },
}

struct AxisIndex {
    rows: bool,
    row: usize,
    column: usize,
    detail: Option<(usize, usize)>,
}

pub(super) struct MatrixState {
    rows_grouped: bool,
    columns_grouped: bool,
    row: Outlet,
    column: Inlet,
    selected: Option<(Wire, Wire)>,
    graph_selected: (Option<Wire>, Option<Port>, Option<Wire>),
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum OutputGroup {
    Port(Outlet),
    Module(OperatorId),
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum InputGroup {
    Port(Inlet),
    Module(OperatorId),
}

fn groups<T>(ports: &[T], operator: impl Fn(&T) -> OperatorId) -> (Vec<usize>, Vec<usize>) {
    let mut starts = Vec::new();
    let mut membership = Vec::new();
    let mut previous = None;
    for (index, port) in ports.iter().enumerate() {
        let id = operator(port);
        if previous != Some(id) {
            starts.push(index);
            previous = Some(id);
        }
        membership.push(starts.len() - 1);
    }
    (starts, membership)
}

fn axis_label(path: &str, suffix: &str, width: usize) -> String {
    let suffix_width = crate::terminal::text::display_width(suffix);
    if suffix_width >= width {
        return crate::terminal::text::truncate(suffix, width);
    }
    let path = crate::terminal::text::plain_text(path).replace('\n', " ");
    let available = width - suffix_width;
    if crate::terminal::text::display_width(&path) <= available {
        return format!("{path}{suffix}");
    }
    let mut start = path.len();
    for (index, _) in path.char_indices().rev() {
        if crate::terminal::text::display_width(&path[index..]) + 1 > available {
            break;
        }
        start = index;
    }
    format!("…{}{suffix}", &path[start..])
}

impl Matrix {
    pub fn new(view: &ModelView<'_>, scene: &Scene, selected: Option<Wire>) -> Option<Self> {
        let graph = view.flow_layout()?;
        let nodes: Vec<_> = (0..graph.count()).map(|index| graph.layer(index)).collect();
        let mut sockets = BTreeMap::new();
        for &index in &nodes {
            for (port, _) in view.document.ports_for(view.layers[index].id) {
                sockets.insert(port, [0.0, 0.0]);
            }
        }
        let mut routes = view.tensor_routes(
            graph,
            0..graph.count(),
            &nodes,
            &sockets,
            scene.width,
            scene.labels.len(),
            false,
        );
        let mut seen = BTreeSet::new();
        routes.retain(|route| seen.insert((route.wire, route.displayed)));
        let mut cells: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut grouped_cells: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for (index, route) in routes.iter().enumerate() {
            cells
                .entry((route.displayed.from, route.displayed.to))
                .or_default()
                .push(index);
            let from = route.displayed.from;
            let to = route.displayed.to;
            for key in [
                (OutputGroup::Module(from.operator), InputGroup::Port(to)),
                (OutputGroup::Port(from), InputGroup::Module(to.operator)),
                (
                    OutputGroup::Module(from.operator),
                    InputGroup::Module(to.operator),
                ),
            ] {
                grouped_cells.entry(key).or_default().push(index);
            }
        }
        let mut outputs: BTreeSet<_> = routes.iter().map(|route| route.displayed.from).collect();
        let mut inputs: BTreeSet<_> = routes.iter().map(|route| route.displayed.to).collect();
        for &index in &nodes {
            for (port, _) in view.document.ports_for(view.layers[index].id) {
                match port {
                    Port::Output(port) => {
                        outputs.insert(port);
                    }
                    Port::Input(port) => {
                        inputs.insert(port);
                    }
                }
            }
        }
        let rows: Vec<_> = outputs.into_iter().collect();
        let columns: Vec<_> = inputs.into_iter().collect();
        if rows.is_empty() || columns.is_empty() {
            return None;
        }
        // Only exact visible port identities carry negative evidence. Never
        // project an absence through a collapsed boundary or override a route.
        let disconnected: BTreeSet<_> = view
            .disconnected
            .iter()
            .copied()
            .filter(|wire| {
                rows.binary_search(&wire.from).is_ok()
                    && columns.binary_search(&wire.to).is_ok()
                    && !cells.contains_key(&(wire.from, wire.to))
            })
            .collect();
        let mut grouped_disconnected = BTreeMap::new();
        for wire in &disconnected {
            for key in [
                (
                    OutputGroup::Module(wire.from.operator),
                    InputGroup::Port(wire.to),
                ),
                (
                    OutputGroup::Port(wire.from),
                    InputGroup::Module(wire.to.operator),
                ),
                (
                    OutputGroup::Module(wire.from.operator),
                    InputGroup::Module(wire.to.operator),
                ),
            ] {
                *grouped_disconnected.entry(key).or_insert(0) += 1;
            }
        }
        let (row_starts, row_group) = groups(&rows, |port| port.operator);
        let (column_starts, column_group) = groups(&columns, |port| port.operator);
        let selected_display = selected.and_then(|wire| {
            scene
                .handoffs
                .iter()
                .find(|route| route.wire == wire)
                .map(|route| route.displayed)
        });
        let route = selected.and_then(|wire| {
            routes.iter().find(|route| {
                route.wire == wire
                    && selected_display.is_none_or(|display| route.displayed == display)
            })
        });
        let row = route
            .and_then(|route| rows.iter().position(|port| *port == route.displayed.from))
            .or_else(|| {
                view.selected_pair
                    .and_then(|pair| rows.binary_search(&pair.from).ok())
            })
            .or_else(|| match view.selected_port {
                Some(Port::Output(port)) => rows.binary_search(&port).ok(),
                _ => None,
            })
            .unwrap_or(0);
        let column = route
            .and_then(|route| columns.iter().position(|port| *port == route.displayed.to))
            .or_else(|| {
                view.selected_pair
                    .and_then(|pair| columns.binary_search(&pair.to).ok())
            })
            .or_else(|| match view.selected_port {
                Some(Port::Input(port)) => columns.binary_search(&port).ok(),
                _ => None,
            })
            .unwrap_or(0);
        let choice = selected
            .and_then(|wire| {
                routes
                    .iter()
                    .filter(|route| {
                        route.displayed.from == rows[row] && route.displayed.to == columns[column]
                    })
                    .position(|route| route.wire == wire)
            })
            .unwrap_or(0);
        Some(Self {
            scope: view.graph_scope(),
            routes,
            cells,
            grouped_cells,
            disconnected,
            grouped_disconnected,
            rows,
            columns,
            row_starts,
            column_starts,
            row_group,
            column_group,
            rows_grouped: false,
            columns_grouped: false,
            row,
            column,
            choice,
            axis: None,
            mismatch_status: None,
        })
    }

    /// Restore presentation state by port identity after rebuilding from the
    /// current graph. A selection made in the graph takes precedence.
    pub fn snapshot(
        &self,
        graph_selected: (Option<Wire>, Option<Port>, Option<Wire>),
    ) -> (GraphScope, MatrixState) {
        (
            self.scope,
            MatrixState {
                rows_grouped: self.rows_grouped,
                columns_grouped: self.columns_grouped,
                row: self.rows[self.row],
                column: self.columns[self.column],
                selected: self.selected_wire(),
                graph_selected,
            },
        )
    }

    pub fn restore(
        &mut self,
        previous: &MatrixState,
        graph_selected: (Option<Wire>, Option<Port>, Option<Wire>),
    ) {
        let graph_selection_changed = previous.graph_selected != graph_selected;
        let selected = if graph_selection_changed {
            self.selected_wire()
        } else {
            previous.selected
        };
        self.rows_grouped = previous.rows_grouped;
        self.columns_grouped = previous.columns_grouped;
        if !graph_selection_changed {
            if let (Ok(row), Ok(column)) = (
                self.rows.binary_search(&previous.row),
                self.columns.binary_search(&previous.column),
            ) {
                self.row = row;
                self.column = column;
            }
        }
        self.choice = selected
            .and_then(|(wire, displayed)| {
                self.indices(self.logical_row(), self.logical_column())
                    .iter()
                    .position(|&index| {
                        self.routes[index].wire == wire && self.routes[index].displayed == displayed
                    })
            })
            .unwrap_or(0);
    }

    fn logical_row(&self) -> usize {
        if self.rows_grouped {
            self.row_group[self.row]
        } else {
            self.row
        }
    }
    fn logical_column(&self) -> usize {
        if self.columns_grouped {
            self.column_group[self.column]
        } else {
            self.column
        }
    }
    fn row_count(&self) -> usize {
        if self.rows_grouped {
            self.row_starts.len()
        } else {
            self.rows.len()
        }
    }
    fn column_count(&self) -> usize {
        if self.columns_grouped {
            self.column_starts.len()
        } else {
            self.columns.len()
        }
    }
    fn row_index(&self, logical: usize) -> usize {
        if self.rows_grouped {
            self.row_starts[logical]
        } else {
            logical
        }
    }
    fn column_index(&self, logical: usize) -> usize {
        if self.columns_grouped {
            self.column_starts[logical]
        } else {
            logical
        }
    }

    fn indices(&self, row: usize, column: usize) -> &[usize] {
        let output = self.rows[self.row_index(row)];
        let input = self.columns[self.column_index(column)];
        if !self.rows_grouped && !self.columns_grouped {
            return self.cells.get(&(output, input)).map_or(&[], Vec::as_slice);
        }
        let output = if self.rows_grouped {
            OutputGroup::Module(output.operator)
        } else {
            OutputGroup::Port(output)
        };
        let input = if self.columns_grouped {
            InputGroup::Module(input.operator)
        } else {
            InputGroup::Port(input)
        };
        self.grouped_cells
            .get(&(output, input))
            .map_or(&[], Vec::as_slice)
    }

    fn disconnected_count(&self, row: usize, column: usize) -> usize {
        let from = self.rows[self.row_index(row)];
        let to = self.columns[self.column_index(column)];
        if !self.rows_grouped && !self.columns_grouped {
            return usize::from(self.disconnected.contains(&Wire { from, to }));
        }
        let output = if self.rows_grouped {
            OutputGroup::Module(from.operator)
        } else {
            OutputGroup::Port(from)
        };
        let input = if self.columns_grouped {
            InputGroup::Module(to.operator)
        } else {
            InputGroup::Port(to)
        };
        self.grouped_disconnected
            .get(&(output, input))
            .copied()
            .unwrap_or(0)
    }

    fn pair_count(&self, row: usize, column: usize) -> usize {
        let row_start = self.row_index(row);
        let row_end = if self.rows_grouped {
            self.row_starts
                .get(row + 1)
                .copied()
                .unwrap_or(self.rows.len())
        } else {
            row_start + 1
        };
        let column_start = self.column_index(column);
        let column_end = if self.columns_grouped {
            self.column_starts
                .get(column + 1)
                .copied()
                .unwrap_or(self.columns.len())
        } else {
            column_start + 1
        };
        (row_end - row_start).saturating_mul(column_end - column_start)
    }

    pub fn toggle_groups(&mut self, rows: bool, columns: bool) {
        self.mismatch_status = None;
        let selected = self.selected_wire();
        if rows {
            self.rows_grouped = !self.rows_grouped;
        }
        if columns {
            self.columns_grouped = !self.columns_grouped;
        }
        if let Some((wire, displayed)) = selected {
            if let Some(index) = self.rows.iter().position(|port| *port == displayed.from) {
                self.row = index;
            }
            if let Some(index) = self.columns.iter().position(|port| *port == displayed.to) {
                self.column = index;
            }
            self.choice = self
                .indices(self.logical_row(), self.logical_column())
                .iter()
                .position(|&index| {
                    self.routes[index].wire == wire && self.routes[index].displayed == displayed
                })
                .unwrap_or(0);
        } else {
            self.choice = 0;
        }
    }

    fn windows(&self, width: usize, height: usize) -> (usize, usize, usize, usize) {
        let row_count = height
            .saturating_sub(
                (if self.rows_grouped || self.columns_grouped {
                    11
                } else {
                    9
                }) + usize::from(width < 58),
            )
            .max(1);
        let column_count = width.saturating_sub(6).checked_div(4).unwrap_or(1).max(1);
        let row_start = self.logical_row().saturating_sub(row_count - 1);
        let column_start = self.logical_column().saturating_sub(column_count - 1);
        (row_start, row_count, column_start, column_count)
    }

    pub fn movement(&mut self, row: isize, column: isize) {
        if let Some(axis) = &mut self.axis {
            if axis.rows {
                axis.row = axis.row.saturating_add_signed(row).min(
                    if self.rows_grouped {
                        self.row_starts.len()
                    } else {
                        self.rows.len()
                    } - 1,
                );
            } else {
                axis.column = axis.column.saturating_add_signed(row).min(
                    if self.columns_grouped {
                        self.column_starts.len()
                    } else {
                        self.columns.len()
                    } - 1,
                );
            }
            return;
        }
        let previous = (self.logical_row(), self.logical_column());
        let next_row = previous
            .0
            .saturating_add_signed(row)
            .min(self.row_count() - 1);
        let next_column = previous
            .1
            .saturating_add_signed(column)
            .min(self.column_count() - 1);
        if previous != (next_row, next_column) {
            self.mismatch_status = None;
            self.row = self.row_index(next_row);
            self.column = self.column_index(next_column);
            self.choice = 0;
        }
    }

    pub fn page(&mut self, width: usize, height: usize, forward: bool) {
        let count = if self.axis.is_some() {
            height.saturating_sub(4).max(1)
        } else {
            self.windows(width, height).1
        };
        let count = isize::try_from(count).unwrap_or(isize::MAX);
        self.movement(if forward { count } else { -count }, 0);
    }

    pub fn cycle(&mut self) {
        let count = self
            .indices(self.logical_row(), self.logical_column())
            .len();
        if count > 0 {
            self.mismatch_status = None;
            self.choice = (self.choice + 1) % count;
        }
    }

    /// Visit recorded mismatches by logical cell in stable row/column order.
    /// A mixed grouped cell selects an underlying mismatching route.
    pub fn jump_mismatch(&mut self, forward: bool) {
        let mut cells: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (index, route) in self.routes.iter().enumerate() {
            if route.alignment != Alignment::Mismatch {
                continue;
            }
            let Ok(row) = self.rows.binary_search(&route.displayed.from) else {
                continue;
            };
            let Ok(column) = self.columns.binary_search(&route.displayed.to) else {
                continue;
            };
            let row = if self.rows_grouped {
                self.row_group[row]
            } else {
                row
            };
            let column = if self.columns_grouped {
                self.column_group[column]
            } else {
                column
            };
            cells.entry((row, column)).or_insert(index);
        }
        if cells.is_empty() {
            self.mismatch_status = Some(MismatchStatus::Empty);
            return;
        }
        let current = (self.logical_row(), self.logical_column());
        let entry = if forward {
            cells
                .range((
                    std::ops::Bound::Excluded(current),
                    std::ops::Bound::Unbounded,
                ))
                .next()
                .or_else(|| cells.first_key_value())
        } else {
            cells
                .range((
                    std::ops::Bound::Unbounded,
                    std::ops::Bound::Excluded(current),
                ))
                .next_back()
                .or_else(|| cells.last_key_value())
        };
        let (&(row, column), &route_index) = entry.unwrap();
        if row != current.0 {
            self.row = self.row_index(row);
        }
        if column != current.1 {
            self.column = self.column_index(column);
        }
        self.choice = self
            .indices(row, column)
            .iter()
            .position(|&candidate| candidate == route_index)
            .unwrap_or(0);
        self.mismatch_status = Some(MismatchStatus::Found {
            position: cells.keys().position(|&key| key == (row, column)).unwrap() + 1,
            total: cells.len(),
        });
    }

    pub fn cell_at(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Option<(usize, usize)> {
        if x >= width || y >= height {
            return None;
        }
        if let Some(axis) = &self.axis {
            if axis.detail.is_some() {
                return None;
            }
            let count = height.saturating_sub(4).max(1);
            let focus = if axis.rows { axis.row } else { axis.column };
            let total = if axis.rows {
                self.row_count()
            } else {
                self.column_count()
            };
            let index = focus
                .saturating_sub(count - 1)
                .saturating_add(y.checked_sub(2)?);
            return (y < 2 + count && index < total).then_some((index, usize::MAX));
        }
        let (row_start, row_count, column_start, column_count) = self.windows(width, height);
        let row = y.checked_sub(3)?;
        let column = x.checked_sub(6)? / 4;
        let row = row_start + row;
        let column = column_start + column;
        if row >= self.row_count()
            || column >= self.column_count()
            || row >= row_start + row_count
            || column >= column_start + column_count
        {
            return None;
        }
        Some((row, column))
    }

    pub fn click(&mut self, x: usize, y: usize, width: usize, height: usize) -> bool {
        let Some((row, column)) = self.cell_at(x, y, width, height) else {
            return false;
        };
        if column == usize::MAX {
            if let Some(axis) = &mut self.axis {
                if axis.rows {
                    axis.row = row;
                } else {
                    axis.column = row;
                }
            }
            self.confirm_axis();
            return true;
        }
        self.mismatch_status = None;
        if (self.logical_row(), self.logical_column()) == (row, column) {
            self.cycle();
        } else {
            self.row = self.row_index(row);
            self.column = self.column_index(column);
            self.choice = 0;
        }
        true
    }

    pub fn selected_wire(&self) -> Option<(Wire, Wire)> {
        let index = *self
            .indices(self.logical_row(), self.logical_column())
            .get(self.choice)?;
        let route = &self.routes[index];
        Some((route.wire, route.displayed))
    }

    /// Recorded routes in this loaded scope, including off-page projections.
    pub(super) fn scope_wires(&self) -> Vec<(Wire, Wire)> {
        self.routes
            .iter()
            .map(|route| (route.wire, route.displayed))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn selected_pair(&self) -> Option<Wire> {
        (!self.rows_grouped && !self.columns_grouped).then_some(Wire {
            from: self.rows[self.row],
            to: self.columns[self.column],
        })
    }

    pub fn axis_open(&self) -> bool {
        self.axis.is_some()
    }

    pub fn toggle_axis(&mut self) {
        if self.axis.is_some() {
            self.axis = None;
        } else {
            self.axis = Some(AxisIndex {
                rows: true,
                row: self.logical_row(),
                column: self.logical_column(),
                detail: None,
            });
        }
    }

    pub fn switch_axis(&mut self) {
        if let Some(axis) = &mut self.axis {
            axis.rows = !axis.rows;
        }
    }

    pub fn axis_details_open(&self) -> bool {
        self.axis.as_ref().is_some_and(|axis| axis.detail.is_some())
    }

    pub fn toggle_axis_details(&mut self) {
        if let Some(axis) = &mut self.axis {
            axis.detail = if axis.detail.is_some() {
                None
            } else {
                Some((0, 0))
            };
        }
    }

    fn axis_details_lines(
        &self,
        view: &ModelView<'_>,
        width: usize,
        locale: Locale,
    ) -> Vec<(String, usize, usize)> {
        let Some(axis) = &self.axis else {
            return Vec::new();
        };
        let (operator, direction, slots, grouped) = if axis.rows {
            let start = self.row_index(axis.row);
            let end = if self.rows_grouped {
                self.row_starts
                    .get(axis.row + 1)
                    .copied()
                    .unwrap_or(self.rows.len())
            } else {
                start + 1
            };
            (
                self.rows[start].operator,
                "out",
                self.rows[start..end]
                    .iter()
                    .map(|port| port.slot)
                    .collect::<Vec<_>>(),
                self.rows_grouped,
            )
        } else {
            let start = self.column_index(axis.column);
            let end = if self.columns_grouped {
                self.column_starts
                    .get(axis.column + 1)
                    .copied()
                    .unwrap_or(self.columns.len())
            } else {
                start + 1
            };
            (
                self.columns[start].operator,
                "in",
                self.columns[start..end]
                    .iter()
                    .map(|port| port.slot)
                    .collect::<Vec<_>>(),
                self.columns_grouped,
            )
        };
        let path = view
            .document
            .operator(operator)
            .map_or("?", |item| item.path.as_str());
        let mut paragraphs = vec![
            locale
                .text("Module path", "模块路径", "模組路徑")
                .to_owned(),
            crate::terminal::text::plain_text(path).replace('\n', " "),
            if axis.rows {
                locale.text("Output ports", "输出端口", "輸出連接埠")
            } else {
                locale.text("Input ports", "输入端口", "輸入連接埠")
            }
            .to_owned(),
        ];
        if grouped {
            paragraphs.push(format!(
                "{} {}",
                locale.text("Group members", "分组端口数", "分組連接埠數"),
                slots.len()
            ));
        }
        paragraphs.extend(slots.iter().map(|slot| format!("{direction}[{slot}]")));
        let mut result = Vec::new();
        for (paragraph_index, paragraph) in paragraphs.iter().enumerate() {
            let mut start = 0;
            let mut columns = 0;
            for (index, ch) in paragraph.char_indices() {
                let char_width = crate::terminal::text::display_width(&ch.to_string());
                if columns + char_width > width.max(1) && index > start {
                    result.push((paragraph[start..index].to_owned(), paragraph_index, start));
                    start = index;
                    columns = 0;
                }
                columns += char_width;
            }
            if start < paragraph.len() {
                result.push((paragraph[start..].to_owned(), paragraph_index, start));
            }
        }
        result
    }

    fn axis_detail_offset(&self, lines: &[(String, usize, usize)], height: usize) -> usize {
        let anchor = self
            .axis
            .as_ref()
            .and_then(|axis| axis.detail)
            .unwrap_or((0, 0));
        let position = lines
            .iter()
            .enumerate()
            .filter(|(_, (_, paragraph, byte))| (*paragraph, *byte) <= anchor)
            .map(|(index, _)| index)
            .next_back()
            .unwrap_or(0);
        position.min(lines.len().saturating_sub(height.saturating_sub(3).max(1)))
    }

    pub fn scroll_axis_details(
        &mut self,
        view: &ModelView<'_>,
        width: usize,
        height: usize,
        locale: Locale,
        forward: bool,
        page: bool,
    ) {
        let lines = self.axis_details_lines(view, width, locale);
        let offset = self.axis_detail_offset(&lines, height);
        let available = height.saturating_sub(3).max(1);
        let step = if page { available } else { 1 };
        let next = if forward {
            offset
                .saturating_add(step)
                .min(lines.len().saturating_sub(available))
        } else {
            offset.saturating_sub(step)
        };
        if let Some((_, paragraph, byte)) = lines.get(next) {
            if let Some(axis) = &mut self.axis {
                axis.detail = Some((*paragraph, *byte));
            }
        }
    }

    pub fn confirm_axis(&mut self) {
        if let Some(axis) = self.axis.take() {
            let previous = (self.logical_row(), self.logical_column());
            if previous.0 != axis.row {
                self.row = self.row_index(axis.row);
            }
            if previous.1 != axis.column {
                self.column = self.column_index(axis.column);
            }
            if previous != (axis.row, axis.column) {
                self.choice = 0;
                self.mismatch_status = None;
            }
        }
    }

    fn render_axis(
        &self,
        view: &ModelView<'_>,
        width: usize,
        height: usize,
        locale: Locale,
        axis: &AxisIndex,
    ) -> String {
        let mut output = String::new();
        if axis.detail.is_some() {
            push(
                &mut output,
                locale.text("Axis port details", "轴端口详情", "軸連接埠詳情"),
                width,
            );
            let lines = self.axis_details_lines(view, width, locale);
            let available = height.saturating_sub(3).max(1);
            let offset = self.axis_detail_offset(&lines, height);
            for (line, _, _) in lines.iter().skip(offset).take(available) {
                push(&mut output, line, width);
            }
            push(
                &mut output,
                &format!(
                    "{} {}/{}",
                    locale.text("Lines", "行", "行"),
                    offset + 1,
                    lines.len()
                ),
                width,
            );
            push(
                &mut output,
                locale.text(
                    "↑↓/Pg scroll · v/Esc back",
                    "↑↓/Pg 滚动 · v/Esc 返回",
                    "↑↓/Pg 捲動 · v/Esc 返回",
                ),
                width,
            );
            return output;
        }
        let (focus, total) = if axis.rows {
            (axis.row, self.row_count())
        } else {
            (axis.column, self.column_count())
        };
        push(
            &mut output,
            if axis.rows {
                locale.text(
                    "Axis index · output rows",
                    "轴索引 · 输出行",
                    "軸索引 · 輸出列",
                )
            } else {
                locale.text(
                    "Axis index · input columns",
                    "轴索引 · 输入列",
                    "軸索引 · 輸入欄",
                )
            },
            width,
        );
        push(
            &mut output,
            &format!(
                "{} {}/{}",
                locale.text("Selected", "已选", "已選"),
                focus + 1,
                total
            ),
            width,
        );
        let count = height.saturating_sub(4).max(1);
        let start = focus.saturating_sub(count - 1);
        for index in start..(start + count).min(total) {
            let (operator, slot, grouped, group_count, direction) = if axis.rows {
                let first = self.row_index(index);
                let end = if self.rows_grouped {
                    self.row_starts
                        .get(index + 1)
                        .copied()
                        .unwrap_or(self.rows.len())
                } else {
                    first + 1
                };
                (
                    self.rows[first].operator,
                    self.rows[first].slot,
                    self.rows_grouped,
                    end - first,
                    "out",
                )
            } else {
                let first = self.column_index(index);
                let end = if self.columns_grouped {
                    self.column_starts
                        .get(index + 1)
                        .copied()
                        .unwrap_or(self.columns.len())
                } else {
                    first + 1
                };
                (
                    self.columns[first].operator,
                    self.columns[first].slot,
                    self.columns_grouped,
                    end - first,
                    "in",
                )
            };
            let path = view
                .document
                .operator(operator)
                .map_or("?", |operator| operator.path.as_str());
            let prefix = format!(
                "{} {:>4} ",
                if index == focus { '>' } else { ' ' },
                index + 1
            );
            let suffix = if grouped {
                format!(".{direction}[*] ({group_count})")
            } else {
                format!(".{direction}[{slot}]")
            };
            let budget = width.saturating_sub(crate::terminal::text::display_width(&prefix));
            let label = axis_label(path, &suffix, budget);
            push(&mut output, &format!("{prefix}{label}"), width);
        }
        push(
            &mut output,
            locale.text(
                "Tab axis · ↑/↓ choose · v details",
                "Tab 切轴 · ↑/↓ 选择 · v 详情",
                "Tab 切軸 · ↑/↓ 選擇 · v 詳情",
            ),
            width,
        );
        push(
            &mut output,
            locale.text(
                "Enter/click · locate   Esc/a · cancel",
                "Enter/单击定位 · Esc/a 取消",
                "Enter/單擊定位 · Esc/a 取消",
            ),
            width,
        );
        output
    }

    pub fn render(
        &self,
        view: &ModelView<'_>,
        width: usize,
        height: usize,
        locale: Locale,
    ) -> String {
        if let Some(axis) = &self.axis {
            return self.render_axis(view, width, height, locale, axis);
        }
        let mut output = String::new();
        let (row_start, row_count, column_start, column_count) = self.windows(width, height);
        push(
            &mut output,
            locale.text(
                "Port matrix · loaded scope",
                "端口矩阵 · 已加载范围",
                "連接埠矩陣 · 已載入範圍",
            ),
            width,
        );
        let subtitle = match &self.mismatch_status {
            Some(MismatchStatus::Empty) => locale
                .text(
                    "No recorded mismatch in scope",
                    "当前范围无已记录错配",
                    "目前範圍無已記錄不符",
                )
                .to_owned(),
            Some(MismatchStatus::Found { position, total }) => format!(
                "{} {position}/{total} · n/p",
                locale.text("Mismatch cell", "错配单元格", "不符儲存格")
            ),
            None => locale
                .text(
                    "Rows output · columns input · a index",
                    "行 输出 · 列 输入 · a 索引",
                    "列 輸出 · 欄 輸入 · a 索引",
                )
                .to_owned(),
        };
        push(&mut output, &subtitle, width);
        let mut header = String::from("      ");
        for column in column_start..(column_start + column_count).min(self.column_count()) {
            if column == self.logical_column() && column < 999 {
                header.push_str(&format!("{:>3}v", column + 1));
            } else {
                header.push_str(&format!("{:>4}", column + 1));
            }
        }
        push(&mut output, &header, width);
        for row in row_start..(row_start + row_count).min(self.row_count()) {
            let mut line = format!(
                "{}{:>4} ",
                if row == self.logical_row() { '>' } else { ' ' },
                row + 1
            );
            for column in column_start..(column_start + column_count).min(self.column_count()) {
                let mut routes = self
                    .indices(row, column)
                    .iter()
                    .map(|&index| &self.routes[index]);
                let first = routes.next();
                let mixed = first
                    .is_some_and(|route| routes.any(|other| other.alignment != route.alignment));
                let mark = if mixed {
                    '!'
                } else {
                    match first.map(|route| route.alignment) {
                        Some(Alignment::Mismatch) => '×',
                        Some(Alignment::Match | Alignment::Features) => '=',
                        Some(Alignment::Unknown) => 'o',
                        None if self.disconnected_count(row, column)
                            == self.pair_count(row, column) =>
                        {
                            '-'
                        }
                        None => '?',
                    }
                };
                if (row, column) == (self.logical_row(), self.logical_column()) {
                    line.push_str(&format!(" [{mark}]"));
                } else if column == self.logical_column() {
                    line.push_str(&format!(" |{mark} "));
                } else {
                    line.push_str(&format!("  {mark} "));
                }
            }
            push(&mut output, &line, width);
        }
        let path = |id| {
            view.document
                .operator(id)
                .map_or("?", |operator| operator.path.as_str())
        };
        let from = self.rows[self.row];
        let to = self.columns[self.column];
        let grouped = self.rows_grouped || self.columns_grouped;
        let row_end = if self.rows_grouped {
            self.row_starts
                .get(self.logical_row() + 1)
                .copied()
                .unwrap_or(self.rows.len())
        } else {
            self.row + 1
        };
        let column_end = if self.columns_grouped {
            self.column_starts
                .get(self.logical_column() + 1)
                .copied()
                .unwrap_or(self.columns.len())
        } else {
            self.column + 1
        };
        let row_start = self.row_index(self.logical_row());
        let column_start = self.column_index(self.logical_column());
        let row_size = row_end - row_start;
        let column_size = column_end - column_start;
        let label = |name: &str, slot: u32, grouped: bool, count: usize| {
            if grouped {
                format!("{name}[*] ({count})")
            } else {
                format!("{name}[{slot}]")
            }
        };
        push(
            &mut output,
            &format!(
                "{} → {}",
                label(path(from.operator), from.slot, self.rows_grouped, row_size),
                label(
                    path(to.operator),
                    to.slot,
                    self.columns_grouped,
                    column_size
                )
            ),
            width,
        );
        let indices = self.indices(self.logical_row(), self.logical_column());
        let count = indices.len();
        if grouped {
            let mismatch = indices
                .iter()
                .filter(|&&index| self.routes[index].alignment == Alignment::Mismatch)
                .count();
            let unknown = indices
                .iter()
                .filter(|&&index| self.routes[index].alignment == Alignment::Unknown)
                .count();
            let pairs: BTreeSet<_> = indices
                .iter()
                .map(|&index| {
                    (
                        self.routes[index].displayed.from,
                        self.routes[index].displayed.to,
                    )
                })
                .collect();
            let consumers: BTreeSet<_> = pairs.iter().map(|(_, input)| *input).collect();
            push(
                &mut output,
                &format!(
                    "{} {count} · {} {mismatch} · ? {unknown} · {} {}",
                    locale.text("Recorded", "已记录", "已記錄"),
                    locale.text("mismatch", "错配", "不符"),
                    locale.text("consumers", "消费者", "消費者"),
                    consumers.len()
                ),
                width,
            );
            let total = row_size.saturating_mul(column_size);
            let absent = self.disconnected_count(self.logical_row(), self.logical_column());
            let unknown = total.saturating_sub(pairs.len()).saturating_sub(absent);
            let coverage = if absent > 0 {
                format!(
                    "{} {absent} · ? {unknown}",
                    locale.text("no wire", "无连线", "無連線")
                )
            } else if pairs.len() == total {
                locale
                    .text("all recorded", "全部已记录", "全部已記錄")
                    .to_owned()
            } else {
                locale
                    .text("remaining unknown", "其余未知", "其餘未知")
                    .to_owned()
            };
            push(
                &mut output,
                &format!(
                    "{} {}/{} · {coverage}",
                    locale.text("Pairs", "端口对", "連接埠對"),
                    pairs.len(),
                    total
                ),
                width,
            );
        }
        if count > 1 || (grouped && count == 1) {
            if let Some((wire, _)) = self.selected_wire() {
                push(
                    &mut output,
                    &format!(
                        "{}[{}] → {}[{}]",
                        path(wire.from.operator),
                        wire.from.slot,
                        path(wire.to.operator),
                        wire.to.slot
                    ),
                    width,
                );
            }
            push(
                &mut output,
                &format!(
                    "{} {}/{} · w",
                    locale.text("Underlying wire", "底层连线", "底層連線"),
                    self.choice + 1,
                    count
                ),
                width,
            );
        } else if count == 0 {
            push(
                &mut output,
                if self.disconnected_count(self.logical_row(), self.logical_column())
                    == self.pair_count(self.logical_row(), self.logical_column())
                {
                    locale.text("Declared no wire", "声明无连线", "聲明無連線")
                } else {
                    locale.text("No recorded wire", "无已记录连线", "無已記錄連線")
                },
                width,
            );
            push(
                &mut output,
                if grouped {
                    locale.text(
                        "Expand groups to select a pair",
                        "展开分组以选择端口对",
                        "展開分組以選擇連接埠對",
                    )
                } else {
                    locale.text(
                        "Enter · select port pair",
                        "Enter · 选择端口对",
                        "Enter · 選擇連接埠對",
                    )
                },
                width,
            );
        }
        push(
            &mut output,
            locale.text(
                "= match  × mismatch  o shape unknown",
                "= 对齐  × 错配  o 形状未知",
                "= 對齊  × 不符  o 形狀未知",
            ),
            width,
        );
        push(
            &mut output,
            locale.text(
                "! mixed  - no wire  ? unknown",
                "! 混合  - 无连线  ? 未知",
                "! 混合  - 無連線  ? 未知",
            ),
            width,
        );
        if width < 58 {
            push(
                &mut output,
                locale.text(
                    "↑↓/wheel · Pg · n/p mismatch",
                    "方向键/滚轮 · Pg · n/p 错配",
                    "方向鍵/滾輪 · Pg · n/p 不符",
                ),
                width,
            );
            push(
                &mut output,
                locale.text(
                    "Enter/click · r/c · a · Esc",
                    "Enter/单击 · r/c · a · Esc",
                    "Enter/單擊 · r/c · a · Esc",
                ),
                width,
            );
        } else {
            push(
                &mut output,
                locale.text(
                    "Arrows/wheel · Pg · n/p mismatch · Enter/click · r/c group · a · Esc",
                    "方向键/滚轮 · Pg · n/p 错配 · Enter/单击 · r/c 分组 · a · Esc",
                    "方向鍵/滾輪 · Pg · n/p 不符 · Enter/單擊 · r/c 分組 · a · Esc",
                ),
                width,
            );
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatch_navigation_wraps_and_preserves_the_underlying_wire_in_groups() {
        let offline =
            super::super::offline::Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts);
        let view = offline.view();
        let scene = view.scene(40, 24, false, None);
        let mut matrix = Matrix::new(&view, &scene, None).unwrap();
        for route in &mut matrix.routes {
            route.alignment = Alignment::Match;
        }
        let initial = (matrix.row, matrix.column, matrix.selected_wire());
        matrix.jump_mismatch(true);
        assert_eq!((matrix.row, matrix.column, matrix.selected_wire()), initial);
        assert!(
            matrix
                .render(&view, 40, 24, Locale::English)
                .contains("No recorded mismatch in scope")
        );
        let first = 0;
        let second = (1..matrix.routes.len())
            .find(|&i| matrix.routes[i].displayed != matrix.routes[first].displayed)
            .unwrap();
        matrix.routes[first].alignment = Alignment::Mismatch;
        matrix.routes[second].alignment = Alignment::Mismatch;
        matrix.jump_mismatch(true);
        let first_selected = matrix.selected_wire().unwrap().0;
        assert!(matches!(
            matrix.mismatch_status,
            Some(MismatchStatus::Found { total: 2, .. })
        ));
        matrix.jump_mismatch(true);
        let second_selected = matrix.selected_wire().unwrap().0;
        assert_ne!(first_selected, second_selected);
        matrix.jump_mismatch(true);
        assert_eq!(matrix.selected_wire().unwrap().0, first_selected);
        matrix.jump_mismatch(false);
        assert_eq!(matrix.selected_wire().unwrap().0, second_selected);
        let rendered = matrix.render(&view, 40, 24, Locale::English);
        assert!(rendered.lines().nth(2).unwrap().contains('v'));
        assert!(
            rendered
                .lines()
                .skip(3)
                .any(|line| line.starts_with('>') && line.contains("[×]"))
        );
        assert!(
            rendered
                .lines()
                .all(|line| crate::terminal::text::display_width(line) <= 40)
        );

        for route in &mut matrix.routes {
            route.alignment = Alignment::Match;
        }
        let members = matrix
            .grouped_cells
            .iter()
            .find_map(|((out, input), indices)| {
                (matches!(out, OutputGroup::Module(_))
                    && matches!(input, InputGroup::Module(_))
                    && indices.len() > 1)
                    .then_some(indices.clone())
            })
            .unwrap();
        let mismatch = members[1];
        let expected = matrix.routes[mismatch].wire;
        matrix.routes[mismatch].alignment = Alignment::Mismatch;
        matrix.toggle_groups(true, true);
        matrix.jump_mismatch(true);
        assert_eq!(matrix.selected_wire().unwrap().0, expected);
        let grouped = matrix.render(&view, 40, 24, Locale::English);
        assert!(grouped.contains("[!]"), "{grouped}");
        assert!(grouped.contains("Mismatch cell 1/1"), "{grouped}");
    }

    #[test]
    fn mismatch_jump_reveals_a_connection_outside_the_current_graph_page() {
        let mut modules = vec![serde_json::json!({"path":"model", "type":"Sequential"})];
        modules
            .extend((0..36).map(
                |index| serde_json::json!({"path":format!("model.{index}"), "type":"Linear"}),
            ));
        let view = ModelView::new(Some(&modules));
        let scene = view.scene(40, 24, false, None);
        let mut matrix = Matrix::new(&view, &scene, None).unwrap();
        let remote = matrix
            .routes
            .iter()
            .position(|route| {
                !scene
                    .handoffs
                    .iter()
                    .any(|visible| visible.wire == route.wire)
            })
            .unwrap();
        for route in &mut matrix.routes {
            route.alignment = Alignment::Match;
        }
        matrix.routes[remote].alignment = Alignment::Mismatch;
        let expected = matrix.routes[remote].wire;
        let revision = view.document.revision();
        matrix.jump_mismatch(true);
        assert_eq!(matrix.selected_wire().unwrap().0, expected);
        let (row_start, rows, column_start, columns) = matrix.windows(40, 24);
        assert!((row_start..row_start + rows).contains(&matrix.logical_row()));
        assert!((column_start..column_start + columns).contains(&matrix.logical_column()));
        let rendered = matrix.render(&view, 40, 24, Locale::English);
        assert!(rendered.contains("[×]"), "{rendered}");
        assert_eq!(view.document.revision(), revision);
    }

    #[test]
    fn axis_details_preserve_long_unicode_paths_and_resize_anchor() {
        let root = format!("模型{}", " 编码器 segment_".repeat(14));
        let first = format!("{root}.0");
        let modules = vec![
            serde_json::json!({"path":root,"type":"Sequential"}),
            serde_json::json!({"path":first,"type":"Linear"}),
            serde_json::json!({"path":format!("{root}.1"),"type":"Linear"}),
        ];
        let view = ModelView::new(Some(&modules));
        let scene = view.scene(40, 24, false, None);
        let mut matrix = Matrix::new(&view, &scene, None).unwrap();
        let operator = view
            .layers
            .iter()
            .find(|layer| layer.path == first)
            .unwrap()
            .id;
        matrix.row = matrix
            .rows
            .iter()
            .position(|port| port.operator == operator)
            .unwrap();
        let before = (matrix.row, matrix.column, matrix.selected_wire());
        matrix.toggle_axis();
        matrix.toggle_axis_details();
        assert!(matrix.cell_at(2, 3, 30, 8).is_none());
        let lines = matrix.axis_details_lines(&view, 30, Locale::English);
        let all = lines
            .iter()
            .map(|(line, _, _)| line.as_str())
            .collect::<String>();
        assert!(all.contains(&first), "{all}");
        assert!(all.contains("out[0]"));
        for _ in 0..3 {
            matrix.scroll_axis_details(&view, 30, 8, Locale::English, true, false);
        }
        let index = matrix.axis_detail_offset(&lines, 8);
        let anchor = (lines[index].1, lines[index].2);
        let resized = matrix.axis_details_lines(&view, 50, Locale::English);
        let new_index = matrix.axis_detail_offset(&resized, 8);
        assert_eq!(resized[new_index].1, anchor.0);
        assert!(resized[new_index].2 <= anchor.1);
        for locale in [Locale::English, Locale::Simplified, Locale::Traditional] {
            let screen = matrix.render(&view, 30, 8, locale);
            assert!(screen.lines().count() <= 8, "{screen}");
            assert!(
                screen
                    .lines()
                    .all(|line| crate::terminal::text::display_width(line) <= 30)
            );
        }
        matrix.toggle_axis_details();
        matrix.toggle_axis();
        assert_eq!((matrix.row, matrix.column, matrix.selected_wire()), before);
    }

    #[test]
    fn axis_index_stages_navigation_and_locates_grouped_ports() {
        let label = axis_label("模型.非常长的编码器.注意力", ".out[*] (12)", 20);
        assert!(
            label.starts_with('…') && label.ends_with(".out[*] (12)"),
            "{label}"
        );
        assert!(crate::terminal::text::display_width(&label) <= 20);
        let offline =
            super::super::offline::Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts);
        let view = offline.view();
        let scene = view.scene(40, 24, false, None);
        let mut matrix = Matrix::new(&view, &scene, None).unwrap();
        let initial = (
            matrix.logical_row(),
            matrix.logical_column(),
            matrix.selected_wire(),
        );
        matrix.toggle_axis();
        let rows = matrix.render(&view, 40, 24, Locale::English);
        assert!(rows.contains("Axis index · output rows"));
        assert!(rows.contains("model.encoder.out[0]"));
        assert!(
            rows.lines()
                .all(|line| crate::terminal::text::display_width(line) <= 40)
        );
        matrix.movement(1, 0);
        assert_eq!(
            (
                matrix.logical_row(),
                matrix.logical_column(),
                matrix.selected_wire()
            ),
            initial
        );
        matrix.toggle_axis();
        assert_eq!(
            (
                matrix.logical_row(),
                matrix.logical_column(),
                matrix.selected_wire()
            ),
            initial
        );
        matrix.toggle_groups(true, true);
        if matrix.rows.len() > 1 && matrix.rows[0].operator == matrix.rows[1].operator {
            matrix.row = 1;
        }
        let before = (matrix.row, matrix.column, matrix.selected_wire());
        matrix.toggle_axis();
        matrix.toggle_axis_details();
        let grouped_detail = matrix.render(&view, 40, 24, Locale::English);
        assert!(
            grouped_detail.contains("Group members 2"),
            "{grouped_detail}"
        );
        assert!(
            grouped_detail.contains("out[0]") && grouped_detail.contains("out[1]"),
            "{grouped_detail}"
        );
        matrix.toggle_axis_details();
        matrix.confirm_axis();
        assert_eq!((matrix.row, matrix.column, matrix.selected_wire()), before);
        matrix.toggle_axis();
        matrix.switch_axis();
        let columns = matrix.render(&view, 40, 24, Locale::English);
        assert!(columns.contains("Axis index · input columns"));
        assert!(columns.contains(".in[*]"));
        matrix.movement(1, 0);
        let target = matrix.axis.as_ref().unwrap().column;
        matrix.confirm_axis();
        assert_eq!(matrix.logical_column(), target);
        assert!(!matrix.axis_open());
        matrix.toggle_axis();
        let target = matrix.cell_at(2, 3, 40, 24).unwrap().0;
        assert!(matrix.click(2, 3, 40, 24));
        assert_eq!(matrix.logical_row(), target);
        assert!(!matrix.axis_open());
    }

    #[test]
    fn matrix_reopening_restores_cells_and_respects_new_graph_selection() {
        let offline =
            super::super::offline::Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts);
        let view = offline.view();
        let scene = view.scene(80, 30, false, None);
        let mut old = Matrix::new(&view, &scene, None).unwrap();
        let absent = *view.disconnected.iter().next().unwrap();
        old.row = old.rows.binary_search(&absent.from).unwrap();
        old.column = old.columns.binary_search(&absent.to).unwrap();
        assert!(old.selected_wire().is_none());
        old.toggle_groups(true, false);
        let (_, state) = old.snapshot((None, None, None));

        let mut restored = Matrix::new(&view, &scene, None).unwrap();
        restored.restore(&state, (None, None, None));
        assert!(restored.rows_grouped && !restored.columns_grouped);
        assert_eq!(restored.rows[restored.row], absent.from);
        assert_eq!(restored.columns[restored.column], absent.to);

        let wire = scene.handoffs.first().unwrap().wire;
        let mut changed = Matrix::new(&view, &scene, Some(wire)).unwrap();
        changed.restore(&state, (Some(wire), None, None));
        assert!(changed.rows_grouped);
        assert_eq!(changed.selected_wire().map(|(wire, _)| wire), Some(wire));
    }

    #[test]
    fn matrix_follows_graph_port_even_when_no_wire_is_selected() {
        let offline =
            super::super::offline::Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts);
        let mut view = offline.view();
        let scene = view.scene(80, 30, false, None);
        let mut previous = Matrix::new(&view, &scene, None).unwrap();
        assert!(previous.rows.len() > 1 && previous.columns.len() > 1);
        previous.toggle_groups(true, true);
        let (_, state) = previous.snapshot((None, None, None));

        let output = *previous.rows.last().unwrap();
        view.selected_port = Some(Port::Output(output));
        let mut matrix = Matrix::new(&view, &scene, None).unwrap();
        matrix.restore(&state, (None, view.selected_port, None));
        assert_eq!(matrix.rows[matrix.row], output);
        assert!(matrix.rows_grouped && matrix.columns_grouped);

        let input = *previous.columns.last().unwrap();
        view.selected_port = Some(Port::Input(input));
        let mut matrix = Matrix::new(&view, &scene, None).unwrap();
        matrix.restore(&state, (None, view.selected_port, None));
        assert_eq!(matrix.columns[matrix.column], input);
    }

    #[test]
    fn matrix_history_is_independent_for_each_structure_scope() {
        let mut view = ModelView::new(None);
        let scene = view.scene(80, 30, false, None);
        let mut outer = Matrix::new(&view, &scene, None).unwrap();
        outer.toggle_groups(true, false);
        let (outer_scope, outer_state) = outer.snapshot((None, None, None));

        let mut inner = None;
        for index in 0..view.layers.len() {
            view.zoom(index);
            let scene = view.scene(80, 30, false, None);
            if let Some(matrix) = Matrix::new(&view, &scene, None) {
                inner = Some(matrix);
                break;
            }
            view.zoom_out();
        }
        let mut inner = inner.expect("fixture should contain a nested scope with ports");
        inner.toggle_groups(false, true);
        let (inner_scope, inner_state) = inner.snapshot((None, None, None));
        assert!(outer_scope != inner_scope);
        let mut history = std::collections::HashMap::new();
        history.insert(outer_scope, outer_state);
        history.insert(inner_scope, inner_state);
        assert!(view.zoom_out());

        let scene = view.scene(80, 30, false, None);
        let mut reopened = Matrix::new(&view, &scene, None).unwrap();
        reopened.restore(&history[&view.graph_scope()], (None, None, None));
        assert!(reopened.rows_grouped && !reopened.columns_grouped);
        view.zoom(view.indices[&inner_scope.focus.unwrap()]);
        let scene = view.scene(80, 30, false, None);
        let mut reopened = Matrix::new(&view, &scene, None).unwrap();
        reopened.restore(&history[&view.graph_scope()], (None, None, None));
        assert!(!reopened.rows_grouped && reopened.columns_grouped);
    }

    #[test]
    fn matrix_uses_displayed_port_identity_and_preserves_the_selected_wire() {
        let offline =
            super::super::offline::Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts);
        let view = offline.view();
        let scene = view.scene(80, 30, false, None);
        let wire = scene.handoffs.first().unwrap().wire;
        let mut matrix = Matrix::new(&view, &scene, Some(wire)).unwrap();
        assert_eq!(matrix.selected_wire().map(|(wire, _)| wire), Some(wire));
        let text = matrix.render(&view, 80, 30, Locale::English);
        assert!(text.contains("Port matrix"));
        assert!(text.contains("? unknown"));
        let (row_start, _, column_start, _) = matrix.windows(80, 30);
        let x = 6 + (matrix.column - column_start) * 4;
        let y = 3 + matrix.row - row_start;
        assert!(matrix.click(x, y, 80, 30));
        assert_eq!(matrix.selected_wire().map(|(wire, _)| wire), Some(wire));
    }

    #[test]
    fn declared_absence_is_distinct_from_unknown_and_invalidated_by_topology_edits() {
        for case in [
            model_maven_tui::diagnosis::Case::CrossedPorts,
            model_maven_tui::diagnosis::Case::TwinMerge,
        ] {
            let offline = super::super::offline::Offline::new(case);
            let mut view = offline.view();
            let absent = *view.disconnected.iter().next().unwrap();
            let old_source = view.document.source(absent.to).unwrap();
            let scene = view.scene(80, 30, false, None);
            let mut matrix = Matrix::new(&view, &scene, None).unwrap();
            matrix.row = matrix
                .rows
                .iter()
                .position(|port| *port == absent.from)
                .unwrap();
            matrix.column = matrix
                .columns
                .iter()
                .position(|port| *port == absent.to)
                .unwrap();
            assert_eq!(matrix.disconnected_count(matrix.row, matrix.column), 1);
            assert!(matrix.selected_wire().is_none());
            let rendered = matrix.render(&view, 80, 30, Locale::English);
            assert!(rendered.contains("[-]"), "{rendered}");
            assert!(rendered.contains("Declared no wire"), "{rendered}");
            assert!(rendered.contains("? unknown"), "{rendered}");
            matrix.toggle_groups(false, true);
            let grouped = matrix.render(&view, 80, 30, Locale::English);
            assert!(grouped.contains("no wire 1"), "{grouped}");
            assert!(grouped.contains("? 1"), "{grouped}");

            view.apply_segment(super::super::Segment {
                base_revision: view.document.revision(),
                edits: vec![
                    super::super::Edit::Disconnect(Wire {
                        from: old_source,
                        to: absent.to,
                    }),
                    super::super::Edit::Connect(absent),
                ],
            })
            .unwrap();
            assert_eq!(view.document.source(absent.to), Some(absent.from));
            assert!(view.disconnected.is_empty());
            let scene = view.scene(80, 30, false, None);
            let matrix = Matrix::new(&view, &scene, Some(absent)).unwrap();
            assert_eq!(matrix.selected_wire().map(|(wire, _)| wire), Some(absent));
        }
    }

    #[test]
    fn matrix_cycles_distinct_wires_behind_a_shared_boundary() {
        let offline =
            super::super::offline::Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts);
        let view = offline.view();
        let scene = view.scene(80, 30, false, None);
        let mut matrix = Matrix::new(&view, &scene, None).unwrap();
        let first = matrix.routes[0].clone();
        matrix.row = matrix
            .rows
            .iter()
            .position(|port| *port == first.displayed.from)
            .unwrap();
        matrix.column = matrix
            .columns
            .iter()
            .position(|port| *port == first.displayed.to)
            .unwrap();
        let mut second = first.clone();
        second.wire.from.slot += 10;
        second.alignment = Alignment::Mismatch;
        matrix.routes.push(second.clone());
        matrix
            .cells
            .get_mut(&(first.displayed.from, first.displayed.to))
            .unwrap()
            .push(matrix.routes.len() - 1);
        assert_eq!(matrix.selected_wire().unwrap().0, first.wire);
        matrix.cycle();
        assert_eq!(matrix.selected_wire().unwrap().0, second.wire);
        assert!(
            matrix
                .render(&view, 80, 30, Locale::English)
                .contains("2/2")
        );
        matrix.cycle();
        assert_eq!(matrix.selected_wire().unwrap().0, first.wire);
    }

    #[test]
    fn matrix_includes_connections_beyond_the_visible_graph_page() {
        let mut modules = vec![serde_json::json!({"path": "model", "type": "Sequential"})];
        modules.extend(
            (0..20).map(
                |index| serde_json::json!({"path": format!("model.{index}"), "type": "Linear"}),
            ),
        );
        let view = ModelView::new(Some(&modules));
        let scene = view.scene(40, 24, false, None);
        let mut matrix = Matrix::new(&view, &scene, None).unwrap();
        assert!(matrix.disconnected.is_empty());
        assert!(matrix.routes.len() > scene.handoffs.len());
        let remote = matrix
            .routes
            .iter()
            .find(|route| {
                !scene
                    .handoffs
                    .iter()
                    .any(|visible| visible.wire == route.wire)
            })
            .unwrap();
        let wire = remote.wire;
        matrix.row = matrix
            .rows
            .iter()
            .position(|port| *port == remote.displayed.from)
            .unwrap();
        matrix.column = matrix
            .columns
            .iter()
            .position(|port| *port == remote.displayed.to)
            .unwrap();
        assert_eq!(
            matrix.selected_wire().map(|(selected, _)| selected),
            Some(wire)
        );
    }

    #[test]
    fn matrix_paging_and_narrow_layout_keep_visible_cells_addressable() {
        let mut modules = vec![serde_json::json!({"path":"model", "type":"Sequential"})];
        modules
            .extend((0..20).map(
                |index| serde_json::json!({"path":format!("model.{index}"), "type":"Linear"}),
            ));
        let view = ModelView::new(Some(&modules));
        let scene = view.scene(40, 24, false, None);
        let mut matrix = Matrix::new(&view, &scene, None).unwrap();
        assert_eq!(matrix.cell_at(6, 3, 40, 14), Some((0, 0)));
        assert!(matrix.cell_at(40, 3, 40, 14).is_none());
        matrix.page(40, 14, true);
        assert_eq!(matrix.logical_row(), 4);
        assert_eq!(matrix.cell_at(6, 3, 40, 14), Some((1, 0)));
        matrix.page(40, 14, false);
        assert_eq!(matrix.logical_row(), 0);
        for grouped in [false, true] {
            if grouped {
                matrix.toggle_groups(true, true);
            }
            let screen = matrix.render(&view, 40, 14, Locale::English);
            assert!(screen.lines().count() <= 14, "{screen}");
            assert!(screen.contains("Esc"), "{screen}");
            assert!(
                screen
                    .lines()
                    .all(|line| crate::terminal::text::display_width(line) <= 40),
                "{screen}"
            );
        }
    }

    #[test]
    fn independent_module_groups_report_coverage_and_preserve_the_selected_wire() {
        let offline =
            super::super::offline::Offline::new(model_maven_tui::diagnosis::Case::CrossedPorts);
        let view = offline.view();
        let merge = view
            .layers
            .iter()
            .find(|layer| layer.path == "net.merge")
            .unwrap()
            .id;
        let original = view
            .document
            .wires()
            .find(|wire| wire.to.operator == merge && wire.to.slot == 0)
            .unwrap();
        let scene = view.scene(80, 30, false, None);
        let mut matrix = Matrix::new(&view, &scene, Some(original)).unwrap();
        matrix.toggle_groups(true, false);
        assert_eq!(matrix.selected_wire().unwrap().0, original);
        let row_grouped = matrix.render(&view, 80, 30, Locale::English);
        assert!(
            row_grouped.contains("Pairs 1/2 · remaining unknown"),
            "{row_grouped}"
        );
        matrix.toggle_groups(false, true);
        assert_eq!(matrix.selected_wire().unwrap().0, original);
        let grouped = matrix.render(&view, 80, 30, Locale::English);
        assert!(
            grouped.contains("Pairs 2/4 · remaining unknown"),
            "{grouped}"
        );
        assert!(grouped.contains("consumers 2"), "{grouped}");
        matrix.cycle();
        let alternate = matrix.selected_wire().unwrap().0;
        assert_ne!(alternate, original);
        matrix.toggle_groups(true, false);
        assert_eq!(matrix.selected_wire().unwrap().0, alternate);
        matrix.toggle_groups(false, true);
        assert_eq!(matrix.selected_wire().unwrap().0, alternate);
        assert_eq!(matrix.rows[matrix.row], alternate.from);
        assert_eq!(matrix.columns[matrix.column], alternate.to);
    }
}
