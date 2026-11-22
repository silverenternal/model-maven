use super::{
    Bookmark, Edit, EditOverlay, GraphDocument, LayoutCache, ModelView, Operator, Port, Segment,
};
use super::{EditError, LinkInk, Wire};
use crate::terminal::locale::Locale;
use crate::workbench::{Preferences, Preset};
use model_maven_tui::diagnosis::{
    Adjustment, Call, Case, Coverage, Evidence, Fixture, Origin, Outcome, RecheckSample, Repair,
    Report,
};
use model_maven_tui::graph_document::Shape;
use std::{borrow::Cow, collections::BTreeMap};

mod display;
#[cfg(unix)]
pub(super) use display::{axis_bars, axis_label};

pub(super) const ROWS: usize = 7;
const EVIDENCE_TRACE_KEY: u8 = b'u';
#[cfg(unix)]
const DETAIL_EXPAND_KEY: u8 = b'v';

#[derive(Clone, Copy)]
enum Stage {
    Evidence,
    Preview(Draft),
    Committed,
    Rechecked { sample: RecheckSample },
}

#[derive(Clone, Copy)]
struct Draft {
    revision: u64,
    existing: Wire,
    unavailable: bool,
    error: Option<DraftError>,
    resume: Option<ReceiptScreen>,
    original_view: Bookmark,
    original_trail_len: usize,
}

#[derive(Clone, Copy)]
enum ReceiptScreen {
    Committed,
    Rechecked { sample: RecheckSample },
}

#[derive(Clone, Copy)]
enum DraftError {
    SourceUnavailable,
    AlreadyConnected,
    Stale,
}

#[derive(Clone, Copy, PartialEq)]
enum DetailMode {
    Main,
    CallEvidence {
        input: usize,
        trace_unavailable: bool,
    },
    #[cfg(unix)]
    Full {
        detail: FullDetail,
        offset: usize,
    },
    Tracing {
        input: usize,
    },
    Settings,
    Axes {
        zoom: u8,
    },
    Compare {
        side: Comparison,
        original: Bookmark,
    },
    Samples,
    Help,
    #[cfg(unix)]
    GraphHelp,
    Receipts,
}

#[cfg(unix)]
#[derive(Clone, Copy, PartialEq)]
enum FullDetail {
    Call {
        input: usize,
        trace_unavailable: bool,
    },
    Preview,
    Comparison {
        side: Comparison,
        original: Bookmark,
    },
}

struct InputEvidence<'a> {
    wire: Option<Wire>,
    actual: Option<(&'a Shape, Origin)>,
    expected: Option<(&'a Shape, Origin)>,
}

#[cfg(unix)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Shortcut {
    Previous,
    Next,
    SwitchPane,
    Expand,
}

#[cfg(unix)]
impl Shortcut {
    const ALL: [Self; 4] = [Self::Previous, Self::Next, Self::SwitchPane, Self::Expand];

    fn key(self) -> char {
        match self {
            Self::Previous => 'k',
            Self::Next => 'j',
            Self::SwitchPane => 'g',
            Self::Expand => 'o',
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Comparison {
    Before,
    After,
}

struct RecheckRejection {
    mismatch: RecheckMismatch,
    expected: String,
    received: String,
}

#[derive(Clone, Copy)]
enum RecheckMismatch {
    Batch,
    Model,
    Snapshot,
    Target,
    Revision,
}

#[derive(Clone, Copy)]
enum PreferenceNotice {
    Defaults,
    Partial,
    SaveFailed,
}

impl RecheckMismatch {
    fn label(self, locale: Locale) -> &'static str {
        match self {
            Self::Batch => {
                locale.text("Recheck batch mismatch", "重验批次不匹配", "重驗批次不相符")
            }
            Self::Model => {
                locale.text("Recheck model mismatch", "重验模型不匹配", "重驗模型不相符")
            }
            Self::Snapshot => locale.text(
                "Recheck variant or snapshot mismatch",
                "重验变体或快照不匹配",
                "重驗變體或快照不相符",
            ),
            Self::Target => locale.text(
                "Recheck target mismatch",
                "重验目标不匹配",
                "重驗目標不相符",
            ),
            Self::Revision => locale.text(
                "Recheck graph revision mismatch",
                "重验图版本不匹配",
                "重驗圖版本不相符",
            ),
        }
    }
}

impl Comparison {
    fn alternate(self) -> Self {
        match self {
            Self::Before => Self::After,
            Self::After => Self::Before,
        }
    }
}

pub(super) struct Offline {
    fixture: Fixture,
    large: bool,
    selected: usize,
    repair_index: usize,
    stage: Stage,
    preferences: Preferences,
    mode: DetailMode,
    preference_error: Option<PreferenceNotice>,
    recheck_mismatch: Option<RecheckRejection>,
    receipts: Vec<Option<RecheckSample>>,
}

pub(super) enum Exit {
    Quit,
    Welcome,
    Edit,
    EditSource,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    Preview,
    CallEvidence,
    Confirm,
    Recheck(RecheckSample),
    Samples,
    Settings,
    Axes,
    Next,
    Source,
    Compare,
    Help,
    Receipts,
    Welcome,
}

impl Action {
    const RECHECK: [RecheckSample; 4] = [
        RecheckSample::Passing,
        RecheckSample::Failing,
        RecheckSample::Truncated,
        RecheckSample::DifferentBatch,
    ];

    const ALL: [Self; 14] = [
        Self::Preview,
        Self::CallEvidence,
        Self::Confirm,
        Self::Source,
        Self::Samples,
        Self::Recheck(RecheckSample::Passing),
        Self::Recheck(RecheckSample::Failing),
        Self::Next,
        Self::Compare,
        Self::Axes,
        Self::Settings,
        Self::Receipts,
        Self::Welcome,
        Self::Help,
    ];

    fn key(self, preset: Preset) -> char {
        match (self, preset) {
            (Self::Preview, Preset::Default) => 'e',
            (Self::Preview, Preset::Shortcuts) => 'p',
            (Self::CallEvidence, _) => 'i',
            (Self::Recheck(RecheckSample::Passing), Preset::Default) => '1',
            (Self::Recheck(RecheckSample::Passing), Preset::Shortcuts) => 't',
            (Self::Recheck(RecheckSample::Failing), Preset::Default) => '2',
            (Self::Recheck(RecheckSample::Failing), Preset::Shortcuts) => 'f',
            (Self::Recheck(RecheckSample::Truncated), _) => '3',
            (Self::Recheck(RecheckSample::DifferentBatch), _) => '4',
            (Self::Samples, _) => 'r',
            (Self::Confirm, _) => 'y',
            (Self::Settings, _) => 's',
            (Self::Axes, _) => 'a',
            (Self::Next, _) => 'n',
            (Self::Source, _) => 'x',
            (Self::Compare, _) => 'c',
            (Self::Help, _) => '?',
            (Self::Receipts, _) => 'l',
            (Self::Welcome, _) => 'm',
        }
    }

    fn label(self, locale: Locale) -> &'static str {
        match self {
            Self::Preview => locale.text("Preview", "预览", "預覽"),
            Self::CallEvidence => locale.text("Call evidence", "调用证据", "呼叫證據"),
            Self::Confirm => locale.text("Confirm segment", "确认本段", "確認本段"),
            Self::Source => locale.text("Change candidate source", "切换候选来源", "切換候選來源"),
            Self::Recheck(RecheckSample::Passing) => {
                locale.text("Passing sample", "通过样例", "通過樣例")
            }
            Self::Recheck(RecheckSample::Failing) => {
                locale.text("Failing sample", "失败样例", "失敗樣例")
            }
            Self::Recheck(RecheckSample::Truncated) => {
                locale.text("Truncated trace", "截断记录样例", "截斷記錄樣例")
            }
            Self::Recheck(RecheckSample::DifferentBatch) => {
                locale.text("Different batch", "不同批次样例", "不同批次樣例")
            }
            Self::Samples => locale.text("Recheck samples", "重验样例", "重驗樣例"),
            Self::Next => locale.text("Next edit", "下一段", "下一段"),
            Self::Compare => locale.text("Compare evidence", "对照证据", "對照證據"),
            Self::Axes => locale.text("Inspect last axis", "查看末轴", "查看末軸"),
            Self::Settings => locale.text("Settings", "设置", "設定"),
            Self::Help => locale.text("Help", "帮助", "說明"),
            Self::Receipts => locale.text("Receipts", "回执", "回執"),
            Self::Welcome => locale.text("Welcome", "欢迎页", "歡迎頁"),
        }
    }
}

impl Offline {
    pub(super) fn return_to_welcome(&self, key: u8) -> bool {
        !self.modal_open()
            && char::from(key.to_ascii_lowercase()) == Action::Welcome.key(self.preferences.preset)
    }

    #[cfg(unix)]
    pub(super) fn shortcut(&self, key: u8, details_focused: bool) -> Option<Shortcut> {
        if self.preferences.preset != Preset::Shortcuts || self.modal_open() {
            return None;
        }
        Shortcut::ALL.into_iter().find(|&action| {
            !(details_focused && action == Shortcut::Expand)
                && action.key() == char::from(key.to_ascii_lowercase())
        })
    }

    #[cfg(unix)]
    pub(super) fn graph_help_open(&self) -> bool {
        self.mode == DetailMode::GraphHelp
    }

    #[cfg(unix)]
    pub(super) fn close_graph_help(&mut self) {
        if self.graph_help_open() {
            self.mode = DetailMode::Main;
        }
    }

    #[cfg(unix)]
    pub(super) fn open_graph_panel(&mut self, key: u8) -> bool {
        let key = char::from(key.to_ascii_lowercase());
        let settings = Action::Settings;
        self.mode = match key {
            '?' => DetailMode::GraphHelp,
            key if key == settings.key(self.preferences.preset) && self.available(settings) => {
                DetailMode::Settings
            }
            _ => return false,
        };
        true
    }

    pub fn new(case: Case) -> Self {
        let fixture = Fixture::load(case);
        let selected = fixture
            .report
            .focus()
            .and_then(|focus| {
                fixture
                    .report
                    .calls
                    .iter()
                    .position(|call| call.id == focus.id)
            })
            .unwrap_or_default();
        let loaded = Preferences::path()
            .as_deref()
            .map(Preferences::load_for_diagnosis)
            .transpose();
        let (preferences, preference_error) = match loaded {
            Ok(Some((preferences, true))) if preferences != Preferences::default() => {
                (preferences, Some(PreferenceNotice::Partial))
            }
            Ok(Some((_, true))) | Err(_) => {
                (Preferences::default(), Some(PreferenceNotice::Defaults))
            }
            Ok(Some((preferences, false))) => (preferences, None),
            Ok(None) => (Preferences::default(), None),
        };
        Self {
            fixture,
            large: matches!(case, Case::Large),
            selected,
            repair_index: 0,
            stage: Stage::Evidence,
            preferences,
            mode: DetailMode::Main,
            preference_error,
            recheck_mismatch: None,
            receipts: Vec::new(),
        }
    }

    pub fn view(&self) -> ModelView<'static> {
        let fixture = &self.fixture;
        let matching = fixture.snapshot.id == fixture.report.snapshot;
        let mut edits = Vec::new();
        if matching {
            if self.large {
                for (id, path) in [
                    (super::large::DIAGNOSIS_GROUP, "diagnostic"),
                    (super::large::SOURCE_GROUP, "source"),
                ] {
                    edits.push(Edit::AddOperator(
                        id,
                        Operator {
                            path: path.into(),
                            kind: "Block".into(),
                            parent: None,
                            attributes: Cow::Owned(serde_json::Value::Null),
                        },
                    ));
                }
            }
            let producer = fixture
                .report
                .focus()
                .and_then(|call| call.incoming.as_ref())
                .filter(|edge| edge.origin == Origin::Observation)
                .map(|edge| edge.value.from.operator);
            for module in &fixture.snapshot.modules {
                let group = self.large.then_some(if producer == Some(module.id) {
                    super::large::SOURCE_GROUP
                } else {
                    super::large::DIAGNOSIS_GROUP
                });
                edits.push(Edit::AddOperator(
                    module.id,
                    Operator {
                        path: module.path.into(),
                        kind: module.kind.into(),
                        parent: module.parent.or(group),
                        attributes: Cow::Owned(serde_json::Value::Null),
                    },
                ));
            }
            let mut outputs = BTreeMap::new();
            let mut inputs = BTreeMap::new();
            for (wire, actual, declared) in self.observed_inputs() {
                outputs
                    .entry(wire.from)
                    .and_modify(|shape| {
                        if *shape != actual {
                            *shape = None;
                        }
                    })
                    .or_insert(actual);
                inputs
                    .entry(wire.to)
                    .and_modify(|(source, shape)| {
                        if *source != Some(wire.from) {
                            *source = None;
                        }
                        if *shape != declared {
                            *shape = None;
                        }
                    })
                    .or_insert((Some(wire.from), declared));
            }
            for (output, shape) in &fixture.snapshot.outputs {
                outputs.insert(*output, Some(&shape.value));
            }
            edits.extend(
                outputs
                    .into_iter()
                    .map(|(output, shape)| Edit::AddPort(Port::Output(output), shape.cloned())),
            );
            for (to, (source, shape)) in inputs {
                edits.push(Edit::AddPort(Port::Input(to), shape.cloned()));
                // The document has one connection per socket. Divergent calls
                // retain their observations in the report.
                if let Some(from) = source {
                    edits.push(Edit::Connect(Wire { from, to }));
                }
            }
            if self.large {
                super::large::append(&mut edits);
            }
        }
        let mut document = GraphDocument::default();
        document
            .apply(Segment {
                base_revision: 0,
                edits,
            })
            .expect("offline fixture is valid");
        let mut view = ModelView::from_document(document, false, LayoutCache::default());
        if matching {
            view.disconnected = fixture.snapshot.disconnected.iter().copied().collect();
        }
        view.offline = true;
        view.style = self.preferences.style;
        if self.large {
            view.hierarchy = true;
            view.selected = view.indices[&super::large::DIAGNOSIS_GROUP];
            view.remember_view();
            view.hierarchy = false;
            view.focus = Some(view.selected);
        }
        self.reveal(&mut view);
        view
    }

    fn observed_inputs(&self) -> impl Iterator<Item = (Wire, Option<&Shape>, Option<&Shape>)> {
        let report = &self.fixture.report;
        let snapshot = &self.fixture.snapshot;
        report.calls.iter().flat_map(move |call| {
            let module = snapshot.resolve(report, call.id);
            let primary = call
                .incoming
                .iter()
                .filter(|edge| edge.origin == Origin::Observation)
                .map(|edge| {
                    (
                        edge.value,
                        call.input
                            .as_ref()
                            .filter(|shape| shape.origin == Origin::Observation)
                            .map(|shape| &shape.value),
                        call.expected.as_ref().map(|shape| &shape.value),
                    )
                });
            primary
                .chain(
                    call.other_inputs
                        .iter()
                        .map(|input| (input.incoming, Some(&input.actual), Some(&input.declared))),
                )
                .filter(move |(wire, _, _)| {
                    module == Some(wire.to.operator)
                        && snapshot
                            .modules
                            .iter()
                            .any(|module| module.id == wire.from.operator)
                })
        })
    }

    fn reveal(&self, view: &mut ModelView<'_>) {
        let report = &self.fixture.report;
        let call = &report.calls[self.selected];
        view.hover = None;
        view.hover_handoff = None;
        view.selected_wire = None;
        let Some(module) = self.fixture.snapshot.resolve(report, call.id) else {
            return;
        };
        if let Some(&index) = view.indices.get(&module) {
            view.selected = index;
            let group = self
                .large
                .then(|| view.indices[&super::large::DIAGNOSIS_GROUP]);
            if let Some(group) = group {
                if view.focus != Some(group) || view.hierarchy {
                    view.focus = Some(group);
                    view.hierarchy = false;
                    view.viewport = super::Viewport::default();
                }
            } else if view
                .focus
                .is_some_and(|focus| focus != index && !view.children[focus].contains(&index))
            {
                view.focus = None;
                view.viewport = super::Viewport::default();
            }
        }
        view.selected_wire = call
            .incoming
            .as_ref()
            .filter(|edge| {
                edge.origin == Origin::Observation
                    && edge.value.to.operator == module
                    && view.document.source(edge.value.to) == Some(edge.value.from)
            })
            .map(|edge| edge.value);
    }

    pub fn navigate(&mut self, next: bool, view: &mut ModelView<'_>) {
        if let DetailMode::Compare { side, .. } = &mut self.mode {
            *side = if next {
                Comparison::After
            } else {
                Comparison::Before
            };
            return;
        }
        if let DetailMode::CallEvidence { input, .. } = self.mode {
            let last = self.evidence_call().1.other_inputs.len();
            let input = if next {
                (input + 1).min(last)
            } else {
                input.saturating_sub(1)
            };
            self.mode = DetailMode::CallEvidence {
                input,
                trace_unavailable: false,
            };
            return;
        }
        if self.modal_open()
            || matches!(
                self.stage,
                Stage::Preview(_) | Stage::Committed | Stage::Rechecked { .. }
            )
        {
            return;
        }
        let last = self.fixture.report.calls.len() - 1;
        self.selected = if next {
            (self.selected + 1).min(last)
        } else {
            self.selected.saturating_sub(1)
        };
        self.reveal(view);
    }

    #[cfg(unix)]
    pub fn select_graph_call(&mut self, view: &ModelView<'_>) {
        if !matches!(self.stage, Stage::Evidence) {
            return;
        }
        let Some(layer) = view.layers.get(view.selected) else {
            return;
        };
        let report = &self.fixture.report;
        let matches = |call: &model_maven_tui::diagnosis::Call| {
            self.fixture.snapshot.resolve(report, call.id) == Some(layer.id)
        };
        if !matches(&report.calls[self.selected]) {
            if let Some(index) = report.calls.iter().position(matches) {
                self.selected = index;
            }
        }
    }

    pub fn cancel(&mut self, view: &mut ModelView<'_>) -> bool {
        let mode = std::mem::replace(&mut self.mode, DetailMode::Main);
        #[cfg(unix)]
        if let DetailMode::Full { detail, .. } = mode {
            self.mode = match detail {
                FullDetail::Call {
                    input,
                    trace_unavailable,
                } => DetailMode::CallEvidence {
                    input,
                    trace_unavailable,
                },
                FullDetail::Preview => DetailMode::Main,
                FullDetail::Comparison { side, original } => DetailMode::Compare { side, original },
            };
            return true;
        }
        if let DetailMode::Compare { original, .. } = mode {
            view.restore_view(original);
            return true;
        }
        if let DetailMode::Tracing { input } = mode {
            view.zoom_out();
            self.mode = DetailMode::CallEvidence {
                input,
                trace_unavailable: false,
            };
            return true;
        }
        if mode != DetailMode::Main {
            return true;
        }
        if self.recheck_mismatch.take().is_some() {
            return true;
        }
        if matches!(self.stage, Stage::Rechecked { .. }) {
            self.mode = DetailMode::Receipts;
            return true;
        }
        self.stage = match self.stage {
            Stage::Preview(draft) => {
                view.trail.truncate(draft.original_trail_len);
                view.restore_view(draft.original_view);
                match draft.resume {
                    Some(ReceiptScreen::Committed) => Stage::Committed,
                    Some(ReceiptScreen::Rechecked { sample }) => Stage::Rechecked { sample },
                    None => Stage::Evidence,
                }
            }
            _ => return false,
        };
        true
    }

    pub fn input(&mut self, key: u8, view: &mut ModelView<'_>) -> bool {
        let key = key.to_ascii_lowercase();
        #[cfg(unix)]
        if self.full_detail_open() {
            if matches!(key, b'b' | DETAIL_EXPAND_KEY) {
                self.cancel(view);
            }
            return key != 27;
        }
        if let DetailMode::CallEvidence { input, .. } = self.mode {
            match key {
                b'b' | b'i' => {
                    self.cancel(view);
                }
                b'j' | b'k' if self.preferences.preset == Preset::Shortcuts => {
                    self.navigate(key == b'j', view)
                }
                EVIDENCE_TRACE_KEY => {
                    let wire = self.evidence_input(input).wire;
                    match wire {
                        Some(wire) if view.follow_observed_source(wire) => {
                            self.mode = DetailMode::Tracing { input }
                        }
                        Some(_) => {
                            self.mode = DetailMode::CallEvidence {
                                input,
                                trace_unavailable: true,
                            }
                        }
                        None => {}
                    }
                }
                _ => {}
            }
            return key != 27;
        }
        if let DetailMode::Tracing { .. } = self.mode {
            if matches!(key, b'b' | EVIDENCE_TRACE_KEY) {
                self.cancel(view);
            }
            return key != 27;
        }
        #[cfg(unix)]
        if self.mode == DetailMode::GraphHelp {
            if matches!(key, b'?' | b'b') {
                self.mode = DetailMode::Main;
            }
            return key != 27;
        }
        if self.mode == DetailMode::Help {
            if matches!(key, b'?' | b'b') {
                self.mode = DetailMode::Main;
            }
            return key != 27;
        }
        if self.mode == DetailMode::Receipts {
            if matches!(key, b'l' | b'b') {
                self.mode = DetailMode::Main;
            }
            return key != 27;
        }
        if self.mode == DetailMode::Samples {
            if let Some(sample) = Action::RECHECK.into_iter().find(|sample| {
                Action::Recheck(*sample).key(self.preferences.preset) == char::from(key)
            }) {
                self.load_recheck(sample, view.document.revision());
                self.mode = DetailMode::Main;
            } else if key == b'b' {
                self.mode = DetailMode::Main;
            }
            return key != 27;
        }
        if let DetailMode::Compare { side, .. } = &mut self.mode {
            match key {
                b'c' => *side = side.alternate(),
                b'b' => {
                    self.cancel(view);
                }
                b'j' | b'k' if self.preferences.preset == Preset::Shortcuts => {
                    self.navigate(key == b'j', view);
                }
                _ => {}
            }
            return key != 27;
        }
        if matches!(self.mode, DetailMode::Axes { .. }) {
            match key {
                b'a' | b'b' => self.mode = DetailMode::Main,
                b'+' | b'=' | b'-' => {
                    self.zoom_axes(key != b'-');
                }
                _ => {}
            }
            return key != 27;
        }
        if self.mode == DetailMode::Settings {
            let mut amended = self.preferences;
            match key {
                b'1' => amended.cycle_style(),
                b'2' => amended.cycle_preset(),
                b'r' => amended = Preferences::default(),
                b'b' => self.mode = DetailMode::Main,
                27 => return false,
                _ => return true,
            }
            if self.mode == DetailMode::Settings {
                let saved = Preferences::path().is_some_and(|path| amended.save(&path).is_ok());
                self.preference_error = (!saved).then_some(PreferenceNotice::SaveFailed);
                if saved {
                    self.preferences = amended;
                    view.style = amended.style;
                }
            }
            return true;
        }
        let report = &self.fixture.report;
        let action = Action::ALL.into_iter().find(|action| {
            self.available(*action)
                && (action.key(self.preferences.preset) == char::from(key)
                    || (*action == Action::Preview && matches!(key, b'\r' | b'\n')))
        });
        match (self.stage, action) {
            (_, Some(Action::CallEvidence)) => {
                self.mode = DetailMode::CallEvidence {
                    input: 0,
                    trace_unavailable: false,
                };
            }
            (_, Some(Action::Help)) => self.mode = DetailMode::Help,
            (_, Some(Action::Receipts)) => self.mode = DetailMode::Receipts,
            (_, Some(Action::Samples)) => self.mode = DetailMode::Samples,
            (_, Some(Action::Axes)) => self.mode = DetailMode::Axes { zoom: 8 },
            (Stage::Rechecked { .. }, Some(Action::Compare)) => {
                self.mode = DetailMode::Compare {
                    side: Comparison::Before,
                    original: view.bookmark(),
                };
                self.show_input_flow(view, self.repair().receipt.target);
            }
            (
                Stage::Evidence | Stage::Committed | Stage::Rechecked { .. },
                Some(Action::Settings),
            ) => {
                self.mode = DetailMode::Settings;
            }
            (Stage::Evidence, Some(Action::Preview)) => {
                let existing = self.repair().preview.affected;
                self.begin_preview(view, existing, false, None);
            }
            (Stage::Preview(mut draft), Some(Action::Source)) => {
                draft.unavailable = !draft.unavailable;
                draft.error = None;
                self.stage = Stage::Preview(draft);
            }
            (stage @ (Stage::Committed | Stage::Rechecked { .. }), Some(Action::Source)) => {
                let to = self.repair().preview.target;
                let Some(from) = view.document.source(to) else {
                    return false;
                };
                let resume = if let Stage::Rechecked { sample } = stage {
                    ReceiptScreen::Rechecked { sample }
                } else {
                    ReceiptScreen::Committed
                };
                self.recheck_mismatch = None;
                self.begin_preview(view, Wire { from, to }, true, Some(resume));
            }
            (Stage::Preview(draft), Some(Action::Confirm)) => self.commit(draft, view),
            (Stage::Committed | Stage::Rechecked { .. }, Some(Action::Next)) => {
                self.repair_index += 1;
                let call = self.repair().call;
                self.selected = report
                    .calls
                    .iter()
                    .position(|entry| entry.id == call)
                    .expect("repair refers to an observed call");
                self.recheck_mismatch = None;
                self.stage = Stage::Evidence;
                self.reveal(view);
            }
            (Stage::Committed | Stage::Rechecked { .. }, Some(Action::Recheck(sample))) => {
                self.load_recheck(sample, view.document.revision())
            }
            _ => return false,
        }
        true
    }

    fn begin_preview(
        &mut self,
        view: &mut ModelView<'_>,
        existing: Wire,
        unavailable: bool,
        resume: Option<ReceiptScreen>,
    ) {
        self.stage = Stage::Preview(Draft {
            revision: view.document.revision(),
            existing,
            unavailable,
            error: None,
            resume,
            original_view: view.bookmark(),
            original_trail_len: view.trail.len(),
        });
        self.show_input_flow(view, existing.to);
    }

    fn show_input_flow(&self, view: &mut ModelView<'_>, to: super::Inlet) {
        self.reveal(view);
        let Some(&index) = view.indices.get(&to.operator) else {
            return;
        };
        view.selected = index;
        view.selected_wire = view.document.source(to).map(|from| Wire { from, to });
        let prior_scope = view.focus;
        if view
            .available_flow()
            .is_none_or(|flow| !flow.contains(index))
        {
            view.focus = view.parents[index];
        }
        if view.hierarchy || prior_scope != view.focus {
            view.hierarchy = false;
            view.viewport = super::Viewport::default();
        }
    }

    fn available(&self, action: Action) -> bool {
        let receipt = matches!(self.stage, Stage::Committed | Stage::Rechecked { .. });
        let preview = matches!(self.stage, Stage::Preview(_));
        match action {
            Action::Preview => matches!(self.stage, Stage::Evidence) && self.can_edit(),
            Action::Confirm => preview,
            Action::Source => {
                (preview || receipt)
                    && matches!(self.repair().preview.change, Adjustment::Source { .. })
            }
            Action::Recheck(_) | Action::Samples => receipt,
            Action::Next => receipt && self.repair_index + 1 < self.fixture.repairs.len(),
            Action::Compare => matches!(self.stage, Stage::Rechecked { .. }),
            Action::Settings | Action::CallEvidence => !preview,
            Action::Receipts => !self.receipts.is_empty(),
            Action::Axes | Action::Help | Action::Welcome => true,
        }
    }

    fn commit(&mut self, mut draft: Draft, view: &mut ModelView<'_>) {
        let repair = self.repair();
        let preview = &repair.preview;
        let receipt = &repair.receipt;
        let bound = self.can_edit()
            && preview.report == self.fixture.report.id
            && receipt.preview_report == preview.report
            && receipt.target == preview.target
            && view.document.source(preview.target) == Some(draft.existing.from);
        let port = Port::Input(preview.target);
        let mut wire = draft.existing;
        let edits = match &preview.change {
            Adjustment::Shape { before, after } if view.document.shape(port) == Some(before) => {
                vec![Edit::SetShape(port, Some(after.value.clone()))]
            }
            Adjustment::Source { after, unavailable } => {
                wire.from = if draft.unavailable {
                    *unavailable
                } else {
                    after.value
                };
                vec![Edit::Disconnect(draft.existing), Edit::Connect(wire)]
            }
            _ => Vec::new(),
        };
        let result = if !bound || edits.is_empty() || draft.revision != view.document.revision() {
            Err(DraftError::Stale)
        } else if draft.resume.is_some() && wire == draft.existing {
            Err(DraftError::AlreadyConnected)
        } else {
            // A follow-up may test rejection at the current revision. Only
            // a recorded candidate may publish a new graph revision.
            let recorded = draft.revision.checked_add(1) == Some(receipt.graph_revision);
            let missing_source = draft.unavailable
                && !view
                    .document
                    .ports_for(wire.from.operator)
                    .any(|(port, _)| port == Port::Output(wire.from));
            if (recorded && !draft.unavailable) || missing_source {
                view.apply_segment(Segment {
                    base_revision: draft.revision,
                    edits,
                })
                .map_err(|error| match error {
                    EditError::MissingPort(Port::Output(_)) => DraftError::SourceUnavailable,
                    _ => DraftError::Stale,
                })
                .map(|_| ())
            } else {
                Err(DraftError::Stale)
            }
        };
        match result {
            Ok(()) => {
                self.show_input_flow(view, wire.to);
                self.receipts.push(None);
                self.stage = Stage::Committed;
            }
            Err(error) => {
                draft.error = Some(error);
                self.stage = Stage::Preview(draft);
            }
        }
    }

    pub fn modal_open(&self) -> bool {
        self.mode != DetailMode::Main
    }

    #[cfg(unix)]
    pub(super) fn full_detail_open(&self) -> bool {
        matches!(self.mode, DetailMode::Full { .. })
    }

    #[cfg(unix)]
    pub(super) fn open_full_detail(&mut self, key: u8) -> bool {
        if key.to_ascii_lowercase() != DETAIL_EXPAND_KEY {
            return false;
        }
        let detail = match self.mode {
            DetailMode::CallEvidence {
                input,
                trace_unavailable,
            } => FullDetail::Call {
                input,
                trace_unavailable,
            },
            DetailMode::Main if matches!(self.stage, Stage::Preview(_)) => FullDetail::Preview,
            DetailMode::Compare { side, original } => FullDetail::Comparison { side, original },
            _ => return false,
        };
        self.mode = DetailMode::Full { detail, offset: 0 };
        true
    }

    pub(super) fn confirmed_target(&self) -> Option<super::Inlet> {
        matches!(self.stage, Stage::Committed).then(|| self.repair().preview.target)
    }

    pub fn zoom_axes(&mut self, inward: bool) -> bool {
        let DetailMode::Axes { zoom } = &mut self.mode else {
            return false;
        };
        let next = if inward {
            (*zoom + 1).min(10)
        } else {
            (*zoom - 1).max(5)
        };
        let changed = *zoom != next;
        *zoom = next;
        changed
    }

    fn axis_evidence(&self) -> ([Option<&Evidence<Shape>>; 2], Option<usize>) {
        let call = &self.fixture.report.calls[self.selected];
        let evidence = [call.input.as_ref(), call.expected.as_ref()];
        match self.stage {
            Stage::Preview(draft) => match &self.repair().preview.change {
                Adjustment::Shape { after, .. } => ([evidence[0], Some(after)], Some(1)),
                Adjustment::Source { after, .. } => {
                    let candidate = self
                        .fixture
                        .snapshot
                        .outputs
                        .iter()
                        .find(|(outlet, _)| !draft.unavailable && *outlet == after.value)
                        .map(|(_, shape)| shape);
                    ([candidate, evidence[1]], Some(0))
                }
            },
            Stage::Rechecked { sample } => (
                self.rechecked_call(self.repair().recheck(sample))
                    .map_or([None; 2], |call| {
                        [call.input.as_ref(), call.expected.as_ref()]
                    }),
                None,
            ),
            _ => (evidence, None),
        }
    }

    #[cfg(unix)]
    pub fn axis_view(&self) -> Option<super::axes::Comparison> {
        let DetailMode::Axes { zoom } = self.mode else {
            return None;
        };
        let (evidence, candidate) = self.axis_evidence();
        let shapes = evidence.map(|entry| entry.map(|entry| &entry.value));
        Some(super::axes::Comparison::from_shapes(
            shapes, candidate, zoom,
        ))
    }

    pub fn graph_overlay(&self, view: &ModelView<'_>) -> Option<EditOverlay> {
        if let DetailMode::Compare { side, .. } = self.mode {
            let to = self.repair().receipt.target;
            let current = super::Wire {
                from: view.document.source(to)?,
                to,
            };
            let (report, call) = self.comparison_evidence(side)?;
            let observed = call
                .incoming
                .as_ref()
                .filter(|edge| edge.origin == Origin::Observation)
                .map(|edge| edge.value);
            let ink = match (&report.coverage.value, report.outcome) {
                (Coverage::Truncated, _) => LinkInk::Unknown,
                (_, Outcome::Passed) => LinkInk::Confirmed,
                _ => LinkInk::Draft,
            };
            return Some(EditOverlay::Evidence {
                current,
                observed,
                ink,
            });
        }
        let Stage::Preview(draft) = self.stage else {
            return None;
        };
        let preview = &self.repair().preview;
        let candidate = match &preview.change {
            Adjustment::Source { after, .. } if !draft.unavailable => Some(super::Wire {
                from: after.value,
                to: preview.target,
            }),
            _ => None,
        };
        Some(EditOverlay::Draft {
            existing: draft.existing,
            candidate: candidate.filter(|wire| *wire != draft.existing),
        })
    }

    fn can_edit(&self) -> bool {
        let Some(repair) = self.fixture.repairs.get(self.repair_index) else {
            return false;
        };
        let preview = &repair.preview;
        let report = &self.fixture.report;
        let call = &report.calls[self.selected];
        let bound = call.id == repair.call
            && call.incoming.as_ref().is_some_and(|edge| {
                edge.origin == Origin::Observation && edge.value == preview.affected
            })
            && self.fixture.snapshot.resolve(report, call.id) == Some(preview.target.operator);
        if !bound {
            return false;
        }
        let mut observations = self
            .observed_inputs()
            .filter(|(wire, _, _)| wire.to == preview.target)
            .peekable();
        observations.peek().is_some()
            && observations.all(|(wire, _, expected)| {
                wire == preview.affected
                    && match &preview.change {
                        Adjustment::Shape { before, .. } => expected == Some(before),
                        Adjustment::Source { .. } => true,
                    }
            })
    }

    fn repair(&self) -> &Repair {
        &self.fixture.repairs[self.repair_index]
    }

    fn evidence_call(&self) -> (&Report, &Call) {
        if let Stage::Rechecked { sample } = self.stage {
            let report = self.repair().recheck(sample);
            let call = self
                .rechecked_call(report)
                .expect("accepted recheck contains the edited call");
            return (report, call);
        }
        let report = &self.fixture.report;
        (report, &report.calls[self.selected])
    }

    #[cfg(unix)]
    pub(super) fn failure_path(&self) -> super::failure::Path {
        use super::failure::{Input, Path, Stop};
        let (report, call) = self.evidence_call();
        let stop = match report.outcome {
            Outcome::FailedAt(id) if id == call.id => Stop::Failure,
            Outcome::LastObserved(id) if id == call.id => Stop::LastObserved,
            Outcome::Passed => Stop::Passed,
            _ => Stop::Call,
        };
        let inputs = (0..=call.other_inputs.len())
            .map(|index| {
                let evidence = self.evidence_input(index);
                Input {
                    wire: evidence.wire,
                    actual: evidence
                        .actual
                        .map(|(shape, origin)| (shape.clone(), origin)),
                    expected: evidence
                        .expected
                        .map(|(shape, origin)| (shape.clone(), origin)),
                    label: None,
                    metadata: None,
                    matches: Vec::new(),
                }
            })
            .collect();
        Path::new(
            call.id.alias.to_owned(),
            Some(u64::from(call.id.invocation)),
            Some(format!("{} · {}", report.id, report.batch)),
            stop,
            inputs,
        )
    }

    #[cfg(unix)]
    pub(super) fn repair_diff(
        &self,
        view: &ModelView<'_>,
        locale: Locale,
    ) -> Option<super::diff::Diff> {
        use model_maven_tui::diagnosis::WeightImpact;
        let draft = match self.stage {
            Stage::Evidence => return None,
            Stage::Preview(draft) => Some(draft),
            _ => None,
        };
        let repair = self.repair();
        let preview = &repair.preview;
        let target = format!(
            "{}.in[{}]",
            self.module_path(preview.target.operator),
            preview.target.slot
        );
        let old = draft.map_or(preview.affected.from, |draft| draft.existing.from);
        let (new, before_shape, after_shape) = match &preview.change {
            Adjustment::Shape { before, after } => (old, before.label(), after.value.label()),
            Adjustment::Source { after, unavailable } => (
                if draft.is_some_and(|draft| draft.unavailable) {
                    *unavailable
                } else {
                    after.value
                },
                super::shape_label(view.document.shape(Port::Input(preview.target))),
                super::shape_label(view.document.shape(Port::Input(preview.target))),
            ),
        };
        let status = if let Some(error) = draft.and_then(|draft| draft.error) {
            match error {
                DraftError::SourceUnavailable => locale.text(
                    "Rejected · source unavailable",
                    "已拒绝 · 来源不可用",
                    "已拒絕 · 來源不可用",
                ),
                DraftError::AlreadyConnected => locale.text(
                    "Rejected · already connected",
                    "已拒绝 · 已存在连接",
                    "已拒絕 · 已存在連接",
                ),
                DraftError::Stale => locale.text(
                    "Rejected · graph revision changed",
                    "已拒绝 · 图版本已变化",
                    "已拒絕 · 圖版本已變化",
                ),
            }
        } else if self.recheck_mismatch.is_some() {
            locale.text(
                "Recheck rejected · previous evidence retained",
                "重验已拒绝 · 保留之前的证据",
                "重驗已拒絕 · 保留先前證據",
            )
        } else {
            match self.stage {
                Stage::Preview(_) => locale.text(
                    "Preview · not committed",
                    "预览 · 尚未提交",
                    "預覽 · 尚未提交",
                ),
                Stage::Committed => locale.text(
                    "Committed · awaiting recheck",
                    "已提交 · 待重验",
                    "已提交 · 待重驗",
                ),
                Stage::Rechecked { sample } => {
                    let report = repair.recheck(sample);
                    if report.coverage.value == Coverage::Truncated
                        || matches!(report.outcome, Outcome::LastObserved(_))
                    {
                        locale.text(
                            "Recheck · coverage incomplete",
                            "重验 · 覆盖不足",
                            "重驗 · 覆蓋不足",
                        )
                    } else if matches!(report.outcome, Outcome::Passed) {
                        locale.text(
                            "Recheck · sample passed",
                            "重验 · 样例通过",
                            "重驗 · 樣例通過",
                        )
                    } else {
                        locale.text(
                            "Recheck · still failing",
                            "重验 · 仍然失败",
                            "重驗 · 仍然失敗",
                        )
                    }
                }
                Stage::Evidence => unreachable!(),
            }
        };
        let weights = match preview.weights.as_ref().map(|value| &value.value) {
            Some(WeightImpact::Reset(name)) => format!(
                "{} {name}",
                locale.text("Predicted reset", "预计重置", "預計重設")
            ),
            Some(WeightImpact::Reuse(name)) => format!(
                "{} {name}",
                locale.text("Predicted reuse", "预计复用", "預計重用")
            ),
            None => locale
                .text("Weight effect unknown", "权重影响未知", "權重影響未知")
                .to_owned(),
        };
        let mut new_source = self.output_label(new);
        if draft.is_some_and(|draft| draft.unavailable) {
            new_source.push_str(locale.text(" · unavailable", " · 不可用", " · 不可用"));
        }
        let before = vec![
            locale
                .text("Before · recorded", "修改前 · 已记录", "修改前 · 已記錄")
                .into(),
            format!("┌─ {}", self.output_label(old)),
            "│".into(),
            format!("└─▶ {target}"),
            format!("{} {before_shape}", locale.text("Input", "输入", "輸入")),
            format!(
                "{} {}",
                locale.text("Variant", "变体", "變體"),
                self.fixture.report.snapshot.variant
            ),
        ];
        let after = vec![
            if draft.is_some() {
                locale.text("Candidate · preview", "候选 · 预览", "候選 · 預覽")
            } else {
                locale.text("After · committed", "修改后 · 已提交", "修改後 · 已提交")
            }
            .into(),
            format!("┌─ {new_source}"),
            "│".into(),
            format!("└─▶ {target}"),
            format!("{} {after_shape}", locale.text("Input", "输入", "輸入")),
            weights.clone(),
        ];
        let mut details = vec![
            format!("{} {target}", locale.text("Target", "目标", "目標")),
            format!(
                "{}: {} → {}",
                locale.text("Source", "来源", "來源"),
                self.output_label(old),
                new_source
            ),
            format!(
                "{}: {before_shape} → {after_shape}",
                locale.text("Input", "输入", "輸入")
            ),
            weights,
            format!(
                "{}: {} / {} / {}",
                locale.text("Before", "修改前", "修改前"),
                self.fixture.report.snapshot.model.name,
                self.fixture.report.snapshot.variant,
                self.fixture.report.id
            ),
            format!(
                "{}: {}",
                locale.text("Original batch", "原批次", "原批次"),
                self.fixture.report.batch
            ),
            status.to_owned(),
        ];
        if draft.is_none() {
            details.push(format!(
                "{}: {} / {} · v{}",
                locale.text("Receipt", "回执", "回執"),
                repair.receipt.snapshot.variant,
                repair.receipt.snapshot.snapshot,
                repair.receipt.graph_revision
            ));
        }
        if let Stage::Rechecked { sample } = self.stage {
            let report = repair.recheck(sample);
            details.push(format!(
                "{}: {} / {} / {} · {}",
                locale.text("Recheck", "重验", "重驗"),
                report.snapshot.model.name,
                report.snapshot.variant,
                report.id,
                report.batch
            ));
        }
        if let Some(rejection) = &self.recheck_mismatch {
            details.push(format!(
                "{}: {} ≠ {}",
                rejection.mismatch.label(locale),
                rejection.expected,
                rejection.received
            ));
        }
        Some(super::diff::Diff::new(
            target,
            status.to_owned(),
            before,
            after,
            details,
        ))
    }

    fn evidence_input(&self, index: usize) -> InputEvidence<'_> {
        let (report, call) = self.evidence_call();
        let module = if matches!(self.stage, Stage::Rechecked { .. }) {
            Some(self.repair().receipt.target.operator)
        } else {
            self.fixture.snapshot.resolve(report, call.id)
        };
        if let Some(input) = index
            .checked_sub(1)
            .and_then(|index| call.other_inputs.get(index))
        {
            return InputEvidence {
                wire: (module == Some(input.incoming.to.operator)).then_some(input.incoming),
                actual: Some((&input.actual, Origin::Observation)),
                expected: Some((&input.declared, Origin::Declaration)),
            };
        }
        InputEvidence {
            wire: call.incoming.as_ref().and_then(|edge| {
                (edge.origin == Origin::Observation && module == Some(edge.value.to.operator))
                    .then_some(edge.value)
            }),
            actual: call
                .input
                .as_ref()
                .map(|shape| (&shape.value, shape.origin)),
            expected: call
                .expected
                .as_ref()
                .map(|shape| (&shape.value, shape.origin)),
        }
    }

    fn load_recheck(&mut self, sample: RecheckSample, revision: u64) {
        let candidate = self.repair().recheck(sample);
        self.recheck_mismatch = self.recheck_rejection(candidate, revision);
        if self.recheck_mismatch.is_none() {
            self.receipts[self.repair_index] = Some(sample);
            self.stage = Stage::Rechecked { sample };
        }
    }

    fn rechecked_call<'a>(
        &self,
        report: &'a Report,
    ) -> Option<&'a model_maven_tui::diagnosis::Call> {
        report
            .calls
            .iter()
            .find(|call| call.id == self.repair().call)
    }

    fn recheck_rejection(&self, candidate: &Report, revision: u64) -> Option<RecheckRejection> {
        let repair = self.repair();
        let receipt = &repair.receipt;
        let rejection = |mismatch, expected, received| {
            Some(RecheckRejection {
                mismatch,
                expected,
                received,
            })
        };
        if candidate.batch != self.fixture.report.batch {
            return rejection(
                RecheckMismatch::Batch,
                self.fixture.report.batch.into(),
                candidate.batch.into(),
            );
        }
        if candidate.snapshot.model != receipt.snapshot.model {
            let model_label = |model: model_maven_tui::diagnosis::Model| {
                format!("{}/{}", model.project, model.name)
            };
            return rejection(
                RecheckMismatch::Model,
                model_label(receipt.snapshot.model),
                model_label(candidate.snapshot.model),
            );
        }
        if candidate.snapshot != receipt.snapshot {
            let variant_label = |id: model_maven_tui::diagnosis::SnapshotId| {
                format!("{}/{}", id.variant, id.snapshot)
            };
            return rejection(
                RecheckMismatch::Snapshot,
                variant_label(receipt.snapshot),
                variant_label(candidate.snapshot),
            );
        }
        if revision != receipt.graph_revision {
            return rejection(
                RecheckMismatch::Revision,
                format!("v{}", receipt.graph_revision),
                format!("v{revision}"),
            );
        }
        let target = self.rechecked_call(candidate);
        if target
            .and_then(|call| call.incoming.as_ref())
            .is_some_and(|edge| {
                edge.origin == Origin::Observation && edge.value.to == receipt.target
            })
        {
            return None;
        }
        let target_label = |call: model_maven_tui::diagnosis::CallId,
                            port: Option<super::Inlet>| {
            let path = port.map_or(call.alias, |port| self.module_path(port.operator));
            let slot = port.map_or_else(|| "?".into(), |port| port.slot.to_string());
            format!("{path}.in[{slot}] #{}", call.invocation)
        };
        let received = target
            .or_else(|| candidate.focus())
            .or_else(|| candidate.calls.first());
        rejection(
            RecheckMismatch::Target,
            target_label(repair.call, Some(receipt.target)),
            received.map_or_else(
                || "?".into(),
                |call| {
                    target_label(
                        call.id,
                        call.incoming
                            .as_ref()
                            .filter(|edge| edge.origin == Origin::Observation)
                            .map(|edge| edge.value.to),
                    )
                },
            ),
        )
    }

    fn comparison_evidence(
        &self,
        side: Comparison,
    ) -> Option<(&Report, &model_maven_tui::diagnosis::Call)> {
        let Stage::Rechecked { sample } = self.stage else {
            return None;
        };
        let report = match side {
            Comparison::Before => &self.fixture.report,
            Comparison::After => self.repair().recheck(sample),
        };
        Some((report, self.rechecked_call(report)?))
    }

    fn module_path(&self, id: super::OperatorId) -> &str {
        self.fixture
            .snapshot
            .modules
            .iter()
            .find(|module| module.id == id)
            .map_or("?", |module| module.path)
    }

    fn output_label(&self, port: super::Outlet) -> String {
        format!("{}.out[{}]", self.module_path(port.operator), port.slot)
    }

    pub fn run_line(&mut self, view: &mut ModelView<'_>, locale: Locale) -> std::io::Result<Exit> {
        use std::io::{self, IsTerminal, Write};
        let input_stream = io::stdin();
        let show_prompt = input_stream.is_terminal();
        let mut output = io::stderr().lock();
        let mut input = String::new();
        loop {
            let width =
                crate::terminal::layout::content_width(crate::terminal::layout::viewport_width());
            write!(output, "{}", self.render_transcript(view, locale, width))?;
            if show_prompt {
                write!(output, "> ")?;
            }
            output.flush()?;
            input.clear();
            if input_stream.read_line(&mut input)? == 0 {
                return Ok(Exit::Quit);
            }
            match input.trim().as_bytes() {
                &[key] => match key.to_ascii_lowercase() {
                    b'q' => return Ok(Exit::Quit),
                    key if self.return_to_welcome(key) => return Ok(Exit::Welcome),
                    b'j' => self.navigate(true, view),
                    b'k' => self.navigate(false, view),
                    b'b' => {
                        self.cancel(view);
                    }
                    key => {
                        self.input(key, view);
                    }
                },
                [] => {
                    self.input(b'\n', view);
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::display::axis_bars;
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn details_action_keys_are_unique_for_both_presets() {
        for preset in [Preset::Default, Preset::Shortcuts] {
            let keys: BTreeSet<_> = Action::ALL
                .into_iter()
                .map(|action| action.key(preset))
                .collect();
            assert_eq!(keys.len(), Action::ALL.len());
            assert!(!keys.contains(&'q'));
            #[cfg(unix)]
            if preset == Preset::Shortcuts {
                let shortcuts: BTreeSet<_> = Shortcut::ALL.into_iter().map(Shortcut::key).collect();
                assert_eq!(shortcuts.len(), Shortcut::ALL.len());
                assert!(keys.is_disjoint(&shortcuts));
                for reserved in [
                    'b', 'w', 'u', 'd', '[', ']', '+', '=', '-', 'r', 'h', '?', 'q', 's',
                ] {
                    assert!(!shortcuts.contains(&reserved));
                }
            }
            let sample_keys: BTreeSet<_> = Action::RECHECK
                .into_iter()
                .map(|sample| Action::Recheck(sample).key(preset))
                .collect();
            assert_eq!(sample_keys.len(), Action::RECHECK.len());
            assert!(!sample_keys.contains(&'b'));
        }
    }

    #[cfg(unix)]
    #[test]
    fn failure_path_preserves_call_evidence_and_stops_at_unknown_boundaries() {
        let mut offline = Offline::new(Case::Mismatch);
        let mut view = offline.view();
        let path = offline.failure_path();
        let text = path.render(&view, 80, 24, Locale::English, Some(&offline));
        assert!(text.contains("Failure · model.head #2"));
        assert!(text.contains("2×48 (observed)"));
        assert!(text.contains("…×32 (declared)"));
        assert!(text.contains("shared scale"));
        offline.navigate(false, &mut view);
        let previous =
            offline
                .failure_path()
                .render(&view, 80, 24, Locale::English, Some(&offline));
        assert!(previous.contains("Observed call · model.head #1"));
        assert!(previous.contains("2×32 (observed)"));
        assert!(previous.contains("Source not recorded"));

        let missing = Offline::new(Case::SnapshotMissing);
        let text =
            missing
                .failure_path()
                .render(&missing.view(), 80, 24, Locale::English, Some(&missing));
        assert!(text.contains("Last observed · failure location unknown"));
        assert!(text.contains("Source not recorded"));
    }

    #[cfg(unix)]
    #[test]
    fn failure_path_exposes_conflicting_sources_without_asserting_graph_wiring() {
        let mut offline = Offline::new(Case::Mismatch);
        let alternate = offline.fixture.snapshot.outputs[0].0;
        let main = offline.fixture.report.calls[1]
            .incoming
            .as_ref()
            .unwrap()
            .value;
        offline.fixture.report.calls[0].incoming = Some(Evidence {
            value: Wire {
                from: alternate,
                to: main.to,
            },
            origin: Origin::Observation,
        });
        let mut view = offline.view();
        assert!(view.document.source(main.to).is_none());
        let first = offline
            .failure_path()
            .render(&view, 80, 24, Locale::English, Some(&offline));
        assert!(first.contains("model.encoder.out[0]"));
        assert!(first.contains("overview differs or has no edge"));
        offline.navigate(false, &mut view);
        let second = offline
            .failure_path()
            .render(&view, 80, 24, Locale::English, Some(&offline));
        assert!(second.contains("net.adapter.out[0]"));
        assert!(second.contains("overview differs or has no edge"));
        assert!(view.selected_wire.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn repair_diff_tracks_preview_commit_and_recheck_without_mutating_the_graph() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        assert!(offline.repair_diff(&view, Locale::English).is_none());
        assert!(offline.input(b'e', &mut view));
        let before = view.bookmark();
        let revision = view.document.revision();
        let mut diff = offline.repair_diff(&view, Locale::English).unwrap();
        let wide = diff.render(100, 24, Locale::English);
        assert!(wide.contains("Preview · not committed"));
        assert!(wide.contains("…×32"));
        assert!(wide.contains("…×48"));
        assert!(wide.contains("Predicted reset"));
        let narrow = diff.render(40, 24, Locale::English);
        assert!(narrow.contains("Before · recorded"));
        diff.change_side();
        assert!(
            diff.render(40, 24, Locale::English)
                .contains("Candidate · preview")
        );
        diff.toggle_details();
        assert!(
            diff.render(80, 24, Locale::English)
                .contains("Original batch: changing-batch-01")
        );
        assert_eq!(view.document.revision(), revision);
        assert!(view.bookmark() == before);
        assert!(offline.input(b'y', &mut view));
        assert!(
            offline
                .repair_diff(&view, Locale::English)
                .unwrap()
                .render(100, 24, Locale::English)
                .contains("Committed · awaiting recheck")
        );
        assert!(offline.input(b'r', &mut view));
        assert!(offline.input(b'2', &mut view));
        let failed = offline.repair_diff(&view, Locale::English).unwrap();
        assert!(
            failed
                .render(100, 24, Locale::English)
                .contains("Recheck · still failing")
        );
        assert!(offline.input(b'r', &mut view));
        assert!(offline.input(b'3', &mut view));
        assert!(
            offline
                .repair_diff(&view, Locale::English)
                .unwrap()
                .render(100, 24, Locale::English)
                .contains("coverage incomplete")
        );
        assert!(offline.input(b'r', &mut view));
        assert!(offline.input(b'4', &mut view));
        let mut rejected = offline.repair_diff(&view, Locale::English).unwrap();
        assert!(
            rejected
                .render(100, 24, Locale::English)
                .contains("Recheck rejected")
        );
        rejected.toggle_details();
        let details = rejected.render(100, 24, Locale::English);
        assert!(details.contains("different-batch-02"));
        assert!(details.contains("changing-batch-01"));
    }

    #[cfg(unix)]
    #[test]
    fn preview_overlay_and_target_survive_graph_preset_changes() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        assert!(offline.input(b'e', &mut view));
        view.edit_overlay = offline.graph_overlay(&view);
        let overlay = view.edit_overlay;
        let target = offline.repair().preview.affected;
        assert_eq!(view.selected_wire, Some(target));
        assert!(matches!(overlay, Some(EditOverlay::Draft { .. })));

        for preset in [3, 4, 1, 2, 4, 1] {
            assert!(view.select_graph_preset(preset));
            view.edit_overlay = offline.graph_overlay(&view);
            assert!(view.edit_overlay == overlay);
            assert_eq!(view.selected_wire, Some(target));

            for pixels in [false, true] {
                let scene = view.scene(100, 30, pixels, None);
                let marker = scene.edit_mark.expect("preview marker remains visible");
                let route = scene
                    .handoffs
                    .iter()
                    .find(|route| route.wire == target)
                    .expect("edited input remains in the current graph preset");
                assert_eq!(marker.point, route.to);
                assert!(marker.ink == LinkInk::Draft);
            }
        }
        assert_eq!(view.document.revision(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn repair_diff_keeps_rejected_rewire_and_original_revision_distinct() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        for key in b"eyne" {
            assert!(offline.input(*key, &mut view));
        }
        let revision = view.document.revision();
        let candidate =
            offline
                .repair_diff(&view, Locale::English)
                .unwrap()
                .render(100, 24, Locale::English);
        assert!(candidate.contains("model.encoder.out[0]"));
        assert!(candidate.contains("net.adapter.out[0]"));
        assert!(offline.input(b'x', &mut view));
        let missing =
            offline
                .repair_diff(&view, Locale::English)
                .unwrap()
                .render(100, 24, Locale::English);
        assert!(missing.contains("net.adapter.out[9] · unavailable"));
        assert!(offline.input(b'y', &mut view));
        let rejected =
            offline
                .repair_diff(&view, Locale::English)
                .unwrap()
                .render(100, 24, Locale::English);
        assert!(rejected.contains("Rejected · source unavailable"));
        assert_eq!(view.document.revision(), revision);
    }

    #[test]
    fn repeated_calls_share_one_socket_without_sharing_their_actual_shapes() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let calls = &mut offline.fixture.report.calls;
        calls[0].incoming = calls[1].incoming.clone();
        calls[0].expected = calls[1].expected.clone();
        let wire = calls[1].incoming.as_ref().unwrap().value;
        let mut view = offline.view();
        assert_eq!(view.document.source(wire.to), Some(wire.from));
        assert_eq!(view.document.shape(Port::Output(wire.from)), None);
        assert_eq!(
            view.document.shape(Port::Input(wire.to)),
            Some(&Shape::Features(32))
        );
        assert_eq!(view.selected_wire, Some(wire));
        assert!(offline.can_edit());

        offline.navigate(false, &mut view);
        assert_eq!(view.selected_wire, Some(wire));
        assert!(
            offline
                .render(Locale::English, 80, true, false)
                .contains("Observed 2×32")
        );
        offline.navigate(true, &mut view);
        assert!(
            offline
                .render(Locale::English, 80, true, false)
                .contains("Observed 2×48")
        );
    }

    #[test]
    fn conflicting_producers_remain_call_evidence_without_becoming_a_graph_edge() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let alternate = offline.fixture.snapshot.outputs[0].0;
        let calls = &mut offline.fixture.report.calls;
        let main = calls[1].incoming.as_ref().unwrap().value;
        let alternate = Wire {
            from: alternate,
            to: main.to,
        };
        calls[0].incoming = Some(Evidence {
            value: alternate,
            origin: Origin::Observation,
        });
        calls[0].expected = calls[1].expected.clone();
        let mut view = offline.view();
        assert_eq!(
            view.document.shape(Port::Input(main.to)),
            Some(&Shape::Features(32))
        );
        assert_eq!(view.document.source(main.to), None);
        assert_eq!(view.selected_wire, None);
        assert!(!offline.input(b'e', &mut view));
        assert_eq!(offline.evidence_input(0).wire, Some(main));
        offline.navigate(false, &mut view);
        assert_eq!(offline.evidence_input(0).wire, Some(alternate));
        assert_eq!(view.selected_wire, None);

        offline.fixture.report.calls[0].id.module = Some(super::super::OperatorId(u64::MAX));
        let mut view = offline.view();
        assert_eq!(view.document.source(main.to), Some(main.from));
        offline.navigate(true, &mut view);
        assert_eq!(view.selected_wire, Some(main));
        offline.navigate(false, &mut view);
        assert_eq!(view.selected_wire, None);
        assert!(
            offline
                .render(Locale::English, 80, true, false)
                .contains("Observed 2×32")
        );
    }

    #[test]
    fn declared_source_does_not_become_an_observed_edge_or_edit_target() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let selected = offline.selected;
        let edge = offline.fixture.report.calls[selected]
            .incoming
            .as_mut()
            .unwrap();
        edge.origin = Origin::Declaration;
        let wire = edge.value;
        let mut view = offline.view();
        assert!(view.document.source(wire.to).is_none());
        assert!(!offline.input(b'e', &mut view));
        assert!(offline.input(b'i', &mut view));
        assert!(offline.input(b'u', &mut view));
        assert!(matches!(offline.mode, DetailMode::CallEvidence { .. }));
        let details = offline.render(Locale::English, 80, true, false);
        assert!(details.contains("Expected …×32 (declared)"));
        assert!(details.contains("Source unknown"));
        assert!(!details.contains("u Producer"));
    }

    #[cfg(unix)]
    #[test]
    fn trace_restores_the_camera_and_the_prior_selected_wire() {
        let mut offline = Offline::new(Case::Large);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        view.camera.orbit(0.23, 0.18);
        view.viewport.zoom(1.15, [30.0, 6.0]);
        let original = (
            view.camera,
            view.viewport,
            view.focus,
            view.selected,
            view.selected_wire,
        );
        let trail = view.trail.len();
        let wire = original.4.unwrap();
        for key in b"iu" {
            assert!(offline.input(*key, &mut view));
        }
        assert_eq!(view.layers[view.selected].id, wire.from.operator);
        assert_ne!(view.focus, original.2);
        assert_eq!(view.trail.len(), trail + 1);
        assert!(offline.cancel(&mut view));
        assert!(
            (
                view.camera,
                view.viewport,
                view.focus,
                view.selected,
                view.selected_wire
            ) == original
        );
        assert_eq!(view.trail.len(), trail);
        assert!(matches!(
            offline.mode,
            DetailMode::CallEvidence { input: 0, .. }
        ));
        assert_eq!(view.document.revision(), 1);
    }

    #[test]
    fn stale_preview_keeps_the_existing_graph_and_can_return_to_evidence() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        assert!(offline.input(b'e', &mut view));
        let target = Port::Input(offline.repair().preview.target);
        view.apply_segment(Segment {
            base_revision: view.document.revision(),
            edits: vec![Edit::SetShape(
                target,
                Some(super::super::Shape::Features(24)),
            )],
        })
        .unwrap();
        let revision = view.document.revision();
        assert!(offline.input(b'y', &mut view));
        assert!(matches!(
            offline.stage,
            Stage::Preview(Draft {
                error: Some(DraftError::Stale),
                ..
            })
        ));
        assert_eq!(view.document.revision(), revision);
        assert_eq!(
            view.document
                .shape(target)
                .and_then(super::super::Shape::last),
            Some(24)
        );
        assert!(offline.cancel(&mut view));
        assert!(matches!(offline.stage, Stage::Evidence));
    }

    #[cfg(unix)]
    #[test]
    fn preview_from_a_leaf_shows_the_candidate_at_the_same_socket_and_restores_the_view() {
        let mut offline = Offline::new(Case::TwinMerge);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        for key in b"eyn" {
            assert!(offline.input(*key, &mut view));
        }
        let current = offline.repair().preview.affected;
        view.zoom(view.indices[&current.to.operator]);
        view.hierarchy = true;
        view.viewport.zoom(1.35, [20.0, 5.0]);
        view.camera.orbit(0.3, 0.1);
        let original = view.bookmark();
        let trail_len = view.trail.len();

        assert!(offline.input(b'e', &mut view));
        assert!(view.flow_layout().is_some());
        view.edit_overlay = offline.graph_overlay(&view);
        for pixels in [false, true] {
            let scene = view.scene(120, 24, pixels, None);
            let mark = scene.edit_mark.unwrap();
            let candidate = scene.candidate.as_ref().unwrap();
            let existing = scene
                .handoffs
                .iter()
                .find(|route| route.wire == current)
                .unwrap();
            assert_eq!(candidate.wire.to, current.to);
            assert_eq!(mark.point, existing.to);
            assert_eq!(candidate.to, existing.to);
            assert_eq!(view.document.source(current.to), Some(current.from));
        }

        view.zoom(view.indices[&current.to.operator]);
        assert!(offline.cancel(&mut view));
        assert!(view.bookmark() == original);
        assert_eq!(view.trail.len(), trail_len);
        assert_eq!(view.document.revision(), 2);
    }

    #[test]
    fn twin_merge_edits_keep_each_input_and_receipt_bound_to_its_operator() {
        let mut offline = Offline::new(Case::TwinMerge);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        let [a, b] = std::array::from_fn(|i| offline.fixture.repairs[i].preview.affected);
        let contracts = |document: &GraphDocument<'_>| {
            [a, b].map(|wire| document.shape(Port::Input(wire.to)).and_then(Shape::last))
        };
        assert_eq!((a.to.slot, b.to.slot), (0, 0));
        assert_ne!(a.to.operator, b.to.operator);
        let initial_wires = view.document.wires().collect::<Vec<_>>();
        assert_eq!(initial_wires.len(), 4);
        for wire in [a, b] {
            let module = view.document.operator(wire.to.operator).unwrap();
            assert_eq!(module.kind, "Merge");
            let side = super::super::Inlet { slot: 1, ..wire.to };
            let source = view.document.source(side).unwrap();
            assert_ne!(wire.from, source);
            assert_eq!(
                view.document.operator(source.operator).unwrap().path,
                "model.residual"
            );
        }
        assert!(offline.input(b'e', &mut view));
        assert_eq!(view.document.revision(), 1);
        assert_eq!(contracts(&view.document), [Some(32), Some(48)]);
        assert!(offline.input(b'y', &mut view));
        assert_eq!(offline.confirmed_target(), Some(a.to));
        assert_eq!(view.document.revision(), 2);
        assert_eq!(contracts(&view.document), [Some(48), Some(48)]);
        assert!(view.document.wires().eq(initial_wires.iter().copied()));

        for key in b"ne" {
            assert!(offline.input(*key, &mut view));
        }
        assert_eq!(view.document.revision(), 2);
        assert!(view.document.wires().eq(initial_wires.iter().copied()));
        let Some(EditOverlay::Draft {
            candidate: Some(rewired),
            ..
        }) = offline.graph_overlay(&view)
        else {
            panic!("missing candidate for merge B");
        };
        assert_eq!(rewired.to, b.to);
        assert_ne!(rewired.from, b.from);
        assert!(offline.input(b'y', &mut view));
        assert_eq!(offline.confirmed_target(), Some(b.to));
        assert_eq!(view.selected_wire, Some(rewired));
        assert_eq!(view.document.revision(), 3);
        assert_eq!(contracts(&view.document), [Some(48), Some(48)]);
        let expected = initial_wires
            .into_iter()
            .map(|wire| if wire == b { rewired } else { wire });
        assert!(view.document.wires().eq(expected.clone()));

        for key in b"xy" {
            assert!(offline.input(*key, &mut view));
        }
        assert!(matches!(offline.stage, Stage::Preview(Draft {
            existing, unavailable: true, error: Some(DraftError::SourceUnavailable), ..
        }) if existing == rewired));
        assert_eq!(view.selected_wire, Some(rewired));
        assert_eq!(view.document.revision(), 3);
        assert_eq!(contracts(&view.document), [Some(48), Some(48)]);
        assert!(view.document.wires().eq(expected));
        assert_eq!(offline.receipts, [None, None]);
        assert!(offline.input(b'x', &mut view));
        assert!(matches!(offline.stage, Stage::Preview(Draft {
            existing, unavailable: false, error: None, ..
        }) if existing == rewired));
        assert!(offline.cancel(&mut view));
        assert_eq!(offline.confirmed_target(), Some(b.to));
    }

    #[test]
    fn axis_comparison_uses_one_scale_and_preserves_unknown_extent() {
        assert_eq!(axis_bars([Some(32), Some(32)], 12), ["├━━━━━━━━━━━━┤"; 2]);
        assert_eq!(
            axis_bars([Some(48), Some(32)], 12),
            ["├━━━━━━━━━━━━┤", "├━━━━━━━━┤"]
        );
        assert_eq!(
            axis_bars([None, Some(32)], 12),
            ["┆┄┄┄┄┄┄┄┄┄┄┄┄?", "├━━━━━━━━━━━━┤"]
        );
    }

    #[test]
    fn incompatible_recheck_preserves_the_last_matching_result() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        for key in [b'e', b'y', b'1'] {
            assert!(offline.input(key, &mut view));
        }
        for key in b"r4" {
            assert!(offline.input(*key, &mut view));
        }
        let frame = offline.render(Locale::English, 80, true, false);
        assert!(frame.contains("Recheck batch mismatch"));
        assert!(frame.contains("changing-batch-01 ≠ different-batch-02"));
        assert!(frame.contains("Passed · 2×48 → …×48"));
        assert_eq!(view.document.revision(), 2);
        assert_eq!(offline.receipts, [Some(RecheckSample::Passing)]);
        assert!(offline.cancel(&mut view));
        assert!(offline.recheck_mismatch.is_none());
        assert!(offline.cancel(&mut view));
        assert!(
            offline
                .render(Locale::English, 80, true, false)
                .contains("✓ Sample passed")
        );
    }

    #[test]
    fn incompatible_rechecks_report_the_expected_and_received_identity() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        for key in b"ey1" {
            assert!(offline.input(*key, &mut view));
        }
        let rejected = |offline: &Offline, revision, reason, expected, received| {
            let error = offline
                .recheck_rejection(&offline.repair().failed, revision)
                .unwrap();
            assert_eq!(error.mismatch.label(Locale::English), reason);
            assert_eq!(error.expected, expected);
            assert_eq!(error.received, received);
        };
        rejected(&offline, 3, "Recheck graph revision mismatch", "v2", "v3");
        let snapshot = offline.repair().receipt.snapshot;
        offline.fixture.repairs[0].failed.snapshot.model.name = "other-model";
        rejected(
            &offline,
            2,
            "Recheck model mismatch",
            "offline-demo/contract-repair",
            "offline-demo/other-model",
        );
        offline.fixture.repairs[0].failed.snapshot.model = snapshot.model;
        offline.fixture.repairs[0].failed.snapshot.variant = "other-variant";
        rejected(
            &offline,
            2,
            "Recheck variant or snapshot mismatch",
            "repair-1/structure-2",
            "other-variant/structure-2",
        );
        offline.fixture.repairs[0].failed.snapshot = snapshot;
        offline.fixture.repairs[0].failed.calls[0]
            .incoming
            .as_mut()
            .unwrap()
            .origin = Origin::Declaration;
        rejected(
            &offline,
            2,
            "Recheck target mismatch",
            "net.head.in[0] #2",
            "model.head.in[?] #2",
        );
        offline.fixture.repairs[0].failed.calls[0]
            .incoming
            .as_mut()
            .unwrap()
            .origin = Origin::Observation;
        offline.fixture.repairs[0].failed.calls[0]
            .incoming
            .as_mut()
            .unwrap()
            .value
            .to
            .slot = 1;
        rejected(
            &offline,
            2,
            "Recheck target mismatch",
            "net.head.in[0] #2",
            "net.head.in[1] #2",
        );
        assert!(offline.input(b'2', &mut view));
        assert!(
            offline
                .render(Locale::English, 100, true, false)
                .contains("✓ Passed")
        );
    }

    #[test]
    fn comparison_preserves_camera_and_keeps_truncated_evidence_explicit() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        view.camera.orbit(1.0, 0.20);
        let camera = view.camera;
        let viewport = view.viewport;
        for key in b"eyr3cc" {
            assert!(offline.input(*key, &mut view));
        }
        let frame = offline.render(Locale::English, 80, true, false);
        assert!(frame.contains("After · ? Coverage incomplete"));
        assert!(frame.contains("Target captured · remaining trace unknown"));
        assert!(view.camera == camera);
        assert!(view.viewport == viewport);
        assert_eq!(view.document.revision(), 2);
        assert!(matches!(
            offline.graph_overlay(&view),
            Some(EditOverlay::Evidence {
                ink: LinkInk::Unknown,
                ..
            })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn comparison_from_a_leaf_marks_the_same_input_and_returns_to_that_view() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        for key in b"eyr3" {
            assert!(offline.input(*key, &mut view));
        }
        let target = offline.repair().receipt.target;
        view.zoom(view.indices[&target.operator]);
        view.hierarchy = true;
        view.viewport.zoom(1.35, [20.0, 5.0]);
        view.camera.orbit(0.3, 0.1);
        let original = view.bookmark();
        let trail_len = view.trail.len();
        let mut points = [[0.0; 2]; 2];

        for ink in [LinkInk::Draft, LinkInk::Unknown] {
            assert!(offline.input(b'c', &mut view));
            assert!(view.flow_layout().is_some());
            assert!(view.camera == original.camera);
            view.edit_overlay = offline.graph_overlay(&view);
            for (backend, pixels) in [false, true].into_iter().enumerate() {
                let scene = view.scene(120, 24, pixels, None);
                let handoff = scene
                    .handoffs
                    .iter()
                    .find(|route| route.wire.to == target)
                    .unwrap();
                let mark = scene.edit_mark.unwrap();
                assert_eq!(mark.point, handoff.to);
                assert!(mark.ink == ink);
                if ink == LinkInk::Draft {
                    points[backend] = mark.point;
                } else {
                    assert_eq!(points[backend], mark.point);
                }
            }
        }
        assert!(offline.input(b'b', &mut view));
        assert!(view.bookmark() == original);
        assert_eq!(view.trail.len(), trail_len);
        assert_eq!(view.document.revision(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn graph_keeps_committed_hit_targets_while_showing_an_alternate_wire() {
        let mut offline = Offline::new(Case::Mismatch);
        offline.preferences = Preferences::default();
        let mut view = offline.view();
        for key in b"eyne" {
            assert!(offline.input(*key, &mut view));
        }
        view.edit_overlay = offline.graph_overlay(&view);
        let baseline = offline.repair().preview.affected;
        let scene = view.scene(100, 30, true, None);
        let proposed = scene.candidate.as_ref().unwrap();
        assert_eq!(proposed.wire.to, baseline.to);
        assert_ne!(proposed.wire.from, baseline.from);
        let committed = scene
            .handoffs
            .iter()
            .find(|route| route.wire == baseline)
            .unwrap();
        assert_eq!(proposed.to, committed.to);
        assert!(
            scene
                .handoffs
                .iter()
                .all(|route| route.wire != proposed.wire)
        );
        assert_eq!(view.document.source(baseline.to), Some(baseline.from));
        assert_eq!(view.document.revision(), 2);
    }
}
