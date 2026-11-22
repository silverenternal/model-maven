//! Offline diagnostic evidence, independent of the model renderer and CLI.
use crate::graph_document::{Inlet, OperatorId, Outlet, Shape, Wire};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Origin {
    Declaration,
    Observation,
    Preview,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evidence<T> {
    pub value: T,
    pub origin: Origin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Model {
    pub project: &'static str,
    pub name: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotId {
    pub model: Model,
    pub variant: &'static str,
    pub snapshot: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CallId {
    pub module: Option<OperatorId>,
    pub alias: &'static str,
    pub invocation: u32,
}

#[derive(Debug)]
pub struct Call {
    pub id: CallId,
    pub input: Option<Evidence<Shape>>,
    pub output: Option<Evidence<Shape>>,
    pub incoming: Option<Evidence<Wire>>,
    pub expected: Option<Evidence<Shape>>,
    pub other_inputs: Vec<ObservedInput>,
}

#[derive(Debug)]
pub struct ObservedInput {
    pub incoming: Wire,
    pub actual: Shape,
    pub declared: Shape,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Coverage {
    Captured,
    Truncated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    FailedAt(CallId),
    LastObserved(CallId),
    Passed,
}

#[derive(Debug)]
pub struct Report {
    pub id: &'static str,
    pub snapshot: SnapshotId,
    pub batch: &'static str,
    pub outcome: Outcome,
    pub calls: Vec<Call>,
    pub source: Option<Evidence<(&'static str, u32)>>,
    pub coverage: Evidence<Coverage>,
}

impl Report {
    pub fn focus(&self) -> Option<&Call> {
        let id = match self.outcome {
            Outcome::FailedAt(id) | Outcome::LastObserved(id) => id,
            Outcome::Passed => return None,
        };
        self.calls.iter().find(|call| call.id == id)
    }
}

#[derive(Debug)]
pub struct Module {
    pub id: OperatorId,
    pub path: &'static str,
    pub aliases: &'static [&'static str],
    pub kind: &'static str,
    pub parent: Option<OperatorId>,
}

#[derive(Debug)]
pub struct Snapshot {
    pub id: SnapshotId,
    pub modules: Vec<Module>,
    pub outputs: Vec<(Outlet, Evidence<Shape>)>,
    /// Explicit fixture declarations; omitted pairs are not assumed absent.
    pub disconnected: Vec<Wire>,
}

impl Snapshot {
    pub fn resolve(&self, report: &Report, call: CallId) -> Option<OperatorId> {
        if self.id != report.snapshot {
            return None;
        }
        if let Some(id) = call.module {
            return self
                .modules
                .iter()
                .any(|module| module.id == id)
                .then_some(id);
        }
        let mut matches = self
            .modules
            .iter()
            .filter(|module| module.path == call.alias || module.aliases.contains(&call.alias));
        let matched = matches.next()?;
        matches.next().is_none().then_some(matched.id)
    }
}

#[derive(Debug)]
pub struct Preview {
    pub report: &'static str,
    pub target: Inlet,
    pub affected: Wire,
    pub change: Adjustment,
    pub weights: Option<Evidence<WeightImpact>>,
}

#[derive(Debug)]
pub enum Adjustment {
    Shape {
        before: Shape,
        after: Evidence<Shape>,
    },
    Source {
        after: Evidence<Outlet>,
        unavailable: Outlet,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum WeightImpact {
    Reset(&'static str),
    Reuse(&'static str),
}

#[derive(Debug)]
pub struct Receipt {
    pub preview_report: &'static str,
    pub target: Inlet,
    pub snapshot: SnapshotId,
    pub graph_revision: u64,
}

#[derive(Debug)]
pub struct Repair {
    pub call: CallId,
    pub preview: Preview,
    pub receipt: Receipt,
    pub passed: Report,
    pub failed: Report,
    pub truncated: Report,
    pub different_batch: Report,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecheckSample {
    Passing,
    Failing,
    Truncated,
    DifferentBatch,
}

impl Repair {
    pub fn recheck(&self, sample: RecheckSample) -> &Report {
        match sample {
            RecheckSample::Passing => &self.passed,
            RecheckSample::Failing => &self.failed,
            RecheckSample::Truncated => &self.truncated,
            RecheckSample::DifferentBatch => &self.different_batch,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Case {
    Mismatch,
    Large,
    TwinMerge,
    CrossedPorts,
    LastObserved,
    SnapshotMissing,
}

#[derive(Debug)]
pub struct Fixture {
    pub source: &'static str,
    pub snapshot: Snapshot,
    pub report: Report,
    pub repairs: Vec<Repair>,
}

impl Fixture {
    pub fn load(case: Case) -> Self {
        let twin = matches!(case, Case::TwinMerge);
        let crossed = matches!(case, Case::CrossedPorts);
        let (head_path, head_alias, head_weight, head_kind, merge_path, merge_alias, merge_weight) =
            if twin {
                (
                    "net.merge_a",
                    "model.merge_a",
                    "net.merge_a.weight",
                    "Merge",
                    "net.merge_b",
                    "model.merge_b",
                    "net.merge_b.weight",
                )
            } else {
                (
                    "net.head",
                    "model.head",
                    "net.head.weight",
                    "Linear",
                    "net.merge",
                    "model.merge",
                    "net.merge.weight",
                )
            };
        let baseline = SnapshotId {
            model: Model {
                project: "offline-demo",
                name: match case {
                    Case::TwinMerge => "twin-merge",
                    Case::CrossedPorts => "crossed-ports",
                    _ => "contract-repair",
                },
            },
            variant: "baseline",
            snapshot: "structure-1",
        };
        let candidate = SnapshotId {
            variant: "repair-1",
            snapshot: "structure-2",
            ..baseline
        };
        let encode = OperatorId(1);
        let head = OperatorId(2);
        let target = Inlet {
            operator: head,
            slot: 0,
        };
        let wire = Wire {
            from: Outlet {
                operator: encode,
                slot: 0,
            },
            to: target,
        };
        let head_call = |invocation| CallId {
            module: (invocation == 1).then_some(head),
            alias: head_alias,
            invocation,
        };
        let actual = |size| Evidence {
            value: Shape::Dimensions(vec![Some(2), Some(size)]),
            origin: Origin::Observation,
        };
        let declaration = |size| Evidence {
            value: Shape::Features(size),
            origin: Origin::Declaration,
        };
        let repair_snapshot = SnapshotId {
            variant: "repair-2",
            snapshot: "structure-3",
            ..baseline
        };
        let merge = OperatorId(3);
        let adapter = OperatorId(4);
        let residual = OperatorId(6);
        let merge_call = CallId {
            module: Some(merge),
            alias: merge_alias,
            invocation: 1,
        };
        let merge_target = Inlet {
            operator: merge,
            slot: 0,
        };
        let merge_wire = Wire {
            from: Outlet {
                operator: encode,
                slot: u32::from(crossed),
            },
            to: merge_target,
        };
        let alternate = Outlet {
            operator: adapter,
            slot: 0,
        };
        let changed_wire = Wire {
            from: alternate,
            to: merge_target,
        };
        let other_inputs = |to: Inlet| {
            let source = if twin {
                Some(Outlet {
                    operator: residual,
                    slot: 0,
                })
            } else if crossed && to.operator == merge {
                Some(wire.from)
            } else {
                None
            };
            source
                .map(|from| ObservedInput {
                    incoming: Wire {
                        from,
                        to: Inlet { slot: 1, ..to },
                    },
                    actual: actual(48).value,
                    declared: declaration(48).value,
                })
                .into_iter()
                .collect()
        };
        let observed_call = |id, size, expected, incoming: Wire, passed: bool| Call {
            id,
            input: Some(actual(size)),
            output: passed.then(|| actual(16)),
            incoming: Some(Evidence {
                value: incoming,
                origin: Origin::Observation,
            }),
            expected: Some(declaration(expected)),
            other_inputs: other_inputs(incoming.to),
        };
        let recheck = |id, snapshot, passed, affected: CallId, calls| Report {
            id,
            snapshot,
            batch: "changing-batch-01",
            outcome: if passed {
                Outcome::Passed
            } else {
                Outcome::FailedAt(affected)
            },
            calls,
            source: (!passed).then_some(Evidence {
                value: ("demo.py", 12),
                origin: Origin::Observation,
            }),
            coverage: Evidence {
                value: Coverage::Captured,
                origin: Origin::Observation,
            },
        };
        let alternate_rechecks = |snapshot, affected, incoming, truncated_id, batch_id| {
            let mut truncated = recheck(
                truncated_id,
                snapshot,
                true,
                affected,
                vec![observed_call(affected, 48, 48, incoming, true)],
            );
            truncated.outcome = Outcome::LastObserved(affected);
            truncated.coverage.value = Coverage::Truncated;
            let mut different_batch = recheck(
                batch_id,
                snapshot,
                true,
                affected,
                vec![observed_call(affected, 48, 48, incoming, true)],
            );
            different_batch.batch = "different-batch-02";
            (truncated, different_batch)
        };
        let exact = matches!(
            case,
            Case::Mismatch | Case::Large | Case::TwinMerge | Case::CrossedPorts
        );
        let repairs = if exact {
            let (head_truncated, head_different_batch) = alternate_rechecks(
                candidate,
                head_call(2),
                wire,
                "recheck-truncated",
                "recheck-other-batch",
            );
            let (merge_truncated, merge_different_batch) = alternate_rechecks(
                repair_snapshot,
                merge_call,
                changed_wire,
                "rewire-truncated",
                "rewire-other-batch",
            );
            vec![
                Repair {
                    call: head_call(2),
                    preview: Preview {
                        report: "failed-48-32",
                        target,
                        affected: wire,
                        change: Adjustment::Shape {
                            before: Shape::Features(32),
                            after: Evidence {
                                value: Shape::Features(48),
                                origin: Origin::Preview,
                            },
                        },
                        weights: Some(Evidence {
                            value: WeightImpact::Reset(head_weight),
                            origin: Origin::Preview,
                        }),
                    },
                    receipt: Receipt {
                        preview_report: "failed-48-32",
                        target,
                        snapshot: candidate,
                        graph_revision: 2,
                    },
                    passed: recheck(
                        "recheck-passed",
                        candidate,
                        true,
                        head_call(2),
                        vec![observed_call(head_call(2), 48, 48, wire, true)],
                    ),
                    failed: recheck(
                        "recheck-failed",
                        candidate,
                        false,
                        head_call(2),
                        vec![observed_call(head_call(2), 64, 48, wire, false)],
                    ),
                    truncated: head_truncated,
                    different_batch: head_different_batch,
                },
                Repair {
                    call: merge_call,
                    preview: Preview {
                        report: "failed-48-32",
                        target: merge_target,
                        affected: merge_wire,
                        change: Adjustment::Source {
                            after: Evidence {
                                value: alternate,
                                origin: Origin::Preview,
                            },
                            unavailable: Outlet {
                                slot: 9,
                                ..alternate
                            },
                        },
                        weights: Some(Evidence {
                            value: WeightImpact::Reuse(merge_weight),
                            origin: Origin::Preview,
                        }),
                    },
                    receipt: Receipt {
                        preview_report: "failed-48-32",
                        target: merge_target,
                        snapshot: repair_snapshot,
                        graph_revision: 3,
                    },
                    passed: recheck(
                        "rewire-passed",
                        repair_snapshot,
                        true,
                        merge_call,
                        vec![
                            observed_call(head_call(2), 48, 48, wire, true),
                            observed_call(merge_call, 48, 48, changed_wire, true),
                        ],
                    ),
                    failed: recheck(
                        "rewire-failed",
                        repair_snapshot,
                        false,
                        merge_call,
                        vec![
                            observed_call(head_call(2), 48, 48, wire, true),
                            observed_call(merge_call, 64, 48, changed_wire, false),
                        ],
                    ),
                    truncated: merge_truncated,
                    different_batch: merge_different_batch,
                },
            ]
        } else {
            Vec::new()
        };
        let mut calls = vec![
            Call {
                id: head_call(1),
                input: Some(actual(32)),
                output: Some(actual(16)),
                incoming: None,
                expected: None,
                other_inputs: Vec::new(),
            },
            Call {
                id: head_call(2),
                input: Some(actual(48)),
                output: None,
                incoming: exact.then_some(Evidence {
                    value: wire,
                    origin: Origin::Observation,
                }),
                expected: exact.then(|| declaration(32)),
                other_inputs: other_inputs(target),
            },
        ];
        if exact {
            // The offline trace includes a second captured contract. It has
            // its own call identity and socket even though both use slot 0.
            calls.push(observed_call(merge_call, 48, 48, merge_wire, true));
        }
        Self {
            source: match case {
                Case::Large => "offline-48-32-large",
                Case::TwinMerge => "offline-48-32-twin-merge",
                Case::CrossedPorts => "offline-48-32-crossed-ports",
                _ => "offline-48-32",
            },
            snapshot: Snapshot {
                id: baseline,
                modules: vec![
                    Module {
                        id: encode,
                        path: "model.encoder",
                        aliases: &[],
                        kind: "Encoder",
                        parent: None,
                    },
                    Module {
                        id: head,
                        path: head_path,
                        aliases: if twin {
                            &["model.merge_a"]
                        } else {
                            &["model.head"]
                        },
                        kind: head_kind,
                        parent: None,
                    },
                    Module {
                        id: merge,
                        path: merge_path,
                        aliases: if twin {
                            &["model.merge_b"]
                        } else {
                            &["model.merge"]
                        },
                        kind: "Merge",
                        parent: None,
                    },
                    Module {
                        id: adapter,
                        path: "net.adapter",
                        aliases: &[],
                        kind: "Adapter",
                        parent: None,
                    },
                ]
                .into_iter()
                .chain(twin.then_some(Module {
                    id: residual,
                    path: "model.residual",
                    aliases: &[],
                    kind: "Residual",
                    parent: None,
                }))
                .collect(),
                outputs: exact
                    .then(|| (alternate, declaration(48)))
                    .into_iter()
                    .collect(),
                disconnected: (twin || crossed)
                    .then_some(Wire {
                        from: alternate,
                        to: merge_target,
                    })
                    .into_iter()
                    .collect(),
            },
            report: Report {
                id: "failed-48-32",
                snapshot: if matches!(case, Case::SnapshotMissing) {
                    SnapshotId {
                        snapshot: "structure-missing",
                        ..baseline
                    }
                } else {
                    baseline
                },
                batch: "changing-batch-01",
                outcome: if exact {
                    Outcome::FailedAt(head_call(2))
                } else {
                    Outcome::LastObserved(head_call(2))
                },
                calls,
                source: Some(Evidence {
                    value: ("demo.py", 12),
                    origin: Origin::Observation,
                }),
                coverage: Evidence {
                    value: if exact {
                        Coverage::Captured
                    } else {
                        Coverage::Truncated
                    },
                    origin: Origin::Observation,
                },
            },
            repairs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_preserves_call_identity_and_evidence_boundaries() {
        let exact = Fixture::load(Case::Mismatch);
        let focus = exact.report.focus().unwrap();
        assert_eq!(focus.id.invocation, 2);
        assert_eq!(
            exact.snapshot.resolve(&exact.report, focus.id),
            Some(OperatorId(2))
        );
        assert_eq!(focus.input.as_ref().unwrap().value.last(), Some(48));
        assert_eq!(focus.expected.as_ref().unwrap().value, Shape::Features(32));
        assert_eq!(focus.incoming.as_ref().unwrap().origin, Origin::Observation);
        assert_eq!(exact.repairs.len(), 2);
        assert_ne!(
            exact.repairs[0].preview.target,
            exact.repairs[1].preview.target
        );
        for repair in &exact.repairs {
            let preview = &repair.preview;
            let receipt = &repair.receipt;
            assert_eq!(preview.report, exact.report.id);
            assert_eq!(receipt.preview_report, preview.report);
            assert_eq!(receipt.target, preview.target);
            assert_ne!(receipt.snapshot, exact.report.snapshot);
            for recheck in [&repair.passed, &repair.failed] {
                assert_eq!(recheck.batch, exact.report.batch);
                assert_eq!(recheck.snapshot, receipt.snapshot);
                assert_ne!(recheck.id, exact.report.id);
            }
        }
        for case in [Case::LastObserved, Case::SnapshotMissing] {
            let fixture = Fixture::load(case);
            let focus = fixture.report.focus().unwrap();
            assert_eq!(fixture.report.outcome, Outcome::LastObserved(focus.id));
            assert!(focus.expected.is_none() && focus.incoming.is_none());
            assert_eq!(fixture.report.coverage.value, Coverage::Truncated);
            assert!(fixture.repairs.is_empty());
            assert_eq!(
                fixture
                    .snapshot
                    .resolve(&fixture.report, focus.id)
                    .is_some(),
                matches!(case, Case::LastObserved),
            );
        }
    }
}
