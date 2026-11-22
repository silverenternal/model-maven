//! A repeatable, grouped graph used by the interactive offline load case.
use super::{Edit, Inlet, Operator, OperatorId, Outlet, Port, Shape, Wire};
use serde_json::json;
use std::borrow::Cow;

pub(super) const GROUPS: u64 = 32;
pub(super) const PER_GROUP: u64 = 32;
pub(super) const DIAGNOSIS_GROUP: OperatorId = OperatorId(5);
pub(super) const SOURCE_GROUP: OperatorId = OperatorId(11_000);
pub(super) const FIRST: u64 = 100;

pub(super) fn append(edits: &mut Vec<Edit<'static>>) {
    let mut seed = 0x5a17_4832_u32;
    let shape = Shape::Features(32);
    for group in 0..GROUPS {
        let parent = OperatorId(FIRST + GROUPS * PER_GROUP + group);
        let path = format!("block{group:02}");
        edits.push(Edit::AddOperator(
            parent,
            Operator {
                path: path.clone(),
                kind: "Block".into(),
                parent: None,
                attributes: Cow::Owned(json!({"stack":3})),
            },
        ));
        for offset in 0..PER_GROUP {
            let id = OperatorId(FIRST + group * PER_GROUP + offset);
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            edits.extend([
                Edit::AddOperator(
                    id,
                    Operator {
                        path: format!("{path}.layer{offset:02}"),
                        kind: "Linear".into(),
                        parent: Some(parent),
                        attributes: Cow::Owned(json!({"parameter_count": 256 + seed % 104_857})),
                    },
                ),
                Edit::AddPort(
                    Port::Output(Outlet {
                        operator: id,
                        slot: 0,
                    }),
                    Some(shape.clone()),
                ),
            ]);
            let mut connect = |source, slot| {
                let to = Inlet { operator: id, slot };
                let from = Outlet {
                    operator: OperatorId(source),
                    slot: 0,
                };
                edits.push(Edit::AddPort(Port::Input(to), Some(shape.clone())));
                edits.push(Edit::Connect(Wire { from, to }));
            };
            if id.0 > FIRST {
                connect(id.0 - 1, 0);
            }
            if offset % 8 == 4 {
                connect(id.0 - 2, 1);
            }
        }
    }
}
