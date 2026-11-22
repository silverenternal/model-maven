//! Semantic graph shared by an editor and its preview. Display slices are not edit targets.
use serde_json::Value;
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    fmt,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct OperatorId(pub u64);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Inlet {
    pub operator: OperatorId,
    pub slot: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Outlet {
    pub operator: OperatorId,
    pub slot: u32,
}

/// One producer tensor can feed several independent input sockets.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Wire {
    pub from: Outlet,
    pub to: Inlet,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Port {
    Input(Inlet),
    Output(Outlet),
}

impl Port {
    pub fn operator(self) -> OperatorId {
        match self {
            Self::Input(port) => port.operator,
            Self::Output(port) => port.operator,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Shape {
    Dimensions(Vec<Option<u64>>),
    Features(u64),
}

impl Shape {
    pub fn parse(value: &Value) -> Option<Self> {
        let dimensions = value.as_array()?;
        if dimensions.len() > 16
            || dimensions
                .iter()
                .any(|axis| !axis.is_null() && axis.as_u64().is_none() && axis.as_i64() != Some(-1))
        {
            return None;
        }
        Some(Self::Dimensions(
            dimensions.iter().map(Value::as_u64).collect(),
        ))
    }

    pub fn last(&self) -> Option<u64> {
        match self {
            Self::Dimensions(axes) => axes.last().copied().flatten(),
            Self::Features(size) => Some(*size),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Dimensions(axes) if axes.is_empty() => "[]".to_owned(),
            Self::Dimensions(axes) => axes
                .iter()
                .map(|axis| axis.map_or_else(|| "?".into(), |size| size.to_string()))
                .collect::<Vec<_>>()
                .join("×"),
            Self::Features(size) => format!("…×{size}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Operator<'a> {
    pub path: String,
    pub kind: String,
    pub parent: Option<OperatorId>,
    pub attributes: Cow<'a, Value>,
}

#[derive(Debug)]
pub enum Edit<'a> {
    AddOperator(OperatorId, Operator<'a>),
    RemoveOperator(OperatorId),
    MoveOperator {
        id: OperatorId,
        parent: Option<OperatorId>,
        path: String,
    },
    SetKind(OperatorId, String),
    SetAttribute(OperatorId, String, Option<Value>),
    AddPort(Port, Option<Shape>),
    SetShape(Port, Option<Shape>),
    RemovePort(Port),
    Connect(Wire),
    Disconnect(Wire),
}

#[derive(Debug)]
pub struct Segment<'a> {
    pub base_revision: u64,
    /// Changes are staged together. An added operator or port can be used by a
    /// later edit in this segment. Reference validity is checked at commit.
    pub edits: Vec<Edit<'a>>,
}

#[derive(Debug, Default)]
pub struct ChangeSet {
    pub revision: u64,
    pub operators: BTreeSet<OperatorId>,
    pub attributes: BTreeSet<(OperatorId, String)>,
    pub ports: BTreeSet<Port>,
    pub inlets: BTreeSet<Inlet>,
    pub topology_changed: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum EditError {
    Stale { expected: u64, actual: u64 },
    RevisionOverflow,
    OperatorAlreadyUsed(OperatorId),
    MissingOperator(OperatorId),
    InvalidParent(OperatorId),
    PortAlreadyExists(Port),
    MissingPort(Port),
    OccupiedInlet(Inlet),
    MissingWire(Wire),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for EditError {}

#[derive(Default)]
pub struct GraphDocument<'a> {
    revision: u64,
    operators: BTreeMap<OperatorId, Operator<'a>>,
    retired: BTreeSet<OperatorId>,
    ports: BTreeMap<Port, Option<Shape>>,
    retired_ports: BTreeSet<Port>,
    wiring: BTreeMap<Inlet, Outlet>,
    subscribers: BTreeMap<Outlet, BTreeSet<Inlet>>,
}

impl<'a> GraphDocument<'a> {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn operators(&self) -> impl Iterator<Item = (OperatorId, &Operator<'a>)> {
        self.operators.iter().map(|(&id, operator)| (id, operator))
    }

    pub fn operator(&self, id: OperatorId) -> Option<&Operator<'a>> {
        self.operators.get(&id)
    }

    pub fn shape(&self, port: Port) -> Option<&Shape> {
        self.ports.get(&port).and_then(Option::as_ref)
    }

    pub fn ports_for(&self, operator: OperatorId) -> impl Iterator<Item = (Port, Option<&Shape>)> {
        let inputs = |slot| Port::Input(Inlet { operator, slot });
        let outputs = |slot| Port::Output(Outlet { operator, slot });
        self.ports
            .range(inputs(0)..=inputs(u32::MAX))
            .chain(self.ports.range(outputs(0)..=outputs(u32::MAX)))
            .map(|(&port, shape)| (port, shape.as_ref()))
    }

    pub fn wires(&self) -> impl Iterator<Item = Wire> + '_ {
        self.wiring.iter().map(|(&to, &from)| Wire { from, to })
    }

    pub fn source(&self, to: Inlet) -> Option<Outlet> {
        self.wiring.get(&to).copied()
    }

    pub fn consumers(&self, from: Outlet) -> impl Iterator<Item = Inlet> + '_ {
        self.subscribers.get(&from).into_iter().flatten().copied()
    }

    pub fn apply(&mut self, segment: Segment<'a>) -> Result<ChangeSet, EditError> {
        if self.revision != segment.base_revision {
            return Err(EditError::Stale {
                expected: self.revision,
                actual: segment.base_revision,
            });
        }
        let mut change = ChangeSet {
            revision: self.revision,
            ..ChangeSet::default()
        };
        if segment.edits.is_empty() {
            return Ok(change);
        }
        change.revision = self
            .revision
            .checked_add(1)
            .ok_or(EditError::RevisionOverflow)?;
        let mut operators = Overlay::new(&self.operators);
        let mut ports = Overlay::new(&self.ports);
        let mut wires = Overlay::new(&self.wiring);
        for edit in segment.edits {
            match edit {
                Edit::AddOperator(id, operator) => {
                    if self.retired.contains(&id) || operators.contains_key(&id) {
                        return Err(EditError::OperatorAlreadyUsed(id));
                    }
                    operators.put(id, operator);
                    change.topology_changed = true;
                }
                Edit::RemoveOperator(id) => {
                    if operators.get(&id).is_none() {
                        return Err(EditError::MissingOperator(id));
                    }
                    operators.remove(id);
                    change.topology_changed = true;
                }
                Edit::MoveOperator { id, parent, path } => {
                    let operator = operators
                        .get_mut(id)
                        .ok_or(EditError::MissingOperator(id))?;
                    change.topology_changed |= operator.parent != parent;
                    operator.parent = parent;
                    operator.path = path;
                }
                Edit::SetKind(id, kind) => {
                    operators
                        .get_mut(id)
                        .ok_or(EditError::MissingOperator(id))?
                        .kind = kind;
                }
                Edit::SetAttribute(id, name, value) => {
                    let operator = operators
                        .get_mut(id)
                        .ok_or(EditError::MissingOperator(id))?;
                    let attrs = operator.attributes.to_mut();
                    if !attrs.is_object() {
                        *attrs = Value::Object(Default::default());
                    }
                    let attrs = attrs.as_object_mut().unwrap();
                    if let Some(value) = value {
                        attrs.insert(name.clone(), value);
                    } else {
                        attrs.remove(&name);
                    }
                    change.attributes.insert((id, name));
                }
                Edit::AddPort(port, shape) => {
                    if self.retired_ports.contains(&port) || ports.contains_key(&port) {
                        return Err(EditError::PortAlreadyExists(port));
                    }
                    ports.put(port, shape);
                }
                Edit::SetShape(port, shape) => {
                    if ports.get(&port).is_none() {
                        return Err(EditError::MissingPort(port));
                    }
                    ports.put(port, shape);
                }
                Edit::RemovePort(port) => {
                    if ports.get(&port).is_none() {
                        return Err(EditError::MissingPort(port));
                    }
                    ports.remove(port);
                }
                Edit::Connect(wire) => {
                    if wires.get(&wire.to).is_some() {
                        return Err(EditError::OccupiedInlet(wire.to));
                    }
                    wires.put(wire.to, wire.from);
                    change.topology_changed = true;
                }
                Edit::Disconnect(wire) => {
                    if wires.get(&wire.to) != Some(&wire.from) {
                        return Err(EditError::MissingWire(wire));
                    }
                    wires.remove(wire.to);
                    change.topology_changed = true;
                }
            }
        }
        for (&id, revised) in &operators.pending {
            if revised.is_some() {
                let mut parent = revised.as_ref().and_then(|operator| operator.parent);
                let mut visited = BTreeSet::from([id]);
                while let Some(ancestor) = parent {
                    if !visited.insert(ancestor) {
                        return Err(EditError::InvalidParent(id));
                    }
                    parent = operators
                        .get(&ancestor)
                        .ok_or(EditError::MissingOperator(ancestor))?
                        .parent;
                }
            } else {
                // Removing an operator requires removing its children or keeping
                // their edits in a separate surviving operator.
                if operators
                    .iter()
                    .any(|(_, operator)| operator.parent == Some(id))
                {
                    return Err(EditError::InvalidParent(id));
                }
                if let Some(port) = ports
                    .iter()
                    .map(|(&port, _)| port)
                    .find(|port| port.operator() == id)
                {
                    return Err(EditError::MissingOperator(port.operator()));
                }
            }
        }
        for (&port, revised) in &ports.pending {
            if revised.is_some() && operators.get(&port.operator()).is_none() {
                return Err(EditError::MissingOperator(port.operator()));
            }
            if revised.is_none()
                && wires
                    .iter()
                    .any(|(&to, &from)| port == Port::Input(to) || port == Port::Output(from))
            {
                return Err(EditError::MissingPort(port));
            }
        }
        for (&to, &from) in wires
            .pending
            .iter()
            .filter_map(|(to, from)| Some((to, from.as_ref()?)))
        {
            for port in [Port::Input(to), Port::Output(from)] {
                if ports.get(&port).is_none() {
                    return Err(EditError::MissingPort(port));
                }
            }
        }
        change.operators.extend(operators.pending.keys().copied());
        change.ports.extend(ports.pending.keys().copied());
        change.inlets.extend(wires.pending.keys().copied());
        for &port in &change.ports {
            match port {
                Port::Input(to) if wires.get(&to).is_some() => {
                    change.inlets.insert(to);
                }
                Port::Output(from) => change.inlets.extend(self.consumers(from)),
                Port::Input(_) => {}
            }
        }
        for (&id, revised) in &operators.pending {
            if revised.is_none() {
                self.retired.insert(id);
            }
        }
        self.retired_ports.extend(
            ports
                .pending
                .iter()
                .filter_map(|(&port, value)| value.is_none().then_some(port)),
        );
        let changes = (operators.pending, ports.pending, wires.pending);
        for (&to, &source) in &changes.2 {
            if let Some(old) = self.wiring.get(&to) {
                let subscribers = self.subscribers.get_mut(old).unwrap();
                subscribers.remove(&to);
                if subscribers.is_empty() {
                    self.subscribers.remove(old);
                }
            }
            if let Some(source) = source {
                self.subscribers.entry(source).or_default().insert(to);
            }
        }
        commit(&mut self.operators, changes.0);
        commit(&mut self.ports, changes.1);
        commit(&mut self.wiring, changes.2);
        self.revision = change.revision;
        Ok(change)
    }
}

struct Overlay<'a, K, V> {
    original: &'a BTreeMap<K, V>,
    pending: BTreeMap<K, Option<V>>,
}

impl<'a, K: Ord, V> Overlay<'a, K, V> {
    fn new(original: &'a BTreeMap<K, V>) -> Self {
        Self {
            original,
            pending: BTreeMap::new(),
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.pending
            .get(key)
            .map_or_else(|| self.original.get(key), Option::as_ref)
    }

    fn contains_key(&self, key: &K) -> bool {
        self.pending.contains_key(key) || self.original.contains_key(key)
    }

    fn get_mut(&mut self, key: K) -> Option<&mut V>
    where
        V: Clone,
    {
        self.pending
            .entry(key)
            .or_insert_with_key(|key| self.original.get(key).cloned())
            .as_mut()
    }

    fn put(&mut self, key: K, value: V) {
        self.pending.insert(key, Some(value));
    }

    fn remove(&mut self, key: K) {
        self.pending.insert(key, None);
    }

    fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.original
            .iter()
            .filter(|(key, _)| !self.pending.contains_key(key))
            .chain(
                self.pending
                    .iter()
                    .filter_map(|(key, value)| Some((key, value.as_ref()?))),
            )
    }
}

fn commit<K: Ord, V>(destination: &mut BTreeMap<K, V>, pending: BTreeMap<K, Option<V>>) {
    for (key, value) in pending {
        if let Some(value) = value {
            destination.insert(key, value);
        } else {
            destination.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn segments_publish_atomically_and_track_the_output_tensor_fanout() {
        let [producer, a, b] = [OperatorId(1), OperatorId(2), OperatorId(3)];
        let from = Outlet {
            operator: producer,
            slot: 0,
        };
        let [a, b] = [a, b].map(|operator| Inlet { operator, slot: 0 });
        let mut doc = GraphDocument::default();
        let mut edits = Vec::new();
        for id in [producer, a.operator, b.operator] {
            edits.push(Edit::AddOperator(
                id,
                Operator {
                    path: id.0.to_string(),
                    kind: "Linear".into(),
                    parent: None,
                    attributes: Cow::Owned(json!({})),
                },
            ));
        }
        edits.extend([
            Edit::AddPort(Port::Output(from), Some(Shape::Features(32))),
            Edit::AddPort(Port::Input(a), None),
            Edit::AddPort(Port::Input(b), None),
            Edit::Connect(Wire { from, to: a }),
            Edit::Connect(Wire { from, to: b }),
        ]);
        doc.apply(Segment {
            base_revision: 0,
            edits,
        })
        .unwrap();

        let update = doc
            .apply(Segment {
                base_revision: 1,
                edits: vec![Edit::SetShape(
                    Port::Output(from),
                    Some(Shape::Features(48)),
                )],
            })
            .unwrap();
        assert_eq!(update.inlets, BTreeSet::from([a, b]));
        assert!(!update.topology_changed);
        let revision = update.revision;
        let rejected = doc.apply(Segment {
            base_revision: revision,
            edits: vec![
                Edit::SetShape(Port::Output(from), Some(Shape::Features(64))),
                Edit::RemovePort(Port::Input(b)),
            ],
        });
        assert_eq!(
            rejected.unwrap_err(),
            EditError::MissingPort(Port::Input(b))
        );
        assert_eq!(doc.revision(), revision);
        assert_eq!(doc.shape(Port::Output(from)), Some(&Shape::Features(48)));
        assert_eq!(doc.consumers(from).collect::<Vec<_>>(), [a, b]);
        assert_eq!(
            doc.apply(Segment {
                base_revision: 1,
                edits: vec![Edit::Disconnect(Wire { from, to: a })]
            })
            .unwrap_err(),
            EditError::Stale {
                expected: revision,
                actual: 1
            }
        );

        doc.apply(Segment {
            base_revision: revision,
            edits: vec![
                Edit::Disconnect(Wire { from, to: b }),
                Edit::RemovePort(Port::Input(b)),
            ],
        })
        .unwrap();
        assert_eq!(doc.consumers(from).collect::<Vec<_>>(), [a]);
        assert_eq!(
            doc.apply(Segment {
                base_revision: doc.revision(),
                edits: vec![Edit::AddPort(Port::Input(b), None)]
            })
            .unwrap_err(),
            EditError::PortAlreadyExists(Port::Input(b))
        );
    }

    #[test]
    fn rejects_containment_cycles_even_when_entered_from_another_operator() {
        let mut doc = GraphDocument::default();
        let edits = [(1, 2), (2, 3), (3, 2)]
            .into_iter()
            .map(|(id, parent)| {
                Edit::AddOperator(
                    OperatorId(id),
                    Operator {
                        path: id.to_string(),
                        kind: "Module".into(),
                        parent: Some(OperatorId(parent)),
                        attributes: Cow::Owned(json!({})),
                    },
                )
            })
            .collect();
        assert!(matches!(
            doc.apply(Segment {
                base_revision: 0,
                edits
            }),
            Err(EditError::InvalidParent(_))
        ));
        assert_eq!(doc.operators().count(), 0);
    }
}
