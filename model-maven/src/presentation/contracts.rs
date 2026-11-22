use serde_json::Value;
use std::collections::BTreeMap;

/// Stable identifier for a terminal page. IDs are strings so extensions can add pages
/// without taking a dependency on a closed enum.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PageId(pub String);

impl PageId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NavigationResult {
    Stay,
    Open(PageId),
    RunAction(String),
    Back,
    Quit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Availability {
    Available,
    Conditional(String),
    Unsupported(String),
    Unknown(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProgressState {
    NotStarted,
    Running,
    Completed,
    Failed,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CancellationState {
    Supported,
    Unsupported,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskProgress {
    pub state: ProgressState,
    pub completed: Option<u64>,
    pub total: Option<u64>,
    pub message: Option<String>,
    pub cancellation: CancellationState,
}

impl TaskProgress {
    /// Use this when a backend has not supplied an event or an authoritative run state.
    pub fn unknown() -> Self {
        Self {
            state: ProgressState::Unknown,
            completed: None,
            total: None,
            message: None,
            cancellation: CancellationState::Unknown,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionDescriptor {
    pub id: String,
    pub label: String,
    pub availability: Availability,
    pub confirmation: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldDescriptor {
    pub path: String,
    pub label: String,
    pub description: Option<String>,
    pub required: bool,
    pub current_value: Option<Value>,
    pub value_source: FieldValueSource,
    pub default: Option<Value>,
    pub schema: Value,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldValueSource {
    Current,
    UserInput,
    Draft,
    Default,
    Unset,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PageDescriptor {
    pub id: PageId,
    pub title: String,
    pub summary: String,
    pub actions: Vec<ActionDescriptor>,
}

#[derive(Default)]
pub struct PageRegistry {
    pages: BTreeMap<PageId, PageDescriptor>,
}

impl PageRegistry {
    pub fn register(&mut self, page: PageDescriptor) -> Result<(), String> {
        if self.pages.contains_key(&page.id) {
            return Err(format!("页面 ID 已注册：{}", page.id.0));
        }
        self.pages.insert(page.id.clone(), page);
        Ok(())
    }

    pub fn get(&self, id: &PageId) -> Option<&PageDescriptor> {
        self.pages.get(id)
    }

    pub fn pages(&self) -> impl Iterator<Item = &PageDescriptor> {
        self.pages.values()
    }
}

impl ActionDescriptor {
    pub fn available(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            availability: Availability::Available,
            confirmation: None,
        }
    }
}

impl FieldDescriptor {
    pub fn from_schema(
        path: impl Into<String>,
        schema: Value,
        required: bool,
        current_value: Option<Value>,
    ) -> Self {
        let path = path.into();
        let value_source = if current_value.is_some() {
            FieldValueSource::Current
        } else if schema.get("default").is_some() {
            FieldValueSource::Default
        } else {
            FieldValueSource::Unset
        };
        Self {
            label: path.clone(),
            path,
            description: schema["description"].as_str().map(str::to_owned),
            required,
            current_value,
            value_source,
            default: schema.get("default").cloned(),
            schema,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(id: &str) -> PageDescriptor {
        PageDescriptor {
            id: PageId::new(id),
            title: id.into(),
            summary: String::new(),
            actions: Vec::new(),
        }
    }

    #[test]
    fn page_registry_rejects_duplicate_ids_and_resolves_registered_pages() {
        let mut registry = PageRegistry::default();
        registry.register(page("home")).unwrap();
        assert_eq!(registry.pages().count(), 1);
        assert_eq!(registry.get(&PageId::new("home")).unwrap().title, "home");
        assert!(registry.register(page("home")).is_err());
    }

    #[test]
    fn missing_backend_events_are_reported_as_unknown() {
        let progress = TaskProgress::unknown();
        assert_eq!(progress.state, ProgressState::Unknown);
        assert_eq!(progress.cancellation, CancellationState::Unknown);
        assert_eq!(progress.completed, None);
        assert_eq!(progress.total, None);
    }
}
