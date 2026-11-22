use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectConfigUpdate {
    pub path: PathBuf,
    pub revision: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectConfigUpdateError {
    Conflict {
        expected_revision: String,
        current_revision: String,
    },
    Failed(String),
}

impl ProjectConfigUpdateError {
    pub fn message(&self) -> String {
        match self {
            Self::Conflict {
                expected_revision,
                current_revision,
            } => format!(
                "项目配置已变化；基准 revision={expected_revision}，当前 revision={current_revision}。"
            ),
            Self::Failed(message) => message.clone(),
        }
    }
}

/// Stores the project's human-selected default Python model and batch entrypoints.
pub trait ProjectConfigStore {
    fn load(&self, root: &Path) -> Result<Option<Value>, String>;
    fn create(&self, root: &Path, config: &Value) -> Result<PathBuf, String>;
    fn update(&self, root: &Path, expected: &Value, config: &Value) -> Result<PathBuf, String>;

    fn update_versioned(
        &self,
        root: &Path,
        expected_revision: &str,
        config: &Value,
    ) -> Result<ProjectConfigUpdate, ProjectConfigUpdateError> {
        let current = self
            .load(root)
            .map_err(ProjectConfigUpdateError::Failed)?
            .ok_or_else(|| ProjectConfigUpdateError::Failed("项目配置不存在。".into()))?;
        let current_revision = crate::domain::input::content_revision(&current)
            .map_err(ProjectConfigUpdateError::Failed)?;
        if current_revision != expected_revision {
            return Err(ProjectConfigUpdateError::Conflict {
                expected_revision: expected_revision.into(),
                current_revision,
            });
        }
        let path = self
            .update(root, &current, config)
            .map_err(ProjectConfigUpdateError::Failed)?;
        let revision = crate::domain::input::content_revision(config)
            .map_err(ProjectConfigUpdateError::Failed)?;
        Ok(ProjectConfigUpdate { path, revision })
    }

    /// Returns previous saved snapshots, newest first. Implementations that do not
    /// preserve history may keep the default empty result.
    fn history(&self, _root: &Path) -> Result<Vec<Value>, String> {
        Ok(Vec::new())
    }
}
