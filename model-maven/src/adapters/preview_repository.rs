//! Port implementation for project-local preview inputs and lifecycle state.

use super::previews::FilePreviewStore;
use crate::ports::{
    PreviewFingerprints, PreviewRepository, PreviewRepositoryError, PreviewTransforms,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
};

impl PreviewRepository for FilePreviewStore {
    fn project_root(&self) -> Result<PathBuf, PreviewRepositoryError> {
        std::env::current_dir()
            .and_then(|path| path.canonicalize())
            .map_err(|error| PreviewRepositoryError::ProjectUnavailable(error.to_string()))
    }

    fn read_transforms(
        &self,
        path: Option<&Path>,
    ) -> Result<PreviewTransforms, PreviewRepositoryError> {
        let Some(path) = path else {
            return Ok(PreviewTransforms {
                steps: Vec::new(),
                sha256: format!("{:x}", Sha256::digest(b"[]")),
            });
        };
        let metadata = fs::symlink_metadata(path)
            .map_err(|error| PreviewRepositoryError::TransformsReadFailed(error.to_string()))?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.len() > 2 * 1024 * 1024
        {
            return Err(PreviewRepositoryError::TransformsPathInvalid(
                "变换配置必须是小于 2 MiB 的普通 JSON 文件。".into(),
            ));
        }
        let bytes = fs::read(path)
            .map_err(|error| PreviewRepositoryError::TransformsReadFailed(error.to_string()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| PreviewRepositoryError::TransformsInvalid(error.to_string()))?;
        let Some(steps) = value.as_array() else {
            return Err(PreviewRepositoryError::TransformsInvalid(
                "变换配置根值必须是 JSON 数组。".into(),
            ));
        };
        Ok(PreviewTransforms {
            steps: steps.clone(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        })
    }

    fn project_fingerprints(
        &self,
        root: &Path,
        source_entrypoint: &str,
        renderer_entrypoint: Option<&str>,
        transforms: &[Value],
    ) -> Result<PreviewFingerprints, PreviewRepositoryError> {
        let project_python_sha256 =
            project_source_digest(root).map_err(PreviewRepositoryError::SourceFingerprintFailed)?;
        let lock_path = root.join(".model-maven/environment.lock.json");
        let environment_lock_sha256 = match fs::read(&lock_path) {
            Ok(bytes) => Some(format!("{:x}", Sha256::digest(&bytes))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(PreviewRepositoryError::EnvironmentReadFailed(
                    error.to_string(),
                ));
            }
        };
        let custom_transform_entrypoints_are_local = transforms
            .iter()
            .filter(|step| step.get("type").and_then(Value::as_str) == Some("custom"))
            .map(|step| {
                step.get("entrypoint")
                    .and_then(Value::as_str)
                    .is_some_and(|entrypoint| entrypoint_is_project_local(root, entrypoint))
            })
            .collect();
        Ok(PreviewFingerprints {
            project_python_sha256,
            environment_lock_sha256,
            source_entrypoint_is_local: entrypoint_is_project_local(root, source_entrypoint),
            renderer_entrypoint_is_local: renderer_entrypoint
                .map(|entrypoint| entrypoint_is_project_local(root, entrypoint)),
            custom_transform_entrypoints_are_local,
        })
    }

    fn begin_preview(&self, root: &Path, signature: &str) -> Result<Value, String> {
        Self::begin(root, signature)
    }

    fn mark_preview_running(
        &self,
        root: &Path,
        request_id: &str,
        generation: u64,
    ) -> Result<bool, String> {
        Self::mark_running(root, request_id, generation)
    }

    fn cancel_preview(&self, root: &Path, request_id: &str, generation: u64) -> Result<(), String> {
        Self::cancel(root, request_id, generation).map(|_| ())
    }

    fn fail_preview(
        &self,
        root: &Path,
        request_id: &str,
        generation: u64,
        message: &str,
    ) -> Result<(), String> {
        Self::fail(root, request_id, generation, message).map(|_| ())
    }

    fn complete_preview(
        &self,
        root: &Path,
        request_id: &str,
        generation: u64,
        signature: &str,
    ) -> Result<Value, String> {
        Self::complete(root, request_id, generation, signature)
    }

    fn current_preview(&self, root: &Path) -> Result<Value, String> {
        Self::current(root)
    }
}

fn entrypoint_is_project_local(root: &Path, value: &str) -> bool {
    let module = value
        .split_once(':')
        .map(|(module, _)| module)
        .unwrap_or_default();
    let path = root.join(module.replace('.', "/"));
    path.with_extension("py").is_file() || path.join("__init__.py").is_file()
}

fn project_source_digest(root: &Path) -> Result<String, String> {
    fn visit(root: &Path, directory: &Path, digest: &mut Sha256) -> Result<(), String> {
        let mut entries = fs::read_dir(directory)
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if matches!(
                name.as_ref(),
                ".git" | ".venv" | "venv" | "target" | ".model-maven" | "__pycache__"
            ) {
                continue;
            }
            let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
            if metadata.file_type().is_symlink() {
                continue;
            }
            if metadata.is_dir() {
                visit(root, &path, digest)?;
            } else if metadata.is_file()
                && path.extension().is_some_and(|extension| extension == "py")
            {
                let relative = path.strip_prefix(root).map_err(|error| error.to_string())?;
                digest.update(relative.to_string_lossy().as_bytes());
                digest.update([0]);
                let mut file = File::open(&path).map_err(|error| error.to_string())?;
                std::io::copy(&mut file, digest).map_err(|error| error.to_string())?;
                digest.update([0]);
            }
        }
        Ok(())
    }
    let mut digest = Sha256::new();
    visit(root, root, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}
