use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

const SESSION_FILE: &str = "ui-session.json";
const FORM_DRAFT_FILE: &str = "ui-form-drafts.json";
const PAGE_DRAFT_FILE: &str = "ui-page-session.json";
const MAX_SESSION_BYTES: u64 = 64 * 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OnboardingDraft {
    pub schema_version: u32,
    pub project_root: String,
    pub model: String,
    pub batch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FormDraft {
    pub schema_revision: String,
    pub values: Map<String, Value>,
    pub stale: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PageDraft {
    pub schema_version: u32,
    pub project_root: String,
    pub page_id: String,
    pub config_revision: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FormDraftDocument {
    schema_version: u32,
    project_root: String,
    drafts: BTreeMap<String, StoredFormDraft>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StoredFormDraft {
    schema_revision: String,
    values: Map<String, Value>,
}

impl OnboardingDraft {
    pub fn new(
        root: &Path,
        model: &str,
        batch: Option<&str>,
        config_revision: Option<String>,
    ) -> Result<Self, String> {
        let project_root = root
            .canonicalize()
            .map_err(|error| format!("无法固定会话项目路径：{error}"))?
            .to_string_lossy()
            .into_owned();
        let draft = Self {
            schema_version: 1,
            project_root,
            model: model.to_owned(),
            batch: batch.map(str::to_owned),
            config_revision,
            source_fingerprint: source_fingerprint(root, model, batch),
        };
        draft.validate(root)?;
        Ok(draft)
    }

    fn validate(&self, root: &Path) -> Result<(), String> {
        let canonical_root = root
            .canonicalize()
            .map_err(|error| format!("无法读取会话项目路径：{error}"))?;
        if self.schema_version != 1
            || self.project_root != canonical_root.to_string_lossy()
            || self.model.trim().is_empty()
            || self.model.len() > 4096
            || self.batch.as_ref().is_some_and(|batch| batch.len() > 4096)
            || self.config_revision.as_ref().is_some_and(|revision| {
                revision.len() != 64 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            || self.source_fingerprint.as_ref().is_some_and(|fingerprint| {
                fingerprint.len() != 64 || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        {
            return Err("向导会话版本、项目路径或入口内容无效。".into());
        }
        Ok(())
    }
}

/// Fingerprint a Python entrypoint and its statically discoverable local imports without
/// importing or executing project code. External modules are intentionally unverifiable and
/// require an explicit recheck on restore.
pub fn source_fingerprint(root: &Path, model: &str, batch: Option<&str>) -> Option<String> {
    let mut digest = Sha256::new();
    for entrypoint in std::iter::once(model).chain(batch) {
        digest.update(entrypoint.as_bytes());
        digest.update([0]);
        digest.update(module_source_fingerprint(root, entrypoint)?.as_bytes());
        digest.update([0]);
    }
    Some(format!("{:x}", digest.finalize()))
}

fn module_source_fingerprint(root: &Path, entrypoint: &str) -> Option<String> {
    let (module, _) = entrypoint.split_once(':')?;
    let root = root.canonicalize().ok()?;
    let entrypoint_path = resolve_local_module(&root, module)?;
    let mut pending = vec![entrypoint_path];
    let mut sources = BTreeMap::<PathBuf, Vec<u8>>::new();
    let mut total_bytes = 0_u64;
    while let Some(source) = pending.pop() {
        if sources.contains_key(&source) {
            continue;
        }
        if sources.len() >= 4096 {
            return None;
        }
        let bytes = fs::read(&source).ok()?;
        total_bytes = total_bytes.checked_add(bytes.len() as u64)?;
        if total_bytes > 64 * 1024 * 1024 {
            return None;
        }
        let text = std::str::from_utf8(&bytes).ok()?;
        if has_unresolved_dynamic_imports(text) {
            return None;
        }
        for imported in local_imports(&root, &source, text)? {
            if !sources.contains_key(&imported) {
                pending.push(imported);
            }
        }
        for initializer in package_initializers(&root, &source)? {
            if !sources.contains_key(&initializer) {
                pending.push(initializer);
            }
        }
        sources.insert(source, bytes);
    }
    let mut digest = Sha256::new();
    for (source, bytes) in sources {
        digest.update(
            source
                .strip_prefix(&root)
                .ok()?
                .to_string_lossy()
                .as_bytes(),
        );
        digest.update([0]);
        digest.update(bytes);
        digest.update([0]);
    }
    Some(format!("{:x}", digest.finalize()))
}

fn valid_module_name(module: &str) -> bool {
    !module.is_empty()
        && module.split('.').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|character| character == '_' || character.is_ascii_alphanumeric())
        })
}

fn resolve_local_module(root: &Path, module: &str) -> Option<PathBuf> {
    if !valid_module_name(module) {
        return None;
    }
    let relative = module.split('.').collect::<PathBuf>();
    let candidates = [
        root.join(relative.with_extension("py")),
        root.join(module.split('.').collect::<PathBuf>())
            .join("__init__.py"),
    ];
    candidates
        .iter()
        .find(|candidate| candidate.is_file())?
        .canonicalize()
        .ok()
        .filter(|candidate| candidate.starts_with(root))
}

fn package_initializers(root: &Path, source: &Path) -> Option<Vec<PathBuf>> {
    let relative = source.strip_prefix(root).ok()?;
    let mut directory = root.to_path_buf();
    let mut initializers = Vec::new();
    for component in relative.parent()?.components() {
        directory.push(component.as_os_str());
        let initializer = directory.join("__init__.py");
        if initializer.is_file() {
            let initializer = initializer.canonicalize().ok()?;
            if initializer.starts_with(root) {
                initializers.push(initializer);
            }
        }
    }
    Some(initializers)
}

fn has_unresolved_dynamic_imports(source: &str) -> bool {
    [
        "import_module",
        "__import__",
        "sys.path",
        "runpy.run_module",
        "runpy.run_path",
        "spec_from_file_location",
        "module_from_spec",
        "exec_module",
    ]
    .iter()
    .any(|marker| source.contains(marker))
}

fn local_imports(root: &Path, source_path: &Path, source: &str) -> Option<Vec<PathBuf>> {
    let mut modules = BTreeSet::new();
    let mut pending = String::new();
    let mut nesting = 0_i32;
    for line in source.lines() {
        let line = line.split('#').next()?.trim();
        if line.is_empty() {
            continue;
        }
        if pending.is_empty() && !line.starts_with("import ") && !line.starts_with("from ") {
            continue;
        }
        if !pending.is_empty() {
            pending.push(' ');
        }
        pending.push_str(line.trim_end_matches('\\').trim());
        nesting += line.chars().filter(|character| *character == '(').count() as i32;
        nesting -= line.chars().filter(|character| *character == ')').count() as i32;
        if nesting > 0 || line.ends_with('\\') {
            continue;
        }
        for statement in pending.split(';') {
            collect_import_modules(root, source_path, statement.trim(), &mut modules);
        }
        pending.clear();
        nesting = 0;
    }
    if !pending.is_empty() {
        return None;
    }
    let mut files = BTreeSet::new();
    for module in modules {
        if let Some(path) = resolve_local_module(root, &module) {
            files.insert(path);
        }
    }
    Some(files.into_iter().collect())
}

fn collect_import_modules(
    root: &Path,
    source_path: &Path,
    statement: &str,
    modules: &mut BTreeSet<String>,
) {
    if let Some(imports) = statement.strip_prefix("import ") {
        for imported in imports.split(',') {
            let module = imported.split_whitespace().next().unwrap_or_default();
            if valid_module_name(module) {
                modules.insert(module.to_owned());
            }
        }
        return;
    }
    let Some(rest) = statement.strip_prefix("from ") else {
        return;
    };
    let Some((specifier, imported_names)) = rest.split_once(" import ") else {
        return;
    };
    let level = specifier
        .chars()
        .take_while(|character| *character == '.')
        .count();
    let module_tail = &specifier[level..];
    if !module_tail.is_empty() && !valid_module_name(module_tail) {
        return;
    }
    let mut base = if level == 0 {
        Vec::new()
    } else {
        module_package(root, source_path).unwrap_or_default()
    };
    if level > 0 {
        for _ in 1..level {
            if base.pop().is_none() {
                return;
            }
        }
    }
    if !module_tail.is_empty() {
        base.extend(module_tail.split('.').map(str::to_owned));
    }
    if !base.is_empty() {
        modules.insert(base.join("."));
    }
    let prefix = base.join(".");
    for imported in imported_names
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
    {
        let name = imported.split_whitespace().next().unwrap_or_default();
        if name == "*" || !valid_module_name(name) {
            continue;
        }
        let child = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}.{name}")
        };
        modules.insert(child);
    }
}

fn module_package(root: &Path, source_path: &Path) -> Option<Vec<String>> {
    let relative = source_path.strip_prefix(root).ok()?;
    let components = relative
        .parent()?
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if components
        .iter()
        .any(|component| !valid_module_name(component))
    {
        return None;
    }
    Some(components)
}

pub fn source_is_stale(root: &Path, draft: &OnboardingDraft) -> bool {
    match (
        draft.source_fingerprint.as_deref(),
        source_fingerprint(root, &draft.model, draft.batch.as_deref()),
    ) {
        (Some(saved), Some(current)) => saved != current,
        // If the local source cannot be fingerprinted (for example, an installed package),
        // don't silently assume it is unchanged across process restarts.
        _ => true,
    }
}

fn session_dir(root: &Path, create: bool) -> Result<PathBuf, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("无法读取项目目录：{error}"))?;
    let directory = root.join(".model-maven");
    match fs::symlink_metadata(&directory) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            Err("拒绝在非普通目录 .model-maven 中保存会话。".into())
        }
        Ok(_) => Ok(directory),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && create => {
            fs::create_dir(&directory).map_err(|error| format!("无法创建会话目录：{error}"))?;
            Ok(directory)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(directory),
        Err(error) => Err(format!("无法读取会话目录：{error}")),
    }
}

fn session_path(root: &Path, create: bool) -> Result<PathBuf, String> {
    Ok(session_dir(root, create)?.join(SESSION_FILE))
}

fn form_draft_path(root: &Path, create: bool) -> Result<PathBuf, String> {
    Ok(session_dir(root, create)?.join(FORM_DRAFT_FILE))
}

fn page_draft_path(root: &Path, create: bool) -> Result<PathBuf, String> {
    Ok(session_dir(root, create)?.join(PAGE_DRAFT_FILE))
}

fn validate_page_draft(root: &Path, draft: &PageDraft) -> Result<(), String> {
    let canonical_root = root
        .canonicalize()
        .map_err(|error| format!("无法读取页面会话项目路径：{error}"))?
        .to_string_lossy()
        .into_owned();
    if draft.schema_version != 1
        || draft.project_root != canonical_root
        || draft.page_id.is_empty()
        || draft.page_id.len() > 128
        || !draft
            .page_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        || draft
            .config_revision
            .as_ref()
            .is_some_and(|revision| !is_sha256(revision))
    {
        return Err("页面会话版本、项目路径或页面标识无效。".into());
    }
    Ok(())
}

pub fn load_page_draft(root: &Path) -> Result<Option<PageDraft>, String> {
    let path = page_draft_path(root, false)?;
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("无法读取页面会话：{error}")),
    };
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > MAX_SESSION_BYTES
    {
        return Err("页面会话不是普通文件或超过 64 KiB；不会自动覆盖。".into());
    }
    let bytes = fs::read(&path).map_err(|error| format!("无法读取页面会话：{error}"))?;
    let draft: PageDraft = serde_json::from_slice(&bytes)
        .map_err(|_| "页面会话格式损坏；不会自动覆盖。".to_owned())?;
    validate_page_draft(root, &draft)?;
    Ok(Some(draft))
}

pub fn save_page_draft(
    root: &Path,
    page_id: &str,
    config_revision: Option<String>,
) -> Result<(), String> {
    let project_root = root
        .canonicalize()
        .map_err(|error| format!("无法固定页面会话项目路径：{error}"))?
        .to_string_lossy()
        .into_owned();
    let draft = PageDraft {
        schema_version: 1,
        project_root,
        page_id: page_id.to_owned(),
        config_revision,
    };
    validate_page_draft(root, &draft)?;
    let directory = session_dir(root, true)?;
    let path = directory.join(PAGE_DRAFT_FILE);
    if let Ok(metadata) = fs::symlink_metadata(&path)
        && (metadata.file_type().is_symlink() || !metadata.is_file())
    {
        return Err("拒绝覆盖非普通页面会话文件。".into());
    }
    let bytes =
        serde_json::to_vec_pretty(&draft).map_err(|error| format!("无法编码页面会话：{error}"))?;
    let temporary = directory.join(format!(".ui-page-session.{}.tmp", std::process::id()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| {
        let mut file = options
            .open(&temporary)
            .map_err(|error| format!("无法创建页面会话临时文件：{error}"))?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("无法写入页面会话：{error}"))?;
        fs::rename(&temporary, &path).map_err(|error| format!("无法原子保存页面会话：{error}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

pub fn clear_page_draft(root: &Path) -> Result<(), String> {
    let path = page_draft_path(root, false)?;
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err("拒绝移除非普通页面会话文件。".into())
        }
        Ok(_) => fs::remove_file(path).map_err(|error| format!("无法移除页面会话：{error}")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("无法检查页面会话：{error}")),
    }
}

fn empty_form_document(root: &Path) -> Result<FormDraftDocument, String> {
    let project_root = root
        .canonicalize()
        .map_err(|error| format!("无法固定表单草稿项目路径：{error}"))?
        .to_string_lossy()
        .into_owned();
    Ok(FormDraftDocument {
        schema_version: 1,
        project_root,
        drafts: BTreeMap::new(),
    })
}

fn read_form_document(root: &Path) -> Result<Option<FormDraftDocument>, String> {
    let path = form_draft_path(root, false)?;
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("无法读取表单草稿：{error}")),
    };
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > MAX_SESSION_BYTES
    {
        return Err("表单草稿不是普通文件或超过 64 KiB；不会自动覆盖。".into());
    }
    let bytes = fs::read(&path).map_err(|error| format!("无法读取表单草稿：{error}"))?;
    let document: FormDraftDocument = serde_json::from_slice(&bytes)
        .map_err(|_| "表单草稿格式损坏；不会自动覆盖。".to_owned())?;
    let expected_root = root
        .canonicalize()
        .map_err(|error| format!("无法读取项目路径：{error}"))?
        .to_string_lossy()
        .into_owned();
    if document.schema_version != 1 || document.project_root != expected_root {
        return Err("表单草稿版本或项目路径不匹配；不会自动恢复。".into());
    }
    if document.drafts.iter().any(|(id, draft)| {
        id.is_empty()
            || id.len() > 128
            || !is_sha256(&draft.schema_revision)
            || contains_sensitive_key(&Value::Object(draft.values.clone()))
    }) {
        return Err("表单草稿包含无效或敏感字段；不会自动恢复。".into());
    }
    Ok(Some(document))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn contains_sensitive_key(value: &Value) -> bool {
    match value {
        Value::Object(values) => values.iter().any(|(key, value)| {
            let normalized = key
                .chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>();
            [
                "password",
                "secret",
                "token",
                "credential",
                "authorization",
                "apikey",
                "privatekey",
            ]
            .iter()
            .any(|marker| normalized.contains(marker))
                || contains_sensitive_key(value)
        }),
        Value::Array(values) => values.iter().any(contains_sensitive_key),
        _ => false,
    }
}

fn save_form_document(root: &Path, document: &FormDraftDocument) -> Result<(), String> {
    let directory = session_dir(root, true)?;
    let path = directory.join(FORM_DRAFT_FILE);
    if let Ok(metadata) = fs::symlink_metadata(&path)
        && (metadata.file_type().is_symlink() || !metadata.is_file())
    {
        return Err("拒绝覆盖非普通表单草稿文件。".into());
    }
    let bytes = serde_json::to_vec_pretty(document)
        .map_err(|error| format!("无法编码表单草稿：{error}"))?;
    if bytes.len() as u64 > MAX_SESSION_BYTES {
        return Err("表单草稿超过 64 KiB；请先完成或丢弃旧草稿。".into());
    }
    let temporary = directory.join(format!(".ui-form-drafts.{}.tmp", std::process::id()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| {
        let mut file = options
            .open(&temporary)
            .map_err(|error| format!("无法创建表单草稿临时文件：{error}"))?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("无法写入表单草稿：{error}"))?;
        fs::rename(&temporary, &path).map_err(|error| format!("无法原子保存表单草稿：{error}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

pub fn load_form_draft(
    root: &Path,
    form_id: &str,
    schema_revision: &str,
) -> Result<Option<FormDraft>, String> {
    let Some(document) = read_form_document(root)? else {
        return Ok(None);
    };
    Ok(document.drafts.get(form_id).map(|draft| FormDraft {
        schema_revision: draft.schema_revision.clone(),
        values: draft.values.clone(),
        stale: draft.schema_revision != schema_revision,
    }))
}

pub fn save_form_draft(
    root: &Path,
    form_id: &str,
    schema_revision: &str,
    values: &Map<String, Value>,
) -> Result<(), String> {
    if form_id.is_empty() || form_id.len() > 128 || !is_sha256(schema_revision) {
        return Err("表单草稿标识或 Schema revision 无效。".into());
    }
    if contains_sensitive_key(&Value::Object(values.clone())) {
        return Err("表单草稿包含敏感字段；拒绝持久化。".into());
    }
    let mut document = read_form_document(root)?.unwrap_or(empty_form_document(root)?);
    document.drafts.insert(
        form_id.to_owned(),
        StoredFormDraft {
            schema_revision: schema_revision.to_owned(),
            values: values.clone(),
        },
    );
    save_form_document(root, &document)
}

pub fn clear_form_draft(root: &Path, form_id: &str) -> Result<(), String> {
    let Some(mut document) = read_form_document(root)? else {
        return Ok(());
    };
    document.drafts.remove(form_id);
    if document.drafts.is_empty() {
        let path = form_draft_path(root, false)?;
        fs::remove_file(path).map_err(|error| format!("无法移除表单草稿：{error}"))
    } else {
        save_form_document(root, &document)
    }
}

pub fn load(root: &Path) -> Result<Option<OnboardingDraft>, String> {
    let path = session_path(root, false)?;
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("无法读取向导会话：{error}")),
    };
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > MAX_SESSION_BYTES
    {
        return Err("向导会话不是普通文件或超过 64 KiB；不会自动覆盖。".into());
    }
    let bytes = fs::read(&path).map_err(|error| format!("无法读取向导会话：{error}"))?;
    let draft: OnboardingDraft = serde_json::from_slice(&bytes)
        .map_err(|_| "向导会话格式损坏；不会自动覆盖。".to_owned())?;
    draft.validate(root)?;
    Ok(Some(draft))
}

pub fn save(root: &Path, draft: &OnboardingDraft) -> Result<(), String> {
    draft.validate(root)?;
    let directory = session_dir(root, true)?;
    let path = directory.join(SESSION_FILE);
    if let Ok(metadata) = fs::symlink_metadata(&path)
        && (metadata.file_type().is_symlink() || !metadata.is_file())
    {
        return Err("拒绝覆盖非普通向导会话文件。".into());
    }
    let bytes =
        serde_json::to_vec_pretty(draft).map_err(|error| format!("无法编码向导会话：{error}"))?;
    if bytes.len() as u64 > MAX_SESSION_BYTES {
        return Err("向导会话超过 64 KiB。".into());
    }
    let temporary = directory.join(format!(".ui-session.{}.tmp", std::process::id()));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| {
        let mut file = options
            .open(&temporary)
            .map_err(|error| format!("无法创建临时会话文件：{error}"))?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("无法写入向导会话：{error}"))?;
        fs::rename(&temporary, &path).map_err(|error| format!("无法原子保存向导会话：{error}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

pub fn clear(root: &Path) -> Result<(), String> {
    let path = session_path(root, false)?;
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err("拒绝移除非普通向导会话文件。".into())
        }
        Ok(_) => fs::remove_file(path).map_err(|error| format!("无法移除向导会话：{error}")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("无法检查向导会话：{error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OnboardingDraft, clear, clear_form_draft, clear_page_draft, load, load_form_draft,
        load_page_draft, save, save_form_draft, save_page_draft,
    };
    use serde_json::{Map, Value};

    #[test]
    fn draft_round_trip_persists_only_confirmed_non_sensitive_entry_names() {
        let root = tempfile::tempdir().unwrap();
        let draft = OnboardingDraft::new(
            root.path(),
            "sample_model:build_model",
            Some("sample_data:example_batch"),
            None,
        )
        .unwrap();
        save(root.path(), &draft).unwrap();
        assert_eq!(load(root.path()).unwrap(), Some(draft));
        let bytes = std::fs::read(root.path().join(".model-maven/ui-session.json")).unwrap();
        let document: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(document["schema_version"], 1);
        assert!(document.get("password").is_none());
        clear(root.path()).unwrap();
        assert!(load(root.path()).unwrap().is_none());
    }

    #[test]
    fn draft_fingerprints_local_python_sources_and_detects_changes() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("models.py");
        std::fs::write(&source, "def build(): return 1\n").unwrap();
        let draft = OnboardingDraft::new(root.path(), "models:build", None, None).unwrap();
        assert!(draft.source_fingerprint.is_some());
        assert!(!super::source_is_stale(root.path(), &draft));
        std::fs::write(&source, "def build(): return 2\n").unwrap();
        assert!(super::source_is_stale(root.path(), &draft));
    }

    #[test]
    fn draft_fingerprint_tracks_transitive_local_imports_but_ignores_unrelated_files() {
        let root = tempfile::tempdir().unwrap();
        let package = root.path().join("model_parts");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(package.join("__init__.py"), "from .builder import build\n").unwrap();
        std::fs::write(
            package.join("builder.py"),
            "from .layers import make_layer\ndef build(): return make_layer()\n",
        )
        .unwrap();
        std::fs::write(package.join("layers.py"), "from .shape import WIDTH\n").unwrap();
        std::fs::write(package.join("shape.py"), "WIDTH = 32\n").unwrap();
        let draft = OnboardingDraft::new(root.path(), "model_parts:build", None, None).unwrap();
        assert!(!super::source_is_stale(root.path(), &draft));

        std::fs::write(root.path().join("unrelated.py"), "VALUE = 7\n").unwrap();
        assert!(!super::source_is_stale(root.path(), &draft));

        std::fs::write(package.join("shape.py"), "WIDTH = 64\n").unwrap();
        assert!(super::source_is_stale(root.path(), &draft));
    }

    #[test]
    fn draft_requires_revalidation_for_dynamic_import_graphs() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("models.py"),
            "import importlib\ndef build(): return importlib.import_module('custom_model')\n",
        )
        .unwrap();
        let draft = OnboardingDraft::new(root.path(), "models:build", None, None).unwrap();
        assert!(draft.source_fingerprint.is_none());
        assert!(super::source_is_stale(root.path(), &draft));
    }

    #[test]
    fn draft_requires_revalidation_for_dynamic_module_loaders() {
        for (name, source) in [
            (
                "runpy_loader",
                "import runpy\ndef build(): return runpy.run_path('plugin.py')\n",
            ),
            (
                "spec_loader",
                "from importlib.util import spec_from_file_location\ndef build(): return spec_from_file_location('plugin', 'plugin.py')\n",
            ),
            (
                "module_loader",
                "from importlib.util import module_from_spec\ndef build(spec): return module_from_spec(spec)\n",
            ),
        ] {
            let root = tempfile::tempdir().unwrap();
            std::fs::write(root.path().join("models.py"), source).unwrap();
            let draft = OnboardingDraft::new(root.path(), "models:build", None, None).unwrap();
            assert!(
                draft.source_fingerprint.is_none(),
                "{name} must not be treated as a statically closed source graph"
            );
            assert!(super::source_is_stale(root.path(), &draft));
        }
    }

    #[test]
    fn draft_requires_revalidation_when_entrypoint_source_is_not_local() {
        let root = tempfile::tempdir().unwrap();
        let draft =
            OnboardingDraft::new(root.path(), "installed.models:build", None, None).unwrap();
        assert!(draft.source_fingerprint.is_none());
        assert!(super::source_is_stale(root.path(), &draft));
    }

    #[test]
    fn form_drafts_round_trip_are_revision_bound_and_clear_independently() {
        let root = tempfile::tempdir().unwrap();
        let form_id = "a".repeat(64);
        let revision = "b".repeat(64);
        let other_revision = "c".repeat(64);
        let mut values = Map::new();
        values.insert("width".into(), Value::from(32));
        save_form_draft(root.path(), &form_id, &revision, &values).unwrap();

        let loaded = load_form_draft(root.path(), &form_id, &revision)
            .unwrap()
            .unwrap();
        assert!(!loaded.stale);
        assert_eq!(loaded.values["width"], 32);
        assert!(
            load_form_draft(root.path(), &form_id, &other_revision)
                .unwrap()
                .unwrap()
                .stale
        );
        clear_form_draft(root.path(), &form_id).unwrap();
        assert!(
            load_form_draft(root.path(), &form_id, &revision)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn form_draft_storage_rejects_sensitive_keys() {
        let root = tempfile::tempdir().unwrap();
        let mut values = Map::new();
        values.insert("api_token".into(), Value::from("must-not-persist"));
        values.insert("api_key".into(), Value::from("must-not-persist"));
        assert!(save_form_draft(root.path(), &"a".repeat(64), &"b".repeat(64), &values).is_err());
        assert!(
            !root
                .path()
                .join(".model-maven/ui-form-drafts.json")
                .exists()
        );
    }

    #[test]
    fn page_selection_round_trips_and_is_bound_to_the_project() {
        let root = tempfile::tempdir().unwrap();
        let revision = "c".repeat(64);
        save_page_draft(root.path(), "components", Some(revision.clone())).unwrap();
        let draft = load_page_draft(root.path()).unwrap().unwrap();
        assert_eq!(draft.page_id, "components");
        assert_eq!(draft.config_revision.as_deref(), Some(revision.as_str()));
        assert_eq!(
            draft.project_root,
            root.path().canonicalize().unwrap().to_string_lossy()
        );
        clear_page_draft(root.path()).unwrap();
        assert!(load_page_draft(root.path()).unwrap().is_none());
    }

    #[test]
    fn page_selection_rejects_invalid_page_ids_and_symlink_targets() {
        let root = tempfile::tempdir().unwrap();
        assert!(save_page_draft(root.path(), "../model", None).is_err());
        std::fs::create_dir_all(root.path().join(".model-maven")).unwrap();
        let outside = root.path().join("outside.json");
        std::fs::write(&outside, "preserve").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(
                &outside,
                root.path().join(".model-maven/ui-page-session.json"),
            )
            .unwrap();
            assert!(save_page_draft(root.path(), "components", None).is_err());
            assert!(clear_page_draft(root.path()).is_err());
            assert_eq!(std::fs::read_to_string(outside).unwrap(), "preserve");
        }
    }

    #[test]
    fn draft_fingerprint_includes_the_optional_batch_factory_source() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("models.py"), "def build(): return 1\n").unwrap();
        std::fs::write(root.path().join("batches.py"), "def example(): return 1\n").unwrap();
        let draft =
            OnboardingDraft::new(root.path(), "models:build", Some("batches:example"), None)
                .unwrap();
        assert!(!super::source_is_stale(root.path(), &draft));
        std::fs::write(root.path().join("batches.py"), "def example(): return 2\n").unwrap();
        assert!(super::source_is_stale(root.path(), &draft));
    }

    #[cfg(unix)]
    #[test]
    fn session_store_refuses_symlink_targets_and_corrupt_content() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".model-maven")).unwrap();
        let outside = root.path().join("outside.json");
        std::fs::write(&outside, "preserve").unwrap();
        symlink(&outside, root.path().join(".model-maven/ui-session.json")).unwrap();
        let draft = OnboardingDraft::new(root.path(), "m:f", None, None).unwrap();
        assert!(save(root.path(), &draft).is_err());
        assert!(clear(root.path()).is_err());
        assert_eq!(std::fs::read_to_string(outside).unwrap(), "preserve");
        std::fs::remove_file(root.path().join(".model-maven/ui-session.json")).unwrap();
        std::fs::write(root.path().join(".model-maven/ui-session.json"), "broken").unwrap();
        assert!(load(root.path()).is_err());
    }
}
