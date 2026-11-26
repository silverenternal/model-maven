use super::*;
use std::path::{Path, PathBuf};

const AUTHORIZATION_IDENTITY_MAX_FILES: usize = 50_000;
const AUTHORIZATION_IDENTITY_MAX_BYTES: u64 = 768 * 1024 * 1024;

pub(super) fn authorization_identity(
    request: &Value,
    root: &Path,
    config_revision: Option<&str>,
    input_sources: &Map<String, Value>,
    python_explicit: bool,
) -> Value {
    let mut incomplete = Vec::new();
    let operation = request["operation"].as_str().unwrap_or("unknown");
    let action_class = match operation {
        "inspect" | "layers" | "probe" | "environment_library_probe" => "user_code",
        "train" | "training_probe" | "generic_loop_train" | "generic_loop_trial" => "training",
        "edit" if request["edit"]["dry_run"] == true => "user_code",
        "edit" | "semantic_mark" | "semantic_supervise" | "assembly_patch" => "persistent_write",
        _ => "unknown",
    };
    let reusable_action = matches!(action_class, "user_code")
        && matches!(operation, "inspect" | "layers" | "probe");
    if !reusable_action {
        incomplete.push("此操作类别不允许会话复用。".to_owned());
    }
    let input_digest = request["expected_input_content"].as_str()
        .filter(|digest| is_sha256(digest));
    // Structural inspection and layer listing construct the model but do not
    // load a batch. A project default batch must not make their authorization
    // permanently non-reusable; forward probes still require actual input
    // content identity when a batch factory is selected.
    if operation == "probe" && request["batch"].is_string() && input_digest.is_none() {
        incomplete.push("输入工厂的实际输出内容没有可复核摘要。".to_owned());
    }
    if operation == "probe"
        && request["probe_device"].is_null()
        && request["variant_device_policy"].is_null()
        && request["variant"].is_null()
    {
        incomplete.push("模型工厂的有效设备放置尚未观察。".to_owned());
    }
    let display_summary = match operation {
        "inspect" => "Runs the selected model factory to inspect the current model; user code may have external side effects.",
        "layers" => "Runs the selected model factory to inspect layers; user code may have external side effects.",
        "probe" => "Runs the selected model and input for a forward pass; user code and Python startup hooks may have external side effects.",
        _ => "Runs the selected project operation; external side effects may occur.",
    };
    if !incomplete.is_empty() {
        // No later inventory can make this identity reusable. Avoid traversing
        // potentially large Python installations for a per-run confirmation.
        return json!({
            "complete": false,
            "identity_digest": Value::Null,
            "components": {},
            "incomplete_reasons": incomplete,
            "action_class": action_class,
            "display_summary": display_summary,
        });
    }

    let runtime = &request["runtime"];
    let python_raw = runtime["python_executable"].as_str();
    let selected_python = python_raw.and_then(resolve_python_selection);
    let python_path = selected_python.as_ref().and_then(|path| path.canonicalize().ok());
    let python_file_digest = python_path.as_deref().and_then(hash_file);
    if python_path.is_none() || python_file_digest.is_none() {
        incomplete.push("无法解析并校验实际 Python 可执行文件。".to_owned());
    }
    let python_source = python_raw.map(|raw| python_selection_source(root, raw, python_explicit)).unwrap_or("unknown");
    if python_source == "unknown" {
        incomplete.push("Python 解释器来源未知。".to_owned());
    }
    let mut virtualenv_config_digest = None;
    if let (Some(selected), Some(resolved)) = (selected_python.as_deref(), python_path.as_deref()) {
        if let Some(prefix) = selected.parent().and_then(Path::parent) {
            let config = prefix.join("pyvenv.cfg");
            match std::fs::read_to_string(&config) {
                Ok(contents) => {
                    use sha2::{Digest, Sha256};
                    virtualenv_config_digest = hash_file(&config);
                    let read_digest = format!("{:x}", Sha256::digest(contents.as_bytes()));
                    let home = contents.lines().filter_map(|line| line.split_once('='))
                        .find(|(key, _)| key.trim().eq_ignore_ascii_case("home"))
                        .map(|(_, value)| PathBuf::from(value.trim()))
                        .and_then(|path| path.canonicalize().ok());
                    let actual_parent = resolved.parent().and_then(|path| path.canonicalize().ok());
                    if virtualenv_config_digest.as_deref() != Some(read_digest.as_str())
                        || home.is_none() || home != actual_parent {
                        incomplete.push("虚拟环境的基础解释器路径无法从所选可执行文件核对；标准库身份未知。".to_owned());
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
                Err(_) => incomplete.push("虚拟环境配置无法读取；基础解释器身份未知。".to_owned()),
            }
        }
    }
    if std::env::var_os("PYTHONHOME").is_some_and(|value| !value.is_empty()) {
        incomplete.push("显式 PYTHONHOME 的运行库路径尚未完整核对。".to_owned());
    }
    if !incomplete.is_empty() {
        // Missing interpreter identity or an unverified base cannot become
        // reusable by scanning project files or an unrelated Python tree.
        return json!({
            "complete": false,
            "identity_digest": Value::Null,
            "components": {},
            "incomplete_reasons": incomplete,
            "action_class": action_class,
            "display_summary": display_summary,
        });
    }

    let mut source_roots = vec![root.to_path_buf()];
    if let Some(roots) = runtime["project_import_roots"].as_array() {
        for relative in roots.iter().filter_map(Value::as_str) {
            source_roots.push(root.join(relative));
        }
    }
    if let Some(directory) = runtime["working_directory"].as_str() {
        source_roots.push(PathBuf::from(directory));
    } else {
        source_roots.push(root.to_path_buf());
    }
    let pythonpath = std::env::var_os("PYTHONPATH");
    if let Some(paths) = pythonpath.as_deref() {
        for path in std::env::split_paths(paths) {
            if path.as_os_str().is_empty() { continue; }
            source_roots.push(path);
        }
    }
    let source_digest = inventory_digest(&source_roots, InventoryKind::ProjectSources);
    if let Err(reason) = &source_digest {
        incomplete.push(format!("项目或模块搜索路径中的源码不能完整枚举和摘要：{reason}。"));
    }

    let mut runtime_roots = python_path.as_deref()
        .map(|path| python_runtime_roots(path, selected_python.as_deref().unwrap_or(path)))
        .unwrap_or_default();
    if let Some(paths) = pythonpath.as_deref() {
        runtime_roots.extend(std::env::split_paths(paths).filter(|path| !path.as_os_str().is_empty()));
    }
    let runtime_digest = if runtime_roots.is_empty() {
        Err("无法定位 Python 标准库或安装包目录。".to_owned())
    } else {
        inventory_digest(&runtime_roots, InventoryKind::RuntimeCode)
    };
    if let Err(reason) = &runtime_digest {
        incomplete.push(format!("Python 标准库、安装包或扩展模块身份不能完整校验：{reason}。"));
    }

    let environment = environment_digest();
    if environment.is_err() {
        incomplete.push("进程环境含不可比较的值。".to_owned());
    }
    let mut stable_request = request.clone();
    if let Some(object) = stable_request.as_object_mut() {
        for key in ["request_id", "progress_path", "stop_file", "log_path", "run_id", "run_directory"] {
            object.remove(key);
        }
    }
    let selection = json!({
        "request": stable_request.clone(),
        "project_config_revision": config_revision,
        "input_sources": input_sources,
        "input_content_sha256": input_digest,
        "python_selection_source": python_source,
        "python_executable_sha256": python_file_digest,
        "virtualenv_config_sha256": virtualenv_config_digest,
        "project_source_inventory_sha256": source_digest.as_ref().ok(),
        "python_runtime_inventory_sha256": runtime_digest.as_ref().ok(),
        "process_environment_sha256": environment.as_ref().ok(),
    });
    let selection_digest = crate::domain::input::content_revision(&selection).ok();
    if selection_digest.is_none() {
        incomplete.push("规范化授权身份不能序列化。".to_owned());
    }
    incomplete.sort();
    incomplete.dedup();

    let complete = incomplete.is_empty() && selection_digest.is_some();
    let components = json!({
        "request_and_project": crate::domain::input::content_revision(&json!({
            "project_root": root,
            "config_revision": config_revision,
            "input_sources": input_sources,
            "request": stable_request
        })).ok(),
        "python_selection": crate::domain::input::content_revision(&json!({
            "source": python_source,
            "executable_sha256": python_file_digest,
            "virtualenv_config_sha256": virtualenv_config_digest,
            "runtime": runtime
        })).ok(),
        "source_inventory": source_digest.ok(),
        "runtime_inventory": runtime_digest.ok(),
        "environment": environment.ok(),
    });
    let digest = if complete { selection_digest } else { None };
    json!({
        "complete": complete,
        "identity_digest": digest,
        "components": components,
        "incomplete_reasons": incomplete,
        "action_class": action_class,
        "display_summary": display_summary,
    })
}

#[derive(Clone, Copy)]
enum InventoryKind { ProjectSources, RuntimeCode }

fn inventory_digest(roots: &[PathBuf], kind: InventoryKind) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut canonical_roots = Vec::new();
    for root in roots {
        let canonical = root.canonicalize().map_err(|error| error.to_string())?;
        if !canonical.is_dir() && !(canonical.is_file()
            && canonical.extension().and_then(|ext| ext.to_str()) == Some("zip")) {
            return Err("inventory root is neither a directory nor a ZIP archive".into());
        }
        if !canonical_roots.contains(&canonical) { canonical_roots.push(canonical); }
    }
    canonical_roots.sort();
    let mut records = Vec::<(String, String)>::new();
    let mut bytes = 0_u64;
    // Refuse oversized installations before reading hundreds of MiB solely
    // to discover that the identity cannot be reused.
    for (root_index, root) in canonical_roots.iter().enumerate() {
        inventory_root(root, root_index, kind, &mut records, &mut bytes, true)?;
    }
    records.clear();
    bytes = 0;
    for (root_index, root) in canonical_roots.iter().enumerate() {
        inventory_root(root, root_index, kind, &mut records, &mut bytes, false)?;
    }
    if records.is_empty() { return Err("inventory contained no relevant source files".into()); }
    records.sort();
    let mut hasher = Sha256::new();
    for (path, digest) in records {
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update(digest.as_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn inventory_root(
    root: &Path, root_index: usize, kind: InventoryKind,
    records: &mut Vec<(String, String)>, bytes: &mut u64, scan_only: bool,
) -> Result<(), String> {
    if root.is_dir() {
        return inventory_walk(root, root, root_index, kind, records, bytes, &mut Vec::new(), scan_only);
    }
    let metadata = std::fs::metadata(root).map_err(|error| error.to_string())?;
    if !metadata.is_file() || records.len() >= AUTHORIZATION_IDENTITY_MAX_FILES {
        return Err("身份清单 ZIP 不可读或超过文件数量上限".into());
    }
    let next = bytes.saturating_add(metadata.len());
    if next > AUTHORIZATION_IDENTITY_MAX_BYTES { return Err("身份清单超过字节数上限".into()); }
    *bytes = next;
    let digest = if scan_only { String::new() } else {
        hash_file(root).ok_or_else(|| "ZIP 无法读取或计算摘要".to_owned())?
    };
    records.push((format!("{root_index}:<zip-root>"), digest));
    Ok(())
}

fn inventory_walk(
    root: &Path,
    directory: &Path,
    root_index: usize,
    kind: InventoryKind,
    records: &mut Vec<(String, String)>,
    bytes: &mut u64,
    ancestors: &mut Vec<PathBuf>,
    scan_only: bool,
) -> Result<(), String> {
    let canonical_directory = directory.canonicalize().map_err(|error| error.to_string())?;
    if ancestors.len() >= 128 || ancestors.contains(&canonical_directory) {
        return Err(format!("身份清单目录循环或层级过深：{}", directory.display()));
    }
    ancestors.push(canonical_directory);
    let mut entries = std::fs::read_dir(directory).map_err(|error| error.to_string())?
        .map(|entry| entry.map_err(|error| error.to_string())).collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let metadata = std::fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() {
            if matches!(kind, InventoryKind::ProjectSources) {
                return Err(format!("项目身份清单中存在符号链接：{}", path.display()));
            }
            if records.len() >= AUTHORIZATION_IDENTITY_MAX_FILES { return Err("身份清单超过文件数量上限".into()); }
            let link = std::fs::read_link(&path).map_err(|error| format!("无法读取链接 {}：{error}", path.display()))?;
            let target = path.canonicalize().map_err(|error| format!("无法解析链接 {}：{error}", path.display()))?;
            let target_meta = std::fs::metadata(&target).map_err(|error| error.to_string())?;
            let relative = path.strip_prefix(root).map_err(|error| error.to_string())?;
            let link_text = link.to_str().ok_or("链接目标不是 UTF-8")?;
            let target_text = target.to_str().ok_or("链接规范路径不是 UTF-8")?;
            let file_digest = if target_meta.is_file() {
                check_python_path_file(&path, &target, target_meta.len())?;
                let next = bytes.saturating_add(target_meta.len());
                if next > AUTHORIZATION_IDENTITY_MAX_BYTES { return Err("身份清单超过字节数上限".into()); }
                *bytes = next;
                Some(if scan_only { String::new() } else { hash_file(&target).ok_or("链接目标无法读取或计算摘要")? })
            } else if target_meta.is_dir() { None } else {
                return Err(format!("链接目标不是普通文件或目录：{}", path.display()));
            };
            let digest = crate::domain::input::content_revision(&json!({
                "link": link_text, "target": target_text, "file_sha256": file_digest
            })).map_err(|error| error.to_string())?;
            records.push((format!("{root_index}:{}", relative.to_string_lossy()), digest));
            if target_meta.is_dir() {
                inventory_walk(root, &path, root_index, kind, records, bytes, ancestors, scan_only)?;
            }
            let after = std::fs::metadata(&target).map_err(|error| error.to_string())?;
            if std::fs::read_link(&path).map_err(|error| error.to_string())? != link
                || path.canonicalize().map_err(|error| error.to_string())? != target
                || (target_meta.is_file() && !same_authorization_file(&target_meta, &after))
                || (target_meta.is_dir() && (target_meta.len() != after.len() || target_meta.modified().ok() != after.modified().ok())) {
                return Err(format!("链接或目标在身份检查期间变化：{}", path.display()));
            }
            continue;
        }
        if metadata.is_dir() {
            if matches!(kind, InventoryKind::ProjectSources)
                && matches!(entry.file_name().to_str(), Some(".git" | ".model-maven" | "__pycache__" | ".venv" | "venv" | "node_modules" | "target" | "dist")) {
                continue;
            }
            inventory_walk(root, &path, root_index, kind, records, bytes, ancestors, scan_only)?;
            continue;
        }
        if !metadata.is_file() || !relevant_inventory_file(&path, kind) { continue; }
        check_python_path_file(&path, &path, metadata.len())?;
        if records.len() >= AUTHORIZATION_IDENTITY_MAX_FILES { return Err("身份清单超过文件数量上限".into()); }
        let next = bytes.saturating_add(metadata.len());
        if next > AUTHORIZATION_IDENTITY_MAX_BYTES { return Err("身份清单超过字节数上限".into()); }
        *bytes = next;
        let digest = if scan_only { String::new() } else {
            hash_file(&path).ok_or_else(|| "文件无法读取或计算摘要".to_owned())?
        };
        let after = std::fs::metadata(&path).map_err(|error| error.to_string())?;
        if !same_authorization_file(&metadata, &after) {
            return Err(format!("文件在身份检查期间变化：{}", path.display()));
        }
        let relative = path.strip_prefix(root).map_err(|error| error.to_string())?;
        records.push((format!("{root_index}:{}", relative.to_string_lossy()), digest));
    }
    ancestors.pop();
    Ok(())
}

fn check_python_path_file(name: &Path, target: &Path, size: u64) -> Result<(), String> {
    if name.extension().and_then(|ext| ext.to_str()) != Some("pth") { return Ok(()); }
    // .pth files can execute imports or add paths outside the inventoried roots.
    if size > 1024 * 1024 { return Err("Python .pth 文件过大，无法核对启动路径".into()); }
    let contents = std::fs::read_to_string(target)
        .map_err(|_| "Python .pth 文件无法读取或不是 UTF-8".to_owned())?;
    if contents.lines().any(|line| {
        let line = line.trim();
        !line.is_empty() && !line.starts_with('#')
    }) {
        return Err(format!("Python .pth 文件可能执行代码或添加未核对的路径：{}", name.display()));
    }
    Ok(())
}

fn relevant_inventory_file(path: &Path, kind: InventoryKind) -> bool {
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
    match kind {
        // Python can execute sourceless bytecode, import code from zip files,
        // and process .pth files (including executable import lines) at startup.
        // A change to any of these must invalidate a reusable authorization.
        InventoryKind::ProjectSources => matches!(path.extension().and_then(|ext| ext.to_str()), Some("py" | "pyi" | "pyc" | "pth" | "zip" | "so" | "pyd" | "dll" | "dylib")),
        InventoryKind::RuntimeCode => matches!(path.extension().and_then(|ext| ext.to_str()), Some("py" | "pyi" | "pyc" | "pth" | "zip" | "so" | "pyd" | "dll" | "dylib"))
            || matches!(name, "METADATA" | "RECORD" | "direct_url.json"),
    }
}

fn resolve_python_selection(raw: &str) -> Option<PathBuf> {
    let path = PathBuf::from(raw);
    let has_separator = raw.contains('/') || raw.contains('\\');
    let candidates: Vec<PathBuf> = if path.is_absolute() || has_separator {
        vec![if path.is_absolute() { path } else { std::env::current_dir().ok()?.join(path) }]
    } else {
        std::env::split_paths(&std::env::var_os("PATH")?).map(|directory| directory.join(&path)).collect()
    };
    for candidate in candidates {
        let Ok(canonical) = candidate.canonicalize() else { continue; };
        if !canonical.is_file() { continue; }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if canonical.metadata().ok()?.permissions().mode() & 0o111 == 0 { continue; }
        }
        return Some(candidate);
    }
    None
}

fn python_runtime_roots(executable: &Path, selected: &Path) -> Vec<PathBuf> {
    let mut prefixes = Vec::new();
    if let Some(prefix) = executable.parent().and_then(Path::parent) { prefixes.push(prefix.to_path_buf()); }
    if let Some(prefix) = selected.parent().and_then(Path::parent) {
        prefixes.push(prefix.to_path_buf());
    }
    let mut roots = Vec::new();
    for prefix in prefixes {
        let windows = prefix.join("Lib");
        if windows.is_dir() { roots.push(windows); }
        let lib = prefix.join("lib");
        if let Ok(entries) = std::fs::read_dir(lib) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("python") && (path.is_dir()
                    || (path.is_file() && name.ends_with(".zip"))) { roots.push(path); }
            }
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

fn environment_digest() -> Result<String, String> {
    let mut environment = BTreeMap::new();
    for (key, value) in std::env::vars_os() {
        let key = key.into_string().map_err(|_| "non-UTF-8 environment key".to_owned())?;
        if key == "MODEL_MAVEN_IF_AUTHORIZATION_IDENTITY" { continue; }
        let value = value.into_string().map_err(|_| "non-UTF-8 environment value".to_owned())?;
        environment.insert(key, value);
    }
    crate::domain::input::content_revision(&json!(environment))
}

fn python_selection_source(root: &Path, raw: &str, explicit: bool) -> &'static str {
    if explicit { return "explicit"; }
    let config = std::fs::read(root.join(".model-maven/project.json"))
        .ok().and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    if config.as_ref().and_then(|config| config["runtime_python"].as_str()) == Some(raw) {
        return "project_configuration";
    }
    if let Some(environment) = std::fs::read(root.join(".model-maven/environment.json"))
        .ok().and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
    {
        if environment["schema_version"] == 1 {
            if let Some(venv) = environment["venv"].as_str() {
                if raw.starts_with(venv) { return "project_environment"; }
            }
        }
    } else if raw.starts_with(".venv/") || raw.starts_with(".venv\\") {
        return "project_environment";
    }
    "path"
}

fn same_authorization_file(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    if !left.is_file() || !right.is_file() || left.len() != right.len() { return false; }
    if left.modified().ok().zip(right.modified().ok()).is_none_or(|(a, b)| a != b) { return false; }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if left.dev() != right.dev() || left.ino() != right.ino()
            || left.ctime() != right.ctime() || left.ctime_nsec() != right.ctime_nsec() { return false; }
    }
    true
}

fn hash_file(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > 256 * 1024 * 1024 { return None; }
    let mut file = std::fs::File::open(path).ok()?;
    if !same_authorization_file(&metadata, &file.metadata().ok()?) { return None; }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes_read = 0_u64;
    loop {
        let count = file.read(&mut buffer).ok()?;
        if count == 0 { break; }
        bytes_read = bytes_read.checked_add(count as u64)?;
        if bytes_read > metadata.len() { return None; }
        hasher.update(&buffer[..count]);
    }
    if bytes_read != metadata.len() || !same_authorization_file(&metadata, &file.metadata().ok()?)
        || !same_authorization_file(&metadata, &std::fs::metadata(path).ok()?) { return None; }
    Some(format!("{:x}", hasher.finalize()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn variant_base_token(variant: Option<&Value>) -> Result<String, String> {
    variant.map_or_else(
        || Ok("absent".to_owned()),
        crate::domain::input::content_revision,
    )
}

pub(super) struct PreparedRunInputs {
    pub variant_name: String,
    pub variant_path: PathBuf,
    pub variant: Option<Value>,
    pub model_inputs: super::model::ModelInputs,
    pub batch: super::data::BatchInput,
    _variant_lock: Option<Box<dyn VariantLock>>,
    _source_lock: Option<Box<dyn SourceEditLock>>,
}

impl<B: Backend> Application<B> {
    pub(super) fn prepare_run_inputs(
        &self,
        cli: &Cli,
        job: &mut Job,
        root: &Path,
        source_target: bool,
    ) -> Result<PreparedRunInputs, Report> {
        if cli.if_variant_base.is_some()
            && !matches!(job.operation, "probe" | "semantic_mark" | "semantic_supervise" | "training_probe" | "train")
            && (job.edit.is_none() || source_target || job.dry_run)
        {
            return Err(Report::error(
                "edit",
                "VARIANT_BASE_UNSUPPORTED",
                "变体基准条件只支持实际的变体编辑提交。",
                "先使用 --dry-run 获取基准，再对变体提交指定条件。",
            ));
        }
        let variant_name = cli.variant.as_deref().unwrap_or("working").to_owned();
        let external_loop = matches!(job.operation, "training_probe" | "train")
            && job.training_mode == Some("external-loop");
        let variant_path = self
            .variants
            .path(root, &variant_name)
            .map_err(|error| Report::error(job.operation, "VARIANT_NAME_INVALID", error, ""))?;
        let writes_variant = job.operation == "edit" && !source_target && !job.dry_run;
        let use_variant = cli.variant.is_some()
            || (job.layer_context && !source_target)
            || writes_variant
            || (job.inputs.model.is_none() && !external_loop)
            || job.variant_snapshot.is_some();
        let mut variant = if use_variant {
            self.variants
                .load(root, &variant_name)
                .map_err(|error| Report::error(job.operation, "VARIANT_READ_FAILED", error, ""))?
        } else {
            None
        };
        if let Some(snapshot) = &job.variant_snapshot {
            variant = Some(snapshot.clone());
        }
        let observed_variant = variant.clone();
        if variant.is_none() && cli.variant.is_some() && job.operation == "edit" {
            self.resolve_variant_edit_defaults(root, job)?;
        }
        if variant.is_none() && cli.variant.is_some() && job.operation != "edit" {
            return Err(Report::error(
                job.operation,
                "VARIANT_NOT_FOUND",
                format!("找不到变体：{variant_name}"),
                "先创建变体或省略 --variant 并提供 --model。",
            ));
        }
        if variant.is_none() && job.inputs.model.is_none() && !external_loop {
            return Err(Report::error(
                job.operation,
                "MODEL_REQUIRED",
                "缺少模型入口。",
                "例如 --model examples.models:build_shared；前向再加 --batch examples.models:shared_batch。",
            ));
        }
        let model_inputs = self.resolve_model_inputs(job, root)?;
        let batch = super::data::batch_input(&job.inputs, job.operation)?;

        // Keep both guards alive until Backend execution and edit finalization return.
        let variant_lock = if writes_variant
            || (matches!(job.operation, "probe" | "semantic_mark" | "semantic_supervise" | "training_probe" | "train")
                && cli.if_variant_base.is_some())
        {
            let lock = self
                .variants
                .lock(root, &variant_name)
                .map_err(|error| Report::error(job.operation, "VARIANT_LOCKED", error, ""))?;
            variant = self
                .variants
                .load(root, &variant_name)
                .map_err(|error| Report::error(job.operation, "VARIANT_READ_FAILED", error, ""))?;
            if cli.if_variant_base.is_none() && job.variant_snapshot.is_none() && variant != observed_variant {
                return Err(Report::error(
                    job.operation,
                    "VARIANT_BASE_CONFLICT",
                    "准备操作期间变体发生变化。",
                    "本次操作未执行；重新预览并核对变体。",
                ));
            }
            if let Some(expected) = &cli.if_variant_base {
                if expected != "absent"
                    && (expected.len() != 64
                        || !expected
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
                {
                    return Err(Report::error(
                        job.operation,
                        "VARIANT_BASE_INVALID",
                        "变体基准格式无效。",
                        "使用预览报告提供的 token。",
                    ));
                }
                let actual = variant_base_token(variant.as_ref()).map_err(|error| {
                    Report::error(
                        job.operation,
                        "VARIANT_BASE_UNAVAILABLE",
                        error,
                        "重新预览。",
                    )
                })?;
                if *expected != actual {
                    return Err(Report::error(
                        job.operation,
                        "VARIANT_BASE_CONFLICT",
                        "变体与指定的基准不同。",
                        "本次操作未执行；重新诊断并核对变体。",
                    ));
                }
                if writes_variant
                    && expected == "absent"
                    && cli.seed.is_none()
                    && (cli.if_source_manifest.is_some() || cli.if_input_content.is_some())
                {
                    return Err(Report::error(
                        "edit",
                        "VARIANT_PREVIEW_SEED_REQUIRED",
                        "新变体的条件提交未指定预览时的初始化种子，重新生成的候选可能与预览不同。",
                        "使用预览报告中的 initialization_seed_decimal 作为 --seed；其他设备和用户随机源仍可能不同。",
                    ));
                }
            }
            if matches!(job.operation, "training_probe" | "train") {
                // The checked value is copied into the worker request; do not
                // hold the cooperative edit lock throughout a training run.
                drop(lock);
                None
            } else {
                Some(lock)
            }
        } else {
            None
        };
        let source_lock = if source_target && !job.dry_run {
            Some(self.source_edits.lock(root).map_err(|error| {
                Report::error("edit", "SOURCE_EDIT_LOCKED", error, "稍后重试源码编辑。")
            })?)
        } else {
            None
        };
        self.verify_model_inputs(job, root, &model_inputs, variant.as_ref())?;

        Ok(PreparedRunInputs {
            variant_name,
            variant_path,
            variant,
            model_inputs,
            batch,
            _variant_lock: variant_lock,
            _source_lock: source_lock,
        })
    }
}
