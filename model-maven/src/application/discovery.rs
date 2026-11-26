//! Bounded, read-only suggestions. This is intentionally not a Python parser.
use crate::domain::Report;
use serde_json::{Value, json};
use std::{collections::{HashSet, VecDeque}, fs, io::Read, path::{Component, Path, PathBuf}};

const MAX_DIRS: usize = 128;
const MAX_FILES: usize = 512;
const MAX_ENTRIES: usize = 2048;
const MAX_FILE_BYTES: u64 = 512 * 1024;
const MAX_CANDIDATES: usize = 512;

fn identifier(value: &str) -> bool {
    super::python_identifier(value)
}

fn module_name(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    let mut parts = Vec::new();
    for component in relative.parent()?.components() {
        let name = component.as_os_str().to_str()?;
        if !identifier(name) { return None; }
        parts.push(name.to_owned());
    }
    let stem = relative.file_stem()?.to_str()?;
    if stem != "__init__" {
        if !identifier(stem) { return None; }
        parts.push(stem.to_owned());
    }
    (!parts.is_empty()).then(|| parts.join("."))
}

fn kind(name: &str) -> Option<&'static str> {
    match name {
        "model" | "build_model" | "create_model" | "make_model" => Some("model"),
        "batch" | "build_batch" | "create_batch" | "sample_batch" => Some("batch"),
        "step" | "training_step" | "train_step" => Some("step"),
        "run_loop" | "training_loop" | "train_loop" => Some("loop"),
        // These are name hints only. The UI still labels every result as
        // unverified, and user selection never imports or executes it.
        _ if name.strip_prefix("build_").is_some_and(|rest| {
            rest.strip_suffix("_model").is_some_and(|part| !part.is_empty())
        }) => Some("model"),
        _ if name.strip_suffix("_batch").is_some_and(|part| !part.is_empty()) => Some("batch"),
        _ => None,
    }
}

pub(super) fn discover(import_roots: &[String]) -> Report {
    let root = match std::env::current_dir().and_then(|p| p.canonicalize()) {
        Ok(root) => root,
        Err(error) => return Report::error("project_discover", "PROJECT_UNAVAILABLE", error.to_string(), "检查项目目录。"),
    };
    if import_roots.len() > 32 {
        return Report::error("project_discover", "IMPORT_ROOTS_INVALID", "模块搜索目录最多 32 项。", "减少搜索目录后重试。");
    }
    let mut roots = Vec::new();
    for value in import_roots {
        let path = Path::new(value);
        if value.is_empty() || value.len() > 1024 || value.contains('\\') || value.contains('\0')
            || value.split('/').any(|part| part.is_empty() || part == "." || part == ".." || part == ".git" || part == "__pycache__")
            || path.is_absolute() || path.components().any(|part| !matches!(part, Component::Normal(_)))
            || path.starts_with(".model-maven/runs") {
            return Report::error("project_discover", "IMPORT_ROOTS_INVALID", "模块搜索目录无效或属于保留目录。", "选择项目内不经过符号链接的目录。");
        }
        let candidate = root.join(path);
        if candidate.canonicalize().ok().as_deref() != Some(candidate.as_path()) || !candidate.is_dir() {
            return Report::error("project_discover", "IMPORT_ROOT_UNAVAILABLE", format!("模块目录不可用：{value}"), "选择现有项目目录，且不能经过符号链接。");
        }
        if !roots.contains(&candidate) { roots.push(candidate); }
    }
    // Recognize common source-tree roots as static module-name alternatives.
    // These are suggestions only: selecting one here must never mutate the
    // project's runtime import paths, which are configured in project settings.
    let mut layout_roots = Vec::new();
    for name in ["src", "python", "lib"] {
        let candidate = root.join(name);
        if fs::symlink_metadata(&candidate).is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            && candidate.canonicalize().ok().as_deref() == Some(candidate.as_path())
        {
            layout_roots.push(candidate);
        }
    }
    let mut queue: VecDeque<PathBuf> = roots.iter().cloned().collect();
    queue.push_back(root.clone());
    let mut visited = HashSet::new();
    let mut candidates: Vec<Value> = Vec::new();
    let mut other_callables: Vec<Value> = Vec::new();
    let mut source_files: Vec<String> = Vec::new();
    let mut environment_candidates: Vec<String> = Vec::new();
    let mut directories = 0usize;
    let mut files = 0usize;
    let mut skipped = 0usize;
    let mut omitted_directories = Vec::new();
    let mut omitted_directory_count = 0usize;
    let active_environments: HashSet<PathBuf> = ["VIRTUAL_ENV", "CONDA_PREFIX"]
        .iter().filter_map(|name| std::env::var_os(*name))
        .filter_map(|path| PathBuf::from(path).canonicalize().ok()).collect();
    let mut excluded_roots: HashSet<PathBuf> = active_environments.iter()
        .filter(|path| path.starts_with(&root) && *path != &root).cloned().collect();
    let mut truncated = false;
    while let Some(dir) = queue.pop_front() {
        if !visited.insert(dir.clone()) { continue; }
        if directories >= MAX_DIRS || files >= MAX_FILES || candidates.len() >= MAX_CANDIDATES {
            truncated = true;
            break;
        }
        directories += 1;
        let iter = match fs::read_dir(&dir) {
            Ok(iter) => iter,
            Err(_) => { skipped += 1; continue; }
        };
        let mut entries: Vec<PathBuf> = Vec::new();
        for entry in iter {
            if entries.len() >= MAX_ENTRIES { truncated = true; break; }
            match entry { Ok(item) => entries.push(item.path()), Err(_) => skipped += 1 }
        }
        entries.sort();
        for path in entries {
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else { skipped += 1; continue; };
            let selected = roots.iter().any(|selected| path.starts_with(selected) || selected.starts_with(&path));
            let metadata = match fs::symlink_metadata(&path) { Ok(value) => value, Err(_) => { skipped += 1; continue; } };
            if metadata.file_type().is_symlink() { skipped += 1; continue; }
            let boundary = if metadata.is_dir() && (path.join("pyvenv.cfg").is_file() || active_environments.contains(&path)) {
                Some("python_environment")
            } else if metadata.is_dir() && name == "target" && path.join(".rustc_info.json").is_file() {
                Some("cargo_build_output")
            } else if metadata.is_dir() && name == "node_modules" {
                Some("package_dependencies")
            } else { None };
            if boundary.is_some() { excluded_roots.insert(path.clone()); }
            // Skip observed environments and tool caches, not arbitrary source
            // directories that happen to be named env, venv, target, or build.
            let omitted = if name == ".git" || name == "__pycache__" || path.starts_with(root.join(".model-maven/runs")) {
                Some("tool_metadata")
            } else if !selected && name.starts_with('.') {
                Some("hidden")
            } else if !selected && boundary.is_some() {
                boundary
            } else if !selected && path.ancestors().skip(1).any(|parent| excluded_roots.contains(parent)) {
                Some("dependency_boundary")
            } else { None };
            if let Some(reason) = omitted {
                skipped += 1;
                if metadata.is_dir() {
                    omitted_directory_count += 1;
                    if omitted_directories.len() < 32 {
                        omitted_directories.push(json!({"path":path.strip_prefix(&root).unwrap_or(&path).to_string_lossy(),"reason":reason}));
                    }
                }
                continue;
            }
            if metadata.is_dir() {
                if queue.len() >= MAX_DIRS { truncated = true; } else { queue.push_back(path); }
                continue;
            }
            if metadata.is_file() && path.parent() == Some(root.as_path())
                && matches!(path.extension().and_then(|s| s.to_str()), Some("toml" | "lock" | "yaml" | "yml" | "txt"))
                && environment_candidates.len() < MAX_FILES {
                environment_candidates.push(name.to_owned());
            }
            if !metadata.is_file() || path.extension().and_then(|s| s.to_str()) != Some("py") { continue; }
            if files >= MAX_FILES { truncated = true; break; }
            files += 1;
            if let Ok(relative) = path.strip_prefix(&root)
                && let Some(name) = relative.to_str() { source_files.push(name.replace('\\', "/")); }
            if metadata.len() > MAX_FILE_BYTES { skipped += 1; continue; }
            // The worker prepends explicitly selected import roots in user
            // order. Then offer conventional source-tree roots as static
            // alternatives before falling back to the project root. None of
            // these suggestions changes the runtime import-path selection.
            let module = roots.iter().find_map(|selected| module_name(selected, &path)
                .map(|module| (module, "explicit_import_root", selected)))
                .or_else(|| layout_roots.iter().find_map(|selected|
                module_name(selected, &path).map(|module| (module, "conventional_layout", selected))))
                .or_else(|| module_name(&root, &path).map(|module| (module, "project_root", &root)));
            let Some((module, module_basis, module_root)) = module else { skipped += 1; continue; };
            let mut module_alternatives = Vec::new();
            for (source_root, basis) in roots.iter().map(|path| (path, "explicit_import_root"))
                .chain(layout_roots.iter().map(|path| (path, "conventional_layout")))
                .chain(std::iter::once((&root, "project_root")))
            {
                let Some(alternative) = module_name(source_root, &path) else { continue; };
                let entrypoint = format!("{alternative}:");
                if module_alternatives.iter().any(|item: &Value| item["module"] == alternative) { continue; }
                module_alternatives.push(json!({"module":alternative,"basis":basis,
                    "import_root":source_root.strip_prefix(&root).unwrap_or(source_root).to_string_lossy(),
                    "runtime_path_configured":false,"entrypoint_prefix":entrypoint}));
            }
            let mut source = String::new();
            let read = fs::File::open(&path)
                .and_then(|file| file.take(MAX_FILE_BYTES + 1).read_to_string(&mut source));
            if read.is_err() || source.len() as u64 > MAX_FILE_BYTES { skipped += 1; continue; }
            for (index, line) in source.lines().enumerate() {
                // Only obvious top-level, single-line declarations are suggested.
                let (rest, declaration) = if let Some(rest) = line.strip_prefix("def ") {
                    (rest, "function")
                } else if let Some(rest) = line.strip_prefix("class ") {
                    (rest, "class")
                } else { continue; };
                let name = rest.split(['(', ':', ' ']).next().unwrap_or("");
                if !identifier(name) || name.starts_with('_') { continue; }
                let suggested_kind = if declaration == "function" { kind(name) } else { None };
                // This scan deliberately does not import a project module.
                // Keep that boundary on each item so consumers cannot treat a
                // syntactically plausible entrypoint as an adopted or checked
                // factory merely because it appears in the candidate list.
                let item = json!({"kind":suggested_kind.unwrap_or("callable"),"entrypoint":format!("{module}:{name}"),
                    "declaration":declaration,"file":path.strip_prefix(&root).unwrap_or(&path).to_string_lossy(),"line":index + 1,
                    "status":"static_candidate","imported":false,"verified":false,"role":"suggestion_only",
                    "module_basis":module_basis,"suggested_import_root":module_root.strip_prefix(&root).unwrap_or(module_root).to_string_lossy(),
                    "entrypoint_alternatives":module_alternatives.iter().map(|alternative| json!({
                        "entrypoint":format!("{}:{name}",alternative["module"].as_str().unwrap_or("")),
                        "basis":alternative["basis"],"import_root":alternative["import_root"],
                        "runtime_path_configured":false})).collect::<Vec<_>>()});
                if suggested_kind.is_some() {
                    candidates.push(item);
                } else if other_callables.len() < MAX_CANDIDATES {
                    other_callables.push(item);
                } else {
                    truncated = true;
                }
                if candidates.len() >= MAX_CANDIDATES { truncated = true; break; }
            }
        }
    }
    let remaining = MAX_CANDIDATES.saturating_sub(candidates.len());
    if other_callables.len() > remaining { truncated = true; }
    candidates.extend(other_callables.into_iter().take(remaining));
    source_files.sort(); source_files.dedup(); environment_candidates.sort(); environment_candidates.dedup();
    Report::ok("project_discover", json!({"candidates":candidates,"scanned_files":files,"source_files":source_files,"environment_candidates":environment_candidates,
        "skipped":skipped,"truncated":truncated,"omitted_directory_count":omitted_directory_count,
        "omitted_directories":omitted_directories,"executed_project_code":false,"prioritized_import_roots":import_roots,
        "candidate_status":"static_candidate","runtime_path_configuration":"not_performed",
        "suggested_import_roots":layout_roots.iter().map(|path| path.strip_prefix(&root).unwrap_or(path).to_string_lossy()).collect::<Vec<_>>(),
        "scope":"top-level functions and classes; roles and entrypoints are suggestions only; candidates are not imported or verified; imports, signatures, package paths and runtime behavior are not verified"}))
}
