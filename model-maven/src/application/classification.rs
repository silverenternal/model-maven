use super::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
};
use sha2::{Digest, Sha256};

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ClassificationRecipe {
    version: u32,
    kind: String,
    preset: String,
    data: String,
    classes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) training: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    supervision: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    runtime: Option<Value>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableRuntime {
    #[serde(default)]
    working_directory: Option<String>,
    #[serde(default)]
    project_import_roots: Vec<String>,
}

fn portable_runtime(value: &Value, root: Option<&Path>) -> Result<Value, String> {
    let runtime: PortableRuntime = serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
    if runtime.project_import_roots.len() > 32 { return Err("模块目录最多 32 项。".into()); }
    let mut seen = BTreeSet::new();
    for name in runtime.project_import_roots.iter().chain(runtime.working_directory.iter()) {
        if super::scaffold::validate_supervision_value(&json!({"file":name,"sha256":"0".repeat(64)})).is_err() {
            return Err(format!("运行路径必须是项目内相对目录：{name}"));
        }
        if let Some(root) = root {
            let path = root.join(name);
            if !path.canonicalize().is_ok_and(|resolved| resolved == path && resolved.is_dir()) {
                return Err(format!("运行目录不可用或经过符号链接：{name}"));
            }
        }
    }
    for name in &runtime.project_import_roots {
        if !seen.insert(name) { return Err("模块搜索目录不能重复。".into()); }
    }
    let directory = runtime.working_directory.as_ref().map(|name| match root {
        Some(root) => root.join(name).to_string_lossy().into_owned(),
        None => name.clone(),
    });
    Ok(json!({"working_directory":directory,"project_import_roots":runtime.project_import_roots}))
}

pub(super) fn read_classification_recipe(path: &Path) -> Result<ClassificationRecipe, String> {
    let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > 128 * 1024 { return Err("分类声明须为不超过 128 KiB 的普通文件。".into()); }
    let mut bytes = Vec::new();
    fs::File::open(path).map_err(|e| e.to_string())?.take(128 * 1024 + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() > 128 * 1024 { return Err("分类声明过大。".into()); }
    parse_classification_recipe(&bytes)
}

pub(super) fn parse_classification_recipe(bytes: &[u8]) -> Result<ClassificationRecipe, String> {
    let recipe: ClassificationRecipe = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if recipe.version != 1 || recipe.kind != "image_classification" || recipe.data.is_empty() || recipe.data.chars().any(char::is_control) {
        return Err("不支持的分类项目声明。".into());
    }
    if let Some(training) = &recipe.training { super::scaffold::validate_training_value(training)?; }
    if let Some(reference) = &recipe.supervision { super::scaffold::validate_supervision_value(reference)?; }
    if let Some(reference) = &recipe.source { super::scaffold::validate_supervision_value(reference)?; }
    if let Some(runtime) = &recipe.runtime { portable_runtime(runtime, None)?; }
    Ok(recipe)
}

fn classification_source_reference(root: &Path, reference: &Value) -> Result<String, String> {
    super::scaffold::validate_supervision_value(reference)?;
    let name = reference["file"].as_str().ok_or("源码快照路径无效。")?;
    if name == "classification.py" { return Err("源码快照不能与生成入口 classification.py 同名。".into()); }
    let path = root.join(name);
    let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
    if !metadata.is_file() || metadata.len() > 2 * 1024 * 1024 { return Err("源码快照须为不超过 2 MiB 的普通文件。".into()); }
    let resolved = path.canonicalize().map_err(|error| error.to_string())?;
    if !resolved.starts_with(root) { return Err("源码快照必须位于项目内。".into()); }
    let mut bytes = Vec::new();
    fs::File::open(path).map_err(|error| error.to_string())?.take(2 * 1024 * 1024 + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
    if bytes.len() > 2 * 1024 * 1024 || super::scaffold::sha256_bytes(&bytes) != reference["sha256"].as_str().unwrap_or("") {
        return Err("源码快照已变化或超过 2 MiB，请检查引用。".into());
    }
    String::from_utf8(bytes).map_err(|_| "源码快照须为 UTF-8。".into())
}

fn bundle_digest(path: &Path) -> Result<(u64, String), String> {
    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new(); let mut total = 0u64; let mut chunk = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut chunk).map_err(|error| error.to_string())?;
        if n == 0 { break; }
        total = total.checked_add(n as u64).ok_or("文件过大。")?;
        if total > 512 * 1024 * 1024 { return Err("单个文件超过 512 MiB。".into()); }
        hash.update(&chunk[..n]);
    }
    Ok((total, format!("{:x}", hash.finalize())))
}

impl<B: Backend> Application<B> {
    pub(super) fn bundle_classification_recipe(&self, graph: bool, scaffold: bool, output: &str, includes: &[String], apply: bool, if_plan: Option<&str>) -> Report {
        let fail = |code, message: String| Report::error("recipe_bundle", code, message, "检查项目文件并重新预览。 ");
        if output.is_empty() || output.len() > 128 || output == "." || output == ".."
            || !output.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) {
            return fail("BUNDLE_OUTPUT_INVALID", "输出必须是项目根目录下的新目录名。".into());
        }
        if includes.len() > 128 { return fail("BUNDLE_INPUT_INVALID", "最多选择 128 个文件或目录。".into()); }
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root, Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
        };
        let destination = root.join(output);
        match fs::symlink_metadata(&destination) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
            Ok(_) => return fail("BUNDLE_OUTPUT_EXISTS", "输出目录已存在，不能覆盖。".into()),
            Err(error) => return fail("BUNDLE_OUTPUT_INVALID", error.to_string()),
        }
        let mut declaration_revision = None;
        let mut scaffold_reference = None;
        let recipe = if graph {
            let path = root.join("model-graph.json");
            let bytes = match super::scaffold::read_recipe_for_update(&path) {
                Ok(bytes) => bytes, Err(error) => return fail("GRAPH_INVALID", error),
            };
            let revision = super::scaffold::sha256_bytes(&bytes);
            let checked = super::graph::plan(&path, &revision);
            if checked.status != "success" { return fail("GRAPH_INVALID", "图声明无效或在读取时变化，请先检查图。".into()); }
            scaffold_reference = match super::graph::bundle_reference(&bytes) {
                Ok(reference) => reference, Err(error) => return fail("GRAPH_INVALID", error),
            };
            declaration_revision = Some(revision);
            None
        } else if scaffold {
            let path = root.join("model-maven.json");
            let bytes = match super::scaffold::read_recipe_for_update(&path) {
                Ok(bytes) => bytes, Err(error) => return fail("RECIPE_INVALID", error),
            };
            scaffold_reference = match super::scaffold::bundle_reference(&bytes) {
                Ok(reference) => reference, Err(error) => return fail("RECIPE_INVALID", error),
            };
            declaration_revision = Some(super::scaffold::sha256_bytes(&bytes));
            None
        } else {
            Some(match read_classification_recipe(&root.join("model-maven.json")) {
                Ok(recipe) => recipe, Err(error) => return fail("RECIPE_INVALID", error),
            })
        };
        let mut selected = vec![if graph { "model-graph.json" } else { "model-maven.json" }.to_owned()];
        if let Some(recipe) = &recipe {
            for reference in [&recipe.source, &recipe.supervision].into_iter().flatten() {
                selected.push(reference["file"].as_str().unwrap_or("").to_owned());
            }
        }
        if let Some(reference) = &scaffold_reference {
            selected.push(reference["file"].as_str().unwrap_or("").to_owned());
        }
        selected.extend(includes.iter().cloned());
        let mut pending = selected;
        let mut files = BTreeMap::<String, (u64, String)>::new();
        let mut dirs = BTreeSet::<String>::new();
        let mut total = 0u64;
        while let Some(name) = pending.pop() {
            if name == output || name.starts_with(&format!("{output}/")) || (recipe.is_some() && name == "classification.py") {
                return fail("BUNDLE_INPUT_INVALID", format!("不能包含输出目录或重建目标 classification.py：{name}。请先创建源码快照。"));
            }
            if super::scaffold::validate_supervision_value(&json!({"file":name,"sha256":"0".repeat(64)})).is_err() {
                return fail("BUNDLE_INPUT_INVALID", format!("路径必须是项目内的相对路径：{name}"));
            }
            let mut path = root.clone();
            for part in Path::new(&name).components() {
                path.push(part.as_os_str());
                let metadata = match fs::symlink_metadata(&path) { Ok(value) => value, Err(error) => return fail("BUNDLE_INPUT_INVALID", error.to_string()) };
                if metadata.file_type().is_symlink() { return fail("BUNDLE_INPUT_INVALID", format!("不能包含符号链接：{name}")); }
            }
            let metadata = match fs::metadata(&path) { Ok(value) => value, Err(error) => return fail("BUNDLE_INPUT_INVALID", error.to_string()) };
            if metadata.is_dir() {
                if !dirs.insert(name.clone()) { continue; }
                let entries = match fs::read_dir(&path) { Ok(value) => value, Err(error) => return fail("BUNDLE_INPUT_INVALID", error.to_string()) };
                for entry in entries {
                    let entry = match entry { Ok(value) => value, Err(error) => return fail("BUNDLE_INPUT_INVALID", error.to_string()) };
                    let Some(child) = entry.file_name().to_str().map(str::to_owned) else { return fail("BUNDLE_INPUT_INVALID", "文件名必须是 UTF-8。".into()); };
                    pending.push(format!("{name}/{child}"));
                    if pending.len() + files.len() + dirs.len() > 10_000 { return fail("BUNDLE_TOO_LARGE", "文件包最多包含 10000 项。".into()); }
                }
            } else if metadata.is_file() {
                if files.contains_key(&name) { continue; }
                if metadata.len() > 512 * 1024 * 1024 { return fail("BUNDLE_TOO_LARGE", format!("单个文件超过 512 MiB：{name}")); }
                total = match total.checked_add(metadata.len()) { Some(value) if value <= 2 * 1024 * 1024 * 1024 => value, _ => return fail("BUNDLE_TOO_LARGE", "文件包超过 2 GiB。".into()) };
                let (size, digest) = match bundle_digest(&path) { Ok(value) => value, Err(error) => return fail("BUNDLE_INPUT_INVALID", error) };
                if size != metadata.len() { return fail("BUNDLE_INPUT_CHANGED", format!("文件在读取时变化：{name}")); }
                files.insert(name, (size, digest));
            } else { return fail("BUNDLE_INPUT_INVALID", format!("不是普通文件或目录：{name}")); }
        }
        if let Some(expected) = declaration_revision {
            let declaration = if graph { "model-graph.json" } else { "model-maven.json" };
            if files.get(declaration).is_none_or(|(_, actual)| actual != &expected) {
                return fail("BUNDLE_INPUT_CHANGED", "项目声明在检查期间变化，请重新预览。".into());
            }
        }
        if let Some(reference) = &scaffold_reference {
            let name = reference["file"].as_str().unwrap_or("");
            if files.get(name).is_none_or(|(_, hash)| hash != reference["sha256"].as_str().unwrap_or("")) {
                return fail("BUNDLE_INPUT_CHANGED", format!("监督引用文件与声明不一致：{name}"));
            }
        }
        if let Some(recipe) = &recipe {
            for reference in [&recipe.source, &recipe.supervision].into_iter().flatten() {
                let name = reference["file"].as_str().unwrap_or("");
                if files.get(name).is_none_or(|(_, hash)| hash != reference["sha256"].as_str().unwrap_or("")) {
                    return fail("BUNDLE_INPUT_CHANGED", format!("引用文件与声明不一致：{name}"));
                }
            }
        }
        let mut missing_runtime = Vec::<String>::new();
        if let Some(runtime) = recipe.as_ref().and_then(|recipe| recipe.runtime.as_ref()) {
            let parsed: PortableRuntime = match serde_json::from_value(runtime.clone()) {
                Ok(value) => value, Err(error) => return fail("RECIPE_INVALID", error.to_string()),
            };
            for path in parsed.project_import_roots.iter().chain(parsed.working_directory.iter()) {
                let prefix = format!("{path}/");
                if !dirs.contains(path) && !dirs.iter().any(|entry| entry.starts_with(&prefix))
                    && !files.keys().any(|entry| entry.starts_with(&prefix)) && !missing_runtime.contains(path) {
                    missing_runtime.push(path.clone());
                }
            }
        }
        // Store only portable runtime hints, never the project configuration or
        // interpreter path: those may contain machine-specific values or secrets.
        let runtime_hint = if graph || scaffold {
            let runtime = match self.project_config.load(&root) {
                Ok(Some(config)) => {
                    let paths = &config["runtime_paths"];
                    let directory = paths["working_directory"].as_str();
                    let working = match directory {
                        None => json!({"kind":"default_project"}),
                        Some(name) => match Path::new(name).strip_prefix(&root) {
                            Ok(relative) if relative.as_os_str().is_empty() => json!({"kind":"project_root"}),
                            Ok(relative) => {
                                let path = relative.to_string_lossy().replace('\\', "/");
                                let prefix = format!("{path}/");
                                if !dirs.contains(&path) && !files.keys().any(|entry| entry.starts_with(&prefix))
                                    && !missing_runtime.contains(&path) { missing_runtime.push(path.clone()); }
                                json!({"kind":"project_relative","path":path})
                            },
                            Err(_) => json!({"kind":"external","path_included":false}),
                        },
                    };
                    let roots = paths["project_import_roots"].as_array().cloned().unwrap_or_default();
                    for value in &roots {
                        if let Some(path) = value.as_str() {
                            let prefix = format!("{path}/");
                            if !dirs.contains(path) && !files.keys().any(|entry| entry.starts_with(&prefix))
                                && !missing_runtime.iter().any(|item| item == path) {
                                missing_runtime.push(path.to_owned());
                            }
                        }
                    }
                    json!({"status":"recorded","working_directory":working,"project_import_roots":roots})
                },
                Ok(None) => json!({"status":"project_unconfigured"}),
                Err(_) => json!({"status":"unavailable"}),
            };
            Some(json!({"format":"model-maven-transfer-runtime-v1","runtime_paths":runtime,
                "note":"Runtime hints only; select an interpreter and restore paths explicitly. External resources and dependencies are not included automatically."}))
        } else { None };
        let hint_name = "model-maven-transfer.json";
        if runtime_hint.is_some() && (files.contains_key(hint_name) || dirs.contains(hint_name)) {
            return fail("BUNDLE_INPUT_INVALID", format!("{hint_name} 是生成的迁移提示文件，请勿另外选择同名文件。"));
        }
        let hint_bytes = match runtime_hint.as_ref().map(serde_json::to_vec_pretty).transpose() {
            Ok(bytes) => bytes, Err(error) => return fail("BUNDLE_PLAN_FAILED", error.to_string()),
        };
        let generated_files = hint_bytes.as_ref().map(|bytes| json!({(hint_name):[bytes.len(), super::scaffold::sha256_bytes(bytes)]})).unwrap_or_else(|| json!({}));
        let package_bytes = total + hint_bytes.as_ref().map_or(0, |bytes| bytes.len() as u64);
        let data_directory_selected = recipe.as_ref().is_some_and(|recipe| !Path::new(&recipe.data).is_absolute() && includes.iter().any(|entry| {
            entry == &recipe.data || recipe.data.starts_with(&format!("{entry}/"))
        }));
        // Filename heuristics only: candidates are presented for review, not
        // treated as an exhaustive secret scanner and never opened for content.
        let sensitive_candidates = files.keys().filter(|path| {
            let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
            name == ".env" || name.starts_with(".env.") || name == "id_rsa" || name == "id_ed25519"
                || name.ends_with(".pem") || name.ends_with(".p12") || name.ends_with(".pfx")
                || name.contains("credentials") || name.contains("secret")
        }).cloned().collect::<Vec<_>>();
        let plan = json!({"kind":if graph {"graph"} else if scaffold {"scaffold"} else {"classification"},"project_root":root,"output":output,"directories":dirs,"files":files,"total_bytes":package_bytes,
            "data_path":recipe.as_ref().map(|recipe| &recipe.data),"includes":includes,"has_source_snapshot":recipe.as_ref().is_some_and(|recipe| recipe.source.is_some()),
            "missing_runtime_directories":missing_runtime,"data_directory_selected":data_directory_selected,
            "runtime_hint":runtime_hint,"generated_files":generated_files,"potentially_sensitive_paths":sensitive_candidates,
            "note":if graph || scaffold {"仅复制声明、监督引用与明确选择的文件；动态导入、外部资源、凭据和环境是否完整未知，不会安装依赖或运行代码。"} else {"仅复制列出的文件；请显式选择数据和环境文件。不会安装依赖或运行代码。"}});
        let revision = match crate::domain::input::content_revision(&plan) { Ok(value) => value, Err(error) => return fail("BUNDLE_PLAN_FAILED", error) };
        if !apply { return Report::ok("recipe_bundle", json!({"committed":false,"plan_revision":revision,"plan":plan})); }
        if if_plan != Some(revision.as_str()) { return fail("BUNDLE_PLAN_CONFLICT", "文件或计划已变化，请重新预览。".into()); }
        if let Err(error) = fs::create_dir(&destination) { return fail("BUNDLE_OUTPUT_EXISTS", error.to_string()); }
        let copy = (|| -> Result<(), String> {
            for dir in &dirs { fs::create_dir_all(destination.join(dir)).map_err(|error| error.to_string())?; }
            for (name, (size, expected)) in &files {
                let target = destination.join(name);
                if let Some(parent) = target.parent() { fs::create_dir_all(parent).map_err(|error| error.to_string())?; }
                let mut source = fs::File::open(root.join(name)).map_err(|error| error.to_string())?;
                let mut target = OpenOptions::new().write(true).create_new(true).open(target).map_err(|error| error.to_string())?;
                let mut hash = Sha256::new(); let mut copied = 0u64; let mut chunk = [0u8; 64 * 1024];
                loop {
                    let n = source.read(&mut chunk).map_err(|error| error.to_string())?;
                    if n == 0 { break; }
                    copied = copied.checked_add(n as u64).ok_or("文件过大。")?;
                    if copied > *size { return Err(format!("文件在复制期间变化：{name}")); }
                    hash.update(&chunk[..n]); target.write_all(&chunk[..n]).map_err(|error| error.to_string())?;
                }
                if copied != *size || format!("{:x}", hash.finalize()) != *expected { return Err(format!("文件在复制期间变化：{name}")); }
                target.sync_all().map_err(|error| error.to_string())?;
            }
            if let Some(bytes) = &hint_bytes {
                let mut target = OpenOptions::new().write(true).create_new(true).open(destination.join(hint_name)).map_err(|error| error.to_string())?;
                target.write_all(bytes).map_err(|error| error.to_string())?;
                target.sync_all().map_err(|error| error.to_string())?;
            }
            Ok(())
        })();
        if let Err(error) = copy { return fail("BUNDLE_INCOMPLETE", format!("{error}；不完整的输出保留在 {}，请检查后手动处理。", destination.display())); }
        Report::ok("recipe_bundle", json!({"committed":true,"plan_revision":revision,"output":destination,"file_count":files.len() + usize::from(hint_bytes.is_some()),"total_bytes":package_bytes}))
    }

    pub(super) fn update_classification_recipe(&self, training: Option<&str>, supervision: Option<&str>, source: Option<&str>, runtime: Option<&str>,
        expected: &str, apply: bool, if_plan: Option<&str>) -> Report {
        let fail = |code, message: String| Report::error("recipe_update", code, message, "检查声明并重新预览。 ");
        if training.is_none() && supervision.is_none() && source.is_none() && runtime.is_none() { return fail("RECIPE_UPDATE_EMPTY", "请选择要修改的声明。".into()); }
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root, Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
        };
        let path = root.join("model-maven.json");
        let bytes = match super::scaffold::read_recipe_for_update(&path) {
            Ok(bytes) => bytes, Err(error) => return fail("RECIPE_INVALID", error),
        };
        if super::scaffold::sha256_bytes(&bytes) != expected { return fail("RECIPE_CONFLICT", "声明已变化。".into()); }
        let mut recipe = match read_classification_recipe(&path) {
            Ok(recipe) => recipe, Err(error) => return fail("RECIPE_INVALID", error),
        };
        let previous = match serde_json::to_value(&recipe) { Ok(value) => value, Err(error) => return fail("RECIPE_INVALID", error.to_string()) };
        for (raw, slot, label) in [(training, &mut recipe.training, "training"), (supervision, &mut recipe.supervision, "supervision"), (source, &mut recipe.source, "source"), (runtime, &mut recipe.runtime, "runtime")] {
            if let Some(raw) = raw {
                if raw.len() > 65_536 { return fail("RECIPE_INVALID", format!("{label} 过大。")); }
                let value: Value = match serde_json::from_str(raw) { Ok(value) => value, Err(error) => return fail("RECIPE_INVALID", error.to_string()) };
                if value.is_null() { *slot = None; continue; }
                let valid = match label {
                    "training" => super::scaffold::validate_training_value(&value),
                    "runtime" => portable_runtime(&value, Some(&root)).map(|_| ()),
                    _ => super::scaffold::validate_supervision_value(&value),
                };
                if let Err(error) = valid { return fail("RECIPE_INVALID", error); }
                *slot = Some(if label == "runtime" { portable_runtime(&value, None).expect("runtime already validated") } else { value });
            }
        }
        let next = match serde_json::to_value(&recipe) { Ok(value) => value, Err(error) => return fail("RECIPE_INVALID", error.to_string()) };
        let plan = json!({"project_root":root,"recipe_revision":expected,"previous":previous,"recipe":next});
        let revision = match crate::domain::input::content_revision(&plan) { Ok(value) => value, Err(error) => return fail("RECIPE_PLAN_FAILED", error) };
        if !apply { return Report::ok("recipe_update", json!({"committed":false,"plan_revision":revision,"plan":plan})); }
        if if_plan != Some(revision.as_str()) { return fail("RECIPE_PLAN_CONFLICT", "声明计划已变化，请重新预览。".into()); }
        let contents = match serde_json::to_string_pretty(&recipe) { Ok(value) => format!("{value}\n"), Err(error) => return fail("RECIPE_INVALID", error.to_string()) };
        if contents.len() > 128 * 1024 { return fail("RECIPE_INVALID", "声明不能超过 128 KiB。".into()); }
        if let Err(error) = super::scaffold::replace_recipe(&path, &bytes, contents.as_bytes()) { return fail("RECIPE_CONFLICT", error); }
        Report::ok("recipe_update", json!({"committed":true,"plan_revision":revision,"recipe_revision":super::scaffold::sha256_bytes(contents.as_bytes())}))
    }

    pub(super) fn create_classification(
        &self,
        cli: &Cli,
        preset: Option<&str>,
        data: Option<&Path>,
        recipe_path: Option<&Path>,
        apply: bool,
        if_plan: Option<&str>,
    ) -> Report {
        let fail = |code, message: String| {
            Report::error(
                "create",
                code,
                message,
                "检查目录和运行环境，然后重新预览。 ",
            )
        };
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
        };
        let declared = match recipe_path.map(read_classification_recipe).transpose() {
            Ok(value) => value,
            Err(error) => return fail("CLASSIFICATION_RECIPE_INVALID", error),
        };
        let preset = declared.as_ref().map(|r| r.preset.as_str()).or(preset).unwrap_or("");
        let Some(data) = declared.as_ref().map(|r| Path::new(&r.data)).or(data) else {
            return fail("CLASSIFICATION_RECIPE_INVALID", "需要数据目录。".into());
        };
        let data_root = match root.join(data).canonicalize() {
            Ok(path) if path.is_dir() => path,
            Ok(_) => return fail("DATA_DIRECTORY_INVALID", "数据路径不是目录。".into()),
            Err(error) => return fail("DATA_DIRECTORY_INVALID", error.to_string()),
        };
        let classes = match classification_classes(&data_root) {
            Ok(classes) => classes,
            Err(error) => return fail("CLASSIFICATION_DATA_INVALID", error),
        };
        if declared.as_ref().is_some_and(|recipe| recipe.classes != classes) {
            return fail("CLASSIFICATION_CLASSES_CHANGED", "数据类别或顺序与声明不一致；请检查数据目录，避免标签改变。".into());
        }
        if let Err(error) = validate_classification_preset_targets(&root) {
            return fail("CREATE_TARGET_CONFLICT", error);
        }
        let model = "classification:model".to_owned();
        let batch = "classification:batch".to_owned();
        let init = |dry_run| {
            self.initialize_project(InitOptions {
                model: Some(&model),
                batch: Some(&batch),
                injections: &[],
                injection_params: &[],
                batch_binding: Some(&BatchBinding::Envelope),
                batch_params: &[],
                dry_run,
            })
        };
        let init_preview = init(true);
        if !init_preview.success() {
            return init_preview;
        }
        if preset == "resnet18" {
            let check = self.run(
                cli,
                &Command::Model {
                    command: ModelCommand::Inspect {
                        path: None,
                        inputs: Inputs {
                            model: Some("model_maven_worker.data.presets:build_resnet18".into()),
                            model_params: vec!["num_classes=2".into(), "weights=\"random\"".into()],
                            ..Inputs::default()
                        },
                    },
                },
            );
            if !check.success() {
                return check;
            }
        }
        let data_value = data_root.strip_prefix(&root).unwrap_or(&data_root);
        let Some(data_text) = data_value.to_str() else {
            return fail("DATA_DIRECTORY_INVALID", "数据路径不是 UTF-8。".into());
        };
        let recipe = ClassificationRecipe { version: 1, kind: "image_classification".into(), preset: preset.into(),
            data: data_text.replace('\\', "/"), classes: classes.clone(),
            training: declared.as_ref().and_then(|recipe| recipe.training.clone()),
            supervision: declared.as_ref().and_then(|recipe| recipe.supervision.clone()),
            source: declared.as_ref().and_then(|recipe| recipe.source.clone()),
            runtime: declared.as_ref().and_then(|recipe| recipe.runtime.clone()) };
        let recipe_path = root.join("model-maven.json");
        let recipe_exists = match fs::symlink_metadata(&recipe_path) {
            Ok(_) => match read_classification_recipe(&recipe_path) {
                Ok(existing) if existing == recipe => true,
                Ok(_) => return fail("CREATE_TARGET_CONFLICT", "model-maven.json 已存在且与此分类项目不同。".into()),
                Err(error) => return fail("CREATE_TARGET_CONFLICT", error),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return fail("CREATE_TARGET_CONFLICT", error.to_string()),
        };
        let recipe_bytes = match serde_json::to_string_pretty(&recipe) {
            Ok(value) => format!("{value}\n").into_bytes(),
            Err(error) => return fail("CREATE_PLAN_FAILED", error.to_string()),
        };
        let generated = match classification_preset_source(
            preset,
            classes.len(),
            &data_text.replace('\\', "/"),
        ) {
            Ok(source) => source,
            Err(error) => return fail("PRESET_INVALID", error),
        };
        let source = match &recipe.source {
            Some(reference) => match classification_source_reference(&root, reference) {
                Ok(source) => source,
                Err(error) => return fail("CLASSIFICATION_SOURCE_INVALID", error),
            },
            None => generated,
        };
        let mut planned_config = init_preview.data["config"].clone();
        if let Some(runtime) = &recipe.runtime {
            match portable_runtime(runtime, Some(&root)) {
                Ok(paths) => planned_config["runtime_paths"] = paths,
                Err(error) => return fail("CLASSIFICATION_RUNTIME_INVALID", error),
            }
        }
        let plan = json!({"version":1,"project_root":root,"data_root":data_root,"classes":classes,
            "preset":preset,"source":source,"recipe":recipe,"recipe_exists":recipe_exists,"config":planned_config,
            "files":["classification.py", ".model-maven/project.json", "model-maven.json"]});
        let revision = match crate::domain::input::content_revision(&plan) {
            Ok(revision) => revision,
            Err(error) => return fail("CREATE_PLAN_FAILED", error),
        };
        if !apply {
            return Report::ok(
                "create",
                json!({"committed":false,"plan":plan,"plan_revision":revision,
                "scope":"类名和文件存在性检查；不验证图像内容，也不固定数据内容。Tiny CNN 环境需在前向时验证。"}),
            );
        }
        if if_plan != Some(revision.as_str()) {
            return fail(
                "CREATE_PLAN_CONFLICT",
                "请使用当前预览的 plan_revision 确认；数据类别或计划发生变化时需重新预览。".into(),
            );
        }
        let created =
            match create_classification_preset_files(&root, &data_root, preset, classes.len(), Some(&source)) {
                Ok(source) => source,
                Err(error) => return fail("CREATE_SOURCE_FAILED", error),
            };
        if !recipe_exists {
            let result = OpenOptions::new().write(true).create_new(true).open(&recipe_path)
                .and_then(|mut file| file.write_all(&recipe_bytes).and_then(|()| file.sync_all()));
            if let Err(error) = result {
                if fs::read(root.join("classification.py")).is_ok_and(|bytes| bytes == created.as_bytes()) { let _ = fs::remove_file(root.join("classification.py")); }
                if fs::read(&recipe_path).is_ok_and(|bytes| bytes == recipe_bytes) { let _ = fs::remove_file(&recipe_path); }
                return fail("CREATE_RECIPE_FAILED", error.to_string());
            }
        }
        if let Err(error) = self.project_config.create(&root, &planned_config) {
            match self.project_config.load(&root) {
                Ok(Some(actual)) if actual == planned_config => {},
                Ok(None) => {
                    let path = root.join("classification.py");
                    if std::fs::read(&path).is_ok_and(|current| current == created.as_bytes()) { let _ = std::fs::remove_file(path); }
                    if !recipe_exists && fs::read(&recipe_path).is_ok_and(|bytes| bytes == recipe_bytes) { let _ = fs::remove_file(&recipe_path); }
                    return fail("CREATE_CONFIG_FAILED", error);
                },
                _ => return fail("CREATE_CONFIG_OUTCOME_UNKNOWN", format!("{error}；项目配置状态无法确认，已保留源码和声明以供检查。")),
            }
        }
        Report::ok(
            "create",
            json!({"committed":true,"plan_revision":revision,
            "files":plan["files"],"model":model,"batch":batch,"classes":classes,
            "note":"已创建可编辑的项目文件；尚未执行前向。"}),
        )
    }
}

pub(crate) fn classification_classes(root: &Path) -> Result<Vec<String>, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("无法读取数据目录：{error}"))?;
    if !root.is_dir() {
        return Err("数据路径不是目录。".into());
    }
    let entries = fs::read_dir(&root).map_err(|error| format!("无法列出数据目录：{error}"))?;
    let mut classes = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("读取数据目录条目失败：{error}"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "类别目录名必须是有效 UTF-8 文本。".to_owned())?;
        let file_type = entry
            .file_type()
            .map_err(|error| format!("无法检查数据目录条目 `{name}`：{error}"))?;
        if !name.starts_with('.') && file_type.is_dir() {
            if !classification_class_has_images(&entry.path())? {
                return Err(format!(
                    "类别 `{name}` 中没有受支持的图像文件（jpg/jpeg/png/bmp/tif/tiff/webp）。"
                ));
            }
            classes.push(name);
        }
    }
    classes.sort_by_key(|name| name.to_lowercase());
    if classes.len() < 2 {
        return Err(format!(
            "需要至少两个类别子目录；在 {} 中只找到 {} 个。每类图片应放在一个子目录里。",
            root.display(),
            classes.len()
        ));
    }
    Ok(classes)
}

fn classification_class_has_images(root: &Path) -> Result<bool, String> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory)
            .map_err(|error| format!("无法读取类别目录 {}：{error}", directory.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("读取类别目录条目失败：{error}"))?;
            let file_type = entry
                .file_type()
                .map_err(|error| format!("无法检查图像条目：{error}"))?;
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file()
                && entry
                    .path()
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        matches!(
                            extension.to_ascii_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "bmp" | "tif" | "tiff" | "webp"
                        )
                    })
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub(crate) fn classification_preset_source(
    preset: &str,
    num_classes: usize,
    data_root: &str,
) -> Result<String, String> {
    let data_root =
        serde_json::to_string(data_root).map_err(|error| format!("无法编码数据目录：{error}"))?;
    let (preset_name, builder, model_args, transforms) = match preset {
        "tiny-cnn" => (
            "Tiny CNN",
            "build_tiny_cnn",
            "",
            r#"[{"type": "Resize", "size": 32}, {"type": "ToTensor"}]"#,
        ),
        "resnet18" => (
            "ResNet18 (random initialization; no weight download)",
            "build_resnet18",
            ", weights=\"random\"",
            r#"[{"type": "Resize", "size": 224}, {"type": "ToTensor"}, {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}]"#,
        ),
        _ => return Err(format!("未知分类预设：{preset}")),
    };
    Ok(format!(
        r#"# Editable {preset_name} image-classification starter created by Model-Maven.

# Preset: {preset_name}
# Change NUM_CLASSES, DATA_ROOT, or the model/batch functions as your project grows.
# Images are read in place from DATA_ROOT/<class-name>; they are never moved.
# ResNet18 uses random weights. Preparing pretrained weights is a separate explicit operation.

from model_maven_worker.data.presets import (
    {builder},
    classification_batch as _classification_batch,
    classification_step,
)

NUM_CLASSES = {num_classes}
DATA_ROOT = {data_root}


def model():
    return {builder}(num_classes=NUM_CLASSES{model_args})


def batch():
    classification = _classification_batch(
        root=DATA_ROOT,
        batch_size=8,
        split="train",
        split_ratios={{"train": 1.0}},
        transforms={transforms},
    )
    return {{
        "args": [classification["inputs"]],
        "kwargs": {{}},
        "loss_inputs": {{
            "targets": classification["targets"],
            "dataset": classification["dataset"],
        }},
    }}


def step(model, batch, context):
    return classification_step(model, batch, context)


batch.state_dict = _classification_batch.state_dict
batch.load_state_dict = _classification_batch.load_state_dict
"#
    ))
}

pub(crate) fn validate_classification_preset_targets(root: &Path) -> Result<(), String> {
    let source_path = root.join("classification.py");
    let metadata_path = root.join(".model-maven");
    match fs::symlink_metadata(&metadata_path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(".model-maven 必须是普通项目目录；为避免写入链接目标，预设已停止。".into());
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("无法检查 .model-maven 目录：{error}")),
    }
    let config_path = metadata_path.join("project.json");
    for (label, path) in [("项目配置", &config_path), ("分类入口", &source_path)] {
        match fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(format!(
                    "{} 已存在，预设不会覆盖它。可先备份或选择其他项目目录。",
                    path.strip_prefix(root).unwrap_or(path).display()
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("无法检查{label}路径：{error}")),
        }
    }
    Ok(())
}

pub(crate) fn create_classification_preset_files(
    root: &Path,
    data_root: &Path,
    preset: &str,
    num_classes: usize,
    source_override: Option<&str>,
) -> Result<String, String> {
    validate_classification_preset_targets(root)?;
    let source_path = root.join("classification.py");

    let data_value = match data_root.strip_prefix(root) {
        Ok(relative) => relative
            .to_str()
            .ok_or_else(|| "数据目录相对路径必须是有效 UTF-8 文本。".to_owned())?
            .replace('\\', "/"),
        Err(_) => data_root
            .to_str()
            .ok_or_else(|| "数据目录路径必须是有效 UTF-8 文本。".to_owned())?
            .to_owned(),
    };
    let source = match source_override { Some(source) => source.to_owned(), None => classification_preset_source(preset, num_classes, &data_value)? };
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&source_path)
        .map_err(|error| format!("无法安全创建 classification.py：{error}"))?;
    if let Err(error) = file
        .write_all(source.as_bytes())
        .and_then(|()| file.sync_all())
    {
        let _ = fs::remove_file(&source_path);
        return Err(format!("写入分类入口失败：{error}"));
    }
    Ok(source)
}
