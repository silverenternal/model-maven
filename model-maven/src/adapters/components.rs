use crate::ports::ComponentResolver;
use fs2::FileExt;
use semver::{Version, VersionReq};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, DirBuilder, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

const MANIFEST_LIMIT: u64 = 256 * 1024;
const PARAM_DEPTH_LIMIT: usize = 16;
const DEPENDENCY_RESOLUTION_LIMIT: usize = 4096;

#[derive(Clone, Default)]
struct DependencySolveState {
    selected: BTreeMap<String, Value>,
    constraints: BTreeMap<String, Vec<(VersionReq, String)>>,
}

pub struct FileComponentRegistry;

fn version_tuple(value: &str) -> (u64, u64, u64) {
    let mut parts = value
        .split('.')
        .map(|part| part.parse::<u64>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

impl FileComponentRegistry {
    fn component_id(id: &str) -> Result<(&str, &str), String> {
        let Some((namespace, name)) = id.split_once('/') else {
            return Err("组件 ID 必须使用 namespace/name 格式。".into());
        };
        let valid_segment = |segment: &str| {
            !segment.is_empty()
                && segment.len() <= 64
                && segment != "."
                && segment != ".."
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        };
        if !valid_segment(namespace) || !valid_segment(name) || name.contains('/') {
            return Err(
                "组件 ID 只能包含一个斜线，namespace/name 两段各限字母、数字、-、_。".into(),
            );
        }
        Ok((namespace, name))
    }

    fn requested_component(value: &str) -> Result<(&str, &str, &str, Option<&str>), String> {
        let (id, version) = match value.split_once('@') {
            Some((id, version)) if !version.is_empty() && !version.contains('@') => {
                if !Self::valid_version(version) {
                    return Err("显式组件版本使用 MAJOR.MINOR.PATCH 数字格式。".into());
                }
                (id, Some(version))
            }
            Some(_) => return Err("组件版本引用应为 namespace/name@MAJOR.MINOR.PATCH。".into()),
            None => (value, None),
        };
        let (namespace, name) = Self::component_id(id)?;
        Ok((namespace, name, id, version))
    }

    fn ensure_directory(path: &Path, create: bool) -> Result<bool, String> {
        match fs::symlink_metadata(path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                Err(format!("拒绝通过符号链接访问组件目录：{}", path.display()))
            }
            Ok(meta) if !meta.is_dir() => Err(format!("组件路径不是目录：{}", path.display())),
            Ok(_) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && !create => Ok(false),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let mut builder = DirBuilder::new();
                #[cfg(unix)]
                {
                    use std::os::unix::fs::DirBuilderExt;
                    builder.mode(0o700);
                }
                builder.create(path).map_err(|error| error.to_string())?;
                Ok(true)
            }
            Err(error) => Err(error.to_string()),
        }
    }

    fn root(root: &Path, create: bool) -> Result<Option<PathBuf>, String> {
        let metadata = root.join(".model-maven");
        if !Self::ensure_directory(&metadata, create)? {
            return Ok(None);
        }
        let components = metadata.join("components");
        if !Self::ensure_directory(&components, create)? {
            return Ok(None);
        }
        Ok(Some(components))
    }

    fn path(root: &Path, id: &str, version: &str, create_parent: bool) -> Result<PathBuf, String> {
        let (namespace, name) = Self::component_id(id)?;
        let components =
            Self::root(root, create_parent)?.ok_or_else(|| "项目尚未注册组件。".to_owned())?;
        let namespace_dir = components.join(namespace);
        if !Self::ensure_directory(&namespace_dir, create_parent)? {
            return Err(format!("找不到组件：{id}"));
        }
        let component_dir = namespace_dir.join(name);
        if !Self::ensure_directory(&component_dir, create_parent)? {
            return Err(format!("找不到组件：{id}"));
        }
        Ok(component_dir.join(format!("{version}.json")))
    }

    fn component_directory(root: &Path, id: &str) -> Result<Option<PathBuf>, String> {
        let (namespace, name) = Self::component_id(id)?;
        let Some(components) = Self::root(root, false)? else {
            return Ok(None);
        };
        let namespace_dir = components.join(namespace);
        if !Self::ensure_directory(&namespace_dir, false)? {
            return Ok(None);
        }
        let component_dir = namespace_dir.join(name);
        if !Self::ensure_directory(&component_dir, false)? {
            return Ok(None);
        }
        Ok(Some(component_dir))
    }

    fn verification_path(
        root: &Path,
        id: &str,
        version: &str,
        create: bool,
    ) -> Result<Option<PathBuf>, String> {
        if !Self::valid_version(version) {
            return Err("组件验证记录版本格式无效。".into());
        }
        let (namespace, name) = Self::component_id(id)?;
        let metadata = root.join(".model-maven");
        if !Self::ensure_directory(&metadata, create)? {
            return Ok(None);
        }
        let verifications = metadata.join("component-verifications");
        if !Self::ensure_directory(&verifications, create)? {
            return Ok(None);
        }
        let namespace_dir = verifications.join(namespace);
        if !Self::ensure_directory(&namespace_dir, create)? {
            return Ok(None);
        }
        let component_dir = namespace_dir.join(name);
        if !Self::ensure_directory(&component_dir, create)? {
            return Ok(None);
        }
        Ok(Some(component_dir.join(format!("{version}.json"))))
    }

    fn entrypoint(value: &str, label: &str) -> Result<(), String> {
        let Some((module, attribute)) = value.split_once(':') else {
            return Err(format!("{label} 必须是 module:callable 入口。"));
        };
        let valid_path = |path: &str| !path.is_empty() && path.split('.').all(Self::identifier);
        if value.matches(':').count() != 1 || !valid_path(module) || !valid_path(attribute) {
            return Err(format!(
                "{label} 只能由 Python 模块和属性标识符组成，不接受表达式。"
            ));
        }
        Ok(())
    }

    fn identifier(value: &str) -> bool {
        let mut bytes = value.bytes();
        matches!(bytes.next(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'_'))
            && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    }

    fn valid_version(value: &str) -> bool {
        let parts = value.split('.').collect::<Vec<_>>();
        parts.len() == 3
            && parts.iter().all(|part| {
                !part.is_empty()
                    && part.bytes().all(|byte| byte.is_ascii_digit())
                    && (part == &"0" || !part.starts_with('0'))
            })
    }

    fn valid_sha256(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn valid_package_file(value: &str) -> bool {
        !value.is_empty()
            && !value.contains('\\')
            && Path::new(value)
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_)))
    }

    fn validate_package_source(manifest: &Value) -> Result<(), String> {
        let Some(source) = manifest.get("package_source") else {
            return Ok(());
        };
        let source = source.as_object().ok_or("package_source 必须是对象。")?;
        let allowed = [
            "distribution",
            "distribution_version",
            "entry_point",
            "manifest_file",
            "manifest_sha256",
            "distribution_root",
            "module_sources",
        ];
        if let Some(unknown) = source.keys().find(|key| !allowed.contains(&key.as_str())) {
            return Err(format!("package_source 不支持字段：{unknown}"));
        }
        for field in ["distribution", "distribution_version", "entry_point"] {
            let value = source
                .get(field)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("package_source 缺少 {field}。"))?;
            if value.is_empty()
                || value.len() > 256
                || value.chars().any(char::is_control)
                || value.chars().any(char::is_whitespace)
            {
                return Err(format!("package_source.{field} 格式无效。"));
            }
        }
        let entry_point = source["entry_point"].as_str().expect("validated string");
        if !entry_point
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
            || !entry_point
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
        {
            return Err("package_source.entry_point 名称格式无效。".into());
        }
        let manifest_file = source
            .get("manifest_file")
            .and_then(Value::as_str)
            .ok_or("package_source 缺少 manifest_file。")?;
        if !Self::valid_package_file(manifest_file) {
            return Err("package_source.manifest_file 必须是安全的相对 POSIX 路径。".into());
        }
        let manifest_hash = source
            .get("manifest_sha256")
            .and_then(Value::as_str)
            .ok_or("package_source 缺少 manifest_sha256。")?;
        if !Self::valid_sha256(manifest_hash) {
            return Err("package_source.manifest_sha256 必须是小写 SHA-256。".into());
        }
        let distribution_root = source
            .get("distribution_root")
            .and_then(Value::as_str)
            .ok_or("package_source 缺少 distribution_root。")?;
        if !Path::new(distribution_root).is_absolute() {
            return Err("package_source.distribution_root 必须是绝对路径。".into());
        }
        let module_sources = source
            .get("module_sources")
            .and_then(Value::as_object)
            .ok_or("package_source.module_sources 必须是对象。")?;
        if !module_sources.contains_key("entrypoint")
            || module_sources
                .keys()
                .any(|role| !matches!(role.as_str(), "entrypoint" | "fixture"))
        {
            return Err(
                "package_source.module_sources 必须包含 entrypoint，且只接受 entrypoint/fixture。"
                    .into(),
            );
        }
        if manifest.get("conformance_fixture").is_some() != module_sources.contains_key("fixture") {
            return Err(
                "package_source 的 fixture 源码指纹必须与 conformance_fixture 对应。".into(),
            );
        }
        for (role, fingerprint) in module_sources {
            let fingerprint = fingerprint
                .as_object()
                .ok_or_else(|| format!("package_source.module_sources.{role} 必须是对象。"))?;
            if fingerprint
                .keys()
                .any(|key| !matches!(key.as_str(), "file" | "sha256"))
            {
                return Err(format!("package_source.module_sources.{role} 含未知字段。"));
            }
            let file = fingerprint
                .get("file")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("package_source.module_sources.{role} 缺少 file。"))?;
            if !Self::valid_package_file(file) {
                return Err(format!(
                    "package_source.module_sources.{role}.file 路径无效。"
                ));
            }
            let hash = fingerprint
                .get("sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("package_source.module_sources.{role} 缺少 sha256。"))?;
            if !Self::valid_sha256(hash) {
                return Err(format!(
                    "package_source.module_sources.{role}.sha256 必须是小写 SHA-256。"
                ));
            }
        }
        Ok(())
    }

    fn validate_schema_definition(schema: &Value, depth: usize) -> Result<(), String> {
        if depth > PARAM_DEPTH_LIMIT {
            return Err("参数 Schema 嵌套超过 16 层。".into());
        }
        let object = schema
            .as_object()
            .ok_or("参数 Schema 每一层都必须是对象。")?;
        let allowed = [
            "type",
            "description",
            "default",
            "enum",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "minimum",
            "maximum",
        ];
        if let Some(unknown) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
            return Err(format!("不支持的参数 Schema 关键字：{unknown}"));
        }
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or("参数 Schema 缺少 type。")?;
        if !matches!(
            kind,
            "object" | "array" | "string" | "integer" | "number" | "boolean"
        ) {
            return Err(format!("不支持的参数 Schema 类型：{kind}"));
        }
        if let Some(description) = object.get("description")
            && !description.is_string()
        {
            return Err("Schema description 必须是字符串。".into());
        }
        if let Some(values) = object.get("enum")
            && (!values.is_array() || values.as_array().is_some_and(Vec::is_empty))
        {
            return Err("Schema enum 必须是非空数组。".into());
        }
        if let Some(values) = object.get("enum").and_then(Value::as_array) {
            for value in values {
                Self::validate_parameter_value(value, schema, "enum", depth)?;
            }
        }
        for field in ["minimum", "maximum"] {
            if object
                .get(field)
                .is_some_and(|value| value.as_f64().is_none())
            {
                return Err(format!("Schema {field} 必须是数字。"));
            }
        }
        if object
            .get("minimum")
            .and_then(Value::as_f64)
            .zip(object.get("maximum").and_then(Value::as_f64))
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return Err("Schema minimum 不能大于 maximum。".into());
        }
        if let Some(default) = object.get("default") {
            Self::validate_parameter_value(default, schema, "default", depth)?;
        }
        if kind == "object" {
            let properties = object
                .get("properties")
                .and_then(Value::as_object)
                .ok_or("object Schema 必须声明 properties 对象。")?;
            if properties.len() > 256 {
                return Err("单个对象 Schema 最多声明 256 个参数。".into());
            }
            for (name, property) in properties {
                if name.is_empty() || name.chars().any(char::is_control) {
                    return Err("Schema 属性名不能为空或包含控制字符。".into());
                }
                Self::validate_schema_definition(property, depth + 1)?;
            }
            if let Some(required) = object.get("required") {
                let required = required.as_array().ok_or("required 必须是字符串数组。")?;
                let mut seen = std::collections::HashSet::new();
                for item in required {
                    let name = item.as_str().ok_or("required 必须是字符串数组。")?;
                    if !properties.contains_key(name) || !seen.insert(name) {
                        return Err(format!("required 参数不存在或重复：{name}"));
                    }
                }
            }
            if let Some(additional) = object.get("additionalProperties")
                && !additional.is_boolean()
            {
                return Err("additionalProperties 当前只接受布尔值。".into());
            }
        } else if kind == "array" {
            let items = object.get("items").ok_or("array Schema 必须声明 items。")?;
            Self::validate_schema_definition(items, depth + 1)?;
        }
        Ok(())
    }

    fn validate_parameter_value(
        value: &Value,
        schema: &Value,
        path: &str,
        depth: usize,
    ) -> Result<(), String> {
        if depth > PARAM_DEPTH_LIMIT {
            return Err(format!("参数 {path} 嵌套过深。"));
        }
        let kind = schema["type"].as_str().ok_or("参数 Schema 缺少 type。")?;
        let matches_type = match kind {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            _ => false,
        };
        if !matches_type {
            return Err(format!("参数 {path} 应为 {kind}。"));
        }
        if let Some(choices) = schema["enum"].as_array()
            && !choices.contains(value)
        {
            return Err(format!("参数 {path} 不在允许值中。"));
        }
        if let Some(number) = value.as_f64() {
            if let Some(bound) = schema["minimum"].as_f64()
                && number < bound
            {
                return Err(format!("参数 {path} 低于 minimum 限制 {bound}。"));
            }
            if let Some(bound) = schema["maximum"].as_f64()
                && number > bound
            {
                return Err(format!("参数 {path} 超出 maximum 限制 {bound}。"));
            }
        }
        if let Some(object) = value.as_object() {
            let properties = schema["properties"]
                .as_object()
                .ok_or("object Schema 缺少 properties。")?;
            if schema["additionalProperties"] == false
                && let Some(unknown) = object.keys().find(|name| !properties.contains_key(*name))
            {
                return Err(format!("参数 {path}.{unknown} 未在组件 Schema 中声明。"));
            }
            if let Some(required) = schema["required"].as_array() {
                for name in required.iter().filter_map(Value::as_str) {
                    if !object.contains_key(name) {
                        return Err(format!("缺少必填组件参数：{path}.{name}"));
                    }
                }
            }
            for (name, child) in object {
                if let Some(child_schema) = properties.get(name) {
                    Self::validate_parameter_value(
                        child,
                        child_schema,
                        &format!("{path}.{name}"),
                        depth + 1,
                    )?;
                }
            }
        }
        if let Some(items) = value.as_array() {
            let item_schema = &schema["items"];
            for (index, child) in items.iter().enumerate() {
                Self::validate_parameter_value(
                    child,
                    item_schema,
                    &format!("{path}[{index}]"),
                    depth + 1,
                )?;
            }
        }
        Ok(())
    }

    fn validate_manifest(manifest: &Value) -> Result<(), String> {
        let object = manifest
            .as_object()
            .ok_or("组件清单顶层必须是 JSON 对象。")?;
        let allowed = [
            "schema_version",
            "namespace",
            "component_id",
            "component_version",
            "extension_api_version",
            "kind",
            "entrypoint",
            "description",
            "params_schema",
            "capabilities",
            "dependencies",
            "docs",
            "port_contracts",
            "renderer",
            "state_schema",
            "conformance_fixture",
            "package_source",
        ];
        if let Some(unknown) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
            return Err(format!("组件清单不支持字段：{unknown}"));
        }
        if manifest["schema_version"] != 1 || manifest["extension_api_version"] != 1 {
            return Err("当前只支持 manifest schema 和 extension API v1。".into());
        }
        let id = manifest["component_id"]
            .as_str()
            .ok_or("清单缺少 component_id。")?;
        let (namespace, _) = Self::component_id(id)?;
        if manifest["namespace"].as_str() != Some(namespace) {
            return Err("namespace 必须与 component_id 的第一段一致。".into());
        }
        let version = manifest["component_version"]
            .as_str()
            .ok_or("清单缺少 component_version。")?;
        if !Self::valid_version(version) {
            return Err("component_version 使用 MAJOR.MINOR.PATCH 数字格式。".into());
        }
        let kind = manifest["kind"].as_str().ok_or("清单缺少 kind。")?;
        if !matches!(
            kind,
            "model"
                | "layer"
                | "dataset"
                | "transform"
                | "loss"
                | "metric"
                | "trainer"
                | "layer-edit-provider"
        ) {
            return Err(format!("不支持的组件 kind：{kind}"));
        }
        Self::entrypoint(
            manifest["entrypoint"]
                .as_str()
                .ok_or("清单缺少 entrypoint。")?,
            "entrypoint",
        )?;
        if let Some(description) = manifest.get("description")
            && description
                .as_str()
                .is_none_or(|text| text.chars().count() > 500)
        {
            return Err("description 必须是最多 500 字符的字符串。".into());
        }
        Self::validate_package_source(manifest)?;
        let schema = manifest
            .get("params_schema")
            .ok_or("清单缺少 params_schema。")?;
        if schema["type"] != "object" {
            return Err("组件 params_schema 顶层必须是 object。".into());
        }
        Self::validate_schema_definition(schema, 0)?;
        if !manifest["capabilities"].is_object() {
            return Err("capabilities 必须是对象。".into());
        }
        for (name, capability) in manifest["capabilities"]
            .as_object()
            .expect("checked object")
        {
            if name.is_empty() || name.chars().any(char::is_control) {
                return Err("能力名称不能为空或包含控制字符。".into());
            }
            let state = capability
                .as_str()
                .or_else(|| capability["state"].as_str())
                .ok_or_else(|| format!("能力 {name} 必须声明 state。"))?;
            if !matches!(
                state,
                "supported" | "unsupported" | "unknown" | "conditional"
            ) {
                return Err(format!("能力 {name} 的 state 无效：{state}"));
            }
            let reason = capability.get("reason").and_then(Value::as_str);
            if state != "supported" && reason.is_none_or(str::is_empty) {
                return Err(format!("能力 {name} 的 {state} 状态必须说明 reason。"));
            }
            if capability.is_object() {
                if let Some(unknown) = capability.as_object().and_then(|object| {
                    object
                        .keys()
                        .find(|key| !matches!(key.as_str(), "state" | "reason" | "requires"))
                }) {
                    return Err(format!("能力 {name} 不支持字段：{unknown}"));
                }
                if let Some(reason) = capability.get("reason")
                    && !reason.is_string()
                {
                    return Err(format!("能力 {name} 的 reason 必须是字符串。"));
                }
                if let Some(requires) = capability.get("requires")
                    && requires.as_array().is_none_or(|items| {
                        let mut seen = BTreeSet::new();
                        items.iter().any(|item| {
                            item.as_str()
                                .is_none_or(|name| name.is_empty() || !seen.insert(name))
                        })
                    })
                {
                    return Err(format!(
                        "能力 {name} 的 requires 必须是唯一的非空字符串数组。"
                    ));
                }
            }
        }
        if let Some(dependencies) = manifest.get("dependencies") {
            let dependencies = dependencies.as_array().ok_or("dependencies 必须是数组。")?;
            let mut dependency_ids = BTreeSet::new();
            for dependency in dependencies {
                let dependency = dependency.as_object().ok_or("dependency 项必须是对象。")?;
                if dependency
                    .keys()
                    .any(|key| !matches!(key.as_str(), "component_id" | "version" | "optional"))
                {
                    return Err("dependency 只接受 component_id、version 和 optional。".into());
                }
                let id = dependency
                    .get("component_id")
                    .and_then(Value::as_str)
                    .ok_or("dependency 缺少 component_id。")?;
                Self::component_id(id)?;
                if !dependency_ids.insert(id) {
                    return Err(format!("dependency {id} 在同一组件清单中重复。"));
                }
                let requirement = dependency
                    .get("version")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| format!("dependency {id} 缺少版本约束字符串。"))?;
                VersionReq::parse(requirement).map_err(|error| {
                    format!("dependency {id} 的版本约束必须是有效的 SemVer range：{error}")
                })?;
                if dependency
                    .get("optional")
                    .is_some_and(|value| !value.is_boolean())
                {
                    return Err(format!("dependency {id} 的 optional 必须是布尔值。"));
                }
            }
        }
        for key in ["port_contracts", "renderer", "state_schema"] {
            if manifest.get(key).is_some_and(|value| !value.is_object()) {
                return Err(format!("{key} 必须是 JSON 对象。"));
            }
        }
        if let Some(docs) = manifest.get("docs")
            && !docs.is_string()
            && !docs.is_object()
        {
            return Err("docs 必须是字符串或对象。".into());
        }
        if let Some(fixture) = manifest.get("conformance_fixture")
            && let Some(fixture) = fixture.as_str()
        {
            Self::entrypoint(fixture, "conformance_fixture")?;
        } else if manifest.get("conformance_fixture").is_some() {
            return Err("conformance_fixture 必须是 module:callable 字符串。".into());
        }
        Ok(())
    }

    fn read_manifest(
        path: &Path,
        expected_id: Option<&str>,
        expected_version: Option<&str>,
    ) -> Result<Value, String> {
        let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!("拒绝读取非普通组件清单：{}", path.display()));
        }
        if metadata.len() > MANIFEST_LIMIT {
            return Err(format!(
                "组件清单超过 {} KiB：{}",
                MANIFEST_LIMIT / 1024,
                path.display()
            ));
        }
        let contents = fs::read(path).map_err(|error| error.to_string())?;
        let manifest: Value = serde_json::from_slice(&contents)
            .map_err(|error| format!("组件清单 JSON 无效 {}：{error}", path.display()))?;
        Self::validate_manifest(&manifest)
            .map_err(|error| format!("组件清单无效 {}：{error}", path.display()))?;
        if expected_id.is_some_and(|id| manifest["component_id"].as_str() != Some(id)) {
            return Err("组件清单 ID 与文件路径不一致。".into());
        }
        if expected_version
            .is_some_and(|version| manifest["component_version"].as_str() != Some(version))
        {
            return Err("组件清单版本与文件路径不一致。".into());
        }
        Ok(manifest)
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn descriptor(path: &Path, manifest: Value) -> Result<Value, String> {
        let bytes = fs::read(path).map_err(|error| error.to_string())?;
        Ok(json!({
            "manifest":manifest,
            "manifest_path":path.to_string_lossy(),
            "manifest_sha256":Self::sha256(&bytes)
        }))
    }

    fn apply_defaults(value: &mut Map<String, Value>, schema: &Value) {
        let Some(properties) = schema["properties"].as_object() else {
            return;
        };
        for (name, property_schema) in properties {
            if !value.contains_key(name)
                && let Some(default) = property_schema.get("default")
            {
                value.insert(name.clone(), default.clone());
            }
            if let Some(Value::Object(child)) = value.get_mut(name) {
                Self::apply_defaults(child, property_schema);
            }
            if let Some(Value::Array(items)) = value.get_mut(name)
                && let Some(item_schema) = property_schema.get("items")
            {
                for item in items {
                    if let Value::Object(child) = item {
                        Self::apply_defaults(child, item_schema);
                    }
                }
            }
        }
    }

    fn add_dependency_constraints(
        descriptor: &Value,
        state: &mut DependencySolveState,
    ) -> Result<(), String> {
        let parent = descriptor["manifest"]["component_id"]
            .as_str()
            .ok_or("组件清单缺少 component_id。")?;
        let Some(dependencies) = descriptor["manifest"]["dependencies"].as_array() else {
            return Ok(());
        };
        for dependency in dependencies {
            if dependency["optional"] == true {
                continue;
            }
            let id = dependency["component_id"]
                .as_str()
                .ok_or("组件依赖缺少 component_id。")?;
            let requirement = dependency["version"]
                .as_str()
                .ok_or_else(|| format!("组件依赖 {id} 缺少版本约束。"))?;
            let requirement = VersionReq::parse(requirement)
                .map_err(|error| format!("组件 {parent} 的依赖 {id} 版本约束无效：{error}"))?;
            state
                .constraints
                .entry(id.to_owned())
                .or_default()
                .push((requirement, parent.to_owned()));
        }
        Ok(())
    }

    fn solve_dependency_constraints(
        available: &[Value],
        state: DependencySolveState,
        attempts: &mut usize,
        last_failure: &mut Option<String>,
    ) -> Result<Option<DependencySolveState>, String> {
        if state.selected.len() > 256 {
            return Err("组件依赖图超过 256 个节点。".into());
        }
        for (id, descriptor) in &state.selected {
            let version = descriptor["manifest"]["component_version"]
                .as_str()
                .ok_or_else(|| format!("组件 {id} 缺少版本。"))?;
            let version = Version::parse(version)
                .map_err(|error| format!("组件 {id} 的版本无效：{error}"))?;
            if let Some(constraints) = state.constraints.get(id)
                && constraints
                    .iter()
                    .any(|(requirement, _)| !requirement.matches(&version))
            {
                let descriptions = constraints
                    .iter()
                    .map(|(requirement, parent)| format!("{parent} 要求 {requirement}"))
                    .collect::<Vec<_>>()
                    .join("；");
                *last_failure = Some(format!("组件 {id}@{version} 与约束冲突：{descriptions}"));
                return Ok(None);
            }
        }

        let Some(id) = state
            .constraints
            .keys()
            .find(|id| !state.selected.contains_key(*id))
            .cloned()
        else {
            return Ok(Some(state));
        };
        let constraints = state.constraints.get(&id).cloned().unwrap_or_default();
        let candidates: Vec<&Value> = available
            .iter()
            .filter(|descriptor| descriptor["manifest"]["component_id"] == id)
            .filter(|descriptor| {
                descriptor["manifest"]["component_version"]
                    .as_str()
                    .and_then(|version| Version::parse(version).ok())
                    .is_some_and(|version| {
                        constraints
                            .iter()
                            .all(|(requirement, _)| requirement.matches(&version))
                    })
            })
            .collect();
        if candidates.is_empty() {
            let requested = constraints
                .iter()
                .map(|(requirement, parent)| format!("{parent} 要求 {requirement}"))
                .collect::<Vec<_>>()
                .join("；");
            let available_versions = available
                .iter()
                .filter(|descriptor| descriptor["manifest"]["component_id"] == id)
                .filter_map(|descriptor| descriptor["manifest"]["component_version"].as_str())
                .collect::<Vec<_>>();
            *last_failure = Some(if available_versions.is_empty() {
                format!("缺少组件依赖 {id}（{requested}）。")
            } else {
                format!(
                    "组件依赖 {id} 没有满足约束的已登记版本（{requested}）；可用版本：{}。",
                    available_versions.join(", ")
                )
            });
            return Ok(None);
        }

        for candidate in candidates {
            *attempts += 1;
            if *attempts > DEPENDENCY_RESOLUTION_LIMIT {
                return Err(format!(
                    "组件依赖求解超过 {DEPENDENCY_RESOLUTION_LIMIT} 个候选分支；收窄版本约束后重试。"
                ));
            }
            let mut branch = state.clone();
            branch.selected.insert(id.clone(), candidate.clone());
            Self::add_dependency_constraints(candidate, &mut branch)?;
            if let Some(solution) =
                Self::solve_dependency_constraints(available, branch, attempts, last_failure)?
            {
                return Ok(Some(solution));
            }
        }
        Ok(None)
    }

    fn dependency_cycle(selected: &BTreeMap<String, Value>) -> Option<Vec<String>> {
        fn visit(
            id: &str,
            selected: &BTreeMap<String, Value>,
            active: &mut Vec<String>,
            complete: &mut BTreeSet<String>,
        ) -> Option<Vec<String>> {
            if let Some(start) = active.iter().position(|item| item == id) {
                let mut cycle = active[start..].to_vec();
                cycle.push(id.to_owned());
                return Some(cycle);
            }
            if complete.contains(id) {
                return None;
            }
            active.push(id.to_owned());
            if let Some(dependencies) = selected[id]["manifest"]["dependencies"].as_array() {
                for dependency in dependencies {
                    if dependency["optional"] == true {
                        continue;
                    }
                    let Some(target) = dependency["component_id"].as_str() else {
                        continue;
                    };
                    if selected.contains_key(target)
                        && let Some(cycle) = visit(target, selected, active, complete)
                    {
                        return Some(cycle);
                    }
                }
            }
            active.pop();
            complete.insert(id.to_owned());
            None
        }

        let mut active = Vec::new();
        let mut complete = BTreeSet::new();
        for id in selected.keys() {
            if let Some(cycle) = visit(id, selected, &mut active, &mut complete) {
                return Some(cycle);
            }
        }
        None
    }

    fn resolve_dependency_graph(root: &Path, descriptor: &Value) -> Result<Value, String> {
        let root_id = descriptor["manifest"]["component_id"]
            .as_str()
            .ok_or("组件清单缺少 component_id。")?
            .to_owned();
        let root_version = descriptor["manifest"]["component_version"]
            .as_str()
            .ok_or("组件清单缺少 component_version。")?;
        let mut state = DependencySolveState::default();
        state.selected.insert(root_id.clone(), descriptor.clone());
        Self::add_dependency_constraints(descriptor, &mut state)?;
        let available = Self::all(root)?;
        let mut attempts = 0;
        let mut last_failure = None;
        let solution = Self::solve_dependency_constraints(
            &available,
            state,
            &mut attempts,
            &mut last_failure,
        )?
        .ok_or_else(|| {
            last_failure.unwrap_or_else(|| "组件依赖版本约束无法同时满足。".to_owned())
        })?;

        if let Some(cycle) = Self::dependency_cycle(&solution.selected) {
            return Err(format!("组件依赖存在循环：{}。", cycle.join(" -> ")));
        }

        let mut components = Vec::new();
        for (id, dependency) in &solution.selected {
            if id == &root_id {
                continue;
            }
            components.push(json!({
                "id":id,
                "version":dependency["manifest"]["component_version"],
                "kind":dependency["manifest"]["kind"],
                "entrypoint":dependency["manifest"]["entrypoint"],
                "path":Path::new(dependency["manifest_path"].as_str().unwrap_or_default())
                    .strip_prefix(root)
                    .map_err(|_| format!("组件依赖 {id} 的清单位于项目外。"))?
                    .to_string_lossy().replace('\\', "/"),
                "sha256":dependency["manifest_sha256"]
            }));
        }
        let mut edges = Vec::new();
        let mut optional_skipped = Vec::new();
        for (parent, selected) in &solution.selected {
            let Some(dependencies) = selected["manifest"]["dependencies"].as_array() else {
                continue;
            };
            for dependency in dependencies {
                let target = dependency["component_id"].as_str().unwrap_or_default();
                let constraint = dependency["version"].as_str().unwrap_or_default();
                let optional = dependency["optional"] == true;
                if optional {
                    let resolved_compatibly =
                        solution.selected.get(target).is_some_and(|resolved| {
                            resolved["manifest"]["component_version"]
                                .as_str()
                                .and_then(|version| Version::parse(version).ok())
                                .zip(VersionReq::parse(constraint).ok())
                                .is_some_and(|(version, requirement)| requirement.matches(&version))
                        });
                    if resolved_compatibly {
                        edges.push(json!({
                            "from":parent,
                            "component_id":target,
                            "constraint":constraint,
                            "version":solution.selected[target]["manifest"]["component_version"],
                            "optional":true,
                            "status":"resolved_via_required_path"
                        }));
                    } else {
                        optional_skipped.push(json!({
                            "from":parent,
                            "component_id":target,
                            "constraint":constraint,
                            "status":if solution.selected.contains_key(target) { "incompatible_optional" } else { "not_requested" }
                        }));
                    }
                    continue;
                }
                if let Some(resolved) = solution.selected.get(target) {
                    edges.push(json!({
                        "from":parent,
                        "component_id":target,
                        "constraint":constraint,
                        "version":resolved["manifest"]["component_version"],
                        "optional":false,
                        "status":"resolved"
                    }));
                }
            }
        }
        Ok(json!({
            "status":"resolved",
            "root":{"id":root_id,"version":root_version},
            "components":components,
            "edges":edges,
            "optional_skipped":optional_skipped,
            "candidate_branches":attempts
        }))
    }

    fn all(root: &Path) -> Result<Vec<Value>, String> {
        let Some(components) = Self::root(root, false)? else {
            return Ok(Vec::new());
        };
        let mut rows = Vec::new();
        for namespace in fs::read_dir(&components).map_err(|error| error.to_string())? {
            let namespace = namespace.map_err(|error| error.to_string())?;
            let namespace_name = namespace.file_name().to_string_lossy().into_owned();
            let metadata = namespace.file_type().map_err(|error| error.to_string())?;
            if metadata.is_symlink() {
                return Err(format!("组件目录含符号链接：{namespace_name}"));
            }
            if !metadata.is_dir() {
                continue;
            }
            for component in fs::read_dir(namespace.path()).map_err(|error| error.to_string())? {
                let component = component.map_err(|error| error.to_string())?;
                let component_name = component.file_name().to_string_lossy().into_owned();
                let component_meta = component.file_type().map_err(|error| error.to_string())?;
                if component_meta.is_symlink() {
                    return Err(format!(
                        "组件目录含符号链接：{namespace_name}/{component_name}"
                    ));
                }
                if !component_meta.is_dir() {
                    continue;
                }
                for entry in fs::read_dir(component.path()).map_err(|error| error.to_string())? {
                    let entry = entry.map_err(|error| error.to_string())?;
                    if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
                        continue;
                    }
                    let filename = entry.file_name().to_string_lossy().into_owned();
                    if entry
                        .file_type()
                        .map_err(|error| error.to_string())?
                        .is_symlink()
                    {
                        return Err(format!(
                            "组件清单含符号链接：{namespace_name}/{component_name}/{filename}"
                        ));
                    }
                    let version = filename.trim_end_matches(".json");
                    if !Self::valid_version(version) {
                        return Err(format!("组件清单文件名版本无效：{filename}"));
                    }
                    let id = format!("{namespace_name}/{component_name}");
                    let manifest = Self::read_manifest(&entry.path(), Some(&id), Some(version))?;
                    rows.push(Self::descriptor(&entry.path(), manifest)?);
                    if rows.len() > 4096 {
                        return Err("组件注册表超过当前解析上限 4096 个版本。".into());
                    }
                }
            }
        }
        rows.sort_by(|left, right| {
            left["manifest"]["component_id"]
                .as_str()
                .cmp(&right["manifest"]["component_id"].as_str())
                .then_with(|| {
                    let left_version = left["manifest"]["component_version"]
                        .as_str()
                        .unwrap_or("0.0.0");
                    let right_version = right["manifest"]["component_version"]
                        .as_str()
                        .unwrap_or("0.0.0");
                    version_tuple(right_version).cmp(&version_tuple(left_version))
                })
        });
        Ok(rows)
    }
}

impl ComponentResolver for FileComponentRegistry {
    fn list(&self, root: &Path, query: Option<&str>) -> Result<Vec<Value>, String> {
        let query = query.map(str::to_lowercase);
        Ok(Self::all(root)?
            .into_iter()
            .filter(|item| {
                query.as_ref().is_none_or(|query| {
                    let manifest = &item["manifest"];
                    manifest["component_id"]
                        .as_str()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(query)
                        || manifest["description"]
                            .as_str()
                            .unwrap_or("")
                            .to_lowercase()
                            .contains(query)
                        || manifest["component_version"]
                            .as_str()
                            .unwrap_or("")
                            .to_lowercase()
                            .contains(query)
                        || manifest["kind"]
                            .as_str()
                            .unwrap_or("")
                            .to_lowercase()
                            .contains(query)
                        || manifest["entrypoint"]
                            .as_str()
                            .unwrap_or("")
                            .to_lowercase()
                            .contains(query)
                })
            })
            .collect())
    }

    fn inspect(&self, root: &Path, id: &str) -> Result<Value, String> {
        let (_, _, component_id, requested_version) = Self::requested_component(id)?;
        let Some(directory) = Self::component_directory(root, component_id)? else {
            return Err(format!("找不到组件版本：{id}"));
        };
        let mut matches = Vec::new();
        if let Some(version) = requested_version {
            let path = directory.join(format!("{version}.json"));
            let manifest = Self::read_manifest(&path, Some(component_id), Some(version))
                .map_err(|error| format!("找不到或无法读取组件版本 {id}：{error}"))?;
            matches.push(Self::descriptor(&path, manifest)?);
        } else {
            for entry in fs::read_dir(&directory).map_err(|error| error.to_string())? {
                let entry = entry.map_err(|error| error.to_string())?;
                if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
                    continue;
                }
                let filename = entry.file_name().to_string_lossy().into_owned();
                if entry
                    .file_type()
                    .map_err(|error| error.to_string())?
                    .is_symlink()
                {
                    return Err(format!("组件清单含符号链接：{component_id}/{filename}"));
                }
                let version = filename.trim_end_matches(".json");
                if !Self::valid_version(version) {
                    return Err(format!("组件清单文件名版本无效：{filename}"));
                }
                let path = entry.path();
                let manifest = Self::read_manifest(&path, Some(component_id), Some(version))?;
                matches.push(Self::descriptor(&path, manifest)?);
            }
        }
        matches.sort_by(|left, right| {
            version_tuple(
                right["manifest"]["component_version"]
                    .as_str()
                    .unwrap_or("0.0.0"),
            )
            .cmp(&version_tuple(
                left["manifest"]["component_version"]
                    .as_str()
                    .unwrap_or("0.0.0"),
            ))
        });
        matches
            .into_iter()
            .next()
            .ok_or_else(|| format!("找不到组件版本：{id}"))
    }

    fn resolve_dependencies(&self, root: &Path, id: &str) -> Result<Value, String> {
        let descriptor = self.inspect(root, id)?;
        Self::resolve_dependency_graph(root, &descriptor)
    }

    fn register(&self, root: &Path, manifest: &Value) -> Result<Value, String> {
        Self::validate_manifest(manifest)?;
        let id = manifest["component_id"]
            .as_str()
            .ok_or("清单缺少 component_id。")?;
        let version = manifest["component_version"]
            .as_str()
            .ok_or("清单缺少 component_version。")?;
        let path = Self::path(root, id, version, true)?;
        if fs::symlink_metadata(&path).is_ok() {
            return Err(format!(
                "组件 {id}@{version} 已存在；注册不会覆盖已有版本。"
            ));
        }
        let parent = path.parent().ok_or("组件清单缺少父目录。")?;
        let mut temporary =
            tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
        serde_json::to_writer_pretty(&mut temporary, manifest)
            .map_err(|error| error.to_string())?;
        temporary
            .write_all(b"\n")
            .map_err(|error| error.to_string())?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|error| error.to_string())?;
        temporary
            .persist_noclobber(&path)
            .map_err(|error| error.to_string())?;
        crate::adapters::sync_directory(parent)
            .map_err(|error| error.to_string())?;
        self.inspect(root, &format!("{id}@{version}"))
    }

    fn resolve_layer(
        &self,
        root: &Path,
        id: &str,
        params: &Map<String, Value>,
    ) -> Result<Value, String> {
        let descriptor = self.inspect(root, id)?;
        let manifest = &descriptor["manifest"];
        if manifest["kind"] != "layer" {
            return Err(format!(
                "组件 {id} 的 kind 为 {}，不能用于 Layer 编辑。",
                manifest["kind"]
            ));
        }
        let mut resolved_params = params.clone();
        Self::apply_defaults(&mut resolved_params, &manifest["params_schema"]);
        let params = Value::Object(resolved_params.clone());
        Self::validate_parameter_value(&params, &manifest["params_schema"], "params", 0)?;
        let dependency_resolution = Self::resolve_dependency_graph(root, &descriptor)?;
        let mut resolved = descriptor;
        resolved["resolved_params"] = json!(resolved_params);
        resolved["resolved_dependencies"] = dependency_resolution["components"].clone();
        resolved["dependency_resolution"] = dependency_resolution;
        Ok(resolved)
    }

    fn resolve_layer_edit_provider(
        &self,
        root: &Path,
        id: &str,
        payload: &Map<String, Value>,
    ) -> Result<Value, String> {
        let descriptor = self.inspect(root, id)?;
        let manifest = &descriptor["manifest"];
        if manifest["kind"] != "layer-edit-provider" {
            return Err(format!(
                "组件 {id} 的 kind 为 {}，不能用于自定义 Layer 编辑。",
                manifest["kind"]
            ));
        }
        let mut resolved_payload = payload.clone();
        Self::apply_defaults(&mut resolved_payload, &manifest["params_schema"]);
        let value = Value::Object(resolved_payload.clone());
        Self::validate_parameter_value(&value, &manifest["params_schema"], "payload", 0)?;
        let dependency_resolution = Self::resolve_dependency_graph(root, &descriptor)?;
        let mut resolved = descriptor;
        resolved["resolved_payload"] = json!(resolved_payload);
        resolved["resolved_dependencies"] = dependency_resolution["components"].clone();
        resolved["dependency_resolution"] = dependency_resolution;
        Ok(resolved)
    }

    fn resolve_transform(
        &self,
        root: &Path,
        id: &str,
        params: &Map<String, Value>,
    ) -> Result<Value, String> {
        let descriptor = self.inspect(root, id)?;
        let manifest = &descriptor["manifest"];
        if manifest["kind"] != "transform" {
            return Err(format!(
                "组件 {id} 的 kind 为 {}，不能用于 transform 流水线。",
                manifest["kind"]
            ));
        }
        let mut resolved_params = params.clone();
        Self::apply_defaults(&mut resolved_params, &manifest["params_schema"]);
        let params = Value::Object(resolved_params.clone());
        Self::validate_parameter_value(&params, &manifest["params_schema"], "params", 0)?;
        let dependency_resolution = Self::resolve_dependency_graph(root, &descriptor)?;
        let mut resolved = descriptor;
        resolved["resolved_params"] = json!(resolved_params);
        resolved["resolved_dependencies"] = dependency_resolution["components"].clone();
        resolved["dependency_resolution"] = dependency_resolution;
        Ok(resolved)
    }

    fn resolve_component_for_test(
        &self,
        root: &Path,
        id: &str,
        kind: &str,
        params: &Map<String, Value>,
    ) -> Result<Value, String> {
        if !matches!(kind, "dataset" | "loss" | "metric" | "trainer") {
            return Err(format!("kind={kind} 没有通用值组件测试解析器。"));
        }
        let descriptor = self.inspect(root, id)?;
        let manifest = &descriptor["manifest"];
        if manifest["kind"] != kind {
            return Err(format!(
                "组件 {id} 的 kind 为 {}，不能用于 {kind} 测试。",
                manifest["kind"]
            ));
        }
        let mut resolved_params = params.clone();
        Self::apply_defaults(&mut resolved_params, &manifest["params_schema"]);
        let value = Value::Object(resolved_params.clone());
        Self::validate_parameter_value(&value, &manifest["params_schema"], "params", 0)?;
        let dependency_resolution = Self::resolve_dependency_graph(root, &descriptor)?;
        let mut resolved = descriptor;
        resolved["resolved_params"] = json!(resolved_params);
        resolved["resolved_dependencies"] = dependency_resolution["components"].clone();
        resolved["dependency_resolution"] = dependency_resolution;
        Ok(resolved)
    }

    fn component_verification(
        &self,
        root: &Path,
        id: &str,
        version: &str,
    ) -> Result<Option<Value>, String> {
        let Some(path) = Self::verification_path(root, id, version, false)? else {
            return Ok(None);
        };
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.to_string()),
        };
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!("组件验证记录不是普通文件：{}", path.display()));
        }
        if metadata.len() > MANIFEST_LIMIT {
            return Err("组件验证记录超过 256 KiB。".into());
        }
        let verification: Value =
            serde_json::from_slice(&fs::read(&path).map_err(|error| error.to_string())?)
                .map_err(|error| format!("组件验证记录 JSON 无效：{error}"))?;
        if verification["schema_version"] != 1
            || verification["component_id"] != id
            || verification["component_version"] != version
            || !verification["verifications"].is_array()
        {
            return Err(format!("组件验证记录身份或格式无效：{}", path.display()));
        }
        Ok(Some(verification))
    }

    fn record_component_verification(
        &self,
        root: &Path,
        id: &str,
        version: &str,
        verification: &Value,
    ) -> Result<(), String> {
        let descriptor = self.inspect(root, &format!("{id}@{version}"))?;
        if descriptor["manifest"]["kind"] != "layer-edit-provider" {
            return Err("只有 Layer Edit Provider 可以写入 Layer 编辑验证记录。".into());
        }
        if verification["manifest_sha256"] != descriptor["manifest_sha256"]
            || verification["component_id"] != id
            || verification["component_version"] != version
            || verification["capability"] != "layer_edit"
            || verification["status"] != "verified"
            || verification["suite_version"] != 2
            || verification["payload_sha256"].as_str().is_none()
            || verification["source_fingerprints"]["provider"]["sha256"]
                .as_str()
                .is_none()
            || verification["source_fingerprints"]["fixture"]["sha256"]
                .as_str()
                .is_none()
        {
            return Err("拒绝保存与当前组件版本/能力不匹配的验证记录。".into());
        }

        let Some(path) = Self::verification_path(root, id, version, true)? else {
            return Err("无法创建组件验证目录。".into());
        };
        let lock_path = path.with_extension("lock");
        if fs::symlink_metadata(&lock_path).is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err("拒绝通过符号链接锁定组件验证记录。".into());
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|error| error.to_string())?;
        lock.lock_exclusive().map_err(|error| error.to_string())?;
        let mut verifications = self
            .component_verification(root, id, version)?
            .filter(|existing| existing["manifest_sha256"] == descriptor["manifest_sha256"])
            .and_then(|existing| existing["verifications"].as_array().cloned())
            .unwrap_or_default();
        verifications.retain(|entry| {
            entry["payload_sha256"] != verification["payload_sha256"]
                || entry["source_fingerprints"] != verification["source_fingerprints"]
                || entry["python_version"] != verification["python_version"]
                || entry["torch_version"] != verification["torch_version"]
        });
        verifications.push(verification.clone());
        if verifications.len() > 128 {
            verifications.drain(..verifications.len() - 128);
        }
        let ledger = json!({
            "schema_version":1,
            "component_id":id,
            "component_version":version,
            "manifest_sha256":descriptor["manifest_sha256"],
            "verifications":verifications
        });
        let parent = path.parent().ok_or("组件验证记录缺少父目录。")?;
        let mut temporary =
            tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
        serde_json::to_writer_pretty(&mut temporary, &ledger).map_err(|error| error.to_string())?;
        temporary
            .write_all(b"\n")
            .map_err(|error| error.to_string())?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|error| error.to_string())?;
        temporary
            .persist(&path)
            .map_err(|error| error.error.to_string())?;
        crate::adapters::sync_directory(parent)
            .map_err(|error| error.to_string())?;
        Ok(())
    }
}
