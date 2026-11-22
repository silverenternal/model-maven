use super::*;
use crate::{
    cli::{DataCommand, PreviewDelivery},
    domain::capability::annotate_worker_request,
};
use sha2::{Digest, Sha256};

pub(super) fn handle<B: Backend>(app: &Application<B>, cli: &Cli, command: &DataCommand) -> Report {
    match command {
        DataCommand::Presets { query } => {
            let mut request = serde_json::json!({
                "protocol_version": crate::domain::PROTOCOL_VERSION,
                "operation": "preset_catalog",
                "query": query,
            });
            annotate_worker_request(&mut request);
            app.backend.execute(&request)
        }
        DataCommand::Index {
            root,
            verify,
            exclude_files,
            apply_exclusions,
            clear_exclusions,
        } => {
            let project_root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
                Ok(path) => path,
                Err(error) => {
                    return Report::error(
                        "data_index",
                        "PROJECT_UNAVAILABLE",
                        error.to_string(),
                        "在可读的 Model-Maven 项目目录中重试。",
                    );
                }
            };
            let mut request = serde_json::json!({
                "protocol_version": crate::domain::PROTOCOL_VERSION,
                "operation": "data_index",
                "dataset_root": root,
                "project_root": project_root,
                "verify": verify,
                "exclude_files": exclude_files,
                "apply_exclusions": apply_exclusions,
                "clear_exclusions": clear_exclusions,
                "runtime": {"python_executable": cli.python.as_deref().unwrap_or("python3")}
            });
            annotate_worker_request(&mut request);
            app.backend.execute(&request)
        }
        DataCommand::CsvTemplate { apply, if_plan } => {
            csv_template(app, *apply, if_plan.as_deref())
        }
        DataCommand::Preview {
            dataset,
            batch,
            parameters: raw_parameters,
            split,
            samples,
            scan_limit,
            fingerprint,
            transforms,
            renderer,
            renderer_entrypoint,
            renderer_version,
            delivery,
            ssh_host,
            ssh_user,
            ssh_port,
            destination,
        } => preview(
            app,
            cli,
            dataset.as_deref(),
            batch.as_deref(),
            raw_parameters,
            split,
            *samples,
            *scan_limit,
            fingerprint.as_deref(),
            transforms.as_deref(),
            renderer,
            renderer_entrypoint.as_deref(),
            renderer_version.as_deref(),
            *delivery,
            ssh_host.as_deref(),
            ssh_user.as_deref(),
            *ssh_port,
            destination,
        ),
    }
}

fn csv_template<B: Backend>(app: &Application<B>, apply: bool, if_plan: Option<&str>) -> Report {
    let operation = "data_csv_template";
    let fail = |code, message: String| {
        Report::error(
            operation,
            code,
            message,
            "检查项目目录及计划后重试；不会覆盖现有文件。",
        )
    };
    let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
        Ok(path) => path,
        Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
    };
    let source = include_str!("templates/csv_batch.py");
    let target = root.join("model_csv.py");
    match std::fs::symlink_metadata(&target) {
        Ok(_) => {
            return fail(
                "CSV_TEMPLATE_EXISTS",
                format!("{} 已存在。", target.display()),
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return fail("CSV_TEMPLATE_UNAVAILABLE", error.to_string()),
    }
    let plan = json!({
        "version":1,"project_root":root,"file":"model_csv.py","entrypoint":"model_csv:batch",
        "source":source,"format":"typed_csv_v2","dtypes":["float32","float64","float16","int64","bool"],
        "max_bytes":16777216,"max_rows":100000,"max_selected_columns":256,"max_selected_values":2000000,
        "sampling":"sequential_cyclic","automatic_splits":false,"automatic_shuffle":false
    });
    let revision = match crate::domain::input::content_revision(&plan) {
        Ok(revision) => revision,
        Err(error) => return fail("CSV_TEMPLATE_PLAN_FAILED", error),
    };
    if !apply {
        return Report::ok(
            operation,
            json!({"committed":false,"plan":plan,"plan_revision":revision}),
        );
    }
    if if_plan != Some(revision.as_str()) {
        return fail(
            "CSV_TEMPLATE_PLAN_CONFLICT",
            "计划修订不匹配；请重新预览。".into(),
        );
    }
    match app
        .project_source_writer
        .write_new(&root, Path::new("model_csv.py"), source.as_bytes())
    {
        Ok(path) => Report::ok(
            operation,
            json!({"committed":true,"plan_revision":revision,"path":path,"entrypoint":"model_csv:batch"}),
        ),
        Err((code, message)) => fail(code, message),
    }
}

#[allow(clippy::too_many_arguments)]
fn preview<B: Backend>(
    app: &Application<B>,
    cli: &Cli,
    dataset: Option<&str>,
    batch: Option<&str>,
    raw_parameters: &[String],
    split: &str,
    samples: usize,
    scan_limit: Option<u32>,
    fingerprint: Option<&str>,
    transforms_path: Option<&Path>,
    renderer: &str,
    renderer_entrypoint: Option<&str>,
    renderer_version: Option<&str>,
    delivery: PreviewDelivery,
    ssh_host: Option<&str>,
    ssh_user: Option<&str>,
    ssh_port: Option<u16>,
    destination: &Path,
) -> Report {
    let operation = "preview";
    if dataset.is_some() == batch.is_some() {
        return Report::error(
            operation,
            "PREVIEW_SOURCE_REQUIRED",
            "请选择一个 Dataset 工厂或批次工厂。",
            "使用 --dataset module:factory 或 --batch module:factory。",
        );
    }
    if scan_limit
        .is_some_and(|limit| dataset.is_none() || (limit as usize) < samples || limit > 100_000)
    {
        return Report::error(
            operation,
            "DATA_SCAN_LIMIT_INVALID",
            "扫描上限仅适用于 Dataset，且必须不小于样本数并不超过 100000。",
            "调整 --scan-limit 或 --samples 后重试。",
        );
    }
    if split.trim().is_empty() || split.len() > 128 || split.chars().any(char::is_control) {
        return Report::error(
            operation,
            "PREVIEW_SPLIT_INVALID",
            "切分名称无效。",
            "使用 train、validation、test 或数据工厂支持的切分名称。",
        );
    }
    let fingerprint = fingerprint.map(str::trim).filter(|value| !value.is_empty());
    if fingerprint.is_some_and(|value| value.len() > 256 || value.chars().any(char::is_control)) {
        return Report::error(
            operation,
            "PREVIEW_FINGERPRINT_INVALID",
            "数据指纹须为 1 到 256 个可见字符。",
            "使用数据提供者版本或切分 ID；修改数据时更新指纹。",
        );
    }
    let params = match parameters(raw_parameters) {
        Ok(params) => params,
        Err(error) => {
            return Report::error(
                operation,
                "PARAMETER_INVALID",
                error,
                "参数格式为 NAME=JSON。",
            );
        }
    };
    let transforms = match configured_transforms(app, transforms_path, split) {
        Ok(result) => result,
        Err(error) => {
            let (code, message) = match error {
                PreviewRepositoryError::TransformsPathInvalid(message) => {
                    ("PREVIEW_TRANSFORMS_PATH_INVALID", message)
                }
                PreviewRepositoryError::TransformsReadFailed(message) => {
                    ("PREVIEW_TRANSFORMS_READ_FAILED", message)
                }
                PreviewRepositoryError::TransformsInvalid(message) => {
                    ("PREVIEW_TRANSFORMS_INVALID", message)
                }
                other => ("PREVIEW_TRANSFORMS_READ_FAILED", other.message().to_owned()),
            };
            return Report::error(
                operation,
                code,
                message,
                "变换文件必须是有效的 JSON 步骤数组；步骤顺序按文件原样执行。",
            );
        }
    };
    let root = match app.previews.project_root() {
        Ok(root) => root,
        Err(error) => {
            return Report::error(
                operation,
                "PROJECT_UNAVAILABLE",
                error.message(),
                "进入模型工程目录后重试。",
            );
        }
    };
    let source_kind = if dataset.is_some() {
        "dataset"
    } else {
        "batch"
    };
    let source_entrypoint = dataset.or(batch).unwrap_or_default();
    if !valid_entrypoint(source_entrypoint) {
        return Report::error(
            operation,
            "PREVIEW_SOURCE_INVALID",
            "来源入口必须是 module:callable。",
            "使用项目内可导入的 Python Dataset 或批次工厂。 ",
        );
    }
    if renderer == "custom"
        && (renderer_entrypoint.is_none_or(|value| !valid_entrypoint(value))
            || renderer_version.is_none_or(|value| {
                value.trim().is_empty() || value.len() > 128 || value.chars().any(char::is_control)
            }))
    {
        return Report::error(
            operation,
            "PREVIEW_RENDERER_INVALID",
            "项目 renderer 需要有效的 module:callable 入口和 1 到 128 位版本标记。",
            "提供 --renderer custom --renderer-entrypoint module:render --renderer-version v1。",
        );
    }
    if renderer != "custom" && (renderer_entrypoint.is_some() || renderer_version.is_some()) {
        return Report::error(
            operation,
            "PREVIEW_RENDERER_OPTIONS_UNUSED",
            "只有 custom renderer 可以设置 renderer 入口和版本。",
            "移除这两个参数，或将 --renderer 设置为 custom。",
        );
    }
    let fingerprints = match app.previews.project_fingerprints(
        &root,
        source_entrypoint,
        renderer_entrypoint,
        &transforms.steps,
    ) {
        Ok(fingerprints) => fingerprints,
        Err(PreviewRepositoryError::SourceFingerprintFailed(message)) => {
            return Report::error(
                operation,
                "PREVIEW_SOURCE_FINGERPRINT_FAILED",
                message,
                "修复项目 Python 文件读取权限后重试。",
            );
        }
        Err(PreviewRepositoryError::EnvironmentReadFailed(message)) => {
            return Report::error(
                operation,
                "PREVIEW_ENVIRONMENT_READ_FAILED",
                message,
                "检查项目依赖锁文件权限。",
            );
        }
        Err(error) => {
            return Report::error(
                operation,
                "PREVIEW_SOURCE_FINGERPRINT_FAILED",
                error.message(),
                "修复项目 Python 文件读取权限后重试。",
            );
        }
    };
    let seed = cli.seed.unwrap_or(0);
    let mut request_identity = json!({
        "source_kind":source_kind,
        "source_entrypoint":source_entrypoint,
        "source_parameters":params,
        "split":split,
        "samples":samples,
        "seed":seed,
        "sampling_contract":"indexed-rank-v1-or-large-counter-v2-stream-window-v1",
        "scan_limit":scan_limit,
        "fingerprint":fingerprint,
        "transforms_sha256":transforms.sha256,
        "renderer":renderer,
        "renderer_entrypoint":renderer_entrypoint,
        "renderer_version":renderer_version.unwrap_or("1"),
        "project_python_sha256":fingerprints.project_python_sha256,
        "environment_lock_sha256":fingerprints.environment_lock_sha256,
        "python_executable":cli.python.as_deref().unwrap_or("python3"),
    });
    // Unknown data identity or an unlocked runtime must never turn a prior result into a cache hit.
    let cacheable = fingerprint.is_some()
        && fingerprints.environment_lock_sha256.is_some()
        && fingerprints.source_entrypoint_is_local
        && fingerprints.renderer_entrypoint_is_local.unwrap_or(true)
        && fingerprints
            .custom_transform_entrypoints_are_local
            .iter()
            .all(|is_local| *is_local);
    if !cacheable {
        request_identity["request_nonce"] = json!(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .to_string()
        );
    }
    let base_signature = match serde_json::to_vec(&request_identity) {
        Ok(bytes) => sha256(&bytes),
        Err(error) => {
            return Report::error(
                operation,
                "PREVIEW_IDENTITY_FAILED",
                error.to_string(),
                "移除非 JSON 数据工厂参数后重试。",
            );
        }
    };
    let lease = match app.previews.begin_preview(&root, &base_signature) {
        Ok(lease) => lease,
        Err(error) => {
            return Report::error(
                operation,
                "PREVIEW_REQUEST_FAILED",
                error,
                "检查 .model-maven/previews 目录及权限。",
            );
        }
    };
    if lease["cached"] == true {
        return delivery_report(
            &root,
            &lease["publication"],
            delivery,
            ssh_host,
            ssh_user,
            ssh_port,
            destination,
            true,
        );
    }
    let request_id = lease["request_id"].as_str().unwrap_or_default();
    let generation = lease["generation"].as_u64().unwrap_or_default();
    let request_directory = lease["request_directory"].as_str().unwrap_or_default();
    if !app
        .previews
        .mark_preview_running(&root, request_id, generation)
        .unwrap_or(false)
    {
        return Report::error(
            operation,
            "PREVIEW_SUPERSEDED",
            "此预览请求已被更新的请求取代。",
            "查看当前预览状态，或重新提交当前输入。 ",
        );
    }
    let mut request = json!({
        "protocol_version":PROTOCOL_VERSION,
        "worker_api_version":crate::domain::capability::WORKER_API_VERSION,
        "operation":"preview",
        "project_root":root,
        "runtime":{
            "python_executable":cli.python.as_deref().unwrap_or("python3"),
            "timeout_seconds":cli.timeout.unwrap_or(120)
        },
        "preview_source":{
            "kind":source_kind,
            "entrypoint":source_entrypoint,
            "parameters":params,
            "binding":Value::Null
        },
        "samples":samples,
        "scan_limit":scan_limit,
        "split":split,
        "seed":seed,
        "fingerprint":fingerprint,
        "transforms":transforms.steps,
        "preview_request":{
            "request_id":request_id,
            "generation":generation,
            "base_signature":base_signature,
            "request_directory":request_directory,
            "renderer":match renderer {
                "summary" => "builtin.summary",
                "custom" => "project.custom",
                _ => "builtin.auto",
            },
            "renderer_entrypoint":renderer_entrypoint,
            "renderer_version":renderer_version.unwrap_or("1"),
            "title":"Model-Maven data preview"
        }
    });
    annotate_worker_request(&mut request);
    let mut worker_report = app.backend.execute(&request);
    if !worker_report.success() {
        let diagnostic = worker_report.diagnostics.first();
        if diagnostic.is_some_and(|diagnostic| diagnostic.code == "USER_INTERRUPTED") {
            let _ = app.previews.cancel_preview(&root, request_id, generation);
        } else {
            let message = diagnostic
                .map(|diagnostic| diagnostic.message.as_str())
                .unwrap_or("renderer failed");
            let _ = app
                .previews
                .fail_preview(&root, request_id, generation, message);
            if let Ok(state) = app.previews.current_preview(&root) {
                attach_stale_previous_preview(&mut worker_report, &state, request_id, generation);
            }
        }
        return worker_report;
    }
    let publication =
        match app
            .previews
            .complete_preview(&root, request_id, generation, &base_signature)
        {
            Ok(result) => result,
            Err(error) => {
                let _ = app
                    .previews
                    .fail_preview(&root, request_id, generation, &error);
                let mut report = Report::error(
                    operation,
                    "PREVIEW_PUBLICATION_FAILED",
                    error,
                    "请求产物未发布；上一份成功预览仍保留。 ",
                );
                if let Ok(state) = app.previews.current_preview(&root) {
                    attach_stale_previous_preview(&mut report, &state, request_id, generation);
                }
                return report;
            }
        };
    if publication["published"] != true {
        return Report::ok(
            operation,
            json!({
                "status":publication["status"],
                "published":false,
                "request_id":request_id,
                "generation":generation,
                "current":app.previews.current_preview(&root).ok(),
                "note":"较新的预览请求已接管当前结果；本请求仅保留为历史产物。"
            }),
        );
    }
    let transform_observations = worker_report.data["transform_observations"].clone();
    let mut report = delivery_report(
        &root,
        &publication["publication"],
        delivery,
        ssh_host,
        ssh_user,
        ssh_port,
        destination,
        false,
    );
    report.data["transform_observations"] = transform_observations;
    report
}

fn configured_transforms<B: Backend>(
    app: &Application<B>,
    path: Option<&Path>,
    split: &str,
) -> Result<crate::ports::PreviewTransforms, PreviewRepositoryError> {
    if path.is_some() {
        return app.previews.read_transforms(path);
    }
    let root = app.previews.project_root()?;
    let config = app
        .project_config
        .load(&root)
        .map_err(PreviewRepositoryError::ProjectUnavailable)?;
    let Some(config) = config else {
        return app.previews.read_transforms(None);
    };
    let augment_split = if split == "train" { "train" } else { "eval" };
    let Some(augment) = config.get("augment") else {
        return app.previews.read_transforms(None);
    };
    let Some(augment) = augment.as_object() else {
        return Err(PreviewRepositoryError::TransformsInvalid(
            "项目配置中的 augment 必须是 train/eval 切分对象。".into(),
        ));
    };
    let steps = match augment.get(augment_split) {
        None => return app.previews.read_transforms(None),
        Some(Value::Array(steps)) => steps,
        Some(_) => {
            return Err(PreviewRepositoryError::TransformsInvalid(format!(
                "项目配置中的 augment.{augment_split} 必须是有序步骤数组。"
            )));
        }
    };
    let mut pipeline = Vec::with_capacity(steps.len());
    for (index, stored) in steps.iter().enumerate() {
        let Some(object) = stored.as_object() else {
            return Err(PreviewRepositoryError::TransformsInvalid(format!(
                "项目配置中的 augment.{augment_split}[{index}] 不是对象。"
            )));
        };
        let mut step = serde_json::Map::new();
        for (key, value) in object {
            if key == "id" || key == "target" {
                continue;
            }
            if key == "parameters" {
                let Some(parameters) = value.as_object() else {
                    return Err(PreviewRepositoryError::TransformsInvalid(format!(
                        "增强步骤 {} 的 parameters 必须是对象。",
                        stored["id"].as_str().unwrap_or("<unknown>")
                    )));
                };
                step.extend(parameters.clone());
                continue;
            }
            step.insert(key.clone(), value.clone());
        }
        pipeline.push(Value::Object(step));
    }
    let encoded = serde_json::to_vec(steps)
        .map_err(|error| PreviewRepositoryError::TransformsInvalid(error.to_string()))?;
    Ok(crate::ports::PreviewTransforms {
        steps: pipeline,
        sha256: format!("{:x}", Sha256::digest(encoded)),
    })
}

fn attach_stale_previous_preview(
    report: &mut Report,
    state: &Value,
    request_id: &str,
    generation: u64,
) {
    let current = &state["current"];
    let previous = &state["last_success"];
    if current["request_id"] != request_id
        || current["generation"] != generation
        || current["status"] != "failed"
        || state["last_success_stale"] != true
        || !previous.is_object()
    {
        return;
    }
    if !report.data.is_object() {
        report.data = json!({});
    }
    report.data["previous_preview"] = previous.clone();
    report.data["previous_result_stale"] = json!(true);
}

// This report binds delivery configuration to the exact publication and cache result.
#[allow(clippy::too_many_arguments)]
fn delivery_report(
    root: &Path,
    publication: &Value,
    delivery: PreviewDelivery,
    ssh_host: Option<&str>,
    ssh_user: Option<&str>,
    ssh_port: Option<u16>,
    destination: &Path,
    cached: bool,
) -> Report {
    let html = publication["artifacts"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|artifact| {
            artifact["media_type"]
                .as_str()
                .is_some_and(|mime| mime.starts_with("text/html"))
        });
    let Some(html) = html else {
        return Report::error(
            "preview",
            "PREVIEW_HTML_MISSING",
            "预览没有生成 HTML 文档。",
            "检查所选 renderer 是否返回了可查看的产物。",
        );
    };
    let uri = html["uri"].as_str().unwrap_or_default();
    let absolute_path = root.join(uri);
    let delivery_data = match delivery {
        PreviewDelivery::Local => json!({
            "mode":"local",
            "path":absolute_path,
            "opened":false,
            "transferred":false,
            "note":"文件已生成在当前执行环境；命令不会自动打开或传输它。"
        }),
        PreviewDelivery::Ssh => {
            let (Some(host), Some(user), Some(port)) = (ssh_host, ssh_user, ssh_port) else {
                return Report::error(
                    "preview",
                    "PREVIEW_SSH_CONFIRMATION_REQUIRED",
                    "SSH 交付需要用户明确提供主机、用户名和端口。",
                    "指定 --ssh-host、--ssh-user 和 --ssh-port；CLI 只生成命令，不自动连接。 ",
                );
            };
            if !valid_ssh_field(host, true) || !valid_ssh_field(user, false) {
                return Report::error(
                    "preview",
                    "PREVIEW_SSH_ADDRESS_INVALID",
                    "SSH 主机或用户名包含不支持的字符。",
                    "主机使用 DNS 名称/IP，用户名使用字母、数字、点、下划线或连字符。 ",
                );
            }
            let remote = absolute_path.to_string_lossy().replace('\\', "/");
            let command = format!(
                "scp -P {port} {user}@{host}:{} {}",
                shell_quote(&remote),
                shell_quote(&destination.to_string_lossy())
            );
            json!({
                "mode":"ssh",
                "remote_path":remote,
                "local_destination":destination,
                "scp_command":command,
                "opened":false,
                "transferred":false,
                "note":"请在目标本机核对并执行该命令；Model-Maven 不会连接 SSH 或自动传输。"
            })
        }
    };
    Report::ok(
        "preview",
        json!({
            "status":"succeeded",
            "cached":cached || publication["cached"] == true,
            "published":true,
            "request_id":publication["request_id"],
            "generation":publication["generation"],
            "input_signature":publication["input_signature"],
            "scope":publication["scope"],
            "sample_count":publication["sample_count"],
            "rendered_image_count":publication["rendered_image_count"],
            "artifacts":publication["artifacts"],
            "delivery":delivery_data,
            "previous_result_stale":false
        }),
    )
}

fn valid_entrypoint(value: &str) -> bool {
    let Some((module, callable)) = value.split_once(':') else {
        return false;
    };
    let valid_identifier = |part: &str| {
        !part.is_empty()
            && part.split('.').all(|name| {
                let mut bytes = name.bytes();
                matches!(bytes.next(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'_'))
                    && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            })
    };
    !value.contains(char::is_whitespace) && valid_identifier(module) && valid_identifier(callable)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_ssh_field(value: &str, host: bool) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'.' | b'_' | b'-')
                || (host && matches!(byte, b':' | b'[' | b']'))
        })
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
