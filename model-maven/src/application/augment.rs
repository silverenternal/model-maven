use super::*;
use crate::cli::AugmentCommand;

fn parse_parameters(raw: &[String]) -> Result<Map<String, Value>, String> {
    let mut values = Map::new();
    for item in raw {
        let Some((name, encoded)) = item.split_once('=') else {
            return Err(format!("参数须为 NAME=JSON：{item}"));
        };
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("参数名无效：{name}"));
        }
        let value: Value = serde_json::from_str(encoded)
            .map_err(|error| format!("参数 {name} 不是有效 JSON：{error}"))?;
        if values.insert(name.to_owned(), value).is_some() {
            return Err(format!("参数重复：{name}"));
        }
    }
    Ok(values)
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

pub(super) fn valid_target(target: &str) -> bool {
    !target.is_empty()
        && target.len() <= 256
        && !target.starts_with('.')
        && !target.ends_with('.')
        && target.split('.').all(|part| {
            !part.is_empty()
                && (part.bytes().all(|byte| byte.is_ascii_digit())
                    || ((part.as_bytes()[0].is_ascii_alphabetic() || part.as_bytes()[0] == b'_')
                        && part
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')))
        })
}

fn validate_step(
    kind: &str,
    parameters: &Map<String, Value>,
    entrypoint: Option<&str>,
) -> Result<(), String> {
    let allowed: &[&str] = match kind {
        "Resize" => &["size", "interpolation"],
        "RandomResizedCrop" => &["size", "scale"],
        "ToTensor" => &[],
        "Normalize" => &["mean", "std"],
        "RandomHorizontalFlip" => &["p"],
        "custom" => {
            return if entrypoint.is_some_and(|value| value.contains(':')) {
                Ok(())
            } else {
                Err("custom 需要 module:callable 工厂入口。".into())
            };
        }
        _ => return Err(format!("未知变换类型：{kind}")),
    };
    if let Some(unknown) = parameters
        .keys()
        .find(|key| !allowed.contains(&key.as_str()))
    {
        return Err(format!("{kind} 不支持参数 {unknown}。"));
    }
    match kind {
        "Resize" => {
            let size = &parameters["size"];
            let positive = |value: &Value| value.as_u64().is_some_and(|number| number > 0);
            if !(positive(size)
                || size
                    .as_array()
                    .is_some_and(|pair| pair.len() == 2 && pair.iter().all(positive)))
            {
                return Err("Resize.size 必须是正整数或两个正整数组成的数组。".into());
            }
            if parameters.get("interpolation").is_some_and(|value| {
                !matches!(
                    value.as_str(),
                    Some("bilinear" | "nearest" | "bicubic" | "area")
                )
            }) {
                return Err(
                    "Resize.interpolation 必须是 bilinear、nearest、bicubic 或 area。".into(),
                );
            }
        }
        "RandomResizedCrop" => {
            let size = &parameters["size"];
            let positive = |value: &Value| value.as_u64().is_some_and(|number| number > 0);
            if !(positive(size)
                || size
                    .as_array()
                    .is_some_and(|pair| pair.len() == 2 && pair.iter().all(positive)))
            {
                return Err("RandomResizedCrop.size 必须是正整数或两个正整数组成的数组。".into());
            }
            if !parameters.get("scale").is_some_and(|value| {
                value.as_array().is_some_and(|pair| {
                    pair.len() == 2
                        && pair[0].as_f64().is_some_and(|number| number > 0.0)
                        && pair[1].as_f64().is_some_and(|number| {
                            number <= 1.0 && number >= pair[0].as_f64().unwrap_or(2.0)
                        })
                })
            }) {
                return Err(
                    "RandomResizedCrop.scale 必须是 0 < min <= max <= 1 的二元数组。".into(),
                );
            }
        }
        "Normalize" => {
            for name in ["mean", "std"] {
                let value = &parameters[name];
                let values = value
                    .as_array()
                    .map_or_else(|| vec![value], |items| items.iter().collect());
                if values.is_empty()
                    || values
                        .iter()
                        .any(|number| number.as_f64().is_none_or(|value| !value.is_finite()))
                {
                    return Err(format!("Normalize.{name} 必须是有限数字或非空数字数组。"));
                }
                if name == "std"
                    && values
                        .iter()
                        .any(|number| number.as_f64().is_some_and(|value| value <= 0.0))
                {
                    return Err("Normalize.std 中每个值都必须大于零。".into());
                }
            }
        }
        "RandomHorizontalFlip"
            if parameters.get("p").is_some_and(|value| {
                value
                    .as_f64()
                    .is_none_or(|value| !(0.0..=1.0).contains(&value))
            }) =>
        {
            return Err("RandomHorizontalFlip.p 必须在 0 到 1 之间。".into());
        }
        _ => {}
    }
    Ok(())
}

fn fail(operation: &'static str, code: &'static str, message: impl Into<String>) -> Report {
    Report::error(
        operation,
        code,
        message,
        "运行 m-model augment list 查看当前流水线。",
    )
}

impl<B: Backend> Application<B> {
    pub(super) fn attach_training_augmentation(
        &self,
        job: &mut Job,
        root: &Path,
        request: &mut Value,
        expected_config_revision: Option<&str>,
    ) -> Result<(), Report> {
        let config = self.project_config.load(root).map_err(|error| {
            Report::error(
                job.operation,
                "PROJECT_CONFIG_INVALID",
                error,
                "检查 .model-maven/project.json；训练没有启动。",
            )
        })?;
        let Some(config) = config else {
            if expected_config_revision.is_some() {
                return Err(Report::error(
                    job.operation,
                    "PROJECT_CONFIG_CONFLICT",
                    "项目配置已不存在，运行未启动。",
                    "重新查看训练计划。 ",
                ));
            }
            request["train_augment"] = json!([]);
            return Ok(());
        };
        let steps = match config.get("augment") {
            None => Vec::new(),
            Some(Value::Object(augment)) => match augment.get("train") {
                None => Vec::new(),
                Some(Value::Array(steps)) => steps.clone(),
                Some(_) => {
                    return Err(Report::error(
                        job.operation,
                        "AUGMENT_CONFIG_INVALID",
                        "augment.train 必须是有序步骤数组。",
                        "修复 .model-maven/project.json 或使用 m-model augment list；训练没有启动。",
                    ));
                }
            },
            Some(_) => {
                return Err(Report::error(
                    job.operation,
                    "AUGMENT_CONFIG_INVALID",
                    "augment 必须是按 train/eval 划分的对象。",
                    "修复 .model-maven/project.json；训练没有启动。",
                ));
            }
        };
        for (index, step) in steps.iter().enumerate() {
            if !step.is_object() {
                return Err(Report::error(
                    job.operation,
                    "AUGMENT_CONFIG_INVALID",
                    format!("augment.train[{index}] 必须是对象。"),
                    "修复该步骤或使用 m-model augment remove 删除；训练没有启动。",
                ));
            }
            if step["target"]
                .as_str()
                .is_none_or(|target| !valid_target(target))
            {
                return Err(Report::error(
                    job.operation,
                    "AUGMENT_TARGET_REQUIRED",
                    format!(
                        "训练增强步骤 {} 没有有效 target。",
                        step["id"].as_str().unwrap_or("<unknown>")
                    ),
                    "使用 m-model augment set <id> --target args.0（或 kwargs.images）后重试。",
                ));
            }
        }
        let revision = crate::domain::input::content_revision(&config).map_err(|error| {
            Report::error(
                job.operation,
                "PROJECT_CONFIG_INVALID",
                error,
                "检查项目配置。",
            )
        })?;
        if expected_config_revision.is_some_and(|expected| expected != revision) {
            return Err(Report::error(
                job.operation,
                "PROJECT_CONFIG_CONFLICT",
                "项目配置在准备训练时发生变化，运行未启动。",
                "重新查看训练计划。 ",
            ));
        }
        job.config_revision = Some(revision.clone());
        job.input_sources.insert(
            "train_augment".into(),
            json!({"kind":"project_config","reference":".model-maven/project.json#/augment/train","revision":revision}),
        );
        request["train_augment"] = json!(steps);
        Ok(())
    }

    pub(super) fn augment(&self, command: &AugmentCommand) -> Report {
        let (operation, split) = match command {
            AugmentCommand::List { split } => ("augment_list", split.as_str()),
            AugmentCommand::Add { split, .. } => ("augment_add", split.as_str()),
            AugmentCommand::Set { split, .. } => ("augment_set", split.as_str()),
            AugmentCommand::Move { split, .. } => ("augment_move", split.as_str()),
            AugmentCommand::Remove { split, .. } => ("augment_remove", split.as_str()),
        };
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root,
            Err(error) => return fail(operation, "PROJECT_UNAVAILABLE", error.to_string()),
        };
        let Some(mut config) = (match self.project_config.load(&root) {
            Ok(config) => config,
            Err(error) => return fail(operation, "PROJECT_CONFIG_INVALID", error),
        }) else {
            return fail(
                operation,
                "PROJECT_NOT_INITIALIZED",
                "先运行 m-model init 初始化项目。",
            );
        };
        let revision = match crate::domain::input::content_revision(&config) {
            Ok(revision) => revision,
            Err(error) => return fail(operation, "PROJECT_CONFIG_INVALID", error),
        };
        let options = match command {
            AugmentCommand::List { .. } => None,
            AugmentCommand::Add { edit, .. }
            | AugmentCommand::Set { edit, .. }
            | AugmentCommand::Move { edit, .. }
            | AugmentCommand::Remove { edit, .. } => Some(edit),
        };
        if options
            .and_then(|edit| edit.if_config_revision.as_deref())
            .is_some_and(|expected| expected != revision)
        {
            return fail(
                operation,
                "PROJECT_CONFIG_CONFLICT",
                "项目配置已发生变化；增强操作没有执行。请重新预览。 ",
            );
        }
        if config.get("augment").is_none() {
            config["augment"] = json!({"train": [], "eval": []});
        }
        if !config["augment"][split].is_array() {
            return fail(
                operation,
                "AUGMENT_CONFIG_INVALID",
                format!("augment.{split} 必须是步骤数组。"),
            );
        }
        let steps = config["augment"][split]
            .as_array_mut()
            .expect("checked array");
        let find = |steps: &[Value], id: &str| {
            steps
                .iter()
                .position(|step| step["id"].as_str() == Some(id))
        };
        match command {
            AugmentCommand::List { .. } => {
                return Report::ok(
                    operation,
                    json!({"split":split,"steps":steps,"config_revision":revision}),
                );
            }
            AugmentCommand::Add {
                kind,
                id,
                entrypoint,
                target,
                scale,
                parameters,
                ..
            } => {
                if !valid_id(id) {
                    return fail(
                        operation,
                        "AUGMENT_ID_INVALID",
                        "步骤 ID 只能包含字母、数字、下划线和连字符，长度不超过 64。",
                    );
                }
                if find(steps, id).is_some() {
                    return fail(
                        operation,
                        "AUGMENT_ID_EXISTS",
                        format!("步骤 ID 已存在：{id}"),
                    );
                }
                if !matches!(
                    kind.as_str(),
                    "Resize"
                        | "RandomResizedCrop"
                        | "ToTensor"
                        | "Normalize"
                        | "RandomHorizontalFlip"
                        | "custom"
                ) {
                    return fail(
                        operation,
                        "AUGMENT_TYPE_UNSUPPORTED",
                        format!(
                            "未知变换类型 {kind}。可用类型：Resize、RandomResizedCrop、ToTensor、Normalize、RandomHorizontalFlip、custom。"
                        ),
                    );
                }
                let mut parameters = match parse_parameters(parameters) {
                    Ok(values) => values,
                    Err(error) => return fail(operation, "AUGMENT_PARAMETERS_INVALID", error),
                };
                if let Some(scale) = scale {
                    let Some((minimum, maximum)) = scale.split_once(',') else {
                        return fail(
                            operation,
                            "AUGMENT_SCALE_INVALID",
                            "--scale 格式须为 MIN,MAX。",
                        );
                    };
                    let (Ok(minimum), Ok(maximum)) =
                        (minimum.trim().parse::<f64>(), maximum.trim().parse::<f64>())
                    else {
                        return fail(
                            operation,
                            "AUGMENT_SCALE_INVALID",
                            "--scale 必须是有限数字范围。",
                        );
                    };
                    if !minimum.is_finite()
                        || !maximum.is_finite()
                        || minimum <= 0.0
                        || minimum > maximum
                        || maximum > 1.0
                    {
                        return fail(
                            operation,
                            "AUGMENT_SCALE_INVALID",
                            "--scale 必须满足 0 < MIN <= MAX <= 1。",
                        );
                    }
                    parameters.insert("scale".into(), json!([minimum, maximum]));
                }
                if (kind == "custom") != entrypoint.is_some() {
                    return fail(
                        operation,
                        "AUGMENT_ENTRYPOINT_INVALID",
                        "custom 必须提供 --entrypoint，内置变换不能提供此参数。",
                    );
                }
                if let Err(error) = validate_step(kind, &parameters, entrypoint.as_deref()) {
                    return fail(operation, "AUGMENT_STEP_INVALID", error);
                }
                if target.as_deref().is_some_and(|value| !valid_target(value)) {
                    return fail(
                        operation,
                        "AUGMENT_TARGET_INVALID",
                        "目标路径须由点分隔的字段名或数组索引组成，例如 args.0 或 kwargs.images。",
                    );
                }
                let mut step = json!({"id":id,"type":kind,"parameters":parameters});
                if let Some(target) = target {
                    step["target"] = json!(target);
                }
                if let Some(entrypoint) = entrypoint {
                    if !entrypoint.contains(':') || entrypoint.chars().any(char::is_whitespace) {
                        return fail(
                            operation,
                            "AUGMENT_ENTRYPOINT_INVALID",
                            "入口格式须为 module:callable。",
                        );
                    }
                    step["entrypoint"] = json!(entrypoint);
                }
                steps.push(step);
            }
            AugmentCommand::Set {
                id,
                parameters,
                scale,
                target,
                ..
            } => {
                if parameters.is_empty() && scale.is_none() && target.is_none() {
                    return fail(
                        operation,
                        "AUGMENT_UPDATE_REQUIRED",
                        "至少指定一个 --param、--scale 或 --target。",
                    );
                }
                let Some(index) = find(steps, id) else {
                    return fail(
                        operation,
                        "AUGMENT_ID_NOT_FOUND",
                        format!("找不到步骤：{id}"),
                    );
                };
                let mut parameters = match parse_parameters(parameters) {
                    Ok(values) => values,
                    Err(error) => return fail(operation, "AUGMENT_PARAMETERS_INVALID", error),
                };
                if let Some(scale) = scale {
                    let Some((minimum, maximum)) = scale.split_once(',') else {
                        return fail(
                            operation,
                            "AUGMENT_SCALE_INVALID",
                            "--scale 格式须为 MIN,MAX。",
                        );
                    };
                    let (Ok(minimum), Ok(maximum)) =
                        (minimum.trim().parse::<f64>(), maximum.trim().parse::<f64>())
                    else {
                        return fail(
                            operation,
                            "AUGMENT_SCALE_INVALID",
                            "--scale 必须是有限数字范围。",
                        );
                    };
                    if !minimum.is_finite()
                        || !maximum.is_finite()
                        || minimum <= 0.0
                        || minimum > maximum
                        || maximum > 1.0
                    {
                        return fail(
                            operation,
                            "AUGMENT_SCALE_INVALID",
                            "--scale 必须满足 0 < MIN <= MAX <= 1。",
                        );
                    }
                    parameters.insert("scale".into(), json!([minimum, maximum]));
                }
                if !steps[index]["parameters"].is_object() {
                    steps[index]["parameters"] = json!({});
                }
                let kind = steps[index]["type"].as_str().unwrap_or_default().to_owned();
                let entrypoint = steps[index]["entrypoint"].as_str().map(str::to_owned);
                let target_update = if let Some(target) = target {
                    if !valid_target(target) {
                        return fail(
                            operation,
                            "AUGMENT_TARGET_INVALID",
                            "目标路径须由点分隔的字段名或数组索引组成，例如 args.0 或 kwargs.images。",
                        );
                    }
                    Some(target)
                } else {
                    None
                };
                {
                    let existing = steps[index]["parameters"]
                        .as_object_mut()
                        .expect("object set above");
                    existing.extend(parameters);
                    if let Err(error) = validate_step(&kind, existing, entrypoint.as_deref()) {
                        return fail(operation, "AUGMENT_STEP_INVALID", error);
                    }
                }
                if let Some(target) = target_update {
                    steps[index]["target"] = json!(target);
                }
            }
            AugmentCommand::Move { id, before, .. } => {
                let Some(index) = find(steps, id) else {
                    return fail(
                        operation,
                        "AUGMENT_ID_NOT_FOUND",
                        format!("找不到步骤：{id}"),
                    );
                };
                if before.as_deref() == Some(id.as_str()) {
                    return fail(
                        operation,
                        "AUGMENT_MOVE_INVALID",
                        "步骤不能移动到自身之前。",
                    );
                }
                let target = match before {
                    Some(before) => match find(steps, before) {
                        Some(index) => index,
                        None => {
                            return fail(
                                operation,
                                "AUGMENT_ID_NOT_FOUND",
                                format!("找不到目标步骤：{before}"),
                            );
                        }
                    },
                    None => steps.len() - 1,
                };
                let moved = steps.remove(index);
                let destination = if before.is_none() {
                    steps.len()
                } else if index < target {
                    target - 1
                } else {
                    target
                };
                steps.insert(destination, moved);
            }
            AugmentCommand::Remove { id, .. } => {
                let Some(index) = find(steps, id) else {
                    return fail(
                        operation,
                        "AUGMENT_ID_NOT_FOUND",
                        format!("找不到步骤：{id}"),
                    );
                };
                steps.remove(index);
            }
        }
        let steps_snapshot = steps.clone();
        if options.is_some_and(|edit| edit.dry_run) {
            return Report::ok(
                operation,
                json!({"committed":false,"split":split,"steps":steps_snapshot,"base_config_revision":revision,"validation":{"configuration":"passed","runtime":"not_run"}}),
            );
        }
        match self
            .project_config
            .update_versioned(&root, &revision, &config)
        {
            Ok(update) => Report::ok(
                operation,
                json!({"committed":true,"split":split,"steps":steps_snapshot,"path":update.path,"base_config_revision":revision,"config_revision":update.revision}),
            ),
            Err(ProjectConfigUpdateError::Conflict { .. }) => fail(
                operation,
                "PROJECT_CONFIG_CONFLICT",
                "项目配置在提交期间发生变化；增强操作未写入。",
            ),
            Err(error) => fail(operation, "PROJECT_CONFIG_UPDATE_FAILED", error.message()),
        }
    }
}
