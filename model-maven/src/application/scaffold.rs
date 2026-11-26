use super::*;
use std::{
    collections::{BTreeMap, HashSet},
    fmt::Write as _,
    fs,
    io::{Read, Write as _},
};
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct InputSpec {
    name: String,
    shape: Vec<usize>,
    dynamic_axes: Vec<usize>,
    dtype: Option<String>,
}

const RECIPE_FILE: &str = "model-maven.json";
const BUILD_PREFIX: &str = "model_maven_build_";
const RECIPE_LIMIT: u64 = 128 * 1024;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaffoldRecipe {
    version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data: Option<RecipeData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    training: Option<RecipeTraining>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    supervision: Option<RecipeSupervision>,
    inputs: Vec<InputSpec>,
    outputs: Vec<RecipeOutput>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    routes: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    components: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    component_params: BTreeMap<String, Map<String, Value>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    encoder_layers: BTreeMap<String, Vec<EncoderLayer>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    encoder_widths: BTreeMap<String, usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    native_encoders: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeSupervision {
    file: String,
    sha256: String,
}

fn validate_supervision_reference(reference: &RecipeSupervision) -> Result<(), String> {
    let path = Path::new(&reference.file);
    if reference.file.is_empty() || reference.file.len() > 1024 || reference.file.contains(['\\', ':'])
        || reference.file.chars().any(char::is_control) || path.is_absolute()
        || !path.components().all(|part| matches!(part, std::path::Component::Normal(_)))
        || reference.sha256.len() != 64
        || !reference.sha256.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        return Err("监督引用须使用项目相对路径和有效 SHA-256。".into());
    }
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeData {
    factory: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    binding: Option<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    params: Map<String, Value>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeTraining {
    mode: String,
    entry: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    params: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    optimizer: Option<RecipeOptimizer>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeOptimizer {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    factory: Option<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    params: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    lr: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    weight_decay: Option<f64>,
}

fn validate_training_profile(profile: &RecipeTraining) -> Result<(), String> {
    if !matches!(profile.mode.as_str(), "managed-step" | "manual-step" | "external-loop")
        || profile.entry.len() > 240 || !(profile.entry == "$model:step" || python_entrypoint(&profile.entry))
        || (profile.entry == "$model:step" && profile.mode == "external-loop")
        || profile.params.len() > 64 || !profile.params.keys().all(|name| python_identifier(name))
        || serde_json::to_vec(&profile.params).map_err(|e| e.to_string())?.len() > 65_536 {
        return Err("训练声明的模式、入口或参数无效。".into());
    }
    if let Some(optimizer) = &profile.optimizer {
        if profile.mode == "external-loop"
            || optimizer.factory.as_deref().is_some_and(|f| f.len() > 240 || !python_entrypoint(f))
            || (optimizer.factory.is_some() && (profile.mode != "managed-step" || optimizer.lr.is_some() || optimizer.weight_decay.is_some()))
            || (optimizer.factory.is_none() && !optimizer.params.is_empty())
            || optimizer.params.len() > 64 || !optimizer.params.keys().all(|name| python_identifier(name))
            || serde_json::to_vec(&optimizer.params).map_err(|e| e.to_string())?.len() > 65_536
            || optimizer.lr.is_some_and(|value| !value.is_finite() || value <= 0.0)
            || optimizer.weight_decay.is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err("训练声明的优化器设置无效或与训练模式不兼容。".into());
        }
    }
    Ok(())
}

pub(super) fn validate_training_value(value: &Value) -> Result<(), String> {
    let profile: RecipeTraining = serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
    validate_training_profile(&profile)
}

pub(super) fn validate_supervision_value(value: &Value) -> Result<(), String> {
    let reference: RecipeSupervision = serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
    validate_supervision_reference(&reference)
}

fn read_training_profile(path: &Path) -> Result<(TrainingMode, RecipeTraining), String> {
    let bytes = read_recipe_for_update(path)?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    let profile = if value["kind"] == "image_classification" {
        let recipe = super::classification::read_classification_recipe(path)?;
        serde_json::from_value(recipe.training.ok_or("配置未声明训练入口。")?).map_err(|error| format!("{error}"))?
    } else if value.get("nodes").is_some() && value.get("edges").is_some() {
        serde_json::from_value(super::graph::training_profile(path, &bytes)?).map_err(|error| format!("{error}"))?
    } else {
        ScaffoldRecipe::parse(&bytes)?.training.ok_or("配置未声明训练入口。")?
    };
    validate_training_profile(&profile)?;
    let mode = match profile.mode.as_str() {
        "managed-step" => TrainingMode::ManagedStep,
        "manual-step" => TrainingMode::ManualStep,
        "external-loop" => TrainingMode::ExternalLoop,
        _ => return Err("训练模式无效。".into()),
    };
    Ok((mode, profile))
}

pub(crate) fn normalize_training_command(command: &Command) -> Result<Option<Command>, Report> {
        let recipe_path = match command {
            Command::Try { inputs, .. } | Command::Train { inputs, .. } => inputs.training_recipe.as_ref(),
            _ => None,
        };
        if let Some(path) = recipe_path {
            let mut normalized = command.clone();
            let (inputs, mode, step, training_loop, params, loop_params, lr, weight_decay, incompatible) = match &mut normalized {
                Command::Try { inputs, training_mode, step, training_loop, params, loop_params, lr, weight_decay, repeat, .. } =>
                    (inputs, training_mode, step, training_loop, params, loop_params, lr, weight_decay, repeat.is_some()),
                Command::Train { inputs, training_mode, step, training_loop, params, loop_params, lr, weight_decay, resume, migrate_from, .. } =>
                    (inputs, training_mode, step, training_loop, params, loop_params, lr, weight_decay, resume.is_some() || migrate_from.is_some()),
                _ => unreachable!(),
            };
            if incompatible || mode.is_some() || step.is_some() || training_loop.is_some() || !params.is_empty() || !loop_params.is_empty() {
                return Err(Report::error("training_recipe", "TRAINING_RECIPE_CONFLICT",
                    "训练声明不能与训练入口、模式、参数、重复或恢复选项同时使用。", "移除冲突选项后重试。"));
            }
            let (selected_mode, profile) = match read_training_profile(path) {
                Ok(value) => value,
                Err(error) => return Err(Report::error("training_recipe", "TRAINING_RECIPE_INVALID", error, "检查声明文件中的 training 字段。")),
            };
            if profile.optimizer.is_some() && (lr.is_some() || weight_decay.is_some() || inputs.optimizer_factory.is_some() || !inputs.optimizer_params.is_empty()) {
                return Err(Report::error("training_recipe", "TRAINING_RECIPE_CONFLICT", "训练声明中的优化器不能与命令行优化器设置同时使用。", "移除冲突设置后重试。"));
            }
            let values = profile.params.iter().map(|(name, value)| format!("{name}={value}")).collect();
            if let Some(optimizer) = profile.optimizer {
                *lr = optimizer.lr;
                *weight_decay = optimizer.weight_decay;
                inputs.optimizer_factory = optimizer.factory;
                inputs.optimizer_params = optimizer.params.iter().map(|(name, value)| format!("{name}={value}")).collect();
            }
            let entry = profile.entry;
            let entry = if entry == "$model:step" {
                let model = if let Some(model) = inputs.model.as_deref() {
                    model.to_owned()
                } else {
                    let root = std::env::current_dir().and_then(|value| value.canonicalize())
                        .map_err(|error| Report::error("training_recipe", "TRAINING_RECIPE_MODEL_UNAVAILABLE", error.to_string(), "选择模型项目。"))?;
                    let config = crate::adapters::project_config::FileProjectConfigStore.load(&root)
                        .map_err(|error| Report::error("training_recipe", "TRAINING_RECIPE_MODEL_UNAVAILABLE", error, "检查项目模型配置。"))?;
                    config.and_then(|value| value["model"].as_str().map(str::to_owned)).unwrap_or_default()
                };
                let module = model.strip_suffix(":model").ok_or_else(|| Report::error("training_recipe", "TRAINING_RECIPE_MODEL_UNAVAILABLE",
                    "当前模型入口不是 module:model，无法解析当前模块的 step。", "选择使用 module:model 的模型，或在声明中填写明确训练入口。"))?;
                let resolved = format!("{module}:step");
                if !python_entrypoint(&resolved) { return Err(Report::error("training_recipe", "TRAINING_RECIPE_MODEL_UNAVAILABLE", "当前模型模块无效。", "检查项目模型入口。")); }
                resolved
            } else { entry };
            inputs.training_recipe = None;
            *mode = Some(selected_mode);
            if selected_mode == TrainingMode::ExternalLoop {
                *training_loop = Some(entry);
                *loop_params = values;
            } else {
                *step = Some(entry);
                *params = values;
            }
            return Ok(Some(normalized));
        }
    Ok(None)
}

fn recipe_binding(value: Option<&str>) -> Result<Option<BatchBinding>, String> {
    Ok(match value {
        None => None,
        Some("envelope") => Some(BatchBinding::Envelope),
        Some("single") => Some(BatchBinding::Single),
        Some("single-raw") => Some(BatchBinding::SingleRaw),
        Some("args") => Some(BatchBinding::Args),
        Some("kwargs") => Some(BatchBinding::Kwargs),
        Some(_) => return Err("数据绑定方式无效。".into()),
    })
}

fn binding_name(binding: &BatchBinding) -> &'static str {
    match binding {
        BatchBinding::Envelope => "envelope", BatchBinding::Single => "single",
        BatchBinding::SingleRaw => "single-raw", BatchBinding::Args => "args",
        BatchBinding::Kwargs => "kwargs",
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum EncoderLayer {
    Linear { #[serde(default, skip_serializing_if = "Option::is_none")] id: Option<String>, features: usize },
    Relu { #[serde(default, skip_serializing_if = "Option::is_none")] id: Option<String> },
    Gelu { #[serde(default, skip_serializing_if = "Option::is_none")] id: Option<String> },
    LayerNorm { #[serde(default, skip_serializing_if = "Option::is_none")] id: Option<String> },
    Dropout { #[serde(default, skip_serializing_if = "Option::is_none")] id: Option<String>, probability: f64 },
    Factory { #[serde(default, skip_serializing_if = "Option::is_none")] id: Option<String>, reference: String, output_width: usize, #[serde(default)] params: Map<String, Value> },
    Residual { #[serde(default, skip_serializing_if = "Option::is_none")] id: Option<String>, branch: Vec<EncoderLayer> },
    Identity { #[serde(default, skip_serializing_if = "Option::is_none")] id: Option<String> },
    Parallel { #[serde(default, skip_serializing_if = "Option::is_none")] id: Option<String>, branches: Vec<Vec<EncoderLayer>>, merge: String },
}

impl EncoderLayer {
    fn id(&self) -> Option<&str> {
        match self {
            Self::Linear { id, .. } | Self::Relu { id } | Self::Gelu { id } | Self::LayerNorm { id }
            | Self::Dropout { id, .. } | Self::Factory { id, .. } | Self::Residual { id, .. }
            | Self::Identity { id } | Self::Parallel { id, .. } => id.as_deref(),
        }
    }
    fn replace_id(&mut self, value: String) {
        match self {
            Self::Linear { id, .. } | Self::Relu { id } | Self::Gelu { id } | Self::LayerNorm { id }
            | Self::Dropout { id, .. } | Self::Factory { id, .. } | Self::Residual { id, .. }
            | Self::Identity { id } | Self::Parallel { id, .. } => *id = Some(value),
        }
    }
}

pub(crate) fn scaffold_encoder_layer_catalog() -> Value {
    json!([
        {"label":"Linear","defaults":EncoderLayer::Linear { id: None, features: 16 }},
        {"label":"ReLU","defaults":EncoderLayer::Relu { id: None }},
        {"label":"GELU","defaults":EncoderLayer::Gelu { id: None }},
        {"label":"LayerNorm","defaults":EncoderLayer::LayerNorm { id: None }},
        {"label":"Dropout","defaults":EncoderLayer::Dropout { id: None, probability: 0.1 }},
    ])
}

struct ResolvedEncoder {
    input_width: usize,
    output_width: usize,
    constructors: Vec<String>,
    width_checks: Vec<(usize, usize, String)>,
    layer_paths: Vec<(String, String)>,
}

type LayerStacks = BTreeMap<String, Vec<EncoderLayer>>;

fn collect_layer_paths(layers: &[EncoderLayer], prefix: &str, output: &mut Vec<(String, String)>) {
    for (index, layer) in layers.iter().enumerate() {
        let path = format!("{prefix}.{index}");
        if let Some(id) = layer.id() { output.push((id.to_owned(), path.clone())); }
        match layer {
            EncoderLayer::Residual { branch, .. } => collect_layer_paths(branch, &format!("{path}.branch"), output),
            EncoderLayer::Parallel { branches, .. } => for (branch_index, branch) in branches.iter().enumerate() {
                collect_layer_paths(branch, &format!("{path}.branches.{branch_index}"), output);
            },
            _ => {}
        }
    }
}

fn stack_expression(stack: &ResolvedEncoder) -> String {
    let args = stack.constructors.join(", ");
    if stack.width_checks.is_empty() { format!("nn.Sequential({args})") } else {
        let checks = stack.width_checks.iter().map(|(i, w, r)| {
            format!("{i}: ({w}, {})", serde_json::to_string(r).expect("string serialization"))
        }).collect::<Vec<_>>().join(", ");
        format!("_CheckedStack({{{checks}}}, {args})")
    }
}

fn resolve_stack(layers: &[EncoderLayer], slot: &str, input_width: usize, depth: usize,
    layer_count: &mut usize, parameter_count: &mut usize, ids: &mut HashSet<String>) -> Result<ResolvedEncoder, String> {
    if layers.is_empty() || layers.len() > 16 || depth > 8 {
        return Err(format!("编码器 {slot} 的层组须包含 1 至 16 层，嵌套不超过 8 层。"));
    }
    *layer_count = layer_count.checked_add(layers.len()).ok_or("模型层数过大。")?;
    if *layer_count > 64 { return Err("一个模型最多包含 64 层（包括残差分支）。".into()); }
    let mut width = input_width;
    let mut constructors = Vec::with_capacity(layers.len());
    let mut width_checks = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        if let Some(id) = layer.id() {
            if id.len() != 34 || !id.starts_with("n_") || !id[2..].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) || !ids.insert(id.to_owned()) {
                return Err(format!("编码器 {slot} 第 {} 层的稳定 ID 无效或重复。", index + 1));
            }
        }
        let code = match layer {
            EncoderLayer::Linear { features, .. } if (1..=65_536).contains(features) => {
                *parameter_count = (width + 1).checked_mul(*features)
                    .and_then(|count| parameter_count.checked_add(count)).ok_or("编码器参数量过大。")?;
                let code = format!("nn.Linear({width}, {features})");
                width = *features;
                code
            }
            EncoderLayer::Relu { .. } => "nn.ReLU()".to_owned(),
            EncoderLayer::Gelu { .. } => "nn.GELU()".to_owned(),
            EncoderLayer::Identity { .. } => "nn.Identity()".to_owned(),
            EncoderLayer::LayerNorm { .. } => {
                *parameter_count = parameter_count.checked_add(width * 2).ok_or("编码器参数量过大。")?;
                format!("nn.LayerNorm({width})")
            }
            EncoderLayer::Dropout { probability, .. } if (0.0..1.0).contains(probability) => format!("nn.Dropout(p={probability})"),
            EncoderLayer::Factory { reference, output_width, params, .. }
                if python_entrypoint(reference) && reference.len() <= 240
                    && (1..=65_536).contains(output_width) && params.len() <= 64
                    && params.keys().all(|name| python_identifier(name)) => {
                width_checks.push((index, *output_width, reference.clone()));
                let encoded = serde_json::to_string(params).map_err(|error| error.to_string())?;
                if encoded.len() > 65_536 { return Err("自定义层参数不能超过 65536 字节。".into()); }
                let reference = serde_json::to_string(reference).map_err(|error| error.to_string())?;
                let encoded = serde_json::to_string(&encoded).map_err(|error| error.to_string())?;
                width = *output_width;
                format!("_make_layer({reference}, {encoded})")
            }
            EncoderLayer::Residual { branch, .. } => {
                let nested = resolve_stack(branch, slot, width, depth + 1, layer_count, parameter_count, ids)?;
                if nested.output_width != width {
                    return Err(format!("编码器 {slot} 第 {} 层残差分支输出宽度 {} 与输入宽度 {width} 不一致。", index + 1, nested.output_width));
                }
                format!("_Residual({})", stack_expression(&nested))
            }
            EncoderLayer::Parallel { branches, merge, .. } => {
                if !(2..=8).contains(&branches.len()) || !matches!(merge.as_str(), "add" | "concat") {
                    return Err(format!("编码器 {slot} 第 {} 层并行组须有 2 至 8 个分支，汇合方式为 add 或 concat。", index + 1));
                }
                let mut outputs = Vec::with_capacity(branches.len());
                let mut stacks = Vec::with_capacity(branches.len());
                for branch in branches {
                    let nested = resolve_stack(branch, slot, width, depth + 1, layer_count, parameter_count, ids)?;
                    outputs.push(nested.output_width);
                    stacks.push(stack_expression(&nested));
                }
                if merge == "add" && outputs.iter().any(|value| *value != outputs[0]) {
                    return Err(format!("编码器 {slot} 第 {} 层相加分支输出宽度不一致：{outputs:?}。", index + 1));
                }
                width = if merge == "add" { outputs[0] } else {
                    outputs.iter().try_fold(0usize, |sum, value| sum.checked_add(*value))
                        .filter(|sum| *sum <= 1_000_000).ok_or("并行分支输出宽度过大。")?
                };
                format!("_Parallel(\"{merge}\", {})", stacks.join(", "))
            }
            _ => return Err(format!("编码器 {slot} 第 {} 层的类型或参数无效。", index + 1)),
        };
        if *parameter_count > 10_000_000 {
            return Err(format!("编码器 {slot} 第 {} 层使内置编码器参数合计超过 10000000。", index + 1));
        }
        constructors.push(code);
    }
    let mut layer_paths = Vec::new();
    collect_layer_paths(layers, "", &mut layer_paths);
    Ok(ResolvedEncoder { input_width, output_width: width, constructors, width_checks, layer_paths })
}

fn flattened_width(input: &InputSpec) -> usize {
    input.shape.iter().enumerate()
        .filter(|(axis, _)| !input.dynamic_axes.contains(axis))
        .map(|(_, dimension)| dimension)
        .product()
}

fn resolve_encoder_layers(
    inputs: &[InputSpec],
    routes: &[usize],
    injections: &[String],
    declarations: &[String],
) -> Result<(LayerStacks, BTreeMap<String, ResolvedEncoder>), String> {
    let mut stacks = LayerStacks::new();
    let mut encoders = BTreeMap::new();
    let mut parameter_count = 0usize;
    let mut layer_count = 0usize;
    let mut ids = HashSet::new();
    for declaration in declarations {
        let (slot, raw) = declaration.split_once('=').ok_or("层配置需使用 SLOT=JSON。")?;
        let index = inputs.iter().position(|input| format!("{}_encoder", input.name) == slot)
            .ok_or_else(|| format!("不存在编码器 {slot}。"))?;
        if stacks.contains_key(slot) || injections.iter().any(|binding| binding.split_once('=').is_some_and(|(name, _)| name == slot)) {
            return Err(format!("编码器 {slot} 只能选择一组内置层或一个工厂。"));
        }
        let layers: Vec<EncoderLayer> = serde_json::from_str(raw)
            .map_err(|error| format!("编码器 {slot} 的层配置无效：{error}"))?;
        let input_width = flattened_width(&inputs[routes[index]]);
        let resolved = resolve_stack(&layers, slot, input_width, 0, &mut layer_count, &mut parameter_count, &mut ids)?;
        stacks.insert(slot.to_owned(), layers);
        encoders.insert(slot.to_owned(), resolved);
    }
    Ok((stacks, encoders))
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeOutput {
    name: String,
    features: usize,
    loss: String,
    weight: f64,
}

impl ScaffoldRecipe {
    fn read(path: &Path) -> Result<Self, String> {
        if !fs::metadata(path).map_err(|error| error.to_string())?.is_file() {
            return Err("配置路径必须是普通文件。".into());
        }
        let file = fs::File::open(path).map_err(|error| error.to_string())?;
        if !file.metadata().map_err(|error| error.to_string())?.is_file() {
            return Err("配置路径必须是普通文件。".into());
        }
        let mut bytes = Vec::new();
        file.take(RECIPE_LIMIT + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
        Self::parse(&bytes)
    }

    fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() as u64 > RECIPE_LIMIT {
            return Err("配置文件不能超过 128 KiB。".into());
        }
        let recipe: Self = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        if recipe.version != 1 {
            return Err(format!("不支持配置版本 {}。", recipe.version));
        }
        if let Some(reference) = &recipe.supervision { validate_supervision_reference(reference)?; }
        Ok(recipe)
    }

    fn arguments(&self) -> Result<(Vec<String>, Vec<String>), String> {
        let inputs = self.inputs.iter().map(|input| {
            if !python_identifier(&input.name)
                || input.dynamic_axes.iter().any(|axis| *axis >= input.shape.len())
                || input.dynamic_axes.iter().collect::<HashSet<_>>().len() != input.dynamic_axes.len()
            {
                return Err(format!("输入 {} 的名称或动态轴无效。", input.name));
            }
            if input.dtype.as_ref().is_some_and(|dtype| dtype.contains(['=', ':', ','])) {
                return Err(format!("输入 {} 的数据类型无效。", input.name));
            }
            let shape = input.shape.iter().enumerate().map(|(axis, dim)| {
                if input.dynamic_axes.contains(&axis) { format!("?{dim}") } else { dim.to_string() }
            }).collect::<Vec<_>>().join(",");
            let dtype = input.dtype.as_ref().map(|dtype| format!(":{dtype}")).unwrap_or_default();
            Ok(format!("{}{dtype}={shape}", input.name))
        }).collect::<Result<Vec<_>, String>>()?;
        let outputs = self.outputs.iter().map(|output| {
            if !python_identifier(&output.name) || Objective::parse(&output.loss).is_err() {
                return Err(format!("输出 {} 的名称或损失无效。", output.name));
            }
            Ok(format!("{}={}:{}@{}", output.name, output.features, output.loss, output.weight))
        }).collect::<Result<Vec<_>, String>>()?;
        Ok((inputs, outputs))
    }
}

pub(super) fn bundle_reference(bytes: &[u8]) -> Result<Option<Value>, String> {
    let recipe = ScaffoldRecipe::parse(bytes)?;
    recipe.arguments()?;
    recipe.supervision.as_ref().map(serde_json::to_value).transpose().map_err(|error| error.to_string())
}

fn graph_training_from_recipe(profile: &RecipeTraining) -> Result<Value, String> {
    validate_training_profile(profile)?;
    let mut value = json!({"mode":profile.mode,"step":profile.entry,"step_params":profile.params});
    if let Some(optimizer) = &profile.optimizer { value["optimizer"] = json!(optimizer); }
    Ok(value)
}

struct EditableGraphBuilder<'a> {
    nodes: &'a mut Vec<Value>, edges: &'a mut Vec<Value>, definitions: &'a mut Vec<Value>,
    used: &'a mut HashSet<String>, branch: usize, counter: usize, dtype_definitions: Vec<String>, nested: bool,
}
impl EditableGraphBuilder<'_> {
    fn fresh(&mut self, preferred: &str) -> String {
        let mut candidate = preferred.to_owned();
        let mut index = 1;
        while !self.used.insert(candidate.clone()) { candidate = format!("{preferred}_{index}"); index += 1; }
        candidate
    }
    fn unary(&mut self, source: &str, node: Value, id: &str) {
        self.edges.push(json!({"from":{"node":source,"port":"value"},"to":{"node":id,"port":"value"}}));
        self.nodes.push(node);
    }
    fn checked(&mut self, source: &str, batch: &str, width: Option<usize>) -> String {
        let id = self.fresh("m_branch_check");
        let mut node = json!({"id":id,"kind":if width.is_some() {"assert_flat"} else {"assert_batch"},"inputs":["value","batch"],"outputs":["value"]});
        if let Some(width) = width { node["width"] = json!(width); }
        self.unary(source, node, &id);
        self.edges.push(json!({"from":{"node":batch,"port":"value"},"to":{"node":id,"port":"batch"}}));
        id
    }
    fn after(&mut self, source: &str, dependency: &str) -> String {
        let id = self.fresh("m_branch_order");
        self.unary(source, json!({"id":id,"kind":"after","inputs":["value","after"],"outputs":["value"]}), &id);
        self.edges.push(json!({"from":{"node":dependency,"port":"value"},"to":{"node":id,"port":"after"}}));
        id
    }
    fn stack(&mut self, layers: &[EncoderLayer], mut previous: String, mut width: usize, batch: &str) -> Result<(String, usize), String> {
        for layer in layers {
            self.counter += 1;
            let suffix = self.counter;
            let preferred = layer.id().map(str::to_owned).unwrap_or_else(|| format!("m_layer_{suffix}"));
            match layer {
                EncoderLayer::Residual { branch, .. } => {
                    self.nested = true;
                    let copy = self.fresh(&format!("m_residual_copy_{suffix}"));
                    self.unary(&previous, json!({"id":copy,"kind":"clone","inputs":["value"],"outputs":["value"]}), &copy);
                    let (branch_end, branch_width) = self.stack(branch, copy, width, batch)?;
                    if branch_width != width { return Err("残差分支宽度不一致。".into()); }
                    let id = self.fresh(&preferred);
                    self.nodes.push(json!({"id":id,"kind":"add","inputs":["skip","branch"],"outputs":["value"]}));
                    for (source, port) in [(&previous, "skip"), (&branch_end, "branch")] {
                        self.edges.push(json!({"from":{"node":source,"port":"value"},"to":{"node":id,"port":port}}));
                    }
                    previous = id;
                },
                EncoderLayer::Parallel { branches, merge, .. } => {
                    self.nested = true;
                    let mut ends: Vec<String> = Vec::new();
                    let mut widths = Vec::new();
                    for (index, branch) in branches.iter().enumerate() {
                        let source = if let Some(prior) = ends.last() { self.after(&previous, prior) } else { previous.clone() };
                        let copy = self.fresh(&format!("m_parallel_copy_{suffix}_{index}"));
                        self.unary(&source, json!({"id":copy,"kind":"clone","inputs":["value"],"outputs":["value"]}), &copy);
                        let (end, branch_width) = self.stack(branch, copy, width, batch)?;
                        ends.push(end); widths.push(branch_width);
                    }
                    let last = ends.last().cloned().ok_or("并行组没有分支。")?;
                    for index in 0..ends.len() {
                        let value = if index + 1 == ends.len() { ends[index].clone() } else { self.after(&ends[index], &last) };
                        ends[index] = self.checked(&value, batch, None);
                    }
                    let id = self.fresh(&preferred);
                    let ports = (0..ends.len()).map(|index| format!("branch_{index}")).collect::<Vec<_>>();
                    let mut node = json!({"id":id,"kind":if merge == "add" {"add"} else {"concat"},"inputs":ports,"outputs":["value"]});
                    if merge == "concat" { node["axis"] = json!(1); }
                    self.nodes.push(node);
                    for (index, source) in ends.iter().enumerate() {
                        self.edges.push(json!({"from":{"node":source,"port":"value"},"to":{"node":id,"port":format!("branch_{index}")}}));
                    }
                    width = if merge == "add" { widths[0] } else { widths.iter().sum() };
                    previous = id;
                },
                _ => {
                    let id = self.fresh(&preferred);
                    let definition = format!("m_layer_def_{}_{suffix}", self.branch);
                    let (factory, params) = match layer {
                        EncoderLayer::Linear { features, .. } => { let result = ("torch.nn:Linear", json!({"in_features":width,"out_features":features})); width = *features; result },
                        EncoderLayer::Relu { .. } => ("torch.nn:ReLU", json!({})),
                        EncoderLayer::Gelu { .. } => ("torch.nn:GELU", json!({})),
                        EncoderLayer::LayerNorm { .. } => ("torch.nn:LayerNorm", json!({"normalized_shape":width})),
                        EncoderLayer::Dropout { probability, .. } => ("torch.nn:Dropout", json!({"p":probability})),
                        EncoderLayer::Identity { .. } => ("torch.nn:Identity", json!({})),
                        EncoderLayer::Factory { reference, output_width, params, .. } => { width = *output_width; (reference.as_str(), json!(params)) },
                        _ => return Err("层类型不受支持。".into()),
                    };
                    self.definitions.push(json!({"id":definition,"factory":factory,"params":params}));
                    self.dtype_definitions.push(definition.clone());
                    self.unary(&previous, json!({"id":id,"kind":"call","definition":definition,"call_style":"positional","inputs":["value"],"outputs":["value"]}), &id);
                    previous = if matches!(layer, EncoderLayer::Factory { .. }) { self.checked(&id, batch, Some(width)) } else { id };
                }
            }
        }
        Ok((previous, width))
    }
}

fn editable_graph_from_recipe(recipe: &ScaffoldRecipe) -> Result<Value, String> {
    if recipe.inputs.is_empty() || recipe.outputs.is_empty() {
        return Err("可编辑迁移需要具名输入和输出。".into());
    }
    let (input_args, output_args) = recipe.arguments()?;
    let (inputs, _) = inputs_from_args(&input_args)?;
    let route_args = recipe.routes.iter().map(|(slot, source)| format!("{slot}={source}")).collect::<Vec<_>>();
    let routes = resolve_routes(&inputs, &route_args)?;
    if recipe.encoder_layers.keys().any(|key| !inputs.iter().any(|input| *key == format!("{}_encoder", input.name))) {
        return Err("编码器没有对应输入。".into());
    }
    let valid_slot = |slot: &str| slot == "fusion" || slot == "head"
        || inputs.iter().any(|input| format!("{}_encoder", input.name) == slot);
    if recipe.components.iter().any(|(slot, reference)| !valid_slot(slot)
        || reference.len() > 240 || !(reference.starts_with("component:") || python_entrypoint(reference)))
        || recipe.component_params.iter().any(|(slot, params)| !recipe.components.contains_key(slot)
            || params.len() > 64 || serde_json::to_vec(params).map_or(true, |value| value.len() > 65_536))
        || recipe.encoder_widths.iter().any(|(slot, width)| !valid_slot(slot) || slot == "fusion" || slot == "head"
            || !recipe.components.contains_key(slot) || !(1..=1_000_000).contains(width))
        || recipe.native_encoders.iter().any(|slot| !valid_slot(slot) || slot == "fusion" || slot == "head")
        || recipe.native_encoders.iter().collect::<HashSet<_>>().len() != recipe.native_encoders.len() {
        return Err("组件接缝、参数、宽度或原始输出声明无效。".into());
    }
    let mut used: HashSet<String> = recipe.inputs.iter().map(|item| item.name.clone())
        .chain(recipe.outputs.iter().map(|item| item.name.clone())).collect();
    fn fresh(used: &mut HashSet<String>, base: &str) -> String {
        let mut candidate = base.to_owned();
        let mut index = 1;
        while !used.insert(candidate.clone()) { candidate = format!("{base}_{index}"); index += 1; }
        candidate
    }
    let output_ids = recipe.outputs.iter().enumerate().map(|(index, output)| {
        if recipe.inputs.iter().any(|input| input.name == output.name) {
            fresh(&mut used, &format!("m_output_{index}"))
        } else { output.name.clone() }
    }).collect::<Vec<_>>();
    let mut nodes = inputs.iter().map(|input| json!({"id":input.name,"kind":"input","inputs":[],"outputs":["value"]})).collect::<Vec<_>>();
    let mut edges = Vec::new();
    let mut definitions = Vec::new();
    let mut branch_ends: Vec<String> = Vec::new();
    let mut total_width = 0usize;
    let (mut layer_count, mut parameter_count, mut layer_ids) = (0, 0, HashSet::new());
    for (branch, input) in inputs.iter().enumerate() {
        let source = &inputs[routes[branch]];
        let slot = format!("{}_encoder", input.name);
        let mut width = flattened_width(source);
        let layers = recipe.encoder_layers.get(&slot).map(Vec::as_slice).unwrap_or(&[]);
        let component = recipe.components.get(&slot);
        let native = recipe.native_encoders.contains(&slot);
        if !layers.is_empty() && (component.is_some() || native) {
            return Err(format!("编码器 {slot} 的层栈不能同时使用组件或原始输出。"));
        }
        if !layers.is_empty() {
            resolve_stack(layers, &slot, width, 0, &mut layer_count, &mut parameter_count, &mut layer_ids)?;
        }
        let mut builder = EditableGraphBuilder { nodes: &mut nodes, edges: &mut edges, definitions: &mut definitions,
            used: &mut used, branch, counter: 0, dtype_definitions: Vec::new(), nested: false };
        // The original starter evaluates encoders in declaration order. Keep
        // that order even when independent branches contain random or stateful
        // user layers; the edge is only an execution dependency.
        let mut previous = if let Some(prior) = branch_ends.last() {
            builder.after(&source.name, prior)
        } else { source.name.clone() };
        if let Some(factory) = component {
            let definition = format!("m_encoder_def_{branch}");
            let id = builder.fresh(&format!("m_encoder_{branch}"));
            let params = recipe.component_params.get(&slot).cloned().unwrap_or_default();
            builder.definitions.push(json!({"id":definition,"factory":factory,"params":params}));
            builder.unary(&previous, json!({"id":id,"kind":"call","definition":definition,"call_style":"positional","inputs":["value"],"outputs":["value"]}), &id);
            previous = id;
            width = recipe.encoder_widths.get(&slot).copied().unwrap_or(width);
        }
        if !native && !source.dynamic_axes.is_empty() {
            let id = builder.fresh(&format!("m_dynamic_mean_{branch}"));
            builder.unary(&previous, json!({"id":id,"kind":"mean_dynamic","axes":source.dynamic_axes,"inputs":["value"],"outputs":["value"]}), &id);
            previous = id;
        }
        if !native {
            let flatten = builder.fresh(&format!("m_flatten_{branch}"));
            let flatten_def = format!("m_flatten_def_{branch}");
            builder.definitions.push(json!({"id":flatten_def,"factory":"torch.nn:Flatten","params":{"start_dim":1}}));
            builder.unary(&previous, json!({"id":flatten,"kind":"call","definition":flatten_def,"call_style":"positional","inputs":["value"],"outputs":["value"]}), &flatten);
            previous = flatten;
        }
        let mut encoder_cast = None;
        if !layers.is_empty() {
            let id = builder.fresh(&format!("m_encoder_cast_{branch}"));
            builder.unary(&previous, json!({"id":id,"kind":"cast","inputs":["value"],"outputs":["value"]}), &id);
            previous = id.clone(); encoder_cast = Some(id);
        }
        let (end, output_width) = builder.stack(layers, previous, width, &source.name)?;
        width = output_width;
        if let Some(id) = encoder_cast {
            let node = builder.nodes.iter_mut().find(|node| node["id"] == id).ok_or("缺少类型转换节点。")?;
            node["dtype_definitions"] = json!(builder.dtype_definitions);
            if !builder.nested && !recipe.components.contains_key("fusion") { node["track_sequence"] = json!(true); }
        }
        let previous = if recipe.components.contains_key("fusion") { end } else {
            let cast = builder.fresh(&format!("m_fusion_cast_{branch}"));
            builder.unary(&end, json!({"id":cast,"kind":"cast","inputs":["value"],"outputs":["value"]}), &cast);
            cast
        };
        drop(builder);
        branch_ends.push(previous);
        total_width = total_width.checked_add(width).filter(|value| *value <= 1_000_000).ok_or("融合输出宽度过大。")?;
    }
    named_outputs_from_args(&output_args, total_width)?;
    let previous = if let Some(factory) = recipe.components.get("fusion") {
        let id = fresh(&mut used, "m_custom_fusion");
        let ports = (0..branch_ends.len()).map(|index| format!("input_{index}")).collect::<Vec<_>>();
        definitions.push(json!({"id":"m_custom_fusion_def","factory":factory,
            "params":recipe.component_params.get("fusion").cloned().unwrap_or_default()}));
        nodes.push(json!({"id":id,"kind":"call","definition":"m_custom_fusion_def","call_style":"list","inputs":ports,"outputs":["value"]}));
        for (index, source) in branch_ends.iter().enumerate() {
            edges.push(json!({"from":{"node":source,"port":"value"},"to":{"node":id,"port":format!("input_{index}")}}));
        }
        id
    } else if branch_ends.len() == 1 { branch_ends.remove(0) } else {
        let id = fresh(&mut used, "m_fusion_concat");
        let ports = (0..branch_ends.len()).map(|index| format!("input_{index}")).collect::<Vec<_>>();
        nodes.push(json!({"id":id,"kind":"concat","axis":1,"inputs":ports,"outputs":["value"]}));
        for (index, source) in branch_ends.iter().enumerate() {
            edges.push(json!({"from":{"node":source,"port":"value"},"to":{"node":id,"port":format!("input_{index}")}}));
        }
        id
    };
    if let Some(factory) = recipe.components.get("head") {
        let id = fresh(&mut used, "m_custom_head");
        definitions.push(json!({"id":"m_custom_head_def","factory":factory,
            "params":recipe.component_params.get("head").cloned().unwrap_or_default()}));
        let ports = recipe.outputs.iter().map(|output| output.name.clone()).collect::<Vec<_>>();
        nodes.push(json!({"id":id,"kind":"call","definition":"m_custom_head_def","call_style":"positional",
            "output_style":"mapping","inputs":["value"],"outputs":ports}));
        edges.push(json!({"from":{"node":previous,"port":"value"},"to":{"node":id,"port":"value"}}));
        for (index, output) in recipe.outputs.iter().enumerate() {
            let output_id = &output_ids[index];
            let mut node = json!({"id":output_id,"kind":"output","inputs":["value"],"outputs":[]});
            if output_id != &output.name { node["output_name"] = json!(output.name); }
            nodes.push(node);
            edges.push(json!({"from":{"node":id,"port":output.name},"to":{"node":output_id,"port":"value"}}));
        }
    } else { for (index, output) in recipe.outputs.iter().enumerate() {
        let head = fresh(&mut used, &format!("m_head_{index}"));
        let definition = format!("m_head_def_{index}");
        definitions.push(json!({"id":definition,"factory":"torch.nn:Linear","params":{"in_features":total_width,"out_features":output.features}}));
        nodes.push(json!({"id":head,"kind":"call","definition":definition,"call_style":"positional","inputs":["value"],"outputs":["value"]}));
        let output_id = &output_ids[index];
        let mut node = json!({"id":output_id,"kind":"output","inputs":["value"],"outputs":[]});
        if output_id != &output.name { node["output_name"] = json!(output.name); }
        nodes.push(node);
        edges.push(json!({"from":{"node":previous,"port":"value"},"to":{"node":head,"port":"value"}}));
        edges.push(json!({"from":{"node":head,"port":"value"},"to":{"node":output_id,"port":"value"}}));
    }}
    let objectives = recipe.outputs.iter().enumerate().map(|(index, output)| json!({"output":output_ids[index],"target":output.name,
        "loss":output.loss.strip_prefix("custom:").unwrap_or(&output.loss),"weight":output.weight})).collect::<Vec<_>>();
    let mut graph = json!({"version":1,"definitions":definitions,"nodes":nodes,"edges":edges,"objectives":objectives});
    if let Some(data) = &recipe.data { graph["data"] = json!(data); }
    if let Some(reference) = &recipe.supervision { graph["supervision"] = json!(reference); }
    if let Some(training) = &recipe.training { graph["training"] = graph_training_from_recipe(training)?; }
    Ok(graph)
}

/// Construct a read-only wrapper proposal. It references the original model
/// factory rather than pretending to recover editable internal layers.
pub(super) fn graph_import_preview<B: Backend>(app: &Application<B>, path: &Path, expected: &str, model: &str,
    expected_project: Option<&str>, editable: bool) -> Report {
    let fail = |code, message: String| Report::error("graph_import_scaffold", code, message,
        "原声明和源码未修改；检查模型工厂、名称和声明后重试。");
    let bytes = match read_recipe_for_update(path) {
        Ok(bytes) => bytes, Err(error) => return fail("RECIPE_READ_FAILED", error),
    };
    let revision = sha256_bytes(&bytes);
    if revision != expected { return fail("RECIPE_CONFLICT", "声明文件内容已变化。".into()); }
    if model.len() > 240 || !python_entrypoint(model) {
        return fail("GRAPH_IMPORT_MODEL_INVALID", "模型工厂须为 module:callable。".into());
    }
    let project_data = if let Some(expected_project) = expected_project {
        let root = match std::env::current_dir().and_then(|value| value.canonicalize()) {
            Ok(value) => value, Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
        };
        if !path.canonicalize().is_ok_and(|value| value.parent() == Some(root.as_path())) {
            return fail("GRAPH_IMPORT_PROJECT_INVALID", "引用项目数据时，声明须位于项目根目录。".into());
        }
        let config = match app.project_config.load(&root) {
            Ok(Some(value)) => value, Ok(None) => return fail("PROJECT_NOT_INITIALIZED", "项目配置不存在。".into()),
            Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
        };
        if crate::domain::input::content_revision(&config).ok().as_deref() != Some(expected_project)
            || config["model"].as_str() != Some(model) {
            return fail("PROJECT_CONFIG_CONFLICT", "项目配置或模型入口已变化。".into());
        }
        config["batch"].as_str().map(|factory| {
            let mut data = json!({"factory":factory,"params":config.get("batch_params").cloned().unwrap_or_else(|| json!({}))});
            if let Some(binding) = config["batch_binding"].as_str() { data["binding"] = json!(binding); }
            data
        })
    } else { None };
    let raw: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value, Err(error) => return fail("RECIPE_INVALID", error.to_string()),
    };
    let content = match String::from_utf8(bytes.clone()) {
        Ok(value) => value, Err(error) => return fail("RECIPE_INVALID", error.to_string()),
    };
    if raw.get("kind").and_then(Value::as_str) == Some("image_classification") {
        if editable { return fail("GRAPH_IMPORT_UNSUPPORTED", "分类模型内部结构不能从分类声明可靠推断；请使用包装预览或源码。".into()); }
        // The classification source may have been edited or replaced. Its
        // preset and class list do not prove the current callable's signature,
        // output structure, transforms, or loss, so none is reconstructed.
        if let Err(error) = super::classification::parse_classification_recipe(&bytes) {
            return fail("RECIPE_INVALID", error);
        }
        let mut graph = json!({"version":1,
            "definitions":[{"id":"original_model","factory":model,"params":{}}],
            "nodes":[{"id":"images","kind":"input","inputs":[],"outputs":["value"]},
                {"id":"wrapped_model","kind":"call","definition":"original_model","call_style":"positional","inputs":["images"],"outputs":["value"]},
                {"id":"prediction","kind":"output","inputs":["value"],"outputs":[]}],
            "edges":[{"from":{"node":"images","port":"value"},"to":{"node":"wrapped_model","port":"images"}},
                {"from":{"node":"wrapped_model","port":"value"},"to":{"node":"prediction","port":"value"}}],
            "objectives":[]});
        if let Some(reference) = raw.get("supervision").filter(|value| !value.is_null()) {
            graph["supervision"] = reference.clone();
        }
        if let Some(data) = &project_data { graph["data"] = data.clone(); }
        if let Some(value) = raw.get("training").filter(|value| !value.is_null()) {
            let profile: RecipeTraining = match serde_json::from_value(value.clone()) {
                Ok(profile) => profile, Err(error) => return fail("RECIPE_INVALID", error.to_string()),
            };
            graph["training"] = match graph_training_from_recipe(&profile) {
                Ok(training) => training, Err(error) => return fail("RECIPE_INVALID", error),
            };
        }
        graph["legacy_import"] = json!({"kind":"image_classification","sha256":revision,"content":content});
        if serde_json::to_vec(&graph).map_or(true, |value| value.len() > 128 * 1024) {
            return fail("GRAPH_IMPORT_TOO_LARGE", "保留旧声明的图超过 128 KiB；原文件未修改。".into());
        }
        return Report::ok("graph_import_scaffold", json!({"committed":false,"recipe_sha256":revision,"graph":graph,
            "representation":"opaque_model_wrapper","source_kind":"image_classification","project_revision":expected_project,
            "data_origin":if project_data.is_some() {"project_config"} else {"not_declared"},
            "limitations":["图按单个位置参数及单个返回值包装工厂；源码可能已修改，必须在运行前核对。",
                "项目批次入口如存在仅作为引用保留；分类变换、类别及损失未转换，训练兼容性未验证。",
                "原模型内部层及权重未转换；监督引用不会自动导入或重新绑定。"],
            "runtime_verified":false}));
    }
    let recipe = match ScaffoldRecipe::parse(&bytes) {
        Ok(recipe) => recipe, Err(error) => return fail("RECIPE_INVALID", error),
    };
    if let Err(error) = recipe.arguments() { return fail("RECIPE_INVALID", error); }
    if recipe.inputs.is_empty() || recipe.outputs.is_empty() || recipe.inputs.len() > 32 || recipe.outputs.len() > 32 {
        return fail("GRAPH_IMPORT_UNSUPPORTED", "需要 1 至 32 个具名输入和输出；请继续使用原声明或源码。".into());
    }
    let mut names = HashSet::new();
    for group in [recipe.inputs.iter().map(|item| &item.name).collect::<Vec<_>>(), recipe.outputs.iter().map(|item| &item.name).collect::<Vec<_>>()] {
        let mut group_names = HashSet::new();
        for name in group {
            if name.len() > 128 || !python_identifier(name) || !group_names.insert(name.clone()) {
                return fail("GRAPH_IMPORT_UNSUPPORTED", format!("同一输入或输出组中的名称须为唯一标识：{name}。"));
            }
            names.insert(name.clone());
        }
    }
    if editable {
        let mut graph = match editable_graph_from_recipe(&recipe) {
            Ok(graph) => graph, Err(error) => return fail("GRAPH_IMPORT_UNSUPPORTED", error),
        };
        let data_origin = if graph.get("data").is_some() { "recipe" } else if let Some(data) = &project_data {
            graph["data"] = data.clone(); "project_config"
        } else { "not_declared" };
        graph["legacy_import"] = json!({"kind":"scaffold","sha256":revision,"content":content});
        if serde_json::to_vec(&graph).map_or(true, |value| value.len() > 128 * 1024) {
            return fail("GRAPH_IMPORT_TOO_LARGE", "保留旧声明的图超过 128 KiB；原文件未修改。".into());
        }
        return Report::ok("graph_import_scaffold", json!({"committed":false,"recipe_sha256":revision,"graph":graph,
            "representation":"editable_recipe_layers","project_revision":expected_project,"data_origin":data_origin,"runtime_verified":false,
            "limitations":["此图仅由声明推导；现有源码、组件注入和模型工厂可能已修改，未比较运行行为。",
                "权重和检查点未迁移；训练声明如存在已保留但未经验证，监督引用需重新导入和绑定。",
                "层栈 dtype 引用保留迁移时的组件顺序；重排或移除层后须检查类型转换节点引用。",
                "自定义融合接收按声明顺序排列的列表；自定义输出头须返回与声明输出名一致的字典，实际工厂需在运行时核对。"]}));
    }
    let mut wrapper = "wrapped_model".to_owned();
    let mut suffix = 1;
    while names.contains(&wrapper) { wrapper = format!("wrapped_model_{suffix}"); suffix += 1; }
    let input_names = recipe.inputs.iter().map(|item| item.name.clone()).collect::<Vec<_>>();
    let output_names = recipe.outputs.iter().map(|item| item.name.clone()).collect::<Vec<_>>();
    let output_ids = recipe.outputs.iter().enumerate().map(|(index, output)| {
        if input_names.contains(&output.name) {
            let base = format!("m_output_{index}");
            let mut id = base.clone();
            let mut suffix = 1;
            while names.contains(&id) || id == wrapper { id = format!("{base}_{suffix}"); suffix += 1; }
            names.insert(id.clone()); id
        } else { output.name.clone() }
    }).collect::<Vec<_>>();
    let mut nodes = input_names.iter().map(|name| json!({"id":name,"kind":"input","inputs":[],"outputs":["value"]})).collect::<Vec<_>>();
    nodes.push(json!({"id":wrapper,"kind":"call","definition":"original_model","call_style":"keyword",
        "output_style":"mapping","inputs":input_names,"outputs":output_names}));
    nodes.extend(recipe.outputs.iter().enumerate().map(|(index, item)| {
        let mut node = json!({"id":output_ids[index],"kind":"output","inputs":["value"],"outputs":[]});
        if output_ids[index] != item.name { node["output_name"] = json!(item.name); }
        node
    }));
    let mut edges = recipe.inputs.iter().map(|item| json!({"from":{"node":item.name,"port":"value"},
        "to":{"node":wrapper,"port":item.name}})).collect::<Vec<_>>();
    edges.extend(recipe.outputs.iter().enumerate().map(|(index, item)| json!({"from":{"node":wrapper,"port":item.name},
        "to":{"node":output_ids[index],"port":"value"}})));
    let objectives = recipe.outputs.iter().enumerate().map(|(index, item)| json!({"output":output_ids[index],"target":item.name,
        "loss":item.loss.strip_prefix("custom:").unwrap_or(&item.loss),"weight":item.weight})).collect::<Vec<_>>();
    let mut graph = json!({"version":1,"definitions":[{"id":"original_model","factory":model,"params":{}}],
        "nodes":nodes,"edges":edges,"objectives":objectives});
    let data_origin = if let Some(data) = recipe.data { graph["data"] = json!(data); "recipe" }
        else if let Some(data) = &project_data { graph["data"] = data.clone(); "project_config" }
        else { "not_declared" };
    if let Some(reference) = recipe.supervision { graph["supervision"] = json!(reference); }
    if let Some(training) = &recipe.training {
        graph["training"] = match graph_training_from_recipe(training) {
            Ok(training) => training, Err(error) => return fail("RECIPE_INVALID", error),
        };
    }
    graph["legacy_import"] = json!({"kind":"scaffold","sha256":revision,"content":content});
    if serde_json::to_vec(&graph).map_or(true, |value| value.len() > 128 * 1024) {
        return fail("GRAPH_IMPORT_TOO_LARGE", "保留旧声明的图超过 128 KiB；原文件未修改。".into());
    }
    Report::ok("graph_import_scaffold", json!({"committed":false,"recipe_sha256":revision,"graph":graph,
        "representation":"opaque_model_wrapper","project_revision":expected_project,"data_origin":data_origin,
        "limitations":["原模型内部层、路由、组件参数和权重不转换为图节点；原声明及源码必须保留。",
            "工厂在此未导入；运行时须核对工厂参数、输入和具名输出。",
            "训练声明如存在已保留但未经验证；监督引用不会自动导入或重新绑定。"],
        "runtime_verified":false}))
}

fn inputs_from_args(args: &[String]) -> Result<(Vec<InputSpec>, usize), String> {
    if args.is_empty() || args.len() > 32 {
        return Err("需要 1 至 32 个具名输入。".into());
    }
    let mut names = HashSet::new();
    let mut normalized_names = std::collections::HashMap::<String, String>::new();
    let mut total = 0usize;
    let mut sample_total = 0usize;
    let mut inputs = Vec::new();
    for arg in args {
        let (typed_name, dimensions) = arg
            .split_once('=')
            .ok_or("输入格式应为 name[:dtype]=3,16,16。")?;
        let (name, dtype) = match typed_name.split_once(':') {
            Some((name, dtype)) => (name, Some(dtype)),
            None => (typed_name, None),
        };
        if let Some(dtype) = dtype {
            let mut dtype_chars = dtype.chars();
            if dtype.len() > 64 || !dtype_chars.next().is_some_and(|c| c.is_ascii_alphabetic())
                || !dtype_chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(format!(
                    "输入 `{name}` 的 dtype `{dtype}` 必须是 PyTorch dtype 的 ASCII 名称。"
                ));
            }
        }
        if name.len() > 48
            || name.starts_with('_')
            || !python_identifier(name)
            || !names.insert(name.to_owned())
        {
            return Err(format!(
                "输入名 `{name}` 必须唯一、符合 Python 标识符规则、不能以下划线开头且不超过 48 字节。"
            ));
        }
        let normalized: String = name.nfkc().collect();
        if normalized.starts_with('_') {
            return Err(format!("输入名 `{name}` 的 NFKC 形式以下划线开头，请选择其他名称。"));
        }
        if let Some(previous) = normalized_names.insert(normalized, name.to_owned()) {
            return Err(format!("输入名 `{name}` 与 `{previous}` 的 NFKC 形式相同；请选择可区分的名称。"));
        }
        let mut dynamic_axes = Vec::new();
        let shape: Vec<usize> = dimensions
            .split(',')
            .enumerate()
            .map(|(axis, part)| {
                let (dynamic, value) = match part.strip_prefix('?') {
                    Some(value) => (true, value),
                    None => (false, part),
                };
                if dynamic {
                    dynamic_axes.push(axis);
                }
                value.parse::<usize>()
            })
            .collect::<Result<_, _>>()
            .map_err(|_| "形状必须由正整数或 ?代表长度组成，以逗号分隔。")?;
        if shape.is_empty() || shape.len() > 8 || shape.contains(&0) {
            return Err("每个输入需要 1 至 8 个正维度，不包含批次轴。".into());
        }
        let sample_size = shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or("输入形状过大。")?;
        sample_total = sample_total
            .checked_add(sample_size)
            .ok_or("输入形状过大。")?;
        if sample_total > 1_000_000 {
            return Err("合计代表样本特征不能超过 1000000。".into());
        }
        let size = shape
            .iter()
            .enumerate()
            .try_fold(1usize, |a, (axis, b)| {
                a.checked_mul(if dynamic_axes.contains(&axis) { 1 } else { *b })
            })
            .ok_or("输入形状过大。")?;
        total = total.checked_add(size).ok_or("输入形状过大。")?;
        if total > 1_000_000 {
            return Err("合计输入特征不能超过 1000000。".into());
        }
        inputs.push(InputSpec {
            name: name.to_owned(),
            shape,
            dynamic_axes,
            dtype: dtype.map(str::to_owned),
        });
    }
    Ok((inputs, total))
}

fn resolve_routes(inputs: &[InputSpec], args: &[String]) -> Result<Vec<usize>, String> {
    let mut routes: Vec<usize> = (0..inputs.len()).collect();
    let mut assigned = HashSet::new();
    for arg in args {
        let (slot, input) = arg.split_once('=').ok_or("接线格式应为 encoder=输入名。")?;
        let target = inputs
            .iter()
            .position(|item| format!("{}_encoder", item.name) == slot)
            .ok_or_else(|| {
                let normalized: String = slot.nfkc().collect();
                match inputs.iter().map(|item| format!("{}_encoder", item.name))
                    .find(|candidate| candidate.nfkc().eq(normalized.chars())) {
                    Some(candidate) => format!("未知编码器接缝 `{slot}`；它的 NFKC 形式与 `{candidate}` 相同，请精确使用已声明的名称。"),
                    None => format!("未知编码器接缝 `{slot}`。"),
                }
            })?;
        let source = inputs
            .iter()
            .position(|item| item.name == input)
            .ok_or_else(|| {
                let normalized: String = input.nfkc().collect();
                match inputs.iter().find(|candidate| candidate.name.nfkc().eq(normalized.chars())) {
                    Some(candidate) => format!("未知输入 `{input}`；它的 NFKC 形式与 `{}` 相同，请精确使用已声明的名称。", candidate.name),
                    None => format!("未知输入 `{input}`。"),
                }
            })?;
        if !assigned.insert(target) {
            return Err(format!("接缝 `{slot}` 被重复接线。"));
        }
        routes[target] = source;
    }
    Ok(routes)
}

#[derive(Clone, PartialEq, Eq)]
enum Objective {
    Mse,
    CrossEntropy,
    Custom(String),
}
impl Objective {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "mse" => Ok(Self::Mse),
            "ce" | "cross_entropy" => Ok(Self::CrossEntropy),
            _ if value.strip_prefix("custom:").is_some_and(|reference| reference.len() <= 240 && python_entrypoint(reference)) => Ok(Self::Custom(value.to_owned())),
            _ => Err("损失必须是 mse、ce 或 custom:module:callable。".into()),
        }
    }
    fn name(&self) -> &str {
        match self {
            Self::Mse => "mse",
            Self::CrossEntropy => "cross_entropy",
            Self::Custom(reference) => reference,
        }
    }
}

fn named_outputs_from_args(
    args: &[String],
    features: usize,
) -> Result<Vec<(String, usize, Objective, f64)>, String> {
    if args.len() > 32 {
        return Err("最多声明 32 个具名输出。".into());
    }
    let mut seen = HashSet::new();
    let mut normalized_names = std::collections::HashMap::<String, String>::new();
    let mut total = 0usize;
    let mut result = Vec::new();
    for arg in args {
        let (name, raw) = arg.split_once('=').ok_or("输出格式应为 name=特征数。")?;
        if name.len() > 48
            || name.starts_with('_')
            || !python_identifier(name)
            || !seen.insert(name.to_owned())
        {
            return Err(format!(
                "输出名 `{name}` 必须唯一、符合 Python 标识符规则、不能以下划线开头且不超过 48 字节。"
            ));
        }
        let normalized: String = name.nfkc().collect();
        if normalized.starts_with('_') {
            return Err(format!("输出名 `{name}` 的 NFKC 形式以下划线开头，请选择其他名称。"));
        }
        if let Some(previous) = normalized_names.insert(normalized, name.to_owned()) {
            return Err(format!("输出名 `{name}` 与 `{previous}` 的 NFKC 形式相同；请选择可区分的名称。"));
        }
        let (descriptor, raw_weight) = raw.split_once('@').unwrap_or((raw, "1"));
        let weight = raw_weight
            .parse::<f64>()
            .map_err(|_| "损失权重必须是正的有限数字。")?;
        if !weight.is_finite() || weight <= 0.0 {
            return Err("损失权重必须是正的有限数字。".into());
        }
        let (raw_width, raw_loss) = descriptor.split_once(':').unwrap_or((descriptor, "mse"));
        let objective = Objective::parse(raw_loss)?;
        let width = raw_width
            .parse::<usize>()
            .map_err(|_| "输出特征数必须为正整数。")?;
        if width == 0 || width > 65_536 {
            return Err("输出特征数必须在 1 至 65536 之间。".into());
        }
        if objective == Objective::CrossEntropy && width < 2 {
            return Err("交叉熵输出至少需要两个类别。".into());
        }
        total = total.checked_add(width).ok_or("输出特征总数过大。")?;
        result.push((name.to_owned(), width, objective, weight));
    }
    if features.checked_mul(total).is_none_or(|n| n > 10_000_000) {
        return Err("所有默认输出头的总权重不能超过 10000000。".into());
    }
    Ok(result)
}

fn source(
    inputs: &[InputSpec],
    routes: &[usize],
    native_encoders: &HashSet<String>,
    layered_encoders: &BTreeMap<String, ResolvedEncoder>,
    encoder_widths: &BTreeMap<String, usize>,
    output: usize,
    named_outputs: &[(String, usize, Objective, f64)],
    objective: Objective,
) -> String {
    let mut text = String::from(
        "# Editable Model-Maven starter. Replace encoders, fusion, head and step as needed.\n# Batch values and targets are synthetic; they are not a dataset.\n# Default Concat casts features to the PyTorch default float dtype for its Linear head.\n# Replace fusion when your head uses another dtype or output contract.\nimport torch\nfrom torch import nn\n\n\ndef _resolve_dtype(name):\n    dtype = getattr(torch, name, None)\n    if not isinstance(dtype, torch.dtype):\n        raise ValueError(f\"PyTorch dtype not available: {name}\")\n    return dtype\n\n\nclass Concat(nn.Module):\n    def forward(self, values):\n        return torch.cat([value.to(dtype=torch.get_default_dtype()) for value in values], dim=1)\n\n\n",
    );
    let has_mse = (named_outputs.is_empty() && matches!(&objective, Objective::Mse))
        || named_outputs.iter().any(|(_, _, loss, _)| matches!(loss, Objective::Mse));
    if has_mse {
        text.push_str(r#"def _mse_loss(prediction, target, name):
    # Reject implicit broadcasting: an accidental target width must not train silently.
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError(f"MSE output {name!r} and its target must both be tensors")
    if prediction.shape != target.shape:
        raise ValueError(f"MSE output {name!r} shape {tuple(prediction.shape)} does not match target {tuple(target.shape)}")
    return (prediction - target).square().mean()


"#);
    }
    let has_custom_loss = matches!(&objective, Objective::Custom(_)) || named_outputs.iter().any(|(_, _, loss, _)| matches!(loss, Objective::Custom(_)));
    if has_custom_loss {
        text.push_str("import importlib\n\n\ndef _custom_loss(reference, prediction, target):\n    # User code is imported only when this training step actually runs.\n    module, name = reference.split(':', 1)\n    function = importlib.import_module(module)\n    for part in name.split('.'):\n        function = getattr(function, part)\n    loss = function(prediction, target)\n    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:\n        raise ValueError(f'{reference} must return a scalar loss tensor')\n    return loss.reshape(())\n\n\n");
    }
    if layered_encoders.values().any(|encoder| encoder.constructors.iter().any(|value| value.contains("_make_layer("))) {
        text.push_str("import importlib\nimport json\n\n\ndef _make_layer(reference, parameters_json):\n    module, name = reference.split(':', 1)\n    factory = importlib.import_module(module)\n    for part in name.split('.'):\n        factory = getattr(factory, part)\n    layer = factory(**json.loads(parameters_json))\n    if not isinstance(layer, nn.Module):\n        raise TypeError(f'{reference} did not return a torch.nn.Module')\n    return layer\n\n\n");
    }
    if layered_encoders.values().any(|encoder| !encoder.width_checks.is_empty() || encoder.constructors.iter().any(|value| value.contains("_CheckedStack("))) {
        text.push_str("class _CheckedStack(nn.Sequential):\n    # Factory layers in this stack must preserve the batch and return flat features.\n    def __init__(self, checks, *layers):\n        super().__init__(*layers)\n        self._width_checks = checks\n\n    def forward(self, value):\n        batch_size = value.shape[0]\n        for index, layer in enumerate(self):\n            value = layer(value)\n            if index in self._width_checks:\n                width, reference = self._width_checks[index]\n                if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != batch_size or value.shape[1] != width:\n                    actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__\n                    raise ValueError(f'Layer {index + 1} ({reference}) declared output (batch, {width}); got {actual}')\n        return value\n\n\n");
    }
    if layered_encoders.values().any(|encoder| encoder.constructors.iter().any(|value| value.contains("_Residual("))) {
        text.push_str("class _Residual(nn.Module):\n    # Protect the skip connection from in-place branch operations.\n    def __init__(self, branch):\n        super().__init__()\n        self.branch = branch\n\n    def forward(self, value):\n        result = self.branch(value.clone())\n        if not isinstance(result, torch.Tensor) or result.shape != value.shape:\n            raise ValueError(f'Residual branch shape {getattr(result, \"shape\", None)} does not match input {value.shape}')\n        return value + result\n\n\n");
    }
    if layered_encoders.values().any(|encoder| encoder.constructors.iter().any(|value| value.contains("_Parallel("))) {
        text.push_str("class _Parallel(nn.Module):\n    # Each branch sees the same unmodified input.\n    def __init__(self, merge, *branches):\n        super().__init__()\n        self.merge = merge\n        self.branches = nn.ModuleList(branches)\n\n    def forward(self, value):\n        outputs = [branch(value.clone()) for branch in self.branches]\n        if any(not isinstance(item, torch.Tensor) or item.ndim != 2 or item.shape[0] != value.shape[0] for item in outputs):\n            raise ValueError('Parallel branches must return rank-2 tensors with the input batch size')\n        if self.merge == 'concat':\n            return torch.cat(outputs, dim=1)\n        if any(item.shape != outputs[0].shape for item in outputs[1:]):\n            raise ValueError('Parallel add requires equal branch shapes')\n        result = outputs[0]\n        for item in outputs[1:]:\n            result = result + item\n        return result\n\n\n");
    }
    if layered_encoders.values().any(|encoder| !encoder.layer_paths.is_empty()) {
        text.push_str("# Declarative IDs map to module paths only for the generated default encoders.\n# Injected replacement encoders require a fresh observation.\nMODEL_MAVEN_LAYER_PATHS = {\n");
        for input in inputs {
            if let Some(encoder) = layered_encoders.get(&format!("{}_encoder", input.name)) {
                for (id, suffix) in &encoder.layer_paths {
                    let path = format!("model.encoders.enc_{}{suffix}", input.name);
                    writeln!(text, "    {}: {},", serde_json::to_string(id).unwrap(), serde_json::to_string(&path).unwrap()).unwrap();
                }
            }
        }
        text.push_str("}\n\n\n");
    }
    if !named_outputs.is_empty() {
        text.push_str("class NamedHeads(nn.Module):\n    def __init__(self, width):\n        super().__init__()\n        self.heads = nn.ModuleDict({\n");
        for (name, width, _, _) in named_outputs {
            writeln!(
                text,
                "            \"head_{name}\": nn.Linear(width, {width}),"
            )
            .unwrap();
        }
        text.push_str("        })\n\n    def forward(self, features):\n        return {\n");
        for (name, _, _, _) in named_outputs {
            writeln!(
                text,
                "            \"{name}\": self.heads[\"head_{name}\"](features),"
            )
            .unwrap();
        }
        text.push_str("        }\n\n\n");
    }
    let has_unicode_inputs = inputs.iter().any(|input| !input.name.is_ascii());
    let mut params = inputs
        .iter()
        .filter(|input| input.name.is_ascii())
        .map(|input| format!("{}_encoder=None", input.name))
        .chain([
            "fusion=None".into(),
            "head=None".into(),
            "input_routes=None".into(),
            "encoder_output_modes=None".into(),
        ])
        .collect::<Vec<_>>();
    if has_unicode_inputs { params.push("**_m_unicode_encoders".into()); }
    let params = params.join(", ");
    writeln!(text, "class Model(nn.Module):\n    def __init__(self, {params}):\n        super().__init__()").unwrap();
    if has_unicode_inputs {
        let slots = inputs.iter().filter(|input| !input.name.is_ascii())
            .map(|input| format!("\"{}_encoder\"", input.name)).collect::<Vec<_>>().join(", ");
        writeln!(text, "        _m_unknown = set(_m_unicode_encoders) - {{{slots}}}\n        if _m_unknown:\n            raise TypeError(f\"Unknown encoder arguments: {{sorted(_m_unknown)}}\")").unwrap();
    }
    if !layered_encoders.is_empty() {
        text.push_str("        self.default_layer_stacks = set()\n");
        for slot in layered_encoders.keys() {
            let value = if slot.is_ascii() { slot.clone() } else { format!("_m_unicode_encoders.get(\"{slot}\")") };
            writeln!(text, "        if {value} is None:\n            self.default_layer_stacks.add(\"{slot}\")").unwrap();
        }
    }
    text.push_str("        self.encoders = nn.ModuleDict({\n");
    for input in inputs {
        if let Some(encoder) = layered_encoders.get(&format!("{}_encoder", input.name)) {
            let value = if input.name.is_ascii() { format!("{}_encoder", input.name) }
                else { format!("_m_unicode_encoders.get(\"{}_encoder\")", input.name) };
            let stack = if encoder.width_checks.is_empty() { "nn.Sequential" } else { "_CheckedStack" };
            writeln!(text, "            \"enc_{}\": {value} if {value} is not None else {stack}(", input.name).unwrap();
            if !encoder.width_checks.is_empty() {
                let checks = encoder.width_checks.iter().map(|(index, width, reference)| {
                    format!("{index}: ({width}, {})", serde_json::to_string(reference).expect("string encoding"))
                }).collect::<Vec<_>>().join(", ");
                writeln!(text, "                {{{checks}}},").unwrap();
            }
            for (index, constructor) in encoder.constructors.iter().enumerate() {
                // Nested constructors currently share their enclosing expression's line.
                let prefix = format!(".{index}");
                let ids = encoder.layer_paths.iter().filter_map(|(id, path)| {
                    (path == &prefix || path.strip_prefix(&prefix).is_some_and(|suffix| suffix.starts_with('.')))
                        .then_some(id.as_str())
                }).collect::<Vec<_>>();
                if !ids.is_empty() {
                    writeln!(text, "                # Model-Maven layer IDs (constructor below): {}", ids.join(" ")).unwrap();
                }
                writeln!(text, "                {constructor},").unwrap();
            }
            text.push_str("            ),\n");
            continue;
        }
        let value = if input.name.is_ascii() { format!("{}_encoder", input.name) }
            else { format!("_m_unicode_encoders.get(\"{}_encoder\")", input.name) };
        writeln!(text, "            \"enc_{}\": {value} if {value} is not None else nn.Identity(),", input.name).unwrap();
    }
    text.push_str("        })\n        self.input_names = {");
    for (index, input) in inputs.iter().enumerate() {
        if index > 0 { text.push_str(", "); }
        write!(text, "\"{}\"", input.name).unwrap();
    }
    text.push_str("}\n        self.input_routes = {\n");
    for (index, input) in inputs.iter().enumerate() {
        writeln!(
            text,
            "            \"{}_encoder\": \"{}\",",
            input.name, inputs[routes[index]].name
        )
        .unwrap();
    }
    text.push_str("        }\n        self.input_features = {\n");
    for input in inputs {
        let features = flattened_width(input);
        writeln!(text, "            \"{}\": {},", input.name, features).unwrap();
    }
    text.push_str("        }\n        self.input_dynamic_axes = {\n");
    for input in inputs {
        writeln!(
            text,
            "            \"{}\": {:?},",
            input.name, input.dynamic_axes
        )
        .unwrap();
    }
    text.push_str("        }\n        if input_routes is not None:\n            if not isinstance(input_routes, dict):\n                raise ValueError(\"input_routes must be an encoder-to-input dictionary\")\n            for slot, source in input_routes.items():\n                if slot not in self.input_routes or source not in self.input_features:\n                    raise ValueError(f\"Unknown route {slot}={source}\")\n                self.input_routes[slot] = source\n        # A construction-time declaration, not proof of a runtime tensor edge.\n        self._model_maven_declared_routes = [\n            {\"slot\": slot, \"input\": source, \"module_path\": \"model.encoders.enc_\" + slot[:-8]}\n            for slot, source in self.input_routes.items()\n        ]\n        self.fusion = fusion if fusion is not None else Concat()\n");
    let native: Vec<_> = inputs
        .iter()
        .map(|input| format!("{}_encoder", input.name))
        .filter(|slot| native_encoders.contains(slot))
        .collect();
    writeln!(text, "        self.native_encoder_outputs = {native:?}").unwrap();
    text.push_str("        if encoder_output_modes is not None:\n            if not isinstance(encoder_output_modes, dict):\n                raise ValueError(\"encoder_output_modes must be an encoder-to-mode dictionary\")\n            for slot, mode in encoder_output_modes.items():\n                if slot not in self.input_routes or mode not in (\"native\", \"flatten\"):\n                    raise ValueError(f\"Unknown encoder output mode {slot}={mode}\")\n                if slot in self.native_encoder_outputs:\n                    self.native_encoder_outputs.remove(slot)\n                if mode == \"native\":\n                    self.native_encoder_outputs.append(slot)\n");
    let total_output: usize = if named_outputs.is_empty() {
        output
    } else {
        named_outputs.iter().map(|(_, width, _, _)| width).sum()
    };
    let default_head = if named_outputs.is_empty() {
        format!("nn.Linear(width, {output})")
    } else {
        "NamedHeads(width)".into()
    };
    if !layered_encoders.is_empty() {
        text.push_str("        # Declared stacks receive flat floating point features.\n        self.encoder_layer_widths = {\n");
        for (slot, encoder) in layered_encoders {
            writeln!(text, "            \"{slot}\": ({}, {}),", encoder.input_width, encoder.output_width).unwrap();
        }
        text.push_str("        }\n        for slot in self.default_layer_stacks:\n            input_width, _ = self.encoder_layer_widths[slot]\n            source = self.input_routes[slot]\n            if self.input_features[source] != input_width or slot in self.native_encoder_outputs:\n                raise ValueError(f\"Route for {slot} must supply {input_width} flattened features\")\n");
    }
    if !encoder_widths.is_empty() {
        text.push_str("        self.component_encoder_widths = {\n");
        for (slot, width) in encoder_widths {
            writeln!(text, "            \"{slot}\": {width},").unwrap();
        }
        text.push_str("        }\n");
    }
    let width = if !encoder_widths.is_empty() && !layered_encoders.is_empty() {
        "sum(self.encoder_layer_widths[slot][1] if slot in self.encoder_layer_widths else self.component_encoder_widths.get(slot, self.input_features[source]) for slot, source in self.input_routes.items())"
    } else if !encoder_widths.is_empty() {
        "sum(self.component_encoder_widths.get(slot, self.input_features[source]) for slot, source in self.input_routes.items())"
    } else if layered_encoders.is_empty() {
        "sum(self.input_features[source] for source in self.input_routes.values())"
    } else {
        "sum(self.encoder_layer_widths[slot][1] if slot in self.encoder_layer_widths else self.input_features[source] for slot, source in self.input_routes.items())"
    };
    write!(text, "        width = {width}\n        if head is None and (width > 1000000 or width * {total_output} > 10000000):\n            raise ValueError(\"Routed default head is too large; provide a custom head or change routes\")\n        self.head = head if head is not None else {default_head}\n\n    def _encode(self, slot, inputs):\n        source = self.input_routes[slot]\n").unwrap();
    if layered_encoders.is_empty() {
        text.push_str("        value = self.encoders[\"enc_\" + slot[:-8]](inputs[source])\n");
    } else {
        text.push_str("        encoder = self.encoders[\"enc_\" + slot[:-8]]\n        stacked = slot in self.default_layer_stacks\n        value = inputs[source] if stacked else encoder(inputs[source])\n");
    }
    text.push_str("        if slot in self.native_encoder_outputs:\n            return value\n        axes = self.input_dynamic_axes[source]\n        if axes:\n            if not (value.is_floating_point() or value.is_complex()):\n                value = value.to(dtype=torch.get_default_dtype())\n            value = value.mean(dim=tuple(axis + 1 for axis in axes), keepdim=True)\n");
    if layered_encoders.is_empty() {
        text.push_str("        return value.flatten(start_dim=1)\n");
    } else {
        text.push_str("        flattened = value.flatten(start_dim=1)\n        return encoder(flattened.to(dtype=self._stack_dtype(encoder))) if stacked else flattened\n");
    }
    if !layered_encoders.is_empty() {
        text.push_str("\n    def _stack_dtype(self, encoder):\n        parameter = next(encoder.parameters(), None)\n        return parameter.dtype if parameter is not None else torch.get_default_dtype()\n");
    }
    writeln!(text, "\n    def forward(self, **inputs):\n        if inputs.keys() != self.input_names:\n            missing = self.input_names - inputs.keys()\n            unknown = inputs.keys() - self.input_names\n            raise ValueError(f\"Input keys differ; missing={{sorted(missing)}}, unknown={{sorted(unknown)}}\")\n        values = [").unwrap();
    for input in inputs {
        writeln!(
            text,
            "            self._encode(\"{}_encoder\", inputs),",
            input.name
        )
        .unwrap();
    }
    text.push_str("        ]\n        return self.head(self.fusion(values))\n\n\n");
    let mut args = inputs
        .iter()
        .filter(|input| input.name.is_ascii())
        .map(|input| format!("{}_encoder={}_encoder", input.name, input.name))
        .chain([
            "fusion=fusion".into(),
            "head=head".into(),
            "input_routes=input_routes".into(),
            "encoder_output_modes=encoder_output_modes".into(),
        ])
        .collect::<Vec<_>>();
    if has_unicode_inputs { args.push("**_m_unicode_encoders".into()); }
    let args = args.join(", ");
    let mut batch_params = vec!["batch_size=2".to_owned()];
    for input in inputs {
        if !input.name.is_ascii() { continue; }
        for axis in &input.dynamic_axes {
            batch_params.push(format!(
                "{}_dim{}={}",
                input.name,
                axis + 1,
                input.shape[*axis]
            ));
        }
    }
    let has_unicode_dimensions = inputs.iter().any(|input| !input.name.is_ascii() && !input.dynamic_axes.is_empty());
    if has_unicode_dimensions { batch_params.push("**_m_dynamic_dims".into()); }
    writeln!(text, "def model({params}):\n    # For a non-ASCII input name, pass its encoder as **{{\"name_encoder\": module}}.\n    return Model({args})\n\n\n# Non-native encoder outputs are averaged on dynamic axes and flattened.\n# Batch lengths below only control synthetic samples; they do not constrain a dataset.\n# Non-ASCII dynamic dimensions are passed as **{{\"name_dimN\": length}}.\ndef batch({}):", batch_params.join(", ")).unwrap();
    if has_unicode_dimensions {
        for (index, input) in inputs.iter().enumerate().filter(|(_, input)| !input.name.is_ascii()) {
            for axis in &input.dynamic_axes {
                writeln!(text, "    _m_input_{index}_dim{} = _m_dynamic_dims.pop(\"{}_dim{}\", {})", axis + 1, input.name, axis + 1, input.shape[*axis]).unwrap();
            }
        }
        text.push_str("    if _m_dynamic_dims:\n        raise TypeError(f\"Unknown dynamic dimensions: {sorted(_m_dynamic_dims)}\")\n");
    }
    for param in &batch_params {
        if param.starts_with("**") { continue; }
        let name = param.split_once('=').unwrap().0;
        writeln!(text, "    if not isinstance({name}, int) or isinstance({name}, bool) or {name} < 1:\n        raise ValueError(\"{name} must be a positive integer\")").unwrap();
    }
    for (index, input) in inputs.iter().enumerate().filter(|(_, input)| !input.name.is_ascii()) {
        for axis in &input.dynamic_axes {
            let name = format!("_m_input_{index}_dim{}", axis + 1);
            writeln!(text, "    if not isinstance({name}, int) or isinstance({name}, bool) or {name} < 1:\n        raise ValueError(\"{}_dim{} must be a positive integer\")", input.name, axis + 1).unwrap();
        }
    }
    text.push_str("    return {\n        \"args\": [],\n        \"kwargs\": {\n");
    for (index, input) in inputs.iter().enumerate() {
        let dims = input
            .shape
            .iter()
            .enumerate()
            .map(|(axis, value)| {
                if input.dynamic_axes.contains(&axis) {
                    if input.name.is_ascii() { format!("{}_dim{}", input.name, axis + 1) }
                    else { format!("_m_input_{index}_dim{}", axis + 1) }
                } else {
                    value.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let dtype = input.dtype.as_ref()
            .map(|name| format!("_resolve_dtype(\"{name}\")"))
            .unwrap_or_else(|| "torch.get_default_dtype()".to_owned());
        writeln!(
            text,
            "            \"{}\": torch.ones((batch_size, {dims}), dtype={dtype}),",
            input.name
        )
        .unwrap();
    }
    if named_outputs.is_empty() {
        match objective {
            Objective::Mse => writeln!(text, "        }},\n        \"loss_inputs\": {{\"target\": torch.zeros((batch_size, {output}), dtype=torch.get_default_dtype())}},\n    }}\n\n\ndef step(model, batch, context):\n    prediction = model(**batch[\"kwargs\"])\n    return {{\"loss\": _mse_loss(prediction, batch[\"loss_inputs\"][\"target\"], \"target\")}}\n").unwrap(),
            Objective::CrossEntropy => writeln!(text, "        }},\n        \"loss_inputs\": {{\"target\": torch.arange(batch_size, dtype=torch.int64) % {output}}},\n    }}\n\n\ndef step(model, batch, context):\n    logits = model(**batch[\"kwargs\"])\n    return {{\"loss\": nn.functional.cross_entropy(logits, batch[\"loss_inputs\"][\"target\"])}}\n").unwrap(),
            Objective::Custom(ref reference) => writeln!(text, "        }},\n        \"loss_inputs\": {{\"target\": torch.zeros((batch_size, {output}), dtype=torch.get_default_dtype())}},\n    }}\n\n\ndef step(model, batch, context):\n    prediction = model(**batch[\"kwargs\"])\n    return {{\"loss\": _custom_loss({}, prediction, batch[\"loss_inputs\"][\"target\"])}}\n", serde_json::to_string(reference.strip_prefix("custom:").unwrap()).unwrap()).unwrap(),
        }
    } else {
        text.push_str("        },\n        \"loss_inputs\": {\"targets\": {\n");
        for (name, width, objective, _) in named_outputs {
            match objective {
                Objective::Mse => writeln!(text, "            \"{name}\": torch.zeros((batch_size, {width}), dtype=torch.get_default_dtype()),").unwrap(),
                Objective::CrossEntropy => writeln!(text, "            \"{name}\": torch.arange(batch_size, dtype=torch.int64) % {width},").unwrap(),
                Objective::Custom(_) => writeln!(text, "            \"{name}\": torch.zeros((batch_size, {width}), dtype=torch.get_default_dtype()),  # Replace with your target contract.").unwrap(),
            }
        }
        text.push_str("        }},\n    }\n\n\ndef step(model, batch, context):\n    prediction = model(**batch[\"kwargs\"])\n    targets = batch[\"loss_inputs\"][\"targets\"]\n    losses = {\n");
        for (name, _, objective, _) in named_outputs {
            match objective {
                Objective::Mse => writeln!(text, "        \"{name}\": _mse_loss(prediction[\"{name}\"], targets[\"{name}\"], \"{name}\"),").unwrap(),
                Objective::CrossEntropy => writeln!(text, "        \"{name}\": nn.functional.cross_entropy(prediction[\"{name}\"], targets[\"{name}\"]),").unwrap(),
                Objective::Custom(reference) => writeln!(text, "        \"{name}\": _custom_loss({}, prediction[\"{name}\"], targets[\"{name}\"]),", serde_json::to_string(reference.strip_prefix("custom:").unwrap()).unwrap()).unwrap(),
            }
        }
        text.push_str("    }\n    # Raw task losses are reported separately; weights affect the optimized total.\n    weights = {\n");
        for (name, _, _, weight) in named_outputs {
            writeln!(text, "        \"{name}\": {weight:?},").unwrap();
        }
        text.push_str("    }\n    return {\"loss\": sum(weights[name] * loss for name, loss in losses.items()), \"losses\": losses}\n");
    }
    text.push_str(r#"
# Model-Maven one-batch evaluator v1
# This evaluates one supplied batch, not the whole dataset. Replace it for
# dataset-wide metrics, custom targets, or task-specific reporting.
def evaluate(model, data, context):
    batch = context.transform_batch(data)
    if not isinstance(batch, dict) or not isinstance(batch.get("kwargs"), dict) or not batch["kwargs"]:
        raise ValueError("The starter evaluator needs one batch envelope with named inputs")
    inputs = list(batch["kwargs"].values())
    if any(not isinstance(value, torch.Tensor) or value.ndim == 0 for value in inputs):
        raise ValueError("Each model input must be a batched tensor")
    count = inputs[0].shape[0]
    if count == 0 or any(value.shape[0] != count for value in inputs):
        raise ValueError("Model inputs must have the same nonempty batch dimension")
    if context.max_samples is not None and count > context.max_samples:
        raise ValueError("The batch exceeds the sample limit; reduce batch_size or increase the limit")

    def to_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(context.device) if context.device is not None else value
        if isinstance(value, dict):
            return {key: to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(to_device(item) for item in value)
        return value

    result = step(model, to_device(batch), context)
    context.report_samples(count)
    return {"metrics": {"loss": result["loss"]}}
"#);
    if has_custom_loss {
        text.push_str("\n# The synthetic batch is stateless; a user loss may hold state.\n# Step resume requires an explicit state contract.\nbatch.MODEL_MAVEN_STATELESS = True\n");
    } else {
        text.push_str("\n# These starter callables have no cursor or mutable state. If you add state,\n# remove the declaration and implement state_dict/load_state_dict for resume.\nbatch.MODEL_MAVEN_STATELESS = True\nstep.MODEL_MAVEN_STATELESS = True\n");
    }
    text
}

impl<B: Backend> Application<B> {
    pub(super) fn sync_scaffold_recipe(&self, sync_data: bool, training: Option<&str>, expected_recipe: &str,
        expected_config: &str, apply: bool, if_plan: Option<&str>) -> Report {
        let fail = |code, message: String| Report::error("recipe_sync", code, message, "检查项目与声明，然后重新预览。 ");
        if !sync_data && training.is_none() { return fail("RECIPE_SYNC_EMPTY", "请选择输入或训练设置。".into()); }
        let root = match std::env::current_dir().and_then(|path| path.canonicalize()) {
            Ok(root) => root, Err(error) => return fail("PROJECT_UNAVAILABLE", error.to_string()),
        };
        let path = root.join(RECIPE_FILE);
        let bytes = match read_recipe_for_update(&path) { Ok(bytes) => bytes, Err(error) => return fail("RECIPE_INVALID", error) };
        if sha256_bytes(&bytes) != expected_recipe { return fail("RECIPE_CONFLICT", "声明已变化。".into()); }
        let mut recipe = match ScaffoldRecipe::parse(&bytes) { Ok(recipe) => recipe, Err(error) => return fail("RECIPE_INVALID", error) };
        let current = match self.project_config.load(&root) {
            Ok(Some(current)) => current,
            Ok(None) => return fail("PROJECT_NOT_INITIALIZED", "项目尚未配置。".into()),
            Err(error) => return fail("PROJECT_CONFIG_INVALID", error),
        };
        let revision = match crate::domain::input::content_revision(&current) { Ok(value) => value, Err(error) => return fail("PROJECT_CONFIG_INVALID", error) };
        if revision != expected_config { return fail("PROJECT_CONFIG_CONFLICT", "项目配置已变化。".into()); }
        let module = match current["model"].as_str().and_then(|model| model.strip_suffix(":model")).filter(|module| {
            *module == "model_starter" || module.strip_prefix(BUILD_PREFIX).is_some_and(|hash| hash.len() == 24 && hash.bytes().all(|b| b.is_ascii_hexdigit()))
        }) {
            Some(module) => module,
            None => return fail("RECIPE_MODEL_UNSUPPORTED", "项目当前没有使用由画布生成的模型。".into()),
        };
        let previous_data = json!(recipe.data);
        let previous_training = json!(recipe.training);
        if sync_data {
            let factory = match current["batch"].as_str() {
                Some(factory) if python_entrypoint(factory) => factory,
                _ => return fail("RECIPE_INPUT_INVALID", "项目没有可声明的输入工厂。".into()),
            };
            let params = if current["batch_params"].is_null() { Map::new() }
                else { match current["batch_params"].as_object() {
                    Some(params) => params.clone(), None => return fail("RECIPE_INPUT_INVALID", "输入参数不是对象。".into()),
                }};
            if params.len() > 64 || !params.keys().all(|name| !name.is_empty() && !name.contains('=') && !name.chars().any(char::is_control))
                || serde_json::to_vec(&params).map_or(true, |value| value.len() > 65_536) {
                return fail("RECIPE_INPUT_INVALID", "输入参数名称或大小不受声明支持。".into());
            }
            let binding = current["batch_binding"].as_str();
            if let Err(error) = recipe_binding(binding) { return fail("RECIPE_INPUT_INVALID", error); }
            let synthetic = factory == format!("{module}:batch") && binding == Some("envelope") && params.is_empty();
            recipe.data = if synthetic { None } else { Some(RecipeData { factory: factory.to_owned(), binding: binding.map(str::to_owned), params }) };
        }
        if let Some(raw) = training {
            if raw.len() > 65_536 { return fail("RECIPE_TRAINING_INVALID", "训练声明过大。".into()); }
            let profile = match serde_json::from_str::<Option<RecipeTraining>>(raw) {
                Ok(value) => value, Err(error) => return fail("RECIPE_TRAINING_INVALID", error.to_string()),
            };
            if let Some(profile) = &profile && let Err(error) = validate_training_profile(profile) { return fail("RECIPE_TRAINING_INVALID", error); }
            recipe.training = profile;
        }
        let mut plan = json!({"recipe_revision":expected_recipe,"config_revision":revision,"model":current["model"],
            "sync_data":sync_data,"sync_training":training.is_some(),"previous_data":previous_data,"data":recipe.data,
            "previous_training":previous_training,"training":recipe.training});
        let plan_revision = match crate::domain::input::content_revision(&plan) { Ok(value) => value, Err(error) => return fail("RECIPE_PLAN_FAILED", error) };
        if !apply {
            let mut fields = Vec::new(); redact_run_inputs(&mut plan, "$.plan", &mut fields);
            return Report::ok("recipe_sync", json!({"committed":false,"plan_revision":plan_revision,"plan":plan,"redacted_fields":fields}));
        }
        if if_plan != Some(plan_revision.as_str()) { return fail("RECIPE_PLAN_CONFLICT", "声明计划已变化，请重新预览。".into()); }
        let contents = match serde_json::to_string_pretty(&recipe) { Ok(value) => format!("{value}\n"), Err(error) => return fail("RECIPE_INVALID", error.to_string()) };
        if contents.len() > RECIPE_LIMIT as usize { return fail("RECIPE_INVALID", "声明不能超过 128 KiB。".into()); }
        let lock = match lock_project_config(&root) { Ok(lock) => lock, Err(error) => return fail("PROJECT_CONFIG_CONFLICT", error) };
        let latest = match self.project_config.load(&root) { Ok(Some(value)) => value, _ => return fail("PROJECT_CONFIG_CONFLICT", "无法重新读取项目配置。".into()) };
        if crate::domain::input::content_revision(&latest).ok().as_deref() != Some(revision.as_str()) {
            return fail("PROJECT_CONFIG_CONFLICT", "项目配置已变化。".into());
        }
        if let Err(error) = replace_recipe(&path, &bytes, contents.as_bytes()) { return fail("RECIPE_CONFLICT", error); }
        drop(lock);
        Report::ok("recipe_sync", json!({"committed":true,"plan_revision":plan_revision,
            "recipe_revision":sha256_bytes(contents.as_bytes()),"config_revision":revision,
            "sync_data":sync_data,"sync_training":training.is_some(),"note":"只更新声明；未重新生成源码或运行模型。"}))
    }

    pub(super) fn scaffold_from_recipe(&self, recipe: &Path, write_recipe: bool, update_recipe: Option<&str>, reconcile_assembly: bool, apply: bool, if_plan: Option<&str>) -> Report {
        let fail = |error| Report::error(
            "scaffold", "SCAFFOLD_RECIPE_INVALID", error, "检查 model-maven.json 的字段与版本。",
        );
        let recipe = match ScaffoldRecipe::read(recipe) {
            Ok(recipe) => recipe,
            Err(error) => return fail(error),
        };
        let (inputs, outputs) = match recipe.arguments() {
            Ok(arguments) => arguments,
            Err(error) => return fail(error),
        };
        if recipe.routes.iter().chain(&recipe.components).any(|(slot, value)| {
            !python_identifier(slot) || value.contains('=')
        }) {
            return fail("接缝名称或绑定值无效。".into());
        }
        let route_args: Vec<_> = recipe.routes.iter().map(|(slot, input)| format!("{slot}={input}")).collect();
        let injections: Vec<_> = recipe.components.iter().map(|(slot, factory)| format!("{slot}={factory}")).collect();
        let mut injection_params = Vec::new();
        let component_params_slots = recipe.component_params.keys().cloned().collect::<Vec<_>>();
        for (slot, values) in &recipe.component_params {
            if !recipe.components.contains_key(slot) {
                return fail(format!("组件参数引用了未声明的接缝 {slot}。"));
            }
            for (name, value) in values {
                injection_params.push(format!("{slot}.{name}={value}"));
            }
        }
        let encoder_layers = match recipe.encoder_layers.iter().map(|(slot, layers)| {
            serde_json::to_string(layers).map(|encoded| format!("{slot}={encoded}"))
        }).collect::<Result<Vec<_>, _>>() {
            Ok(declarations) => declarations,
            Err(error) => return fail(error.to_string()),
        };
        let encoder_widths = recipe.encoder_widths.iter()
            .map(|(slot, width)| format!("{slot}={width}")).collect::<Vec<_>>();
        let binding = match recipe_binding(recipe.data.as_ref().and_then(|data| data.binding.as_deref())) {
            Ok(value) => value,
            Err(error) => return fail(error),
        };
        let batch_params = recipe.data.as_ref().map(|data| data.params.iter()
            .map(|(name, value)| format!("{name}={value}")).collect::<Vec<_>>()).unwrap_or_default();
        let training = match recipe.training.as_ref().map(serde_json::to_string).transpose() {
            Ok(value) => value,
            Err(error) => return fail(error.to_string()),
        };
        let supervision = match recipe.supervision.as_ref().map(serde_json::to_string).transpose() {
            Ok(value) => value,
            Err(error) => return fail(error.to_string()),
        };
        self.scaffold_model(
            None, recipe.data.as_ref().map(|data| data.factory.as_str()), binding.as_ref(), &batch_params, &injections, &injection_params, &inputs, &route_args, &recipe.native_encoders, &encoder_layers, &encoder_widths,
            None, &outputs, None, write_recipe, update_recipe, reconcile_assembly, &component_params_slots, training.as_deref(), supervision.as_deref(), apply, if_plan,
        )
    }

    pub(super) fn scaffold_model(
        &self,
        module: Option<&str>,
        external_batch: Option<&str>,
        external_batch_binding: Option<&BatchBinding>,
        batch_params: &[String],
        injections: &[String],
        injection_params: &[String],
        args: &[String],
        route_args: &[String],
        native_args: &[String],
        encoder_layer_args: &[String],
        encoder_width_args: &[String],
        output_dim: Option<usize>,
        output_args: &[String],
        loss: Option<&str>,
        write_recipe: bool,
        update_recipe: Option<&str>,
        reconcile_assembly: bool,
        component_params_slots: &[String],
        training_profile: Option<&str>,
        supervision_reference: Option<&str>,
        apply: bool,
        if_plan: Option<&str>,
    ) -> Report {
        let fail = |code, msg: String| {
            Report::error("scaffold", code, msg, "修正输入或目标目录后重新预览。 ")
        };
        let supervision = match supervision_reference {
            Some(raw) if raw.len() <= 4096 => match serde_json::from_str::<RecipeSupervision>(raw) {
                Ok(reference) => match validate_supervision_reference(&reference) {
                    Ok(()) => Some(reference),
                    Err(error) => return fail("SCAFFOLD_SUPERVISION_INVALID", error),
                },
                Err(error) => return fail("SCAFFOLD_SUPERVISION_INVALID", error.to_string()),
            },
            Some(_) => return fail("SCAFFOLD_SUPERVISION_INVALID", "监督引用过大。".into()),
            None => None,
        };
        let training = match training_profile {
            Some(raw) if raw.len() <= 65_536 => match serde_json::from_str::<RecipeTraining>(raw) {
                Ok(profile) => match validate_training_profile(&profile) {
                    Ok(()) => Some(profile),
                    Err(error) => return fail("SCAFFOLD_TRAINING_INVALID", error),
                },
                Err(error) => return fail("SCAFFOLD_TRAINING_INVALID", error.to_string()),
            },
            Some(_) => return fail("SCAFFOLD_TRAINING_INVALID", "训练声明过大。".into()),
            None => None,
        };
        let mut explicit_component_params = match injection_parameters(injection_params) {
            Ok(params) => params,
            Err(error) => return fail("MODEL_INJECTION_INVALID", error),
        };
        let mut declared_slots = HashSet::new();
        for slot in component_params_slots {
            if !python_identifier(slot) || !declared_slots.insert(slot)
                || !injections.iter().any(|binding| binding.split_once('=').is_some_and(|(name, _)| name == slot)) {
                return fail("MODEL_INJECTION_INVALID", format!("组件参数接缝 {slot} 无效或重复。"));
            }
            explicit_component_params.entry(slot.clone()).or_default();
        }
        let module = module.unwrap_or("model_starter");
        const PYTHON_KEYWORDS: &[&str] = &[
            "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
            "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
            "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
            "return", "try", "while", "with", "yield",
        ];
        if module.len() > 200
            || !module
                .split('.')
                .all(|part| python_identifier(part) && !PYTHON_KEYWORDS.contains(&part))
        {
            return fail(
                "SCAFFOLD_MODULE_INVALID",
                "模块路径须由 Python 标识符以点分隔，例如 models.custom。".into(),
            );
        }
        let mut source_relative = format!("{}.py", module.replace('.', "/"));
        if external_batch.is_some_and(|entry| !python_entrypoint(entry)) {
            return fail(
                "SCAFFOLD_BATCH_INVALID",
                "已有批次工厂须为 module:callable。".into(),
            );
        }
        if external_batch.is_none() && external_batch_binding.is_some() {
            return fail(
                "SCAFFOLD_BATCH_INVALID",
                "批次绑定需要提供已有批次工厂。".into(),
            );
        }
        if external_batch.is_none() && !batch_params.is_empty() {
            return fail(
                "SCAFFOLD_BATCH_INVALID",
                "批次参数需要提供已有批次工厂。".into(),
            );
        }
        let (inputs, _) = match inputs_from_args(args) {
            Ok(x) => x,
            Err(e) => return fail("SCAFFOLD_INPUT_INVALID", e),
        };
        let mut native_encoders = HashSet::new();
        for slot in native_args {
            if !inputs
                .iter()
                .any(|input| format!("{}_encoder", input.name) == *slot)
                || !native_encoders.insert(slot.clone())
            {
                return fail(
                    "SCAFFOLD_NATIVE_ENCODER_INVALID",
                    format!("编码器接缝 {slot} 不存在或重复。"),
                );
            }
        }
        for injection in injections {
            let Some((slot, _)) = injection.split_once('=') else {
                return fail(
                    "SCAFFOLD_SEAM_INVALID",
                    "接缝绑定格式须为 SLOT=MODULE:FACTORY。".into(),
                );
            };
            if slot != "fusion"
                && slot != "head"
                && !inputs
                    .iter()
                    .any(|input| format!("{}_encoder", input.name) == slot)
            {
                return fail(
                    "SCAFFOLD_SEAM_INVALID",
                    format!("骨架没有接缝 {slot}；可选输入编码器、fusion 或 head。"),
                );
            }
        }
        let routes = match resolve_routes(&inputs, route_args) {
            Ok(routes) => routes,
            Err(error) => return fail("SCAFFOLD_ROUTE_INVALID", error),
        };
        let (encoder_layers, resolved_encoders) = match resolve_encoder_layers(&inputs, &routes, injections, encoder_layer_args) {
            Ok(resolved) => resolved,
            Err(error) => return fail("SCAFFOLD_ENCODER_LAYER_INVALID", error),
        };
        if let Some(slot) = native_encoders.iter().find(|slot| resolved_encoders.contains_key(*slot)) {
            return fail("SCAFFOLD_NATIVE_ENCODER_INVALID", format!("编码器 {slot} 的内置层序列需要展平输入；请使用组件或关闭原始输出。"));
        }
        let mut encoder_widths = BTreeMap::new();
        for item in encoder_width_args {
            let Some((slot, raw)) = item.split_once('=') else {
                return fail("SCAFFOLD_ENCODER_WIDTH_INVALID", "编码器宽度格式应为 SLOT=WIDTH。".into());
            };
            let width = raw.parse::<usize>().ok().filter(|width| (1..=1_000_000).contains(width));
            let exists = inputs.iter().any(|input| format!("{}_encoder", input.name) == slot);
            let injected = injections.iter().any(|binding| binding.split_once('=').is_some_and(|(name, _)| name == slot));
            if !exists || !injected || width.is_none() || encoder_widths.contains_key(slot) {
                return fail("SCAFFOLD_ENCODER_WIDTH_INVALID", format!("编码器 {slot} 的组件宽度无效或重复。"));
            }
            encoder_widths.insert(slot.to_owned(), width.unwrap());
        }
        let features: usize = inputs.iter().zip(&routes).map(|(input, &route)| {
            let slot = format!("{}_encoder", input.name);
            encoder_widths.get(&slot).copied().unwrap_or_else(|| resolved_encoders.get(&slot).map_or_else(
                || flattened_width(&inputs[route]),
                |encoder| encoder.output_width,
            ))
        }).sum();
        if features > 1_000_000 {
            return fail(
                "SCAFFOLD_ROUTE_INVALID",
                "接线后的融合特征不能超过 1000000。".into(),
            );
        }
        let named_outputs = match named_outputs_from_args(output_args, features) {
            Ok(value) => value,
            Err(error) => return fail("SCAFFOLD_OUTPUT_INVALID", error),
        };
        if (output_dim.is_some() || loss.is_some()) && !named_outputs.is_empty() {
            return fail(
                "SCAFFOLD_OUTPUT_INVALID",
                "具名输出不能与 --output-dim 或 --loss 同时使用。".into(),
            );
        }
        let output = output_dim.unwrap_or(features);
        let objective = match Objective::parse(loss.unwrap_or("mse")) {
            Ok(value) => value,
            Err(error) => return fail("SCAFFOLD_OUTPUT_INVALID", error),
        };
        if named_outputs.is_empty() && objective == Objective::CrossEntropy && output < 2 {
            return fail(
                "SCAFFOLD_OUTPUT_INVALID",
                "交叉熵输出至少需要两个类别。".into(),
            );
        }
        if named_outputs.is_empty()
            && (output == 0
                || output > 65_536
                || features.checked_mul(output).is_none_or(|n| n > 10_000_000))
        {
            return fail(
                "SCAFFOLD_OUTPUT_INVALID",
                "输出维度或默认线性层参数量过大（最多 10000000 个权重）。".into(),
            );
        }
        let root = match std::env::current_dir().and_then(|p| p.canonicalize()) {
            Ok(p) => p,
            Err(e) => return fail("PROJECT_UNAVAILABLE", e.to_string()),
        };
        let metadata = root.join(".model-maven");
        match fs::symlink_metadata(&metadata) {
            Ok(m) if m.file_type().is_symlink() || !m.is_dir() => {
                return fail(
                    "SCAFFOLD_TARGET_CONFLICT",
                    "项目元数据目录必须是普通目录。".into(),
                );
            }
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return fail("SCAFFOLD_TARGET_CONFLICT", e.to_string());
            }
            _ => {}
        }
        let source = source(&inputs, &routes, &native_encoders, &resolved_encoders, &encoder_widths, output, &named_outputs, objective.clone());
        let update = if let Some(expected) = update_recipe {
            match self.prepare_recipe_update(&root, expected, reconcile_assembly, injections, injection_params) {
                Ok(update) => {
                    let mut binding_order = injections.iter().collect::<Vec<_>>();
                    binding_order.sort_unstable();
                    let build_revision = crate::domain::input::content_revision(&json!({
                        "source": source,
                        "components": binding_order,
                    }));
                    let build_revision = match build_revision {
                        Ok(revision) => revision,
                        Err(error) => return fail("SCAFFOLD_PLAN_FAILED", error),
                    };
                    source_relative = format!("{BUILD_PREFIX}{}.py", &build_revision[..24]);
                    Some(update)
                }
                Err(error) => return fail("SCAFFOLD_RECIPE_CONFLICT", error),
            }
        } else {
            None
        };
        let mut files = vec![source_relative.clone(), ".model-maven/project.json".to_owned()];
        if write_recipe { files.push(RECIPE_FILE.to_owned()); }
        for file in files.iter().filter(|_| update.is_none()).map(|file| root.join(file)) {
            match fs::symlink_metadata(&file) {
                Ok(_) => {
                    return fail(
                        "SCAFFOLD_TARGET_CONFLICT",
                        format!("{} 已存在，不会覆盖。", file.display()),
                    );
                }
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    return fail("SCAFFOLD_TARGET_CONFLICT", e.to_string());
                }
                _ => {}
            }
        }
        let module = source_relative.strip_suffix(".py").unwrap_or_default().replace('/', ".");
        let model = format!("{module}:model");
        let previous_config = update.as_ref().and_then(|update| update.config.as_ref());
        let retained_batch = previous_config.and_then(|current| {
            let old_module = current["model"].as_str()?.strip_suffix(":model")?;
            let batch = current["batch"].as_str()?;
            let default_batch = batch == format!("{old_module}:batch")
                && current["batch_binding"] == "envelope"
                && current["batch_params"].as_object().is_none_or(|params| params.is_empty());
            (!default_batch).then_some(batch)
        });
        if let (Some(current), Some(factory)) = (previous_config, external_batch) {
            let requested_params = match parameters(batch_params) {
                Ok(value) => value,
                Err(error) => return fail("SCAFFOLD_BATCH_INVALID", error),
            };
            let requested_binding = external_batch_binding.map(binding_name);
            if current["batch"].as_str() != Some(factory)
                || current["batch_binding"].as_str() != requested_binding
                || current["batch_params"].as_object().map_or(!requested_params.is_empty(), |params| params != &requested_params) {
                return fail("SCAFFOLD_BATCH_CONFLICT", "声明的数据绑定与当前项目不一致；请先在项目中更新绑定并重新载入。".into());
            }
        }
        let batch = retained_batch.or(external_batch)
            .map(str::to_owned).unwrap_or_else(|| format!("{module}:batch"));
        let batch_source = if retained_batch.is_some() {
            "existing"
        } else if external_batch.is_some() {
            "external"
        } else {
            "synthetic"
        };
        let binding = if external_batch.is_some() {
            external_batch_binding
        } else {
            Some(&BatchBinding::Envelope)
        };
        let init = |dry_run| {
            self.initialize_project(InitOptions {
                model: Some(&model),
                batch: Some(&batch),
                injections,
                injection_params,
                batch_binding: binding,
                batch_params,
                dry_run,
            })
        };
        let config = if let Some((update, current)) = update.as_ref()
            .and_then(|update| update.config.as_ref().map(|current| (update, current)))
        {
            let mut bindings = match model_injections(injections, injection_params, &*self.components, &root) {
                Ok(values) => values,
                Err(error) => return fail("MODEL_INJECTION_INVALID", error),
            };
            for (slot, factory) in injections.iter().filter_map(|entry| entry.split_once('=')) {
                if (reconcile_assembly || (update.recipe.components.get(slot).is_some_and(|previous| previous == factory)
                    && !explicit_component_params.contains_key(slot)))
                    && let Some(value) = current["model_injections"].get(slot)
                {
                    bindings.insert(slot.to_owned(), value.clone());
                }
            }
            let mut next = current.clone();
            next["model"] = json!(model);
            next["batch"] = json!(batch);
            next["model_injections"] = json!(bindings);
            if reconcile_assembly && let Some(params) = next.as_object_mut()
                .and_then(|config| config.get_mut("model_params"))
                .and_then(Value::as_object_mut) {
                // The reviewed routes are now baked into the generated model.
                params.remove("input_routes");
                params.remove("encoder_output_modes");
            }
            next
        } else {
            let preview = init(true);
            if !preview.success() {
                return preview;
            }
            preview.data["config"].clone()
        };
        let descriptors: Vec<Value> = inputs.iter().map(|i| json!({"name": i.name, "shape": i.shape, "dynamic_axes": i.dynamic_axes, "dynamic_reduction":if native_encoders.contains(&format!("{}_encoder", i.name)) { "none_native_encoder_output" } else { "mean_after_encoder" }, "encoder_output":if native_encoders.contains(&format!("{}_encoder", i.name)) { "native" } else { "flatten" }, "dtype":i.dtype, "dtype_source":if i.dtype.is_some() { "explicit_runtime_lookup" } else { "pytorch_default_at_batch_execution" }, "encoder_slot":format!("{}_encoder", i.name)})).collect();
        let connections: Vec<Value> = inputs.iter().enumerate().map(|(index, input)| json!({"encoder_slot":format!("{}_encoder", input.name), "input":inputs[routes[index]].name})).collect();
        let output_descriptor = if named_outputs.is_empty() {
            if objective == Objective::Mse {
                json!({"features":output,"dtype":null,"dtype_source":"pytorch_default_at_construction"})
            } else {
                json!({"features":output,"dtype":null,"dtype_source":"pytorch_default_at_construction","objective":objective.name()})
            }
        } else {
            json!({"kind":"named","heads":named_outputs.iter().map(|(name, width, objective, weight)| json!({"name":name,"features":width,"dtype":null,"dtype_source":"pytorch_default_at_construction","objective":objective.name(),"weight":weight})).collect::<Vec<_>>()})
        };
        let starter_step = format!("{module}:step");
        let starter_evaluator = format!("{module}:evaluate");
        let mut plan = json!({"version":1,"project_root":root,"inputs":descriptors,"routes":connections,"output":output_descriptor,"slots":["fusion","head"],"source":source,"config":config,"files":files,"batch_source":batch_source,"starter_step":starter_step,"starter_evaluator":starter_evaluator});
        let mut layer_id_paths = serde_json::Map::new();
        for input in &inputs {
            if let Some(encoder) = resolved_encoders.get(&format!("{}_encoder", input.name)) {
                for (id, suffix) in &encoder.layer_paths {
                    layer_id_paths.insert(id.clone(), json!(format!("model.encoders.enc_{}{suffix}", input.name)));
                }
            }
        }
        if !layer_id_paths.is_empty() {
            plan["layer_id_paths"] = json!({"scope":"generated_default_encoders_only", "paths":layer_id_paths});
            let marker = "# Model-Maven layer IDs (constructor below): ";
            let mut locations = serde_json::Map::new();
            for (index, line) in source.lines().enumerate() {
                if let Some(ids) = line.trim_start().strip_prefix(marker) {
                    for id in ids.split_whitespace() {
                        locations.insert(id.to_owned(), json!({"line":index + 2}));
                    }
                }
            }
            plan["layer_source_locations"] = json!({
                "scope":"proposed_generated_default_encoders_only",
                "source_sha256":sha256_bytes(source.as_bytes()),
                "granularity":"constructor_line_nested_layers_share_line",
                "locations":locations,
            });
        }
        if !resolved_encoders.is_empty() {
            let dimensions = resolved_encoders.iter().map(|(slot, encoder)| {
                (slot.clone(), json!({"input":encoder.input_width,"output":encoder.output_width}))
            }).collect::<serde_json::Map<_, _>>();
            plan["encoder_dimensions"] = json!(dimensions);
        }
        if let Some(update) = &update {
            let source_state = previous_source_state(&root, update);
            plan["update"] = json!({
                "recipe_revision": sha256_bytes(&update.bytes),
                "previous_recipe": update.recipe,
                "project_config_revision": update.config.as_ref().and_then(|value| crate::domain::input::content_revision(value).ok()),
                "previous_model": update.config.as_ref().map(|config| &config["model"]),
                "previous_batch": update.config.as_ref().map(|config| &config["batch"]),
                "previous_source": source_state,
            });
        }
        let recipe_source = if write_recipe {
            let recipe = ScaffoldRecipe {
                version: 1,
                training,
                supervision,
                data: match external_batch {
                    Some(factory) => match parameters(batch_params) {
                        Ok(params) => Some(RecipeData { factory: factory.to_owned(), binding: external_batch_binding.map(|binding| binding_name(binding).to_owned()), params }),
                        Err(error) => return fail("SCAFFOLD_BATCH_INVALID", error),
                    },
                    None => None,
                },
                encoder_layers,
                encoder_widths,
                native_encoders: inputs.iter().map(|input| format!("{}_encoder", input.name))
                    .filter(|slot| native_encoders.contains(slot)).collect(),
                component_params: explicit_component_params,
                routes: inputs.iter().zip(&routes).filter_map(|(input, &index)| {
                    let source = &inputs[index].name;
                    (&input.name != source).then(|| (format!("{}_encoder", input.name), source.clone()))
                }).collect(),
                components: injections.iter().filter_map(|binding| {
                    binding.split_once('=').map(|(slot, factory)| (slot.to_owned(), factory.to_owned()))
                }).collect(),
                inputs,
                outputs: named_outputs.iter().map(|(name, features, objective, weight)| RecipeOutput {
                    name: name.clone(), features: *features, loss: objective.name().to_owned(), weight: *weight,
                }).collect(),
            };
            let value = match serde_json::to_value(&recipe) {
                Ok(value) => value,
                Err(error) => return fail("SCAFFOLD_PLAN_FAILED", error.to_string()),
            };
            let mut contents = match serde_json::to_string_pretty(&value) {
                Ok(contents) => contents,
                Err(error) => return fail("SCAFFOLD_PLAN_FAILED", error.to_string()),
            };
            contents.push('\n');
            plan["recipe"] = value;
            Some(contents)
        } else {
            None
        };
        let revision = match crate::domain::input::content_revision(&plan) {
            Ok(r) => r,
            Err(e) => return fail("SCAFFOLD_PLAN_FAILED", e),
        };
        if !apply {
            let mut public_plan = plan.clone();
            let mut redacted_fields = Vec::new();
            redact_run_inputs(&mut public_plan, "$.plan", &mut redacted_fields);
            return Report::ok(
                "scaffold",
                json!({"committed":false,"plan_revision":revision,"plan":public_plan,"redacted_fields":redacted_fields,"note":if batch_source != "synthetic" { "已有批次工厂尚未执行或验证；生成文件仍包含合成示例，示例损失需与真实目标匹配。" } else { "骨架使用合成输入与占位损失；运行前向或训练前应替换为项目数据和目标。" }}),
            );
        }
        if if_plan != Some(revision.as_str()) {
            return fail(
                "SCAFFOLD_PLAN_CONFLICT",
                "计划修订不匹配；请重新预览。".into(),
            );
        }
        if let Some(update) = update {
            let Some(contents) = recipe_source else {
                return fail("SCAFFOLD_PLAN_FAILED", "缺少已审核的配置文件内容。".into());
            };
            if let Err(error) = self.apply_recipe_update(
                &root, &update, &source_relative, source.as_bytes(), contents.as_bytes(), &plan["config"],
            ) {
                return fail("SCAFFOLD_RECIPE_UPDATE_FAILED", error);
            }
            return Report::ok("scaffold", json!({
                "committed":true,"updated":true,"plan_revision":revision,"model":model,"batch":batch,
                "files":plan["files"],"starter_step":plan["starter_step"],"starter_evaluator":plan["starter_evaluator"],"batch_source":batch_source,
                "note":"新构建已激活，旧 Python 文件保留。实际前向尚未执行。"
            }));
        }
        let path = match self.project_source_writer.write_new(
            &root,
            Path::new(&source_relative),
            source.as_bytes(),
        ) {
            Ok(p) => p,
            Err((code, msg)) => return fail(code, msg),
        };
        let recipe_path = if let Some(contents) = &recipe_source {
            match self.project_source_writer.write_new(&root, Path::new(RECIPE_FILE), contents.as_bytes()) {
                Ok(path) => Some(path),
                Err((code, message)) => {
                    remove_unchanged_generated_file(&path, source.as_bytes());
                    return fail(code, message);
                }
            }
        } else {
            None
        };
        let setup = init(false);
        if !setup.success() {
            remove_unchanged_generated_file(&path, source.as_bytes());
            if let (Some(path), Some(contents)) = (recipe_path, recipe_source) {
                remove_unchanged_generated_file(&path, contents.as_bytes());
            }
            return setup;
        }
        Report::ok(
            "scaffold",
            json!({"committed":true,"plan_revision":revision,"model":model,"batch":batch,"files":plan["files"],"starter_step":plan["starter_step"],"starter_evaluator":plan["starter_evaluator"],"batch_source":batch_source,"note":"骨架已创建；实际前向尚未执行。示例步骤和目标应由用户审核，尤其是使用已有批次时。"}),
        )
    }

    fn prepare_recipe_update(&self, root: &Path, expected: &str, reconcile: bool, injections: &[String], injection_params: &[String]) -> Result<RecipeUpdate, String> {
        let bytes = read_recipe_for_update(&root.join(RECIPE_FILE))?;
        if sha256_bytes(&bytes) != expected {
            return Err("model-maven.json 已变化，请重新打开画布。".into());
        }
        let recipe = ScaffoldRecipe::parse(&bytes)?;
        let config = self.project_config.load(root)?;
        if let Some(current) = &config {
            let _module = current["model"]
                .as_str()
                .and_then(|model| model.strip_suffix(":model"))
                .filter(|module| {
                    *module == "model_starter"
                        || module.strip_prefix(BUILD_PREFIX).is_some_and(|hash| {
                            hash.len() == 24 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                        })
                })
                .ok_or("项目当前使用独立的模型入口。请在新目录使用此配置生成，不会切换现有模型。")?;
            if let Some(params) = current["model_params"].as_object() {
                let only_assembly = params.keys().all(|key| key == "input_routes" || key == "encoder_output_modes")
                    && ["input_routes", "encoder_output_modes"].iter()
                        .all(|key| params.get(*key).is_none_or(|value| value.is_null() || value.is_object()));
                if !params.is_empty() && !(reconcile && only_assembly) {
                    return Err("项目已自定义模型参数；仅支持同步输入接线和编码器输出方式。请检查参数，或在新目录生成。".into());
                }
            } else if !current["model_params"].is_null() {
                return Err("项目模型参数不是对象；无法安全同步画布。".into());
            }
            let saved_bindings = current["model_injections"].as_object();
            if !reconcile && saved_bindings.is_some_and(|bindings| {
                bindings.keys().any(|slot| !recipe.components.contains_key(slot))
            }) {
                return Err("项目包含画布配置之外的组件绑定。请在装配中检查并同步后再编辑画布。".into());
            }
            if reconcile || !recipe.components.is_empty() {
                let declarations = if reconcile { injections.to_vec() } else { recipe.components.iter()
                    .map(|(slot, factory)| format!("{slot}={factory}"))
                    .collect::<Vec<_>>() };
                let declared_bindings = model_injections(&declarations, if reconcile { injection_params } else { &[] }, &*self.components, root)
                    .map_err(|error| format!("无法核对声明中的组件：{error}"))?;
                if reconcile && saved_bindings.map_or(0, |bindings| bindings.len()) != declared_bindings.len() {
                    return Err("同步后的组件列表与项目装配不一致，请重新打开画布。".into());
                }
                for (slot, declared) in &declared_bindings {
                    let Some(actual) = saved_bindings.and_then(|bindings| bindings.get(slot)) else {
                        return Err(format!("项目中的组件 {slot} 已移除，画布声明已过期；请先检查装配。"));
                    };
                    let declared_factory = declared.get("factory").unwrap_or(declared);
                    let actual_factory = actual.get("factory").unwrap_or(actual);
                    if declared_factory != actual_factory
                        || declared.get("component_ref") != actual.get("component_ref")
                        || (reconcile && declared.get("params") != actual.get("params"))
                    {
                        return Err(format!("项目中的组件 {slot} 已变化，画布声明已过期；请先检查装配。"));
                    }
                }
            }
        }
        Ok(RecipeUpdate { bytes, recipe, config })
    }

    fn apply_recipe_update(
        &self, root: &Path, update: &RecipeUpdate, source_path: &str, source: &[u8],
        contents: &[u8], config: &Value,
    ) -> Result<(), String> {
        let _lock = self.source_edits.lock(root)?;
        let recipe_path = root.join(RECIPE_FILE);
        if self.project_config.load(root)? != update.config
            || read_recipe_for_update(&recipe_path)? != update.bytes
        {
            return Err("项目配置或声明已变化，请重新预览。".into());
        }
        let build = root.join(source_path);
        match fs::symlink_metadata(&build) {
            Ok(meta) if meta.is_file() && meta.len() == source.len() as u64 && fs::read(&build).is_ok_and(|bytes| bytes == source) => {}
            Ok(_) => return Err(format!("构建路径 {} 已存在且内容不同。", build.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.project_source_writer.write_new(root, Path::new(source_path), source)
                    .map_err(|(_, message)| message)?;
            }
            Err(error) => return Err(error.to_string()),
        }
        replace_recipe(&recipe_path, &update.bytes, contents)?;
        let saved = match &update.config {
            Some(previous) => self.project_config.update(root, previous, config),
            None => self.project_config.create(root, config),
        };
        if let Err(error) = saved {
            // A failed directory sync can be reported after the new config is
            // visible. Look at the actual entrypoint before restoring the recipe.
            match self.project_config.load(root) {
                Ok(Some(actual)) if &actual == config => return Ok(()),
                Ok(_) => {
                    replace_recipe(&recipe_path, contents, &update.bytes).map_err(|rollback| {
                        format!("项目入口保存失败：{error}。声明恢复也失败：{rollback}。请核对实际文件。")
                    })?;
                }
                Err(read_error) => return Err(format!("项目入口保存失败：{error}。当前结果无法复查：{read_error}。")),
            }
            return Err(error);
        }
        Ok(())
    }
}

struct RecipeUpdate {
    bytes: Vec<u8>,
    recipe: ScaffoldRecipe,
    config: Option<Value>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerMove {
    id: String,
    before: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum LayerEdit {
    Add { slot: String, before: Option<String>, after: Option<String>, layer: EncoderLayer },
    Remove { id: String },
    Replace { id: String, layer: EncoderLayer },
    Set { id: String, fields: Map<String, Value> },
    Copy { id: String, before: Option<String>, after: Option<String>, to_slot: Option<String>, new_ids: Vec<String> },
    Move { id: String, before: Option<String> },
}

fn require_layer_ids(layers: &[EncoderLayer]) -> Result<(), String> {
    for layer in layers {
        if layer.id().is_none() { return Err("新增或复制的每一层（含嵌套层）都需要稳定 ID。".into()); }
        match layer {
            EncoderLayer::Residual { branch, .. } => require_layer_ids(branch)?,
            EncoderLayer::Parallel { branches, .. } => for branch in branches { require_layer_ids(branch)?; },
            _ => {}
        }
    }
    Ok(())
}

fn assign_copy_ids(layer: &mut EncoderLayer, ids: &mut impl Iterator<Item = String>) -> Result<(), String> {
    layer.replace_id(ids.next().ok_or("复制层需要为所有嵌套层提供新 ID。")?);
    match layer {
        EncoderLayer::Residual { branch, .. } => for child in branch { assign_copy_ids(child, ids)?; },
        EncoderLayer::Parallel { branches, .. } => for branch in branches { for child in branch { assign_copy_ids(child, ids)?; } },
        _ => {}
    }
    Ok(())
}

fn set_layer_fields(layer: &mut EncoderLayer, fields: Map<String, Value>) -> Result<(), String> {
    let allowed: &[&str] = match layer {
        EncoderLayer::Linear { .. } => &["features"],
        EncoderLayer::Dropout { .. } => &["probability"],
        EncoderLayer::Factory { .. } => &["reference", "output_width", "params"],
        EncoderLayer::Parallel { .. } => &["merge"],
        _ => &[],
    };
    if fields.is_empty() || fields.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err("该层不支持这些字段；结构和 ID 不能通过 set 修改。".into());
    }
    let mut value = serde_json::to_value(&*layer).map_err(|error| error.to_string())?;
    let target = value.as_object_mut().ok_or("层不是对象。")?;
    for (key, item) in fields { target.insert(key, item); }
    *layer = serde_json::from_value(value).map_err(|error| error.to_string())?;
    Ok(())
}

fn layer_group_mut<'a>(layers: &'a mut Vec<EncoderLayer>, id: &str) -> Option<&'a mut Vec<EncoderLayer>> {
    if layers.iter().any(|layer| layer.id() == Some(id)) { return Some(layers); }
    for layer in layers.iter_mut() {
        match layer {
            EncoderLayer::Residual { branch, .. } => if let Some(group) = layer_group_mut(branch, id) { return Some(group); },
            EncoderLayer::Parallel { branches, .. } => for branch in branches {
                if let Some(group) = layer_group_mut(branch, id) { return Some(group); }
            },
            _ => {}
        }
    }
    None
}

fn collect_identity_ids(layers: &[EncoderLayer], seen: &mut HashSet<String>) -> Result<(), String> {
    for layer in layers {
        if let Some(id) = layer.id() {
            if id.len() != 34 || !id.starts_with("n_") || !id[2..].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) || !seen.insert(id.to_owned()) {
                return Err(format!("层 ID 无效或重复：{id}。"));
            }
        }
        match layer {
            EncoderLayer::Residual { branch, .. } => collect_identity_ids(branch, seen)?,
            EncoderLayer::Parallel { branches, .. } => for branch in branches { collect_identity_ids(branch, seen)?; },
            _ => {}
        }
    }
    Ok(())
}

fn find_recipe_group<'a>(recipe: &'a mut ScaffoldRecipe, id: &str) -> Option<(String, &'a mut Vec<EncoderLayer>)> {
    recipe.encoder_layers.iter_mut().find_map(|(slot, roots)| layer_group_mut(roots, id).map(|group| (slot.clone(), group)))
}

fn find_layer<'a>(layers: &'a [EncoderLayer], id: &str) -> Option<&'a EncoderLayer> {
    for layer in layers {
        if layer.id() == Some(id) { return Some(layer); }
        match layer {
            EncoderLayer::Residual { branch, .. } => if let Some(found) = find_layer(branch, id) { return Some(found); },
            EncoderLayer::Parallel { branches, .. } => for branch in branches {
                if let Some(found) = find_layer(branch, id) { return Some(found); }
            },
            _ => {}
        }
    }
    None
}

fn validate_edit_groups(layers: &[EncoderLayer], depth: usize, count: &mut usize) -> Result<(), String> {
    if layers.is_empty() || layers.len() > 16 || depth > 8 { return Err("层组须包含 1 至 16 层，嵌套不超过 8 层。".into()); }
    *count += layers.len();
    if *count > 64 { return Err("一个模型最多包含 64 层。".into()); }
    for layer in layers {
        match layer {
            EncoderLayer::Residual { branch, .. } => validate_edit_groups(branch, depth + 1, count)?,
            EncoderLayer::Parallel { branches, merge, .. } => {
                if !(2..=8).contains(&branches.len()) || !matches!(merge.as_str(), "add" | "concat") { return Err("并行组或汇合方式无效。".into()); }
                for branch in branches { validate_edit_groups(branch, depth + 1, count)?; }
            }
            EncoderLayer::Linear { features, .. } if !(1..=65_536).contains(features) => return Err("线性层维度无效。".into()),
            EncoderLayer::Dropout { probability, .. } if !(0.0..1.0).contains(probability) => return Err("Dropout 概率无效。".into()),
            EncoderLayer::Factory { reference, output_width, params, .. }
                if !python_entrypoint(reference) || reference.len() > 240 || !(1..=65_536).contains(output_width)
                    || params.len() > 64 || !params.keys().all(|name| python_identifier(name))
                    || serde_json::to_vec(params).map_or(true, |raw| raw.len() > 65_536) => return Err("工厂层声明无效。".into()),
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn plan_layer_edits(path: &Path, expected: &str, operations_path: &Path, expected_operations: &str) -> Report {
    let fail = |code, message: String| Report::error("recipe_layer_edit_plan", code, message, "检查草稿修订、操作和层 ID 后重试。");
    let bytes = match read_recipe_for_update(path) { Ok(value) => value, Err(error) => return fail("RECIPE_LAYER_READ_FAILED", error) };
    let operations_bytes = match read_recipe_for_update(operations_path) { Ok(value) => value, Err(error) => return fail("RECIPE_LAYER_READ_FAILED", error) };
    let revision = sha256_bytes(&bytes);
    let operations_revision = sha256_bytes(&operations_bytes);
    if revision != expected || operations_revision != expected_operations {
        return fail("RECIPE_LAYER_CONFLICT", "草稿或操作文件与请求修订不一致。".into());
    }
    let operations: Vec<LayerEdit> = match serde_json::from_slice(&operations_bytes) {
        Ok(value) => value, Err(error) => return fail("RECIPE_LAYER_INVALID", error.to_string()),
    };
    if operations.is_empty() || operations.len() > 64 { return fail("RECIPE_LAYER_INVALID", "需要 1 至 64 个操作。".into()); }
    let mut recipe = match ScaffoldRecipe::parse(&bytes) { Ok(value) => value, Err(error) => return fail("RECIPE_LAYER_INVALID", error) };
    let mut seen = HashSet::new();
    for layers in recipe.encoder_layers.values() {
        if let Err(error) = collect_identity_ids(layers, &mut seen) { return fail("RECIPE_LAYER_ID_INVALID", error); }
    }
    let mut outcomes = Vec::new();
    for (operation_index, operation) in operations.into_iter().enumerate() {
        let result = (|| -> Result<Value, String> {
            let (action, id, slot, order, layer_digest) = match operation {
                LayerEdit::Add { slot, before, after, layer } => {
                    if before.is_some() && after.is_some() { return Err("不能同时指定 before 和 after。".into()); }
                    if !recipe.inputs.iter().any(|input| format!("{}_encoder", input.name) == slot)
                        || recipe.components.contains_key(&slot) || recipe.native_encoders.contains(&slot) {
                        return Err("目标编码器不存在或已使用组件/原始输出。".into());
                    }
                    require_layer_ids(std::slice::from_ref(&layer))?;
                    let id = layer.id().ok_or("缺少新层 ID。")?.to_owned();
                    let roots = recipe.encoder_layers.entry(slot.clone()).or_default();
                    let anchor = before.as_deref().or(after.as_deref());
                    let group = if let Some(anchor) = anchor { layer_group_mut(roots, anchor).ok_or("目标层不属于指定编码器。")? } else { roots };
                    let position = match anchor {
                        Some(anchor) => group.iter().position(|item| item.id() == Some(anchor)).ok_or("目标层不存在。")? + usize::from(after.is_some()),
                        None => group.len(),
                    };
                    let digest = sha256_bytes(&serde_json::to_vec(&layer).map_err(|error| error.to_string())?);
                    group.insert(position, layer);
                    ("add", id, slot, group.iter().map(|item| item.id()).collect::<Vec<_>>(), Some(digest))
                }
                LayerEdit::Remove { id } => {
                    let top = recipe.encoder_layers.values().any(|roots| roots.iter().any(|layer| layer.id() == Some(id.as_str())));
                    let (slot, group) = find_recipe_group(&mut recipe, &id).ok_or("找不到目标层。")?;
                    if group.len() == 1 && !top { return Err("嵌套分支不能删除最后一层；可改为 Identity。".into()); }
                    let position = group.iter().position(|layer| layer.id() == Some(id.as_str())).ok_or("找不到目标层。")?;
                    group.remove(position);
                    let order = group.iter().map(|layer| layer.id()).collect::<Vec<_>>();
                    ("remove", id, slot, order, None)
                }
                LayerEdit::Replace { id, layer } => {
                    require_layer_ids(std::slice::from_ref(&layer))?;
                    if layer.id() != Some(id.as_str()) { return Err("替换层须保留原 ID；复制应使用新 ID。".into()); }
                    let (slot, group) = find_recipe_group(&mut recipe, &id).ok_or("找不到目标层。")?;
                    let position = group.iter().position(|item| item.id() == Some(id.as_str())).ok_or("找不到目标层。")?;
                    let digest = sha256_bytes(&serde_json::to_vec(&layer).map_err(|error| error.to_string())?);
                    group[position] = layer;
                    ("replace", id, slot, group.iter().map(|item| item.id()).collect::<Vec<_>>(), Some(digest))
                }
                LayerEdit::Set { id, fields } => {
                    let (slot, group) = find_recipe_group(&mut recipe, &id).ok_or("找不到目标层。")?;
                    let layer = group.iter_mut().find(|item| item.id() == Some(id.as_str())).ok_or("找不到目标层。")?;
                    set_layer_fields(layer, fields)?;
                    let digest = sha256_bytes(&serde_json::to_vec(&*layer).map_err(|error| error.to_string())?);
                    ("set", id, slot, group.iter().map(|item| item.id()).collect::<Vec<_>>(), Some(digest))
                }
                LayerEdit::Copy { id, before, after, to_slot, new_ids } => {
                    if before.is_some() && after.is_some() { return Err("不能同时指定 before 和 after。".into()); }
                    let mut copy = recipe.encoder_layers.values().find_map(|layers| find_layer(layers, &id))
                        .ok_or("找不到源层。")?.clone();
                    let mut ids = new_ids.into_iter();
                    assign_copy_ids(&mut copy, &mut ids)?;
                    if ids.next().is_some() { return Err("复制层提供了过多 ID。".into()); }
                    let copy_id = copy.id().ok_or("缺少复制层 ID。")?.to_owned();
                    let anchor = before.as_deref().or(after.as_deref());
                    let (slot, group) = if let Some(target) = to_slot {
                        if !recipe.inputs.iter().any(|input| format!("{}_encoder", input.name) == target)
                            || recipe.components.contains_key(&target) || recipe.native_encoders.contains(&target) {
                            return Err("复制目标编码器不存在或已使用组件/原始输出。".into());
                        }
                        let roots = recipe.encoder_layers.entry(target.clone()).or_default();
                        let group = if let Some(anchor) = anchor { layer_group_mut(roots, anchor).ok_or("复制锚点不属于目标编码器。")? } else { roots };
                        (target, group)
                    } else {
                        let (slot, source_group) = find_recipe_group(&mut recipe, &id).ok_or("找不到源层。")?;
                        if let Some(anchor) = anchor && !source_group.iter().any(|item| item.id() == Some(anchor)) {
                            return Err("复制目标不在同一层组。".into());
                        }
                        (slot, source_group)
                    };
                    let position = match anchor {
                        Some(anchor) => group.iter().position(|item| item.id() == Some(anchor)).ok_or("复制锚点不存在。")? + usize::from(after.is_some()),
                        None => group.len(),
                    };
                    let digest = sha256_bytes(&serde_json::to_vec(&copy).map_err(|error| error.to_string())?);
                    group.insert(position, copy);
                    ("copy", copy_id, slot, group.iter().map(|item| item.id()).collect::<Vec<_>>(), Some(digest))
                }
                LayerEdit::Move { id, before } => {
                    let (slot, group) = find_recipe_group(&mut recipe, &id).ok_or("找不到目标层。")?;
                    let from = group.iter().position(|item| item.id() == Some(id.as_str())).ok_or("找不到目标层。")?;
                    if before.as_deref() != Some(id.as_str()) {
                        let target = match before { Some(ref anchor) => group.iter().position(|item| item.id() == Some(anchor.as_str())).ok_or("移动目标不在同一层组。")?, None => group.len() };
                        let item = group.remove(from);
                        group.insert(if from < target { target - 1 } else { target }, item);
                    }
                    ("move", id, slot, group.iter().map(|item| item.id()).collect::<Vec<_>>(), None)
                }
            };
            if order.iter().any(|id| id.is_none()) { return Err("编辑的层组须先为每一层启用稳定 ID。".into()); }
            let order = order.into_iter().map(|id| id.map(str::to_owned)).collect::<Vec<_>>();
            Ok(json!({"action":action,"id":id,"slot":slot,"order":order,"layer_sha256":layer_digest}))
        })();
        let outcome = match result { Ok(value) => value, Err(error) => return fail("RECIPE_LAYER_INVALID", format!("第 {} 项：{error}", operation_index + 1)) };
        recipe.encoder_layers.retain(|_, layers| !layers.is_empty());
        let mut seen = HashSet::new();
        let mut count = 0;
        for layers in recipe.encoder_layers.values() {
            if let Err(error) = collect_identity_ids(layers, &mut seen).and_then(|()| validate_edit_groups(layers, 0, &mut count)) {
                return fail("RECIPE_LAYER_INVALID", format!("第 {} 项：{error}", operation_index + 1));
            }
        }
        outcomes.push(outcome);
    }
    let after = match serde_json::to_vec(&recipe) { Ok(value) => value, Err(error) => return fail("RECIPE_LAYER_INVALID", error.to_string()) };
    let validation = match recipe_source(&recipe) {
        Ok(source) => json!({"status":"generated_statically","source_sha256":sha256_bytes(source.as_bytes()),"runtime_verified":false}),
        Err(error) => json!({"status":"unavailable_or_invalid","message":error,"runtime_verified":false}),
    };
    let plan = json!({"protocol_version":1,"input_sha256":revision,"operations_sha256":operations_revision,
        "result_sha256":sha256_bytes(&after),"outcomes":outcomes,"validation":validation,"committed":false});
    let plan_revision = match crate::domain::input::content_revision(&plan) { Ok(value) => value, Err(error) => return fail("RECIPE_LAYER_PLAN_FAILED", error) };
    Report::ok("recipe_layer_edit_plan", json!({"plan_revision":plan_revision,"plan":plan,
        "note":"仅规划草稿；未修改文件或运行模型。所有操作只有全部合法才返回计划。"}))
}

pub(super) fn plan_layer_moves(path: &Path, expected: &str, raw_moves: &str) -> Report {
    let fail = |code, message: String| Report::error("recipe_layer_plan", code, message, "检查草稿修订、层 ID 和目标组后重试。");
    let bytes = match read_recipe_for_update(path) {
        Ok(value) => value,
        Err(error) => return fail("RECIPE_LAYER_READ_FAILED", error),
    };
    let revision = sha256_bytes(&bytes);
    if revision != expected { return fail("RECIPE_LAYER_CONFLICT", "草稿内容与请求修订不一致。".into()); }
    if raw_moves.len() > 16_384 { return fail("RECIPE_LAYER_INVALID", "移动请求过大。".into()); }
    let moves: Vec<LayerMove> = match serde_json::from_str(raw_moves) {
        Ok(value) => value,
        Err(error) => return fail("RECIPE_LAYER_INVALID", error.to_string()),
    };
    if moves.is_empty() || moves.len() > 64 { return fail("RECIPE_LAYER_INVALID", "需要 1 至 64 个移动操作。".into()); }
    let mut recipe = match ScaffoldRecipe::parse(&bytes) {
        Ok(value) => value,
        Err(error) => return fail("RECIPE_LAYER_INVALID", error),
    };
    let mut seen = HashSet::new();
    for layers in recipe.encoder_layers.values() {
        if let Err(error) = collect_identity_ids(layers, &mut seen) { return fail("RECIPE_LAYER_ID_INVALID", error); }
    }
    let mut outcomes = Vec::new();
    for (operation_index, operation) in moves.iter().enumerate() {
        let group = recipe.encoder_layers.values_mut().find_map(|layers| layer_group_mut(layers, &operation.id));
        let Some(group) = group else { return fail("RECIPE_LAYER_NOT_FOUND", format!("第 {} 项找不到层 {}。", operation_index + 1, operation.id)); };
        if group.iter().any(|layer| layer.id().is_none()) {
            return fail("RECIPE_LAYER_ID_REQUIRED", "被编辑层组中的每层都需要稳定 ID。".into());
        }
        let Some(from) = group.iter().position(|layer| layer.id() == Some(operation.id.as_str())) else { unreachable!() };
        if operation.before.as_deref() != Some(operation.id.as_str()) {
            let target = match &operation.before {
                Some(id) => match group.iter().position(|layer| layer.id() == Some(id.as_str())) {
                    Some(index) => index,
                    None => return fail("RECIPE_LAYER_TARGET_INVALID", format!("第 {} 项的目标不在同一层组：{id}。", operation_index + 1)),
                },
                None => group.len(),
            };
            let layer = group.remove(from);
            group.insert(if from < target { target - 1 } else { target }, layer);
        }
        outcomes.push(json!({"id":operation.id,"before":operation.before,"order":group.iter().map(|layer| layer.id().unwrap()).collect::<Vec<_>>() }));
    }
    let after = match serde_json::to_vec(&recipe) { Ok(value) => value, Err(error) => return fail("RECIPE_LAYER_INVALID", error.to_string()) };
    let validation = match recipe_source(&recipe) {
        Ok(source) => json!({"status":"generated_statically","source_sha256":sha256_bytes(source.as_bytes()),"runtime_verified":false}),
        Err(error) => json!({"status":"unavailable_or_invalid","message":error,"runtime_verified":false}),
    };
    let plan = json!({"protocol_version":1,"input_sha256":revision,"result_sha256":sha256_bytes(&after),
        "moves":moves,"outcomes":outcomes,"validation":validation,"committed":false});
    let plan_revision = match crate::domain::input::content_revision(&plan) {
        Ok(value) => value,
        Err(error) => return fail("RECIPE_LAYER_PLAN_FAILED", error),
    };
    Report::ok("recipe_layer_plan", json!({"plan_revision":plan_revision,"plan":plan,
        "note":"仅规划草稿，未修改文件、导入组件或运行模型；保存仍需审核生成计划。"}))
}

// A recipe with named outputs contains enough information to reproduce the
// generated source without importing or executing any project code.
fn recipe_source(recipe: &ScaffoldRecipe) -> Result<String, String> {
    if recipe.outputs.is_empty() { return Err("旧配置没有记录输出维度。".into()); }
    let (input_args, output_args) = recipe.arguments()?;
    let (inputs, _) = inputs_from_args(&input_args)?;
    let routes = resolve_routes(&inputs, &recipe.routes.iter().map(|(slot, input)| format!("{slot}={input}")).collect::<Vec<_>>())?;
    let injections = recipe.components.iter().map(|(slot, factory)| format!("{slot}={factory}")).collect::<Vec<_>>();
    let layers = recipe.encoder_layers.iter().map(|(slot, layers)| {
        serde_json::to_string(layers).map(|value| format!("{slot}={value}")).map_err(|error| error.to_string())
    }).collect::<Result<Vec<_>, _>>()?;
    let (_, resolved) = resolve_encoder_layers(&inputs, &routes, &injections, &layers)?;
    let features = inputs.iter().zip(&routes).try_fold(0usize, |total, (input, &route)| {
        let slot = format!("{}_encoder", input.name);
        total.checked_add(recipe.encoder_widths.get(&slot).copied().unwrap_or_else(|| {
            resolved.get(&slot).map_or_else(|| flattened_width(&inputs[route]), |encoder| encoder.output_width)
        })).ok_or("特征数过大。")
    })?;
    let outputs = named_outputs_from_args(&output_args, features)?;
    let native = recipe.native_encoders.iter().cloned().collect();
    Ok(source(&inputs, &routes, &native, &resolved, &recipe.encoder_widths, features, &outputs, Objective::Mse))
}

fn previous_source_state(root: &Path, update: &RecipeUpdate) -> Value {
    let Some(module) = update.config.as_ref().and_then(|config| config["model"].as_str()).and_then(|model| model.strip_suffix(":model")) else {
        return json!({"status":"unknown"});
    };
    // prepare_recipe_update has already restricted module to our generated names.
    let relative = format!("{module}.py");
    let path = root.join(&relative);
    let bytes = (|| -> Option<Vec<u8>> {
        let metadata = fs::symlink_metadata(&path).ok()?;
        if !metadata.is_file() || metadata.len() > 8 * 1024 * 1024 { return None; }
        let mut bytes = Vec::new();
        fs::File::open(path).ok()?.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes).ok()?;
        (bytes.len() <= 8 * 1024 * 1024).then_some(bytes)
    })();
    let Some(bytes) = bytes else { return json!({"status":"unknown", "file":relative}); };
    let status = match recipe_source(&update.recipe) {
        Ok(expected) if expected.as_bytes() == bytes => "unchanged",
        Ok(_) => "different",
        Err(_) => "unknown",
    };
    json!({"status":status, "file":relative, "revision":sha256_bytes(&bytes)})
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn read_recipe_for_update(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !metadata.is_file() || metadata.len() > RECIPE_LIMIT {
        return Err("model-maven.json 必须是小于 128 KiB 的普通文件。".into());
    }
    let file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    file.take(RECIPE_LIMIT + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
    if bytes.len() as u64 > RECIPE_LIMIT { return Err("配置文件不能超过 128 KiB。".into()); }
    Ok(bytes)
}

fn lock_project_config(root: &Path) -> Result<fs::File, String> {
    use fs2::FileExt as _;
    let directory = root.join(".model-maven");
    let metadata = fs::symlink_metadata(&directory).map_err(|error| error.to_string())?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() { return Err("项目元数据目录不可用。".into()); }
    let path = directory.join("project.json.lock");
    if fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.file_type().is_symlink()) { return Err("项目配置锁不能是符号链接。".into()); }
    let mut options = fs::OpenOptions::new();
    options.create(true).truncate(false).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let lock = options.open(path).map_err(|error| error.to_string())?;
    lock.try_lock_exclusive().map_err(|_| "项目配置正在更新，请稍后重试。".to_owned())?;
    Ok(lock)
}

pub(super) fn replace_recipe(path: &Path, expected: &[u8], contents: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("配置目录不存在。")?;
    let mut staged = tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
    staged.write_all(contents).and_then(|()| staged.as_file().sync_all())
        .map_err(|error| error.to_string())?;
    if read_recipe_for_update(path)? != expected {
        return Err("model-maven.json 已变化，请重新打开画布。".into());
    }
    staged.persist(path).map_err(|error| error.error.to_string())?;
    Ok(())
}

fn remove_unchanged_generated_file(path: &Path, expected: &[u8]) {
    if fs::read(path).is_ok_and(|current| current == expected) {
        let _ = fs::remove_file(path);
    }
}
