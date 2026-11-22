use super::*;
use std::{collections::HashSet, fmt::Write as _, fs};

struct InputSpec {
    name: String,
    shape: Vec<usize>,
    dynamic_axes: Vec<usize>,
    dtype: String,
}

fn inputs_from_args(args: &[String]) -> Result<(Vec<InputSpec>, usize), String> {
    if args.is_empty() || args.len() > 8 {
        return Err("需要 1 至 8 个具名输入。".into());
    }
    let mut names = HashSet::new();
    let mut total = 0usize;
    let mut sample_total = 0usize;
    let mut inputs = Vec::new();
    for arg in args {
        let (typed_name, dimensions) = arg
            .split_once('=')
            .ok_or("输入格式应为 name[:dtype]=3,16,16。")?;
        let (name, dtype) = typed_name
            .split_once(':')
            .unwrap_or((typed_name, "float32"));
        if !matches!(dtype, "float32" | "float64" | "float16" | "int64" | "bool") {
            return Err(format!(
                "输入 `{name}` 的 dtype `{dtype}` 不受支持；请选择 float32、float64、float16、int64 或 bool。"
            ));
        }
        let mut chars = name.chars();
        if name.len() > 48
            || !chars.next().is_some_and(|c| c.is_ascii_alphabetic())
            || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            || !names.insert(name.to_owned())
        {
            return Err(format!(
                "输入名 `{name}` 必须唯一，以字母开头且只含 ASCII 字母、数字或下划线。"
            ));
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
        if shape.is_empty() || shape.len() > 4 || shape.contains(&0) {
            return Err("每个输入需要 1 至 4 个正维度，不包含批次轴。".into());
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
            dtype: dtype.to_owned(),
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
            .ok_or_else(|| format!("未知编码器接缝 `{slot}`。"))?;
        let source = inputs
            .iter()
            .position(|item| item.name == input)
            .ok_or_else(|| format!("未知输入 `{input}`。"))?;
        if !assigned.insert(target) {
            return Err(format!("接缝 `{slot}` 被重复接线。"));
        }
        routes[target] = source;
    }
    Ok(routes)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Objective {
    Mse,
    CrossEntropy,
}
impl Objective {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "mse" => Ok(Self::Mse),
            "ce" | "cross_entropy" => Ok(Self::CrossEntropy),
            _ => Err("损失必须是 mse 或 ce（交叉熵）。".into()),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Mse => "mse",
            Self::CrossEntropy => "cross_entropy",
        }
    }
}

fn named_outputs_from_args(
    args: &[String],
    features: usize,
) -> Result<Vec<(String, usize, Objective, f64)>, String> {
    if args.len() > 8 {
        return Err("最多声明 8 个具名输出。".into());
    }
    let mut seen = HashSet::new();
    let mut total = 0usize;
    let mut result = Vec::new();
    for arg in args {
        let (name, raw) = arg.split_once('=').ok_or("输出格式应为 name=特征数。")?;
        let mut chars = name.chars();
        if name.len() > 48
            || !chars.next().is_some_and(|c| c.is_ascii_alphabetic())
            || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            || !seen.insert(name.to_owned())
        {
            return Err(format!(
                "输出名 `{name}` 必须唯一，以字母开头且只含 ASCII 字母、数字或下划线。"
            ));
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
    output: usize,
    named_outputs: &[(String, usize, Objective, f64)],
    objective: Objective,
) -> String {
    let mut text = String::from(
        "# Editable Model-Maven starter. Replace encoders, fusion, head and step as needed.\n# Batch values and targets are synthetic; they are not a dataset.\n# Default Concat casts encoded features to float32 for the default Linear head.\n# Inputs reach encoders with their declared dtype; replace fusion for other contracts.\nimport torch\nfrom torch import nn\n\n\nclass Concat(nn.Module):\n    def forward(self, values):\n        return torch.cat([value.to(dtype=torch.float32) for value in values], dim=1)\n\n\n",
    );
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
    let params = inputs
        .iter()
        .map(|input| format!("{}_encoder=None", input.name))
        .chain([
            "fusion=None".into(),
            "head=None".into(),
            "input_routes=None".into(),
            "encoder_output_modes=None".into(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(text, "class Model(nn.Module):\n    def __init__(self, {params}):\n        super().__init__()\n        self.encoders = nn.ModuleDict({{").unwrap();
    for input in inputs {
        writeln!(
            text,
            "            \"enc_{}\": {}_encoder if {}_encoder is not None else nn.Identity(),",
            input.name, input.name, input.name
        )
        .unwrap();
    }
    text.push_str("        })\n        self.input_routes = {\n");
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
        let features = input
            .shape
            .iter()
            .enumerate()
            .fold(1usize, |n, (axis, value)| {
                n * if input.dynamic_axes.contains(&axis) {
                    1
                } else {
                    *value
                }
            });
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
    writeln!(text, "        width = sum(self.input_features[source] for source in self.input_routes.values())\n        if head is None and (width > 1000000 or width * {total_output} > 10000000):\n            raise ValueError(\"Routed default head is too large; provide a custom head or change routes\")\n        self.head = head if head is not None else {default_head}\n\n    def _encode(self, slot, inputs):\n        source = self.input_routes[slot]\n        value = self.encoders[\"enc_\" + slot[:-8]](inputs[source])\n        if slot in self.native_encoder_outputs:\n            return value\n        axes = self.input_dynamic_axes[source]\n        if axes:\n            value = value.to(dtype=torch.float32).mean(dim=tuple(axis + 1 for axis in axes), keepdim=True)\n        return value.flatten(start_dim=1)\n\n    def forward(self, **inputs):\n        values = [").unwrap();
    for input in inputs {
        writeln!(
            text,
            "            self._encode(\"{}_encoder\", inputs),",
            input.name
        )
        .unwrap();
    }
    text.push_str("        ]\n        return self.head(self.fusion(values))\n\n\n");
    let args = inputs
        .iter()
        .map(|input| format!("{}_encoder={}_encoder", input.name, input.name))
        .chain([
            "fusion=fusion".into(),
            "head=head".into(),
            "input_routes=input_routes".into(),
            "encoder_output_modes=encoder_output_modes".into(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    let mut batch_params = vec!["batch_size=2".to_owned()];
    for input in inputs {
        for axis in &input.dynamic_axes {
            batch_params.push(format!(
                "{}_dim{}={}",
                input.name,
                axis + 1,
                input.shape[*axis]
            ));
        }
    }
    writeln!(text, "def model({params}):\n    return Model({args})\n\n\n# Non-native encoder outputs are averaged on dynamic axes and flattened.\n# Batch lengths below only control synthetic samples; they do not constrain a dataset.\ndef batch({}):", batch_params.join(", ")).unwrap();
    for param in &batch_params {
        let name = param.split_once('=').unwrap().0;
        writeln!(text, "    if not isinstance({name}, int) or isinstance({name}, bool) or {name} < 1:\n        raise ValueError(\"{name} must be a positive integer\")").unwrap();
    }
    text.push_str("    return {\n        \"args\": [],\n        \"kwargs\": {\n");
    for input in inputs {
        let dims = input
            .shape
            .iter()
            .enumerate()
            .map(|(axis, value)| {
                if input.dynamic_axes.contains(&axis) {
                    format!("{}_dim{}", input.name, axis + 1)
                } else {
                    value.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            text,
            "            \"{}\": torch.ones((batch_size, {dims}), dtype=torch.{}),",
            input.name, input.dtype
        )
        .unwrap();
    }
    if named_outputs.is_empty() {
        match objective {
            Objective::Mse => writeln!(text, "        }},\n        \"loss_inputs\": {{\"target\": torch.zeros((batch_size, {output}), dtype=torch.float32)}},\n    }}\n\n\ndef step(model, batch, context):\n    prediction = model(**batch[\"kwargs\"])\n    return {{\"loss\": (prediction - batch[\"loss_inputs\"][\"target\"]).square().mean()}}\n").unwrap(),
            Objective::CrossEntropy => writeln!(text, "        }},\n        \"loss_inputs\": {{\"target\": torch.arange(batch_size, dtype=torch.int64) % {output}}},\n    }}\n\n\ndef step(model, batch, context):\n    logits = model(**batch[\"kwargs\"])\n    return {{\"loss\": nn.functional.cross_entropy(logits, batch[\"loss_inputs\"][\"target\"])}}\n").unwrap(),
        }
    } else {
        text.push_str("        },\n        \"loss_inputs\": {\"targets\": {\n");
        for (name, width, objective, _) in named_outputs {
            match objective {
                Objective::Mse => writeln!(text, "            \"{name}\": torch.zeros((batch_size, {width}), dtype=torch.float32),").unwrap(),
                Objective::CrossEntropy => writeln!(text, "            \"{name}\": torch.arange(batch_size, dtype=torch.int64) % {width},").unwrap(),
            }
        }
        text.push_str("        }},\n    }\n\n\ndef step(model, batch, context):\n    prediction = model(**batch[\"kwargs\"])\n    targets = batch[\"loss_inputs\"][\"targets\"]\n    losses = {\n");
        for (name, _, objective, _) in named_outputs {
            match objective {
                Objective::Mse => writeln!(text, "        \"{name}\": (prediction[\"{name}\"] - targets[\"{name}\"]).square().mean(),").unwrap(),
                Objective::CrossEntropy => writeln!(text, "        \"{name}\": nn.functional.cross_entropy(prediction[\"{name}\"], targets[\"{name}\"]),").unwrap(),
            }
        }
        text.push_str("    }\n    # Raw task losses are reported separately; weights affect the optimized total.\n    weights = {\n");
        for (name, _, _, weight) in named_outputs {
            writeln!(text, "        \"{name}\": {weight:?},").unwrap();
        }
        text.push_str("    }\n    return {\"loss\": sum(weights[name] * loss for name, loss in losses.items()), \"losses\": losses}\n");
    }
    text.push_str("\n# These starter callables have no cursor or mutable state. If you add state,\n# remove the declaration and implement state_dict/load_state_dict for resume.\nbatch.MODEL_MAVEN_STATELESS = True\nstep.MODEL_MAVEN_STATELESS = True\n");
    text
}

impl<B: Backend> Application<B> {
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
        output_dim: Option<usize>,
        output_args: &[String],
        loss: Option<&str>,
        apply: bool,
        if_plan: Option<&str>,
    ) -> Report {
        let fail = |code, msg: String| {
            Report::error("scaffold", code, msg, "修正输入或目标目录后重新预览。 ")
        };
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
        let source_relative = format!("{}.py", module.replace('.', "/"));
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
        let features = routes.iter().try_fold(0usize, |total, &index| {
            let input = &inputs[index];
            let size = input
                .shape
                .iter()
                .enumerate()
                .try_fold(1usize, |value, (axis, dim)| {
                    value.checked_mul(if input.dynamic_axes.contains(&axis) {
                        1
                    } else {
                        *dim
                    })
                })?;
            total.checked_add(size)
        });
        let Some(features) = features.filter(|features| *features <= 1_000_000) else {
            return fail(
                "SCAFFOLD_ROUTE_INVALID",
                "接线后的融合特征不能超过 1000000。".into(),
            );
        };
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
        for file in [root.join(&source_relative), metadata.join("project.json")] {
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
        let model = format!("{module}:model");
        let batch = external_batch
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{module}:batch"));
        let batch_source = if external_batch.is_some() {
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
        let preview = init(true);
        if !preview.success() {
            return preview;
        }
        let source = source(
            &inputs,
            &routes,
            &native_encoders,
            output,
            &named_outputs,
            objective,
        );
        let descriptors: Vec<Value> = inputs.iter().map(|i| json!({"name": i.name, "shape": i.shape, "dynamic_axes": i.dynamic_axes, "dynamic_reduction":if native_encoders.contains(&format!("{}_encoder", i.name)) { "none_native_encoder_output" } else { "mean_after_encoder" }, "encoder_output":if native_encoders.contains(&format!("{}_encoder", i.name)) { "native" } else { "flatten" }, "dtype":i.dtype, "encoder_slot":format!("{}_encoder", i.name)})).collect();
        let connections: Vec<Value> = inputs.iter().enumerate().map(|(index, input)| json!({"encoder_slot":format!("{}_encoder", input.name), "input":inputs[routes[index]].name})).collect();
        let output_descriptor = if named_outputs.is_empty() {
            if objective == Objective::Mse {
                json!({"features":output,"dtype":"float32"})
            } else {
                json!({"features":output,"dtype":"float32","objective":objective.name()})
            }
        } else {
            json!({"kind":"named","heads":named_outputs.iter().map(|(name, width, objective, weight)| json!({"name":name,"features":width,"dtype":"float32","objective":objective.name(),"weight":weight})).collect::<Vec<_>>()})
        };
        let plan = json!({"version":1,"project_root":root,"inputs":descriptors,"routes":connections,"output":output_descriptor,"slots":["fusion","head"],"source":source,"config":preview.data["config"],"files":[source_relative, ".model-maven/project.json"],"batch_source":batch_source});
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
                json!({"committed":false,"plan_revision":revision,"plan":public_plan,"redacted_fields":redacted_fields,"note":if external_batch.is_some() { "已有批次工厂尚未执行或验证；生成文件仍包含合成示例，示例损失需与真实目标匹配。" } else { "骨架使用合成输入与占位损失；运行前向或训练前应替换为项目数据和目标。" }}),
            );
        }
        if if_plan != Some(revision.as_str()) {
            return fail(
                "SCAFFOLD_PLAN_CONFLICT",
                "计划修订不匹配；请重新预览。".into(),
            );
        }
        let path = match self.project_source_writer.write_new(
            &root,
            Path::new(&source_relative),
            source.as_bytes(),
        ) {
            Ok(p) => p,
            Err((code, msg)) => return fail(code, msg),
        };
        let setup = init(false);
        if !setup.success() {
            if fs::read(&path).is_ok_and(|current| current == source.as_bytes()) {
                let _ = fs::remove_file(path);
            }
            return setup;
        }
        Report::ok(
            "scaffold",
            json!({"committed":true,"plan_revision":revision,"model":model,"batch":batch,"files":plan["files"],"note":"骨架已创建；实际前向尚未执行。"}),
        )
    }
}
