# 模型端口契约

`model probe` 可以在只提供模型和一个批次的情况下检查可选端口契约，不需要 DataLoader、Loss、优化器或训练循环。通过 `--contract` 指定 UTF-8 JSON 文件；TTY 工作台的 Probe 表单也可选择该文件。

```sh
m-model model probe \
  --model my_model:build_model \
  --batch my_model:example_batch \
  --contract model.contract.json
```

契约文件示例：

```json
{
  "version": 1,
  "inputs": {
    "args[0]": {"dtype": "float32", "shape": ["B", 4]}
  },
  "outputs": {
    "features": {"dtype": "float32", "shape": ["B", 3]}
  },
  "modules": {
    "model.projection": {
      "inputs": {"args[0]": {"shape": ["B", 4]}},
      "outputs": {"outputs": {"shape": ["B", 3]}}
    }
  },
  "symbols": {"B": {"min": 1, "max": 64}},
  "static": true
}
```

端口路径对应 Probe 展示的嵌套结构：`args[0]`、`kwargs.image`、`features`。`modules` 中的路径使用模型树路径；每个实际调用单独检查，并保留 invocation ID。形状维度可写固定整数、已声明符号或 `null`（不约束）。符号在所有已检查端口间共享，并可声明 `min`、`max` 或 `equal`。

`dtype` 支持 `float32` 和 `torch.float32` 两种写法。`layout` 检查 PyTorch Tensor layout（例如 `torch.strided`）；`memory_format` 支持 `contiguous`、`channels_last` 和 `channels_last_3d`。本版本不推断 NCHW/NHWC 等语义轴顺序，也不会把内存格式误报成语义布局。

契约错误返回 `MODEL_CONTRACT_FAILED`，并附带失败路径、expected、actual 和覆盖范围。FX/Meta 可推演时会报告 `inferred-static`；已证实的维度不匹配会在真实前向前失败。动态 Python 分支或无 Meta kernel 的算子会标记 `unsupported`，随后仍运行真实 Probe；只对本次实际执行的分支报告 `observed-runtime`，未执行的模块端口标记 `unobserved`。单批次成功不能证明未覆盖分支或所有形状范围。

## 局部调用重放与证据复用

对一个已在根前向中观察到的 Layer 调用，可以指定模块路径、调用序号和 `--local`，用刚才捕获的调用输入重放该 Layer：

```sh
m-model model probe model.encoder.block \
  --model my_model:build_model \
  --batch my_model:example_batch \
  --call 2 \
  --local
```

局部重放复制 Layer 和输入 Tensor，只在当前进程内暂存边界，不写入批次值。它报告对应的 `invocation_id` 和输入结构签名；单次最多重放 8 个调用，超出时显式置 `local_probe_reports_truncated`。模块副本不能隔离用户代码访问的全局变量、文件、网络或其他进程外副作用；重放也不包含根模型其余分支、反向传播或权重更新。

跨进程缓存只针对显式提供 `--batch-fingerprint`、且 Dataset 返回稳定 `sample_id(index)` 的输入启用。示例：

```sh
m-model model probe \
  --model my_model:build_model \
  --batch my_data:dataset \
  --batch-fingerprint cifar10/train-v4 \
  --seed 7
```

缓存身份覆盖项目 Python/组件描述源码、模型权重、组件注入、模型和批次参数、变体、契约、Python/PyTorch/已安装包版本、数据切分/样本身份及 Probe 调用选择。只改变运行超时等预算不改变身份。数据提供方应在内容或切分变化时更新 `--batch-fingerprint`；工具按该声明和稳定样本 ID 判断批次身份，不保存或回读原始批次值。任何身份不匹配都重新 Probe；动态/不支持 FX 路径、训练模式、缺失稳定样本身份、无法完整盘点的源码或过大的权重不会复用。缓存仅保留形状、dtype、布局、调用身份和契约检查元数据，文件权限限当前用户，最多保留 64 条记录；不保存原始 Tensor、批次值、批次工厂参数或凭据。

当缓存身份仅因可定位的项目/组件源码发生变化时，静态 FX/Meta 图、未截断调用轨迹、稳定输入内容签名和唯一模块调用身份允许逐调用比较证据。报告 `data.evidence_cache.branch_reuse.status=partial_reuse` 表示复用了匹配且未受影响调用的结构元数据，并列出 `reused` 与 `revalidated` 调用；根模型前向仍会在本次真实执行，`root_forward_executed` 为 `true`，`tensor_reuse` 为 `false`。无法归属的入口/helper 源码变化、动态/不支持图、共享别名或重复调用、输入签名不完整、调用轨迹截断，或样本/权重/依赖/契约/探测范围等身份同时改变时，报告 `whole_probe` 和 `expanded_scope_reason`，不做分支级复用。这是结构证据的增量复用，不会跳过模型计算，也不代表未经执行的分支已验证。

## 手动语义标记（实验性）

结构端口契约只说明 Tensor 的 shape、dtype 等事实，不说明某个端口代表 embedding、logits、action 或其他业务语义。用户可在一次真实 Probe 后，将自定义语义标记绑定到已观察的模块路径和端口：

```sh
m-model model mark \
  --model my_model:build_model \
  --batch my_model:example_batch \
  --path model.encoder.projection \
  --port outputs.features \
  --role embedding \
  --meaning "供检索头使用的潜在表示"
```

命令会重新执行前向，只接受该模块本次实际出现的输入/输出端口；不会复用缺少源码锚点的 Probe 缓存，也不会保存 Tensor 值。标记写入 `.model-maven/semantic-marks.json`，包含用户输入的角色/含义、模块路径、端口、项目内实现文件 SHA-256、模型参数/注入/变体/已加载项目源码指纹及临时 Probe 来源。源码变化后 `model marks` 会报告 stale；添加 `--model` 与 `--batch` 会执行完整绑定审计，检查模块/端口、模型参数、注入、变体及已加载项目源码身份。

```sh
m-model model marks
m-model model marks --model my_model:build_model --batch my_model:example_batch
m-model model retag --id MARK_ID --model my_model:build_model \
  --batch my_model:example_batch --path model.encoder.projection \
  --port outputs.features --role embedding --meaning "检索表示"
m-model model unmark --id MARK_ID
```

外部包源码、歧义路径、截断轨迹及未观察端口不能建立或重新绑定标记。没有提供当前模型/批次时，list 仅复核保存的项目源码锚点；动态分支覆盖须以带模型/批次的审计为准。

语义角色由用户手动填写，不从变量名或 Python 源码自动推断。规则需要先绑定到某个手动标记，再运行单批次只读预览：

```sh
m-model model supervise rule add --mark embedding --kind range --mode require --min -1 --max 1
m-model model supervise rule add --mark embedding --kind range --mode penalty --min -1 --max 1 --weight 0.05
m-model model supervise rule add --mark logits --kind batch-alignment --mode require --target-path batch.kwargs.labels --axis 0
m-model model supervise rule list
m-model model supervise preview --model my_model:build_model --batch my_model:example_batch
```

领域约束可以注册为项目内的 Python callable。validator 用 `module:callable` 引用项目源码，入口签名为 `(value, batch, params)`，返回 `passed`、可选 JSON `summary`，以及可选的同形状布尔 `violations`：

```python
def valid_logits(value, batch, params):
    violations = value.abs() > params["limit"]
    return {
        "passed": not bool(violations.any().item()),
        "summary": {"limit": params["limit"]},
        "violations": violations,
    }
```

```sh
m-model model supervise rule add --mark logits --kind custom-validator \
  --mode require --plugin constraints:valid_logits \\
  --plugin-param limit=20.0 --axis 0 --id bounded-logits
```

validator 仅支持 `observe` / `require`。若提供 `--axis`，须同时返回和标记端口同 shape 的 bool `violations`；工具据此用稳定 sample IDs 定位违规。省略 mask 时只能报告整体通过/失败，不能定位样本。自定义 penalty 也使用 `(value, batch, params)`，但必须返回有限、可微的标量浮点 Tensor：

```python
def energy_penalty(value, batch, params):
    return (value - params.get("target", 0.0)).square().mean()
```

```sh
m-model model supervise rule add --mark representation --kind custom-penalty \
  --mode penalty --plugin constraints:energy_penalty \\
  --plugin-param target=0.0 --weight 0.05 --id representation-energy
```

自定义规则只接受项目内普通 `.py` 文件，入库时记录源码相对路径、大小和 SHA-256；预览/训练会复核源码身份，变更后必须重新添加规则。参数是有限大小的 JSON 值。插件接收标记 Tensor 和 batch Tensor 的副本；penalty 的值副本保留梯度。源码指纹用于可追溯性，不是安全隔离：callable 以当前 Python 进程权限执行，可能访问文件、网络或进程资源。因此只加载你信任的项目代码，不要把凭据放进 `--plugin-param`。当前没有 OS 沙箱、超时/CPU/内存预算或副作用拦截；这不适合执行不可信插件。

`observe` 报告有限值/范围摘要，`require` 预览会报告 `passed` 或 `would_reject`，`penalty` 展示 detached 的加权数值估计。`batch-alignment` 需要用户指定真实批次 Tensor 路径和维度，例如把 logits 的 axis 0 与 `batch.kwargs.labels` 的 axis 0 对齐；它只检查尺寸相等，不猜任务类型或标签语义，且不支持 penalty。预览只观测当前批次实际执行的调用，不保存 Tensor，不反向传播、不更新权重，也不改变训练 Loss；源码或模型绑定变化时拒绝观测并报告 stale。负范围值可直接跟在 `--min` / `--max` 后。

确认预览结果后，可在 `managed-step` / `manual-step` 试跑或正式训练中显式启用规则：

```sh
m-model try --model my_model:build_model --batch my_model:batch \
  --step my_training:step --steps 1 --apply-supervision
m-model train --model my_model:build_model --batch my_model:batch \
  --step my_training:step --steps 100 --apply-supervision
```

开关关闭时不会安装监督 hook，也不会改训练 Loss。启用后，`require` 在 backward/optimizer update 前检查并拒绝违规步骤；`penalty` 将命名的加权范围 penalty 加进 Loss，运行结果记录规则快照、覆盖计数、违反计数、贡献和梯度结果。标记源码、模型参数、注入或变体不匹配会在训练开始前 fail closed。启用规则的正式恢复会把规则快照纳入 checkpoint run signature。

在 `manual-step` 中，`require` 会在标记端口 forward 时检查，并在已追踪 optimizer 更新前复核；penalty 必须通过 `context.backward(total_loss)` 加入目标函数。带 penalty 的 manual-step 只接受一次聚合 backward，避免多次注入造成目标不明确。

**当前边界：** `batch-alignment` 可把手动选定的输出端口与当前 batch Tensor 的一个轴绑定，并在 managed/manual-step 或显式接入的 external-loop 更新前执行；它只证明尺寸相等。对 `finite`/`range`，用户可用 `--axis` 手动指定样本轴，并在 batch envelope 的 `metadata` 中提供唯一稳定字符串 ID，或在自定义 training step 的 forward 前调用 `context.report_sample_ids(ids)`。external-loop 的自有循环须把 forward 放入 `context.supervised_step(batch)`；penalty 使用 `context.backward(loss)`；所有更新模型的 optimizer 需先用 `context.register_optimizer(optimizer)` 安装更新前/后检查。预览与循环都由用户显式控制，未调用这些接口的训练步骤不记为已监督。

```python
def run_loop(model, context):
    optimizer = make_optimizer(model.parameters())
    context.register_optimizer(optimizer)
    for batch in my_data_iterator():
        optimizer.zero_grad(set_to_none=True)
        with context.supervised_step(batch):
            output = model(batch["inputs"])
            loss = task_loss(output, batch["targets"])
            total_loss = context.backward(loss)
            optimizer.step()
        if not context.report_step(total_loss.detach()):
            break
```

`require` 在受监督 forward 或 optimizer 前置钩子处拒绝违规/未观测端口；`penalty` 只通过 `context.backward()` 加入用户目标。使用自定义 optimizer 时，它必须支持 PyTorch 的 step 前后置钩子；不支持时 fail closed。循环仍拥有数据迭代、优化器创建和状态恢复；若要恢复 optimizer、sampler 或环境状态，循环须通过现有 `state_dict/load_state_dict` 钩子保存。对 `finite`/`range`，用户可在 batch envelope 的 `metadata` 中提供唯一稳定字符串 ID，或在受监督的步骤内调用 `context.report_sample_ids(ids)`。当前批次内 ID 数量与轴长匹配时，预览和训练会定位违规 ID；多批次计数和 checkpoint/resume 去重适用于 managed/manual/external 三种显式接入方式。未提供或不匹配时，该次样本定位报告为 unsupported，require/penalty 不会把它算作通过。checkpoint 保存 ID 的 SHA-256 摘要，不保存原始 ID；每条规则最多跟踪 10000 个摘要，超出时只报告下界。没有覆盖状态的旧 checkpoint 会标记历史步数未追踪，不能报完整覆盖。ID 由用户声明，工具不验证它们与原始数据记录的真实性；没有数据集总量，不计算总体覆盖率。任务标签值语义仍未实现。自定义 validator/penalty 已支持项目内版本化 callable，但只做源码身份复核，不提供 OS 沙箱、资源预算或副作用隔离。预览与训练整体 `coverage_status` 仍为 `partial`，预览中的 `would_reject` 仍只是只读预览状态。
