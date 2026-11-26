# 复杂模型探针与覆盖地图

这是仓库自带的合成 Python 示例，供人工检查模型探针、源码编辑和训练入口；不是用户项目、基准或全面通过验收的证明。它不下载权重或数据。样例入口依赖用户选择的 Python 环境中可导入的 PyTorch（`torch`）；本仓库不会替用户安装或选择运行环境。

`VideoTextModel` 包含视频与文本编码器、融合层以及两个输出头；融合层的 `shared_projection` 会被两种模态复用。`ConditionalVideoTextModel` 在此基础上增加一个由 `refine` 参数选择的分支。`direct_batch` 和 `refined_batch` 通过同一种子构造相同值的合成张量，只分别设置 `refine=False/True`；输入封装因此不同，但不保证模型权重、环境或其他随机状态一致。它们适合定位该条件分支，不构成其他随机因素受控的完整对照实验。

为避免改动已有项目配置，可把源码仓库或 npm 安装中的 `examples/` 复制到一个新的空项目目录，并在该目录打开开发中的统一工作区。npm 安装可用 `m-model-tui --examples-dir` 查看示例目录；这是 npm 启动器的选项。复制后保留 `examples/__init__.py` 和 `examples/complex_acceptance/` 的目录结构。按 TUI 当前《接入已有模型》指南选择以下模型和输入工厂（训练步骤可选）：

- 模型：`examples.complex_acceptance.conditional:build_conditional_model`
- 输入：`examples.complex_acceptance.conditional:direct_batch`
- 可选托管训练步骤：`examples.complex_acceptance.training:multitask_step`

在工作区按 `p` 对当前选定的模型和输入执行一次真实前向；按 `P` 设置该输入的 args/kwargs 绑定。`P` 只调整绑定，不会更换输入工厂。工作区可选择不同输入工厂并比较两次观察；本地走查见 [多输入观察记录](../../docs/acceptance/multi-input-comparison.md)。也可以使用下面列出的 `direct_batch` 与 `refined_batch` CLI 入口分别生成报告，再人工核对各自真实记录的模块调用；单次报告未记录某条路径不证明该路径从未执行。不同 CLI 运行也不固定模型初始化、Python 环境或其他随机状态，不应把它们视为完整受控比较。

用户可以打开 `examples/complex_acceptance/conditional.py`、`fusion.py` 等源码，观察并手动编辑；源码 Layer 操作仅支持当前后端能安全定位的构造形式，无法接管时继续手写。可对真实观察到的输出添加语义标记和规则，先预览后再决定是否在受支持的训练方式中启用。

这里提供两种不同序列长度的原有 `changing_batch`，但它使用进程内计数，不保证跨独立进程交替。需要明确比较分支时使用上面两个固定工厂。

## 覆盖地图

| 能力 | 本地样例和入口 | 应检查的真实证据 | 编辑与诊断范围 | 限制/未观测 |
| --- | --- | --- | --- | --- |
| 多文件模型与共享层 | `model.py`、`encoders.py`、`fusion.py`；`conditional:build_conditional_model` + `conditional:direct_batch` | Probe 报告中的模块调用路径、逐次调用序号、输入/输出 shape；`fusion.shared_projection` 在一次前向中应有两次调用 | 可查看并手动改 Python 源；受支持的 Layer 操作可对可定位模块做临时/变体编辑。`model probe` 可按模块路径检查调用 | 调用次数需以实际报告核实；报告只说明已观察到的调用，不证明未记录路径从未执行；非纯 PyTorch 或复杂构造形式可能无法源码接管 |
| 动态分支 | `conditional.py`；分别使用 `direct_batch`（`refine=False`）和 `refined_batch`（`refine=True`） | 两次各自的 Probe 报告；检查 `refinement` 是否在 True 输入报告中被调用，False 输入未观察到它 | 当前批次可做路径/shape 诊断；可编辑能安全定位的模块构造 | 两个批次分别重建随机张量，且不固定模型初始化、完整 RNG 或环境；不构成只改变单一变量的受控实验，也不能覆盖其他分支/输入 |
| 用户模块和接口契约变化 | `contract_repair.py`；`contract_repair:build_model` + 下面的 `data:changing_batch` | 首次真实前向应暴露 `VideoEncoderV2` 输出 48 特征与 `ContractFusion.video_projection` 输入 32 特征之间的运行时契约错误；报告应记录失败位置/源码栈 | 手动接管路径为 `contract_repair.py` → `ContractFusion.__init__` 的 `self.video_projection = nn.Linear(32, 16)`；按模型契约将输入维度修为 48 后再 Probe。源码 Layer 是否能接管须在真实操作中确认 | 该样例故意不匹配；本地走查已从失败报告定位并预览、提交修复，独立前向成功。源码编辑不保证支持任意 Python 重构 |
| 输入动态尺寸 | `data.py`；`data:changing_batch` | 每次真实 Probe 的帧数、token 数、mask 与模块张量 shape；可观察两种长度 | 对当前真实批次做前向和契约诊断 | 依靠进程内计数器交替；独立 CLI 进程可能总从同一状态开始，不能用它保证跨运行对照 |
| 托管训练步骤 | `training.py`；`training:multitask_step` + 条件模型/批次 | 有限 `try` 报告中的步数、总 loss、命名子损失和指标；本步骤包含分类、跨模态对齐、有效帧加权损失 | 本地代码展示 `training_step(model, batch, context, **params)` 契约；托管模式负责框架管理的反向传播/更新 | 本地 CPU 受管试跑完成 1 步，不代表收敛；不覆盖真实数据加载、评估器、分布式训练或持久训练效果 |
| 用户控制更新的手动步骤 | `training.py`；`training:manual_multitask_step` | 有限 `try` 报告和代码路径；步骤 1–2 冻结视频编码器，随后解冻；两个命名优化器组分别更新 | 调用者显式执行 `context.backward` 和 optimizer `step`；运行报告需核对工具记录的协议证据 | 本地 CPU 手动试跑完成 4 步并报告两个优化器组；冻结过程的逐步参数变化未单独核对。参数组重叠反例结果未采集 |
| 用户自带复杂模型 | 当前未提供 | 无 | 待获得用户可分享模型/入口后，按同一地图补充 | `blocked_external`：用户模型、依赖清单、许可/分发授权及真实数据均未提供；不能推断本地合成样例代表其行为 |

### 入口、依赖、许可与证据来源

| 样例 | 代表入口 | 运行依赖 | 许可来源 | 执行时应保存/观察 |
| --- | --- | --- | --- | --- |
| 本地双输入、共享层、动态分支 | `conditional:build_conditional_model` + `conditional:direct_batch` / `refined_batch` | 所选 Python 的 PyTorch (`torch`)，源码不固定版本 | 仓库根目录 `LICENSE`（MIT）；该目录未提供独立 SPDX 标记 | 两份真实 Probe 报告；`shared_projection` 的调用路径/次数、refinement True/False 分支调用及 tensor shape |
| 本地自定义模块契约探针 | `contract_repair:build_model` + `data:changing_batch` | 所选 Python 的 PyTorch；无模型权重或外部数据 | 同上；源码为本仓库本地示例 | Probe 失败报告、失败模块/端口和源码位置；手动修复或支持的编辑后重新 Probe 的结果 |
| 本地训练接缝 | `training:multitask_step` / `manual_multitask_step` + 条件模型/批次 | 所选 Python 的 PyTorch；训练模式和设备由用户选择/实际环境解析 | 同上；未复制第三方训练代码 | 真实 `try` 报告中的模式、步数、loss/子损失/指标、手动反向传播及优化器组证据 |
| 外部用户模型 | 未提供 | 未知，待用户提供环境/依赖 | 未提供模型来源及数据/权重许可 | `blocked_external`；不得以本地示例推断外部模型兼容性 |

## 本地操作入口

以下命令是在示例目录的父级项目目录执行的入口示例，要求已有 `m-model` 二进制/命令和用户选定、含 PyTorch 的 Python 环境。本地已观察两次分支前向及下面的训练步骤；这些命令仅供在自己的环境中复现，不保证其他环境通过。可从源码仓库的项目根目录执行，或将完整 `examples/` 目录复制到新的空项目并保留包目录结构。若使用单独构建的 CLI，将 `m-model` 换成其路径，并通过该 CLI 支持的 `--python` 选择环境。

```sh
m-model model inspect --model examples.complex_acceptance.conditional:build_conditional_model
m-model model probe \
  --model examples.complex_acceptance.conditional:build_conditional_model \
  --batch examples.complex_acceptance.conditional:direct_batch
m-model model probe \
  --model examples.complex_acceptance.conditional:build_conditional_model \
  --batch examples.complex_acceptance.conditional:refined_batch
m-model model probe \
  --model examples.complex_acceptance.contract_repair:build_model \
  --batch examples.complex_acceptance.data:changing_batch
m-model try \
  --model examples.complex_acceptance.conditional:build_conditional_model \
  --batch examples.complex_acceptance.conditional:direct_batch \
  --step examples.complex_acceptance.training:multitask_step \
  --steps 1
m-model try \
  --model examples.complex_acceptance.conditional:build_conditional_model \
  --batch examples.complex_acceptance.conditional:direct_batch \
  --step examples.complex_acceptance.training:manual_multitask_step \
  --training-mode manual-step --steps 1
```

如需在复制的工作区里操作，README 开头的 TUI 步骤是另一条人工路径。若后端支持的源码编辑动作未能定位或接管自定义构造，保留手动编辑后重新 Probe 的路径，并记录具体模块、诊断与结果；不要把图声明或命令成功提交当成真实执行证据。

## 许可和运行材料

该示例由仓库本地 Python 源码组成，没有引入外部模型、数据集或复制的第三方代码片段。仓库根目录提供 MIT License；本目录没有独立逐文件 SPDX 声明，重新分发时应连同仓库许可文本和版权声明检查处理。运行依赖是所选解释器中的 PyTorch；仓库未为示例固定 PyTorch 版本，也未声明这里的 `torch` 依赖等于所有操作系统/设备上的完整兼容环境。CPU/GPU、驱动、Python/PyTorch 版本及其他本地库能力均须从用户选择的环境和实际运行解析，不在示例中硬编码绑定。

## 当前证据状态

- 源码静态检查确认了上述入口、共享模块两次调用的代码路径、条件分支、故意不匹配的维度，以及两种训练步骤的代码结构。
- 2026-11-24 在 Linux x86_64、Python 3.14.10、PyTorch 2.15.0 的独立副本中，工作区以两个输入完成前向：共享投影均观察到两次调用，`refinement` 只在第二次观察中出现；见 [多输入观察记录](../../docs/acceptance/multi-input-comparison.md)。另使用 CPU 和 `--no-supervision` 执行本样例：受管步骤完成 1 步，手动步骤完成 4 步，后者报告 `encoders` 和 `fusion_heads` 两个优化器组。运行回执暂存于 `/tmp/model-maven-complex-review-8x2x2gr3`，临时目录可能被清理；这不证明收敛、持久恢复或其他设备支持。
- 契约错误样例的具体失败与修复范围另见 [已有模型接入](../../docs/existing-project-adaptation.md) 和 [本地工作区走查](../../docs/acceptance/local-workspace-walkthrough.md)；仍需按实际模型判断源码编辑可用性。
- 外部用户模型与授权证据未提供，记为 `blocked_external`；不阻塞上述本地探针和覆盖地图。
- 设备差异、真实多模态数据、生产规模模型、动态分支全路径、泛化输入、性能与训练效果仍未观测。
