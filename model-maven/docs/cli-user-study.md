# CLI 配对观察协议与记录表

**状态：只有研究材料，尚未招募或观察参与者。** 本文不是结果报告。自动化 fixture、开发者自测或演示都不能填成真实用户证据。

## 研究问题

对已经维护复杂 PyTorch 模型的工程师，Model-Maven 是否能在不先接管训练循环的前提下，减少首次有效前向诊断和局部修复的负担？哪些输入形式、模型结构或项目环境会让原有方法更快或更可靠？

研究以 `product.json#/value_validation`、`/quality_and_acceptance/usability_targets`、S4、S3、A6、V0 和 V1 为依据。首位参与者用于观察原有方法并建立比较基线；单人的结果只提供定性信号，不外推为统计结论或商业价值证明。

## 参与者与项目

- 邀请已有复杂 PyTorch 模型、实际批次和近期调试任务的开发者；分类预设成功不能替代这类参与者。
- 参与者自己选择一个近期会做的模型检查/错误定位任务。优先使用原有项目与真实输入；不为了让工具“看起来有用”而注入人工错误或削弱原有脚本、hook、IDE、文档和助手。
- 进入前记录 Python/PyTorch 版本、硬件、数据原格式、已有依赖和可复用代码。记录模型/数据是否能离线使用；必要时由参与者在本机操作，不复制专有源码、样本、密钥或客户数据。
- 参与者可跳过问题、停止会话或撤回记录。录屏和屏幕截图需单独征得同意；默认只记匿名化的事件、计时和代码变更量。

## 任务流程

### 阶段一：原方法基线

让参与者使用平时采用的方法完成其选定任务。观察者不预先要求改成 Model-Maven 适配格式，不移除其平时可用的工具。记录从开始准备到参与者能正确解释模型行为或局部错误的时间、帮助次数、修改和未完成原因。

### 阶段二：Model-Maven 首次诊断

用同一项目或难度相当的真实项目，先只给根模型工厂与需要时的代表批次。参与者依自己的理解完成 inspect/probe、找到并解释目标调用或错误、进行合适的局部修改，再次前向。

首次有效诊断前，不要求 loss、optimizer、trainer 或持久化项目配置。记录额外入口、文件、代码行、环境整理、重复输入，以及是否必须离开 CLI 才能继续。对于没有已知错误的项目，任务可以是解释一个真实 forward 的调用边界；不得伪称发现了错误。

### 阶段三：复杂迭代扩展

只有参与者愿意继续时，才在同一项目逐步尝试 Layer 查找/增删改、变体重载、训练步骤试跑和恢复。每一步独立记录前置条件、候选验证、持久化结果、权重影响、撤销/恢复和退出工具的情况。首次收益与训练接管是不同里程碑。

首位参与者先观察原方法再使用工具。之后的配对会话使用难度相近的任务并交错方法顺序，记录顺序和熟悉效应。两种方法的设备、模型权重初始化、数据和可用帮助尽量一致；无法一致时据实说明。

## 观察者规则与完成判定

- 先让参与者自己操作并说出其预期；不先讲解产品术语或引导点击。卡住后按预先约定的提示级别逐步帮助，并记录每次提示内容。
- 只有参与者完成目标，并能用自己的话说明观测范围、变更来源及保存状态，才算任务成功。命令退出码为 0、看到菜单或生成文件都不单独算成功。
- 对 probe，核对报告中的调用/输入输出是否与真实模型批次相符；对 Layer 编辑，重载模型并检查目标结构；对训练，区分试跑与正式训练/恢复能力。
- 记录旧方法更快、工具诊断错误、需要不接受的代码改造、误以为未覆盖分支已验证、丢失选择、取消不符合预期和放弃等负面结果。
- 不把流程中止、环境准备失败、无效批次或缺失依赖删除出分母。标记为失败、未完成、不适用或外部阻塞，并保留原因。

## 逐项记录指标

每项记录起止事件、秒数或计数，并说明如何观测。未测量填 `未测`，不要估数。

| 维度 | 记录内容 |
| --- | --- |
| 准备与迁移 | 依赖/环境/data 准备时间、迁移时间、冷启动与缓存条件 |
| 首次反馈 | 首次 inspect、首次有效调用/错误证据、首次看到真实样本的时间分别记录 |
| 修改迭代 | 从提出修改到再次有效 probe 的时间、重复运行/输入次数、等待和重算范围 |
| 适配负担 | 新增/修改入口、文件、代码行、手工配置项及必须理解的新概念 |
| 错误与恢复 | 错误定位、修复和恢复时间；提示次数、误取消、是否丢失状态 |
| 任务完成 | 原方法与产品路径的正确性、完成/未完成/不适用、离开工具次数 |
| 理解程度 | 能否解释本次配置、调用边界、未覆盖分支、变体/源码差别和试跑限制 |
| Layer 编辑 | 查找/增/改/删、预览、提交、重载、撤销成本及权重后果理解 |
| 后续训练 | 是否能继续使用原步骤；何时需要试跑、正式训练、状态钩子或恢复 |

## 空白会话记录模板

每次会话复制本节，保存一份独立记录。只使用参与者代号，不记录姓名或联系方式。提交前删除专有路径、模块名、数据标签和其他可识别信息。

```yaml
record_status: template_only
record_id:
date_utc:
observer:
participant_code:
consent_confirmed: false
recording_consent: false

participant_context:
  experience_summary:
  model_task_type:
  python_version:
  pytorch_version:
  hardware:
  data_format_and_location_class: # local/remote; never put credentials or raw paths here
  existing_tools:
  reusable_code_or_templates:
  known_inapplicable_capabilities:

comparison_conditions:
  project_or_fixture: # anonymized description
  model_and_initialization:
  input_or_batch:
  environment_and_cache:
  help_available:
  order:
  differences_that_limit_comparison:

task_events:
  - method: existing_workflow # existing_workflow | model_maven
    task:
    start_event:
    first_inspect_seconds:
    first_valid_diagnostic_seconds:
    first_real_sample_seconds:
    repair_and_repeat_seconds:
    total_seconds:
    added_files:
    added_or_changed_lines:
    repeated_inputs:
    observer_prompts:
    help_or_tool_exits:
    mistaken_or_cancelled_actions:
    completed: # true | false | not_applicable | unknown
    correctness_check:
    participant_explanation:
    failure_or_limitation:

layer_or_training_follow_up:
  attempted:
  operations:
  reload_or_resume_verified:
  weight_or_state_effect_explained:
  extra_files_or_code:
  elapsed_seconds:
  outcome:

summary:
  participant_preference_and_reason:
  where_existing_workflow_was_faster:
  where_model_maven_helped_or_hurt:
  comprehension_or_correctness_regressions:
  unmet_user_need:
  confidence: # low | medium | high, with reason
  follow_up_changes_to_test:
  anonymization_reviewed: false
```

## 汇总与决策

每位参与者单独报告两种方法的条件、步骤耗时、迁移成本、代码量、正确性和限制。配对汇总只在条件可比时计算差异；样本量不足时保留逐人结果和不确定性，不报告显著性或普遍结论。

若原方法更快、适配成本抵消收益、参与者不理解证据边界，或复杂模型无法在产品内继续修改，应记录为反证并据此简化交互/适配。不得通过缩小复杂模型范围来换取较好的预设指标。

收集到真实参与者记录前，`value_validation`、A6 与 QA-04 必须继续保持未完成。本模板本身不构成研究结果。
