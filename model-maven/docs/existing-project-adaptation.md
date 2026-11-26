# 接入已有模型：接口与运行环境

本页整理当前代码提供的接缝，供将已有项目接入统一工作区时选择；它不是对任意项目、框架或硬件的兼容性认证。部分样例的本地走查和未覆盖范围见[统一工作区本地走查](acceptance/local-workspace-walkthrough.md)。

## 先选择最小接入方式

| 项目现状 | 工作区中要提供什么 | 已有代码和边界 |
| --- | --- | --- |
| 有可导入的 PyTorch 模型和代表输入 | 模型工厂、输入工厂 | 工厂用 `module:callable` 标识。可以先静态查找候选，再由用户明确选择；发现是有限扫描，会报告省略目录和截断，支持优先指定项目模块目录。真正导入、构造与前向会执行项目代码。 |
| 模型创建依赖参数或组件 | 工厂参数和可选的具名接缝 | 参数是 JSON 值；模型工厂须能接受接缝对应的关键字（也可通过 `**kwargs`），组件需出现在返回模型的注册模块树中。若运行时无法获取工厂签名，由实际调用判断能否接受参数；调用可能执行用户代码。保留 Python 中的实际算法。 |
| 有自己的单步 loss | `managed-step` 的步骤函数 | 步骤接收 `(model, batch, context)` 并返回包含可微标量 `loss` 的字典；引擎负责受支持的更新流程。复杂优化规则不应强行塞进这个模式。 |
| 多优化器或分阶段更新 | `manual-step` 的步骤函数 | 用户负责更新，使用受支持的 context 接口登记优化器与反向；查看具体模式限制与报告，不能假设任意优化器都受管。 |
| 有完整训练循环 | `external-loop` 入口 | 用户保留采样、前向和更新，循环按边界报告进度并遵守停止请求；恢复需要正确提供状态钩子或明确接受状态重置。当前 worker 即使无模型也要求所选 Python 可导入 PyTorch；其他库仍由用户自行选择和导入，工具不自动安装或绑定。工具不证明循环确实完成了每一步更新。 |
| 只有启动脚本，不能安全导入工厂 | 在源码 buffer 中编写小适配器，或继续使用原有启动方式 | 不会自动执行任意脚本来猜测模型；分布式启动、分离机器执行、任意容器编排及不符合现有接缝的框架不因可导入而自动受支持。 |

可许可的代表性仓库样例清单如下。它们是合成夹具和文档示例，能说明当前接缝覆盖的项目形态；不是外部真实用户项目、性能或生产验证：

| 项目样例 | 代表形态与可检查入口 |
| --- | --- |
| [分类预设适配器](../examples/presets/README.md) | 将适配器文件复制到项目根，调用 `classification:tiny_cnn`、`classification:batch`、`classification:step`，使用 managed-step。 |
| [多模态工作区样例](../examples/complex_acceptance/README.md) | 多输入、共享模块、显式条件分支；入口为 `examples.complex_acceptance.conditional:build_conditional_model`、`direct_batch` 和 `examples.complex_acceptance.training:multitask_step`。 |
| [完整训练循环样例](../examples/external_loop.py) | `build_model` 加 `run_loop`；演示用户自管数据与更新，以及循环的 `state_dict`/`load_state_dict` 恢复钩子。 |
| [Gymnasium 用户循环](../examples/gymnasium_loop.py) | 用户代码自行导入 Gymnasium、创建环境和报告状态；可通过无框架循环入口运行，不要求安装 PyTorch。 |
| `src` 布局和外部工作目录 | 本地独立样例将工厂放在项目 `src`，输入从所选工作目录读取相对路径文件；需明确配置导入目录与已有工作目录。 |

本地走查已覆盖多模态样例前向与 Layer 修改、`src` 布局及用户组件导入、无 PyTorch 环境中的 Gymnasium 循环和恢复，以及用户完整训练循环；具体版本、操作与限制见[走查记录](acceptance/local-workspace-walkthrough.md)。分类预设、所有虚拟环境配置和任意用户仓库未因此得到验证。

| 环境或启动方式 | 最小接入要求 | 当前边界 |
| --- | --- | --- |
| 虚拟环境或 Conda | 选择该环境已有的 Python，并由用户预装项目依赖；必要时指定项目内模块搜索目录。 | 工具不会自动安装、更新或绑定框架；可导入不等于具备专用适配能力。 |
| 单个容器或远程主机 | 在该执行环境内运行 Neovim、CLI 与所选 Python，选择该环境可见的项目路径。 | 不提供跨主机路径映射或自动创建容器；不同容器或远程断线后的恢复未由本地样例证明。 |
| 依赖 shell 脚本的训练入口 | 用户在自己的环境管理器中准备环境，或将所需模型、输入及训练代码包装成可导入工厂。 | 不自动执行脚本、下载权重或安装依赖；无法适配时保留原有启动流程。 |
| 多机/分布式编排 | 用户保留自己现有的启动和监控方式。 | 不声称内置分布式调度或跨主机恢复；普通单设备试跑不能证明分布式支持。 |

## 在同一个工作区接入

1. 在项目目录打开工作区。按 `e` 选择现有 Python；如需跨会话保留，可在同一菜单预览并保存项目选择。按 `E` 检查环境，或主动查询 PyTorch 当前可见的训练设备（CLI 为 `m-model env devices`）。试跑和训练可选本会话观察到的设备、自动选择或手动输入；上下文变化后不再提供旧观察。导入与设备查询可能执行库代码并初始化驱动；查询不会安装库或自动选择设备，设备可见性可能随后变化。
2. 如源码位于项目内子目录，按 `K` 逐个添加或移除模块搜索目录；也可一次输入按优先级排列的项目相对目录。如果项目代码依赖其他当前目录，也可在这里选择已有工作目录。受支持操作读取会话选择或项目默认值，具体约束见[运行时契约](working-directory-contract.md)。
3. 按 `F` 静态查找入口名称；候选不是已验证的模型。没有合适入口时按 `N` 在原生源码 buffer 编写适配器，用 `:write` 保存。按 `a` 接入模型及代表输入并审阅配置。
4. 先检查再主动发起前向；报告只说明实际使用的输入和已观察路径。改变源码、输入或环境后重新验证。试跑和正式训练都可选择已有的三种模式，并填写步骤或循环参数；受管和手动模式可选学习率及权重衰减，留空交给后端默认值。手动步骤中的优化器可自行覆盖这些默认值。确认训练模式及其副作用后再运行；不需要预先配置监督规则。

## 保留自己的训练与实验记录

已有训练框架或实验记录工具时，可继续在用户代码中创建和调用它们，无须先创建监督规则。需要工作区管理单步预算、停止请求和进度时，优先把原有循环接到上表的 `external-loop`；不需要工具构造 PyTorch 模型时，选择无框架用户循环。以下是无框架循环的**适配形状**，`my_project.training` 和 `my_project.tracking` 是用户需替换的已有模块，不是本产品提供的接口：

```python
def run(context, *, experiment_name):
    # 这些导入由所选 Python 在运行时解析，依赖由用户安装和管理。
    from my_project.training import create_trainer
    from my_project.tracking import open_experiment

    saved = context.resume_state or {}
    trainer = create_trainer(resume=saved.get("trainer_state"))
    tracker = open_experiment(experiment_name, resume=saved.get("experiment_id"))
    try:
        while context.remaining_steps and not context.stop_requested:
            metrics = trainer.step()  # 返回字符串名称到有限数字的字典
            tracker.log(metrics)      # 用户自己的实验记录；可能产生网络或文件写入
            # 仅保存用户明确提供的 JSON 状态；不要放密钥或原始样本。
            state = {"trainer_state": trainer.json_state(), "experiment_id": tracker.id}
            if not context.report_step(metrics, state=state):
                break
    finally:
        tracker.close()
```

在工作区按 `y` 或 `T` 选择无框架用户循环，选择 `loop(context, **params)` 签名并填写该入口，例如 `my_project.mm_loop:run`，再填写参数并审阅执行确认。具体参数必须是 JSON；记录工具的认证、网络、目录、离线策略和密钥由用户自己的环境管理，不应放进循环参数或状态。循环实际执行时才导入相关库，Model-Maven 不会安装或自动绑定任何实验平台。

仓库的 [TensorBoard 用户循环样例](../examples/tensorboard_loop.py) 展示了实际写入本地事件文件：将该文件复制到项目中，选择用户已安装 TensorBoard 的 Python，入口填写 `tensorboard_loop:run`，参数填写 `{"log_dir": "/用户选择的可写目录"}`。正式运行后可用自己的 TensorBoard 工具查看该目录；样例指标是合成数值，实际训练须替换循环主体。它只保存游标，不保存模型或优化器；试跑也会写事件文件，但不持久保存循环状态。已在独立 Python 3.14、TensorBoard 2.21.0 环境执行两步并读取两个 scalar 事件；这不代表其他版本、远程跟踪服务或真实训练已验证。

`trainer.json_state()` 必须由用户实现并返回受支持的 JSON 状态；指标也须转成有限数字，不能直接传 PyTorch Tensor。示例不会自动保存任意框架的模型、优化器、数据加载器或实验平台状态。如果这些状态不适合放在 JSON 中，应由用户代码负责其持久化，并在状态中只记录可恢复的引用；从不可信位置加载状态的风险也由该实现决定。`report_step` 在正式运行中保存循环明确提供的状态和进度；试跑不持久保存状态。它不能证明第三方服务已经接收到记录，也不能把该样例当作任意实验平台或分布式训练的兼容性保证。完整 PyTorch 循环与工具监督联动时，还须按[工作区监督边界](workspace-journey.md)实现受支持的 context 协议；否则可不启用监督并继续使用原训练方式。

## CLI 最小接入路径

CLI 接受 Python 的 `module:callable` 入口。`module` 是可导入的模块名，`callable` 是模块中的可调用对象；这里不是文件路径，也不能写任意 Python 表达式。典型布局如下：

```text
my-project/
├── <selected-python-interpreter>
├── my_project/
│   ├── __init__.py
│   └── mm_adapter.py
└── data/                 # 用户自己的数据，可按工作目录规则访问
```

这里的 `my-project`、`my_project` 和文件名仅用于说明根目录可导入的形态；按用户仓库中真实的包名与文件路径替换。将适配器放进原项目已有的可导入包，只提供工作区所需的窄接口：

```python
import torch
from torch import nn
import torch.nn.functional as F


class ExistingModel(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.linear = nn.Linear(width, 2)

    def forward(self, features):
        return self.linear(features)


def build_model(width=16):
    # 在这里调用现有构造逻辑；需要时加载权重。
    return ExistingModel(width=width)


def build_batch():
    # 换成项目自己的单批数据读取；返回值直接交给 step。
    features = torch.randn(8, 16)
    labels = torch.zeros(8, dtype=torch.long)
    return features, labels


def training_step(model, batch, context):
    features, labels = batch
    loss = F.cross_entropy(model(features), labels)
    return {"loss": loss}
```

在项目根目录执行。下面用变量表示用户选定的项目路径与解释器，避免假定固定虚拟环境目录。以下是有限单步试跑和最小正式训练命令；`--python` 显式指定本次 CLI 使用的解释器，`--model-param`、`--step-param`、`--batch-param` 的值是 JSON：

```sh
PROJECT_ROOT=/absolute/path/to/project
PYTHON=/absolute/path/to/environment/python
cd "$PROJECT_ROOT"
m-model --python "$PYTHON" init \
  --model my_project.mm_adapter:build_model \
  --batch my_project.mm_adapter:build_batch

m-model --python "$PYTHON" try \
  --training-mode managed-step \
  --model my_project.mm_adapter:build_model \
  --batch my_project.mm_adapter:build_batch \
  --step my_project.mm_adapter:training_step \
  --steps 1 --device cpu

m-model --python "$PYTHON" train \
  --training-mode managed-step \
  --model my_project.mm_adapter:build_model \
  --batch my_project.mm_adapter:build_batch \
  --step my_project.mm_adapter:training_step \
  --steps 10 --checkpoint-every 5 --device cpu
```

`init` 只保存模型、批次及装配配置；训练步骤需在 `try`/`train` 命令传入。managed-step 要求每步批次工厂返回单批对象，步骤返回含有依赖模型计算图的浮点标量 Tensor 的 `loss` 字典；默认使用运行器的 AdamW、反向传播和更新。也可显式提供 `--optimizer-factory module:callable` 和 `--optimizer-param NAME=JSON`；工厂接收模型及这些参数，并须返回使用当前模型可训练参数的 `torch.optim.Optimizer`。模型运行使用所选解释器，需在该环境预先安装 PyTorch 和项目本身依赖；工具不会替项目安装依赖。实际训练前应先审阅工厂、步骤及其导入模块，因为导入和调用都可能执行项目代码并产生副作用。若参数中有 token、密码或其他凭据，不要传入 `*-param`：参数会进入本地项目配置、运行记录或检查点。请在启动 CLI 前通过现有秘密管理器或进程环境提供环境变量，并由适配器按需读取；Model-Maven 不为这些变量提供项目级秘密存储或脱敏快照。

若模型源文件不在项目根可导入的位置，例如位于源码子目录，使用 `m-model discover --import-root <PROJECT_RELATIVE_DIR>` 作只读候选发现；发现不会导入项目代码。需要运行时加载该目录时，先通过 `m-model config set-runtime-paths --import-root <PROJECT_RELATIVE_DIR>` 预览路径，再按回执 revision 使用 `--apply --if-current <revision>` 保存。也可在同一配置命令中用 `--working-directory <EXISTING_ABSOLUTE_DIRECTORY>` 指定用户代码解析相对路径时的运行目录；省略则使用项目目录。项目根仍用于配置、源码和运行记录。解释器、工作目录和导入目录是彼此独立的选择。配置命令只保存用户明确选择，不创建目录、不安装依赖；相对解释器路径在项目上下文解析。若在多个导入目录中有同名模块，按配置顺序解析，须检查实际导入来源。

接入自有完整 PyTorch 训练循环时，可用 `--training-mode external-loop --loop module:run_loop`。入口按 `run_loop(model, context, **params)` 调用；显式 `--model` 优先，否则使用项目已配置的模型和组件接缝；项目未配置模型时传入 `None`。已配置模型的项目也可以在新运行时指定 `--no-model`，显式向循环传入 `None`；此选项不能与模型、模型参数、组件注入、变体或模型初始化同时使用，重复和续跑沿用历史选择。循环自行取数和更新，并通过 context 报告已完成步骤；可恢复续训还需正确实现循环对象的 `state_dict`/`load_state_dict`，否则恢复会被拒绝，除非显式允许状态重置。循环模式不接受单独 `--batch`，也不使用 CLI 的 `--lr`/`--weight-decay`。它仍属于模型运行，所选 Python 环境需要可导入 PyTorch。另有 `loop-trial`/`loop-run` 是独立的纯 Python 循环接口，可在无模型情形下运行，但只记录用户报告的进度与状态，不提供模型构造或训练更新管理；两种循环入口不要互换。

### 启动与环境选择

- 未显式传入 `--python` 时，新操作优先使用项目配置保存的 Python，然后是项目环境选择或 `.venv`，最后按当前 `PATH` 搜索；找不到或选中的解释器失效时应修正选择，不会静默切换。CLI 与 worker 继承启动它们的进程环境。需要 CUDA、代理、数据凭据等变量时，应在启动命令的 shell/环境管理器中设置；运行记录和环境清单不是完整环境快照。
- 如需持久化用户选择，可用 `m-model config set-python --executable /path/to/python` 预览，然后按回执中的 revision 使用 `--apply --if-current <revision>` 保存。它只保存路径/命令，不验证解释器、不安装依赖。命令选项及项目配置是本地数据，不要将凭据写入其中。
- `requirements.txt`、`pyproject.toml` 锁文件、conda 环境定义、shell 安装脚本和 Dockerfile 都由项目所有者按原项目流程处理。先按该项目的受支持方式准备环境，再把其中实际可用的 Python 解释器作为 `--python` 或项目选择；Model-Maven 不读取这些文件来推断安装步骤，也不会自动运行安装脚本。若依赖只在容器内可用，用户须自行安排容器和在该环境中启动工作区/CLI；当前没有代为启动容器或编排容器任务的通用接缝。
- 从自定义工作目录运行时，建议将其保存到项目 runtime paths，而不是误把它当成项目根目录。用户代码可能自行改变当前目录；工作目录选择不构成进程隔离，也不能限制代码读写权限。
- 不要通过设置 `PYTHONPATH`、全局导入钩子或环境变量来假定 CLI 的项目导入路径；使用项目相对的 `--import-root`/runtime paths。程序启动继承的环境仍可能影响 Python 本身及第三方库行为。

当前边界：入口发现只静态扫描常见 Python 声明，不导入代码，也无法推断任意启动脚本的参数语义；交互式 notebook 内核、必须由 shell 预处理/多进程协调器启动的训练、分布式作业、多机执行、服务托管和任意容器编排没有通用自动接入协议。对这些项目，应沿用原启动器，或在同一源码工作区增加上述工厂/步骤/循环适配层，并由用户明确安排原有资源与生命周期。工具不会为了猜测入口自动运行任意脚本、安装依赖或绑定设备。

## 第二种形态：源码树项目与自有更新步骤

有些仓库把可导入包放在源码子目录，且原训练代码自行执行反向传播和优化器更新。可以在该源码树中添加一个很薄的适配模块，包装项目原有 API；以下 `src/legacy_pkg`、导入名和解释器变量都只是示意，须按实际项目布局替换，不是 Model-Maven 固定目录或包名。命令中的 `src`、工作目录及绝对路径也都要改成用户选择的现有目录；不存在统一要求的目录名：

```text
<project-root>/
├── src/<actual-package>/
│   ├── model.py
│   ├── data.py
│   └── mm_adapter.py       # 新增的项目内适配层
└── <selected-python-interpreter>
```

`src/<actual-package>/mm_adapter.py` 的适配器可按以下契约组织。把示意导入及 `make_model`、`next_batch`、`loss_for` 替换为项目已有函数；批次工厂返回训练步骤消费的一个批次：

```python
import os

from actual_package.model import make_model
from actual_package.data import next_batch, loss_for


def build_model():
    return make_model()  # 保留项目需要的配置、权重加载等逻辑


def build_batch():
    return next_batch(root=os.environ["DATA_ROOT"])  # 将继承的非秘密路径交给项目数据管线


def training_step(model, batch, context):
    optimizer = context.optimizer("main", model.parameters())
    optimizer.zero_grad()
    loss = loss_for(model, batch)
    context.backward(loss)
    optimizer.step()
    return {"loss": loss}
```

该步骤对应 `manual-step`：`context.optimizer(name, parameters)` 默认登记 AdamW 优化器组，`context.backward(loss)` 执行受管理反向传播，用户步骤调用登记优化器的 `step()` 完成更新。需要其他 PyTorch 优化器时，可显式传入顶层工厂和 JSON 参数，例如 `context.optimizer("main", model.parameters(), factory=make_optimizer, factory_params={"lr": 0.001})`；工厂接收参数迭代器，必须返回 `torch.optim.Optimizer`，并按登记顺序且只包含该组登记的模型参数。一个参数不能同时登记到多个组，登记后不能改变参数组；`step` 闭包不受支持。若原算法需要动态参数组或无法遵守这个窄接口，可保留它在自己的完整 `external-loop` 入口内执行，或沿用原训练启动器。自定义优化器工厂及参数会进入运行与 checkpoint 状态，不能放入秘密。

在实际源码根目录和已存在的环境中指定值；`PYTHON` 应填写所选环境中的解释器路径。先预览并按 revision 保存实际源码目录和工作目录，然后按真实包名运行：

```sh
PROJECT_ROOT=/absolute/path/to/project
PYTHON=/absolute/path/to/environment/python
WORKING_DIRECTORY=/existing/absolute/directory
cd "$PROJECT_ROOT"

# 预览项目内源码搜索目录和用户代码的工作目录；将回执 revision 用于下一条 apply 命令。
m-model config set-runtime-paths \
  --working-directory "$WORKING_DIRECTORY" \
  --import-root src
m-model config set-runtime-paths \
  --working-directory "$WORKING_DIRECTORY" \
  --import-root src --apply --if-current <revision-from-preview>

m-model --python "$PYTHON" try \
  --training-mode manual-step \
  --model actual_package.mm_adapter:build_model \
  --batch actual_package.mm_adapter:build_batch \
  --step actual_package.mm_adapter:training_step \
  --steps 1 --device cpu

m-model --python "$PYTHON" train \
  --training-mode manual-step \
  --model actual_package.mm_adapter:build_model \
  --batch actual_package.mm_adapter:build_batch \
  --step actual_package.mm_adapter:training_step \
  --steps 10 --checkpoint-every 5 --device cpu
```

上例中的绝对路径、`actual_package` 和文件名要替换为用户选择的现有路径与实际 Python 模块名；解释器不会因 `src` 布局而自动推断。工作目录用于用户代码的相对路径访问，项目根仍管理配置、源码及运行记录。路径配置限定为已有目录，导入根必须是项目内目录；所选解释器必须已能导入项目所需库。用户若使用外部数据目录，可把其真实绝对目录作为工作目录或在启动 CLI 前通过其环境管理器提供数据路径变量，不要把凭据放入模型、批次或训练参数。项目模块导入、构造、取数、权重加载以及训练步骤均可能运行任意项目代码，执行前应审阅；工具不安装依赖，不自动调用原始 shell 启动脚本。

适配器可从 CLI 和 worker 继承的进程环境读取用户配置的非秘密参数，例如 `DATA_ROOT`；需要凭据时，从现有秘密管理器注入环境变量，不要把真实值写进脚本、命令历史或配置文件：

启动前由用户自己的 shell/环境管理器设置 `DATA_ROOT`（以及需要的密钥），再执行前述 `m-model --python "$PYTHON" ...` 命令。环境变量会传给 worker，但 Model-Maven 不将它们变成工厂参数，也不提供变量快照、秘密存储或密钥脱敏。

```sh
PROJECT_ROOT=/absolute/path/to/project
PYTHON=/absolute/path/to/environment/python
cd "$PROJECT_ROOT"
export DATA_ROOT=/existing/data/directory
# 需要 DATA_TOKEN 等凭据时，由用户已有的秘密管理器向此进程注入。
m-model --python "$PYTHON" try --training-mode manual-step \
  --model actual_package.mm_adapter:build_model \
  --batch actual_package.mm_adapter:build_batch \
  --step actual_package.mm_adapter:training_step --steps 1
```

例如，用户可在自己的适配模块中包装已有 API：

```python
from my_project.model import MyModel
from my_project.data import make_example


def build_model():
    return MyModel()  # 按真实项目填写构造参数及权重加载


def example_batch():
    example = make_example()  # 替换为项目的代表数据读取逻辑
    return {"args": [example.image], "kwargs": {"mask": example.mask}}
```

`args`/`kwargs` envelope 表示真实前向参数；其他普通字典不会自动当作关键字参数，需明确选择 `kwargs`、`args` 或 `single` 绑定。工厂可以读取项目自己的配置，但读取、下载、写入、初始化设备等副作用由项目代码决定，应在执行前审阅。

如果批次对象本身可迭代，却不应被当作数据集采样，选择 `single-raw`：前向将整对象作为一个位置参数传给模型，跳过自动数据集采样；它与沿用历史自动采样行为的 `single` 不同。工作区可在输入绑定中选择这一项，数据预览会按整个对象处理；CLI 数据预览可使用 `--whole-batch`。受管和手动训练可使用 `--batch-device-policy preserve` 跳过步骤前的自动批次迁移；新运行默认 `recursive-strict`，递归迁移张量且不能保留受支持容器类型时中止。工作区试跑和训练会让用户选择这两项，确认前展示策略。自定义不透明对象不会自动迁移，批次工厂或步骤应自行处理设备；显式增强仍可能在迁移策略之前修改批次。旧运行未记录策略时保留历史递归行为，重复试跑可明确改变，续训不能改变。完整循环由用户管理输入。这些路径尚待人工走查。

工作区的重复试跑和续训会在确认中展示记录的批次设备策略；未记录时区分历史递归行为和用户完整循环。若记录了新策略而所选 CLI 未声明支持，工作区不启动操作。严格递归迁移保留受支持容器及 Tensor 的重复对象引用，对自定义映射、循环引用或过深/过大的结构中止；可检查到不同 Tensor 共享存储且需要迁移时也中止，避免静默破坏视图。不可检查的存储关系仍未知，字典键和不透明对象不会迁移。需要原对象语义时应选择 `preserve` 并由用户代码处理设备。这些行为尚待人工验证。

数据集工厂参数中显式填写的 `seed` 会原样传给工厂，由工厂决定是否接受；工具自动生成的采样种子仅在可检查的签名支持该关键字时自动传入。直接从批次工厂得到的数据集实例不会声称工具已将种子传给其工厂。新试跑或训练未指定种子时会生成并记录种子，用户也可显式指定；种子不证明工厂或外部数据源完全确定，相关实现尚待人工验证。

数据集预览中的切分是用户请求，不自动证明数据来源。报告区分从工厂返回字典按键选取、将切分作为参数传入工厂、以及无法核对；传参也不能证明工厂实际采用该切分。若工厂参数已明确包含与所选切分不同的 `split`，工具会在调用工厂前报冲突；导入工厂模块本身仍可能有副作用。直接获得的数据集实例或自定义适配器的切分处理可能未知。此行为尚待人工验证。

没有切分概念的数据源可在工作区数据集预览中将切分留空，或在 CLI 使用 `data preview --dataset module:factory --no-split`；此时默认不会自动套用项目的 train/eval 增强。也可显式选择 `--augment-split train|eval|none`，与数据切分独立；该选项不能与 `--transforms` 文件同时使用。未明确选择增强时保留旧规则：train 切分使用训练增强，其他非空切分使用评估增强。CLI 未提供 `--no-split` 时保留原有默认 train 行为；工作区会检查 CLI 能力，旧版不可用时提示更新或选择兼容选项。该路径尚待人工验证。

预览调用工厂也不能证明得到全新的独立数据源：工厂可能返回共享对象、访问共享游标或消费外部数据流。采样独立性因此报告为未知，预览可能影响用户代码或外部服务管理的状态；应由数据源作者明确处理。

## 环境和数据边界

工作区和 CLI 子进程继承启动它们的环境；项目选择的解释器、运行目录与模块搜索目录不是容器隔离。对于需要环境变量的项目，在启动 Neovim 或 CLI 之前用现有环境管理工具设置；当前没有项目级秘密管理器或可恢复的环境变量快照。不要把令牌写入 JSON 工厂参数、项目配置、示例源码或运行记录。训练快照、环境清单及导出均不能保证固定外部包、动态导入、数据、权重或驱动。历史运行优先使用记录的解释器与路径，失效时应先修复或明确处理冲突，不能理解成已经完整复现环境。
