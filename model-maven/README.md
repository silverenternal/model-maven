# Model-Maven

**The Build System for Deep Learning Models**

Model-Maven is an early engineering preview for building and inspecting models in a Neovim workspace backed by a Rust CLI. You can start from your own Python model, inspect a real batch, edit the source or supported Layers, and run supported training workflows. Python, model libraries and the runtime environment remain your choice.

The currently published npm `0.1.0` predates the unified workspace described below. The current workspace is available from source or as a locally built private Linux x64 preview; it is not yet a new public npm release. See the [release compatibility notes](docs/release/compatibility.md) for the actual scope.

The [shared roadmap](todo.json) tracks the CLI backend and its companion TUI, including new model creation and integration milestones. Use `m-model doctor` for the capabilities of your installed binary.

For a task-oriented walkthrough, start with the [CLI guide](docs/cli-guide.md). The [user-study protocol](docs/cli-user-study.md) is preparation material only; no real-participant comparison has been completed.

## Start the unified workspace from source

You need Neovim 0.10 or newer, Rust 1.88 or newer, the companion [model-maven-tui](https://github.com/silverenternal/model-maven-tui) checkout, and an existing Python environment containing your project's dependencies. For PyTorch model operations that environment needs a compatible PyTorch installation. No framework is installed or selected automatically.

Build the two checkouts and start the workspace in your model project, replacing the paths with your own:

```sh
cargo build --release --locked --manifest-path /path/to/model-maven/Cargo.toml
cargo build --release --locked --manifest-path /path/to/model-maven-tui/Cargo.toml
cd /path/to/your/model-project
/path/to/model-maven-tui/target/release/m-model-tui \
  --cli /path/to/model-maven/target/release/m-model --project "$PWD"
```

For a new model, open an empty project and choose **Open graph canvas**. Start with a single stream, two-input merge or residual scaffold, or add individual nodes. Select a connection to insert an operator, choose a factory from the selected runtime or your own project, and edit its parameters and ports. Drag from an output to an input to connect branches; move a block onto another block to review a supported sequential reorder. Clicking a block opens its details. `Tab`, arrow keys and `Enter` provide keyboard navigation; `n` follows the suggested next action and `m` lists graph actions.

Choose an input or output node to connect a CSV batch, class-folder images or a custom batch factory. For training, select an output and declare its target and loss. Save the graph, review and generate a new Python module, then review its selection as the project model and apply the declared data. These are separate confirmations; an existing source file is retained. Preview a real batch and run a forward pass before preparing a trial. The graph planner checks declared structure and selected data mappings, but only execution can reveal actual import, tensor and runtime behavior. The graph and source workflows described here have not yet completed release acceptance.

The alternate sequential builder remains available for supported layer stacks and saves a `model-maven.json` declaration. Its example models and synthetic input are optional starting points.

For class-folder image datasets, the graph canvas can connect a classification directory as real data. The separate legacy image-classification builder can also create a preset project. Its model choices come from the current CLI and selected runtime; after reviewing the class folders and creation plan, use the result actions to preview a real image batch, run forward and prepare a trial. This path saves an image-classification `model-maven.json` with the preset, data path and ordered class names. Open the image project to save the current training settings, link an exported supervision bundle, or snapshot the saved `classification.py` and link that snapshot. To rebuild, copy the declaration, images and any referenced files into a new project, open it in the TUI, or preview `m-model create --recipe model-maven.json` and apply its plan revision. The data path is resolved in the new project; external absolute paths remain external. Rebuilding checks class order and snapshot integrity, but does not copy or fingerprint image contents, other imported code or the environment. Source snapshots are editable project code: review them and their data paths. Supervision still requires explicit import and rebinding, and does not enable itself during training.

To transfer an image project, choose **Save project runtime paths** to record its project-local working directory and ordered module search directories. Rebuilding resolves those paths in the target project; it does not transfer or select the Python interpreter. Then choose **Package project for transfer**. The declaration and its referenced source snapshot and supervision file are included automatically; choose project-local data, other imported files and environment files explicitly, then review the complete file list before creating a new directory. The picker offers statically discovered source and possible environment files as suggestions; it does not execute imports or prove dependencies are complete. The CLI equivalent is `m-model recipe-bundle --output transfer --include images --include requirements.txt`; preview first and use `--apply --if-plan <revision>` to create it. The package does not discover every import, include external paths, install libraries or set up an environment. Review selected data and credentials before sharing. If no source snapshot is linked, rebuilding uses the preset and excludes edits to `classification.py`.

For a saved explicit graph, choose **Package graph project** in the graph actions. The graph declaration is included; select the model's source, data, supervision and environment files explicitly. Static file suggestions do not establish that every dependency has been found. Review the full file list, including any sensitive files, before creating or sharing the package. The CLI preview is `m-model recipe-bundle --graph --output transfer --include my_model.py`; add `--apply --if-plan <revision>` using the returned revision to create it. The target still needs its own runtime setup and any omitted external resources.

For a saved scaffold declaration, choose **Package project** in the model builder, or preview `m-model recipe-bundle --scaffold --output transfer --include my_model.py`. This includes `model-maven.json` and its referenced supervision file after checking the reference digest. Other source, data and environment files must be selected; the destination still needs the appropriate runtime and supervision rebind. As with graph packages, dynamic imports and external dependencies may be missing.

Graph and scaffold packages contain `model-maven-transfer.json` with project-relative runtime path hints when available. In the target workspace, open runtime paths with `K` and choose **Load migration path hints into this session**. Review available and missing directories before applying them; use **Save current paths to project** separately if they should persist. External working directories and the selected Python interpreter are not copied. File suggestions can include sources observed in earlier forward passes; they may have changed and do not prove dependency coverage.

After connecting a real input, reopen the saved model canvas and choose **Load project input** to include its factory, binding and parameters in `model-maven.json`. Review and save the new build. To rebuild elsewhere, provide the referenced Python factory, data files and environment in the new project, then preview `m-model scaffold --recipe /path/to/model-maven.json --write-recipe` and apply its reported plan revision. Paths in factory parameters remain project-relative; the declaration does not package datasets or install dependencies. Factory parameters are stored in the declaration, so use runtime environment mechanisms for credentials.

The workspace and canvas flag a saved project input that differs from the declaration; the canvas also flags changed training settings selected in the current session. From the input-save or trial result, review and save those settings directly to the declaration without regenerating the model. Alternatively, load them into the canvas and review a new build. A previously captured forward input can be reused for a source-edit preview and its independent recheck; the original capture remains available separately.

From a saved canvas input, **Connect CSV data** opens the CSV setup. Declared input and output names are offered as candidates, and readable ordinary CSV headers can be selected in order; manual names remain available for other adapters or headers. A header is only a local suggestion: preview the input and run the model to check the actual binding.

After changing a starter model's input routes in the assembly workspace, reopen the canvas and choose **Load saved assembly** to bring its component bindings, input routes and encoder output modes into the draft. Review and generate to save them in the declaration as defaults for the new model. Other custom model parameters cannot yet be synchronized by this canvas and are reported instead of silently dropped.

If `model-maven.json` contains a `training` declaration, `m-model try --training-recipe model-maven.json --steps 1` runs a trial with its mode, entry and parameters. `m-model train --training-recipe model-maven.json --steps 100` starts a new training run. Set the Python environment and data independently; the recipe does not install dependencies or enable supervision. Training recipe settings cannot be combined with explicit training mode, entry, step/loop parameters, repeat, resume or migration options. When the declaration includes an optimizer, explicit optimizer options also conflict; remove them to use the declared settings.

In the workspace, the trial and training mode picker places the current project's saved mode (or a still-current session choice) first and labels its source. The matching entry and parameters are presented for review; a different mode remains selectable. For the same model, entry and training mode, the optimizer settings selected in a trial can be reused when preparing training during that session. Changing the project configuration invalidates the previous session defaults. After selecting training settings and an optimizer, use **Load training settings** in the canvas to save them in the declaration; factories and parameters remain visible in the file.

For an existing Python model, connect its model factory and a representative input in the workspace panel. The main buffer is available for editing source and adapters. Choose your Python environment before running a forward observation and review the execution confirmation.

To open the same workspace inside an existing Neovim or LazyVim session, add the companion checkout as a local plugin and run `:ModelMaven`. The [workspace guide](https://github.com/silverenternal/model-maven-tui/blob/main/docs/unified-workspace-guide.md) explains configuration, navigation and exit; the [existing-project guide](docs/existing-project-adaptation.md) covers adapters and runtime paths. If you have built the private packaged preview locally, installation instructions are in its `dist/local-preview/README.md`.

## CLI quick start

Requirements:

- Rust 1.88 or newer (the project uses edition 2024 and let chains).
- An existing Python environment with the dependencies your project needs. `venv`, `pip` and a `requirements.txt` are only needed if you choose the optional environment setup below.
- A Python module in the current project that returns an `nn.Module` and, for a forward probe, a representative batch.

Install the CLI from this repository, then declare dependencies in the model project:

```sh
cargo install --path .
```

For example, the model project's `requirements.txt` can start with:

```text
torch
# Add the packages imported by your model and data code.
```

To use an environment you already manage, select it with `--python /path/to/python` on a command, or configure the project interpreter in the workspace. The following is an **optional** pip-based setup for a project that wants Model-Maven to create and manage a venv:

```sh
m-model env setup
m-model env check
m-model doctor
```

`env setup` resolves wheel packages without installing into the base interpreter, writes `requirements.lock.txt` with exact versions and SHA-256 hashes, creates the selected project venv (default `.venv`), installs from the lock, and runs `pip check`. Declare the packages your project needs in `requirements.txt`; include a compatible PyTorch wheel only when using PyTorch-based model features, and choose indexes and versions appropriate for the machine. Set private indexes and wheelhouse locations in pip config or `PIP_INDEX_URL`/`PIP_FIND_LINKS`, since requirements-file index directives and direct URL dependencies cannot be replayed from the lock. A requirements file may include other project files with `-r` or `-c`, and those inputs are included in the lock freshness check. Source builds, VCS and editable requirements are rejected. The lock does not store package indexes or credentials, vendor wheels, or pin operating-system libraries and accelerator drivers.

Subsequent commands automatically use the environment selected by the last successful setup (default `.venv`). An explicit `--python /path/to/python` always takes precedence. To inspect or update the lock separately, use `m-model env lock` or `m-model env lock --refresh`; setup reuses an unchanged lock and refuses to overwrite an existing incompatible virtual environment. `m-model env check` and a later `m-model env setup` reuse the selected path; `--venv` can override it.

To inspect optional ML/RL packages without changing that environment, use `m-model env libraries`; `m-model env probe --library <id>` explicitly checks importability. Add `--check-runtime` for bounded CPU smoke profiles on selected supported packages. None of these commands install dependencies or bind a Trainer. See [environment adapter boundaries](docs/environment-adapters.md).

To prepare same-ABI offline install materials, use `m-model env wheelhouse archive`, inspect them with `m-model env wheelhouse check`, then create a new environment with `m-model env wheelhouse replay --create --venv .venv-offline`. The destination must not exist; an existing environment is not modified. The new environment is not automatically selected as the project interpreter. The archive uses the active pip source configuration, stores only package names/versions and wheel hashes, and refuses missing, modified, or ABI-incompatible material. It does not include the operating system, accelerator drivers, CUDA/ROCm runtime, or datasets; it is not a cross-machine reproducibility guarantee.

The repository includes a shared-module example:

```sh
m-model init --model examples.models:build_shared --batch examples.models:shared_batch
m-model model inspect
m-model model probe
m-model layer list
```

`init` stores default Python factory entrypoints in `.model-maven/project.json`; it does not rewrite project code or install dependencies. In a terminal, you can run `m-model init` and enter the model factory when prompted, then optionally enter a representative batch factory. `--dry-run` previews the file without creating it. Existing project settings are never overwritten. Later command-line `--model`, `--batch`, or `--batch-binding` values take precedence over saved defaults. These defaults also populate the interactive workbench.

For one-off inspection, skip initialization and pass the entrypoints directly:

```sh
m-model model inspect --model examples.models:build_shared
m-model model probe \
  --model examples.models:build_shared \
  --batch examples.models:shared_batch
m-model model verify \
  --model examples.models:build_classifier \
  --batch examples.models:classification_batch \
  --loss examples.models:classification_loss
m-model try \
  --model examples.models:build_classifier \
  --batch examples.models:classification_batch \
  --step examples.models:classification_training_step \
  --steps 4
```

The probe executes the model's real `forward` on CPU and reports input/output shapes and each observed module invocation. It reports metadata rather than copying tensor contents. The shared layer in this example is called twice; each call is shown separately. If a forward fails, the JSON report includes the last observed module call, its input metadata, and project source frames; the human CLI prints the module path and source location to help you find the contract boundary. The call list is bounded; when it is truncated, the failure location still refers to the most recently observed call.

To give a model block a human-defined meaning without adding hooks or editing `forward`, bind a label to a port seen by Probe:

```sh
m-model model mark \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch \
  --path model.encoder.projection \
  --port outputs.features \
  --role embedding \
  --meaning "Representation consumed by the retrieval head"
```

This reruns a real forward, rejects ports that were not observed, and stores project source/configuration fingerprints in `.model-maven/semantic-marks.json`. The role is always authored by the user. Use `m-model model marks` to list marks and inspect source staleness, or add `--model` and `--batch` to re-probe current module/port/config bindings. `m-model model retag --id ...` explicitly updates a mark after review; `m-model model unmark --id ...` deletes that exact ID. A mark is descriptive until you add a rule; rules stay off unless training explicitly receives `--apply-supervision`, and only implemented rule/mode combinations run. See [model port contracts and semantic marks](docs/model-contracts.md) for coverage limits.

To reuse supervision in another project, the workspace offers **Export supervision** and **Import supervision** in its supervision actions. The export contains the authored roles, rules, parameters and source identities, so choose a destination appropriate for that information. Import previews the marks and verifies both the file hash and the target project's semantic-state revision again when applying. Imported marks are deliberately unbound: run a fresh forward and rebind each mark to an observed port before using its rules. For custom rules, the matching source must also be available and pass its normal identity checks in the target environment. Importing does not install dependencies or execute custom rules.

An existing canvas can link an exported supervision file in `model-maven.json` using a project-relative path and SHA-256. Copy that file along with the declaration when rebuilding elsewhere, then use **Import referenced supervision** and rebind its marks. The reference is not an active rule set and does not automatically enable training supervision; importing verifies the referenced file has not changed.

After a current forward, choose **Inspect a port and its rules** or select an observed tensor-link endpoint to see that port's marks and rules, add a rule, or launch a new supervision preview. The port view can also search up to eight recent non-running run records for matching marks. Earlier results describe their original execution and may use different data or settings; they are not evidence about the current input.

The complex acceptance fixture also demonstrates a local contract repair: a 48-feature video encoder reaches a fusion projection configured for 32 features. The first command is expected to report the mismatch; the Layer edit creates a persistent `working` variant, validates a real batch, and the final command reloads it in a new process:

```sh
m-model model probe \
  --model examples.complex_acceptance.contract_repair:build_model \
  --batch examples.complex_acceptance.data:changing_batch
m-model layer set model.fusion.video_projection \
  --model examples.complex_acceptance.contract_repair:build_model \
  --batch examples.complex_acceptance.data:changing_batch \
  --param in_features=48
m-model --variant working model probe \
  --model examples.complex_acceptance.contract_repair:build_model \
  --batch examples.complex_acceptance.data:changing_batch
```

Run the full offline complex-model acceptance workflow with a built CLI and a PyTorch environment:

```sh
cargo build
python3 scripts/verify_complex_acceptance.py --python python3
```

It checks shared-module calls, dynamic video/text lengths, three named losses, the 48→32 contract diagnosis and local variant repair, then stage-aware two-optimizer training. It compares a two-stage checkpoint resume with an uninterrupted run, verifies the batch cursor and frozen/unfrozen phase statistics, and restores the exported variant and trained weights. This is a small CPU functional fixture, not a performance result or user-value study.

Exercise complete user-owned loops, explicit state hooks, Linux background stop, the interactive TTY path, and accurate capability reporting with:

```sh
python3 scripts/verify_external_loop.py --python python3
```

## Work with an existing model

Expose a model factory and a batch factory in an importable Python module. A model factory can be zero-argument, or accept explicit component seams as keyword arguments. A tensor batch is passed as one positional argument. For structured inputs, return either an explicit envelope or choose a binding:

```python
def example_batch():
    return {
        "args": [video_tensor],
        "kwargs": {"tokens": token_tensor, "mask": attention_mask},
    }
```

```sh
m-model model probe \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch
```

An ordinary dictionary is not assumed to mean keyword arguments. Use `--batch-binding kwargs` for a dictionary of keyword arguments, `args` for a tuple/list of positional arguments, or `single` to pass the complete returned object as one argument. Model factory arguments can be supplied as JSON data, for example `--model-param width=256`.

### Compose a model through declared seams

Keep algorithms and internal network logic in Python. Expose only the modules you want to swap as named factory parameters:

```python
from torch import nn

class VisionModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, image):
        return self.head(self.encoder(image))

def build_model(encoder, head):
    return VisionModel(encoder, head)
```

Bind factories to those seams and save them as project defaults:

```sh
m-model init \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch \
  --inject encoder=my_project.encoders:build_resnet \
  --inject head=my_project.heads:build_classifier
m-model model probe
m-model try --step my_project.training:step --steps 4
```

Direct `module:factory` injection factories take no arguments and must return an `nn.Module`; use a small Python wrapper when construction needs parameters. The root factory must explicitly accept each seam name. Before returning a model, Model-Maven checks that every injected module appears in its registered module tree. Inspection shows the resulting paths, and a probe reports whether each seam ran on that sample. A declared but unvisited conditional branch stays visible as `not_observed`.

For reusable components, register a Layer factory and inject it by its exact component version. Parameters are schema-checked and defaults are filled before construction:

```sh
m-model init \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch \
  --inject encoder=component:vision/resnet@1.2.0 \
  --inject-param encoder.depth=50
```

Registered injections save the resolved parameters plus the component manifest hash and transitive component locks. `init`, inspect, probe, try, train, resume, and variant replay use that saved lock. If a locked manifest or dependency changes, construction stops with a component version error. Direct `module:factory` injections remain available for project-specific code; they take no arguments, so use a Python wrapper when that factory needs parameters.

`init` stores the mapping in `.model-maven/project.json`, so probe, try, train, and resume use the same assembly. Passing one or more `--inject` flags to a later command replaces the complete saved mapping for that invocation. New variants pin the factory mapping and local Python source fingerprints. Direct Python factories still follow the selected Python environment and source fingerprints. `assembly show/add/set/remove` manages the project default's explicitly exposed root factory seams. Each candidate must construct successfully and, when a representative batch is configured, pass a forward probe before it can be saved. TTY edits ask for confirmation; `--dry-run` always leaves the project config unchanged. Existing variants retain their original assembly snapshot. `assembly generate` writes a separate readable adapter for the validated root factory and named injection seams. With global `--variant`, it also lowers supported built-in Layer edit histories for `Sequential`, `ModuleList`, and `ModuleDict`, including transaction order, active cursor, saved initialization seeds, and explicit sharing changes. It refuses to overwrite files and does not serialize arbitrary Python graph internals.

早期生成的自定义骨架可尝试在 TUI 的源码编辑中选择 `migrate-starter`，先审阅差异和候选前向，再确认保存。CLI 对应 `m-model layer migrate-starter --target source --dry-run --require-forward`；确认差异后，用返回的计划摘要通过 `--if-source-plan` 条件提交。该迁移只支持可识别的旧生成结构，不会覆盖不确定的自定义接线。它保留原文件中其他代码，保存后重新构造与验证，失败时尝试回滚。既有检查点不会迁移，固定旧源码的变体可能需要重建；修改接线后旧默认预测头的宽度不会自动调整。

```console
m-model assembly show
m-model assembly set encoder --component vision/encoder@1.2.0 --param depth=50 --dry-run
m-model assembly set encoder --component vision/encoder@1.2.0 --param depth=50
m-model assembly add fusion --factory my_model.blocks:build_fusion
m-model assembly remove old_head
m-model assembly generate --output generated/model_assembly.py

# Use the generated entrypoint for a real run, then export that exact run.
m-model train \
  --model generated.model_assembly:build_model \
  --batch my_project.data:example_batch \
  --step my_project.training:step \
  --steps 100
m-model export ../model-export
```

When training imports the generated entrypoint, the run snapshot records the adapter and its imported project modules. Export then copies and hashes those files, so `restore_model.py` can reconstruct the same assembly outside the project. The export still expects compatible Python packages and runtime libraries to be available; it does not vendor arbitrary dependencies.

An interactive terminal can prompt for a missing add/set binding and collect registered component parameters from its Schema. Use `--json`, `--model-param NAME=JSON`, `--batch-param NAME=JSON`, and optionally `--batch module:factory --batch-binding envelope|single|args|kwargs` in scripts or when a factory requires runtime inputs. `show` reports each module's registered path and whether the configured sample forward observed it. `assembly generate --variant working` supports built-in Layer operations and explicit sharing changes. Registered component refs and custom edit Providers fail closed with a structured diagnostic; the adapter still relies on the project's Python modules and dependencies being importable.

需要检查损失和梯度时，提供一个接收模型输出和原始批次、返回标量 Tensor 的用户损失函数：

```python
import torch.nn.functional as F

def classification_loss(outputs, batch):
    targets = batch["loss_inputs"]["targets"]
    return F.cross_entropy(outputs, targets)
```

```sh
m-model model verify \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch \
  --loss my_project.training:classification_loss
```

`verify` 会执行一次真实前向、用户损失和反向，报告损失及逐参数梯度范数；不会执行 optimizer step，也不是训练循环或 checkpoint 验证。带标签的批次可使用 `args`、`kwargs` 和 `loss_inputs` 组成的 envelope，让同一批次既能调用模型，也能交给损失函数。

`try` 让用户用普通 Python 函数定义一个训练步骤，同时由 CLI 管理一个有上限的 AdamW 迭代：

```python
import torch

def classification_training_step(model, batch, context, label_smoothing=0.0):
    outputs = model(*batch["args"], **batch["kwargs"])
    targets = batch["loss_inputs"]["targets"]
    loss = torch.nn.functional.cross_entropy(
        outputs, targets, label_smoothing=label_smoothing
    )
    context.log("accuracy", (outputs.detach().argmax(-1) == targets).float().mean())
    return {"loss": loss, "outputs": outputs}
```

```sh
m-model try \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch \
  --step my_project.training:classification_training_step \
  --steps 4 --lr 1e-4 --step-param label_smoothing=0.1
```

步骤返回的 `loss` 必须是有限、可微的标量 Tensor；引擎负责 backward、梯度/参数/优化器状态有限性检查和 AdamW 更新。报告包含具名 Loss、指标和有梯度的参数名（最多显示 256 项）。`context.step`、`context.total_steps` 和 `context.log(name, value)` 可用于进度相关逻辑和指标。如果要在运行对照中比较实际使用的样本，可在每个步骤调用 `context.report_sample_ids(sample_ids)`；只保存按步骤顺序汇总的 SHA-256 和样本数，不保存 ID。只要有一步未报告稳定 ID，运行的样本身份就显示为“信息不足”。这是训练代码的显式报告，工具不会独立验证 ID 与批次内容相符。试跑最多 1000 步，运行在前台 worker 中，受 `--timeout` 和 Ctrl+C 控制。当前模型权重及优化器状态只存在于进程内，命令结束即丢弃；`try` 用于尽早验证真实训练步骤，不是正式训练任务、checkpoint 或恢复机制。终端菜单的“训练步骤试跑”也使用同一能力。

需要显式管理多组优化器和阶段时，可选 `--training-mode manual-step`。用户步骤负责清梯度、反向和更新；通过 `context.optimizer(name, parameters, lr=...)` 注册在本次试跑中复用的 AdamW 组，用 `context.backward(loss)` 执行可检查的反向：

```python
import torch

def manual_step(model, batch, context):
    encoder_optimizer = context.optimizer(
        "encoder", model.encoder.parameters(), lr=1e-4
    )
    head_optimizer = context.optimizer(
        "head", model.head.parameters(), lr=3e-4
    )
    encoder_optimizer.zero_grad()
    head_optimizer.zero_grad()
    outputs = model(batch["inputs"])
    loss = torch.nn.functional.cross_entropy(outputs, batch["targets"])
    context.backward(loss)
    encoder_optimizer.step()
    head_optimizer.step()
    return {"loss": loss.detach(), "outputs": outputs}
```

仓库的动态多模态样例可以直接试跑：

```sh
m-model try \
  --training-mode manual-step \
  --model examples.complex_acceptance.model:build_model \
  --batch examples.complex_acceptance.data:changing_batch \
  --step examples.complex_acceptance.training:manual_multitask_step \
  --steps 4
```

同名优化器每步须用相同参数和配置；两个组重复包含同一参数会收到明确错误。用户可在步骤中冻结/解冻参数。当前只注册 AdamW。`try` 的状态只存在于进程内；要保存权重、优化器和可恢复状态，使用正式训练：

```sh
m-model train \
  --training-mode manual-step \
  --model examples.complex_acceptance.model:build_model \
  --batch examples.complex_acceptance.data:changing_batch \
  --step examples.complex_acceptance.training:manual_multitask_step \
  --steps 4 --checkpoint-every 1
```

### 接入完整自有训练循环

已有 epoch、梯度累积、自定义采样器、多个 optimizer 或特殊更新规则时，不必把它拆成 CLI 驱动的逐批步骤。`external-loop` 把完整控制权交给 Python：循环自行读取数据、执行 forward/backward/update；Model-Maven 提供总步数预算、进度与指标事件、参数有限性抽查、协作停止和正式运行记录。它不需要单独的 `--batch`：

```sh
m-model try --training-mode external-loop \
  --model examples.external_loop:build_model \
  --loop examples.external_loop:run_loop \
  --loop-param lr=0.01 \
  --steps 4

m-model train --training-mode external-loop \
  --model examples.external_loop:build_model \
  --loop examples.external_loop:run_loop \
  --loop-param lr=0.01 \
  --steps 100 --checkpoint-every 1
```

完整可运行样例在 [`examples/external_loop.py`](examples/external_loop.py)。循环必须在每次完整更新后调用 `context.report_step(loss, metrics)`；返回 `False` 表示预算已到。需要比较数据选择时，可在对应的 `report_step` 前调用 `context.report_sample_ids(sample_ids)`；每一步都必须提供稳定 ID，工具只保存哈希摘要。`context.stop_requested` 可供循环提前退出，CLI 也会在已报告步骤的 checkpoint 边界协作停止。正式训练会原子保存模型、CPU/Python RNG、统计量和循环状态。若循环有 optimizer、sampler 或 epoch 游标等恢复必需状态，应让入口 callable 提供成对的 `state_dict()` / `load_state_dict(state)`；样例中的 callable 对象会一起保存 optimizer 与数据游标。若没有状态，恢复会要求显式 `--allow-state-reset`。恢复命令用 `--resume <run-id> --additional-steps N`。

报告的 step、loss 和指标由循环声明。CLI 会检查标量有限性及每个报告边界的模型参数有限性，但不会独立证明循环确实执行了 backward、optimizer 更新，或每一数据样本都恰好消费一次；结果 capability 会明确列出这些未验证项。停止是协作式的：正式训练达到 `--timeout` 后，CLI 写入停止请求，并最多等待 10 秒让循环在步骤边界保存 checkpoint；循环不报告边界时会强制终止。试跑和其它 worker 操作仍由 watchdog 直接终止。当前运行环境为 CPU、单机、单进程，不是操作系统安全沙箱。

正式训练在 `.model-maven/runs/<id>/checkpoints/` 原子保存最新 checkpoint。要从指定运行继续，显式给出增加的步数；省略 ID 时使用最近一次正式训练：

```sh
m-model train --resume <run-id> --additional-steps 4 --checkpoint-every 1
m-model train --resume --additional-steps 4
```

恢复会固定原来的模型、批次、步骤、优化器设置、源码指纹及 Python/PyTorch 版本，并恢复模型权重、具名 AdamW 状态、训练统计、CPU/Python RNG 和声明的数据/步骤/循环状态。自定义批次工厂、步骤或完整循环入口应提供成对的 `state_dict`/`load_state_dict` 钩子；无隐式状态的函数可以设置 `MODEL_MAVEN_STATELESS = True`。未声明的状态默认阻止恢复；`--allow-state-reset` 是一次明确接受重置后才继续的选择。训练步骤失败时仍可从最近一次完整 checkpoint 显式恢复；活动任务不会恢复，监督身份不明的旧记录也不会自动选择。对明确指定的后台运行，CLI 会重新核验进程启动身份，确认监督进程已退出后才标为中断并允许按 checkpoint 恢复。checkpoint 不包含用户源码或数据，只保留当前最新恢复点。

续跑和评估会将 checkpoint 流式复制到私有临时文件，在校验清单摘要后从同一副本载入。临时目录（`TMPDIR`）需要容纳一份 checkpoint，空间不足时操作会失败且不载入状态；较大模型还需预留复制和校验时间。条件摘要还用于确认用户指定的版本；不指定条件摘要时只校验本次读取的内容与清单一致。

TUI 会在支持该功能的 CLI 上为正式训练或续跑保存待确认启动标记。如果界面意外退出或回执丢失，重新打开训练记录可查看匹配的运行；检查后输入 `forget 序号` 并确认，才能继续发起新的训练或续跑。清除标记不会停止运行，也不能证明尚未显示的运行不会稍后启动。

训练记录页可输入列表序号或完整 run ID 打开运行；即使完整列表不可用，也可以尝试按已知 ID 重连。全屏终端中的 `watch` 会刷新已报告的进度，返回不会停止训练。

Linux 支持受监督的后台正式训练：

```sh
m-model train --detach --model examples.models:build_classifier \
  --batch examples.models:classification_batch \
  --step examples.models:classification_training_step --steps 1000
m-model runs attach                 # TTY 下持续跟随唯一活动任务
m-model runs attach <run-id> --follow=false
m-model runs stop <run-id>          # 下一个完整步骤后保存 checkpoint 并停止
m-model runs stop <run-id> --force  # 立即中断；若已有 checkpoint，可从该处恢复
```

附加时按 Ctrl+C 只退出查看，训练继续运行。`runs attach/stop` 也可管理仍在运行的前台训练。任务保存独立日志、逐步进度和可验证进程身份；若监督身份消失，CLI 将状态标成 unknown，不会误报为仍在运行。显式恢复 unknown 运行时会再次核验进程身份；Linux worker 还会在监督进程意外退出时收到 parent-death 信号。`--timeout` 是总运行上限；正式训练到时后另有最多 10 秒步骤边界 checkpoint 宽限期，超出宽限期仍会强制终止。当前仍是 Linux、单机 CPU 和单进程训练，未提供 GPU 资源调度、跨环境状态迁移或安全沙箱。终端菜单也可选择后台训练、附加和停止。

要交接或独立恢复已训练模型，可将最近正式训练导出到项目目录之外：

```sh
m-model export ../my-model-export
python ../my-model-export/restore_model.py
```

也可通过 `--run <id>` 指定运行。导出会复制该运行观察到的 Python 源码、本地注册组件清单、模型变体（若有）和 checkpoint，逐文件核验 SHA-256，并生成 `manifest.json`、环境清单与恢复脚本。若活动变体使用已登记的独立 Layer/Provider distribution，导出会解析其 PEP 508 依赖声明并递归核对当前环境中满足约束的 Python distribution，将扩展及依赖文件复制到 `vendor/packages/`；缺包、版本不符、重复来源、editable 安装、`.pth` 启动钩子和无法安全枚举的文件都会中止导出。恢复时优先使用哈希核验过的 bundle 副本，不受同名但已漂移的本地安装影响。`manifest.json` 记录包角色、依赖来源与外部运行时版本；PyTorch 保留为外部依赖，恢复脚本会检查精确版本。该闭包不包含操作系统共享库、驱动、数据集、外部模型权重、私有索引配置或非 Python 项目文件；原生扩展仍要求兼容的 ABI 与系统库。新正式训练会在开始时记录所选 Python 环境中的 distribution 名称、版本、来源类别和 UTC 采集时间，并生成规范化 SHA-256 环境指纹；导出复用这份运行证据。`try --repeat` 和 `train --resume` 会在训练步骤前比较 Python ABI/平台、PyTorch 构建及 distribution inventory，checkpoint 载入时还会独立复核指纹。旧记录没有指纹时回退为 Python/PyTorch 版本比较。该指纹能发现包版本/来源类别变化，但不哈希安装文件，也不可用于重建环境；`requirements.snapshot.txt` 仍是全环境版本快照，不是训练时锁定或可直接重建的环境锁文件。目标已存在时默认拒绝覆盖；`--overwrite` 会先把旧目录改名备份，再提交新包。恢复脚本校验源码、bundle 文件和权重后重建模型，只恢复模型权重；原运行记录仍保留优化器状态以供续训。分享前仍应检查源码中的凭证。

每次 `try` 都会把输入、运行前后观测到的源码文件哈希、起止事件和完整报告保存在 `.model-maven/runs/<id>/`。常见凭证字段会在输入记录中标为 `[REDACTED]`，训练时仍使用原值；不要把秘密嵌在普通字符串或源码里。可以在人类可读列表中找回最近运行，再查看状态、步骤、Loss 和失败原因；`--json` 返回完整证据：

```sh
m-model runs list
m-model runs list --query encoder --status failed
m-model runs list --kept
m-model runs show [run-id]
m-model runs storage [run-id]
m-model runs rename <run-id> "encoder ablation"
m-model runs keep <run-id>
m-model runs unkeep <run-id>
```

正式训练和 `try` 都可用 `--name` 标记运行；重命名和保留操作只更新运行元数据与事件，不修改模型、训练配置或指标。

`runs storage` 按运行记录、日志、进度文件和 checkpoint 汇总本地占用；扫描会跳过符号链接，只统计 `.model-maven/runs` 内文件，不递归计算用户源码或数据集。

清理先预览，再执行预览生成的固定计划：

```sh
# 普通临时/旧 checkpoint 清理
m-model runs clean --dry-run

# 另外预览结束运行的 worker 日志和逐步进度；固定计划仍会保留 run 元数据、结果和事件历史
m-model runs clean --include-logs --include-progress --dry-run

# 如需释放最新恢复权重，单独预览指定运行，再执行预览返回的 plan ID
m-model runs clean --run <run-id> --include-checkpoints --dry-run
m-model runs clean --apply <plan-id> --release-checkpoints
```

在终端工作台首页进入 **产物与清理 → 清理运行文件**，可以交互完成日志/进度选择、预览和确认。

进入 **编辑模型 → 管理根工厂装配接缝**，可查看接缝、按注册组件 Schema 收集参数，或新增/替换/移除组件；候选构造和示例前向通过后才会询问是否保存。工作台支持按页面序号、操作名称或关键词导航，也可输入 `back` 返回。

普通预览返回的计划可用 `m-model runs clean --apply <plan-id>` 执行，不需要 `--release-checkpoints`。

普通计划只考虑已结束、未标记保留运行中的崩溃临时文件和未被有效 `latest.json` 引用的旧 checkpoint。`--include-logs` 和 `--include-progress` 可将 `worker.log` 与 `progress.jsonl` 纳入预览；清理这些文件仍保留 run 元数据、输入、结果和事件历史。活动、状态未知或已保留运行会跳过。最新恢复权重默认受保护。只有为一个明确运行单独预览并列出最新权重后，执行时才可追加 `--release-checkpoints`。计划保留 7 天，执行会重新核对运行状态、保留标记、权重引用、文件大小和 SHA-256；变化的文件会跳过，不扩大删除范围。释放最新权重后，运行摘要和输入仍可查看，但不能再从该运行续训或导出权重。

用 `m-model try --repeat` 重跑最近的训练试跑，也可以传入明确的 run ID。重跑会重新初始化权重，并沿用原步骤、预算、种子和 Python 环境；例如 `--steps 20`、`--lr 0.001` 会作为本次覆盖写入新记录，并关联原 run。原记录里脱敏的模型/批次/步骤参数必须通过对应 `--*-param NAME=JSON` 重新提供。重跑前会比对本地 Python 源码及 Python/PyTorch 版本；来源变化时停止在训练步骤之前，并建议用当前代码启动新的 `try`。

运行记录不复制用户源码或数据；它保存入口、参数和已观察到的本地 Python 文件哈希，便于发现来源变化。`try` 记录不包含模型权重或优化器状态；只有正式 `train` 运行会保存 checkpoint。

## Inspect and edit a model variant

```sh
m-model layer list --model my_project.model:build_model
m-model layer show model.encoder.proj --model my_project.model:build_model

m-model layer add \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch \
  --parent model.encoder --after relu \
  --name dropout --type Dropout --param p=0.2

m-model layer set model.encoder.dropout \
  --batch my_project.data:example_batch --param p=0.1
m-model layer remove model.encoder.dropout \
  --batch my_project.data:example_batch
m-model layer undo
m-model layer redo
```

需要同时改动相邻层时，用 `layer apply` 把多个操作作为一个候选一起验证、一次提交，也能用一次 `undo` 整组撤销。例如同步修改投影层和 Head 的宽度：

```sh
m-model layer apply \
  --operation '{"action":"set","path":"model.encoder.proj","params":{"out_features":16}}' \
  --operation '{"action":"set","path":"model.head","params":{"in_features":16}}' \
  --batch my_project.data:example_batch
```

每项 `--operation` 接受一个 JSON 对象。组内任一操作或最终前向失败，整个候选都不会写入变体。

当相关 Layer 分散在多个 Python 文件时，也可以将多项参数修改作为一个源码事务预览和提交：

```sh
m-model layer apply \
  --operation '{"action":"set","path":"model.encoder.proj","params":{"out_features":16}}' \
  --operation '{"action":"set","path":"model.head","params":{"in_features":16}}' \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch \
  --target source --dry-run
```

检查差异后，移除 `--dry-run` 再执行以提交。源码目标目前只接受 1 到 64 项 `set`，每个 Layer 必须唯一映射到项目 `.py` 文件。候选统一验证后写入事务日志，再逐文件原子替换；发生进程中断时，后续源码操作会从日志完成该事务。模型重载/前向复核失败时整体回滚。事务日志对所有文件保存前后快照；它提供可恢复的一致事务，不承诺其他进程在提交期间看不到中间文件状态。单个文件上限 16 MiB、一次事务快照总量上限 64 MiB、预览响应上限 24 MiB。涉及成组 `add`/`replace`/`remove` 的计划会在写入前拒绝。

Edits are saved as a structure recipe under `.model-maven/variants/working.json`; they do not overwrite the Python source. The first edit needs the model factory. A representative batch is optional for construction-only edits and can be passed to validate the candidate with a real forward before it is saved. Use `--dry-run` to inspect a candidate without committing it, or `--require-forward` to require a batch.

Use named variants to keep separate structural experiments:

```sh
m-model variant list
m-model variant show working
m-model variant copy working wider-encoder
m-model --variant wider-encoder layer set model.encoder.proj \
  --model my_project.model:build_model \
  --param out_features=768
```

Copy creates an independent structure recipe and refuses to overwrite an existing name. It does not copy model source or trained weights. Listing reports unreadable JSON files instead of hiding them; `show` reads the descriptor but does not replay it. Model inspection or a Layer edit performs the source, component and revision checks before applying the recipe.

A variant's seed fixes reconstruction of its base model and saved Layer edits. When running `probe`, `try` or `train` against that variant, an explicit `--seed` sets the execution RNG after the variant is reconstructed and does not change the saved structure. A Layer edit keeps the existing variant seed; omit `--seed`, or choose a new variant name with `--seed` when you want different initialization. Training runs record their execution seed for repeat/resume checks.

The current built-in layer editor supports parameter updates for `Linear`, `Dropout`, `ReLU`, `GELU`, `LayerNorm`, and `Conv2d`. Persistent variant insertions and true removals target native `nn.Sequential`, `nn.ModuleList`, and `nn.ModuleDict` children, including user subclasses that retain PyTorch's standard `_modules` registration semantics. A supplied batch is run through the actual subclass `forward` before the variant is committed. Source edits remain limited to directly assigned native containers. ModuleList positions use numeric indexes; ModuleDict branches use their names. A replacement can use a built-in `--type` or a local factory such as `--component my_project.layers:make_activation`. Unsupported graph edits fail instead of claiming that an unrelated forward path changed.

After a successful edit, the human terminal summary names the target and operation. Built-in edit handlers list preserved or newly initialized parameters, warn when old weights were not migrated, and call out every alias when a shared module is replaced. Provider edits show the Provider name and its returned summary. `--json` retains structured edit effects for scripts and agents.

Layer inspection also reports the project Python line that constructed each observed module when it can trace one, plus the module type definition location. Dynamic factories that cannot be mapped to a project callsite still show the type definition when available.

For a supported, uniquely located built-in Layer constructor, `set --target source` can update constructor parameters in the project `.py` file. `layer apply --target source` can update multiple constructors across project files in one recoverable transaction. For directly assigned `nn.Sequential`, `nn.ModuleList`, and `nn.ModuleDict` containers, `layer apply --target source` can group `add`, `replace` and `remove` operations across containers and project files. The plan rebuilds each ordered container while retaining unchanged child modules and their weights. `ModuleList` paths and insertion anchors use numeric indexes, which are renumbered after insertions and removals. `ModuleDict` paths and anchors use named branches; new names must be valid Python identifiers without dots, and branch order is preserved. Omit `--name` when adding to a `ModuleList`. Preview first; omit `--dry-run` only when you want to write the change:

```sh
# Find a Layer by a case-insensitive path, type, or alias match.
m-model layer list --model my_project.model:build_model --search attention

m-model layer set model.encoder.proj \
  --model my_project.model:build_model \
  --param out_features=768 \
  --target source \
  --dry-run

# Preview source insertion into a Sequential, then repeat without --dry-run to commit it.
m-model layer add --parent model.encoder --after relu --name dropout \
  --type Dropout --param p=0.2 --model examples.models:build_editable \
  --batch examples.models:editable_batch --target source --dry-run

# Once the diff is right, run that add command again without --dry-run.
# ModuleList uses numeric indexes and automatic names, so omit --name.
m-model layer add --parent model.blocks --before 3 --type Dropout \
  --param p=0.1 --model my_project.model:build_model \
  --batch my_project.model:example_batch --target source --dry-run

# ModuleDict uses existing branch names as anchors and a new name for the added branch.
m-model layer add --parent model.branches --before right --name projection \
  --type ReLU --model my_project.model:build_model \
  --batch my_project.model:example_batch --target source --dry-run

m-model layer apply --model examples.models:build_editable \
  --batch examples.models:editable_batch --target source --dry-run \
  --operation '{"action":"add","parent":"model.encoder","after":"relu","name":"dropout","layer_type":"Dropout","params":{"p":0.2}}' \
  --operation '{"action":"replace","path":"model.encoder.relu","layer_type":"GELU","params":{}}'

m-model layer undo --target source
m-model layer redo --target source
```

The source `set` processor supports built-in editable Layer types and parameterless project-defined `nn.Module` constructors with explicit keyword-capable signatures when each constructor call and runtime Layer instance map uniquely to one another. It reports project constructor fields in `layer show`, updates explicit positional or keyword arguments, or adds a missing keyword to a one-line call. The generic handler currently accepts custom modules with no parameters or buffers; weighted custom modules need an explicit state-migration handler. Custom source fields are available only with `--target source`; persistent variants keep their narrower built-in reconstruction handlers. Source structural CRUD supports direct assignments such as `self.encoder = nn.Sequential(...)`, `self.blocks = nn.ModuleList([...])`, and `self.branches = nn.ModuleDict({...})`, with unique source locations and built-in or importable Python factory Layers; custom containers, shared containers/layers, ambiguous or repeated callsites, and compound one-line assignments are rejected. A grouped structure plan may contain up to 64 `add`/`replace`/`remove` operations across those containers and source files. A mixed source plan can apply parameter `set` operations before structural operations; overlapping targets, edits to the same source line, and ambiguous order are rejected. The preview evaluates the combined in-memory candidate; source is reloaded and validated after commit. A local CLI and workspace walkthrough covered coordinated dimension changes and a structural addition; other models and edit patterns still require validation. Python factories are written into the generated source as ordinary imports and explicit JSON parameters, so the resulting code remains readable and runnable without Model-Maven. Registered component factories are resolved and schema-checked before editing; the patch records the selected component version and manifest hash as a source comment, while the project's Python environment remains responsible for installing and pinning the package. Rebuilds preserve child order and unchanged child objects; Sequential names stay intact, ModuleList indexes are renumbered after structural changes, and ModuleDict branch names and order are retained. New/replacement modules get deterministic isolated initialization seeds so they do not consume random numbers intended for later model construction. Source plans save before/after snapshots and a recovery journal under `.model-maven/source-edits`, replace each file atomically, then reload the model in a fresh worker and run a forward check when a batch was supplied. Failed reload/forward validation triggers an automatic rollback of the whole plan. If interrupted during a multi-file commit, the next source history/edit command resumes it from the journal; an unexpected file hash stops recovery and preserves the journal for manual resolution. Undo/redo checks all file hashes and refuses to overwrite later manual edits. The interactive workbench has preview/confirmation flows for grouped source `set` and grouped Sequential/ModuleList/ModuleDict CRUD.

Grouped source plans support parameter `set` operations before structural `add`, `replace` and `remove` operations, subject to the ordering and overlap restrictions above. Custom edit-provider operations cannot be mixed into this source transaction. The structural handler targets directly assigned, uniquely located `nn.Sequential`, `nn.ModuleList`, and `nn.ModuleDict` containers. Custom containers and assembly-target edits are not implemented. For broader custom graph changes, use a persistent variant or a Layer Edit Provider. Source factories are emitted as explicit Python calls; subsequent package upgrades are ordinary source/environment changes and are not pinned by the source editor.

For custom containers or graph-specific edits, add a local provider instead of changing Model-Maven core. A provider may be called directly or registered as a versioned project component. To avoid hand-assembling the package protocol, scaffold a distributable Layer factory or Layer Edit Provider:

```sh
m-model component scaffold acme/gated-block --kind layer --output extensions/gated-block
m-model component scaffold acme/insert-gate --kind layer-edit-provider --version 0.1.0 --output extensions/insert-gate
m-model component install extensions/gated-block
m-model component test acme/gated-block
m-model component install extensions/insert-gate
m-model component test acme/insert-gate
```

The output is a standalone setuptools project with an entry point, static manifest, Python starter implementation, and package README. `--output` must be a new project-relative directory; existing directories and symlinked parent paths are rejected. The Layer example starts as a SiLU factory. The Provider example appends a named SiLU to a selected `nn.Sequential` and includes an explicit fixture for `component test`. Edit the generated module and keep the manifest parameter schema in sync. `component install DIR` builds a wheel and copies it into a SHA-256-addressed `.model-maven/extensions/` directory, then automatically discovers and registers its components. The selected Python environment and its requirements lock are left untouched; runtime dependencies must already be present there, so add them to `requirements.txt` and run `m-model env setup` first. Wheel build isolation may download declared build tools; when those tools are already installed, `--no-build-isolation` skips that download after checking `[build-system].requires`. The build backend executes with your user permissions; install only trusted source packages. The generated Provider uses the API v2 lifecycle; the optional SDK supplies typing contracts, while the runtime conformance suite remains partial.

Remove a registered version with a preview first:

```sh
m-model component uninstall acme/gated-block@0.1.0
m-model component uninstall acme/gated-block@0.1.0 --apply
```

Registered providers get static payload validation and their exact manifest version/hash is recorded in the model variant:

```python
from torch import nn

MODEL_MAVEN_LAYER_EDIT_API = 1

def add_silu(model, operation):
    name = operation["payload"]["name"]
    model.encoder.layers[name] = nn.SiLU()
    return {"added": f"model.encoder.layers.{name}", "type": "SiLU"}
```

```sh
m-model layer custom examples.models:add_silu \
  --payload '{"name":"silu"}' \
  --model examples.models:build_editable \
  --batch examples.models:editable_batch
```

Provider v1 is a deliberately small Python callable contract: the referenced function accepts `(model, operation)` and returns a JSON object, with `MODEL_MAVEN_LAYER_EDIT_API = 1` declared in its module. The operation is replayed from the saved variant and the resulting model is checked with the supplied batch before commit.

To make the provider discoverable and validate its input fields before Python runs, register it with a payload Schema:

```sh
m-model component add example/add-silu \
  --kind layer-edit-provider \
  --entrypoint examples.models:add_silu \
  --version 1.0.0 \
  --description "Insert an activation into the project's custom encoder" \
  --params-schema '{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}' \
  --capabilities '{"layer_edit":{"state":"supported"}}' \
  --conformance-fixture examples.models:provider_fixture

m-model component list --query layer-edit-provider
m-model component show example/add-silu
m-model component test example/add-silu --param 'name="silu"'
m-model layer custom example/add-silu@1.0.0 \
  --payload '{"name":"silu"}' \
  --model examples.models:build_editable \
  --batch examples.models:editable_batch
```

For a registered Layer Edit Provider, `conformance_fixture` is a no-argument callable that returns a fresh disposable model and optional expectations:

```python
def provider_fixture():
    return {
        "model": build_editable(),
        "expected_result": {"added": "model.encoder.layers.silu", "type": "SiLU"},
        "expected_modules": {"encoder.layers.silu": "SiLU"},
    }
```

`component test` passes the schema-validated `--param` values as the Provider payload, checks the declared API v1 callable or v2 lifecycle protocol and JSON result, then checks any declared result/module expectations. Module paths are relative to the fixture model; a `null` expected type means the path must be absent. At least one explicit expectation is needed for `complete_for_fixture_expectations`, and every other declared capability must also be checked or skipped; otherwise the report remains `partial`. A complete check records the exact payload digest, manifest hash, Provider/fixture source hashes, conformance-suite version, and Python/PyTorch versions under `.model-maven/component-verifications`. Registered Provider edits are rejected unless that record matches the requested payload and current code/environment. The terminal workbench runs this check automatically after collecting the payload; in explicit CLI workflows, run `component test` first. The test edits only the disposable fixture model and does not validate saved variant/source persistence, undo, state migration, transitive dependencies, or arbitrary inputs. The local verification record is a workflow guard, not a signed attestation or security sandbox.

The same registered provider can be used inside `layer apply` as a `custom` operation, and each registered custom operation must carry matching verification evidence. The manifest and payload are checked statically, while provider Python executes with the selected interpreter's permissions. API v1 callable providers remain supported. API v2 providers expose `describe`, `can_edit`, `plan`, `candidate`, `persist`, and `rollback`; the worker validates bounded JSON plans and results, records the plan with the variant, and refuses replay if its persisted description drifts. `rollback` is a provider-owned compensation hook; it cannot undo unrelated filesystem, network, or process side effects.

For direct registered-module sharing changes, use a variant `layer apply` operation. Default `replace` behavior keeps all aliases bound to the replacement. To split one named alias while retaining its weights, use `{"action":"sharing","mode":"clone-selected","path":"model.decoder"}`. To connect an explicitly named branch to another module instance, use `{"action":"sharing","mode":"share-selected","source":"model.encoder","targets":["model.decoder"]}`. This replaces each target module and its parameters/buffers with the source instance; provide a representative batch to check forward compatibility. These paths refer to registered module-tree names, never dynamic invocation numbers. Reconnection of custom forward branches with semantics beyond module identity remains a Provider responsibility; the tool does not infer dataflow from arbitrary Python control flow.

In the guided terminal workbench, a registered provider opens a Schema-driven form: required fields, descriptions, defaults, enum choices, nested objects, and numeric bounds are shown while you enter values. Arrays can be entered as JSON. Values are checked against the selected manifest; Model-Maven then runs the fixture check for that payload before applying the edit and pins the component version plus verification evidence. A direct `module:callable` provider still accepts a JSON payload because it has no registered Schema and remains an explicit unregistered-code path.

## Reuse a project component

Register an existing Python Layer factory once, then discover it by a stable ID:

```sh
m-model component add example/scaled-activation \
  --kind layer \
  --entrypoint examples.models:ScaledActivation \
  --version 1.0.0 \
  --description "Scaled activation used by the image encoder" \
  --params-schema '{"type":"object","properties":{"scale":{"type":"number","default":1.0}},"required":[],"additionalProperties":false}' \
  --capabilities '{"construct":{"state":"supported"},"forward_probe":{"state":"supported","requires":["example_batch"]}}'

m-model component list
m-model component check example/scaled-activation
m-model component discover
m-model component test example/scaled-activation \
  --param scale=0.75 --batch examples.models:editable_batch
m-model layer replace model.encoder.relu \
  --component example/scaled-activation@1.0.0 \
  --model examples.models:build_editable \
  --param scale=0.75 \
  --batch examples.models:editable_batch
```

Components can declare required local components and optional features with `--dependencies`:

```sh
m-model component add project/video-encoder \
  --kind layer \
  --entrypoint my_project.encoders:build_video_encoder \
  --dependencies '[{"component_id":"project/attention-kernel","version":"^1.2.0"}]'
m-model component check project/video-encoder
```

`component check` resolves the full registered component graph without importing Python entrypoints. Version constraints use SemVer ranges (`^1.2.0`, `~1.2.0`, `>=1.2.0,<2.0.0`; use `=1.2.3` for an exact version). The resolver chooses the highest compatible local version, backtracks when shared dependencies conflict, and reports missing versions and cycles. Optional dependencies are not activated automatically. A Layer edit records every selected dependency's version and manifest hash in the variant; reload checks that lock and export includes package-backed dependencies. This resolves Model-Maven component manifests only; it does not install Python distributions or solve the interpreter's package environment.

For a separately buildable package using a standard Python entry point and static sidecar, see `examples/extensions/silu-component`.

To exercise the real wheel path without installing into the active Python environment, build the CLI and run the isolated acceptance workflow:

```sh
cargo build
python3 scripts/verify_extension_package.py --python python3
```

The script copies the sample package to a temporary directory, builds and installs its wheel into a temporary site-packages directory, then checks static discovery, factory construction, Layer replacement, and variant reload in a fresh CLI process. It runs one formal training step, exports the checkpoint plus registered distribution files, restores after removing the original installation, verifies a changed installation cannot shadow the bundled copy, and rejects a tampered bundle file. The selected Python needs PyTorch, pip, setuptools, and wheel. Temporary wheel/build artifacts are removed automatically.

Layer Edit Providers can also ship as independent wheels. The reference package in `examples/extensions/silu-provider` declares a static manifest, payload Schema, API v1 entry point, and a disposable conformance fixture. Run its full acceptance workflow with:

```sh
python3 scripts/verify_provider_extension.py --python python3
```

This builds and installs a Provider wheel plus a separate helper dependency in a temporary location, proves discovery does not import the Provider, checks its exact fixture payload, edits a separate project's model, exercises variant undo/redo and fresh-process reload, then trains and exports. The acceptance checks that the helper is recorded as a transitive runtime dependency, Torch is recorded as an external runtime, restore succeeds after the original packages are removed, changed installations do not shadow bundled files, and tampered bundle files are rejected. It covers this PEP 508 wheel case only; it does not establish correctness for arbitrary models, full Provider state migration, operating-system library portability, or a security boundary.

Component descriptors live under `.model-maven/components/<namespace>/<name>/<version>.json`. `component check` validates the static descriptor and parameter Schema, resolves required transitive component dependencies against locally registered versions, detects missing versions/cycles/range conflicts, and does not import Python. Constraints use SemVer range syntax; a plain `1.2.3` is a caret range, while `=1.2.3` pins an exact component version. Optional dependencies are reported and left inactive unless selected through a required path. `component discover` scans the selected interpreter's installed distributions, the project-local packages installed by `m-model component install`, and repeatable `--path SITE_PACKAGES` directories for the standard `model_maven.components` entry-point group. Entry-point values use normal `module:callable` syntax. The corresponding static manifest is stored beside the entry module in `__model_maven__/<entry-point-name>.json`; for example, `silu = demo_ext.layers:make_silu` maps to `demo_ext/__model_maven__/silu.json`. It is read without loading the extension module. Discovery copies validated manifests into the project registry and records the distribution name/version, entry-point name, manifest hash, and Python source hashes for the declared entry point and optional fixture. Identical repeated discovery is idempotent; a same-ID/version content conflict is reported without overwriting the existing project manifest. `--distribution NAME` limits the scan. Discovery does not install or upgrade packages.

`m-model component install DIR` builds the package source as a wheel and installs it under `.model-maven/extensions/<wheel-sha256>`, then discovers and registers its components. By default, pip uses an isolated build environment for `pyproject.toml` build requirements. For a faster build when those requirements are already installed in the selected Python, use `m-model component install DIR --no-build-isolation`; Model-Maven checks `[build-system].requires` first and stops with the missing requirements instead of waiting on an implicit download. The interactive workbench asks which mode to use and defaults to isolation. Both modes check that the wheel's active PEP 508 runtime requirements already exist in the selected Python environment, and neither installs runtime dependencies or changes the environment lock. Put runtime dependencies in the model project's requirements and run `m-model env setup` first. A package build backend executes as the current user, so only install trusted source directories. Package versions are immutable in the component registry: bump the component version when changing a package's manifest or source.

`m-model component uninstall ID@VERSION` only previews a removal. Add `--apply` to unregister that exact component version, remove its Provider verification record, and delete the project wheel when no other registered component uses it. The operation scans saved variants, run records, and project Python imports; it refuses to run while a training process is active or its state is unknown. If a saved variant or statically detected source import still references the component, or Python source scanning is incomplete, `--apply` is rejected until you explicitly add `--force`. The original source and variants remain untouched and may stop loading after a forced removal. Import scanning covers regular Python files and literal `import_module()`/`__import__()` calls; dynamically constructed imports and runtime uses cannot be proven absent. Failed or interrupted cleanup can leave a hidden `.uninstall-*` directory under `.model-maven/extensions`; inspect it before manual removal.

When registering a component, declare dependencies with repeatable flags instead of writing JSON by hand:

```sh
m-model component add project/fusion \
  --kind layer --entrypoint project.layers:make_fusion --version 1.0.0 \
  --dependency 'project/encoder@^2.1.0' \
  --dependency 'project/attention@~1.4.0' \
  --optional-dependency 'project/visualizer@^0.3.0'
```

Use `--dependencies '<JSON array>'` when generating manifests programmatically or supplying a larger dependency list. Both forms can be combined; duplicate component IDs are rejected with a direct validation error.

At runtime, a discovered component checks that its exact distribution version, entry-point metadata, manifest bytes, and recorded entry/fixture source files are still present before Python code runs. Variant references also carry a flat lock of selected transitive component versions and manifest hashes; reload rejects changed dependency manifests. Export resolves package-backed components from that lock. These checks are provenance and staleness checks, not a signature or security sandbox; component manifests do not fingerprint every transitive Python import. Moving the project to a different environment may require rediscovery because distribution locations and files are part of the recorded source. `component test` supports registered Layer factories, Layer Edit Providers, transforms, datasets, losses, metrics, and trainers. A Layer test checks `nn.Module` construction and finite initialized state, can run a finite-output forward probe, and checks a declared `state_roundtrip` by loading the extension-owned state into a fresh instance and comparing fixture forward output. Provider tests execute one edit against a disposable fixture model and check declared API/JSON and structure expectations. Transform tests use the ordered transform pipeline; dataset tests compare a bounded indexed sample; loss tests require a finite scalar and finite input gradients; metric tests compare a finite fixture result. Trainer fixtures run the real external-loop engine to check reported step/loss/metric events and cooperative stop polling. A trainer `timeout` claim is checked through a runner-owned cooperative deadline: the Trainer receives `timeout_requested` and `remaining_time_seconds`, then must stop at a reported step boundary. This cannot interrupt a blocked Python/CUDA call; the CLI watchdog remains the process-level backstop. Component references and fixture/entrypoint source fingerprints are rechecked after supported fixture checks. The JSON report distinguishes passed, skipped, not-run, and untested capabilities; partial coverage is not a general compatibility guarantee. These fixtures do not prove arbitrary-input correctness, all training branches, or OS-level isolation. Omitting `@version` selects the highest locally registered SemVer; the exact selected version and descriptor hash are locked into the variant. Layer editing validates parameters and applies declared defaults before construction. The selected Python factory still runs with the current interpreter's permissions when an edit is constructed or tested. Direct `module:callable` factories remain available without registration as an explicit code path. Component resolution does not install packages or solve the Python environment.

## Human and automation interfaces

Running `m-model` in a terminal opens a guided menu for inspection, forward and gradient checks, Layer listing, add/set/replace/remove, custom Layer providers, variant list/show/copy, variant undo/redo, source Layer CRUD preview/confirmation, grouped source parameter edits, grouped source Sequential/ModuleList/ModuleDict structural CRUD, source undo/redo, component package scaffolding, discovery, dependency checks and Layer testing, project environment setup/lock/check, bounded training-step trials, formal training/resume, background attach/stop, and recent run records. Layer actions use a live model-tree picker: search by path, module name, type, or alias, then choose among matches; the list shows parameter counts, shared instances, and available editable fields before an edit begins. The direct `layer list --search KEYWORD` command also provides case-insensitive path/type/alias filtering. Sequential insertion prompts use child names, ModuleList prompts use numeric indexes and assign names automatically, and ModuleDict prompts use named branches; grouped source plans dry-run and refresh the tree after every structural operation, so subsequent choices follow the candidate model structure. Edits cannot accidentally select the root model. The grouped source edit flow collects up to 64 operations, previews the complete plan, then applies it after one confirmation. It keeps the selected model and batch during that session and asks only for the inputs needed by each action. Source edits show the diff before asking for confirmation. Registered Layer factories and Layer Edit Providers use Schema-driven field prompts, including defaults, choices, and value constraints; before a registered Provider edit, the workbench tests that exact payload against its fixture and stops if the fixture does not establish an explicit expected result. Direct factories and `module:callable` providers accept free-form JSON parameters. Explicit subcommands are available for repeatable work. Add `--json` to get one structured JSON report on stdout; diagnostics and optional worker logs go to stderr. `--timeout SECONDS` bounds a worker operation; Ctrl+C cancels a foreground worker group, while Ctrl+C during background attach exits only the viewer.

The JSON report has a versioned protocol, a stable operation status, machine-readable diagnostics, and a contextual hint. `m-model doctor` reports capabilities for the active Python interpreter. The `python/worker.py` adapter is embedded in the Rust executable, so a built binary does not need the source checkout at runtime.

### Shell completion and non-interactive use

Load dynamic completions from the installed binary. The command prints the shell code to add to your profile:

```sh
m-model completions bash       # choose bash, zsh, fish, powershell, or elvish
source <(m-model completions bash)  # in Bash
m-model --no-interactive data preview --json
```

For Zsh, run `autoload -Uz compinit; compinit` and then `source <(m-model completions zsh)`. For Fish, use `m-model completions fish | source`; for PowerShell, use `m-model completions powershell | Out-String | Invoke-Expression`. These dynamic completions include the live command tree, locally registered component IDs and versions, and Schema-backed parameter names and enum values for `component test`, registered Layer add/replace, component-backed Assembly add/set, and named model injections using `--inject SLOT=component:ID` with `--inject-param SLOT.NAME=JSON`. For declared schemas, completion can also suggest bounded whole-JSON values for enum choices and required nested object fields whose values have defaults or enums. Candidate discovery reads static component manifests and does not import project Python code; arbitrary Python factories and partial free-form JSON are not guessed. Re-source on shell startup so completions stay synchronized with the installed binary. `--no-interactive` prevents prompts even when the command is attached to a terminal; missing required input is returned as a diagnostic.

## Current boundaries

- Execution is CPU-only and limited to model construction, inspection, one-batch forward probing, a user-defined scalar-loss backward check, managed/manual AdamW trials, user-owned full-loop trials and foreground training with explicit opt-in state restoration, managed/manual foreground or Linux-supervised background training with explicit checkpoint resume, verified training-run export, and the implemented variant edits. External loops remain user-owned; when `--apply-supervision` is enabled, only loops using `context.supervised_step`, `context.backward`, and registered optimizer hooks receive the marked-port rule checks. Other forward/backward/optimizer behavior remains unverified. Run inputs/results persist; GPU scheduling and cross-environment state migration are not implemented.
- `m-model env lock/setup/check` resolves wheel-only Python dependencies, records exact package versions and wheel hashes, installs them in the selected project venv (default `.venv`), and checks Python ABI, requirements inputs, package versions, and `pip check`. `m-model env wheelhouse archive/check/replay` stores and verifies local hash-checked wheels and installs them offline into an existing same-ABI venv. The resolved closure contains the declared packages and their dependencies; pip configuration supplies package sources without writing index URLs or credentials to the lock or archive index. The workflow does not build source/VCS/editable dependencies or capture OS/accelerator libraries. Existing incompatible environments are left in place and reported.
- The Python worker runs with the same operating-system permissions as the invoking user. The timeout is process management, not a security sandbox; do not use untrusted model code.
- A variant pins observed Python source, registered component manifest hashes, and the PyTorch version. Changes to pinned files or a PyTorch version mismatch are rejected. Formal training runs additionally fingerprint the Python/ABI/platform, PyTorch build and installed distribution inventory; repeat/resume checks it before running steps. The inventory fingerprint does not hash installed package files and cannot recreate the environment. The recipe records structure, not trained weights, and currently uses an absolute project path.
- Complex dynamic graphs can be inspected only to the extent that their real forward works for the supplied batch. This prototype does not promise static shape inference, exhaustive branch coverage, arbitrary graph rewiring, or training-state correctness.
- The project component catalog supports static descriptors, registered Layer factories, Layer Edit Providers, trainers, and fixture conformance for transforms, datasets, losses, metrics, state roundtrip, trainer events, and cooperative stop. Checks execute the real registered entrypoint with resolved defaults, pin and recheck component references, and verify bounded fixture expectations; transform checks use the ordered data pipeline, loss checks require a finite scalar and finite input gradients, and dataset checks compare a bounded indexed sample. These fixtures do not connect component references automatically to project training configuration or prove arbitrary-input behavior. Source editing supports constructor `set` plans and grouped `nn.Sequential` add/replace/remove transactions spanning project files, with built-in or importable Python factories and a recovery journal. `component scaffold`, `component install`, and reference-aware `component uninstall` provide starter packages, project-local hash-addressed wheel installs, and preview/apply removal; installation leaves the selected environment unchanged, and uninstall checks variants, run records, project imports, and active training. `scripts/verify_component_scaffold.py` verifies this path, including forced-reference guards and the TTY flow. Static discovery of installed Python component distributions is implemented; `scripts/verify_extension_package.py` accepts an independent Layer-factory wheel, and `scripts/verify_provider_extension.py` accepts an independent Provider wheel through fixture validation, model editing, undo/redo, training, hashed export, and offline restore. Export now bundles verified PEP 508 Python dependency closure and records exact external PyTorch runtime requirements; distribution files that symlink outside their package root fail with an explicit external-resource diagnostic instead of copying host files. Environment locking currently supports wheel-only artifacts from the selected pip configuration; it does not vendor wheels or preserve private-index configuration. OS/accelerator portability, generic preemption of blocked trainer calls remains unsupported; finite extension fixtures do not prove arbitrary-input correctness or provide OS-level isolation.

## Developer checks

Exercise the project environment lifecycle entirely offline with synthetic wheels:

```sh
cargo build
python3 scripts/verify_environment_setup.py --binary target/debug/m-model --python python3
```

The acceptance flow covers wheel hashes, lock creation, setup and reuse, `pip check`, custom venv selection, drift rejection, and redaction of direct-URL credentials. Its synthetic `torch` package validates the environment manager only; it does not test real PyTorch model execution or accelerator support.

Exercise generated Layer and Provider packages through project-local wheel installation, automatic discovery, runtime checks, dependency preflight, and environment non-mutation:

```sh
cargo build
python3 scripts/verify_component_scaffold.py --binary target/debug/m-model --python python3
```

This acceptance flow requires a Python interpreter with PyTorch and access to the configured package source for isolated build-system requirements. It uses temporary projects and does not install into that interpreter.

```sh
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
```

PyTorch integration cases use `python3` by default. Set `MODEL_MAVEN_TEST_PYTHON` to the interpreter that has the project's PyTorch installation when it differs from the default.

## Project status

The [shared roadmap](todo.json) defines two release journeys: build a custom model from an empty directory and repair an existing PyTorch model. [Acceptance records](docs/acceptance/README.md) describe the tested scope of earlier CLI and backend snapshots.

For a `model probe` that must execute a new forward instead of reusing eligible
evidence, set `MODEL_MAVEN_FORCE_FRESH_FORWARD=1` for that CLI process. The
report's `forward_execution` field distinguishes `executed` from `reused`;
older versions that do not provide this field leave execution status unknown.
This setting does not pin weights, input data, external libraries, or later
changes, and does not affect other operations.
