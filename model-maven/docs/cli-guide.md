# Model-Maven CLI guide

This guide is organized around model-development tasks. It starts with your existing Python model and a representative batch; you do not need to register a loss, optimizer, trainer, or dataset to get the first forward diagnosis.

## 1. Check your environment

Run these commands from the model project directory:

```sh
m-model doctor
```

`doctor` reports what the selected Python can actually do. If the project needs packages, add them to `requirements.txt`, then prepare its environment:

```sh
m-model env setup
m-model env check
m-model doctor
```

Setup uses the project `.venv` by default. It resolves wheel packages and leaves your base Python unchanged. To reuse an already installed ML runtime such as CUDA/ROCm PyTorch, explicitly opt in before creating the project environment:

```sh
m-model env configure --system-site-packages
m-model env setup
```

The inherited package versions and base-environment dependency warnings are recorded in the lock metadata and checked for drift. Inherited packages are not included in a wheelhouse archive. The default remains an isolated environment; see [environment portability](environment-portability.md) before relying on inherited packages or moving a project to another machine.

## 2. Inspect and run your existing model

Expose a factory that returns your `torch.nn.Module`, plus a function returning one representative batch. For example:

```python
# my_project/model.py
def build_model():
    ...  # return the model you already use

# my_project/data.py
def example_batch():
    ...  # return one batch in the format your model.forward accepts
```

Start with read-only inspection and a real forward pass:

```sh
m-model model inspect --model my_project.model:build_model
m-model model probe \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch
```

`inspect` shows the modules and construction locations the tool can resolve. `probe` runs one real forward on CPU and records the input/output metadata for each observed module call. It does not train the model, inspect tensor values, or prove branches that this batch did not exercise.

To discover the built-in starter models without creating project configuration or importing PyTorch, list or search the preset catalog:

```sh
m-model data presets
m-model data presets resnet
```

The catalog currently includes Tiny CNN and ResNet18 for image classification. It reports input dimensions, model factory, and weight choices; pretrained weights are never downloaded by browsing. Use `--json` for a machine-readable catalog, or run `m-model data presets --help` for command options.

To scan a class-folder dataset and verify image decoding, run `m-model data index --root ./data --verify`. In a terminal, damaged files are listed by relative path and class. Choose only the files to exclude, inspect the exclusion plan, then type `EXCLUDE`. The tool saves the content-bound protocol in `.model-maven/data-exclusions.json`; it never deletes or moves the images. The built-in classification batch applies the saved list only when both the data root and source content fingerprint match. If the dataset changes, it stops and asks you to verify it again. After repairing the files, clear the old protocol only after a fresh full verification: `m-model data index --root ./data --verify --clear-exclusions`, then type `CLEAR`. For scripts, pass `--exclude RELATIVE_PATH --apply-exclusions` explicitly; combine `--json` for a single structured report. JSON inspection never applies exclusions unless the explicit apply flags are provided.

For structured inputs, return an explicit envelope so argument binding is clear:

```python
def example_batch():
    return {
        "args": [image_tensor],
        "kwargs": {"tokens": token_tensor, "mask": attention_mask},
    }
```

An ordinary dictionary is not assumed to be keyword arguments. Use `--batch-binding kwargs` only when the model's `forward` expects those dictionary keys as named arguments.

### What the first terminal flow looks like

The following is a shortened excerpt captured from `m-model` running in a Linux PTY with a temporary `Linear(2, 2)` fixture. The temporary path is abbreviated; the prompt and evidence text are from the actual CLI, not a design mockup.

```text
Model-Maven · 开始一个模型项目
1. 接入已有 Python 模型
2. 打开一个项目目录
3. 从分类预设开始
...
选择操作: 根模型工厂入口: 可选：代表批次工厂入口:
入口处理方式：1 临时查看（不写项目配置），2 保存为默认入口:
首次查看：1 检查模型结构，2 前向查看代表批次:
正在执行 probe（CPU，最多 30 秒）…
model  Linear  参数=6
  创建位置: <临时工程>/cli_acceptance_model.py:5

本批次前向调用:
model · 第 1 次
  输入 args: [Tensor [1,2] torch.float32]
  输出 Tensor [1,2] torch.float32

前向证据 · 仅本次执行观测，不代表所有动态分支
未观测到的分支保持未知，不视为验证通过。
```

This demonstrates the prompt order and the kind of result shown. It is a tiny fixture, not a user study or proof of complex-model adoption cost. To repeat the recorded smoke flow and inspect the complete transcript, run `python3 scripts/verify_cli_interaction.py` after building the binary.

## 3. Diagnose and repair a local contract mismatch

This repository fixture has a 48-feature encoder connected to a projection configured for 32 features. The first probe is expected to fail at that boundary:

```sh
m-model model probe \
  --model examples.complex_acceptance.contract_repair:build_model \
  --batch examples.complex_acceptance.data:changing_batch
```

Keep the first edit in a named variant. Preview is non-mutating; repeat without `--dry-run` to commit it:

```sh
m-model layer set model.fusion.video_projection \
  --model examples.complex_acceptance.contract_repair:build_model \
  --batch examples.complex_acceptance.data:changing_batch \
  --param in_features=48 --dry-run

m-model layer set model.fusion.video_projection \
  --model examples.complex_acceptance.contract_repair:build_model \
  --batch examples.complex_acceptance.data:changing_batch \
  --param in_features=48

m-model --variant working model probe \
  --model examples.complex_acceptance.contract_repair:build_model \
  --batch examples.complex_acceptance.data:changing_batch
```

The persistent variant is a readable structural recipe. It does not overwrite the Python source or contain trained weights. The probe validates the edited model against the supplied batch; it does not prove all dynamic inputs or training behavior.

To add, replace, or remove registered children in supported `Sequential`, `ModuleList`, or `ModuleDict` structures, see the [Layer editing examples in README](../README.md#inspect-and-edit-a-model-variant). Use `--target source --dry-run` when you intend to modify Python source. Review the diff and commit your project to version control before applying source edits. Source editing supports a constrained set of unambiguous constructor/container forms; arbitrary graph rewrites and custom weighted Layer migration are not claimed.

## 4. Run a bounded training-step trial

Once the model and batch work, attach the training step you already own:

```sh
m-model try \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch \
  --step my_project.training:training_step \
  --steps 4 --lr 1e-4
```

The managed-step protocol expects `training_step(model, batch, context, **params)` to return a finite differentiable scalar `loss`. Model-Maven performs the backward pass and a bounded AdamW update. `try` is an experiment; model and optimizer state are not saved when it exits.

For a guided human workflow, run `m-model model supervise guide` (or provide `--model` and `--batch`). It executes a real probe, lists the observed module/port pairs, asks you to choose one and write its role and meaning, then adds one rule and runs a read-only preview. The semantic label always comes from you; unobserved branches remain unknown. The guide does not train or change a loss. `--json` and `--no-interactive` reject this prompt-driven command with a diagnostic; use the individual commands below for scripts and agents.

To bind a marked output to a tensor in the actual batch, use `--kind batch-alignment --mode require --target-path batch.kwargs.labels --axis 0`. The target path and axis are your explicit choices; the rule checks only that the selected dimensions have equal size. It does not verify label values or infer the task. For `finite`/`range`, `--axis 0` explicitly identifies the output's sample axis. Return a batch envelope with `metadata: {"stable_sample_ids": true, "sample_ids": [...]}` to preview per-sample violations. Custom training steps can instead call `context.report_sample_ids(stable_ids)` before the supervised forward. ID count must match the selected tensor axis; otherwise localization is reported as unsupported and never as a pass. Explicit IDs appear in violation evidence and may be retained in run reports; checkpoints store SHA-256 summaries for multi-batch deduplication, not the raw IDs. Counts are capped at 10,000 IDs per rule, after which reports give a lower bound. The user-provided IDs are not checked against source records, and the tool has no dataset-size denominator for an overall coverage fraction. Overall coverage remains partial until identity authenticity and task-specific label meaning are verified.

You can also use the individual commands. Review the measured ports with `model supervise preview`, then add `--apply-supervision` to a managed-step, manual-step, or protocol-aware external-loop run. Rules stay off unless you pass this flag. `require` rejects a violating step before its tracked update; a weighted `penalty` changes the loss and appears as `supervision/<rule-id>`. For `manual-step`, combine task losses and call `context.backward(total_loss)` once. For an external loop, use `context.supervised_step(batch)`, `context.backward(loss)`, and `context.register_optimizer(optimizer)` explicitly; ordinary external loops remain user-owned and unverified. See [manual semantic marks and supervision](model-contracts.md#手动语义标记实验性) for the protocol and limits.

To compare which samples each step consumed, call `context.report_sample_ids(stable_ids)` from the managed or manual step, or `context.report_sample_ids(stable_ids)` immediately before the corresponding `report_step` in an external loop. Every completed step must report stable IDs for the run comparison to show a sample identity digest. Model-Maven stores the ordered SHA-256 summary and count, not the raw IDs. This is an explicit report from your training code; the tool does not verify that the IDs match the batch contents. Missing or unstable IDs remain “information insufficient.”

If your code owns multiple optimizers, update order, data iteration, or the complete loop, use `--training-mode manual-step` or `--training-mode external-loop` instead. Those modes report which actions remain user-owned and unverified. To opt an external loop into semantic supervision, pass `--apply-supervision` and follow the explicit context/optimizer protocol in [manual semantic marks and supervision](model-contracts.md#手动语义标记实验性). Do not split a full loop into a fabricated independent batch just to satisfy a CLI field.

For a resumable run, use `train` and explicitly choose the step budget and checkpoint interval:

```sh
m-model train \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch \
  --step my_project.training:training_step \
  --steps 100 --checkpoint-every 10
```

Resume only from a run you intend to continue:

```sh
m-model runs list
m-model train --resume <run-id> --additional-steps 20
```

After changing model structure, use explicit [weight migration](weight-migration.md) to start a child run from selected parent weights. `--resume` continues the same model/run signature; it is not a structure migration.

The Python entrypoints that carry implicit state need paired `state_dict`/`load_state_dict` hooks, or an explicit stateless declaration. Resume checks the recorded source and environment fingerprints; it refuses to assume changed code or unknown state is compatible.

## 5. Evaluate a saved run and inspect predictions

Evaluation is attached to a completed training run, so the checkpoint, source run, metrics, data identity, and optional preview records remain linked in run history. In the interactive workbench, open **试跑与训练 → 浏览运行列表与详情**, select a completed training run, then choose **evaluate**. Enter your evaluator (`package.module:evaluate`) and data factory (`package.module:build_data`); split defaults to `test`. Sample limits, a data fingerprint, evaluator/data parameters, and up to 32 preview records are optional. Review the displayed plan and confirm before the evaluation starts. The evaluator must explicitly report prediction records through `context.report_prediction`; an opaque custom artifact is never guessed to be a prediction.

For scripts, use the same operation directly:

```sh
m-model evaluate \
  --run <completed-run-id> \
  --evaluator my_project.evaluation:evaluate \
  --data my_project.data:validation_data \
  --split validation \
  --preview-samples 8

m-model runs list
m-model runs show <evaluation-run-id>
```

The run details show the saved metric/provenance summary. Choose **samples** to page through only the preview records the evaluator explicitly reported; select a row to view its prediction and optional target as JSON. An absent sample count or preview remains unknown, not zero.

## 6. Save defaults only when you want a project entry

One-off commands can keep using explicit `--model` and `--batch` arguments. To reuse confirmed entrypoints in later commands and the workbench:

```sh
m-model init \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch --dry-run

# Review the preview, then save:
m-model init \
  --model my_project.model:build_model \
  --batch my_project.data:example_batch
```

`init` writes `.model-maven/project.json`; it does not rewrite your Python code or install packages. It refuses to overwrite existing project settings.

To replace the default batch while keeping the model and component bindings, preview `m-model config set-batch --batch my_project.data:batch --batch-binding envelope --batch-param 'split="train"'`, then repeat with `--apply --if-current <from_revision>`. The command checks the project revision but does not execute the factory. In the separate TUI workbench, **Assemble components → batch** runs a real forward before offering to save. Parameters are stored in local project configuration; do not put credentials there. When the project default batch is used, explicit `--batch-param` keys override matching saved keys; an explicitly different `--batch` does not inherit them. Resume and repeat use the recorded run inputs. Saved data files and Python source are not frozen by this operation.

To start a new editable multi-input model, preview a scaffold first:

```sh
m-model scaffold --input image=3,32,32 --input tokens:int64=16 --input mask:bool=16 --output-dim 4
# After reviewing the generated plan, repeat with --apply --if-plan <plan_revision>.
```

Inputs default to `float32`; `float16`, `float64`, `int64`, and `bool` are also supported. Batch inputs and targets are synthetic. Each input reaches its encoder in the declared dtype. The default fusion converts encoder outputs to `float32` for the default linear head; a custom fusion controls its own dtype contract. Replace the generated input and loss logic with project data before training.

For a standalone, readable wrapper around the validated project factory and named injection seams, run `m-model assembly generate --output generated/model_assembly.py`. Add global `--variant working` to lower that variant's active edit history as well. Supported built-in add/set/replace/remove, transactions, and explicit sharing changes for `Sequential`, `ModuleList`, and `ModuleDict` are emitted as readable Python; the active history cursor and saved initialization seeds are preserved. The generator checks the reconstructed model before writing and never overwrites an existing path.

The adapter still calls your root factory and does not serialize arbitrary Python graph internals or dynamic control flow. Registered components bound to root-factory seams can be reconstructed when their project modules remain importable; registered component references and custom Layer Edit Providers in variant Layer-edit histories fail with a structured unsupported-operation error when their graph semantics cannot be reproduced. Keep those dependencies importable in the destination environment.

To make that adapter the model entrypoint for a training run and carry it through export:

```sh
m-model train \
  --model generated.model_assembly:build_model \
  --batch my_project.data:example_batch \
  --step my_project.training:step \
  --steps 100
m-model export ../model-export
```

The run snapshots imported project modules. Export includes and hashes the generated adapter and its imported model/component files, then `restore_model.py` uses the recorded generated entrypoint to rebuild the model. Python packages and non-Python runtime dependencies remain environment requirements.

## 7. Configure and inspect data augmentation

`m-model augment` saves ordered `train` and `eval` transform steps in `.model-maven/project.json`. For example:

```sh
m-model augment add RandomResizedCrop --id crop --split train --target args.0 --param size=224 --scale 0.6,1.0
m-model augment add ToTensor --id tensor --split train --target args.0
m-model augment move tensor --before crop --split train
m-model augment list --split train
m-model augment set crop --scale 0.7,1.0 --split train
m-model data preview --batch my_project.data:example_batch --split train
```

Built-in transforms include `Resize`, `RandomResizedCrop`, `ToTensor`, `Normalize`, and `RandomHorizontalFlip`; `custom` accepts a project `module:callable`. Built-in parameters are checked before saving. Each step can name a batch target such as `args.0` or `kwargs.images`; training fails before execution if a saved training step lacks a target. Managed/manual training applies the steps automatically. An external loop and evaluator keep control of their data and must call `context.transform_batch(batch)`; if a configured pipeline is skipped, the worker returns an error. The preview command uses the saved pipeline for its split and reports observed transform ports. An explicit `--transforms FILE` takes precedence. Training checkpoint signatures include the exact saved train pipeline.

## 8. Use the interactive workbench

Run `m-model` in a terminal to open the guided workbench. On a new project, choose an existing Python model, open a project directory, or start the Tiny CNN classification preset. The preset asks for a local folder with one image subfolder per class, shows the detected classes, and only writes `classification.py` and project defaults after confirmation. It uses the images in place and probes a real batch. The generated adapter is editable; it is not a replacement for your custom architecture.

The workbench pages include model/Layer inspection and editing, bounded trial/formal training, runs, components, environment, artifacts and project configuration. Labels and available actions depend on the current project and runtime capabilities. If an action is unavailable, use `m-model doctor` and the page's recovery hint; do not infer availability from a menu label alone.

For scripts and agents, use `--json` for one structured report on stdout. Use `--no-interactive` to guarantee a command will not prompt, even in a terminal. See [README](../README.md) for component packages, export, completions, and command-level details.

## Current limits

- The current execution path is CPU-only; it is not an OS security sandbox. Model and extension Python execute with your permissions.
- Forward evidence covers supplied batches only. Dynamic branches that did not run remain unknown.
- Variant edits and source edits have different persistence and weight effects; the CLI reports their limits instead of promising arbitrary rewrites.
- The repository has automated fixtures but no completed comparison study with real users. No speed, usability, or commercial-value claim is made by this guide.

### Run a data preview without leaving the workbench

From the interactive workbench, open **Data & Preview** and configure the Dataset or batch factory. Choose `start` to run the preview in a child process. You can return to the workbench while it runs, then reopen **Data & Preview** to refresh, wait, or request cancellation. A completed request displays its actual report; a cancelled request is recorded as cancelled.

After a successful batch preview, the workbench can create a one-step managed trial from the current model plus the same batch factory and parameters. The trial calls the factory again; it does not train on the preview's temporary sample. Enter the training-step factory and confirm the displayed run plan. Dataset previews do not imply a training batch contract, so they do not offer this shortcut.
