# Optional ML/RL libraries and explicit environments

Model-Maven keeps the project-selected Python interpreter as the source of truth. To inspect common deep-learning and reinforcement-learning packages in that environment, run:

```sh
m-model env libraries
m-model env libraries --library transformers --library gymnasium
m-model --python .venv/bin/python env libraries --library stable-baselines3
m-model env probe --library gymnasium
m-model --python .venv/bin/python env probe --library torch --library timm
m-model env probe --library torch --check-runtime
m-model env probe --library jax --library gymnasium --check-runtime
```

On Windows, use the selected environment's `Scripts/python.exe` path. Without `--python`, the command follows the project's saved environment selection when that virtual environment exists; otherwise it inspects the CLI's base Python. `--venv` can select a project-relative virtual environment explicitly.

The command reads Python distribution metadata only. It does not import third-party packages, run their entry points, install dependencies, change the selected interpreter, or choose an adapter. Its `installed` result means that package metadata is present; it does not prove the package can be imported or initialized on this machine.

`m-model env probe` is a separate, explicit action. It requires one or more `--library` IDs and imports each named candidate in the selected interpreter, so the package's initialization code runs with the current user's permissions. It reports `imported`, `not_installed`, or `import_failed`, including the detected distribution/import version and a sanitized failure type/reason. Probe is allowlisted to the common IDs shown by `m-model env libraries`; arbitrary distributions remain metadata-only. Probe never installs dependencies, changes interpreters, or binds a Trainer/backend. A successful import only proves that import initialization succeeded in that environment.

Add `--check-runtime` when you want a bounded, first-party CPU smoke in addition to import. Current profiles exercise CPU tensor arithmetic for PyTorch, a native NMS operation for torchvision, CPU array arithmetic for JAX, and a bounded Box sample for Gymnasium. Each selected library runs in its own subprocess; at most four libraries can be checked in one command. Other allowlisted libraries report `runtime_check.status: unsupported`, rather than implying they passed. Import and runtime results are separate, so a library can import successfully while its smoke fails. Runtime checks execute third-party code with the selected interpreter's permissions; they are diagnostics, not a sandbox, GPU certification, or evidence that Model-Maven has bound an execution adapter. Recorded live-package acceptance covers PyTorch 2.15.0 CPU and Gymnasium 1.4.0 CPU (explicit package probe and a user-owned two-step CartPole loop). torchvision and JAX branches have fixture-path tests, not real-installation compatibility evidence. These checks do not certify other versions, tasks, hardware or native RL Trainer integration.

Project setup stays isolated by default. If the selected base Python already contains a large ML runtime that should be reused, explicitly run `m-model env configure --system-site-packages` before `m-model env setup`. The project lock records inherited package versions and the base environment's existing `pip check` findings; checks detect inherited drift and new conflicts. Inherited packages are not copied into a wheelhouse, so another machine needs a compatible base Python runtime.

The first-party `pytorch-native` path supports the Model-Maven operations listed in `m-model doctor`. Other packages such as torchvision, timm, Lightning, Transformers, Diffusers, JAX/Flax, Gymnasium, Stable-Baselines3 and Ray/RLlib are reported as package-only unless a dedicated adapter is explicitly implemented. Presence of `lightning` does not start Lightning Trainer, presence of `ray` does not start RLlib, and presence of a framework never changes the current backend.

User model/data/training modules can import libraries installed in the Python environment selected for that command. To install or lock dependencies, add them to the project's requirements and explicitly run `m-model env lock`/`setup`. These commands resolve only the packages the user declared and their dependency closure; a requirements file containing no Torch is valid and can prepare an environment for user-managed JAX, Gymnasium, or other tooling. This does not bind that library as a Model-Maven backend. Model-Maven's model and training operations still require importable PyTorch and fail clearly when the selected interpreter has no Torch.

`env lock` and `env setup` may create an explicitly requested project environment and install its hash-pinned wheels. They do not install extra packages, select a framework, switch the interpreter implicitly, or infer a Trainer from installed package metadata. `m-model` currently executes native `torch.nn.Module` models and its documented training modes; it does not claim generic JAX/Flax, HF Trainer, Lightning Trainer, Diffusers, or RLlib execution support.

An arbitrary `--library DISTRIBUTION` query is metadata-only and does not imply first-party adapter support. Unknown packages are returned with `adapter_status: unknown`; no execution behavior is inferred.
