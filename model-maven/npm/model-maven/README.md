# Model-Maven

This package provides the `m-model` CLI and the `m-model-tui` terminal workbench.
The launcher is designed to use platform-specific optional packages for Linux
x64/arm64 (glibc 2.39 or newer), macOS x64/arm64 and Windows x64/arm64. These
are configured targets, not a guarantee that matching packages have been built,
published or validated for every release. Where a matching package is available,
npm selects it during installation; do not disable optional dependencies. Linux musl
(including Alpine) is not supported. Node.js 18 or newer is required to launch.
For newer platform packages, the launcher reports the selected binary's actual
glibc requirement when the system is too old. Requirements can differ between
builds and commands; check the package manifest for the recorded values.

Python and a compatible PyTorch environment are separate: choose and install
the dependencies needed by your own model. The package does not install PyTorch,
GPU drivers, or Python libraries automatically. Run `m-model doctor` to inspect
the selected environment, or launch `m-model-tui` to start the workbench.

The upcoming unified workspace requires Neovim 0.10 or newer. The npm launcher
package contains the same workspace at `nvim/` for an existing Neovim session;
`m-model-tui` embeds it for standalone launch. Run `m-model-tui --workspace-plugin-dir` to print the plugin directory for
this npm installation. The command reports a path only; it does not require a
platform binary or modify Neovim. With lazy.nvim or LazyVim, an example is:

```lua
local directory = vim.trim(vim.fn.system({ "m-model-tui", "--workspace-plugin-dir" }))
if vim.v.shell_error ~= 0 or vim.fn.isdirectory(directory) ~= 1 then
  vim.notify("Model-Maven npm plugin directory is unavailable", vim.log.levels.WARN)
  return {}
end
return {
  {
    dir = directory,
    main = "model_maven",
    cmd = "ModelMaven",
    opts = { cli = "m-model" },
  },
}
```

Open a project with `:ModelMaven`. For a local installation, ensure its `node_modules/.bin` is on Neovim’s PATH, or specify the local launcher directly. The path-printing option belongs to the npm launcher; a standalone native binary does not expose it. This describes the next
workspace release candidate; it is not a claim about the already published
0.1.0 package. The user's Neovim configuration and Python environment are not
changed automatically.

For the engineering-preview scope and known platform limits, see
[`docs/engineering-preview.md`](docs/engineering-preview.md). The two-input
starter walkthrough is in `examples/from_scratch_walkthrough.md`.

### Installing a candidate alongside an existing version

Only use a version that has actually been published. The following installs it
in a separate writable prefix instead of replacing a global installation; set
`MM_RELEASE` to that exact version first:

```sh
npm install --prefix "$HOME/.local/model-maven-candidate" "@silverenternal/model-maven@$MM_RELEASE"
"$HOME/.local/model-maven-candidate/node_modules/.bin/m-model" --version
"$HOME/.local/model-maven-candidate/node_modules/.bin/m-model-tui" --workspace-plugin-dir
```

To launch this candidate, put that prefix's `node_modules/.bin` first on PATH,
or set `MODEL_MAVEN_CLI` to its `m-model` and invoke its `m-model-tui` by full
path. For an existing Neovim session, obtain the candidate plugin directory
using that same launcher's `--workspace-plugin-dir` and configure the plugin's
`cli` option to the candidate `m-model` path. Close and reopen the workspace
after changing the selected CLI or plugin. A local prefix is not automatically
added to PATH. A private Linux x64 candidate has been installed alongside the
published 0.1.0 package for a local walkthrough; this does not establish support
for other platforms or a published upgrade.

For rollback, select the old launcher, matching platform package and plugin
together, and reopen the workspace; do not assume the old version supports
configuration, run records or checkpoints written by the new one. Keep a
backup of project files and `.model-maven/` before operations that change
them. Switching packages alone does not restore those files or the Python
environment. The published 0.1.0 package does not contain this upcoming
unified workspace. In a local walkthrough, its CLI did not honor the newer
project module-search directory setting and failed to import a model located
only under `src`. If another module with that name is visible, the old runtime
may select it instead. Review and explicitly configure the old environment and
entrypoints before running or training after rollback.

The candidate launcher package also includes `docs/unified-workspace-guide.md`,
`docs/workspace-journey.md`, `docs/existing-project-adaptation.md`, and
copyable examples under `examples/`. The `complex_acceptance` example combines
multiple inputs, a shared layer, and a conditional branch; its README explains
how to copy it into a new project. The examples use synthetic inputs and require
a user-provided Python and PyTorch environment. They have not yet completed the
manual walkthrough for this candidate and do not establish production readiness.
Run `m-model-tui --examples-dir` to locate the installed examples, then copy the
needed files into a writable project; the command reports a path without
modifying files and is available through the npm launcher, not the native binary.
The included `generic_loop.py`, toy `bandit_loop.py` and optional
`gymnasium_loop.py` demonstrate the framework-neutral user-loop interface. A
local Gymnasium FrozenLake walkthrough covered episode-boundary recovery; other
reinforcement-learning libraries and environment state remain the user's
responsibility. Install any optional library yourself in the Python you select.
`tensorboard_loop.py` demonstrates writing local experiment events from a
user-owned loop when TensorBoard is installed separately in that environment;
it uses synthetic metrics and does not automatically bind an experiment tool.

For distributors: `scripts/prepare_npm_bundle.py` discovers the workspace's Lua
modules and plugin files and records a workspace identity
in every staged package's `.manifest.json` and package metadata; the npm launcher checks its declared version and identity against the selected platform package. Platform staging also compares the
native `m-model-tui` embedded workspace byte for byte with the selected TUI
source. Before combining or publishing artifacts, compare the `workspace.digest`
and `workspace.algorithm` in the launcher and every platform manifest; all must
match. This identity check does not establish runtime compatibility or complete
the manual release review.

This is an early release. Model inspection and training execute project Python
code; only open projects you trust. Build and startup checks are configured for
several non-Linux platforms, but a workflow configuration is not evidence that
an artifact has passed them. Full model workflows, GPU and accessibility vary
by environment and must be assessed for the exact release and platform.
