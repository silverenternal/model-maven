# Model-Maven engineering preview

This package is an engineering preview. Install its launcher and native binary
package from the same release; the launcher checks their version, workspace
identity and recorded source revisions. Review the release's platform support
statement before installing: a workflow configuration alone is not evidence
that a platform has passed native use.

The workspace requires Neovim 0.10 or newer. Select a local Python environment
with the libraries your project needs; Model-Maven does not install or bind a
particular PyTorch, reinforcement-learning library, accelerator driver, dataset
or model weight. Project Python code runs with the selected environment's
permissions; inspection and training are not a sandbox for untrusted code.

You can construct graphs from operators, inspect supported existing models,
connect data, declare objectives and supervision, and run supported trials.
A successful observation covers its actual input, path and environment; it is
not proof for all branches, datasets or training conditions. The existing-model
preview labels unobserved connections as unknown. Unsupported structural
changes may require editing the source, then rerunning validation.

Before upgrading or rolling back, preserve the project configuration, source,
checkpoints and data. Run/checkpoint compatibility is versioned independently of
the package version and depends on the selected runtime and user code. An
operation whose outcome is uncertain should be inspected before retrying.
