# Model-Maven

This package provides the `m-model` CLI and the `m-model-tui` terminal workbench.
Platform-specific optional packages provide Linux x64/arm64 (glibc 2.39 or
newer), macOS x64/arm64 and Windows x64/arm64 binaries. npm selects the matching
package during installation; do not disable optional dependencies. Linux musl
(including Alpine) is not supported. Node.js 18 or newer is required to launch.

Python and a compatible PyTorch environment are separate: choose and install
the dependencies needed by your own model. The package does not install PyTorch,
GPU drivers, or Python libraries automatically. Run `m-model doctor` to inspect
the selected environment, or launch `m-model-tui` to start the workbench.

This is an early release. Model inspection and training execute project Python
code; only open projects you trust. Non-Linux platforms receive native build and
startup smoke checks; full model workflows, GPU and accessibility vary by
environment and have not been certified by those checks.
