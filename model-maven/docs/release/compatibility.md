# Release candidate compatibility

This is a review artifact, not a stable or generally supported release.

## Current artifact target

| Artifact | Built for | Runtime requirements |
| --- | --- | --- |
| `m-model` CLI | Linux x86_64, glibc-based Ubuntu 24.04 CI | Python 3 with `venv` and `pip` for worker operations; model work also needs a user-managed compatible PyTorch environment |
| `model-maven-sdk` wheel | Python 3.10+ | Pure Python, optional, no runtime dependencies |

Only the Linux x86_64 artifact is currently built by the release-candidate
workflow. Other Linux distributions, macOS, Windows, ARM, GPU/accelerator
drivers, and non-CPU training are not certified by this artifact. Rust 1.88+
is required when installing from source; users of the review binary do not need
Rust.

The CI jobs prepare a separate Python 3.12 verification environment with CPU
PyTorch 2.6.0 and SDK build tools. These are CI test dependencies, not bundled
runtime libraries or a default environment imposed on model projects. The
workflow pins direct test dependencies; it does not lock every transitive wheel.

The CLI embeds the worker sources, so running `m-model --version` and `m-model
--help` does not require a checkout. Python and PyTorch are separate runtime
requirements and are not bundled. The optional SDK wheel is delivered alongside
the CLI artifact and is not required for ordinary model use.

## Compatibility policy

Worker protocol and persisted state formats are versioned independently. The
CLI rejects unsupported worker/state versions rather than guessing or silently
migrating data. No stable CLI, configuration, extension, or state-format
compatibility guarantee is made for this pre-release snapshot. Check the
project's `docs/contracts/compatibility.json` before upgrading and keep a copy
of project configuration and run artifacts before trying a newer snapshot.

The binary can read/write user project files, model code and checkpoints. It is
not a security sandbox for untrusted Python code. Do not point it at untrusted
projects as though inspection or training were isolated.
