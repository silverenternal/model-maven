# Model-Maven 0.2.0-alpha.1

Pre-release source: `24f0934d6fcc6734cb9254d9ce6ce3b91a127bf1`.

This alpha provides a terminal model-design workspace with editable graph declarations, Python generation, data bindings, supervised training and runtime inspection. It runs against an explicitly selected Python environment; it does not install a deep-learning framework into the user's project automatically. The workspace also has a Neovim/LazyVim plugin entry.

## Install from GitHub assets

Download the launcher `.tgz` and the matching platform `.tgz` from this release, then install both local files with npm. Replace `<platform>` with `linux-x64`, `linux-arm64`, `darwin-x64`, `darwin-arm64`, `win32-x64`, or `win32-arm64`:

```sh
npm install --global --omit=optional ./silverenternal-model-maven-0.2.0-alpha.1.tgz ./silverenternal-model-maven-<platform>-0.2.0-alpha.1.tgz
m-model-tui
```

Node.js 18 or later and Neovim 0.10 or later are required. Use an existing Python environment with PyTorch for PyTorch models; select it explicitly when needed. Package checksums are in `SHA256SUMS`. The npm registry itself still serves the older version; automatic platform selection from the registry is not yet available for this alpha.

## Evidence and scope

- Six platform packages built and passed npm selection and startup checks: Linux x64/ARM64, macOS x64/ARM64 and Windows x64/ARM64 ([build](https://github.com/silverenternal/model-maven/actions/runs/45848198756)).
- On each platform, the installed candidate generated a trainable two-input graph, read two real CSV rows, ran forward and backward, completed a two-step trial and training run, saved a checkpoint and resumed one additional step ([run](https://github.com/silverenternal/model-maven/actions/runs/45849511373)). The batch factory and generated step were explicitly marked stateless for this comparison. Windows ARM used an x64 Python/PyTorch runtime under emulation.
- On Windows x64 and ARM64, an attached Neovim 0.10 terminal exercised 3D preview orbit, zoom, mouse movement, window resize, closing, reopening and a draft refresh. ARM used x64 Neovim emulation. All six platform binaries completed a small installed-TUI authoring walk from an empty project through a two-input addition, save, generation and matching output ([Windows](https://github.com/silverenternal/model-maven/actions/runs/45852062590), [Linux and macOS](https://github.com/silverenternal/model-maven/actions/runs/45853616156)). The complete supervised from-zero TUI journey was not independently repeated on each native platform.
- A fixed graph and a handwritten PyTorch implementation matched on two outputs, loss, gradient and one update. A separate, simple no-parameter TUI authoring comparison matched its handwritten output. Neither comparison measures human productivity or proves commercial value. Details and limitations are in `progress.md` and the JSON evidence files in this directory.

## Known limits

- The alpha has not had human usability evaluation, a production reliability audit or evidence of market demand. Automated terminal interactions do not substitute for them.
- Windows file contents are flushed before replacement, but the product cannot make the same portable directory-fsync power-loss durability guarantee as on Unix.
- Training executes a frozen Python code snapshot. A custom data factory using `Path(__file__)` to find an adjacent CSV can fail because data files are not automatically copied into that snapshot; use an explicitly configured data path. Checkpoint resume requires the callable state hooks or an explicit stateless declaration.
- Platform coverage is bounded to the CI environments and the examples above; GPU hardware, every library, full plugin combinations, and every terminal are not covered.
- The npm registry package is still `0.1.0`; this alpha has not been published to npm.
