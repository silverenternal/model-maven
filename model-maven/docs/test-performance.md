# Test performance notes

## Current baseline

The full workspace gate passed with `cargo test --locked --quiet`. The 57-test `tests/cli.rs` integration binary took 116.35 seconds as part of that run. A later direct run of that target passed 57/57 in 106.82 seconds. These runs are not controlled benchmarks, so the difference is only a signal, not a measured speedup.

Several PyTorch-backed model/edit scenarios crossed the harness's 60-second warning threshold. Those flows launch the real CLI and isolated Python workers to verify persistence and replay, so they are high-value end-to-end checks. Do not delete them just to make the suite faster.

## Changes made

`tests/cli.rs` called `import torch` in 50 independent capability checks. `tests/train_device.rs` and `tests/train_modes.rs` had similar repeated checks. Those test processes now cache PyTorch availability with `OnceLock`; the device tests also cache GPU count. This avoids repeated interpreter starts while preserving the same skip decision for every test in that process.

## How to validate changes efficiently

During an iteration, run the directly affected integration target(s), architecture/compatibility checks as needed, and Clippy. Use `cargo test --locked --quiet` as an integrated gate after cross-module changes or before a release candidate, not after every local edit.

Next performance work should profile repeated worker startup inside the slowest scenarios and compare a small set of bounded test-thread counts under a controlled environment. Keep behavior and failure-path coverage; remove a test only after proving its assertions are covered elsewhere.

An exploratory direct run with `--test-threads=4` was still running after two minutes and was interrupted; it did not produce a valid result and should not be treated as a faster setting. The bottleneck is likely dominated by end-to-end subprocess/worker work rather than only excess harness concurrency.
