# Run artifacts

External-loop training can keep files in its private run directory and register them in the run artifact catalog. The worker does not infer ownership from a filename or extension.

```python
def run_loop(model, context):
    weights = context.artifact_dir / "best-weights.bin"
    weights.write_bytes(export_weights(model))

    context.register_artifact(
        "best-weights",
        ["best-weights.bin"],
        owner="my_project.trainer.v1",
        references=[],
        consumes=[],
        complete=True,
        kind="checkpoint",
    )

    for batch in batches():
        # update model and report completed work
        if not context.report_step(loss_for(batch), {"updates": 1}):
            break

run_loop.MODEL_MAVEN_STATELESS = True
```

`context.artifact_dir` is created under the current run. `files` are paths relative to that directory; registration rejects absolute paths, traversal, symlinks, missing files, duplicate paths, and duplicate artifact IDs. At run finalization, Rust computes size and SHA-256 from the actual files. A declaration is marked complete only when the run completes successfully and the producer declared it complete.

Files written under `artifacts/` but omitted from every registration remain outside the managed catalog. Cleanup does not infer ownership from `.pt`, `.bin`, or another file extension, and does not release those files with a registered checkpoint.

`references` remains a legacy producer-supplied list of opaque strings. Any non-empty list conservatively keeps that checkpoint set out of cleanup. New consumers should use `consumes`, a list of typed `{run_id, artifact_id}` identities. The project resolver verifies that each producer identity exists and builds the reverse consumer index before cleanup. Resuming a run automatically records consumption of its parent's `training-checkpoint` when that checkpoint is registered.

Cleanup fails closed when any project run is active, has an incomplete/legacy artifact manifest, or contains an invalid or dangling typed identity. This keeps old projects and unfinished records protected until their artifact graph is trustworthy. The resolver covers registered run artifacts; it cannot discover arbitrary files copied outside Model-Maven or consumers that were never declared.

`kind="checkpoint"` makes a complete file set eligible for explicit release only when no typed project consumer or legacy reference protects it. Preview one selected set with:

```sh
m-model runs clean --run <run-id> --include-checkpoints --artifact best-weights --dry-run
```

Inspect the fixed plan and apply it only after reviewing the lost recovery capability:

```sh
m-model runs clean --apply <plan-id> --release-checkpoints
```

Inspect the resulting catalog with `m-model runs show <run-id>` and the disk usage with `m-model runs storage <run-id>`. If a declaration cannot be verified, the run still finishes; its artifact manifest is marked incomplete and unregistered files remain protected.

The fixed plan revalidates the complete file set, owner, run-local references, hashes, run state, keep flag, project consumer graph, and hard-link count. A changed member skips the entire set. Successful release records an event and marks each file unavailable while preserving the run report and metrics. A partial filesystem failure is reported and the artifact is marked incomplete; remaining files are not eligible for another release. Hard-linked files are deduplicated in storage totals and excluded from cleanup plans so an external path cannot lose a shared file through run cleanup. Hard-link checks detect shared inodes only; they do not find independent copies or undeclared consumers.
