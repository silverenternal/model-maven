# Variant preview base and conditional edit

For a variant-targeted Layer preview or edit, a compatible CLI reports
`data.variant_base` with a variant `name`, a `token`, and the algorithm
`canonical-json-sha256-v1`. The token is `absent` if the selected variant did
not exist, otherwise it is the lowercase SHA-256 of the complete parsed variant
JSON as serialized by the CLI. It is a content identifier, not the worker's
own `revision` field.
The token is not scoped to a variant name: the caller must retain and pass the
same reported name when applying it.

For a subsequent variant edit, `--if-variant-base TOKEN` acquires the existing
variant lock, reloads the variant, and compares its token while that lock is
held through execution and persistence. A mismatch returns
`VARIANT_BASE_CONFLICT` before the worker executes and does not modify the
variant. An unavailable lock, unreadable file or invalid token is an error.
A successful commit also reports `data.variant_committed` with the name,
algorithm and content token of the value saved. A subsequent `model probe`
with an explicit `--variant` can pass that token via `--if-variant-base`; the
CLI locks and reloads the variant, compares it before worker execution, and
holds the lock throughout the probe. A conflict rejects the probe. The
condition is rejected for other actions, source edits and edit dry runs.
Older CLIs do not provide this contract.

The comparison coordinates writers that honor the CLI's variant lock. It does
not protect against unrelated processes directly modifying files without that
lock, nor does it pin the model's Python source, dependencies, initialization
or batch contents. A matching token therefore does not by itself authorize a
claim that an earlier preview or its candidate forward was reproduced. The
TUI offers a confirmed conditional commit when the preview also has a matching
supported input fingerprint and verified requested mode. It warns that source,
weights and runtime are not pinned. Its subsequent guarded probe requires the
committed variant token and either regenerates the input or replays the opt-in
captured input. If that token is absent or unrecognized, the TUI reports that
the guarded probe was not run. Captured replay covers the supported input
values and structure, not the original Python objects or runtime state.
