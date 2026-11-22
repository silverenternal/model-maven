# Bound forward-input content fingerprints

The compatible CLI worker reports `batch_content` for a newly executed probe,
including a failed forward, and `validation.batch_content` for a newly
executed candidate preview. `sha256-forward-input-v1` fingerprints the bound
`args` and `kwargs` before the forward, including container types and order,
scalar values, tensor dtype, shape, stride, gradient flag, bytes and repeated
object references. Candidate validation also reports `forward_evidence` as
`executed` or `reused`. Reused cached evidence does not acquire a fingerprint
of a newly executed forward.

Coverage is deliberately limited: exact built-in Python tuple/list/dict,
string-keyed dictionaries, basic scalar values, and ordinary contiguous CPU
tensors are supported. Unsupported tensor devices/layouts, tensor subclasses,
distinct tensors sharing storage, unsupported objects, or the 64 MiB, 4,096
item and 32-level limits produce `status: unsupported` without a digest. The
worker does not persist input values; a digest can still reveal equality and
should be treated as potentially sensitive metadata. Hashing supported input
content has a bounded memory and runtime cost.

The TUI compares only complete, recognized digests from the failed diagnosis
and a successfully executed candidate forward. A match is evidence that the
supported bound input content and represented structure match, not proof of
the same sample identity, original object, model source, weights, random state,
dependencies or runtime behavior. Different, missing, unsupported, reused or
unrecognized evidence never becomes a match.

The compatible CLI also accepts `--if-input-content <digest>` on `model probe`
and on a variant-targeted layer edit with `--require-forward`. It regenerates
and binds the input, then checks its supported content before the candidate
forward. A mismatch or unsupported input rejects the operation; a guarded edit
does not save the variant. Guarded operations bypass evidence-cache reuse.
`--if-variant-base` can separately guard the variant's persisted base. Neither
guard saves or replays the original batch or pins model source, weights,
dependencies, random state or execution mode. The content check describes the
input at the time it is computed; user code can have side effects or mutate
objects later. When a preview has matching supported input fingerprints, a
recognized variant base, an explicitly selected model and batch, and a
matching reported execution mode, the TUI offers a default-cancel confirmation
for a guarded variant commit. After an acknowledged commit it runs a guarded
probe using regenerated inputs, or the opt-in captured input, and the returned
committed variant token. The
CLI checks the token under the variant lock and retains that lock during the
probe; its result must be read separately from the commit result. The lock only
coordinates cooperating writers. After any attempted commit the TUI discards session evidence
because a failed or malformed response does not prove that no write occurred.
It separately retains a session receipt containing the displayed commit and
recheck results, identities, and the preceding preview. The receipt is
historical rather than a current-state check; it survives a new inspection,
probe or preview, and is cleared when changing projects or selecting a variant.
The full-screen receipt can reopen the historical preview difference at its
previous diagram position. It separately labels the commit acknowledgement
and the reported recheck outcome. A reported success is shown as unverified
when the recheck's input fingerprint, requested mode, requested snapshot
replay or fresh-forward evidence cannot be matched. Truncated or unreported
call coverage is labelled separately from a complete recorded trace. A `FORWARD_FAILED` diagnostic is distinguished from
guard rejection or an unavailable report. The evidence view includes the
session receipt with its model, batch, variant and guard identifiers and the
displayed commit and recheck output. A preview or commit response with an
unexpected `operation` is not accepted as authorization or acknowledgement;
because a write could already have happened, the latter remains unconfirmed.
The receipt also records the CLI's self-reported version queried after the
operations, or marks it unavailable. That later query does not pin or prove
the executable used for any preceding operation. Diagnosis, preview, commit,
and recheck JSON responses are shown in the receipt when available, wrapped
for display. Each report is limited to the first 65,536 UTF-8 bytes at a
character boundary; the receipt explicitly labels truncated or unavailable
reports. It is a session record and is not saved as a permanent audit log.
The opt-in input capture and replay facility is described in
[input-snapshot-contract.md](input-snapshot-contract.md).
