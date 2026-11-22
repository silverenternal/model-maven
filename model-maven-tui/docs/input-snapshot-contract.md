# Forward input snapshots

A compatible CLI can opt in to capturing a freshly bound diagnostic input with
`--capture-input-to FILE model probe ...`. The file contains actual input
values, encoded rather than encrypted. The caller chooses its location and is
responsible for cleanup. The worker creates a new file exclusively, without
overwriting an existing path, and requests mode `0600` on Unix. Final-component
symlink protection depends on the platform's `O_NOFOLLOW` support; callers must
use a private directory. In the TUI, enter `:snapshot` in the batch form and
select `on` to opt in. The TUI creates a private temporary directory and uses
its snapshot for candidate validation, commit and recheck. Replacing the
diagnosis, switching project or variant, committing, or exiting normally
releases the directory. Abrupt termination can leave input data behind; the
interface states this before capture. Cleanup is best effort and is not secure
erasure. If the CLI does not advertise both snapshot options, the TUI reports
that replay is unavailable rather than running the requested capture silently.
On the tested CLI, a system temporary directory inside the project source tree
may be mistaken for changed source; use a temporary directory outside the project.

Reports use `batch_snapshot.status` of `captured` or `unsupported`; an
unsupported input is still diagnosed but no snapshot is written. Capture occurs
before the actual forward, including when that forward fails. Snapshot capture
does not reuse cached probe evidence. File creation failure rejects the probe
before the explicit model forward.

`--replay-input-from FILE --if-input-content DIGEST` replays bound arguments
without calling the batch factory for a probe or a variant-targeted layer edit
with `--require-forward`. The digest must come from the original diagnosis.
The data-only format loads no arbitrary Python objects and verifies the restored
input fingerprint before forwarding. Missing, malformed, tampered or mismatched
snapshots are rejected; candidate edits do not commit on rejection. Replayed
reports use `batch_snapshot.status: replayed`, or
`validation.batch_snapshot.status` for a candidate. A guarded recheck can also
specify the committed variant's `--if-variant-base` token.

The format supports ordinary contiguous CPU tensors and bounded built-in
containers/scalars, including supported repeated references. Tensor views,
non-leaf, pinned, named and inference tensors, cyclic containers, unsupported
layouts or devices, and inputs over the codec's bounds are not
captured. A reconstructed input matches the stored bound values and represented
structure; it is a new object and does not pin source, weights, execution mode,
random state, external state or dependencies. This is not an encrypted data
store or a guarantee about arbitrary model code after the input is handed to it.
