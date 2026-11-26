# CLI interface discovery v1

`m-model --json --no-interactive --describe-interface` returns one ordinary
protocol-v1 JSON report on stdout without starting Python or loading project code.
`data.interface_version` describes this catalog, `data.cli_version` identifies
the package version, and `data.report_protocol_version` describes the report
envelope. `data.capabilities` advertises boolean capabilities including
`input_snapshot.capture_replay`. It describes the CLI interface, not whether
the selected Python environment or a particular model can perform an operation.
Unknown fields may be ignored; missing or invalid required catalog fields are
protocol errors. A missing or false optional capability means it was not
advertised, while a present non-boolean value is a protocol error. An older CLI
that rejects `--describe-interface` does not prove support for new guarded
operations. Future capabilities are added explicitly as their contracts are
implemented.

For a guarded `model probe --contract FILE --if-contract-sha256 SHA256`,
`model.contract_identity_v1` and `model.contract_preload_v1` advertise the CLI
interface; the worker separately negotiates both features before the operation.
The CLI checks the selected bytes before starting the worker, then the worker
checks and loads them before importing the model or input factory. Python
startup hooks may execute before the worker receives the request. The result's
`contract_report.contract_content_sha256` and `contract_identity_check`
describe the bytes used and whether a condition was checked; older reports may
omit them. A fresh-forward request is distinct from a reused probe report.

For protocol v1, a successful report includes a non-null `data` value and a
nonempty `operation`. The known statuses are `success`, `error`, and
`inconclusive`; unknown statuses, a missing success payload, invalid JSON,
multiple stdout documents, and an exit status inconsistent with the report are
protocol errors. Clients must not reinterpret these as a model failure. Error
reports from older CLIs may omit `data`; their structured diagnostics remain
available. Unknown fields are allowed. stderr can contain logs independently of
the single JSON report on stdout.

The TUI requires advertised base capabilities before project classification or
custom scaffold creation, conditional batch and augmentation changes, and
single-seam assembly changes. Scaffold routes additionally require their route
capability. A missing, legacy or malformed catalog stops these guarded actions;
no write is attempted. This includes basic scaffolding on catalog-less CLIs,
which older TUI builds attempted; compatible read-only inspection and the
separately guarded source-history path remain available where supported.

`project.classification_create` advertises `create --preset tiny-cnn|resnet18
--data DIRECTORY`. Its default mode returns a non-writing plan and revision;
`--apply --if-plan REVISION` recomputes the plan before creating new files.
The revision includes class names and generated source, not image contents;
the data directory is not locked between preflight and creation. A successful
creation does not imply a successful forward or training step.

`project.custom_scaffold` advertises `scaffold --input NAME=DIM,DIM` (repeat
for each named input) and optional `--output-dim FEATURES`. Shapes exclude the
batch dimension. `project.scaffold_runtime_dtype_v1` allows optional `:dtype`
to name a public PyTorch dtype; planning validates only the name's syntax, and
the selected runtime resolves it when the synthetic batch runs. A dimension
`?N` is variable with representative length N, for
example `tokens:int64=?8`. The synthetic batch defaults to two; generated
`batch` parameters expose `batch_size` and, for each variable axis, a name
such as `tokens_dim1` (axes are numbered from one, excluding the batch axis).
For an input with variable axes, the starter preserves floating and complex
encoder output precision, converts non-floating output to the PyTorch default
floating dtype, and averages those axes with dimensions retained; then it flattens each output,
concatenates the results, and applies a fixed-width linear head. Custom encoders
must preserve those axes and compatible remaining feature widths, or the author
must edit the generated model. Averaging is a starter behavior, not an inference
of task semantics. Each `<NAME>_encoder`, `fusion`, and `head` is an
explicit Python factory injection seam. The generated sample step uses a
synthetic target and is only a runnable starting point. Preview is non-writing;
commit requires `--apply --if-plan REVISION` and refuses existing source or
project configuration. Replacing inputs, loss, model behavior, or training logic
remains the project author's responsibility.

`data.csv_batch_template` advertises `data csv-template`: preview returns an
editable `model_csv.py` file plan and revision; `--apply --if-plan REVISION`
creates the new file without overwriting an existing one. Its `model_csv:batch`
callable accepts a CSV path, a mapping of input names to numeric column lists,
a target column list or named target mapping, and a batch size. It produces envelope batches in file order, cycles at the end, and exposes checkpoint
hooks for its cursor and a content-and-mapping fingerprint. File or mapping
changes cause resume to fail. It does not shuffle, split data, infer labels or
support variable-length fields. The bounded starter loads selected
values into memory; CSV is limited to 16 MiB, 100,000 rows, 256 selected columns
or column/type conversions and two million selected values. It is user-owned Python source after creation.
Previewing the template does not load the data or configure the project.

`data.csv_batch_typed` adds fixed-shape typed fields. A simple column list is a
float32 vector as before; a field may instead be an object such as
`{"columns":["p0","p1","p2","p3"],"shape":[2,2],"dtype":"float32"}`.
`data.csv_batch_runtime_dtype_v1` resolves real floating, integer and boolean
dtype names in the selected PyTorch runtime; individual tensor operations can
still be unsupported. Complex and quantized formats require user-provided
parsing. Dimensions must multiply to the column count. For a single scalar target use a `$tensor` wrapper,
for example `{"$tensor":{"columns":["label"],"shape":[],"dtype":"int64"}}`.
For named outputs, each target value can be a list or a typed field object.
Integers parse directly with range checks; booleans accept 0/1/true/false. A
shape or conversion failure occurs before the cursor advances. The default
starter step still uses synthetic-style mean squared error and must be replaced
for classification labels or other task-specific objectives.

`project.scaffold_named_outputs` adds repeatable `scaffold --output NAME=FEATURES`,
for example `--output score=1 --output classes=3`. It is mutually exclusive
with `--output-dim`; a maximum of eight unique names is supported. The starter
returns a dictionary from its default head, creates synthetic targets for each
name, and supplies an illustrative step that sums equally weighted mean-squared
losses while reporting each named loss. This is a runnable placeholder, not a
choice of task-appropriate objectives or labels. A custom `head` injection must
return the declared mapping or the author must replace the example step. Output
names and dimensions are part of the plan revision; no existing project source
or configuration is overwritten.

`project.scaffold_objectives` adds a cross-entropy example objective. For a
single output, `--output-dim 3 --loss cross_entropy` creates three logits and
an integer target per sample. Named heads can declare objectives individually,
e.g. `--output score=1:mse --output classes=3:ce`; omitted objective means MSE.
Cross entropy requires at least two classes and integer labels in `[0, classes)`;
no softmax is applied before the PyTorch loss. By default the generated multi-head step
sums the example losses with equal weight and reports them by name. Synthetic
labels are placeholders, and the project author controls real target mapping,
metrics, masks and custom step behavior. Objective declarations
are included in the preview revision and never override existing source.

`project.scaffold_objective_weights` adds a positive finite per-head weight,
for example `--output score=1:mse@0.25 --output classes=3:ce@2`. An omitted
weight is 1. The optimized total is the sum of weighted losses; the named
losses reported separately are raw, unweighted values. The plan includes each
weight and a change invalidates the confirmed revision. The generated Python
remains editable; these fixed weights do not implement adaptive scheduling,
masking, task-specific normalization or custom metrics.

`project.scaffold_input_routes` adds repeatable `scaffold --route
ENCODER=INPUT`, for example `--input a=4 --input b=2 --route a_encoder=b`.
Each encoder defaults to its same-named input; an explicit route replaces that
edge, and multiple encoders may consume the same input. The plan lists all
routes and hashes them into its revision. The starter's default head width is
computed from the routed representative input shapes; custom encoders must
preserve the assumed flattened widths or the user must replace the head.
This expresses only input-to-encoder edges in generated starters, not an
arbitrary DAG. Unused declared inputs are still present in the starter batch
and are reported by the TUI preview.

`project.scaffold_route_edit` adds `assembly route SLOT INPUT --dry-run` for
recent generated starters that accept `input_routes`. It validates the candidate
using the saved model, bindings and representative batch, without writing.
Commit uses `--if-config-revision` from the preview, repeats the forward and
conditionally writes `model_params.input_routes` to project configuration.
Model parameters follow the configured model; explicit same-model parameters
override individual project keys. `config restore --field model_params` can
restore an earlier configuration. Old starter files without this factory
parameter are rejected by actual construction; their code is never rewritten.
`project.scaffold_route_evidence` adds optional `--if-source-revision` and
`--if-input-sha256` checks using observed project sources and the representative
forward input. The TUI uses source evidence when available and lets users choose
whether to require identical input content. External code, environment, targets
and changes after checks are not pinned. Existing variants and past runs retain
their own snapshots.

`project.scaffold_setting_snapshot` extends route and encoder-output previews
with `--capture-input-to`; a guarded commit may use `--replay-input-from` only
with the matching `--if-input-sha256`. The TUI offers `route+snapshot` and
`encoder-output+snapshot`, creates a private temporary file, and reports its
cleanup on success, cancellation or handled errors. The snapshot contains
forward input values but not targets; a crash can leave it on disk. User code,
future batches and external dependencies remain outside this snapshot.

`assembly add|set|remove --if-config-revision REVISION` conditionally edits
the project configuration using the `base_config_revision` returned by an
earlier `--dry-run` report. The command checks that revision before validation
and uses a conditional configuration update when saving. This protects project
configuration, not Python source, weights, external resources, or runtime
identity; validation reruns and any unpinned changes remain outside this guard.
`assembly show --config-only` reads saved project bindings without loading or
validating the model or its dependencies; both construction and forward are
reported as `not_run`. It can be used to begin repairing an assembly that no
longer constructs or runs, but does not prove that any binding is usable.

`augmentation.conditional_config_edit` advertises `augment add|set|move|remove`
support for `--dry-run` and `--if-config-revision REVISION`. A successful dry
run reports candidate steps, `committed: false`, `base_config_revision`, and
configuration validation; no transform code or data is executed. Commit checks
the supplied revision and uses a conditional update, returning
`PROJECT_CONFIG_CONFLICT` on a mismatch. The revision covers project config,
not transform source, runtime, dataset contents, or external resources.

`training.conditional_config_start` advertises `try` and new `train` support for
`--if-config-revision REVISION`. The CLI checks the project configuration used
for default inputs and rechecks the configuration read for training augmentation
before creating a run. A mismatch or deleted configuration returns
`PROJECT_CONFIG_CONFLICT`. This condition is unavailable for repeat, resume and
migration, which use prior-run snapshots. It does not pin source files, runtime,
variant, dataset contents or separately stored supervision rules.
Before a new TUI trial or training launch, the TUI checks this capability and,
when applicable, the supervision and selected-variant capabilities. Missing,
legacy or malformed catalogs stop that guarded launch with the specific reason.
If request-identity capability discovery itself becomes malformed after the
plan is confirmed, the TUI also stops rather than silently downgrading.

`supervision.conditional_variant_mark_v1` declares that `model mark` can pin an explicitly named variant with `--if-variant-base`; the variant lock is held during observation and mark creation. `supervision.conditional_variant_retag_v1` adds the same variant guard to a retag, together with the semantic-state and observed-anchor guards. `supervision.conditional_variant_preview_v1` similarly pins the selected variant for `model supervise preview`; it does not cover rule edits or other supervision operations.

`training.detached_cli_v1` is true only when this CLI was built for Linux; it declares the CLI background supervisor path, not the selected Python worker or device capabilities. The workspace selects background starts and resumes only when this field is true. A false or absent field selects foreground execution with its own pending-operation handling; the confirmation states the selected mode. Runtime readiness still checks the actual worker.

`training.request_identity` advertises `train --request-id ID`, including
detached starts and resumes. IDs contain 16–64 ASCII letters, digits or hyphens
and must not contain secrets. The ID is recorded in the run and exposed by
`runs list` and detached launch receipts. Under the project run-creation lock,
a retained record with the same ID prevents another run; `RUN_REQUEST_ID_EXISTS`
includes the prior run ID. The TUI uses a fresh random ID to find a single
matching run after a lost receipt and never automatically retries. No match does
not prove that a delayed launch failed, and removal of a run record also removes
this duplicate protection. Older CLIs fall back to a non-attributing run list.
`runs.lookup_request_id` advertises `runs find-request ID`. It scans run
metadata without reading checkpoints or progress, returns at most 64 matches,
and labels status as recorded rather than live. `complete` is false if records
are unreadable, larger than 1 MiB, or matches were truncated. A missing match
never proves that a delayed start failed. TUI pending-start recovery remains
available when the full run list cannot be read; malformed or unavailable
lookup results remain unknown.
For supporting CLIs the TUI creates a project-local pending-start marker before
launch, flushing the file and, on Unix, its containing directories. Power-loss
durability on other platforms has not been verified. A confirmed receipt clears it after display; an interrupted or
uncertain request remains visible in Training runs across TUI restarts. The TUI
blocks another training start or resume until the user inspects and explicitly
forgets pending markers. Forgetting does not cancel a delayed or running job;
missing run records do not prove the request did not execute. The marker contains
only the generated request identity, not the training inputs.

`training.conditional_supervision` advertises `--if-supervision-revision`
for supervised `try` and `train`. `model supervise rule list` returns a
`state_revision` for the complete saved semantic-mark state, including rule
parameters. The worker reads that state once and compares its digest when it
binds supervision; a mismatch returns `SUPERVISION_STATE_CONFLICT` before
the training step or loop executes. This condition is separate from project
configuration and does not pin Python source, data, variant, plugin code or
the runtime. Model construction can execute user code before this check. A
detached launch receipt does not guarantee the check passed; inspect the run
status. A TUI unable to obtain this revision must not silently start a
supervised run under an unguarded confirmation.

`training.conditional_variant_start` advertises `--if-variant-base` for new
`try` and `train` with an explicit `--variant`. `variant show` returns the
canonical unredacted variant's SHA-256 as `variant_base.token` alongside the
redacted variant shown to users. The CLI reads and compares the actual variant
under the cooperative variant lock, copies the checked snapshot into the
worker request, then releases the lock before executing or launching training.
This protects the selected variant value, not project source, data, external
dependencies, or non-cooperating writers. Repeat, resume, and migration use
historical snapshots and cannot use this current-variant condition.

`training.data_fingerprint` advertises `try|train --data-fingerprint VERSION`.
When a user supplies a data version, the TUI checks this capability before
starting the run; absence or a malformed catalog stops the launch. The version
is a user declaration, not a verified data-content hash.
The value is a user-declared dataset/split version, not a verified content
hash, and must not contain credentials. It is saved with run inputs. A repeated
trial inherits it unless explicitly overridden, in which case the override is
recorded. Resume inherits it and rejects a different or newly asserted version;
use a new training run for changed data. Run comparison marks matching declared
versions and matching factory settings as `same` with an explicit warning that
data contents were not independently verified; absent versions remain unknown.
Matching identifiers do not prove identical samples, data ordering or contents.

For variant edits, `layer undo` and `layer redo` support `--dry-run` and
`--require-forward`, as do the other layer edit operations. A client can preview
the candidate with a real forward, then commit with the returned variant base
and forward input digest via `--if-variant-base` and `--if-input-content`.
These conditions protect the variant and the actual input contents; they do not
pin all source, project configuration, or runtime. When an input digest is
specified, the CLI requires a successful, actually executed forward and a
matching worker-reported input digest before saving either a variant or source
edit; an older worker that cannot provide that evidence is rejected. `--dry-run` and
`--require-forward` are not supported for source-target undo or redo. After a
transport interruption, inspect the variant before retrying a commit.

`edit.conditional_source_manifest` advertises an optional
`--if-source-manifest SHA256` condition for variant edits requiring a forward.
When a candidate contains a nonempty, valid worker-recorded project-source and
component-manifest map, the preview reports its canonical JSON digest as
`source_manifest_evidence`. The condition compares that digest after candidate
execution and re-reads the recorded files before saving; a mismatch rejects the
save. Files must be canonical project paths; hashing is streamed and checks
file identity and metadata before and after reading. Missing evidence cannot satisfy the condition. This covers only recorded
files, not external libraries, weights, other resources, future imports, or
changes after the check. The file hashes do not prove that the bytes read from
disk are identical to the code actually loaded by Python.

`edit.conditional_source_plan` advertises optional `--if-source-plan SHA256`
for source-target Layer edits that require a forward. The dry-run receipt's
`source_plan` hashes the complete planned file patches before their contents
are redacted from the report. The commit regenerates the candidate under the
source-edit lock and rejects a different plan with `SOURCE_PLAN_CONFLICT`;
the source store also checks the original file hashes before replacing files.
Successful commits perform a separate reconstruction and forward, with
rollback attempted on validation failure. The plan condition alone does not pin
batch contents, unedited files, configuration, runtime, or external resources;
source undo/redo use their own history and do not accept this condition.

`edit.conditional_source_input` adds `--if-input-content SHA256` to source-target
Layer edits that require a forward. The candidate reports an executed forward and
`sha256-forward-input-v1` input digest; a conditional commit rejects mismatched or
unavailable evidence before changing files, and independently reconstructs and
checks the same digest after writing. If the latter fails, source rollback is
attempted. A previously captured input snapshot can be supplied with
`--replay-input-from` and the corresponding digest for non-repeatable batch
factories; the snapshot contains input values and remains the caller's file to
protect and remove. Digests cover forward args and kwargs, not targets, data
sources, model weights, other files, environment or future input values. Source
undo/redo do not support these conditions.

`edit.conditional_source_config` adds `--if-source-config SHA256` to source
Layer edits that require a forward. The revision must match the project
configuration used for input resolution; the CLI rereads it immediately before
writing and after independent revalidation. A missing or changed revision is
rejected before writing, or causes rollback if detected after writing. These
checks do not lock configuration writers, so a change between checks or after
the final check can still occur; source rollback does not undo a concurrent
configuration edit. Undo/redo do not accept this condition.

`edit.conditional_source_project_manifest` extends `--if-source-manifest` to
source-target Layer edits with a forward. Candidate evidence hashes the
worker-recorded project Python source map; conditional commit requires the same
map and checks those files again before writing. After reconstruction it checks
unchanged recorded files against their prior hashes and edited files against the
approved patch hashes; detected drift triggers a rollback attempt. This evidence
does not prove that the disk bytes were the bytes actually imported, nor cover
future imports, non-Python files, external libraries, data, weights, or changes
between and after the checks. It is distinct from the variant manifest scope,
which can additionally include component manifests.

`layer.builtin_catalog` advertises `layer catalog`, a static, non-Python
catalog of example fields for common PyTorch Layer factories. It is not an
exhaustive allowlist. `catalog_version: 1` and
`scope: common-pytorch-constructor-fields` identify the hint format;
`exhaustive: false` explicitly states its coverage. It does not prove that a given PyTorch version,
constructor argument, device, dtype, or input is usable. Clients can still
accept raw JSON parameters; candidate construction and forward remain the
authoritative checks for a particular environment and model.

`runtime.torch_layer_catalog_v1` advertises the explicit
`env probe --library torch --layer-catalog` query. It imports torch in the
selected Python and reports bounded public `torch.nn.Module` classes and
inspectable constructor parameter names, kinds and required flags. Import and
signature inspection may execute code; no layer is constructed. Truncation,
unavailable signatures and failures are reported. The result is a snapshot,
not proof that a class can be instantiated or used with a given input. The
worker capability `environment.torch_layer_catalog_v1` is required when the
query is requested.

Source Layer generation preserves runtime factory names and keyword strings
when Python source identifier normalization would otherwise change them. Such
calls can use explicit dictionary expansion. The automatic source parameter
editor can address explicit keywords and literal `**` dictionaries with unique
string keys; dynamic expansions, nested dictionary expansions, duplicate keys
and `*args` still require manual source editing. All automatic changes require
the usual source-location and preview checks. Newly generated custom factory
calls use literal dictionaries; this alone does not establish that a factory
call can be mapped uniquely to the constructed class. Earlier generated
`json.loads` calls remain dynamic expressions and require manual editing.

`edit.variant_layer_rng_v1` advertises optional `--layer-rng-device cuda:0`
on variant `layer add` / `layer replace`, and `rng_devices` on individual
variant `layer apply --operation` add/replace JSON objects, for example
`"rng_devices":["cuda:0"]`. The flag may be repeated. Device IDs must be explicit, unique and from one
backend. The Layer seed applies to CPU and those devices only; construction
may initialize the selected backend. Existing operations without the field
retain CPU-only behavior. The field is persisted and replayed by generated
variant adapters, with worker feature `model.variant_layer_rng_v1` required.
Source edits and custom edit providers do not support this option. Seeded
generators do not guarantee deterministic algorithms, input or full-model
weights. The workspace offers CPU-only, an observed indexed device or manual
same-backend device IDs while preparing a variant add/replace; advanced users
can also enter the operation JSON. Device observations may become stale.

`edit.conditional_variant_edit_v1` advertises the workspace's required
preview and conditional commit path: a variant edit can be previewed with a
fresh forward and later committed against its variant base, input digest and
source-manifest evidence. A client must preserve the reviewed seed and request
and recheck its own context; these conditions do not pin all external state.

Success exits 0. Business failures and invalid arguments are JSON error reports
with nonzero exit status; `inconclusive` also has a nonzero exit status. stderr
may contain diagnostic logs and is not the JSON transport. Existing CLI timeout
and interruption diagnostics remain operation-specific. Clients must not retry a
mutating operation solely because transport was interrupted: query its state.
The TUI bounds synchronous report memory. If a confirmed write produces an
oversized report, it drains stdout and waits for the CLI to exit before reporting
an unknown outcome; users must inspect project state before retrying. TUI does
not surface stderr logs in this path.
The TUI may fall back to legacy help-text detection only when an older CLI
returns `CLI_ARGUMENT_INVALID` for this flag; malformed responses and unsupported
catalog versions remain protocol errors.

`training.conditional_resume_checkpoint` adds `train --resume RUN
--if-checkpoint-sha256 SHA256`. The worker verifies the selected latest
checkpoint against its manifest and the supplied lowercase SHA-256 before
loading training state; a mismatch reports `CHECKPOINT_CHANGED`. This pins the
checkpoint contents by loading the same private temporary copy that was
hashed, not source, data or runtime. This requires temporary storage at least
as large as the checkpoint; TMPDIR controls its location. Read or temporary
storage failures return CHECKPOINT_SNAPSHOT_UNAVAILABLE before state loading.
`runs show RUN` exposes the current checkpoint manifest for an explicit user
confirmation; this metadata is not itself an integrity check. Detached requests
can create a run record before the worker rejects a mismatched checkpoint.

Successful resumed training reports `data.resume_checkpoint` with the source
file, completed step, observed SHA-256, optional confirmed SHA-256, and whether
loading used the verified copy. The run summary retains these fields so clients
can show them after reopening. Without a confirmed digest, the SHA-256 was
observed before the legacy path loaded the file; it does not prove identical
bytes were loaded. Failed runs do not necessarily have this success receipt.

`evaluation.conditional_checkpoint` adds `evaluate --if-checkpoint-sha256
SHA256`. Conditional evaluation uses the same verified temporary-copy loading
as conditional resume and reports the actual checkpoint digest in the saved
evaluation. The evaluator is not called if the confirmed digest differs.
This requires temporary space at least as large as the checkpoint; data and
project code are governed by their separate provenance checks. Evaluation
without the condition retains the prior loading behavior for compatibility.

`evaluation.model_device_v1` adds `evaluate --device DEVICE`. The CLI requires the worker capability of the same name when this option is present. After loading checkpoint weights, the worker calls the model's `to()` and exposes the resolved device string as `context.device`; the evaluator is responsible for input placement. Without the option, `context.device` is `None` and prior placement behavior remains. Placement may execute user-defined `to()` code; the reported device is the resolved request, not independent proof of the placement of model state.

For unseeded model observations and edits with a representative input, the CLI requests worker capability `data.runtime_sampling_seed_v1`. When that input resolves to a dataset, the worker chooses and reports a sampling seed instead of silently assuming zero. A plain batch factory retains ownership of its randomness. Historical training requests with no seed keep their legacy sampling policy.

`data.raw_batch_binding_v1` adds `--batch-binding single-raw`: pass the complete batch-factory result as one positional argument without dataset auto-sampling. The distinct binding value is stored in project configuration, run inputs and checkpoint signatures; legacy `single` retains its prior sample-then-bind behavior. Explicit Dataset requests cannot be combined with `single-raw`. New CLI requests require matching worker capability when this binding is used.

The preview worker accepts an omitted binding or `single-raw` for a batch factory; unsupported bindings and bindings attached to a Dataset source are rejected before the factory is called. This validation is not yet manually verified.

Variant edit reports and `variant show` include `initialization_seed_decimal` as a decimal string, so a 64-bit construction seed can be displayed or passed back without floating-point rounding. Assembly generation reports expose the same field when a seed is recorded; a null value means no seed was recorded for that generation. Consumers of older reports must treat a missing string as unavailable rather than reconstructing a large integer from an imprecise JSON number. These additions are not yet manually verified.

`export.conditional_checkpoint` adds `export DIRECTORY --run RUN
--if-checkpoint-sha256 SHA256`. The selected run's checkpoint manifest must
match the supplied digest; the existing export staging additionally verifies
both the input checkpoint and staged copy against that digest before publishing.
A changed manifest returns `CHECKPOINT_CHANGED` before creating the export
destination. Source and extension files are separately checked against the run's
recorded evidence. Export without the condition retains its previous behavior.

`assembly.atomic_patch` advertises `assembly patch`: repeat `--set
SLOT=MODULE:FACTORY` or `--set SLOT=component:ID@VERSION`, optionally provide
`--param SLOT.NAME=JSON`, and/or repeat `--remove SLOT`. A non-writing preview
resolves all bindings and executes a full forward using the saved project model
and representative batch. `--apply --if-config-revision REVISION` repeats
validation and commits the whole mapping with a versioned configuration update.
Duplicate/conflicting operations, missing removed seams, missing batch, stale
configuration and invalid components are rejected. Source, runtime and batch
contents are not pinned between preview and commit; this is seam binding, not
arbitrary Python graph rewiring. Existing variants keep their own snapshots.

`assembly.atomic_patch_evidence` extends batch patch preview with the worker's
observed project Python source digest and forward-input content evidence.
An apply may supply `--if-source-revision SHA256` and/or `--if-input-sha256
SHA256`: the first compares the newly observed source map and rechecks those
files before the configuration update; the second requires the actual forward
input to match before the forward. The TUI uses available source evidence and
lets the user opt into input matching. Inputs that change between invocations
may fail this check unless the optional snapshot mechanism is used. Observed file
hashes do not prove which bytes Python imported or cover external libraries,
weights, target values, future imports, or writes after the checks.

`assembly.single_source_evidence` extends `assembly add`, `set`, and `remove`
previews with `source_evidence` for the project sources observed while validating
the candidate. A subsequent edit may pass the global `--if-source-manifest
SHA256` condition: the CLI compares the newly observed map and rechecks its files
before the versioned configuration update. A mismatch or unavailable evidence
returns `SOURCE_MANIFEST_CONFLICT` without saving. The TUI uses this condition
when available and explains when it is unavailable. A removed component is not
necessarily loaded by the candidate and may therefore be absent from the map;
the conditions do not fix inputs, external code or changes after the checks.

`assembly.single_input_evidence` adds `input_evidence` to single-seam previews.
For `add`, `set`, and `remove`, the global `--if-input-content SHA256` requires
the actual candidate forward input to match the preview's
`sha256-forward-input-v1` digest before the forward. Missing batches, unavailable
input content or a mismatch reject the edit without changing the configuration.
The TUI's `input` confirmation selects this guard; `y` saves without binding
input. Dynamic batches may change between calls.
Targets, later batches, and other external resources are outside this condition.

`assembly.single_input_snapshot` permits `--capture-input-to FILE` on an
`assembly add/set/remove --dry-run`, and `--replay-input-from FILE` on the
corresponding commit with `--if-input-content SHA256`. The TUI supports
`add+snapshot`, `set+snapshot`, and `remove+snapshot`; it verifies capture and
replay receipts, and attempts to remove the private temporary file on success,
cancellation and handled errors. Abnormal termination may leave it behind. The
snapshot contains forward inputs, not targets or future batches.

`component.conditional_uninstall` adds `plan_revision` to component uninstall
previews and accepts `component uninstall ID@VERSION --apply --if-plan SHA256`.
The digest binds the recomputed removal plan, including the component manifest,
observed references, extension sharing and active-run state; a changed plan is
rejected before removal. Existing callers may still use the unconditional
`--apply` form. This is an observation and pre-removal check, not an atomic lock
over all project files, process creation or external dependencies.
When the plan would remove a project-local extension directory, current plans
also contain `removal_tree` (`sha256-removal-tree-v1`, file count and byte count),
covering observed directory names and streamed hashes of regular files. A tree
change before conditional apply changes the plan. Scanning can be expensive and
the check does not lock files against changes after the scan.

`component.project_package_lifecycle` advertises `component scaffold` for a
new Layer or Layer Edit Provider package and `component install` for a
project-relative package. Install builds a wheel, places it in project-local
extension storage and registers its components; it does not select a trainer
or alter the selected Python environment. Build code executes with the user's
permissions and isolated builds may download build dependencies. A failed or
interrupted installation can leave project-local files; inspect before retrying.

Native Layer set/replace/remove edit effects may include
`old_state_not_copied`: names from the prior target module's state dict that
were not copied into the in-memory candidate. This is a module-local effect;
shared tensors can remain reachable through another module. A null or missing
field means unavailable, not an empty list. `preserved` and `initialized`
similarly describe the in-memory candidate rather than migration of any saved
training checkpoint. Custom providers may report different effect fields.

`assembly.atomic_patch_snapshot` permits `assembly patch --capture-input-to
FILE` for non-writing preview and `assembly patch --apply --if-input-sha256
DIGEST --replay-input-from FILE --if-config-revision REVISION` for commit.
The input snapshot contains the actual forward arguments and keyword arguments,
not targets. Replay verifies its content digest; the batch factory is not called
again, although its module is imported for the project source evidence. The
caller owns the snapshot file. The TUI offers `patch+snapshot`, uses a private
temporary location, and attempts cleanup on exit; abnormal termination may
leave the file. This does not guarantee future batches or external resources.

`project.scaffold_custom_module` permits `scaffold --module models.custom` to
create `models/custom.py` and configure `models.custom:model` and
`models.custom:batch`. Dotted Python identifiers determine a path inside the
project; invalid names and keywords are rejected. Preview and revision include
the chosen file, and creation refuses to overwrite it or project configuration.
Parent directories may be created during commit and can remain empty if a later
step fails. Importability still depends on the project's Python environment and
any existing packages; a successful write is not an import or forward proof.
For generated starters the input-route editor recognizes the starter source
marker; removing that marker can make this guided route editor unavailable.

`project.scaffold_external_batch` adds `scaffold --batch MODULE:CALLABLE` and
optional `--batch-binding envelope|single|args|kwargs`. The selected entrypoint
is saved in project configuration and included in the plan revision; without
an explicit binding, the runtime infers it. Preview and creation do not import
or execute the user factory, so they do not establish that it exists, is safe,
or produces the declared inputs or targets. The generated file still includes
an editable synthetic batch and a placeholder step; a real forward and trial
must verify the chosen data and loss contract. Omitting `--batch` retains the
original synthetic envelope default.

`project.scaffold_external_batch_params` adds repeatable `scaffold --batch-param
NAME=JSON` when `--batch` is specified. Parameters are included in the creation
plan revision and saved with the project configuration, without executing the
factory. Structured preview redacts recognized sensitive keys, but the actual
configuration stores values in plain form; command arguments and client drafts
may also expose them, so do not pass credentials. The TUI syntax
`@batchparam.NAME=JSON` uses semicolon-delimited declarations; values containing
semicolons should instead be supplied through the CLI or later batch settings.

`project.scaffold_initial_bindings` supports repeatable `scaffold --inject
SLOT=MODULE:FACTORY|component:ID@VERSION` and `--inject-param SLOT.NAME=JSON`.
Allowed seams are each declared input's `<name>_encoder`, `fusion`, and `head`.
Registered Layer components are resolved against their schema and exact manifest
version during planning; direct factories must accept zero arguments. The plan
revision includes resolved bindings and parameters. Preview and creation do not
construct the components or run a forward; actual model and data compatibility
still requires Probe and a trial. Source code and runtime are not pinned by this
plan. Configuration and TUI drafts store parameter values locally; do not use
credentials. TUI accepts `@seam.SLOT=...` and `@seamparam.SLOT.NAME=JSON` in
its semicolon-delimited input declaration.

`assembly.generated_source_verify` means `assembly generate` first validates
the current assembly, generates source, and re-imports that source in a fresh
worker before publishing the new file. The worker reconstructs the model and,
when a representative batch is configured, runs another forward. The response's
`generated_validation` records construction and forward separately; with no
batch, forward is `not_run`. This process executes user factories and the batch
again and does not fix their external state. It neither loads trained weights nor
vendors project modules, third-party packages or data. The legacy `validated`
field describes current-assembly validation, not checkpoint equivalence.
Current implementations also compare project source manifests observed after
the first validation and the generated-source verification, and recheck the
second manifest before publication. `source_evidence.status` is `checked` for
a nonempty first manifest or `unavailable` when no project files were observed;
older implementations may omit this field. A changed, missing or inconsistent
manifest prevents publication. These observations do not prove which bytes were
imported, cover external dependencies, or eliminate changes after the check.

`assembly.show_runtime_paths_v1` means live assembly inspection uses the selected
Python, working directory and project import roots. `assembly.generate_runtime_paths_v1`
extends the same selection to both generation validation workers, compares
the declared paths and resolves them again before publication. A client must
check the relevant capability before forwarding explicit workspace path
overrides. These capabilities do not prove that external dependencies or data
are identical; implementation remains subject to the planned manual review.
`input_comparison.status` separately reports whether the two actual forward
inputs have matching complete `sha256-forward-input-v1` evidence: `same`,
`different`, `unknown`, or `not_run` when there was no batch. Older responses
may omit this field. A match does not establish output, target, dataset or
trained-weight equivalence; a difference does not automatically reject code
generation.

`project.scaffold_native_encoder_output` adds repeatable `scaffold
--native-encoder SLOT`. For these input encoder seams, the generated model
passes the encoder result to the fusion module unchanged: no implicit dynamic
axis averaging, dtype conversion or flattening is applied. Other seams retain
the default reduction and flattening behavior. The scaffold preview includes
the choice for each encoder; the TUI input syntax is `@native=x_encoder`.
The default fusion and head may be incompatible with native outputs: select
custom components and validate them with a real forward. This is an editable
starter, not a general-purpose graph compiler, and creation does not execute
the selected factories.

`project.scaffold_encoder_output_edit` adds `assembly encoder-output SLOT
native|flatten --dry-run` and a guarded commit with `--if-config-revision`.
It requires a starter accepting `encoder_output_modes` and a saved batch. Both
preview and commit run a representative forward with the proposed override;
commit uses a conditional project-config update. `project.scaffold_encoder_output_evidence`
adds optional `--if-source-revision` and `--if-input-sha256` checks against the
worker-observed project sources and actual forward input. The TUI uses source
evidence when available and offers to bind input content. It reports when
either is unavailable. External code, environment, targets, and changes after
the checks are not pinned. Older starters can use `layer migrate-starter
--target source --dry-run --require-forward` when their structure is recognized;
the candidate is executed and its source diff is returned. A confirmed commit
requires `--if-source-plan`, revalidates the source transaction and checks a
fresh reconstruction, with rollback attempted on failure. Existing user code
outside the recognized expressions is preserved; unsupported forms are rejected.
Preview constructs the candidate once, starting the random generators before
loading its module and components as a fresh worker would. It records the
candidate's source manifest before unloading the temporary module. A nested
package that imports the target module during package initialization, or an
import path/loader that does not resolve normally to the edited Python file,
is rejected because the preview cannot safely reproduce that import.
Existing variants pin old source, checkpoints are not migrated, and the old
default head width is not recomputed when routes change. A
factory signature that cannot accept the assembly options returns
`MODEL_ASSEMBLY_OPTIONS_UNSUPPORTED`. For identifiable local Model-Maven
starters, the worker also checks the constructed model's route and output-mode
declarations against requested overrides; mismatches or missing declarations
return `MODEL_ASSEMBLY_OPTIONS_UNCONFIRMED`. Those declarations are model
self-report, not proof of actual tensor flow. Custom factories outside the
identifiable starter scope retain their own contract.

`assembly.atomic_patch_encoder_output` extends `assembly patch` with repeatable
`--encoder-output SLOT=native|flatten`. The proposed encoder settings and seam
bindings are constructed and forward-tested together and committed in one
conditional project configuration update. The TUI batch input accepts
`@output.text_encoder=flatten` alongside seam bindings. The existing optional
source, input and input-snapshot checks also apply to the combined candidate;
external dependencies, targets and changes after the checks are not pinned.

`assembly.atomic_patch_route` adds repeatable `assembly patch --route SLOT=INPUT`.
The TUI batch field uses `@route.b_encoder=a`; route changes can be combined
with seam bindings and encoder output changes in the same candidate and
conditional config update. The saved factory must accept `input_routes` and
validate the selected names; preview executes the representative forward.

Models may opt into construction-time route declarations by setting the plain
instance attribute `_model_maven_declared_routes` to a list of at most 32
objects with `slot`, `input`, and `module_path` strings. The worker resolves
each path against registered modules and records every registration alias; an
invalid declaration or incomplete module scan is marked unavailable. This is
model-authored metadata and can become stale; it is not runtime dataflow.
New custom scaffolds emit this attribute. The TUI only associates a route with
a graph call when registration is unique and identifies it as self-reported.

`runs.comparison_unverified_data` indicates that `runs compare` does not label
data content as the same merely because batch settings and human-declared
versions match. The `data` dimension reports `status: unknown` in that case,
with `settings_status`, `declared_version_status`, and `content_identity`
reported separately. A changed setting is reported as changed, but does not
prove that the underlying data content differs. Older CLI versions may use
`status: same` for declarations alone; consumers should treat that as unknown
content identity. This release does not independently hash the dataset.

## Capability evidence, readiness and compatibility boundary

The catalogs describe different evidence layers and are not interchangeable:

| Evidence | Source | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| CLI interface capability | `--describe-interface` `data.capabilities` boolean map | This CLI build recognizes a guarded command/schema feature | Python availability, worker support, component conformance, or model behavior |
| Worker feature registry | `compatibility.json#/required_features` and `#/operations`, mirrored by `python/model_maven_worker/protocol.py` | An operation is registered and names the worker features required before dispatch | Whether the selected runtime or model instance can satisfy those features |
| Selected runtime capability | `doctor.data.compatibility` and `doctor.data.capabilities[]` | Worker protocol/API/state format and registered capability state for the selected Python/platform at query time | Model construction, data compatibility, observed path coverage, or safe editability |
| Provider declaration | Component manifest | What the component claims in its versioned description | Runtime conformance until the corresponding check/test is run |
| Model/data instance evidence | Explicit inspect, probe, preview, conformance, or candidate-validation report | Only the paths, inputs and scope actually observed by that operation | Unobserved/dynamic paths, later mutations, external dependencies, or future batches |

An absent capability, failed query, malformed JSON, duplicate ID, invalid version,
unknown status, or incompatible protocol is `unknown`/incompatible evidence. It
must not be converted to `supported` or successful readiness. An older CLI that
rejects `--describe-interface` has unknown interface capability; clients may keep
only separately documented legacy-safe read-only actions. They must stop a
protected write or execution route when its required interface capability cannot
be established. For path values, pass the path as one argv value or a JSON string;
spaces are ordinary path characters and must not be split or shell-expanded by a
client.

`OperationPlan` v2 defines `ready`, `needs_input`, `conditional`, `unsupported`,
and `unknown`; facts and capabilities separately use `available`, `missing`,
`conditional`, `unsupported`, and `unknown`. This is a point-in-time answer for
one selected operation. It is not a project-wide score or a lock on files,
configuration, runtime, data, model weights, or provider code. A missing source
revision means freshness is unknown. Recompute before an operation-specific
commit guard; a revision mismatch is a conflict, not permission to use the old
plan. A successful legacy `Report` is not proof that readiness was known.

Current coverage is partial. The application builds plans for selected command
families, and `ApplicationUiServices` blocks missing, unsupported or unknown
required inputs and capabilities when such a plan exists. A plan that fails
structural validation is an error rather than an absent plan. Required
conditional inputs, capabilities and facts are rejected unless an explicit
operation-specific execution check exists: Layer provider verification is
deferred to its worker, which checks the payload, source fingerprints and
runtime identity before using the provider. Advisory conditions without a
required conditional fact or capability remain visible but do not themselves
assert a failed prerequisite. Commands without a readiness plan do not inherit
this gate. The legacy interactive
training-plan flow now starts only a `ready` plan and rechecks that the plan is
unchanged after confirmation; an unavailable, unknown or conditional plan must
be resolved and prepared again. This is not a claim that the contract is
enforced everywhere or that a matching plan locks external resources.
`DiagnosticDetails`
can carry severity, scope, recovery action availability and evidence certainty,
but current reports primarily provide legacy `code`, `message`, and optional
`hint`. Hints are display text, not executable recovery actions. Recovery advice
must name a compatible next query/action and be revalidated after the user selects
it.

## Query, preview, execution and cancellation

The default/home view is query-only. It may read local metadata and show known
catalog facts, but must not construct a user model, call a user batch/transform,
run training/evaluation code, install packages, download weights, or mutate
project files. When facts are unavailable it reports `unknown` and presents
available inspection/configuration actions. `--describe-interface` does not
start Python or import project code; its Python selection hint may read project
configuration metadata. `doctor` may start the selected Python runtime to inspect
it, but does not construct the selected model. An application readiness plan may
ask `doctor` for runtime capabilities when its required inputs are present, so it
is not a universal zero-process query.

A preview is non-writing only with respect to the target configuration/source
being planned. If its operation runs a forward, imports a selected component, or
invokes user code, that must be stated by that operation's contract. A commit
requires its documented explicit choice and revision/input guards, then
revalidates the candidate. A lost or interrupted commit receipt leaves the
outcome unknown; query the operation's request, run, source history, or current
configuration before retrying.

Cancellation is operation-specific and generally cooperative. Selected preview
flows support cancellation and generation supersession; training stop asks the
user loop to stop at a supported boundary. There is no common cancellable async
operation envelope for every CLI command. A stop request or `USER_INTERRUPTED`
report does not prove that no side effect occurred. Reconcile current operation
state before retrying.

Cache and freshness are likewise operation-specific. Model evidence cache keys
and preview request generations have local identity rules; the application
runtime capability catalog is process-local. No universal stale marker or global
invalidation event exists. If a consumer cannot verify the evidence identity or
source revision it needs, it shows the result as stale/unknown and requests a
fresh query rather than reusing it as current evidence.

## Neovim workspace mapping

| Presentation boundary | Existing source of facts | Remaining boundary |
| --- | --- | --- |
| Project, model, input and environment context | `assembly show --config-only` for saved configuration; session overrides and buffer state in the workspace; `input_sources` and runtime fields in operation reports | A selected path or source revision does not prove an imported module, complete environment, device, or external data identity. Missing and stale evidence remain unknown. |
| Actions and prerequisites | CLI interface capability map, operation-specific `readiness` facts/capabilities/actions, and the current observed report | No universal backend recommendation or complete action catalog exists; workspace suggestions are derived from its current project and evidence. Conditional and unsupported actions retain their reported limits. |
| Preview and commit | Operation-specific preview report and conditions such as config revision, source-plan/manifest and input-content identities | Preview can execute user code even when it does not write the planned target. Guards cover their stated inputs, not arbitrary external state or all side effects. |
| Results and recovery | Versioned report status and diagnostics; operation-specific run/request IDs, current files and the workspace's pending-operation marker | Cancellation and recovery are operation-specific. A lost or malformed receipt is unknown, not a successful rollback or proof that no process ran. Diagnostic `hint` text is not an executable recovery contract. |

The journey states below map the current interface to facts, not to a promised
universal backend state machine. The workspace derives its suggestion from the
available context and last result; another action may replace that result.

| Journey state | Current facts and actions | Unknown or unsupported boundary |
| --- | --- | --- |
| Starting a project | Saved metadata via `assembly show --config-only`; `discover` gives static entrypoint candidates; `init`/`scaffold` offer plans and guarded writes. | A missing or unreadable configuration must not be silently treated as a safe empty project. Static candidates do not prove imports or model behavior. |
| Model selected, no representative input | Session selection and saved model/batch fields identify the configured factories; inspection can construct the model after confirmation. | A declared factory is not an observed sample. User-provided loops may supply their own data; no real dataset is fabricated. |
| Preparing a forward | The confirmation identifies selected model, input, Python and known path/seed/device choices; the CLI capability and worker readiness are checked where supported. | Selection and capability declarations do not prove the instance will run. User code and Python startup hooks may have side effects. |
| Forward failed | `probe` diagnostics, partial calls and source identities can support a report, input change or verified source navigation. | Missing, truncated or ambiguous observations are unknown; the last observed call is not necessarily the root cause, and failed execution may have side effects. |
| Candidate modification | A dry-run `edit` or assembly plan supplies the candidate, validation and operation-specific revision/input/source guards before explicit commit. | Preview can execute code; checks cover only their reported scope, not arbitrary dependencies or external state. Unsupported source forms remain manual edits. |
| Forward observed | A successful `probe` with executed forward records observations for the selected input; module navigation, semantic marks, edit and training preparation are separate choices. | Reused or partial evidence is identified; unobserved branches, external state and other inputs are not proved. A variant result remains separate from the source model. |
| Run state unknown | Run/request lookup, logs and persistent pending markers provide operation-specific recovery paths. | Missing records or a lost receipt do not prove that a process stopped or never started; a user must inspect and decide before retrying. |

For a command with a readiness plan, an inaccessible project directory,
unreadable/invalid project configuration or invalid plan is an error; it is not
silently treated as a command without a plan. An uninitialized project is a
separate state and remains available to operations that accept explicit inputs.

The framework-neutral loop, evaluation, data preview and assembly view, edit and code generation paths have readiness plans. Project operations retain their own guards:

| Route | Existing operation-specific checks | Limit |
| --- | --- | --- |
| `loop-trial` / `loop-run` | A readiness plan checks the loop entry and selected worker declarations, including the signature and detached capability; the operation also validates paths, request and run identity. | The declarations-only worker query does not deliberately import PyTorch; Python startup hooks may still run. Missing metadata is unknown. Invalid or unavailable historical run IDs still use operation-specific diagnostics. User code, dependencies and resume contents remain unverified until execution. |
| `evaluate` | A resolved historical training record supplies the Python for capability checks unless explicitly overridden; the plan checks evaluation, checkpoint and requested device capabilities. | The plan identifies a record, not a valid checkpoint or evaluator. The operation separately validates checkpoint and inputs; user code runs in the selected runtime and snapshots may differ. Missing records retain operation-specific diagnostics. |
| Data preview | A readiness plan records the explicit source type and checks preview, dataset, whole-batch and selected runtime-path capabilities as applicable. | A selected source is not a verified factory or sample; the operation separately validates parameters and paths and may execute user code or renderers. A missing source keeps the operation-specific diagnostic. |
| Assembly view, edits and generation | Runtime assembly view and edits check selected model, applicable batch, runtime paths and worker capabilities; generation also checks code-generation and selected variant requirements. Batch patch and route settings require the saved project batch; `show --config-only` remains static. | A ready plan does not verify components, source, weights or the forward result. Each operation retains its revision checks and may execute user code; project creation/configuration retain their own guards. Invalid generation paths or unavailable variants retain operation-specific diagnostics. |

These checks do not yet provide a single readiness envelope or common
cancellation contract for all routes. See the operation descriptions for exact
requirements; an unlisted operation must not be assumed to have equivalent
guards.

For required runtime capabilities, an explicit unsupported state offers an environment inspection action; an unknown or incompatible worker declaration offers a diagnostic action. An older worker that lacks the declarations-only endpoint remains unknown and the requested operation is not started. These actions do not install dependencies or establish that a particular model is supported.

The workspace displays each readiness action's reported availability. Only an
action explicitly marked `available` can open a known workspace handler;
conditional, unsupported, missing or unknown availability remains advice until
the conditions are reviewed and a fresh plan reports availability. A handler
still performs its own current-context checks and confirmations. Current CLI
execution reports can wrap the complete operation plan under `data.readiness`,
with actions in `data.readiness.readiness.actions`; older direct readiness
objects are also accepted. Missing actions or availability are not inferred.

The current workspace route map is recorded in
`operations.json#/workspace_entry_matrix`; its source is
`../model-maven-tui/lua/model_maven/workspace.lua` keybindings and
`../model-maven-tui/lua/model_maven/transport.lua` request validation. `M.open`
creates the panel and refreshes saved state with `assembly show --config-only`.
Key `I` requests `--describe-interface` and labels returned booleans as CLI
declarations; it may separately display a still-current inspect/probe report as
model-instance evidence. It does not run a model and does not query the selected
worker runtime. Keys `i`/`l` issue model inspection/layer listing, `p` issues a
forward probe, `w`/`W` prepare Layer edit candidates, `A` prepares assembly
changes, `y`/`T` prepare training actions, `R`/`H` reconcile run state, and `C`
interrupts the active client process request. Their CLI report operation and
worker route IDs are listed in the matrix and cross-referenced to
`compatibility.json#/operations`. Key `Z` reads saved variant metadata without
running the model; the selected variant's forward, edit, training and
supervision actions separately check their advertised capabilities and pin the
selected revision where supported. A source module detail can also initiate a
new variant edit with `V`. The reported structure is not proof of execution,
and observations made on a variant are not source-model observations.
`F`, `n`/`a`, `Q`/`S`, and `e`/`E`/`K` are also mapped in the matrix:
static discovery and configuration previews do not establish runtime behavior;
explicit environment inspection can start Python and import libraries; saved
runtime choices do not freeze their contents.

The workspace transport validates report protocol v1, operation, status/exit-code
agreement, successful data presence, and required interface-catalog identity.
Missing CLI or incompatible/malformed catalog stops guarded actions before the
requested operation. For actions with explicit runtime-path overrides it first
checks `runtime.workspace_paths_v1` and the route-specific CLI capability. It
passes command arguments as an argv array, so spaces in a path remain part of one
argument; it does not rely on shell quoting.

If this preliminary capability query fails, the transport reports its bounded
reason and marks the requested action as not started. The query process itself
may have started; this distinction applies to the subsequent model, data, or
training operation, not to arbitrary side effects of a nonconforming CLI.

The TUI also tracks project config revisions, source buffer/disk changes and a
per-session change epoch. A changed context labels prior reports stale and gates
execution; a final check rescans open Python buffers instead of relying on the
initial comparison cache. Cancellation during capability discovery is reported
as cancelled before start. After a request starts, the TUI distinguishes a
completed operation, `USER_INTERRUPTED`, and an unknown cancellation outcome;
unknown outcomes require inspecting affected files or run state before retry.
These are TUI request/session behaviors, not a universal backend stale-event or
async-operation protocol. The capability overview itself is not a unified
per-operation preflight: it combines interface declarations with available
current instance reports, while selected-runtime doctor evidence and action
specific worker feature gates remain separate.

The workspace capability overview reuses these existing sources:

| Source | Reused fields | Boundary |
| --- | --- | --- |
| `--describe-interface` | CLI capability IDs for probe, guarded edits, training and supervision | A declaration is CLI support, not proof for the selected runtime or model. Missing or unavailable metadata is unknown. |
| Current inspect/layers report | Module paths, source-edit and insertion candidates and rejection reasons | Candidates require an individual preview; they do not authorize an edit. |
| Current probe report | Actual forward status and observed input/call paths | Unobserved branches and truncated calls remain unknown. |
| Current training trial | Reported mode, supervision and forward/loss/backward/resume seam information and coverage | Unreported fields remain unknown; a trial does not establish other inputs or training runs. |
| Neovim source buffer | Manual takeover in the existing buffer | Unsaved changes are not executed; save and revalidate explicitly. |

There is still no unified, enforced readiness matrix combining selected-runtime
doctor evidence with model-instance evidence. Runtime checks require explicit
confirmation, and individual operations retain their own capability and
revision guards.
