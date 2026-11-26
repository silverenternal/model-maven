# Explicit graph topology plan v1

`m-model --json graph-plan --graph FILE --if-graph SHA256` reads a regular JSON file of at most 128 KiB, verifies the exact file bytes, and returns a read-only plan. It compares optional user-declared shape contracts and infers merge shapes conditionally. It does not import the declared factories, infer dtype or actual runtime shapes, generate code, execute models, edit the project, or attest that runtime interfaces match declared ports.

当前实现的运行时容器使用组件定义的 `children`（按顺序引用其他定义，最多 64 项），与仅用于显示的 `groups` 不同。默认将子实例逐个作为位置参数传入工厂；声明 `"children_style": "list"` 则把有序子实例列表作为一个位置参数传入，空列表也会传递。工厂仍由所选 Python 运行时解析，必须自行支持所选构造方式并返回 `nn.Module`；规划不会导入或验证工厂。引用必须存在且不得循环。相同定义引用的是同一实例；在容器和其他位置同时注册共享模块时，PyTorch 状态字典可能出现别名，迁移权重前应核查实际状态键和身份。独立复制组件树保留副本内部共享，但不会复制已训练权重。旧 CLI/TUI 应按能力拒绝无法理解的字段。

```json
{
  "version": 1,
  "definitions": [{"id": "shared", "factory": "my_layers:make_block", "params": {}}],
  "nodes": [
    {"id": "x", "kind": "input", "outputs": ["value"]},
    {"id": "first", "kind": "call", "definition": "shared", "inputs": ["value"], "outputs": ["value"]},
    {"id": "second", "kind": "call", "definition": "shared", "inputs": ["value"], "outputs": ["value"]},
    {"id": "sum", "kind": "add", "inputs": ["left", "right"], "outputs": ["value"]},
    {"id": "result", "kind": "output", "inputs": ["value"]}
  ],
  "edges": [
    {"from": {"node": "x", "port": "value"}, "to": {"node": "first", "port": "value"}},
    {"from": {"node": "first", "port": "value"}, "to": {"node": "second", "port": "value"}},
    {"from": {"node": "x", "port": "value"}, "to": {"node": "sum", "port": "left"}},
    {"from": {"node": "second", "port": "value"}, "to": {"node": "sum", "port": "right"}},
    {"from": {"node": "sum", "port": "value"}, "to": {"node": "result", "port": "value"}}
  ]
}
```

IDs and port names are 1–128 ASCII letters, digits, `_` or `-`; optional node labels are display text. Definition IDs are distinct from call-node IDs. Calls referencing the same definition declare shared identity, which the plan reports without promising executable weight sharing. A call has at least one input and one output; inputs have outputs only, outputs have one input only, and `add`/`concat` each have at least two inputs and one output. Every target port accepts at most one incoming edge; a source output can fan out. Unknown fields, invalid references, duplicate IDs/ports, invalid boundaries and cycles produce errors. Loops can be represented only inside a user-supplied component.

An unconnected input port or absent input/output boundary returns `status: incomplete` with details; a structurally connected graph does not establish shape, dtype, factory or execution validity. Topological order is deterministic with lexical tie-breaking. `unused_definitions` and `shared_definitions` are informational. Limits: 512 nodes, 512 definitions, 64 ports per node and 4096 edges. The graph planner and standalone source generator are connected to the graph canvas but not yet to the unified project declaration or training workflow.

Node validation, axis and declared shape errors include `data.node_id`, optional `data.port`, and `data.input_sha256`, so an interactive client can locate the issue only against the matching draft. Other failures may lack a node location. The canvas lists unconnected inputs and missing boundaries and can focus a located node; it does not treat a static plan as runtime verification.

The canvas keeps its display order separate from `nodes` and `edges`. Dragging a node or using the display-order keys changes only the layout, not the saved graph, generated execution order, or tensor connections. Automatic arrangement orders nodes by dependencies, preserving the prior relative order when possible; it refuses cycles or missing endpoints. Layout changes participate in canvas undo/redo and are stored in local editor state per project. Concurrent or invalid layout state is reported rather than deliberately overwritten; layout is a convenience and is not part of the portable graph declaration.

The local layout record also stores the last selected node and viewport when the plane is left normally. Reopening restores a selected node only if it still exists, and clamps the viewport to the available buffer; an older layout without this information remains readable. This is editor state, not a graph edit or portable project declaration. Cross-session and terminal-specific behavior still requires hands-on acceptance.

When the graph file changes outside the canvas, reconciliation can compare and retain the draft or, when its original base is available, offer a conservative three-way merge. The merge refuses unknown top-level fields, differing concurrent structural changes, differing concurrent settings changes, and cross-side structural/settings changes; it does not prove that the resulting graph runs. Review applies the merge only to the undoable draft and updates its save baseline. The user still reviews and saves separately, with a fresh revision check. A recovered draft whose original base is unavailable cannot be merged automatically. This interaction awaits hands-on acceptance.

The focus action shows a selected node and its immediate connected neighbors, counts hidden nodes, and marks incoming edges from outside the view. Toggle it again to return to the full graph; search and diagnostic navigation can also return to the full view. This is a view filter, not a graph edit or an assertion that hidden nodes are irrelevant.

Optional `groups` contain an ID, node IDs, and an optional parent group ID. Each node may belong to at most one group; the planner rejects duplicate groups, missing nodes, overlapping membership, missing parents and group cycles. A group can be folded or moved in the canvas; deleting one retains its nodes and reparents its children. Membership and hierarchy are saved in the graph, while folding is kept in per-project local editor state. Neither changes execution order or creates a nested runtime module; use definition children for supported runtime containers.

Dragging a node onto a group heading changes its membership as one undoable graph edit. Moving a node to an empty grid cell changes only display position; group membership remains unchanged unless explicitly edited.

Connection details display the two explicitly declared endpoint shapes, using `?` for undeclared shapes, and let the user jump to either endpoint. A node menu lists directly connected upstream and downstream nodes. These navigation actions do not infer runtime tensor shapes or alter the graph.

Opening the graph editor first shows the planar grid in the same workspace; `q` returns to the connection view, where `P` reopens it. An empty graph offers a basic structure or a first node. Dragging moves a node between unoccupied cells; keyboard selection and movement are available, and clicking a node opens its graph details without closing the grid. The grid renders orthogonal connection paths; clicking a path or selecting an edge with the keyboard opens the existing connection editor. Accepted graph and layout edits refresh the open grid. Crossings may contain multiple selectable connections. Rendering is bounded to 256 edges and 100,000 path cells and reports truncation; the connection view retains the complete graph. Coordinates are stored in version 2 of the per-project local layout record, with version 1 order/folding records still readable. Moving a node changes neither graph edges nor generated execution. The grid supports up to 32 columns and 512 rows, not unrestricted coordinates, and does not guarantee clear routing in dense graphs.

Within one open plane, wire paths and hit locations are reused while topology, layout and focused-node mode remain unchanged. A document refresh or position change discards that cache; focused views keep only the latest selection. Rendering still rebuilds the visible buffer and has not been benchmarked after this change.

Previously observed operator signatures in the graph editor are discarded when its project, runtime selections or inherited environment variables change. Only a digest of the inherited environment is retained for this comparison; the environment is not recorded in the graph. Changes to packages or source files at the same paths are not detected, so users can explicitly refresh the signature.

Each node displays `i` and/or `o` when it has inputs or outputs. Dragging `o` to another node's `i`, or selecting a source and using `c`, selects the exact ports through the existing graph connection flow. That flow checks occupied inputs, declared shape conflicts and cycles and creates an undoable graph edit. Port selection does not imply runtime shape verification.

Dropping a node on the left or right half of an occupied cell proposes placing it before or after the target in execution order. It is available only when both nodes belong to the same unbranched chain of single-input, single-output component calls. After showing the old and proposed order, confirmation rewires the chain atomically as one undoable graph edit. Declared endpoint shape conflicts and missing ports block it; unknown shapes and factory behavior still require runtime validation. The operation moves the corresponding visual blocks to reflect the new order, preserves display-group membership, and does not reorder arbitrary branching graphs.

During a planar drag, `*` marks the candidate cell and the status describes whether releasing moves only the visual position, opens an execution-order review, or offers port selection. A wire target remains a candidate until ports and compatibility are checked after release. Leaving the canvas or grid, escape, focus loss and resize cancel the gesture; preview never commits a graph edit. The visual behavior still needs real terminal acceptance.

`graph-reorder-plan --graph FILE --if-graph SHA256 --node ID --before ID` or `--after ID` provides the same bounded operation to CLI clients. It checks the input revision and graph structure, returns the old/new order and proposed graph without writing files, and reports conditional shape conflicts. A pre-existing shape conflict does not prevent proposing a repair. The canvas independently compares its proposal with this response before asking the user to apply it to the draft; the normal reviewed graph-save transaction remains responsible for persistence. Older CLI versions can still plan `--before` and the workspace hides the unsupported `--after` option.

The reorder plan also lists replacement connections whose declared dtype names differ. The canvas includes a bounded summary before confirmation, even with an older compatible reorder planner. Names may be aliases and no automatic conversion is performed; these are review hints, not proof of a runtime mismatch.

Dragging near an edge scrolls the planar view vertically or horizontally; leaving the window or pressing Escape cancels the gesture. Repainting preserves the current view, and terminal resize adjusts the window without changing stored positions. The selected node's full ID is shown in the view header. Cross-terminal mouse and resize behavior remains pending manual acceptance.

`A` proposes a dependency-ranked grid arrangement. The user confirms replacement of existing display positions, and the arrangement is a single undoable layout edit. Cycles and missing endpoints prevent the arrangement; it changes no connections and does not assert that graph contracts or runtime behavior are valid.

The planar view's `u` and `Ctrl-r` use the same undo/redo history as the graph editor, including graph edits made before entering the view. The view reloads its nodes, connections and positions after each restore; a changed graph or an empty history is reported rather than applying a stale operation.

The plane's `[m]` menu and shortcuts invoke the same graph actions for basic structures, node creation, planning, saving, source generation, data, forward and trial preparation. Plan feedback that identifies a node focuses it in the plane. Operations retain the graph editor's existing review and conflict checks; opening an action does not mean it has completed successfully.

`S` in the plane starts a continuous save-and-data workflow for an initialized project: review and save the graph, then review applying its declared data to the project input. These remain two independently conditional transactions, not an atomic multi-file commit. If the first step is cancelled or fails, or the graph/project changes before the second, the second step does not start. The data review may also be cancelled after the graph has been saved.

The plane shows whether the graph data matched the last checked project input, differed, was overridden by session input, or could not be established; it does not imply the factory was run. The check caches a bounded project-configuration read using file identity metadata, so it is a navigation hint, not a replacement for launch-time checks. If the project and session input already match after the graph is saved, `S` skips the redundant input write. A missing project is directed to source generation and setup.

`/` searches the current graph's node IDs, types, component IDs, factory references and port names, and selects a matching node in the plane. The selection header summarizes its full ID, type, port counts and factory when declared. Search results are discarded if the graph or layout changes while the picker is open.

Node cards label boundary inputs and outputs, component calls, additions and concatenations with short text codes and highlight their input/output connection points. Selection also has a textual marker, so it does not rely on color. Long IDs are abbreviated on the card and shown in full when selected; visual accessibility and theme contrast still require terminal acceptance.

`f` toggles a wire filter for the selected node, leaving all nodes visible while showing only its adjacent connections. Shared wire cells are marked `*` and let the user choose among represented edges. A wire that exceeds the remaining rendering budget is omitted in full; hidden wires are reported and remain available in the connection list. These measures improve navigation but do not guarantee non-overlapping routes.

Duplicating a group copies its descendants, nodes, internal connections and output objectives as one undoable edit. The user chooses whether the copied calls share the original component instances or use new definitions, preserving sharing within the copy. Connections crossing the group boundary and learned weights are not copied; reconnect ports and review data bindings after duplication. This operation does not turn display groups into runtime containers.

From a selected connection the canvas can insert a single-input, single-output call node. It replaces the connection with two connections in one undoable draft operation and leaves any other fan-out untouched. The component's actual interface and shape remain unverified until planning and execution.

On an empty graph the canvas offers single-stream, two-input addition, and residual scaffold structures. These are editable graph starting points rather than pretrained models. The residual scaffold deliberately leaves its branch input unconnected until the user supplies a branch; neither starting point guarantees compatible shapes, trainable parameters, or runtime behavior.

Removing a single-input, single-output call can optionally reconnect its sole upstream edge to its downstream edges in one undoable operation. This option is withheld for missing or ambiguous endpoints, self-connections and known declared shape conflicts. Unknown shapes remain unknown; removing the node changes computation even when reconnection succeeds.


## Optional declared shapes

Input, call, add and concat nodes can provide `output_shapes`, and any node with input ports can provide `input_shapes`, as objects keyed by the corresponding port name. Each shape is an array of up to 16 dimensions: a positive integer is a declared size and `null` is unknown. An omitted port shape means unknown rank; `[]` means scalar. For example, an input node can include `"output_shapes":{"value":[null, 128]}`. These contracts are authored assertions, not observations of a factory, data sample or model execution.

The graph editor lets users add fixed or dynamic dimensions, edit, reorder or remove them, and save a zero-dimensional scalar or remove the contract. The shape remains a draft until the graph is saved; adding a dimension does not execute a model.

Node ports can be added, renamed, removed or reordered individually. A connected port or a port carrying a semantic label or draft rules cannot be removed or renamed until those associations are handled. The editor asks for confirmation when an edit changes a positional/sequence mapping, discards declared shape contracts, or may require data binding changes; it does not silently rewrite the data declaration.

The planner rejects a connection when both ends declare incompatible ranks or known dimensions. For a connected input, its declared contract can fill dimensions or rank that its upstream leaves unknown; disconnected input contracts do not imply a connected value. It checks known input ranks and known non-concatenated dimensions at `add` and `concat` nodes; `concat` axes, when supplied, must be in range for known input ranks. Equal-size constraints can conditionally fill a dimension from one input when another leaves it unknown; a concatenated axis still requires all contributing sizes to calculate its sum. If the concat axis is omitted, no dimension is assumed to be the concatenation axis. A declared merge output is checked against the conditional inference and can supply dimensions that inference leaves unknown. It returns `output_shapes` by node and port: `null` for unknown rank, or arrays containing `null` for unknown dimensions. Missing inputs, undeclared factory outputs and otherwise unsupported information are never treated as runtime evidence. `shape_status: conditional_on_declared_contracts` and `runtime_verified: false` apply even if every declared dimension is numeric.

## Python source preview

`m-model --json graph-generate --graph FILE --if-graph SHA256 --output my_graph.py` checks the same topology, requires a complete graph and an explicit integer `axis` on each `concat` node, then returns source, a SHA-256 digest and a plan revision without writing or executing it. The output argument is optional for a preview; including it binds the proposed destination into the plan. `node_lines` maps each ID to its comment in that proposed source. `axis` is invalid on other node kinds; planning may still represent a concat whose axis is not yet selected.

The generated `GraphModel` imports PyTorch at execution time and constructs each referenced `module:callable` factory exactly once, in definition order. Factories receive the declared JSON parameters as keyword arguments and must return `torch.nn.Module`. Call nodes sharing a definition call that same module instance; unused definitions are not constructed. Each call may set `call_style` to `positional` (inputs in declared port order) or `keyword` (arguments named for the ports). Omitting it retains the original keyword behavior for existing graphs. New calls created in the canvas use positional arguments; users should choose a style compatible with the component's actual forward signature. Optional `output_style` selects `mapping` (a dictionary with exactly the port names) or `sequence` (a tuple or list whose length and order match the output ports). When omitted, the original behavior remains: one output passes through unchanged and multiple outputs require a named dictionary. The selected style applies even to one port. Reordering positional inputs or sequence outputs changes the mapping and requires confirmation in the canvas. `add` checks equal tensor shapes rather than broadcasting; `concat` uses `torch.cat` along its declared axis.

An optional graph-level `component_storage: "named"` generates an `nn.ModuleDict` with keys derived from definition IDs. Adding or reordering other definitions then does not renumber an unchanged component's checkpoint key. The default for existing declarations remains the original `nn.ModuleList` layout. Switching layouts or changing a definition ID changes checkpoint keys; users must explicitly regenerate source and plan weight migration. Stable names do not prove parameter shapes, values or model behavior are compatible. The canvas exposes the choice only when the selected CLI advertises support.

The generation plan includes `component_prefixes`, a definition-by-definition comparison of legacy positional and named state-dict prefixes for the current graph. The workspace shows a bounded preview and the CLI report carries the full list. These are not a migration plan: the old checkpoint may have come from another graph or definition order, and the graph alone does not expose every state key, shape, dtype or alias group. Verify those against both models and complete the explicit weight migration plan before migrating.

A factory may also be `component:namespace/name@VERSION`. Source generation resolves the registered Layer without importing it, validates and fills its parameter schema, and records the manifest and dependency references in the plan and `GRAPH_COMPONENT_REFS` in the generated file. The plan is recomputed at apply time. These references document generation-time provenance; generated Python does not itself enforce a component lock or install dependencies at runtime, so environment and code compatibility still require verification.

For an existing non-registered factory, the parameter editor can explicitly rediscover the factory in the selected Python and use its observed signature as an editing hint. Import and introspection can run user code and require the normal execution review; selecting another factory does not change the node. A missing signature leaves manual parameter editing available. Signature observation does not instantiate the component or establish forward compatibility. Registered component selection shows up to 512 reported entries and warns at the limit; it does not install dependencies.

Observed signature hints are cleared when the selected CLI, Python, project configuration, working/import paths or common interpreter-selection environment variables change. They are labelled as earlier observations: packages can still change in place without those identifiers changing, so rediscover after modifying an environment. No environment change is inferred to have succeeded merely from its path.

The runtime catalogue may report observed null, boolean, bounded integer and finite numeric constructor defaults. The parameter editor displays them as observations without adding them to the declaration; arbitrary objects and strings are not exposed as defaults. Positional-only and variadic signature entries are omitted from the keyword-parameter suggestions, because generated component factories receive keyword arguments. An existing explicit parameter can still be edited, and factories with required positional-only constructor arguments need an adapter.

The generated source includes runtime checks for explicitly declared `input_shapes` and `output_shapes`. At a checked port, the value must be a tensor of the declared rank and every numeric dimension must match; `null` dimensions accept any size. Errors identify the node, port, expected shape and actual shape. Unspecified shapes do not add checks. These checks do not establish device, factory behavior or model correctness, and are only exercised when the generated model is run.

Nodes can also declare `input_dtypes` or `output_dtypes` as objects mapping port names to PyTorch dtype attribute names such as `float32`. The canvas exposes this when the CLI advertises `project.graph_port_dtype_v1`. Generated code resolves each declared name against the installed PyTorch at runtime and checks the tensor's exact dtype; it does not cast. Invalid or unavailable names fail when the model runs. Missing declarations add no dtype checks. Planning does not import PyTorch or infer actual dtypes. For the bundled CSV adapter, differing declared dtype names produce an `input_dtype_review` advisory; names can be aliases, so the advisory is not proof of a mismatch. When a graph input has a dtype and a fully fixed sample shape, the CSV wizard can suggest that format after the user chooses matching columns.

Connected ports with different declared dtype names are listed in `dtype_advisories`, with both endpoints, and can be located in the graph plan. The canvas also asks for review when making such a connection. Because names may be aliases, these are advisories rather than static incompatibility errors; generated runtime checks remain authoritative for the chosen runtime.

`forward` accepts either one positional input mapping `{input_node_id: {output_port: value}}` or named inputs, and returns `{output_node_id: value}`. A graph with exactly one single-port input also accepts a single non-dictionary positional value. For a named input whose node has one output port, pass the value directly under the node ID. For a multi-port input, pass either a mapping of port names under the node ID, or separate keys `node_id.port` for all its ports. Unexpected or mixed duplicate input keys are rejected. This makes the model callable using a compatible batch's `args` and `kwargs`, but does not itself configure a batch factory. Values on fan-out edges are passed by reference; in-place mutation by a custom module may affect another branch. The generated file also exposes `model()` returning `GraphModel`. The graph is not yet automatically connected to project data, training or the scaffold declaration. A successful preview does not verify a factory, shape, dtype, device, runtime, or custom module behavior.

The optional graph `objectives` list records entries such as `{"output":"scores", "target":"label", "loss":"cross_entropy", "weight":1.0}`. The output must name an output node; the target names a batch target. The canvas can add, edit and remove several losses on an output. The generated module provides an editable managed-step entry `step(model, batch, context, objectives=None)`, which defaults to the declared objectives; callers can explicitly override them. It reads named targets from `batch["loss_inputs"]["targets"]`, or accepts a single unnamed target when all objectives use the same target name. It supports `mse`, `cross_entropy`, or a user `module:callable` loss, and reports individual losses as well as their weighted sum. MSE rejects broadcasting. Configure the matching batch source and training entry; generation does not change project configuration or run training. An empty objective list cannot train. Custom losses execute user code, and all compatibility remains unverified until runtime.

An optional `data` object records a batch `factory`, optional `binding`, and `params`. The canvas can configure a CSV adapter from an input or output node, using graph input IDs, multi-port `node_id.port` keys and declared objective target names as suggestions. From a saved graph the user can review and conditionally apply that declaration to the project batch configuration; this is a separate action and does not create a data snapshot. The CSV adapter limits field names to 128 characters; longer combined node/port names require shorter names or a custom batch factory. Paths and external data remain the user's responsibility, and neither selection nor saving proves that a batch can be loaded.

If a forward or trial request finds different project input settings, the graph canvas offers the existing reviewed application flow. If the project matches but session input overrides remain, it offers to clear them explicitly. After confirmation, it checks the project and graph again and continues the original action; a conflict or uncertain save stops the flow. No factory runs merely to resolve the mismatch. If no graph data is declared, the user may explicitly continue with current project/session input, whose compatibility remains unverified.

The graph canvas also offers a class-folder image mapping through `model_maven_worker.data.presets:graph_image_batch`. It binds one named graph input and one named label target to the existing image indexing, sampling and transformation pipeline. The guide records user-selected split weights, active split, resize and batch size, and optionally applies horizontal flip to the training split only. The three ordered transform lists are part of the data declaration and can be edited explicitly; `split_transforms` and the older single `transforms` parameter are mutually exclusive. Small classes may leave a split empty. Multi-input data and additional labels need another user factory. Creating the declaration does not scan images or verify runtime dependencies; batch preview does.

For this specific built-in image factory, preview verifies its named BCHW input and label batch sizes, splits that known batch into samples and requests the image renderer. The preview records the factory's actual split, distinct from the preview command's optional split request. It uses the factory's declared transforms but does not apply project-level augmentation. Other arbitrary batch factories retain whole-output semantics because their batch axis is unknown. Rendering and training have not yet been manually accepted.

The workspace result describes the image-batch interpretation, rendered count and actual factory split and provides the local HTML path. It does not describe that batch as one unsplit factory result; generated files are not opened or transferred automatically.

The graph image batch checkpoints its own cursor together with hashes of the selected dataset, split and effective batch configuration. On a later batch, a changed identity raises an error before returning the batch and retains the previous cursor. Its cursor is separate from the older classification factory's cursor. These checks do not freeze external files or prove deterministic behavior of arbitrary user transforms; actual checkpoint/recovery acceptance remains pending.

For an explicitly configured graph image pipeline, input or output node details offer ordered transform editing for each split. Transform types and fields come from the selected CLI's augmentation catalogue; the user can add, edit, reorder and remove steps, including custom factory references. A shared explicit pipeline is copied to each split before editing; the active split must have a pipeline. Changes remain in the graph draft until saved and applied, and this editor does not execute or fully validate the transforms. Preview the intended split before training. This flow awaits hands-on acceptance.

During one worker process, graph image batches reuse prior file hashes only when device, inode, size, modification time and change time all match; the directory tree and metadata are rescanned, and selected files are hashed again before decoding. The cache holds at most two roots and 100,000 entries per root and is cleared when checkpoint state is loaded, so the first restored batch hashes the dataset again. Metadata-based reuse cannot detect changes hidden by a filesystem that reports unchanged identity metadata; it is not a filesystem snapshot. Legacy classification calls continue to use full hashing by default.

For datasets up to 100,000 samples, the graph image batch also caches the stratified split plan by dataset fingerprint, split ratios and seed, and retains at most two shuffled split/epoch orders. A changed fingerprint or configuration replaces this cache. It is process-local and cleared on recovery; runtime speed and end-to-end reproducibility have not yet been measured.

Input and output node menus can switch between image splits with positive weights. The change is an undoable draft edit; saving and applying the data configuration remains explicit. Static planning reports graph inputs and objective targets that this single-input, single-target image batch cannot supply. It does not validate image files, model shapes or package availability.

An input node may declare human-readable `semantics` keyed by its output ports; an output node may do so keyed by its input port. These labels persist with the graph and are editable before a model is run. They are annotations only: they do not create observed semantic marks, bind enforcement rules, or prove runtime coverage. Removing a labeled port requires clearing its label first. Older graph implementations that do not recognize the field must reject rather than discard it.

When a saved graph and generated source match and a current forward uniquely identifies the corresponding root-model port, its supervision panel shows the graph label. Choosing to create an observed mark prefills the label as its meaning; the engineer still supplies a role and confirms the mark under the existing source and observation checks. The label alone does not establish the binding.

Boundary ports may also carry `draft_rules`: bounded lists of finite or numeric-range rules with observe/require modes, or a range penalty with positive weight. The graph planner validates their declarations; they do not themselves affect generated code or training. After a current forward identifies a unique port with one current observed mark, its supervision panel can bind a selected draft rule using the existing conditional rule-add command. The operation rechecks graph and generated-source contents and rejects an already identical rule. Binding adds a separate rule to the observed-mark store; the draft remains in the graph and can subsequently diverge. Training applies the bound rule only when supervision is enabled. Port deletion requires first removing its draft rules.

Draft rules can be edited or removed at the selected boundary port. Existing bound rules are separate and do not change when a draft changes; review the mark and explicitly bind the revised rule if needed. The plane shows the count of draft rules without assuming their binding status. Mode labels describe recording, rejection or weighted loss only at supported execution boundaries; a declaration alone does not enable supervision.

With `project.graph_supervision_reference_v1`, a graph may also retain `supervision: {"file": "supervision.json", "sha256": "…"}`. The canvas can link a project-local bundle after reviewing its mark count, export existing supervision, and review an import from a saved graph. Packaging includes the referenced file and checks its digest. Import creates unbound marks: the target project must perform a new forward and rebind before use. Removing the reference does not remove the file or existing rules. Linking and packaging do not prove runtime compatibility; this workflow awaits hands-on acceptance.

The supervision panel compares each draft with rules on the single current mark and reports matching parameters, no equivalent, or unknown. Parameter equality does not establish an identity link or automatic synchronization. Removing a node reports its attached draft-rule count before confirmation.

The graph's optional `training` declaration stores a trial step count and optimizer settings. When the CLI advertises `project.graph_training_step_v1`, it may also store a custom `step` as `module:callable` and a `step_params` object; otherwise the generated graph step is suggested. With `project.graph_training_modes_v1`, `mode` can additionally select a manual step or external loop; these require an explicit entry. A manual step owns updates and can use fallback AdamW settings; an external loop owns its optimizer and stores no optimizer settings. After the saved graph and generated source have been checked, a trial offers the declared mode, entry and parameters for review. Changing the suggested entry prevents reuse of its parameters, optimizer and step-count suggestions. Existing optimizer settings can be retained while editing the same mode. The entry executes user code only during an explicitly started trial and remains unverified until then. Managed and manual steps review graph data before proceeding; an external loop manages its own data and skips that check. The graph and generated source are still checked before launch. These declarations do not update project configuration or perform training.

When `project.graph_training_recipe_v1` is advertised, `try` and `train` also accept the saved graph through `--training-recipe model-graph.json`. This uses its training mode, entry, parameters and applicable optimizer settings; specify the run length separately with `--steps`. The command uses the selected project model and data configuration: it does not apply the graph's data declaration or verify that the selected source was generated from this graph. When no step is declared, the generated-step placeholder resolves against the selected model. Review the model and data before starting; this CLI path has not yet had hands-on acceptance.

For a graph-launched managed trial, the graph draft, saved graph, generated source and declared data mapping are checked again before launch and after confirmation. A change cancels the launch and requires preparing it again. This is a local preflight check, not an atomic lock on arbitrary external source edits or data contents.

Static image-data review reports a missing or invalid label target and a missing or non-string dataset path even when no loss has been declared. It does not open the path or establish that image data can be loaded.

When opening a v1 graph, the workspace treats JSON `null` as absent only for fields that the graph backend defines as optional. A later save may omit those equivalent null fields. Unknown structure fields and nulls in non-optional fields are not silently removed; unsupported declarations remain on disk for inspection or a compatible version.

`graph-import-scaffold --recipe FILE --if-recipe SHA256 --model module:callable` previews an opaque wrapper graph for a version-1 sequential declaration. The command reads but does not import or execute the original model, nor write a graph. It requires unique names within each input or output group and retains declared data, output objectives and the supervision reference. The original model remains one call node: internal layers, routes, constructor settings and weights are not converted, while any declared training profile is carried forward for review. The user must retain the original source and declaration, review factory parameters and runtime compatibility, and import/rebind supervision separately. The workspace can review this proposal and open it as a new unsaved graph only if no graph already exists. This is a partial migration path, not a layer-by-layer conversion or proof of equivalent runtime behavior.

`project.graph_unicode_ids_v1` permits Unicode XID continuation characters in
graph identifiers, alongside ASCII hyphens, within the existing 128-byte limit.
The fixed-cell plane renders Unicode node IDs as stable `u:` tags and displays
the full ID when selected; the underlying declaration and generated names are
not changed. `project.graph_output_name_v1` lets an output node keep an
external name distinct from its unique internal ID. Migration uses this when
an input and output have the same name; generated outputs and loss lookup keep
the original external name. External output names must remain unique.

The same command can preview a version-1 image-classification declaration. Its wrapper assumes one positional input and one output; edited source may differ and must be checked. It carries the supervision reference into the graph, but classification dataset details, transforms, class names, loss, runtime settings, internals and weights are not independently translated into active graph fields; a declared training profile is carried forward for review. The workspace reviews the same proposal using the project's selected model factory, with a revision check before opening the draft. Neither preview changes the original files.

With `--if-project CONFIG_REVISION`, the preview also checks the current project
model and configuration revision. If the declaration has no data binding, the
saved project batch factory, binding and parameters are copied into the graph
proposal; the result identifies their source as `project_config`. Declaration
data takes precedence. This preserves an entrypoint, not proof that its output
matches the graph, and the copied parameters may contain sensitive values.

Graph training can omit trial steps and optimizer settings; absent values stay absent. A `$model:step` entry resolves against the selected model at launch and can therefore refer to a different source after switching models. Review the training entry, data and loss before running.

Both wrapper proposals also contain an inert `legacy_import` snapshot with the
exact original declaration text, its SHA-256 and its source kind. Planning and
saving reject a malformed or changed snapshot. It is retained for recovery and
review only: its data, training and other fields do not configure the graph, and
it contains no model source or weights. The snapshot may contain sensitive project
values and travels with the graph. The combined graph must fit the existing
128 KiB limit; otherwise the import is rejected and the original remains intact.

The `project.graph_cast_parameter_dtype_v1` capability adds a unary `cast`
node. With a `definition`, generated PyTorch code casts its input to that
component's first parameter dtype, falling back to the runtime default float
dtype when the component has no parameters. Without a `definition`, it uses
the runtime default float dtype directly. The node preserves shape; it is not
a component invocation, although the referenced definition is instantiated.
This is a building block for editable migration, not an equivalence guarantee.

With `--editable`, `graph-import-scaffold` instead proposes separate graph
nodes for inputs, their declared encoder routes, dynamic-axis mean reduction, default encoders
consisting of supported linear, activation, normalization, dropout and identity
layers, concatenation, and default named output heads. It keeps declared data, objectives, the supervision
reference and the inert original declaration. User factory layers are carried
as separate calls with an explicit flat-output/batch check. Residual and
parallel groups are expanded into isolated branch inputs, ordered calls,
batch checks and merge nodes. Declared encoder, fusion and head components
are represented as factories with their parameters; fusion receives an ordered
list, and a custom head must return a mapping with the declared output names.
Component availability and actual behavior remain unverified. The proposal is reconstructed from the declaration: edited source,
constructor overrides, weights and optimizer state are not recovered or compared. A declared training profile is retained, but its behavior with the new graph remains unverified. It has not established behavioral equivalence; retain
the original files and verify in the selected runtime.

`project.graph_dynamic_mean_v1` represents a unary `mean_dynamic` node. Its
`axes` are zero-based sample axes, excluding the batch axis. Generated PyTorch
code converts integer inputs to the runtime default floating dtype, retains
floating or complex inputs, and takes the mean with dimensions retained. The
planner propagates reduced dimensions only when the input shape is declared.

`project.graph_factory_contract_v1` permits ordered `dtype_definitions` on a
cast and an `assert_flat` node with value and batch-reference inputs. It checks
that the value is a rank-two tensor with the declared width and the reference
batch size. The cast searches definitions in the listed order for the first
parameter. The supported single-chain reorder operation synchronizes the
corresponding segment of this list and reports that change. Other graph edits
do not automatically rewrite it. Migrated linear stacks set
`track_sequence`, so planning refuses a connected layer chain whose order no
longer agrees with the list, or whose structure is no longer a simple chain.
The workspace can edit the list or explicitly turn tracking off. Incomplete
drafts may still be saved for later repair.

`project.graph_clone_node_v1` provides an explicit tensor copy;
`project.graph_assert_batch_v1` checks rank and batch size, and
`project.graph_after_node_v1` expresses an execution dependency while passing
the first input through. Nested migrations retain an explicit dtype source
order without automatic linear-chain tracking, so edits require review. Linear
stacks feeding a custom fusion also retain explicit untracked dtype order.

`project.graph_list_call_v1` allows a call node to pass its ordered input
ports as one list argument; graph planning does not import the target factory.

It also checks the active image split, positive batch size and the basic relationship between shared and per-split transform declarations. It does not validate the transforms' runtime behavior, dataset contents or split coverage.

Static review of the supplied CSV adapter also checks the declared path and split/weight/seed syntax. `model_csv.py` is editable, so this review describes the supplied adapter contract and does not prove that a customized factory behaves the same way or that the file can be read.

Graph planning reports `data_review` separately from graph connectivity. For the bundled `model_csv:batch` it flags missing or unexpected named inputs, missing objective targets, invalid target fields, overlong input names, non-envelope bindings, invalid declared input field shapes and known conflicts between a CSV input's declared batch shape and the graph input contract. An anonymous CSV target is compatible only when all objectives use the same target name, matching the managed training step. `mapping_consistent` describes these limited static checks only; it does not read the file or verify column contents, runtime dtypes, model behavior or training. Other factories have an unknown static contract. These diagnostics do not prevent saving or source generation.

When an output has a connected, declared shape, CSV objective review also compares the declared batch target shape for MSE and cross entropy. Cross entropy allows either class-index or probability target shapes; dtype and value constraints remain runtime concerns. Each review reports shape consistency, conflict or unknown; custom losses and incomplete shape information remain unknown. These conditional checks do not establish that the model or loss actually runs.

CSV field review also checks the adapter's allowed field keys and the syntax of a declared dtype name without importing PyTorch. For cross entropy, when the declared shapes distinguish index targets from probability targets, a limited set of familiar integer and floating dtype names can produce a `target_dtype_review` advisory. The advisory is not a complete dtype compatibility check; custom names, actual values and runtime behavior remain unverified.

The canvas also accepts an existing batch `module:callable`, its binding and JSON parameters without importing or executing the factory during declaration. The user must supply a batch compatible with the graph and selected training step; the optional preview and forward workflows execute that factory after review.

The graph canvas can open the existing trial workflow across sessions: it regenerates a read-only source proposal from the saved graph and compares its hash with the configured project model file before suggesting the declared custom step or generated managed step. This is a convenience, not a persistent graph-to-training binding. If the file has been edited or a different generator version produces different source, use the workspace trial flow and select the appropriate entry after reviewing the code. Trial preparation does not itself run training.

The canvas also offers formal training via `Y`. It uses the same graph/source check and declared mode, entry, parameters and optimizer suggestions, but asks for the formal step count separately. Managed and manual modes review graph data; an external loop owns its data. Immediately before process launch the canvas checks the graph, generated source, project and applicable data again. Existing training confirmation, request tracking and supervision review remain in effect. This path has not yet had hands-on acceptance.

From the canvas, `R` opens project-wide run history and `Q` opens pending-request inspection. The existing run browser offers the actions available for the selected record, including progress, comparison, evaluation and supported resume or export flows. The list is not automatically filtered to this graph, and an acknowledged start is not evidence of completion. These canvas shortcuts await hands-on acceptance.

When generating another version, the canvas can create an unused module or review a conditional update of the current project model. Updating shows a diff of the exact saved file, refuses dirty or pending editor buffers, and requires a matching file digest at commit; the source-edit journal retains the previous contents for recovery. This is a replacement, not an automatic merge: the diff shows any hand-written changes that would be removed. The TUI limits this review to files of 128 KiB; larger files can be retained while creating a new module. Switching or binding a project model requires a separate configuration review. No data factory runs during configuration.

CLI clients request an update explicitly by passing `--if-source SHA256` together with `--output` to both the preview and its `--apply --if-plan` call. The plan includes the expected prior digest. At commit the graph and source are checked again under the source-edit lock; an external modification produces a conflict. An unchanged source reports `unchanged` without creating a journal entry. The ordinary invocation without `--if-source` still only creates a new file. Graph saving, source updating and project binding are separate operations, so a completed source update does not claim that project configuration or training has been updated.

`graph-draft-probe --graph DRAFT --if-graph SHA256 --snapshot FILE --input-sha256 SHA256 --if-config REVISION` renders a structurally complete draft into memory and executes it in a fresh worker against the previously captured input. It does not save the graph or generated source. Input digest, graph digest, source digest, replay format and execution evidence are checked before success is returned. The draft model is newly constructed; learned weights are not copied. This process isolation is not an operating-system sandbox: component factories and model code can have side effects. Missing weight, environment or source evidence remains unknown, and the captured input remains available after failure. The workspace uses this path for unsaved drafts; saved graphs use independent source-checked replay.

The draft preview uses the selected Python, project import roots, working directory and optional forward seed. `--model-device DEVICE` explicitly moves the newly constructed model using the same device resolution as a regular probe; it does not automatically move the restored input. A device mismatch can therefore fail the forward and is reported as such.

From a graph input or output the user can preview its declared batch through the existing batch preview workflow. Forward from the canvas uses the project's existing model probe after checking the saved graph, generated source, and applied batch declaration; session input overrides must be cleared. Both operations execute selected user code after the existing review. A batch preview does not prove model compatibility; a successful forward is evidence only for the actual input and runtime used.

When a canvas input snapshot is captured and replayed, the workspace compares available fingerprints for registered state before forward, the reported runtime environment, and the reported model-class source. Missing fingerprints remain unknown; identical fingerprints do not cover unregistered state, all dependencies, devices or side effects. A replay is described as passed only when independent forward execution and the captured input digest are both confirmed. The snapshot stays available for further replay until explicitly discarded or the workspace closes.

If a failed or unconfirmed forward still reports a verified replay of the captured input, the result distinguishes the matching input from the unsuccessful or unconfirmed forward and displays available identity comparisons. A matching input alone never establishes a passing forward.

For a current failed forward, the canvas can compare recorded project stack frames with the generated source hash and node line map and focus a possible failing node. The file digest is observed after the exception and does not prove which bytes Python loaded; a matching frame is a navigation hint, not a root-cause diagnosis. Failures without a matching generated `forward` frame remain unlocated.

The graph canvas can also capture one forward input and later replay that private temporary snapshot through the existing workspace workflow, including after generating and selecting another graph model in the same workspace. A graph without declared data can use a reviewed current project or session input. Capture and replay recheck graph and source immediately before launch; capture also checks any declared data binding. The snapshot remains available after replay and is removed when discarded, replaced, or the workspace closes. This confirms only the captured input, not identical weights, runtime, or future batches. Previewing an unsaved graph edit against that snapshot is not yet supported; this path awaits hands-on acceptance.

After a current successful forward, an input port or output node can open the existing semantic supervision panel when its corresponding runtime port is uniquely observed. For multi-port inputs the user chooses the port; flat and nested input forms are matched separately. The panel operates on observed runtime ports and saved marks; a graph declaration or static plan does not establish observation, rule coverage or a passing result.

To create the reviewed source, repeat the same command with `--apply --if-plan PLAN_REVISION`. The plan binds the absolute project root, graph bytes, generated source, output name and node line map. The output must be a new ASCII Python module filename in the project root; existing files are never overwritten. The receipt contains `model` as `module:model`. This only creates source; it does not modify project configuration, copy the graph declaration, import a factory, validate runtime behavior or start training. Keep the original graph file to reconstruct or edit the graph later. The normal project import and data-binding workflows remain explicit.

## Conditional draft save

`graph-edit-plan --graph FILE --if-graph SHA256 --operations FILE
--if-operations SHA256` previews 1–128 operations against a single graph
revision. Supported actions are `add_node`, `update_node`, `copy_node`,
`remove_node`, `add_definition`, `update_definition`, `remove_definition`,
`set_parameter`, `remove_parameter`, `connect`, `disconnect`, `move_before`
and `move_after`. Nodes, definitions and edge endpoints are addressed by stable
IDs; a node removal with dependent edges, objectives or group membership needs
explicit `cascade: true`. A copied call shares its definition and does not copy
edges. Shared-definition edits report the number of affected nodes. Parameter
values, including JSON null, are preserved as supplied. Unknown actions and
fields are rejected, and the complete candidate is structurally planned before
it is returned. Repairable shape, axis and tracked-dtype conflicts are returned
as a failed structural review inside the successful edit preview, so a draft can
be fixed in steps; cycles and other structural errors reject the batch. No
source or graph file is changed. The caller can persist the
returned candidate with `graph-save`, using the original saved graph revision
as its conditional base; a preview alone is not a commit or runtime validation.

`graph-bind --graph model-graph.json --if-graph SHA256
--source MODULE.py --if-config CONFIG_REVISION` previews binding the saved graph's
generated model source to the project. Add `--include-data` to bind the graph's
data factory, binding and parameters in the same versioned configuration update.
The plan explicitly clears previous model parameters and injections. Repeat with
`--apply --if-plan PLAN_REVISION` to commit; the operation rechecks the graph,
generated-source bytes and configuration revision under cooperative locks. The
graph and source remain separate files, and training and supervision remain in
the graph. External editors that ignore cooperative locks can still race; this
does not execute factories or prove runtime compatibility. For a new project,
pass `--if-config absent`; the same reviewed operation creates the initial
configuration with exclusive creation rather than overwriting an existing one.

`m-model --json graph-save --draft DRAFT --if-draft SHA256 --output model-graph.json --if-base absent` previews creating a graph file in the project root. For an existing graph, pass the SHA-256 of its exact current bytes instead of `absent`. The target name must be a simple ASCII JSON filename, and an existing target must parse as a supported version-1 graph; unrelated project files are refused. The draft must parse as a version-1 graph, but it can have unconnected ports, cycles or other semantic errors. The preview includes `structural_review` with either the planner result or its diagnostic; saving a draft does not certify it for generation.

Repeat the request with `--apply --if-plan PLAN_REVISION` to save. The plan binds the canonical project root, destination, base digest, draft digest, unchanged status and structural review. The writer uses a project-root cooperative lock, rechecks the target and uses exclusive creation or staged replacement. Other Model-Maven graph saves respect the lock; an external editor that ignores it can still race the last check and replacement, so users should reconcile concurrent external edits. Existing target symlinks and unsupported files are rejected. The draft's exact bytes are stored; no model code or project configuration is changed.

### Staged graph delivery and recovery

`graph-delivery-plan` is a read-only overview of the graph, generated source and project binding. An explicit `graph-delivery-start` pins its plan revision, persists the exact graph draft in a private project-local journal, then conditionally saves the graph, generates source and binds the project model and declared data. These are separate file commits, not an atomic transaction. Training, loss and supervision remain graph declarations; binding does not enable them or run factories.

If a stage fails or the process is interrupted, the journal remains. `graph-delivery-status` reports whether each current target matches its reviewed before or after revision, conflicts, or cannot be inspected. The status is a snapshot and cannot reserve the targets. `graph-delivery-resume --if-plan SHA256` pins the reviewed recovery record and compares each actual file or configuration revision with the journal's reviewed before/after revisions, skips an already completed stage, and stops if a target has another revision. It retains the record for further inspection rather than overwriting the external change. A successful final identity check removes the record. The journal contains the graph draft and is stored under `.model-maven/graph-delivery.pending.json` with private file permissions; handle it as project data. `graph-delivery-archive --if-plan SHA256` preserves the reviewed recovery record in a private, uniquely named archive and releases the pending slot; it does not roll back already written files. Recovery does not restore a previous state automatically and does not establish that model, data or runtime behavior is correct.

### Generic split-specific batch transforms

The optional `model_maven_worker.data.pipeline:batch` adapter wraps an explicitly selected batch factory and its parameters. Its `split_transforms` maps split names to ordered, path-targeted transform steps; `split` selects one list and `seed` is explicitly supplied. The same seed is used on every call, so stochastic behavior may repeat. A custom step specifies a target such as `kwargs.features`, an explicit `module:callable`, and keyword parameters. The selected split controls the transform list only: the adapter does not infer or rewrite the source factory's split arguments. The source must return a batch; recognized PyTorch Dataset objects must first be batched by a user-selected factory. Separately configured project augmentation may also run. Selection and preview do not prove that the source, transform, or model contracts are compatible.

### Workspace compatibility for declarations

The CLI advertises `project.graph_boundary_annotations_v1` for preserving and
validating boundary `semantics` and `draft_rules`, and
`project.graph_data_objectives_v1` for preserving the graph's data and
objective declarations. The unified workspace requires both before editing;
an older CLI is rejected instead of being allowed to discard unfamiliar
fields. Existing graphs also require the advertised capabilities for any
stored port shapes, dtypes, call output mapping, named component storage,
supervision reference and training mode/step they use. Capabilities describe
parsing and the documented operation contracts; they do not establish that a
particular factory, device, training run or external library will work.
