# Sequential order in inspection reports

For an exact built-in `torch.nn.Sequential`, the CLI inspection module record
may include `sequential_children`, an array of the registered direct child names
in forward order. The names are relative to that record's `path`. The field is
not emitted for differently named Sequential subclasses, which may override
`forward`. A non-built-in class named `Sequential` is explicitly marked with
`sequential_children: null` so it cannot be mistaken for an older report.

The TUI uses this declaration only when every name is nonempty, unique, has no
dot, and resolves to exactly one reported direct child; every reported direct
child must also be listed. Missing, duplicate or filtered records and invalid
declarations therefore produce no inferred Sequential edges. When the report
lists a shared child's registered paths as aliases and provides its instance
ID, the TUI can project those paths into separate positions in the graph. Those
positions refer to the same CLI-reported instance, not separate module objects;
the ID is scoped to that report. An alias with conflicting owners or missing
identity is not projected. Expansion is bounded to 4,096 graph records. The
array establishes the built-in container's structural order, not a completed
forward execution or tensor provenance.

When the field is absent, the existing numeric-only compatibility rule remains
in effect. A null or otherwise invalid field never falls back to that rule.
Older CLIs therefore remain usable, but their numeric-only compatibility rule
cannot distinguish a custom class also named `Sequential` from the built-in
container. A named Sequential without this field stays a module tree.
Both the CLI report change and this TUI change must be distributed together to
enable the new behavior for real projects.

The repair preview's difference view compares only links backed by an explicit,
valid `sequential_children` array. It uses the same direct-child and alias
validation as the structure graph, but does not apply the legacy numeric-only
fallback. `=` means the same ordered pair was declared in both reports; `?`
means the pair is not established on one side. This is not an assertion that
the actual connection was added or removed, nor that either forward ran. The
inspection snapshot and preview base are not necessarily the same revision;
the evidence page retains that warning. Missing or over-4,096-record reports
are reported as unavailable rather than treated as having zero links.

When at least one explicit pair is present, `g` switches the preview difference
between the report summary and a paired node-and-arrow diagram. Both sides use
the same ordered list of pairs and share scrolling; narrow terminals switch
sides with Tab. A pair missing on one side stays `?`, with no arrow drawn on
that side. The requested repair target is marked where it appears. Diagram and
summary scroll positions are kept independently, and the evidence page remains
available. The diagram presents each declared pair separately; it is not a
complete runtime graph or a view of exact tensor ports.
