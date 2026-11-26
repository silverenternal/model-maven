# Layer edit planning v1

`m-model --json recipe-layer-edit-plan --recipe DRAFT --if-recipe SHA256 --operations OPS --if-operations SHA256` reads two bounded regular JSON files. SHA-256 covers each file's exact bytes. Operations are an array of 1–64 objects applied in order to an in-memory copy; an error returns no plan and does not write the project. Pass arguments and parameters through the files rather than command-line JSON. The command does not import user code or require Python.

Each addressed layer and every layer in an edited group needs a unique `n_` plus 32 lowercase hex digits. Nested IDs are checked across the whole model. A result is a proposal, not a commit; save through the separately reviewed and revision-guarded scaffold workflow.

```json
[
  {"action":"add","slot":"input1_encoder","before":null,"after":null,"layer":{"type":"relu","id":"n_00000000000000000000000000000001"}},
  {"action":"set","id":"n_00000000000000000000000000000001","fields":{"type":"not-allowed"}}
]
```

The second operation above is intentionally invalid: `set` cannot change `type`, `id`, or branch structure, and the entire request fails. Supported operations are:

- `add`: `slot`, optional `before` or `after` anchor ID, and a complete `layer` with IDs on all nested layers. Without an anchor, append to the top-level slot. An anchor selects its existing group.
- `remove`: `id`. A nested branch cannot become empty; an empty top-level encoder stack becomes an identity encoder.
- `replace`: `id` and complete `layer`; retain the root ID and provide valid unique IDs for nested layers.
- `set`: `id` and nonempty `fields`: linear `features`, dropout `probability`, factory `reference`/`output_width`/`params`, or parallel `merge`.
- `copy`: `id`, optional `to_slot`, optional `before` or `after` anchor, and `new_ids` in preorder for the copied root and every nested layer. Without `to_slot`, the destination is the source group; with it, the destination is that encoder. Without an anchor, append to the destination. Cross-encoder copies require the `project.recipe_layer_cross_copy_plan_v1` capability.
- `move`: `id`, optional `before` in the same group. Null `before` appends.

Successful reports include the two input digests, per-operation action, affected ID and slot, resulting group order, a digest of added/replaced/modified layer data where applicable, a result declaration digest, and a static source-generation status. The result digest uses the backend's canonical serialization and is not a digest of the user's original formatting. The report omits parameter values. Static generation may report an incompatible or incomplete draft while still returning its structural plan; it never certifies actual shapes, imports, training or runtime behavior.
