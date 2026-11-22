# Explicit weight migration

Use weight migration when a model's structure has changed and a fresh training run should start from selected weights in a completed run. The parent checkpoint remains immutable; Model-Maven creates a new child run, starts its optimizer from empty state, and records the migration report and parent run ID.

```sh
m-model train \
  --model model:build_model \
  --batch data:build_batch \
  --step train:training_step \
  --steps 20 \
  --migrate-from RUN_ID \
  --migration-plan migration.json
```

The migration plan is a JSON object with `schema_version: 1` and three exhaustive lists:

```json
{
  "schema_version": 1,
  "preserve_compatible": [
    {
      "source_key": "encoder.weight",
      "target_key": "encoder.weight",
      "semantic_reason": "The encoder has the same input, output, and representation role."
    }
  ],
  "reset_target": ["head.weight", "head.bias"],
  "drop_source": ["classifier.weight", "classifier.bias"]
}
```

Every state-dict key must appear exactly once: preserved through an explicit mapping, initialized from the target model, or dropped. Preserved tensors require an exact shape and dtype match plus a non-empty semantic reason. Slicing, reshaping, widening, merging, and splitting require an explicit migration adapter and are not inferred from names or matching shapes.

Checkpoints record shared parameter/buffer alias groups. A preserved alias group must be mapped completely and bijectively to the same target alias group. A topology change cannot be described as preserving weights: reset the entire target alias group and drop the corresponding source group, or migration fails closed. Checkpoints without alias-group evidence cannot be used as migration sources.

Migration does not carry optimizer, scheduler, scaler, or step state into the child. The child starts at step zero. Ordinary `--resume` remains a same-run continuation and cannot be combined with `--migrate-from`.
