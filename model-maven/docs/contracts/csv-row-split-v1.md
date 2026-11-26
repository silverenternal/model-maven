# CSV row partitions v1

The editable `model_csv.py` produced by a CLI advertising `data.csv_batch_row_split_v1` accepts the optional batch parameters `split`, `split_weights` and `split_seed`. This capability describes the generated template; existing or edited project adapters are not upgraded and may not implement these parameters.

To enable a partition, set `split` to `train`, `val` or `test`, provide all three nonnegative integer weights (at most 1,000,000 each, with a positive sum), and provide a nonnegative integer seed at most 2^63-1. Omit all three parameters to retain the original all-rows behavior and checkpoint fingerprint. The TUI limits seeds to the exactly representable integer range of its Lua runtime.

For each CSV data row, the adapter hashes the seed and zero-based row position and maps it into the weighted buckets. Identical CSV row order, weights and seed give disjoint, reproducible partitions. This is not stratified; a small partition may be empty and raises an error. Reordering or inserting rows changes membership. Use stable source data and inspect the actual split before relying on it.

The source bytes, normalized mapping, selected split, weights and seed contribute to the adapter's checkpoint fingerprint. Switching split or data mapping requires a separate batch instance/run; an existing checkpoint cannot be silently resumed on a different partition. All CSV rows are still parsed and checked, including rows outside the selected partition. The interface does not automatically choose separate training and evaluation bindings or define transforms.
