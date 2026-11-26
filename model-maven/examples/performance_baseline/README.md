# CPU model cost example

This synthetic residual MLP is repository-authored and covered by the root MIT
license. It has 8,499,300 parameters and uses a generated 34 MB weight file;
it is not a user model or representative of every production architecture.
No pretrained weights or data are downloaded. Run from this directory with a
user-selected Python environment that already has PyTorch:

```sh
python generate_weights.py
python measure.py
python measure.py
m-model --json model probe --model model:model --batch model:batch
m-model --json model probe model.blocks.0 --model model:model --batch model:batch
```

`measure.py` times imports, model construction, reading and applying weights,
input creation, and a full CPU forward separately with four PyTorch CPU threads.
It reports the process peak RSS where Python's `resource` module is available.
The selected-path CLI probe still runs the model; it is not a localized forward.
Weights are written to `weights.pt` in this directory; run in a copy if the
source tree should remain untouched. Neither the script nor these observations
measure cold disk cache, GPU memory, or other environments.
