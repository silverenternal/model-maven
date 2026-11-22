# Classification presets

This example is a thin project adapter over Model-Maven's native model, data,
and managed-step interfaces. The resulting `nn.Module`, batch factory, and
`training_step` use the same `inspect`, `try`, and `train` operations as custom
project code.

Copy `classification.py` into the root of a Model-Maven project. Put images in
class folders such as `data/cats/*.jpg` and `data/dogs/*.jpg`. The sample batch
factory reads those files in place, fingerprints them, applies deterministic
class-folder sampling and the declared transforms, and returns image tensors,
labels, and sample identity. It does not rearrange or delete the source data.

For a two-class Tiny CNN budget try, from the project root:

```sh
m-model init --model classification:tiny_cnn
m-model try \
  --model classification:tiny_cnn --model-param num_classes=2 \
  --batch classification:batch --batch-param root='"data"' \
  --batch-param batch_size=4 --step classification:step \
  --training-mode managed-step --steps 2 --lr 0.001
```

The command creates a real image batch and sends it through the normal managed
AdamW step engine, including loss, gradient, optimizer-state, and finite-value
checks. `m-model init` only saves the root model entry; try-time flags keep this
example self-contained.

`classification.py` also exposes `resnet18(num_classes, weights="random")`; it requires a compatible torchvision in the selected project environment. Random initialization is offline. `weights="imagenet1k-v1"` reads the pinned
official ResNet18 checkpoint from the torch hub cache. If it is absent, the
factory returns `PRETRAINED_WEIGHTS_NOT_CACHED` and does not access the network.
An application that offers explicit download must pass `allow_download=True`
only after the user approves that operation. The checkpoint's URL, cache path,
and SHA-256 are attached to the model as `model_maven_weight_plan` so the
application can include that evidence in the run record; the task-specific
classifier head is newly initialized.

The default Tiny CNN pipeline resizes to 32×32 and converts to float32 CHW.
For ResNet18, pass an explicit transform list to `batch` with Resize(224),
ToTensor, then ImageNet Normalize. The transform registry preserves this order
and reports the actual ports. Small datasets may yield fewer samples than the
requested batch size; the sampler never borrows from validation/test or invents
examples. The batch factory advances a deterministic epoch shuffle and exposes
checkpoint state hooks so training resumes at the next sample.

This is a starting point for image classification, not a restriction on the
product: custom projects can replace the model, batch, transforms, or training
step with native Python callables and continue through the same core protocol.
