"""Project-local wrappers for Model-Maven's classification preset services.

Copy this file into a project directory as ``classification.py``. Keeping these
wrappers in the project makes the chosen preset callable through the same native
model, batch, and managed-step protocols as custom project code.
"""

from model_maven_worker.data.presets import (
    build_resnet18,
    build_tiny_cnn,
    classification_batch as _classification_batch,
    classification_step,
)


def tiny_cnn(num_classes, in_channels=3, width=16):
    return build_tiny_cnn(num_classes=num_classes, in_channels=in_channels, width=width)


def resnet18(num_classes, weights="random", allow_download=False):
    return build_resnet18(
        num_classes=num_classes,
        weights=weights,
        allow_download=allow_download,
    )


def batch(root, batch_size=8, split="train", seed=0, sample_offset=None):
    return _classification_batch(
        root=root,
        batch_size=batch_size,
        split=split,
        split_ratios={"train": 1.0},
        seed=seed,
        sample_offset=sample_offset,
        transforms=[{"type": "Resize", "size": 32}, {"type": "ToTensor"}],
    )


def step(model, batch, context):
    return classification_step(model, batch, context)


# The managed training/checkpoint engine persists this protocol when it sees a
# batch callable with state hooks, so resume continues at the next sample.
batch.state_dict = _classification_batch.state_dict
batch.load_state_dict = _classification_batch.load_state_dict
