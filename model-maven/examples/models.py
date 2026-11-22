from collections import OrderedDict
import torch
from torch import nn


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, left, right):
        return {"left": self.shared(left), "right": self.shared(right)}


def build_shared():
    return SharedModel()


def shared_batch():
    return {"args": [torch.randn(2, 4), torch.randn(3, 4)], "kwargs": {}}


class EditableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ("proj", nn.Linear(8, 8)), ("relu", nn.ReLU()),
        ]))
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        return self.head(self.encoder(x))


def build_editable():
    return EditableModel()


def editable_batch():
    return {"args": [torch.randn(2, 8)], "kwargs": {}}


class ScaledActivation(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, inputs):
        return torch.nn.functional.silu(inputs) * self.scale


def build_projection(out_features=8):
    return nn.Linear(8, out_features)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def forward(self, values):
        return self.projection(values)


def build_classifier():
    return Classifier()


def classification_batch():
    return {
        "args": [torch.randn(2, 4)],
        "kwargs": {},
        "loss_inputs": {"targets": torch.tensor([0, 2])},
    }


def classification_loss(outputs, batch):
    return nn.functional.cross_entropy(outputs, batch["loss_inputs"]["targets"])


def classification_training_step(model, batch, context, label_smoothing=0.0):
    """One user-owned model/loss step; the CLI owns backward and AdamW updates."""
    outputs = model(*batch["args"], **batch["kwargs"])
    targets = batch["loss_inputs"]["targets"]
    loss = nn.functional.cross_entropy(
        outputs, targets, label_smoothing=label_smoothing
    )
    accuracy = (outputs.detach().argmax(dim=-1) == targets).float().mean()
    context.log("accuracy", accuracy)
    return {
        "loss": loss,
        "losses": {"classification": loss.detach()},
        "outputs": outputs,
    }


MODEL_MAVEN_LAYER_EDIT_API = 1


def add_silu(model, operation):
    name = operation["payload"]["name"]
    model.encoder.add_module(name, nn.SiLU())
    return {"added": f"model.encoder.{name}", "type": "SiLU"}
