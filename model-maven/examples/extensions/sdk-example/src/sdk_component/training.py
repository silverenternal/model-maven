"""A plain managed-step callable; the SDK only supplies optional type hints."""

from typing import Any, Mapping

from model_maven_sdk import StepContext
from .losses import classification_loss


def training_step(model, batch: Mapping[str, Any], context: StepContext):
    logits = model(*batch["args"], **batch["kwargs"])
    labels = batch["loss_inputs"]["labels"]
    loss = classification_loss(logits, batch)
    accuracy = (logits.detach().argmax(-1) == labels).float().mean()
    context.log("batch_size", labels.shape[0])
    return {"loss": loss, "metrics": {"accuracy": accuracy}}
