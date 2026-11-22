"""Plain loss callable; it is an algorithm choice, not a required SDK class."""

from torch.nn import functional as F


def classification_loss(outputs, batch):
    return F.cross_entropy(outputs, batch["loss_inputs"]["labels"])
