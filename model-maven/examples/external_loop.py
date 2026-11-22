"""A complete user-owned PyTorch loop with explicit resumable state."""

import torch
from torch import nn
import torch.nn.functional as F


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, inputs):
        return self.linear(inputs)


def build_model():
    return TinyClassifier()


def batch_at(cursor):
    generator = torch.Generator().manual_seed(1000 + cursor)
    inputs = torch.randn(8, 4, generator=generator)
    targets = torch.arange(8) % 3
    return inputs, targets


class ExampleLoop:
    """The CLI calls state_dict/load_state_dict around checkpoint boundaries."""

    def __init__(self):
        self.optimizer_state = None
        self.cursor = 0

    def __call__(self, model, context, lr=0.01):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        model.train()
        while context.remaining_steps:
            if context.stop_requested:
                break
            inputs, targets = batch_at(self.cursor)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            # Keep the exact loop cursor and optimizer state aligned with the
            # model before the CLI observes this completed update.
            self.cursor += 1
            self.optimizer_state = optimizer.state_dict()
            accuracy = (logits.detach().argmax(-1) == targets).float().mean()
            if not context.report_step(loss.detach(), {"accuracy": accuracy}):
                break
        return {"data_cursor": self.cursor}

    def state_dict(self):
        return {"optimizer": self.optimizer_state, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.optimizer_state = state["optimizer"]
        self.cursor = int(state["cursor"])


run_loop = ExampleLoop()
