import torch

_BATCH_INDEX = 0


def changing_batch():
    """Alternate valid sequence lengths so one trial exercises dynamic dimensions."""
    global _BATCH_INDEX
    frames, tokens_length = ((2, 3), (4, 5))[_BATCH_INDEX % 2]
    _BATCH_INDEX += 1
    targets = torch.tensor([0, 3])
    text_mask = torch.ones(2, tokens_length, dtype=torch.bool)
    text_mask[1, -1] = False
    return {
        "args": [
            torch.randn(2, frames, 3, 8, 8),
            torch.randint(0, 32, (2, tokens_length)),
            torch.tensor([frames, max(1, frames - 1)]),
            text_mask,
        ],
        "kwargs": {},
        "loss_inputs": {"targets": targets},
    }


def _batch_state_dict():
    return {"batch_index": _BATCH_INDEX}


def _load_batch_state_dict(state):
    global _BATCH_INDEX
    index = state.get("batch_index")
    if type(index) is not int or index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    _BATCH_INDEX = index


changing_batch.state_dict = _batch_state_dict
changing_batch.load_state_dict = _load_batch_state_dict
