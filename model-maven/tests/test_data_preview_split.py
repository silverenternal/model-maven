import unittest

try:
    import torch
except ImportError:
    torch = None

from model_maven_worker.data.preview import _infer_batch_count, _split_preview_batch


@unittest.skipIf(torch is None, "PyTorch is required for tensor preview tests")
class PreviewBatchSplitTests(unittest.TestCase):
    def test_envelope_keeps_shared_metadata_and_splits_images_and_targets(self):
        batch = {
            "args": [torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)],
            "kwargs": {},
            "loss_inputs": {"targets": torch.tensor([4, 9]), "dataset": {"source": "disk", "classes": ["cat", "dog"]}},
            "metadata": {"split": "train", "epoch": 3},
        }
        count = _infer_batch_count(batch, 12)
        self.assertEqual(count, 2)
        samples = _split_preview_batch(batch, count)
        self.assertEqual(len(samples), 2)
        for index, sample in enumerate(samples):
            self.assertTrue(torch.equal(sample["args"][0], batch["args"][0][index]))
            self.assertEqual(sample["loss_inputs"]["targets"].item(), [4, 9][index])
            self.assertEqual(sample["loss_inputs"]["dataset"]["classes"], ["cat", "dog"])
            self.assertEqual(sample["metadata"]["split"], "train")
            self.assertEqual(sample["metadata"]["epoch"], 3)
            self.assertEqual(sample["kwargs"], {})

    def test_sequence_of_unbatched_samples_is_not_transposed(self):
        first = (torch.zeros(3, 4, 4), 7)
        second = (torch.ones(3, 4, 4), 8)
        samples = _split_preview_batch([first, second], 2)
        self.assertEqual([sample[1] for sample in samples], [7, 8])
        self.assertTrue(torch.equal(samples[0][0], first[0]))
        self.assertTrue(torch.equal(samples[1][0], second[0]))

    def test_tuple_with_shared_scalar_keeps_per_sample_values(self):
        image = torch.ones(2, 3, 4, 4)
        samples = _split_preview_batch((image, "shared"), 2)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[1][1], "shared")
        self.assertEqual(tuple(samples[0][0].shape), (3, 4, 4))


if __name__ == "__main__":
    unittest.main()
