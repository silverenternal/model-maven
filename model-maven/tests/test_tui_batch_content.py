"""Conservative forward-input content identity and explicit coverage limits."""

import unittest

import torch
from model_maven_worker.data.content_identity import MAX_BYTES, fingerprint


class ContentIdentityTests(unittest.TestCase):
    def identity(self, args, kwargs=None):
        return fingerprint(args, kwargs or {}, torch)

    def test_content_not_just_shape_and_dtype(self):
        first = self.identity((torch.tensor([[1.0, 2.0]]),))
        again = self.identity((torch.tensor([[1.0, 2.0]]),))
        changed = self.identity((torch.tensor([[1.0, 3.0]]),))
        self.assertEqual(first["status"], "available")
        self.assertEqual(first["digest"], again["digest"])
        self.assertNotEqual(first["digest"], changed["digest"])

    def test_types_structure_keywords_and_aliasing_are_significant(self):
        tensor = torch.tensor([1.0, 2.0])
        shared = self.identity((tensor, tensor), {"mask": True})
        copied = self.identity((tensor, tensor.clone()), {"mask": True})
        self.assertNotEqual(shared["digest"], copied["digest"])
        self.assertNotEqual(
            self.identity(([1, 2],))["digest"], self.identity(((1, 2),))["digest"]
        )
        self.assertNotEqual(
            self.identity((1,))["digest"], self.identity((True,))["digest"]
        )
        self.assertNotEqual(
            self.identity((), {"a": 1, "b": 2})["digest"],
            self.identity((), {"b": 2, "a": 1})["digest"],
        )
        self.assertNotEqual(
            self.identity((0.0,))["digest"], self.identity((-0.0,))["digest"]
        )

    def test_unsupported_inputs_do_not_get_a_digest(self):
        tensor = torch.arange(6).reshape(2, 3)
        cases = [
            ((object(),), {}),
            ((tensor.t(),), {}),
            ((tensor, tensor.view(-1)), {}),
            (({1: "value"},), {}),
            ((torch.empty(MAX_BYTES + 1, dtype=torch.uint8),), {}),
        ]
        for args, kwargs in cases:
            with self.subTest(args=type(args[0]).__name__):
                result = fingerprint(args, kwargs, torch)
                self.assertEqual(result["status"], "unsupported")
                self.assertNotIn("digest", result)

    def test_empty_scalar_and_bfloat16_tensors(self):
        for tensor in [
            torch.empty(0),
            torch.tensor(2.0),
            torch.ones(2, dtype=torch.bfloat16),
        ]:
            with self.subTest(dtype=tensor.dtype, shape=tensor.shape):
                result = self.identity((tensor,))
                self.assertEqual(result["status"], "available", result)
                self.assertEqual(len(result["digest"]), 64)


if __name__ == "__main__":
    unittest.main()
