"""Exact, bounded forward-input snapshot round trips without object loading."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from model_maven_worker.data.content_identity import fingerprint
from model_maven_worker.data.input_snapshot import SnapshotError, encode, read_file, restore, write_new


class InputSnapshotTests(unittest.TestCase):
    def test_roundtrip_preserves_content_types_aliases_and_tensor_metadata(self):
        tensor = torch.tensor([[1.0, -0.0]], requires_grad=True)
        shared = [tensor, {"value": tensor, "none": None}]
        args = (shared, shared, torch.empty(0, dtype=torch.bfloat16), -0.0, 7, True)
        kwargs = {"mask": torch.tensor([[True, False]]), "name": "样本"}
        digest = fingerprint(args, kwargs, torch)["digest"]
        restored_args, restored_kwargs = restore(encode(args, kwargs, torch), torch, expected_digest=digest)
        self.assertEqual(fingerprint(restored_args, restored_kwargs, torch)["digest"], digest)
        self.assertIs(restored_args[0], restored_args[1])
        self.assertIs(restored_args[0][0], restored_args[0][1]["value"])
        self.assertTrue(restored_args[0][0].requires_grad)
        self.assertIsNot(restored_args[0][0], tensor)
        self.assertEqual(restored_kwargs["name"], "样本")

    def test_different_inputs_and_tampering_are_rejected(self):
        args = (torch.tensor([1., 2.]),)
        digest = fingerprint(args, {}, torch)["digest"]
        data = encode(args, {}, torch)
        changed = fingerprint((torch.tensor([1., 3.]),), {}, torch)["digest"]
        with self.assertRaises(SnapshotError):
            restore(data, torch, expected_digest=changed)
        document = json.loads(data)
        document["tree"][2][0][2][0][6] = "AAAA"
        with self.assertRaises(SnapshotError):
            restore(json.dumps(document).encode(), torch, expected_digest=digest)
        with self.assertRaises(SnapshotError):
            restore(b'{"format":1,"format":2}', torch, expected_digest=digest)

    def test_unsupported_views_cycles_and_types_are_rejected(self):
        cycle = []
        cycle.append(cycle)
        for value in [torch.ones(3, 4).t(), torch.ones(8)[:4], object(), cycle, {1: "x"}]:
            with self.subTest(value=type(value).__name__), self.assertRaises(SnapshotError):
                encode((value,), {}, torch)

    def test_invalid_tensor_shape_or_reference_is_rejected(self):
        args = (torch.ones(2),)
        digest = fingerprint(args, {}, torch)["digest"]
        document = json.loads(encode(args, {}, torch))
        document["tree"][2][0][2][0][3] = [64 * 1024 * 1024]
        with self.assertRaises(SnapshotError):
            restore(json.dumps(document).encode(), torch, expected_digest=digest)
        document["tree"] = ["ref", 999]
        with self.assertRaises(SnapshotError):
            restore(json.dumps(document).encode(), torch, expected_digest=digest)

    def test_files_are_new_private_and_not_followed_at_the_final_component(self):
        payload = encode((torch.ones(2),), {}, torch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.snapshot"
            write_new(path, payload)
            self.assertEqual(read_file(path), payload)
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                link = Path(directory) / "link.snapshot"
                link.symlink_to(path)
                with self.assertRaises(OSError):
                    read_file(link)
            with self.assertRaises(FileExistsError):
                write_new(path, b"other")
            self.assertEqual(read_file(path), payload)
if __name__ == "__main__":
    unittest.main()
