import tempfile
import unittest
from pathlib import Path

from model_maven_worker.data.presets import classification_batch
from model_maven_worker.protocol import UserError


class ClassificationRetryTests(unittest.TestCase):
    def test_invalid_image_does_not_advance_batch_cursor(self):
        from PIL import Image
        classification_batch.load_state_dict({"version": 1, "cursors": {}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("cat", "dog"):
                (root / name).mkdir()
                (root / name / "image.png").write_bytes(b"invalid png")
            with self.assertRaises(UserError) as error:
                classification_batch(root=root, batch_size=1)
            self.assertEqual(error.exception.code, "CLASS_SAMPLE_LOAD_FAILED")
            self.assertEqual(classification_batch.state_dict()["cursors"], {})
            for name in ("cat", "dog"):
                Image.new("RGB", (8, 8), "red").save(root / name / "image.png")
            result = classification_batch(root=root, batch_size=1)
            self.assertEqual(result["dataset"]["coverage"]["actual_count"], 1)
            self.assertEqual(list(classification_batch.state_dict()["cursors"].values()), [1])
        classification_batch.load_state_dict({"version": 1, "cursors": {}})


if __name__ == "__main__":
    unittest.main()
