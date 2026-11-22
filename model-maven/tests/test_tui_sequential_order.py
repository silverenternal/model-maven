"""The TUI may use only explicitly declared built-in Sequential order."""

import unittest
from collections import OrderedDict

import torch

from model_maven_worker.model.inspection import inspect_model


class CustomSequential(torch.nn.Sequential):
    def forward(self, value):
        return value


class Sequential(torch.nn.Sequential):
    def forward(self, value):
        return value


def inspect(model):
    return inspect_model(
        model,
        torch_module=torch,
        source_locations={},
        source_site_counts={},
        layer_container_kind_fn=lambda _module: None,
    )["modules"]


class SequentialOrderTests(unittest.TestCase):
    def test_named_sequence_keeps_registration_order(self):
        model = torch.nn.Sequential(
            OrderedDict(
                [("z_first", torch.nn.Linear(4, 4)), ("a_second", torch.nn.ReLU())]
            )
        )
        rows = inspect(model)
        self.assertEqual(rows[0]["sequential_children"], ["z_first", "a_second"])
        self.assertNotIn("sequential_children", rows[1])

    def test_custom_forward_has_no_sequential_order_claim(self):
        model = CustomSequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        self.assertNotIn("sequential_children", inspect(model)[0])
        same_name = Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        self.assertIsNone(inspect(same_name)[0]["sequential_children"])

    def test_shared_children_keep_both_registered_names(self):
        shared = torch.nn.Linear(4, 4)
        model = torch.nn.Sequential(OrderedDict([("first", shared), ("again", shared)]))
        rows = inspect(model)
        self.assertEqual(rows[0]["sequential_children"], ["first", "again"])
        self.assertEqual(rows[1]["aliases"], ["model.first", "model.again"])


if __name__ == "__main__":
    unittest.main()
