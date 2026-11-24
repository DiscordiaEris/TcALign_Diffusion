import unittest

import torch
from torch import nn

from models.faceid_adapter import install_frozen_lora


class FaceIDLoRATest(unittest.TestCase):
    def test_frozen_lora_matches_fused_output_without_replacing_base_weights(self):
        torch.manual_seed(5)
        layer = nn.Linear(4, 3)
        original = layer.weight.detach().clone()
        down, up = torch.randn(2, 4), torch.randn(3, 2)
        install_frozen_lora(layer, down, up)
        x = torch.randn(2, 4, requires_grad=True)
        expected = nn.functional.linear(x, original + up @ down, layer.bias)
        actual = layer(x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(layer.weight, original)
        actual.square().mean().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertEqual(list(layer.faceid_lora.named_parameters()), [])

    def test_installing_twice_is_rejected(self):
        layer = nn.Linear(4, 3)
        install_frozen_lora(layer, torch.zeros(2, 4), torch.zeros(3, 2))
        with self.assertRaises(ValueError):
            install_frozen_lora(layer, torch.zeros(2, 4), torch.zeros(3, 2))


if __name__ == "__main__":
    unittest.main()
