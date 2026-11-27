"""Contracts for a small trainable residual on frozen private FaceID features."""

import io
import unittest
from copy import deepcopy

import torch
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    IPAdapterAttnProcessor2_0,
)
from torch.nn import functional as F

from models.faceid_adapter import install_frozen_lora
from models.isolated_identity_attention import FrozenIdentityAttnProcessor


class IdentityRefinementTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)

    def attention(self):
        attention = Attention(
            query_dim=32, cross_attention_dim=12, heads=4, dim_head=8,
            bias=True, out_bias=True, dropout=0., norm_num_groups=4,
            residual_connection=True, rescale_output_factor=1.7,
        )
        for layer in (attention.to_q, attention.to_k, attention.to_v, attention.to_out[0]):
            install_frozen_lora(
                layer, torch.randn(2, layer.in_features) * .1,
                torch.randn(layer.out_features, 2) * .1,
            )
        attention.requires_grad_(False)
        processor = IPAdapterAttnProcessor2_0(
            hidden_size=32, cross_attention_dim=12, num_tokens=[4], scale=[2.],
        )
        return attention, processor

    def inputs(self, spatial=False, *, dtype=torch.float32):
        shape = (2, 32, 2, 3) if spatial else (2, 6, 32)
        return (
            torch.randn(shape, dtype=dtype, requires_grad=True),
            torch.randn(2, 5, 12, dtype=dtype),
            torch.randn(2, 1, 4, 12, dtype=dtype, requires_grad=True),
        )

    def test_rank_zero_preserves_default_architecture_and_outputs(self):
        attention, original = self.attention()
        default = FrozenIdentityAttnProcessor(attention, original)
        rank_zero = FrozenIdentityAttnProcessor(attention, original, identity_refinement_rank=0)
        self.assertFalse(hasattr(rank_zero, "identity_refinement"))
        self.assertEqual(set(default.state_dict()), set(rank_zero.state_dict()))
        self.assertFalse(any(parameter.requires_grad for parameter in rank_zero.parameters()))
        hidden, text, image = self.inputs()
        expected = default(attention, hidden, (text, [image]))
        actual = rank_zero(attention, hidden, (text, [image]))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_zero_initialized_rank_two_preserves_bnc_bchw_and_input_gradients(self):
        for spatial in (False, True):
            with self.subTest(spatial=spatial):
                attention, original = self.attention()
                baseline = FrozenIdentityAttnProcessor(attention, original)
                refined = FrozenIdentityAttnProcessor(attention, original, identity_refinement_rank=2)
                self.assertEqual(sum(p.numel() for p in refined.identity_refinement.parameters()), 128)
                self.assertGreater(float(refined.identity_refinement[0].weight.norm()), 0)
                self.assertEqual(int(refined.identity_refinement[1].weight.count_nonzero()), 0)
                self.assertIsNone(refined.identity_refinement[0].bias)
                self.assertIsNone(refined.identity_refinement[1].bias)
                trainable = {name for name, p in refined.named_parameters() if p.requires_grad}
                self.assertEqual(trainable, {
                    "identity_refinement.0.weight", "identity_refinement.1.weight",
                })
                hidden, text, image = self.inputs(spatial)
                mask = torch.tensor([[[0., 0., -10000., 0., 0.]]]).expand(2, -1, -1)
                expected = baseline(attention, hidden, (text, [image]), attention_mask=mask)
                actual = refined(attention, hidden, (text, [image]), attention_mask=mask)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                upstream = torch.randn_like(actual)
                expected_grads = torch.autograd.grad(expected, (hidden, image), upstream,
                                                     retain_graph=True)
                actual_grads = torch.autograd.grad(actual, (hidden, image), upstream)
                for got, wanted in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(got, wanted, atol=0, rtol=0)

    def test_up_learns_first_then_down_learns_without_mutating_frozen_state(self):
        attention, original = self.attention()
        refined = FrozenIdentityAttnProcessor(
            attention, original, train_identity_gain=True, identity_refinement_rank=2,
        )
        # The refinement training configuration retains saved gain values while
        # freezing them, so only the two residual matrices can be optimized.
        refined.identity_gain_logit.requires_grad_(False)
        attention.set_processor(refined)
        before = {name: value.clone() for name, value in attention.state_dict().items()}
        optimizer = torch.optim.SGD([p for p in attention.parameters() if p.requires_grad], lr=.3)
        hidden, text, image = self.inputs()
        attention(hidden, encoder_hidden_states=(text, [image])).square().mean().backward()
        down, up = refined.identity_refinement
        self.assertIsNotNone(down.weight.grad)
        self.assertEqual(int(down.weight.grad.count_nonzero()), 0)
        self.assertGreater(float(up.weight.grad.norm()), 0)
        optimizer.step()
        self.assertEqual(int(up.weight.count_nonzero()), up.weight.numel())
        torch.testing.assert_close(down.weight, before["processor.identity_refinement.0.weight"],
                                   atol=0, rtol=0)
        optimizer.zero_grad(set_to_none=True)
        attention(hidden, encoder_hidden_states=(text, [image])).square().mean().backward()
        self.assertGreater(float(down.weight.grad.norm()), 0)
        optimizer.step()
        for name, value in attention.state_dict().items():
            if name.startswith("processor.identity_refinement."):
                self.assertFalse(torch.equal(value, before[name]), name)
            else:
                torch.testing.assert_close(value, before[name], atol=0, rtol=0, msg=name)
        self.assertTrue(all(p.grad is None for name, p in attention.named_parameters()
                            if "identity_refinement." not in name))
        self.assertGreater(float(hidden.grad.norm()), 0)
        self.assertGreater(float(image.grad.norm()), 0)

    def test_refinement_uses_raw_identity_once_and_keeps_output_lora_without_bias(self):
        attention, original = self.attention()
        with torch.no_grad():
            attention.to_out[0].bias.fill_(.8)
        refined = FrozenIdentityAttnProcessor(attention, original, identity_refinement_rank=2)
        with torch.no_grad():
            refined.identity_refinement[1].weight.normal_(std=.2)
        hidden, text, image = self.inputs()
        captured = []
        hook = refined.to_out_identity.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        )
        actual = refined(attention, hidden, (text, [image]))
        hook.remove()
        raw = captured[0]
        lora = refined.to_out_identity.faceid_lora
        fixed_output = F.linear(raw, refined.to_out_identity.weight)
        lora_output = F.linear(F.linear(raw, lora.down), lora.up)
        residual_output = F.linear(F.linear(raw, refined.identity_refinement[0].weight),
                                   refined.identity_refinement[1].weight)
        text_output = AttnProcessor2_0()(attention, hidden, text)
        expected = text_output + refined.scale[0] * (fixed_output + lora_output + residual_output) / 1.7
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertGreater(float(lora_output.norm()), .01)
        self.assertIsNone(refined.to_out_identity.bias)

    def test_checkpoint_roundtrip_restores_nonzero_refinement_and_saved_gain(self):
        attention, original = self.attention()
        refined = FrozenIdentityAttnProcessor(
            attention, original, train_identity_gain=True, identity_refinement_rank=2,
        )
        with torch.no_grad():
            refined.identity_refinement[1].weight.normal_(std=.15)
            refined.identity_gain_logit.fill_(.73)
        hidden, text, image = self.inputs(spatial=True)
        expected = refined(attention, hidden, (text, [image]))
        stream = io.BytesIO()
        torch.save(refined.state_dict(), stream)
        stream.seek(0)
        restored = FrozenIdentityAttnProcessor(
            attention, deepcopy(original), train_identity_gain=True, identity_refinement_rank=2,
        )
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        torch.testing.assert_close(restored(attention, hidden, (text, [image])), expected,
                                   atol=0, rtol=0)
        self.assertIn("identity_gain_logit", restored.state_dict())
        self.assertTrue(all(p.requires_grad for p in restored.identity_refinement.parameters()))

    def test_refinement_follows_attention_dtype_and_zero_scale_skips_it(self):
        attention, original = self.attention()
        attention.double()
        original.double()
        refined = FrozenIdentityAttnProcessor(attention, original, identity_refinement_rank=2)
        self.assertTrue(all(p.dtype == torch.float64 for p in refined.identity_refinement.parameters()))
        hidden, text, image = self.inputs(dtype=torch.float64)
        refined.scale = [0.]
        actual = refined(attention, hidden, (text, [image]))
        expected = AttnProcessor2_0()(attention, hidden, text)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in refined.identity_refinement.parameters()))

    def test_invalid_refinement_rank_fails_clearly(self):
        attention, original = self.attention()
        for rank in (-1, 1.5):
            with self.subTest(rank=rank), self.assertRaisesRegex(ValueError, "rank"):
                FrozenIdentityAttnProcessor(attention, original, identity_refinement_rank=rank)


if __name__ == "__main__":
    unittest.main()
