import unittest

import torch
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    IPAdapterAttnProcessor2_0,
)

from models.faceid_adapter import install_frozen_lora
from models.isolated_identity_attention import FrozenIdentityAttnProcessor


class IsolatedIdentityAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def attention(self, **kwargs):
        attn = Attention(
            query_dim=32,
            cross_attention_dim=12,
            heads=4,
            dim_head=8,
            bias=True,
            out_bias=True,
            dropout=0.0,
            **kwargs,
        )
        for layer in (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]):
            install_frozen_lora(
                layer,
                torch.randn(2, layer.in_features) * 0.1,
                torch.randn(layer.out_features, 2) * 0.1,
            )
        processor = IPAdapterAttnProcessor2_0(
            hidden_size=32,
            cross_attention_dim=12,
            num_tokens=[4],
            scale=[0.7],
        )
        processor.requires_grad_(False)
        return attn, processor

    def inputs(self, spatial=False):
        hidden = torch.randn(2, 32, 2, 3) if spatial else torch.randn(2, 6, 32)
        return hidden.requires_grad_(), torch.randn(2, 5, 12), torch.randn(2, 1, 4, 12)

    def identity_residual(self, isolated, attn, hidden, text, image):
        isolated.scale = [1.0]
        positive = isolated(attn, hidden, (text, [image]))
        isolated.scale = [0.0]
        return positive - isolated(attn, hidden, (text, [image]))

    def test_initial_output_and_input_gradients_match_original_with_lora_and_bias(self):
        # Losing either LoRA, double-counting output bias, or detaching hidden
        # breaks compatibility with the original frozen pretrained branch.
        for spatial in (False, True):
            with self.subTest(spatial=spatial):
                attn, original = self.attention(
                    norm_num_groups=4,
                    residual_connection=True,
                    rescale_output_factor=1.7,
                )
                isolated = FrozenIdentityAttnProcessor(attn, original)
                hidden, text, image = self.inputs(spatial)
                image.requires_grad_()
                mask = torch.tensor([[[0.0, 0.0, -10000.0, 0.0, 0.0]]]).expand(
                    2, -1, -1
                )
                expected = original(attn, hidden, (text, [image]), attention_mask=mask)
                actual = isolated(attn, hidden, (text, [image]), attention_mask=mask)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                upstream = torch.randn_like(expected)
                expected_grads = torch.autograd.grad(
                    expected,
                    (hidden, image),
                    upstream,
                    retain_graph=True,
                )
                actual_grads = torch.autograd.grad(actual, (hidden, image), upstream)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, atol=2e-6, rtol=2e-5
                    )

    def test_identity_residual_survives_text_projection_updates(self):
        attn, original = self.attention()
        isolated = FrozenIdentityAttnProcessor(attn, original)
        attn.set_processor(isolated)
        hidden, text, image = self.inputs()
        before = self.identity_residual(isolated, attn, hidden, text, image).detach()
        private_before = {
            name: value.detach().clone()
            for name, value in isolated.state_dict().items()
        }
        optimizer = torch.optim.SGD(attn.parameters(), lr=0.1)
        isolated.scale = [1.0]
        isolated(attn, hidden, (text, [image])).square().mean().backward()
        self.assertGreater(float(hidden.grad.norm()), 0)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in isolated.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in isolated.parameters())
        )
        before_text = AttnProcessor2_0()(attn, hidden, text).detach()
        optimizer.step()
        self.assertFalse(
            torch.allclose(before_text, AttnProcessor2_0()(attn, hidden, text))
        )
        torch.testing.assert_close(
            self.identity_residual(isolated, attn, hidden, text, image),
            before,
            atol=2e-6,
            rtol=2e-5,
        )
        with torch.no_grad():
            attn.to_q.weight.add_(torch.randn_like(attn.to_q.weight) * 0.5)
            attn.to_out[0].weight.mul_(0.3)
            attn.to_out[0].bias.add_(0.2)
        torch.testing.assert_close(
            self.identity_residual(isolated, attn, hidden, text, image),
            before,
            atol=2e-6,
            rtol=2e-5,
        )
        for name, value in isolated.state_dict().items():
            torch.testing.assert_close(value, private_before[name], atol=0, rtol=0)

    def test_identity_depends_on_image_tokens_and_keeps_hidden_gradients(self):
        attn, original = self.attention()
        isolated = FrozenIdentityAttnProcessor(attn, original)
        hidden, text, image = self.inputs()
        first = self.identity_residual(isolated, attn, hidden, text, image)
        second = self.identity_residual(isolated, attn, hidden, text, image + 0.5)
        self.assertFalse(torch.allclose(first, second))
        first.square().mean().backward()
        self.assertGreater(float(hidden.grad.norm()), 0)
        self.assertTrue(
            all(parameter.grad is None for parameter in isolated.parameters())
        )

    def test_zero_scale_matches_text_baseline_with_mask_and_spatial_temb(self):
        attn, original = self.attention(
            spatial_norm_dim=16,
            residual_connection=True,
            rescale_output_factor=2.0,
        )
        isolated = FrozenIdentityAttnProcessor(attn, original)
        hidden, text, image = self.inputs(spatial=True)
        temb = torch.randn(2, 16, 2, 3)
        mask = torch.tensor([[[0.0, -10000.0, 0.0, 0.0, 0.0]]]).expand(2, -1, -1)
        original_result = original(attn, hidden, (text, [image]), mask, temb)
        torch.testing.assert_close(
            isolated(attn, hidden, (text, [image]), mask, temb),
            original_result,
            atol=2e-6,
            rtol=2e-5,
        )
        isolated.scale = [0.0]
        torch.testing.assert_close(
            isolated(attn, hidden, (text, [image]), mask, temb),
            AttnProcessor2_0()(attn, hidden, text, mask, temb),
            atol=0,
            rtol=0,
        )

    def test_unsupported_inputs_fail_explicitly(self):
        attn, original = self.attention()
        isolated = FrozenIdentityAttnProcessor(attn, original)
        hidden, text, image = self.inputs()
        with self.assertRaisesRegex(NotImplementedError, "mask"):
            isolated(
                attn, hidden, (text, [image]), ip_adapter_masks=torch.ones(1, 1, 2, 3)
            )
        with self.assertRaisesRegex(ValueError, "single|one"):
            isolated(attn, hidden, (text, [image, image]))
        with self.assertRaisesRegex(ValueError, "single|one"):
            isolated(attn, hidden, (text, [image.expand(-1, 2, -1, -1)]))
        with self.assertRaisesRegex(ValueError, "tuple"):
            isolated(attn, hidden, text)
        isolated.scale = [[0.5, 0.5]]
        with self.assertRaisesRegex(ValueError, "scale"):
            isolated(attn, hidden, (text, [image]))
        multiple = IPAdapterAttnProcessor2_0(32, 12, num_tokens=[4, 4])
        with self.assertRaisesRegex(ValueError, "single|one"):
            FrozenIdentityAttnProcessor(attn, multiple)
        attn.to_out[1].p = 0.1
        with self.assertRaisesRegex(NotImplementedError, "dropout"):
            FrozenIdentityAttnProcessor(attn, original)


if __name__ == "__main__":
    unittest.main()
