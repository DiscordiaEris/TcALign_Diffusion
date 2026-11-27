import io
import unittest
from copy import deepcopy

import torch
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    IPAdapterAttnProcessor2_0,
)

from models.faceid_adapter import install_frozen_lora
from models.isolated_identity_attention import FrozenIdentityAttnProcessor


class IdentityGainTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def attention(self):
        attn = Attention(
            query_dim=32,
            cross_attention_dim=12,
            heads=4,
            dim_head=8,
            bias=True,
            out_bias=True,
            dropout=0.0,
            norm_num_groups=4,
            residual_connection=True,
            rescale_output_factor=1.7,
        )
        for layer in (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]):
            install_frozen_lora(
                layer,
                torch.randn(2, layer.in_features) * 0.1,
                torch.randn(layer.out_features, 2) * 0.1,
            )
        attn.requires_grad_(False)
        original = IPAdapterAttnProcessor2_0(
            hidden_size=32, cross_attention_dim=12, num_tokens=[4], scale=[2.0]
        )
        return attn, original

    def inputs(self, spatial=False):
        shape = (2, 32, 2, 3) if spatial else (2, 6, 32)
        return (
            torch.randn(shape, requires_grad=True),
            torch.randn(2, 5, 12),
            torch.randn(2, 1, 4, 12, requires_grad=True),
        )

    def test_default_has_no_gain_and_remains_frozen(self):
        attn, original = self.attention()
        isolated = FrozenIdentityAttnProcessor(attn, original)
        self.assertNotIn("identity_gain_logit", isolated.state_dict())
        self.assertFalse(any(p.requires_grad for p in isolated.parameters()))

    def test_initial_gain_preserves_outputs_and_input_gradients(self):
        for spatial in (False, True):
            with self.subTest(spatial=spatial):
                attn, original = self.attention()
                fixed = FrozenIdentityAttnProcessor(attn, original)
                calibrated = FrozenIdentityAttnProcessor(
                    attn, original, train_identity_gain=True
                )
                hidden, text, image = self.inputs(spatial)
                expected = fixed(attn, hidden, (text, [image]))
                actual = calibrated(attn, hidden, (text, [image]))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                upstream = torch.randn_like(actual)
                expected_grads = torch.autograd.grad(
                    expected, (hidden, image), upstream, retain_graph=True
                )
                actual_grads = torch.autograd.grad(
                    actual, (hidden, image), upstream
                )
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, atol=0, rtol=0
                    )

    def test_only_scalar_gain_receives_parameter_gradients(self):
        attn, original = self.attention()
        calibrated = FrozenIdentityAttnProcessor(
            attn, original, train_identity_gain=True
        )
        trainable = {
            name: p for name, p in calibrated.named_parameters() if p.requires_grad
        }
        self.assertEqual(list(trainable), ["identity_gain_logit"])
        self.assertEqual(trainable["identity_gain_logit"].shape, torch.Size([]))
        hidden, text, image = self.inputs()
        calibrated(attn, hidden, (text, [image])).square().mean().backward()
        self.assertGreater(float(calibrated.identity_gain_logit.grad.abs()), 0)
        self.assertGreater(float(hidden.grad.norm()), 0)
        self.assertGreater(float(image.grad.norm()), 0)
        self.assertTrue(
            all(p.grad is None for name, p in calibrated.named_parameters()
                if name != "identity_gain_logit")
        )

    def test_static_scale_two_stays_between_one_and_three(self):
        attn, original = self.attention()
        calibrated = FrozenIdentityAttnProcessor(
            attn, original, train_identity_gain=True
        )
        hidden, text, image = self.inputs()
        for logit, effective_scale in ((-100.0, 1.0), (0.0, 2.0), (100.0, 3.0)):
            with self.subTest(logit=logit):
                with torch.no_grad():
                    calibrated.identity_gain_logit.fill_(logit)
                original.scale = [effective_scale]
                expected = original(attn, hidden, (text, [image]))
                actual = calibrated(attn, hidden, (text, [image]))
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_state_dict_restores_learned_gain_and_output(self):
        attn, original = self.attention()
        calibrated = FrozenIdentityAttnProcessor(
            attn, original, train_identity_gain=True
        )
        with torch.no_grad():
            calibrated.identity_gain_logit.fill_(0.73)
        hidden, text, image = self.inputs()
        expected = calibrated(attn, hidden, (text, [image]))
        buffer = io.BytesIO()
        torch.save(calibrated.state_dict(), buffer)
        buffer.seek(0)
        restored = FrozenIdentityAttnProcessor(
            attn, deepcopy(original), train_identity_gain=True
        )
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(
            restored(attn, hidden, (text, [image])), expected, atol=0, rtol=0
        )
        self.assertTrue(restored.identity_gain_logit.requires_grad)

    def test_optimizer_updates_gain_without_changing_frozen_projections(self):
        attn, original = self.attention()
        calibrated = FrozenIdentityAttnProcessor(
            attn, original, train_identity_gain=True
        )
        attn.set_processor(calibrated)
        before = {name: value.clone() for name, value in attn.state_dict().items()}
        optimizer = torch.optim.AdamW(
            [p for p in attn.parameters() if p.requires_grad], lr=0.1
        )
        hidden, text, image = self.inputs()
        attn(hidden, encoder_hidden_states=(text, [image])).square().mean().backward()
        optimizer.step()
        for name, value in attn.state_dict().items():
            if name == "processor.identity_gain_logit":
                self.assertFalse(torch.equal(value, before[name]))
            else:
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_zero_static_scale_skips_gain_and_matches_text(self):
        attn, original = self.attention()
        calibrated = FrozenIdentityAttnProcessor(
            attn, original, train_identity_gain=True
        )
        calibrated.scale = [0.0]
        hidden, text, image = self.inputs()
        expected = AttnProcessor2_0()(attn, hidden, text)
        actual = calibrated(attn, hidden, (text, [image]))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().mean().backward()
        self.assertIsNone(calibrated.identity_gain_logit.grad)
        self.assertGreater(float(hidden.grad.norm()), 0)

    def test_gain_keeps_unsupported_input_errors(self):
        attn, original = self.attention()
        calibrated = FrozenIdentityAttnProcessor(
            attn, original, train_identity_gain=True
        )
        hidden, text, image = self.inputs()
        with self.assertRaisesRegex(NotImplementedError, "mask"):
            calibrated(
                attn, hidden, (text, [image]), ip_adapter_masks=torch.ones(1, 1, 2, 3)
            )
        with self.assertRaisesRegex(ValueError, "single|one"):
            calibrated(attn, hidden, (text, [image, image]))
        calibrated.scale = [[2.0]]
        with self.assertRaisesRegex(ValueError, "scale"):
            calibrated(attn, hidden, (text, [image]))


if __name__ == "__main__":
    unittest.main()
