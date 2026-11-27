import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.inference import load_generation_checkpoint


class LoadedModel:
    def __init__(self, config):
        self.config = config

    def _init_diffusion_modules(self):
        pass

    def load_checkpoint_state(self, state):
        pass

    def to(self, device):
        return self

    def eval(self):
        return self


class InferenceSettingsTest(unittest.TestCase):
    def load(self, saved, defaults=True):
        config = SimpleNamespace(guidance_scale=7.5, inference_strength=1.0,
                                 num_inference_steps=50, use_bf16=False, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"model_state_dict": {}, "global_step": 10, **saved}, path)
            with patch("scripts.inference.TcAlignDiffusion", LoadedModel):
                load_generation_checkpoint(config, path, use_default_guidance=defaults)
        return config

    def test_inference_defaults_round_trip_without_geometry(self):
        config = self.load({"inference_config": {
            "guidance_scale": 1.5, "inference_strength": 0.65,
            "num_inference_steps": 35, "use_bf16": True,
        }})
        self.assertEqual(config.inference_strength, 0.65)
        self.assertEqual(config.guidance_scale, 1.5)
        self.assertEqual(config.num_inference_steps, 35)
        self.assertTrue(config.use_bf16)

    def test_explicit_config_takes_precedence(self):
        config = self.load({"inference_config": {"inference_strength": 0.65}}, defaults=False)
        self.assertEqual(config.inference_strength, 1.0)

    def test_legacy_checkpoint_keeps_legacy_defaults(self):
        config = self.load({})
        self.assertEqual(config.inference_strength, 1.0)

    def test_registered_target_evaluation_uses_saved_coordinate_frame(self):
        config = self.load({"training_config": {"align_paired_targets": True}})
        self.assertTrue(getattr(config, "align_paired_targets", False))

    def test_identity_and_exposure_architecture_is_restored_with_explicit_config(self):
        config = self.load({"inference_config": {
            "faceid_adapter_path": "faceid.bin", "faceid_encoder_path": "buffalo.onnx",
            "faceid_scale": 1.0, "use_photometric_head": True,
        }}, defaults=False)
        self.assertEqual(config.faceid_adapter_path, "faceid.bin")
        self.assertEqual(config.faceid_encoder_path, "buffalo.onnx")
        self.assertEqual(config.faceid_scale, 1.0)
        self.assertTrue(config.use_photometric_head)

    def test_isolated_identity_projection_architecture_round_trips(self):
        config = self.load({"inference_config": {"isolate_faceid_attention": True}}, defaults=False)
        self.assertTrue(getattr(config, "isolate_faceid_attention", False))

    def test_identity_refinement_architecture_restores_before_model_construction(self):
        config = self.load({"inference_config": {
            "identity_gain_only": True, "identity_refinement_rank": 2,
        }}, defaults=False)
        self.assertTrue(config.identity_gain_only)
        self.assertEqual(config.identity_refinement_rank, 2)


if __name__ == "__main__":
    unittest.main()
