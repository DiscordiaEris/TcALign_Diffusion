import unittest
from types import SimpleNamespace

import torch
from torch import nn

from models.tcalign_diffusion import TcAlignDiffusion
from configs.stage2_config import Stage2Config


class IdentityCalibrationStateTest(unittest.TestCase):
    def test_refinement_mode_freezes_global_gains_and_retains_all_checkpoint_weights(self):
        model = self.model()
        model.config.identity_refinement_rank = 2
        model.processor.identity_refinement = nn.Sequential(nn.Linear(3, 2, bias=False), nn.Linear(2, 3, bias=False))
        model.configure_identity_gain_training()
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(names, ['processor.identity_refinement.0.weight', 'processor.identity_refinement.1.weight'])
        self.assertIn('processor.identity_gain_logit', model.checkpoint_state())
        self.assertIn('backbone.weight', model.checkpoint_state())
    def test_calibration_requires_an_identity_objective(self):
        with self.assertRaisesRegex(ValueError, "identity loss"):
            Stage2Config(identity_gain_only=True, isolate_faceid_attention=True,
                faceid_adapter_path="adapter", faceid_encoder_path="encoder", lambda_id=0.)

    def model(self):
        model = TcAlignDiffusion.__new__(TcAlignDiffusion)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(identity_gain_only=True)
        model.backbone = nn.Linear(3, 3)
        model.processor = nn.Module()
        model.processor.identity_gain_logit = nn.Parameter(torch.zeros(()))
        return model

    def test_only_gain_trains_but_finetuned_backbone_round_trips(self):
        model = self.model()
        with torch.no_grad(): model.backbone.weight.fill_(.37)
        model.configure_identity_gain_training()
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad],
                         ["processor.identity_gain_logit"])
        saved = {k: v.clone() for k, v in model.checkpoint_state().items()}
        self.assertIn("backbone.weight", saved)
        other = self.model()
        other.configure_identity_gain_training()
        other.load_checkpoint_state(saved)
        torch.testing.assert_close(other.backbone.weight, model.backbone.weight)
        torch.testing.assert_close(other.processor.identity_gain_logit, model.processor.identity_gain_logit)

    def test_repeated_configuration_keeps_backbone_checkpoint_keys(self):
        model = self.model()
        model.configure_identity_gain_training()
        model.configure_identity_gain_training()
        self.assertIn("backbone.weight", model.checkpoint_state())

    def test_training_keeps_frozen_batch_norm_and_dropout_in_inference_mode(self):
        model = self.model()
        model.clip_encoder = nn.Identity()
        model.text_3d_editor = nn.Linear(3, 3)
        model.backbone = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(.5))
        model.configure_identity_gain_training()
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(all(not module.training for module in model.children()))
        self.assertFalse(model.backbone[0].training)


if __name__ == "__main__": unittest.main()
