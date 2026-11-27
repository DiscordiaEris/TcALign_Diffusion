import unittest
from configs.stage2_config import Stage2Config


class IdentityBackendConfigTest(unittest.TestCase):
    def test_unknown_identity_training_alignment_is_rejected(self):
        config = Stage2Config()
        config.identity_training_alignment = 'unknown'
        with self.assertRaisesRegex(ValueError, 'alignment'):
            config.__post_init__()
    def test_refinement_requires_identity_calibration_mode(self):
        config = Stage2Config()
        config.identity_refinement_rank = 2
        with self.assertRaisesRegex(ValueError, 'refinement'):
            config.__post_init__()
    def test_legacy_config_keeps_facenet_default(self):
        config = Stage2Config()
        self.assertEqual(getattr(config, 'identity_loss_backend', None), 'facenet')

    def test_unknown_backend_is_rejected(self):
        config = Stage2Config()
        config.identity_loss_backend = 'unknown'
        with self.assertRaisesRegex(ValueError, 'backend'):
            config.__post_init__()

    def test_arcface_requires_its_weights_and_preserves_raw_exposure(self):
        config = Stage2Config()
        config.identity_loss_backend = 'arcface'
        with self.assertRaisesRegex(ValueError, 'ArcFace'):
            config.__post_init__()
        config.identity_arcface_weights_path = 'model.onnx'
        config.identity_exposure_normalization = True
        with self.assertRaisesRegex(ValueError, 'exposure'):
            config.__post_init__()
