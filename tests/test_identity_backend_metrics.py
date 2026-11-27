import unittest
import torch
from torch import nn
from test_identity_validation import PairedDataset, SampledModel, IndependentIdentity
from utils.identity_validation import evaluate_sampled_identity


class BackendMetricsTest(unittest.TestCase):
    def test_arcface_is_named_correctly_and_never_exposure_normalized(self):
        class ArcFace(nn.Module):
            metric_name = 'arcface'
            supports_exposure_normalization = False
            def embed(self, image, alignment, *, normalize_exposure=False):
                if normalize_exposure: raise AssertionError('raw ArcFace only')
                return torch.tensor([[1., 0.]])
        model = SampledModel()
        model.identity_loss = ArcFace()
        result = evaluate_sampled_identity(model, PairedDataset(), 'cpu', evaluator=IndependentIdentity())
        m = result['aggregate']
        self.assertEqual(m.get('arcface_raw_valid'), 18)
        self.assertEqual(m.get('arcface_raw_cosine'), 1.)
        self.assertNotIn('facenet_raw_cosine', m)
        self.assertNotIn('arcface_normalized_cosine', m)

    def test_missing_arcface_detection_is_recorded_and_blocks_promotion(self):
        from models.arcface_identity_loss import FaceDetectionError
        from utils.identity_validation import should_replace_identity_checkpoint
        class ArcFace(nn.Module):
            metric_name = 'arcface'
            supports_exposure_normalization = False
            def __init__(self):
                super().__init__(); self.calls = 0
            def embed(self, image, alignment, *, normalize_exposure=False):
                self.calls += 1
                if self.calls == 2: raise FaceDetectionError('missing generated face')
                return torch.tensor([[1., 0.]])
        model=SampledModel();model.identity_loss=ArcFace()
        result=evaluate_sampled_identity(model, PairedDataset(), 'cpu', evaluator=IndependentIdentity())
        self.assertEqual(result['aggregate']['arcface_raw_valid'], 17)
        self.assertIsNone(result['rows'][0]['arcface_raw_cosine'])
        self.assertFalse(should_replace_identity_checkpoint(result, None))

    def test_arcface_baseline_is_protected_alongside_independent_identity(self):
        from utils.identity_validation import should_replace_identity_checkpoint
        result=evaluate_sampled_identity(SampledModel(),PairedDataset(),'cpu',evaluator=IndependentIdentity())
        metrics=dict(result['aggregate'],arcface_raw_valid=18,arcface_raw_cosine=.7)
        self.assertFalse(should_replace_identity_checkpoint(metrics,.7,min_arcface_score=.75))
        self.assertTrue(should_replace_identity_checkpoint(metrics,.7,min_arcface_score=.65))
