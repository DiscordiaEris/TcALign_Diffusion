import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from trainers.base_trainer import BaseTrainer


class IdentityBackendResumeTest(unittest.TestCase):
    def test_resume_cannot_change_the_identity_training_alignment(self):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.config = SimpleNamespace(identity_loss_backend='arcface', identity_training_alignment='paired_target')
        trainer.model=Mock();trainer.optimizer=Mock();trainer.scaler=Mock();trainer.logger=Mock()
        checkpoint={'training_config':{'identity_loss_backend':'arcface','identity_training_alignment':'generated'},
            'epoch':0,'global_step':0,'model_state_dict':{},'optimizer_state_dict':{},'scaler_state_dict':{}}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'old_alignment.pth';torch.save(checkpoint,p)
            with self.assertRaisesRegex(ValueError,'alignment'):
                trainer.load_checkpoint(str(p))
        trainer.model.load_checkpoint_state.assert_not_called()
    def test_resume_cannot_silently_change_the_identity_objective(self):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.config = SimpleNamespace(identity_loss_backend='arcface')
        trainer.model = Mock(); trainer.optimizer = Mock(); trainer.scaler = Mock(); trainer.logger = Mock()
        checkpoint = {'training_config': {'identity_loss_backend':'facenet'},
            'epoch':0, 'global_step':0, 'model_state_dict':{},'optimizer_state_dict':{},'scaler_state_dict':{}}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'legacy.pth';torch.save(checkpoint,p)
            with self.assertRaisesRegex(ValueError, 'identity.*backend'):
                trainer.load_checkpoint(str(p))
        trainer.model.load_checkpoint_state.assert_not_called()
