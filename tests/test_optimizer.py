import unittest
from types import SimpleNamespace

import torch
from torch import nn

from trainers.base_trainer import BaseTrainer
from trainers.stage2_trainer import Stage2Trainer


class OptimizerTest(unittest.TestCase):
    def trainer(self, trainer_type, model, **overrides):
        trainer = trainer_type.__new__(trainer_type)
        trainer.model = model
        trainer.config = SimpleNamespace(
            learning_rate=1e-4, weight_decay=1e-4, beta1=0.9, beta2=0.999,
            **overrides,
        )
        return trainer

    def test_small_pretrained_weights_survive_zero_task_gradients(self):
        # Coupled Adam L2 used to erase small pretrained attention weights even
        # when the task provided no update. Decoupled decay preserves the prior.
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.02)
        optimizer = self.trainer(BaseTrainer, model).configure_optimizers()
        for _ in range(400):
            model.weight.grad = torch.zeros_like(model.weight)
            optimizer.step()
        self.assertGreater(float(model.weight.min()), 0.0199)

    def test_bias_and_normalization_have_no_weight_decay(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
        optimizer = self.trainer(BaseTrainer, model).configure_optimizers()
        group_for = {id(p): group for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(group_for[id(model[0].weight)]["weight_decay"], 1e-4)
        for parameter in (model[0].bias, model[1].weight, model[1].bias):
            self.assertEqual(group_for[id(parameter)]["weight_decay"], 0)

    def test_pretrained_unet_uses_smaller_updates_than_new_conditioning(self):
        model = nn.Module()
        model.diffusion_unet = nn.Module()
        model.diffusion_unet.unet = nn.Module()
        model.diffusion_unet.unet.attn2 = nn.Linear(2, 2, bias=False)
        model.diffusion_unet.unet.conv_in = nn.Linear(2, 2, bias=False)
        model.diffusion_unet.geometry_adapter = nn.Linear(2, 2, bias=False)
        model.diffusion_mapper = nn.Linear(2, 2, bias=False)
        model.frozen = nn.Linear(2, 2).requires_grad_(False)
        trainer = self.trainer(Stage2Trainer, model, unet_learning_rate=1e-5)
        optimizer = trainer.configure_optimizers()
        all_parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(all_parameters), 4)
        self.assertEqual(len({id(p) for p in all_parameters}), 4)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for parameter in all_parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        changes = {name: float((parameter - before[name]).abs().mean())
                   for name, parameter in model.named_parameters()}
        attention_change = changes["diffusion_unet.unet.attn2.weight"]
        for name in ("diffusion_unet.unet.conv_in.weight",
                     "diffusion_unet.geometry_adapter.weight", "diffusion_mapper.weight"):
            self.assertAlmostEqual(changes[name] / attention_change, 10, delta=0.05)
        self.assertEqual(changes["frozen.weight"], 0)


if __name__ == "__main__":
    unittest.main()
