import unittest

import torch

from utils import losses


class RequestedControlLossTest(unittest.TestCase):
    groups = {
        "expression": slice(0, 2),
        "pose": slice(2, 5),
        "lighting": slice(5, 7),
    }

    def control_loss(self, predicted, source, target, attributes):
        return losses.requested_attribute_control_loss(
            predicted, source, target, attributes, self.groups
        )

    def test_copying_source_has_unit_error_for_each_requested_attribute(self):
        source = torch.zeros(3, 7)
        target = torch.tensor([
            [0.3, 0.3, 0., 0., 0., 0., 0.],
            [0., 0., 0., -0.6, 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.04, 0.04],
        ])
        attributes = [["expression"], ["pose"], ["lighting"]]
        for index, requested in enumerate(attributes):
            with self.subTest(attribute=requested):
                value = self.control_loss(source[index:index + 1], source[index:index + 1],
                                          target[index:index + 1], [requested])
                self.assertAlmostEqual(float(value), 1., places=6)

    def test_half_edit_beats_copy_and_reversing_edit_is_worse(self):
        source = torch.zeros(1, 7)
        target = torch.tensor([[0., 0., 0., 0.6, 0., 0., 0.]])
        self.assertAlmostEqual(float(self.control_loss(target / 2, source, target, [["pose"]])),
                               0.25, places=6)
        self.assertAlmostEqual(float(self.control_loss(-target, source, target, [["pose"]])),
                               4., places=6)

    def test_correct_requested_edit_ignores_target_nuisance_changes(self):
        source = torch.zeros(1, 7)
        target = torch.tensor([[2., -2., 0.7, -0.2, 0.3, 0.04, 0.04]])
        predicted = torch.tensor([[0., 0., 0., 0., 0., 0.04, 0.04]], requires_grad=True)
        value = self.control_loss(predicted, source, target, [["lighting"]])
        self.assertEqual(float(value), 0.)
        value.backward()
        self.assertTrue(torch.equal(predicted.grad, torch.zeros_like(predicted)))

    def test_gradients_turn_yaw_both_directions_and_change_requested_lighting(self):
        source = torch.zeros(3, 7)
        target = torch.tensor([
            [2., -2., 0., -0.6, 0., 0.1, 0.1],
            [2., -2., 0., 0.6, 0., 0.1, 0.1],
            [2., -2., 0.5, 0.5, 0.5, -0.04, -0.04],
        ])
        predicted = source.clone().requires_grad_()
        self.control_loss(predicted, source, target, [["pose"], ["pose"], ["lighting"]]).backward()
        self.assertGreater(float(predicted.grad[0, 3]), 0.)
        self.assertLess(float(predicted.grad[1, 3]), 0.)
        self.assertTrue((predicted.grad[2, 5:] > 0).all())
        self.assertTrue(torch.equal(predicted.grad[:, :2], torch.zeros(3, 2)))
        self.assertTrue(torch.equal(predicted.grad[:2, 5:], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(predicted.grad[2, 2:5], torch.zeros(3)))

    def test_tiny_requested_edits_have_finite_gradients(self):
        source = torch.zeros(1, 7)
        target = torch.tensor([[0., 0., 0., 0., 0., 1e-7, 1e-7]])
        predicted = source.clone().requires_grad_()
        value = self.control_loss(predicted, source, target, [["lighting"]])
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertLess(float(predicted.grad.abs().max()), 0.001)

    def test_teachers_are_detached_and_samples_receive_equal_weight(self):
        source = torch.zeros(2, 7, requires_grad=True)
        target = torch.tensor([
            [0.3, 0.3, 0., -0.6, 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.04, 0.04],
        ], requires_grad=True)
        predicted = torch.tensor([
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.04, 0.04],
        ], requires_grad=True)
        value = self.control_loss(predicted, source, target,
                                  [["expression", "pose"], ["lighting"]])
        self.assertAlmostEqual(float(value), 0.5, places=6)
        value.backward()
        self.assertIsNone(source.grad)
        self.assertIsNone(target.grad)


if __name__ == "__main__":
    unittest.main()
