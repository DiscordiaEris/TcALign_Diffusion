import unittest

import torch

import torch

from utils.editing_metrics import pose_scores, summarize


class EditingMetricsTest(unittest.TestCase):
    def test_copying_source_cannot_hide_failed_requested_edit(self):
        from utils.editing_metrics import coefficient_edit_scores
        source = torch.zeros(1, 5)
        target = torch.tensor([[0., 0.5, 0., 0., 0.]])
        scores = coefficient_edit_scores(
            source, target, source, [["pose"]],
            {"pose": slice(0, 3), "expression": slice(3, 5)},
        )
        self.assertGreater(scores["generated_pose_mae_deg"], 9)
        self.assertEqual(scores["generated_pose_mae_deg"], scores["source_copy_pose_mae_deg"])
        self.assertEqual(scores["unrequested_expression_drift"], 0)

    def test_left_prompt_requires_left_in_the_image(self):
        scores = pose_scores(torch.tensor([0., -30., 0.]), torch.tensor([0., -25., 0.]),
                             torch.tensor([0., 8., 0.]), "Turn toward the left side of the image.")
        self.assertEqual(scores["pose_target_side_hit"], 1)
        self.assertEqual(scores["pose_direction_hit"], 1)
        self.assertAlmostEqual(scores["yaw_abs_error_deg"], 5)

    def test_no_turn_cannot_pass_direction(self):
        scores = pose_scores(torch.zeros(3), torch.tensor([0., -25., 0.]),
                             torch.zeros(3), "Turn toward the left side of the image.")
        self.assertEqual(scores["pose_target_side_hit"], 0)
        self.assertEqual(scores["pose_direction_hit"], 0)

    def test_angular_error_wraps(self):
        scores = pose_scores(torch.tensor([0., 179., 0.]), torch.tensor([0., -179., 0.]),
                             torch.zeros(3), "")
        self.assertAlmostEqual(scores["yaw_abs_error_deg"], 2)

    def test_missing_faces_reduce_end_to_end_success(self):
        rows = [
            {"generated_face_detected": 1, "generated_pose_target_side_hit": 1},
            {"generated_face_detected": 0, "generated_pose_target_side_hit": 0},
        ]
        self.assertEqual(summarize(rows)["generated_pose_target_side_hit"], {"mean": 0.5, "n": 2})


if __name__ == "__main__":
    unittest.main()
