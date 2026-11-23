import unittest
from types import SimpleNamespace

from torch.utils.data import Dataset

from data import datasets


class InstructionDataset(Dataset):
    def __init__(self):
        instructions = ["turn left"] * 8 + ["smile"] + ["dim lighting"] * 3
        self.metadata = [(f"source-{index}", f"target-{index}", text)
                         for index, text in enumerate(instructions)]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.metadata[index][2]


class InstructionSamplingTest(unittest.TestCase):
    def setUp(self):
        self.dataset = InstructionDataset()
        self.factory = getattr(datasets, "create_training_sampler", None)
        self.assertIsNotNone(self.factory, "Training must expose optional instruction balancing")

    def test_each_instruction_has_equal_total_probability(self):
        sampler = self.factory(self.dataset, SimpleNamespace(balance_instructions=True, seed=42))
        probabilities = sampler.weights / sampler.weights.sum()
        for instruction in ("turn left", "smile", "dim lighting"):
            selected = [index for index, row in enumerate(self.dataset.metadata)
                        if row[2] == instruction]
            self.assertAlmostEqual(probabilities[selected].sum().item(), 1 / 3)
        self.assertEqual(sampler.num_samples, len(self.dataset))
        self.assertTrue(sampler.replacement)

    def test_old_configs_keep_regular_shuffle(self):
        self.assertIsNone(self.factory(self.dataset, SimpleNamespace(seed=42)))
        self.assertIsNone(self.factory(
            self.dataset, SimpleNamespace(balance_instructions=False, seed=42)))

    def test_seed_reproduces_sampling_without_repeating_every_epoch(self):
        config = SimpleNamespace(balance_instructions=True, seed=42)
        first = self.factory(self.dataset, config)
        second = self.factory(self.dataset, config)
        epoch_one = list(first)
        self.assertEqual(epoch_one, list(second))
        self.assertNotEqual(epoch_one, list(first))


if __name__ == "__main__":
    unittest.main()
