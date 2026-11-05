import unittest

from src.attacks import SPOOF_PROFILES, inject_balanced
from src.data import synthetic


class BalancedAttackTests(unittest.TestCase):
    def test_includes_balanced_classes_and_varied_spoof_profiles(self):
        attacked = inject_balanced(synthetic(400), seed=2)

        self.assertEqual(set(attacked["attack"].unique()), {0, 1, 2})
        self.assertGreaterEqual(attacked.loc[attacked["attack"] == 1, "scenario"].nunique(), 3)
        spoof_events = (attacked["attack"].eq(1) & attacked["attack"].shift(fill_value=0).ne(1)).sum()
        jam_events = (attacked["attack"].eq(2) & attacked["attack"].shift(fill_value=0).ne(2)).sum()
        self.assertEqual(spoof_events, 3)
        self.assertEqual(jam_events, 3)

    def test_small_split_has_each_attack_class_and_keeps_source_unchanged(self):
        source = synthetic(22)
        attacked = inject_balanced(source, seed=7)

        self.assertEqual(set(attacked["attack"].unique()), {0, 1, 2})
        self.assertNotIn("attack", source.columns)
        self.assertNotIn("label", source.columns)
        self.assertNotIn("scenario", source.columns)

    def test_multiple_seeds_cover_all_spoof_profiles(self):
        observed = set()
        for seed in range(len(SPOOF_PROFILES)):
            attacked = inject_balanced(synthetic(400), seed=seed)
            observed.update(attacked.loc[attacked["attack"] == 1, "scenario"].unique())

        self.assertEqual(observed, set(SPOOF_PROFILES))

    def test_spoof_strengths_are_applied_and_recorded(self):
        strengths = (0.1, 0.25, 0.5, 1.0)
        observed = set()
        for seed in range(len(strengths)):
            attacked = inject_balanced(synthetic(600), seed=seed, spoof_strengths=strengths)
            observed.update(attacked.loc[attacked["attack"] == 1, "spoof_strength"].unique())

        self.assertEqual(observed, set(strengths))
        self.assertTrue((attacked.loc[attacked["attack"] != 1, "spoof_strength"] == 0).all())

    def test_spoof_strengths_must_be_finite_and_positive(self):
        for strengths in ((), (0.0,), (-0.1,), (float("inf"),)):
            with self.subTest(strengths=strengths), self.assertRaises(ValueError):
                inject_balanced(synthetic(100), spoof_strengths=strengths)


if __name__ == "__main__":
    unittest.main()
