"""Pure calculation checks; synthetic values are not research outcomes."""

import unittest
import numpy as np
import analyze_schema_results as analysis


class StatisticsTests(unittest.TestCase):
    def test_no_disagreement_has_p_one(self):
        self.assertEqual(analysis.mcnemar_exact(0, 0), 1)

    def test_exact_mcnemar_known_tail(self):
        self.assertEqual(analysis.mcnemar_exact(0, 5), 0.0625)
        self.assertEqual(analysis.mcnemar_exact(5, 0), 0.0625)
        self.assertEqual(analysis.mcnemar_exact(4, 4), 1)

    def test_holm_is_monotone_in_sorted_order(self):
        values = analysis.holm([0.04388, 0.00015, 0.06094])
        for actual, expected in zip(values, [0.08776, 0.00045, 0.08776]):
            self.assertAlmostEqual(actual, expected)

    def test_identical_condition_has_zero_paired_difference(self):
        samples = np.random.default_rng(1).integers(0, 2, size=(1000, 2))
        intervals = analysis.cluster_intervals([[1, 1], [0, 0]], [2, 3], samples)
        self.assertEqual(intervals["difference_ci"], [0, 0])
        self.assertEqual(intervals["object_ci"], intervals["schema_ci"])

    def test_unequal_cluster_sizes_are_case_weighted(self):
        samples = np.array([[0, 1], [0, 1]])
        intervals = analysis.cluster_intervals([[0, 1], [0, 9]], [1, 9], samples)
        self.assertEqual(intervals["difference_ci"], [1, 1])
        self.assertEqual(intervals["schema_ci"], [1, 1])

    def test_primary_family_is_three_models(self):
        self.assertEqual(len(analysis.MODELS), 3)
        self.assertEqual(analysis.REPETITIONS, 20000)
        self.assertIn("procedural_success", analysis.METRICS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
