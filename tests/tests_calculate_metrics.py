import unittest

import pandas as pd

from debate_gpt.results_analysis.calculate_metrics import (
    calculate_cohens_kappa,
    calculate_fleiss_kappa,
)


class TestCalculateMetrics(unittest.TestCase):
    def test_calculate_fleiss_kappa(self):
        test_df = pd.DataFrame(
            {
                "rater_1": [1, 2, 2, 1, 2, 2, 1, 1, 3, 1, 2, 2],
                "rater_2": [1, 2, 1, 2, 1, 2, 3, 2, 3, 2, 3, 1],
                "rater_3": [1, 2, 2, 1, 3, 3, 3, 2, 1, 2, 3, 1],
            }
        )
        fleiss_kappa, lb, ub = calculate_fleiss_kappa(test_df)

        self.assertAlmostEqual(fleiss_kappa, 0.1, 2)
        self.assertAlmostEqual(lb, -0.15, 2)
        self.assertAlmostEqual(ub, 0.35, 2)

    def test_calculate_cohens_kappa(self):
        test_df = pd.DataFrame(
            {
                "rater_1": ["normal"] * 64 + ["abnormal"] * 172,
                "rater_2": ["normal"] * 48
                + ["abnormal"] * 16
                + ["normal"] * 12
                + ["abnormal"] * 160,
            }
        )
        cohens_kappa, lb, ub = calculate_cohens_kappa(test_df)
        self.assertAlmostEqual(cohens_kappa, 0.69, 2)
        self.assertAlmostEqual(ub, 0.80, 2)
        self.assertAlmostEqual(lb, 0.59, 2)
