# encoding: utf-8

"""
Tests for the forecast accuracy evaluation methods.
"""

import unittest
from src import evaluation


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.test_set = [0, 0, 0, 3, 1, 0, 0, 1, 0, 1, 0, 0]
        self.training_set = [0, 2, 0, 1, 0, 11, 0, 0, 0, 0, 2, 0, 6, 3, 0, 0, 0,
                             0, 0, 7, 0, 0, 0, 0]

    def test_naive_prediction(self):
        prediction = [0] * len(self.test_set)
        self.assertAlmostEqual(evaluation.mase(self.training_set, self.test_set,
                                               prediction), 0.1982758621)

    def test_total_avg_prediction(self):
        prediction = [1.333333333] * len(self.test_set)
        self.assertAlmostEqual(evaluation.mase(self.training_set, self.test_set,
                                               prediction), 0.44061302681992)

    def test_mase_holt(self):
        prediction = [0.8611, 0.7673, 0.7165, 0.6658, 0.6150, 0.5642, 0.5135,
                      0.4627, 0.4120, 0.3612, 0.3105, 0.2597]
        self.assertAlmostEqual(evaluation.mase(self.training_set, self.test_set,
                                               prediction), 0.27428490)

        print()

    def test_mase_seasonality(self):
        prediction = [0.8611, 0.7673, 0.7165, 0.6658, 0.6150, 0.5642, 0.5135,
                      0.4627, 0.4120, 0.3612, 0.3105, 0.2597]
        self.assertLess(
            evaluation.mase(self.training_set, self.test_set, prediction),
            evaluation.mase(self.training_set, self.test_set, prediction,
                            seasonal_period=2),
            'useage of a period on non-seasonal data leads to a larger error'
        )

    def test_mase_perfect_prediction(self):
        prediction = self.test_set
        self.assertAlmostEqual(evaluation.mase(self.training_set, self.test_set,
                                               prediction), 0.0)

    def test_mase_symmetry(self):
        over_estimate = [1, 1, 1, 4, 2, 1, 1, 2, 1, 2, 1, 1]
        under_estimate = [-1, -1, -1, 2, 0, -1, -1, 0, -1, 0, -1, -1]
        self.assertEqual(
            evaluation.mase(self.training_set, self.test_set, over_estimate),
            evaluation.mase(self.training_set, self.test_set, under_estimate)
        )

    def test_mase_params(self):
        args = [self.training_set, self.test_set, self.test_set, -1]
        self.assertRaises(ValueError, evaluation.mase, *args)

        args = [self.training_set, [1, 2, 3, 4], [1, 2, 3]]
        self.assertRaises(ValueError, evaluation.mase, *args)
