# encoding: utf-8

"""
Tests for the forcast accuracy evaluation methods.
"""

import unittest
from src import evaluation


class TestEvaluation(unittest.TestCase):
    def test_mase_perfect_forcast(self):
        actual = list(range(100))
        forcast = actual
        self.assertEqual(evaluation.mase(actual, forcast), 0.0)

    def test_mase_bad_forecast(self):
        length = 100
        actual = list(range(length))
        bad_forcast = [10000] * length
        better_forecast = list(range(5, length + 5))

        self.assertGreater(evaluation.mase(actual, bad_forcast),
                           evaluation.mase(actual, better_forecast))
        self.assertGreater(evaluation.mase(actual, bad_forcast), 1.0)

    def test_mase_symmetry(self):
        length = 100
        actual = list(range(length))
        over_estimate = list(range(10, length + 10))
        under_estimate = list(range(-10, length - 10))

        self.assertEqual(evaluation.mase(actual, over_estimate),
                         evaluation.mase(actual, under_estimate))

    def test_mase_error_value(self):
        actual = [0.5, 1.0, 1.2, 0.8, 1.0, 1.1, 1.5, 2.1, 2.0]
        forcast = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0]

        self.assertLess(evaluation.mase(actual, forcast), 1.0)

    def test_mase_seasonality(self):
        actual = [1, 2, 3, 4, 5, 1, 2.2, 3, 4.2, 5] * 10
        forecast = [0.5, 1.5, 3, 3.5, 5] * 20

        self.assertGreater(evaluation.mase(actual, forecast, seasonal_period=5),
                           evaluation.mase(actual, forecast))
