#!/usr/bin/env python3
# encoding: utf-8

"""
Forcasting accuracy measures.
"""

import numpy as np


def mase(actual_values, forcasted_values, seasonal_period=1):
    """
    Calculates the mean absolute scaled error (i.e. a measure for the accuracy
    of univariate forecast predictions). See
    https://en.wikipedia.org/wiki/Mean_absolute_scaled_error  for more details.

    Args:
        actual_values: The true values as list.
        forcasted_values: The predicted values as list.
        seasonal_period: The seasonal period used in the naïve baseline
        forcast method.

    Returns:
       The mean scaled error of the prediction.
    """
    if seasonal_period < 1:
        raise ValueError('seasonal period must be greater than zero.')

    actual_values = np.array(actual_values)
    forcasted_values = np.array(forcasted_values)
    nperiods = len(actual_values)

    forcast_error = np.sum(np.abs(actual_values - forcasted_values))
    mae_naive_forcast = nperiods / (nperiods - seasonal_period) * np.sum(np.abs(
        actual_values[seasonal_period:] - actual_values[:-seasonal_period]))

    return forcast_error / mae_naive_forcast
