#!/usr/bin/env python3
# encoding: utf-8

"""
Forecasting accuracy measures.
"""

import numpy as np


def naive_forecast(past, t=-1, horizon=1, seasonal_period=1):
    """
    Simple forecast method which predicts the next value by taking the current
    value (i.e. value at the end of the past array) using the default parameter.
    By tuning the default parameter it is possible to use the seasonal version
    of this algorithm which calculates: pred_t+h,t = y_t+h-K where pred is the
    list of past values, t is the current time index, h is the prediction
    horizon and K is the seasonal period.
    """
    return past[t - seasonal_period + horizon]

def mase(training_set, test_set, prediction, seasonal_period=1):
    """
    Calculates the mean absolute scaled error (i.e. a measure for the accuracy
    of univariate forecast predictions). See https://www.otexts.org/fpp/2/5
    for more details.

    Args:
        training_set: Data to train the model is here used calculate the
        scaling factor for MASE as list.
        test_set: The true values to compare to the prediction as list.
        prediction: The forecasted values as list.
        seasonal_period: The seasonal period used in the naïve baseline
        forecast method.

    Returns:
       The mean scaled error of the prediction.
    """
    if seasonal_period < 1:
        raise ValueError('Seasonal period must be greater than zero.')

    if len(test_set) != len(prediction):
        raise ValueError('Prediction an test set must have the same length.')

    training_set = np.array(training_set)
    test_set = np.array(test_set)
    prediction = np.array(prediction)

    mae_naive_forecast = np.sum(np.abs(training_set[seasonal_period:] -
      training_set[:-seasonal_period])) / (len(training_set) - seasonal_period)

    forecast_errors = np.abs(test_set - prediction) / mae_naive_forecast

    return forecast_errors.mean()
