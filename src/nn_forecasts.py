#!/usr/bin/env python3
# encoding: utf-8

from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from evaluation import mase
from utils import read_dataset, create_window_array
from nn_common import load_model, iterative_prediction


def forecasting_error_experiment():
    print('load data set...')
    data_df = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    mean = data_df.mean()
    std = data_df.std()
    data_df -= mean
    data_df /= std
    data = data_df['data (in bytes)']

    start_forecast_idx = int(len(data) * 0.90)
    training_data = data[:start_forecast_idx]
    test_data = data[start_forecast_idx:]

    print('calculate forecasts using the naive method...')
    naive_forecast = [data[t-1] for t in range(start_forecast_idx, len(data))]
    naive_forecast_errors = []
    for steps in trange(24):
        forecast = naive_forecast[:len(naive_forecast)-steps]
        error = mase(training_data, test_data[steps:], forecast)
        naive_forecast_errors.append(error)

    print('calculate forecasts using a MLP neural network...')
    mlp_model = load_model('Saturday_192115', 'mse')
    window = create_window_array(139, 288)
    window_size = sum(1 for x in window if x)

    mlp_forecast_errors = []
    for steps in trange(24):
        mlp_forecast = []
        for t in range(start_forecast_idx, len(data) - steps):
            mlp_forecast.append(iterative_prediction(
                mlp_model, data[:t], (1, window_size), window, steps+1))
        mlp_forecast_errors.append(
            mase(training_data, test_data[steps:], mlp_forecast))

    print('calculate forecasts using a LSTM neural network...')
    lstm_model = load_model('Saturday_181721', 'mse')
    window = [True] * 19

    lstm_forecast_errors = []
    for steps in trange(24):
        lstm_forecast = []
        for t in range(start_forecast_idx, len(data) - steps):
            lstm_forecast.append(iterative_prediction(
                lstm_model, data[:t], (1, len(window), 1), window, steps+1))
        lstm_forecast_errors.append(
            mase(training_data, test_data[steps:], lstm_forecast))

    print('calculate forecasts using a deep LSTM neural network...')
    lstm_model = load_model('Saturday_171936', 'mse')
    window = [True] * 14

    deep_lstm_forecast_errors = []
    for steps in trange(24):
        lstm_forecast = []
        for t in range(start_forecast_idx, len(data) - steps):
            lstm_forecast.append(iterative_prediction(
                lstm_model, data[:t], (1, len(window), 1), window, steps+1))
        deep_lstm_forecast_errors.append(
            mase(training_data, test_data[steps:], lstm_forecast))

    plt.ylabel('Error')
    plt.xlabel('Steps')
    plt.plot(naive_forecast_errors, label='naïve')
    plt.plot(mlp_forecast_errors, label='MLP')
    plt.plot(lstm_forecast_errors, label='1 layer LSTM')
    plt.plot(deep_lstm_forecast_errors, label='2 layer LSTM')
    plt.legend(loc='upper left')
    plt.show()


def forecasting_different_horizons():
    print('load data set...')
    data_df = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    mean = data_df.mean()
    std = data_df.std()
    data_df -= mean
    data_df /= std
    data = data_df['data (in bytes)']

    start_forecast_idx = int(len(data) * 0.90)
    test_set = data[start_forecast_idx:]

    print('calculate forecasts using a MLP neural network...')
    mlp_model = load_model('Saturday_192115', 'mse')
    window = create_window_array(139, 288)
    window_size = sum(1 for x in window if x)

    plt.ylabel('data (normalized)')
    plt.xlabel('time')
    plt.plot(test_set, 'r-', label='test set')

    for steps, style in [(1, 'g-'), (24, 'b--')]:
        forecast = []
        for t in range(start_forecast_idx, len(data) - steps):
            forecast.append(iterative_prediction(
                mlp_model, data[:t], (1, window_size), window, steps+1))
        series = pd.Series([np.nan] * steps + forecast, index=test_set.index)
        plt.plot(series, style, label='h={}'.format(steps))

    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    forecasting_error_experiment()
    forecasting_different_horizons()
