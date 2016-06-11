#!/usr/bin/env python3
# encoding: utf-8

"""
Utility functions to handle the network traffic data set.
"""


import pandas as pd
import numpy as np
from itertools import compress


def read_dataset(path):
    """
    Imports a network traffic data set from the datasets directory. The values
    for the transported data are converted from bits into bytes.

    Args:
        path: The path to the data set.

    Returns:
       The data set as pandas DataFrame.
    """

    data = pd.read_csv(path, parse_dates='Time', index_col='Time')
    data = data / 8  # convert bits into bytes
    data.rename(columns={'Internet traffic data (in bits)': 'data (in bytes)'},
                inplace=True)
    return data


def split_dataset(data, window, ratio=0.7, standardize=True):
    """
    Generates training and test data from the data set. Each sample consists of
    a sequence of data points and a target forecast value. The values in the
    data set can be standardized to get data with zero mean and a standard
    deviation of one.

    Args:
        data: The data set as pandas DataFrame.
        window: A list which defines what subset of prior values should be used
        to generate the forecast by the truth values of its elements. For
        example, a list `[1, 0, 1]` indicates that a sequence of the current
        and the second last value should determine the forecast value and the
        last value should not influence the forecast.
        ratio: The ratio (i.e. a value between 0.0 and 1.0) of the data set
        which should be used for training.
        standardize: True leads to a zero mean and standard deviation of one
        data set.

    Returns:
        A 6-tuple consisting of:
        1) A numpy array of training sequences.
        2) A numpy array of the corresponding target values for the training
           sequences.
        3) A numpy array of test sequences.
        4) A numpy array of the corresponding target values for the test
           sequences.
        5) The mean of the original data or 0 if standardize is False.
        6) The standard deviation of the original data or 1 if standardize is
           False.
    """
    if not 0.0 <= ratio <= 1:
        raise ValueError("invalid value for split ratio.")

    ext_window = window + [True]  # include the target value in the sequences

    if standardize:
        # standardizing the data to get a 0 mean and standard deviation of 1
        mean = data['data (in bytes)'].mean()
        std = data['data (in bytes)'].std()
        data['data (in bytes)'] -= mean
        data['data (in bytes)'] /= std
    else:
        mean = 0
        std = 1

    sequences = []
    for idx in range(len(data) - len(ext_window) + 1):
        sequence = data[idx:idx+len(ext_window)]['data (in bytes)'].tolist()
        sequences.append(list(compress(sequence, ext_window)))
    sequences = np.array(sequences)

    split_row = int(len(sequences) * ratio)
    train = sequences[:split_row:]
    test = sequences[split_row::]

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, mean, std


def create_window_array(window, season_lag=None):
    """
    Create a sliding window array (i.e. a array of True and False values that
    is used to determine what past values should be used to to forecast the
    next value).

    Args:
        window: An integer that is converted into its binary representation
        which corresponds to the sliding window. For example, a input of 9
        would correspond to a window of [1, 0, 0, 1].
        season_lag: If value is present three values surrounding the lag are
        included into the sliding window. For example, a a window of 1 (i.e.
        only the last value) and a season lag of 5 leads to a window
        [1, 0, 0, 1, 1, 1].

    Returns:
       The sliding window array.
    """
    window = [int(digit) for digit in bin(window)[2:]]
    if season_lag:
        window += [0] * (season_lag - len(window) - 2) + [1, 1, 1]
    return window
