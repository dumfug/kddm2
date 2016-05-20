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
    for the transported data are converted from bits into GB.

    Args:
        path: The path to the data set.

    Returns:
       The data set as pandas DataFrame.
    """

    data = pd.read_csv(path, parse_dates='Time', index_col='Time')
    data = data / 8 / 2**30 # convert bits into GB
    data.rename(columns={'Internet traffic data (in bits)':
                         'Internet traffic data (in GB)'}, inplace=True)
    return data

def split_dataset(data, window, ratio=0.7):
    """
    Generates training and test data from the data set. Each sample consists of
    a sequence of data points and a target forecast value. The values in the
    data set are standardized to get data with zero mean and a standard
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

    Returns:
        A 6-tuple consisting of:
        1) A numpy array of training sequences.
        2) A numpy array of the corresponding target values for the training
           sequences.
        3) A numpy array of test sequences.
        4) A numpy array of the corresponding target values for the test
           sequences.
        5) The mean of the original data.
        6) The standard deviation of the original data.
    """
    if not 0.0 <= ratio <= 1:
        raise ValueError("invalid value for split ratio.")

    window.append(True) # include the target value in the sequences

    # standardizing the data to get a zero mean and standard deviation of one
    mean = data['Internet traffic data (in GB)'].mean()
    std = data['Internet traffic data (in GB)'].std()
    data['Internet traffic data (in GB)'] -= mean
    data['Internet traffic data (in GB)'] /= std

    sequences = []
    for index in range(len(data) - len(window) + 1):
        sequence = data[index : index + len(window)]['Internet traffic data (in GB)'].tolist()
        sequences.append(list(compress(sequence, window)))
    sequences = np.array(sequences)

    split_row = int(len(sequences) * ratio)
    train = sequences[:split_row:]
    test = sequences[split_row::]

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, mean, std

def _print_test_data():
    data = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    X_test, y_test, _, _, mean, std = split_dataset(data, [True] * 50)
    print("test data: ", X_test)
    print("target values: ", y_test)
    print("mean and std of the original data set: ", mean, "and ", std)

if __name__ == '__main__':
    _print_test_data()
