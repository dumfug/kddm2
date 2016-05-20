#!/usr/bin/env python3
# encoding: utf-8

import time

import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.utils.visualize_util import plot

from utils import read_dataset, split_dataset


def init_model(hidden_neurons, input_dim):
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='sgd', loss='mse')
    return model

def run_network():
    start_time = time.time()

    print('loading and prepare data set...')
    data = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    window = [1] * 20 # use the last 20 values for forecasting
    X_train, y_train, X_test, y_test, mean, std = split_dataset(data, window)

    print('initialize model...')
    model = init_model(hidden_neurons=25, input_dim=sum(1 for x in window if x))

    print('train model...')
    model.fit(X_train, y_train, nb_epoch=20)

    print('\nduration: {0} seconds'.format(time.time() - start_time))

if __name__ == '__main__':
    run_network()
