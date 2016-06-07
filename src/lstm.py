#!/usr/bin/env python3
# encoding: utf-8

import time
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation, Dropout
from utils import read_dataset, split_dataset
from nn_common import plot_result, store_model, load_model


def compile_model(nneurons, loss_fn, dropout=(0.2, 0.2)):
    model = Sequential()
    model.add(LSTM(nneurons[0], input_dim=1, return_sequences=True))
    model.add(Dropout(dropout[0]))
    model.add(LSTM(nneurons[1], return_sequences=False))
    model.add(Dropout(dropout[1]))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='rmsprop', loss=loss_fn)
    return model

def run_network(model=None, save_model=False):
    start_time = time.time()

    print('loading and prepare data set...')
    data = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    window = [1] * 50 # use the last 50 values for forecasting
    X_train, y_train, X_test, y_test, mean, std = split_dataset(data, window, ratio=0.75)

    # reshape s.t. the data has the form (#examples, #values in sequences,
    # dim. of each value in the sequence)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    if model is None:
        print('initialize model...')
        model = compile_model(nneurons=(50, 100), loss_fn='mse')

        print('train model...')
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        model.fit(X_train, y_train, nb_epoch=10, validation_split=0.33,
            callbacks=[early_stopping])

    print('make predictions...')
    prediction = model.predict(X_test).flatten()
    plot_result(prediction, y_test, mean, std)

    if save_model:
        store_model(model)

    print('totoal duration: {:.2f} seconds'.format(time.time() - start_time))

if __name__ == '__main__':
    #m = load_model('Tuesday_155612', 'msle')
    run_network(save_model=True)
