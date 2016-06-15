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
from evaluation import mase


def compile_model(nneurons, loss_fn, dropout=(0.0, 0.0)):
    model = Sequential()
    model.add(LSTM(nneurons[0], input_dim=1, return_sequences=True))
    if dropout[0] > 0:
        model.add(Dropout(dropout[0]))
    model.add(LSTM(nneurons[1], return_sequences=False))
    if dropout[1] > 0:
        model.add(Dropout(dropout[1]))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='rmsprop', loss=loss_fn)
    return model


def run_network(window, model=None, show_plot=False, save_model=False):
    start_time = time.time()

    print('loading and prepare data set...')
    data = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    X_train, y_train, X_test, y_test, mean, std = split_dataset(
        data, window, ratio=0.90, standardize=True)

    # reshape s.t. the data has the form (#examples, #values in sequences,
    # dim. of each value in the sequence)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    if model is None:
        print('initialize model...')
        model = compile_model(
            nneurons=(19, 5), dropout=(0.0, 0.0), loss_fn='mse')
        print('model ', model.summary())
        print('train model...')
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        model.fit(X_train, y_train, nb_epoch=100, validation_split=0.10,
                  callbacks=[early_stopping])

    print('make predictions...')
    prediction = model.predict(X_test).flatten()

    if show_plot:
        plot_result(prediction, y_test, mean, std)
        print('mase = ', mase(y_train, y_test, prediction))

    if save_model:
        store_model(model)

    print('totoal duration: {:.2f} seconds'.format(time.time() - start_time))


def hyper_parameter_search(max_evals=200):
    from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL

    data = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    space = {
        'nneurons1': hp.randint('nneurons1', 15),
        'nneurons2': hp.randint('nneurons2', 15),
        'window': hp.randint('window', 15)
    }

    def objective(params):
        nneurons1 = params['nneurons1']
        nneurons2 = params['nneurons2']
        window = params['window']

        if nneurons1 < 1 or nneurons2 < 1 or window < 1:
            return {'status': STATUS_FAIL}

        X_train, y_train, *_ = split_dataset(
            data, [1] * window, ratio=0.90, standardize=True)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        model = compile_model(
            (nneurons1, nneurons2), loss_fn='mse', dropout=(0.0, 0.0))
        hist = model.fit(
            X_train, y_train, nb_epoch=100, validation_split=0.10,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
            verbose=0)

        return {'loss': hist.history['val_loss'][-1], 'status': STATUS_OK}

    return fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals)


if __name__ == '__main__':
    print('run hyper param search')
    print(hyper_parameter_search(100))
