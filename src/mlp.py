#!/usr/bin/env python3
# encoding: utf-8

import time
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.callbacks import EarlyStopping
from utils import read_dataset, split_dataset, create_window_array
from nn_common import load_model, store_model, plot_result
from evaluation import mase


def compile_model(hidden_neurons, input_dim, loss_fn, activation_fn='sigmoid'):
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=input_dim))
    model.add(Activation(activation_fn))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='sgd', loss=loss_fn)
    return model


def run_network(window, model=None, save_model=False, show_plot=False):
    start_time = time.time()

    print('loading and prepare data set...')
    data = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    X_train, y_train, X_test, y_test, mean, std = split_dataset(
        data, window, ratio=0.90, standardize=True)

    print('number of training samples ', len(y_train))
    print('number of test samples     ', len(y_test))

    if not model:
        print('initialize model...')
        model = compile_model(
            hidden_neurons=25, loss_fn='mse',
            input_dim=sum(1 for x in window if x), activation_fn='tanh')

        print('model ', model.summary())

        print('train model...')
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        model.fit(X_train, y_train, nb_epoch=500, validation_split=0.1,
                  callbacks=[early_stopping])

    print('make predictions...')
    prediction = model.predict(X_test).flatten()

    if show_plot:
        plot_result(prediction, y_test, mean, std)
        print('mase = ', mase(y_train, y_test, prediction))

    if save_model:
        store_model(model)

    print('totoal duration: {:.2f} seconds'.format(time.time() - start_time))


def hyper_parameter_search(max_evals=100):
    from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL

    data = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    space = {
        'nneurons': hp.randint('nneurons', 41),
        'window': hp.randint('window', 2048),
        'season': hp.choice('season', ['no_season', 'half_day', 'full_day']),
        'activation_function': hp.choice('func', ['sigmoid', 'tanh', 'relu'])
    }

    def objective(params):
        nneurons = params['nneurons']

        if params['season'] == 'full_day':
            window = create_window_array(params['window'], season_lag=288)
        if params['season'] == 'half_day':
            window = create_window_array(params['window'], season_lag=168)
        else:
            window = create_window_array(params['window'])

        if not any(window) or nneurons < 2:
            return {'status': STATUS_FAIL}

        X_train, y_train, *_ = split_dataset(
            data, window, ratio=0.90, standardize=True)
        model = compile_model(
            nneurons, input_dim=sum(1 for x in window if x), loss_fn='mse',
            activation_fn=params['activation_function'])
        hist = model.fit(
            X_train, y_train, nb_epoch=50, validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
            verbose=0)

        return {'loss': hist.history['val_loss'][-1], 'status': STATUS_OK}

    return fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals)


if __name__ == '__main__':
    print(hyper_parameter_search(1000))
