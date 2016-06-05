#!/usr/bin/env python3
# encoding: utf-8

import time
from matplotlib import pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Dense
from utils import read_dataset, split_dataset


def plot_result(forecast, actual, mean, std, export_path=None):
    plt.ylabel('data [GB]')
    plt.xlabel('sample')
    plt.plot((std * actual + mean), 'g', label='actual')
    plt.plot((std * forecast + mean), 'r--', label='forecast')
    plt.legend()

    if export_path is None:
        plt.show()
    else:
        plt.savefig(export_path)

def store_model(model):
    model_id = time.strftime('%A_%H%M%S')
    with open('../models/model_{}.json'.format(model_id), 'x') as f:
        f.write(model.to_json())
    model.save_weights('../models/model_{}.h5'.format(model_id))

def load_model(model_id):
    with open('../models/model_{}.json'.format(model_id)) as f:
        model = model_from_json(f.read())
    model.load_weights('../models/model_{}.h5'.format(model_id))
    model.compile(optimizer='sgd', loss='mae') #loss=mean_squared_logarithmic_error
    return model

def compile_model(hidden_neurons, input_dim):
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=input_dim))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='sgd', loss='mae') #loss=mean_squared_logarithmic_error
    return model

def run_network(show_plot=False):
    start_time = time.time()

    print('loading and prepare data set...')
    data = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    window = [1] * 20 # use the last 20 values for forecasting
    X_train, y_train, X_test, y_test, mean, std = split_dataset(data, window)

    print('number of training samples ', len(y_train))
    print('number of test samples     ', len(y_test))

    print('initialize model...')
    model = compile_model(hidden_neurons=25, input_dim=sum(1 for x in window if x))

    print('train model...')
    model.fit(X_train, y_train, nb_epoch=10)

    print('make predictions...')
    prediction = model.predict(X_test)
    prediction = prediction.reshape(len(y_test))

    if show_plot:
        plot_result(prediction, y_test, mean, std)

    print('\ntotoal duration: {:.2f} seconds'.format(time.time() - start_time))
    return model

if __name__ == '__main__':
    run_network(show_plot=True)
