# encoding: utf-8

import time
import numpy as np
from itertools import compress
from matplotlib import pyplot as plt
from keras.models import model_from_json


def store_model(model):
    model_id = time.strftime('%A_%H%M%S')
    with open('../models/model_{}.json'.format(model_id), 'x') as json_file:
        json_file.write(model.to_json())
    model.save_weights('../models/model_{}.h5'.format(model_id))

def load_model(model_id, loss_function):
    with open('../models/model_{}.json'.format(model_id)) as json_file:
        model = model_from_json(json_file.read())
    model.load_weights('../models/model_{}.h5'.format(model_id))
    model.compile(optimizer='sgd', loss=loss_function)
    return model

def plot_result(forecast, actual, mean=0, std=1, export_path=None):
    plt.ylabel('data [GB]')
    plt.xlabel('sample')
    plt.plot((std * actual + mean), 'g', label='actual')
    plt.plot((std * forecast + mean), 'r--', label='forecast')
    plt.legend()

    if export_path is None:
        plt.show()
    else:
        plt.savefig(export_path)

def multi_step_prediction(model, observations, window, horizon):

    if len(observations) < len(window):
        raise ValueError('number of observations is too small')

    predictions = np.array([])
    for _ in range(horizon):
        observations = observations[len(observations)-len(window):]
        x = np.array(list(compress(observations, window))).reshape(1, len(window))
        prediction = model.predict_on_batch(x)
        predictions = np.append(predictions, prediction.flatten())
        observations = np.append(observations, prediction.flatten())
    return np.array(predictions)
