#!/usr/bin/env python3
# encoding: utf-8


import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import compress
from utils import read_dataset, split_dataset
from math import sqrt
from evaluation import mase

def read_model():
    start_time = time.time()

    print('loading and prepare data set...')
    data_5m = read_dataset('../datasets/internet-traffic-data-5minutes.csv')
    data_hoursly = read_dataset('../datasets/internet-traffic-data-hourly.csv')
    data_daily = read_dataset('../datasets/internet-traffic-data-daily.csv')

    # TODO: ändern beim merge...
    data_5m = data_5m['Internet traffic data (in GB)'].tolist()
    data_daily = data_daily['Internet traffic data (in GB)'].tolist()
    data_hoursly = data_hoursly['Internet traffic data (in GB)'].tolist()

    rest_5m = int(len(data_5m) * 0.80) % 288
    start_forecast_idx_5m = int(len(data_5m) * 0.90) - rest_5m
    train_5m = data_5m[:start_forecast_idx_5m]
    test_5m = data_5m[start_forecast_idx_5m:]

    rest_hoursly = int(len(data_hoursly) * 0.80) % 168
    start_forecast_idx_hoursly = int(len(data_hoursly) * 0.90) - rest_hoursly
    train_hoursly = data_hoursly[:start_forecast_idx_hoursly]
    test_hoursly = data_hoursly[start_forecast_idx_hoursly:]

    rest_daily = int(len(data_daily) * 0.80) % 7
    start_forecast_idx_daily = int(len(data_daily) * 0.90) - rest_daily
    train_daily = data_daily[:start_forecast_idx_daily]
    test_daily = data_daily[start_forecast_idx_daily:]


    print('5min: ', len(train_5m), " ", len(test_5m))
    print('hoursly: ', len(train_hoursly), " ", len(test_hoursly))
    print('daily: ', len(train_daily), " ", len(test_daily), "\n")

    plot_daily(train_daily, test_daily)
    plot_hoursly(train_hoursly, test_hoursly)
    plot_5min(train_5m, test_5m)

    #errors
    plot_errors(train_5m, test_5m, train_hoursly, test_hoursly, train_daily, test_daily)


def plot_daily(train, test):
    #linear
    lin_alpha = 0.1
    lin_beta = 0.2
    # error: 1.17685709586

    # multiadaptive
    mult_alpha = 0.0
    mult_beta = 0.0
    mult_gamma = 1.0
    period = 7
    # error: 0.316808789116

    print("Calculate holt-winters for daily traffic ...")
    pred_linear = holts_linear(train, len(test), lin_alpha, lin_beta)
    error = mase(train, test, pred_linear, seasonal_period=1)
    print("Linear Error: ", error)

    pred_adaptive = holts_additive(train, period, mult_alpha, mult_beta, mult_gamma, len(test))
    error = mase(train, test, pred_adaptive, seasonal_period=1)
    print("Adaptive Error: ", error, "\n")

    #plt.title('alpha: %.1f, beta: %.1f, gamma: %.1f, period: %d MSE: %.7f'%(alpha, beta, gamma, period, error))
    plt.xlabel('step')
    plt.ylabel('traffic')
    plt.plot(test, label='series')
    plt.plot(pred_linear, label='linear')
    plt.plot(pred_adaptive, label='adaptive (period: 7)')
    plt.legend(loc=4, scatterpoints=1)
    plt.savefig('daily.png')
    plt.clf();

def plot_hoursly(train, test):
    # linear
    lin_alpha = 0.15
    lin_beta = 0.1
    #Linear Error:  4.34134852342

    # adaptive
    mult_alpha = 0.7
    mult_beta = 0.0
    mult_gamma = 1.0
    period1 = 24 #day
    period2 = 168 #week
    #Adaptive Error (Period 24):  3.24448701448
    #Adaptive Error (Period 168):  5.49937414585

    print("Calculate holt-winters for hour traffic ...")
    pred_linear = holts_linear(train, len(test), lin_alpha, lin_beta)
    error = mase(train, test, pred_linear, seasonal_period=1)
    print("Linear Error: ", error)

    pred_adaptive24 = holts_additive(train, period1, mult_alpha, mult_beta, mult_gamma, len(test))
    error = mase(train, test, pred_adaptive24, seasonal_period=1)
    print("Adaptive Error (Period 24): ", error)

    pred_adaptive168 = holts_additive(train, period2, mult_alpha, mult_beta, mult_gamma, len(test))
    error = mase(train, test, pred_adaptive168, seasonal_period=1)
    print("Adaptive Error (Period 168): ", error, "\n")

    ax1 = plt.subplot()
    ax1.set_xlim([0, len(test)])
    plt.xlabel('step')
    plt.ylabel('traffic')
    plt.plot(test, label='series')
    plt.plot(pred_linear, label='linear')
    plt.plot(pred_adaptive24, label='adaptive (period: 24)')
    plt.plot(pred_adaptive168, label='adaptive (period: 168)')
    plt.legend(loc=4, scatterpoints=1)
    plt.savefig('hoursly.png')
    plt.clf();


def plot_5min(train, test):
    # linear
    #lin_alpha = 0.35
    #lin_beta = 0.08
    #Linear Error:  4.34134852342

    # adaptive
    mult_alpha = 0.9
    mult_beta = 0.65
    mult_gamma = 0.9
    period1 = 12 #hour
    period2 = 288 #day

    print("Calculate holt-winters for 5 min traffic ...")
    #pred_linear = holts_linear(train, len(test), lin_alpha, lin_beta)
    #error = mase(train, test, pred_linear, seasonal_period=1)
    #print("Linear Error: ", error)

    #pred_adaptive_hour = holts_additive(train, period1, mult_alpha, mult_beta, mult_gamma, len(test))
    #error = mase(train, test, pred_adaptive_hour, seasonal_period=1)
    #print("Adaptive Error (Period 12): ", error)

    pred_adaptive_day = holts_additive(train, period2, mult_alpha, mult_beta, mult_gamma, len(test))
    error = mase(train, test, pred_adaptive_day, seasonal_period=1)
    print("Adaptive Error (Period 288): ", error, "\n")

    ax1 = plt.subplot()
    ax1.set_xlim([0, len(test)])
    plt.xlabel('step')
    plt.ylabel('traffic')
    plt.plot(test, label='series')
    #plt.plot(pred_linear, label='linear')
    #plt.plot(pred_adaptive_hour, label='adaptive (Period: 12)')
    plt.plot(pred_adaptive_day, label='adaptive (Period: 288)')
    plt.legend(loc=4, scatterpoints=1)
    plt.savefig('5min.png')
    plt.clf();


def plot_errors(train_5m, test_5m, train_hoursly, test_hoursly, train_daily, test_daily):
    print("Calculate error-plot ...")
    plt.title('Error')
    add_error_for_errorplot_linear(train_daily, test_daily, 0.1, 0.2, 'daily, linear')
    add_error_for_errorplot_adaptive(train_daily, test_daily, 0.0, 0.0, 1.0, 7, 'daily, period: 7')
    add_error_for_errorplot_linear(train_hoursly, test_hoursly, 0.15, 0.1, 'hour, linear')
    add_error_for_errorplot_adaptive(train_hoursly, test_hoursly, 0.7, 0.0, 1.0, 24, 'hour, period: 24')
    add_error_for_errorplot_adaptive(train_hoursly, test_hoursly, 0.7, 0.0, 1.0, 168, 'hour, period: 168')
    add_error_for_errorplot_adaptive(train_5m, test_5m, 0.9, 0.65, 0.9, 288, '5 min, period: 288')

    plt.xlabel('time')
    plt.ylabel('mse')
    plt.legend(loc=2, scatterpoints=1)
    plt.savefig('error.png')
    plt.clf();

def add_error_for_errorplot_adaptive(train, test, alpha, beta, gamma, period, label):
    error = []
    for ran in range(1, len(test), 1):
        pred = holts_additive(train, period, alpha, beta, gamma, ran)
        error.append(mase(train, test[:ran], pred, seasonal_period=1))

    x = np.arange(0, 1+1/len(error), 1/len(error))
    error.insert(0, 0)
    plt.plot(x, error, label=label)

def add_error_for_errorplot_linear(train, test, alpha, beta, label):
    error = []
    for ran in range(1, len(test), 1):
        pred = holts_linear(train, ran, alpha, beta)
        error.append(mase(train, test[:ran], pred, seasonal_period=1))

    x = np.arange(0, 1+1/len(error), 1/len(error))
    error.insert(0, 0)
    plt.plot(x, error, label=label)


def plot_list_multi(test, pred, alpha, beta, gamma, error, fc):
    plt.title('alpha: %.3f, beta: %.3f, gamma: %.3f, RMSE: %.7f'%(alpha, beta, gamma, error))
    plt.plot(test, label='series')
    plt.plot(pred, label='predicted')
    plt.legend(loc=2, scatterpoints=1)
    name = 'multi_fc_{fc}.png'.format(fc=fc)
    plt.savefig(name)
    plt.clf();


def train_for_linear_model(train, test, fc):
    best_alpha = 0;
    best_beta = 0;
    best_mse = 10000000;

    for alpha in np.arange(0.1, 1.0, 0.05):
        for beta in np.arange(0.1, 1.0, 0.05):
            pred = holts_linear(train, fc, alpha, beta)
            error = mase(train, test, pred, seasonal_period=1)

            if error < best_mse:
                best_mse = error
                best_alpha = alpha
                best_beta = beta
                print("New Minimum: Alpha:", best_alpha, " Beta:", best_beta, " MSE:", best_mse)
        print(alpha, "%")

    print("Alpha:", best_alpha, " Beta:", best_beta, " MSE:", best_mse)


def train_for_adaptive_model(train, test, fc, period):
    best_alpha = 0;
    best_beta = 0;
    best_gamma = 0;
    best_mse = 10000000;

    for alpha in np.arange(0.1, 1.0, 0.05):
        for beta in np.arange(0.1, 1.0, 0.05):
            for gamma in np.arange(0.1, 1.0, 0.05):
                pred = holts_additive(train, period, alpha, beta, gamma, fc)
                error = mase(train, test, pred, seasonal_period=1)

                if error < best_mse:
                    best_mse = error
                    best_alpha = alpha
                    best_beta = beta
                    best_gamma = gamma
                    print("New Minimum: Alpha:", best_alpha, " Beta:", best_beta, " Gamma:", best_gamma, " MSE:", best_mse)
        print(alpha, "%")

    print("Alpha:", best_alpha, " Beta:", best_beta, " Gamma:", best_gamma, " MSE:", best_mse)


def holts_linear(x, fc, alpha, beta):
    Y = x[:]
    a = [Y[0]]
    b = [Y[1] - Y[0]]
    y = [a[0] + b[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1] + b[-1])

        a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        y.append(a[i + 1] + b[i + 1])
    return Y[-fc:]


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen


def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals


def holts_additive(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = len(series) - i + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result[len(series):]


def holts_double_sessional(x, m, m2, fc, alpha, beta, gamma, delta, autocorrelation):
    Y = x[:]

    a = [sum(Y[0:m]) / float(m)]
    s = [Y[i] / a[0] for i in range(m)]
    s2 = [Y[i] / a[0] for i in range(0,m2,m)]
    y = [a[0] + s[0] + s2[0]]
    mse = 0

    for i in range(len(Y) + fc):

        if i >= len(Y):
            Y.append(a[-1] +  s[-m] + s2[-m2])

        a.append(alpha * (Y[i] - s2[i] - s[i]) + (1 - alpha) * (a[i]))
        s.append(delta *  (Y[i] - a[i] - s2[i]) + (1 - delta) * s[i])
        s2.append(gamma * (Y[i] - a[i] - s[i]) + (1 - gamma) * s2[i])
        autocorr = autocorrelation * (Y[i] - (a[i] + s[i] + s2[i]))
        y.append(a[i + 1] + s[i + 1] + s2[i + 1] + autocorr)

    #mse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
    return {'Y':Y[-fc:], 'mse':mse}


if __name__ == '__main__':
    read_model()
