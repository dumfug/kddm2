#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.dates as mdates


def read_dataset(path, date_parser=None):
    data = pd.read_csv(path, parse_dates='Time', index_col='Time')
    return data

def plot_full_timeseries(ts, export_path=None):
    ts_scaled = ts / 8 / 2**30 # convert bits into GB
    plt.plot(ts_scaled)

    plt.title('Internet Traffic Data collected at Transatlantic Link')
    plt.ylabel('data [GB]')
    plt.xlabel('time')

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=3))
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    plt.gca().xaxis.set_tick_params(which='major', pad=15)

    if export_path is None:
        plt.show()
    else:
        plt.savefig(export_path)

def plot_all_full_timeseries(ts_5min, ts_hourly, ts_daily, export_path=None):
    ts_scaled = ts_5min / 8 / 2**30 # convert bits into GB
    plt.plot(ts_scaled, label='5 minutes')
    ts_scaled = ts_hourly / 8 / 2**30 # convert bits into GB
    plt.plot(ts_scaled, label='hourly')
    ts_scaled = ts_daily / 8 / 2**30 # convert bits into GB
    plt.plot(ts_scaled, label='daily')

    plt.legend(loc=4, fontsize='medium')
    plt.title('Internet Traffic Data collected at Transatlantic Link')
    plt.ylabel('data [GB]  (logarithmic scale)')
    plt.xlabel('time')

    plt.gca().set_yscale('log')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B'))
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=3))
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    plt.gca().xaxis.set_tick_params(which='major', pad=15)

    if export_path is None:
        plt.show()
    else:
        plt.savefig(export_path)

def plot_interval_of_timeseries(ts, start_day, end_day, export_path=None):
    ts_interval = ts[start_day + ' 00:00:00' : end_day + ' 23:59:59']
    ts_interval = ts_interval / 8 / 2**30 # convert bits into GB

    plt.plot(ts_interval)
    plt.ylabel('data [GB]')
    plt.xlabel('time')
    if start_day == end_day:
        plt.title('Internet Traffic Data collected on ' + start_day)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        plt.title('Data collected between ' + start_day + ' and ' + end_day)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a'))

    if export_path is None:
        plt.show()
    else:
        plt.savefig(export_path)

def plot_autocorrelation(ts, max_lag):
    pd.tools.plotting.autocorrelation_plot(ts)
    plt.gca().set_xlim([0, max_lag])
    plt.show()

def plot_daily_means(ts):
    ext_ts = ts / 8 / 2**30
    ext_ts['weekday'] = ext_ts.index.weekday
    ext_ts.boxplot(column=['Internet traffic data (in bits)'], by='weekday')

    plt.title('')
    plt.ylabel('data [GB]')
    plt.xlabel('time')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    plt.show()

if __name__ == '__main__':
    ts_5minutes = read_dataset('datasets/internet-traffic-data-5minutes.csv')
    ts_hourly = read_dataset('datasets/internet-traffic-data-hourly.csv')
    ts_daily = read_dataset('datasets/internet-traffic-data-daily.csv')

    plot_all_full_timeseries(ts_5minutes, ts_hourly, ts_daily)

    plot_full_timeseries(ts_5minutes)
    plot_full_timeseries(ts_hourly)
    plot_full_timeseries(ts_daily)

    plot_interval_of_timeseries(ts_5minutes, '2005-06-22', '2005-06-22')
    plot_interval_of_timeseries(ts_5minutes, '2005-07-04', '2005-07-10')

    plot_daily_means(ts_5minutes)
