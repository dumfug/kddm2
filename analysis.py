#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.dates as mdates


def read_dataset(path, date_parser=None):
    data = pd.read_csv(path, parse_dates='Time', index_col='Time')
    data = data / 8 / 2**30 # convert bits into GB
    data.rename(columns={'Internet traffic data (in bits)':
      'Internet traffic data (in GB)'}, inplace=True)
    return data

def plot_full_timeseries(ts, export_path=None):
    plt.plot(ts)

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
    plt.plot(ts_5min, label='5 minutes')
    plt.plot(ts_hourly, label='hourly')
    plt.plot(ts_daily, label='daily')

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

def plot_acf(ts, max_lag, ticks, plot_title=''):
    pd.tools.plotting.autocorrelation_plot(ts)
    plt.title(plot_title)
    plt.gca().set_xlim([0, max_lag])
    plt.xticks(np.arange(0, max_lag+1, ticks))
    plt.show()

def plot_daily_means(ts):
    ext_ts = ts.copy()
    ext_ts['weekday'] = ext_ts.index.weekday
    ext_ts.boxplot(column=['Internet traffic data (in GB)'], by='weekday')

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
    plot_acf(ts_hourly, 200, 24, 'ACF hourly data (one week)')
    plot_acf(ts_5minutes, 300, 50, 'ACF 5 min. data (one day)')
