#!/usr/bin/env python3
# encoding: utf-8

import pandas as pd


def read_dataset(path):
    data = pd.read_csv(path, parse_dates='Time', index_col='Time')
    data = data / 8 / 2**30 # convert bits into GB
    data.rename(columns={'Internet traffic data (in bits)':
                         'Internet traffic data (in GB)'}, inplace=True)
    return data
