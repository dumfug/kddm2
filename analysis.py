#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pylab as plt


def read_dataset(path):
    data = pd.read_csv(path, parse_dates='Time', index_col='Time')
    ts = data['Internet traffic data (in bits)']    
    
    print(ts.head())
    plt.plot(ts)
    plt.show()


if __name__ == '__main__':
    read_dataset('datasets/internet-traffic-data-2005-06-2005-07.csv')

