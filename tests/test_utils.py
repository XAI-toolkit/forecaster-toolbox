# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from bson import ObjectId

def test_mean_absolute_percentage_error():
    y_true = np.ones((10,1))
    y_pred = np.ones((10,1))
    
    assert np.mean(np.abs((y_true - y_pred) / y_true)) * 100 == 0

def test_root_mean_squared_error():
    y_true = np.ones((10,1))
    y_pred = np.ones((10,1))
    
    assert sqrt(mean_squared_error(y_true, y_pred)) == 0

def test_series_to_supervised():
    input_data = {
        'var1': [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        'var2': [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0],
        'var3': [3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]
    }
    output_data = {
        'var1(t-1)': [float('nan'),1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        'var2(t-1)': [float('nan'),2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0],
        'var3(t-1)': [float('nan'),3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0],
        'var1(t)': [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        'var2(t)': [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0],
        'var3(t)': [3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0],
    }
    dataset = pd.DataFrame(input_data)
    n_in = 1
    n_out = 1
    dropnan = False

    data = dataset.values
    labels = dataset.columns.tolist()
    n_vars = 1 if type(data) is list else data.shape[1]
    d_f = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(d_f.shift(i))
        names += [('%s(t-%d)' % (labels[j], i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(d_f.shift(-i))
        if i == 0:
            names += [('%s(t)' % (labels[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (labels[j], i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    assert agg.equals(pd.DataFrame(output_data))

def test_objectid_to_string():
    dict_obj = {'key': ObjectId('582431a6a377f26970c543b3')}

    for key in dict_obj:
        if isinstance(dict_obj[key], ObjectId):
            dict_obj[key] = str(dict_obj[key])

    assert isinstance(dict_obj[key], str)
