# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from bson import ObjectId

def test_mean_absolute_percentage_error(x_vector_input):   
    assert np.mean(np.abs((x_vector_input - x_vector_input) / x_vector_input)) * 100 == 0

def test_root_mean_squared_error(x_vector_input):   
    assert sqrt(mean_squared_error(x_vector_input, x_vector_input)) == 0

def test_series_to_supervised(x_dataframe_input, x_dataframe_output, n_in=1, n_out=1, dropnan=False):
    data = x_dataframe_input.values
    labels = x_dataframe_input.columns.tolist()
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

    assert agg.equals(pd.DataFrame(x_dataframe_output))

def test_objectid_to_string(dict_oid_input):
    for key in dict_oid_input:
        if isinstance(dict_oid_input[key], ObjectId):
            dict_oid_input[key] = str(dict_oid_input[key])

    assert isinstance(dict_oid_input[key], str)
