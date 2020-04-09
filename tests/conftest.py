# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import pytest
import numpy as np
import pandas as pd
from bson import ObjectId

@pytest.fixture
def x_vector_input():
    x_vector = np.random.rand(100,1)
    return x_vector

@pytest.fixture
def x_array_input():
    x_array = np.arange(150).reshape(50,3)
    return x_array

@pytest.fixture
def y_array_input():
    y_array = np.arange(50).reshape(50,1)
    return y_array

@pytest.fixture
def y_array_input_arima():
    y_array = np.ones(100)
    return y_array

@pytest.fixture
def x_dataframe_input():
    x_object = {
        'var1': [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        'var2': [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0],
        'var3': [3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]
    }
    return pd.DataFrame(x_object)

@pytest.fixture
def x_dataframe_output():
    x_object = {
        'var1(t-1)': [float('nan'),1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        'var2(t-1)': [float('nan'),2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0],
        'var3(t-1)': [float('nan'),3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0],
        'var1(t)': [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        'var2(t)': [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0],
        'var3(t)': [3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0],
    }
    return pd.DataFrame(x_object)

@pytest.fixture
def dict_oid_input():
    dict_oid = {'key': ObjectId('582431a6a377f26970c543b3')}
    return dict_oid

@pytest.fixture
def forecaster_url_input():
    base_url = 'http://localhost:5000'
    return base_url

@pytest.fixture
def forecasting_horizon_input():
    horizon = 10
    return horizon