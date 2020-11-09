# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import pytest
import numpy as np
import pandas as pd
from bson import ObjectId
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
        'var1(t+1)': [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,float('nan')],
        'var2(t+1)': [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,float('nan')],
        'var3(t+1)': [3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,float('nan')],
    }
    return pd.DataFrame(x_object)

@pytest.fixture
def pipe_input():
    scaler = StandardScaler()
    pipes = [Pipeline([('regressor', LinearRegression())]), Pipeline([('regressor', Lasso())]), Pipeline([('regressor', Ridge())]), Pipeline([('scaler', scaler), ('regressor', SVR(kernel='linear'))]), Pipeline([('scaler', scaler), ('regressor', SVR(kernel='rbf'))]), Pipeline([('regressor', RandomForestRegressor())])]
    return pipes
    
@pytest.fixture
def dict_oid_input():
    dict_oid = {'key': ObjectId('582431a6a377f26970c543b3')}
    return dict_oid

@pytest.fixture
def forecasting_horizon_input():
    horizon = 10
    return horizon

@pytest.fixture
def td_forecasting_project_input():
    project = 'imd_technical_debt'
    return project

@pytest.fixture
def energy_forecasting_project_input():
    project = 'neurasmus'
    return project

@pytest.fixture
def dependability_forecasting_project_input():
    project = 'sdk4ed-healthcare-use-case'
    return project

@pytest.fixture
def forecasting_project_class_level_input():
    project = 'apache_kafka'
    return project

@pytest.fixture
def forecasting_class_number_input():
    classes = 10
    return classes
