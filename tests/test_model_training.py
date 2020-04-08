# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from pmdarima import auto_arima
from pmdarima import ARIMA

def test_grid_search_best_lasso():
    x_array = np.arange(150).reshape(50,3)
    y_array = np.arange(50).reshape(50,1)

    # Chosing hyperparameters based on best score during TimeSeriesSplit Validation
    splits = int(len(x_array) / 30) if len(x_array) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=splits)

    regressor = Lasso()
    pipeline = Pipeline([('regressor', regressor)])
    parameters = {'regressor__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000, 1000000, 10000000]}

    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, parameters, cv=tscv)
    grid_search = grid_search.fit(x_array, y_array.ravel())

    # best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    assert 'regressor__alpha' in best_parameters

def test_grid_search_best_ridge():
    x_array = np.arange(150).reshape(50,3)
    y_array = np.arange(50).reshape(50,1)

    # Chosing hyperparameters based on best score during TimeSeriesSplit Validation
    splits = int(len(x_array) / 30) if len(x_array) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=splits)

    regressor = Ridge()
    pipeline = Pipeline([('regressor', regressor)])
    parameters = {'regressor__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000, 1000000, 10000000]}

    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, parameters, cv=tscv)
    grid_search = grid_search.fit(x_array, y_array.ravel())

    # best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    assert 'regressor__alpha' in best_parameters
    
def test_grid_search_best_svr_linear():
    x_array = np.arange(150).reshape(50,3)
    y_array = np.arange(50).reshape(50,1)

    # Chosing hyperparameters based on best score during TimeSeriesSplit Validation
    splits = int(len(x_array) / 30) if len(x_array) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=splits)
    scaler = StandardScaler()
    
    regressor = SVR(kernel='linear')
    pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
    parameters = {'regressor__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, parameters, cv=tscv)
    grid_search = grid_search.fit(x_array, y_array.ravel())

    # best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    assert 'regressor__C' in best_parameters

def test_grid_search_best_svr_rbf():
    x_array = np.arange(150).reshape(50,3)
    y_array = np.arange(50).reshape(50,1)

    # Chosing hyperparameters based on best score during TimeSeriesSplit Validation
    splits = int(len(x_array) / 30) if len(x_array) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=splits)
    scaler = StandardScaler()
    
    regressor = SVR(kernel='rbf')
    pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
    parameters = {'regressor__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'regressor__gamma' : [1, 0.1, 0.01, 0.001]}

    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, parameters, cv=tscv)
    grid_search = grid_search.fit(x_array, y_array.ravel())

    # best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    assert 'regressor__C' and 'regressor__gamma' in best_parameters

def test_grid_search_best_random_forest():
    x_array = np.arange(150).reshape(50,3)
    y_array = np.arange(50).reshape(50,1)

    # Chosing hyperparameters based on best score during TimeSeriesSplit Validation
    splits = int(len(x_array) / 30) if len(x_array) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=splits)
    
    regressor = RandomForestRegressor()
    pipeline = Pipeline([('regressor', regressor)])
    parameters = {'regressor__n_estimators' : [5, 10, 100, 500], 'regressor__max_depth': [5, 10]}

    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, parameters, cv=tscv)
    grid_search = grid_search.fit(x_array, y_array.ravel())

    # best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    assert 'regressor__n_estimators' and 'regressor__max_depth' in best_parameters

def test_grid_search_best_arima():
    y_array = np.ones(100)

    stepwise_model = auto_arima(y_array, m=1, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)

    assert isinstance(stepwise_model, ARIMA)

def test_cross_validation_best():
    scaler = StandardScaler()
    pipes = pipes = [Pipeline([('regressor', LinearRegression())]), Pipeline([('regressor', Lasso())]), Pipeline([('regressor', Ridge())]), Pipeline([('scaler', scaler), ('regressor', SVR(kernel='linear'))]), Pipeline([('scaler', scaler), ('regressor', SVR(kernel='rbf'))]), Pipeline([('regressor', RandomForestRegressor())])]
    x_array = np.arange(150).reshape(50,3)
    y_array = np.arange(50).reshape(50,1)

    # Chosing regressor based on best score during TimeSeriesSplit Validation
    splits = int(len(x_array) / 30) if len(x_array) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=splits)

    # Scores that will be computed during TimeSeriesSplit Validation
    scorer = {'neg_mean_absolute_error': 'neg_mean_absolute_error', 'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}

    # Perform TimeSeriesSplit Validation and compute metrics
    best_score = float('-inf')
    best_regressor = None
    for pipe in pipes:
        scores = cross_validate(estimator=pipe, X=x_array, y=y_array.ravel(), scoring=scorer, cv=tscv, return_train_score=False)
        if scores['test_r2'].mean() > best_score:
            best_score = scores['test_r2'].mean()
            best_regressor = pipe

#    print(best_regressor['regressor'])
    assert best_regressor['regressor'] is not None
