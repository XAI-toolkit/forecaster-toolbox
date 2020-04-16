# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

from pmdarima import ARIMA

import sys
sys.path.append('..')
from model_training import grid_search_best, arima_search_best, cross_validation_best, create_regressor

def test_grid_search_best_lasso(x_array_input, y_array_input):
    assert 'regressor__alpha' in grid_search_best('lasso', x_array_input, y_array_input)

def test_grid_search_best_ridge(x_array_input, y_array_input):
    assert 'regressor__alpha' in grid_search_best('ridge', x_array_input, y_array_input)

def test_grid_search_best_svr_linear(x_array_input, y_array_input):
    assert 'regressor__C' in grid_search_best('svr_linear', x_array_input, y_array_input)

def test_grid_search_best_svr_rbf(x_array_input, y_array_input):
    assert 'regressor__C' and 'regressor__gamma' in grid_search_best('svr_rbf', x_array_input, y_array_input)

def test_grid_search_best_random_forest(x_array_input, y_array_input):
    assert 'regressor__n_estimators' and 'regressor__max_depth' in grid_search_best('random_forest', x_array_input, y_array_input)

def test_grid_search_best_arima(y_array_input_arima):
    assert isinstance(arima_search_best(y_array_input_arima), ARIMA)

def test_cross_validation_best(pipe_input, x_array_input, y_array_input):
    assert cross_validation_best(pipe_input, x_array_input, y_array_input)['regressor'] is not None

def test_create_regressor_mlr(x_array_input, y_array_input):
    assert create_regressor('mlr', x_array_input, y_array_input)['regressor'] is not None

def test_create_regressor_lasso(x_array_input, y_array_input):
    assert create_regressor('lasso', x_array_input, y_array_input)['regressor'] is not None

def test_create_regressor_ridge(x_array_input, y_array_input):
    assert create_regressor('ridge', x_array_input, y_array_input)['regressor'] is not None

def test_create_regressor_svr_linear(x_array_input, y_array_input):
    assert create_regressor('svr_linear', x_array_input, y_array_input)['regressor'] is not None

def test_create_regressor_svr_rbf(x_array_input, y_array_input):
    assert create_regressor('svr_rbf', x_array_input, y_array_input)['regressor'] is not None

def test_create_regressor_random_forest(x_array_input, y_array_input):
    assert create_regressor('random_forest', x_array_input, y_array_input)['regressor'] is not None

def test_create_regressor_auto(x_array_input, y_array_input):
    assert create_regressor('auto', x_array_input, y_array_input)['regressor'] is not None
