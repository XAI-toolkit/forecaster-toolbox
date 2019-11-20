# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.scorer import make_scorer
from utils import mean_absolute_percentage_error, root_mean_squared_error, series_to_supervised

#===============================================================================
# grid_search_best ()
#===============================================================================
def grid_search_best(reg_type, X, Y):
    """
    Perform Grid Search on a model and return best hyper-parameters based on R2 
    error minimization.
    Arguments:
        reg_type: Type of regressor as a string.
        X: Independent variable values of observations as a NumPy array.
        Y: Dependent variable values of observations as a NumPy array.
    Returns:
        The best model hyper-parameters as a dict.
    """

    # Chosing hyperparameters based on best score during TimeSeriesSplit Validation
    tscv = TimeSeriesSplit(n_splits=5) 
    
    scaler = StandardScaler()
    
    # Create the regressor model and parameters range
    if reg_type == 'lasso_regression':
        regressor = Lasso()
        pipeline = Pipeline([('regressor', regressor)])
        parameters = {'regressor__alpha': [1, 10, 100, 1000, 100000, 1000000, 10000000]}
    elif reg_type == 'ridge_regression':
        regressor = Ridge()
        pipeline = Pipeline([('regressor', regressor)])
        parameters = {'regressor__alpha': [1, 10, 100, 1000, 100000, 1000000, 10000000]}
    elif reg_type == 'svr_linear':
        # Fitting linear SVR to the dataset
        regressor = SVR(kernel = 'linear')
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        parameters = {'regressor__C': [0.1, 1, 10, 100, 1000, 10000, 100000]}
    elif reg_type == 'svr_rbf':
        # Fitting SVR to the dataset
        regressor = SVR(kernel = 'rbf')
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        parameters = {'regressor__C': [0.1, 1, 10, 100, 1000, 10000, 100000], 'regressor__gamma' : [1, 0.1, 0.01, 0.001]}
    elif reg_type == 'random_forest':
        # Fitting Random Forest Regression to the dataset
        regressor = RandomForestRegressor()
        pipeline = Pipeline([('regressor', regressor)])
        parameters = {'regressor__n_estimators' : [5, 10, 100, 500], 'regressor__max_depth': [5, 10]}
        
    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, parameters, cv = tscv)
    grid_search = grid_search.fit(X, Y.ravel())
    
    # best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    
    print('=========================== Grid Search ===========================')
    print(' - Regressor: ', reg_type)
    print(' - Best Parameters: ', best_parameters)
    return best_parameters

#===============================================================================
# cross_validation_best ()
#===============================================================================
def cross_validation_best(pipes, X, Y):
    """
    Perform TimeSeriesSplit Validation to a list of models and return best based
    on R2 error minimization.
    Arguments:
        pipelines: A list of models integrated into a Pipeline.
        X: Independent variable values of observations as a NumPy array.
        Y: Dependent variable values of observations as a NumPy array.
    Returns:
        The best model integrated into a Pipeline.
    """

    # Chosing regressor based on best score during TimeSeriesSplit Validation
    tscv = TimeSeriesSplit(n_splits=5) 
    
    # Scores that will be computed during TimeSeriesSplit Validation
    scorer = {'neg_mean_absolute_error': 'neg_mean_absolute_error', 'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2', 'mean_absolute_percentage_error': make_scorer(mean_absolute_percentage_error, greater_is_better=False), 'root_mean_squared_error': make_scorer(root_mean_squared_error, greater_is_better=False)}
        
    # Perform TimeSeriesSplit Validation and compute metrics
    best_score = float('-inf')
    best_regressor = None
    for pipe in pipes:
        scores = cross_validate(estimator = pipe, X = X, y = Y.ravel(), scoring = scorer, cv = tscv, return_train_score = False)
        if scores['test_r2'].mean() > best_score:
            best_score = scores['test_r2'].mean()
            best_regressor = pipe
    
    print('=================== TimeSeriesSplit Validation ====================')
    print(' - Best Regressor: ', best_regressor)
    print(' - Best Score: ', best_score)
    return best_regressor

#===============================================================================
# create_regressor ()
#===============================================================================
def create_regressor(reg_type, X, Y):   
    """
    Create and train a regressor based on given X and Y values. Regressor type
    can be provided manually by the user or selected automatically based on R2 
    error minimization.
    Arguments:
        reg_type: Type of regressor as a string.
        X: Independent variable values of observations as a NumPy array.
        Y: Dependent variable values of observations as a NumPy array.
    Returns:
        A fitted sklearn model integrated into a Pipeline.
    """
    
    scaler = StandardScaler()
        
    # Create the regressor model
    if reg_type == 'linear_regression':
        # Fitting Multiple Linear Regression to the Training set
        regressor = LinearRegression()
        pipeline = Pipeline([('regressor', regressor)])
    elif reg_type == 'lasso_regression':
        # Fitting Lasso Regression to the Training set
        best_parameters = grid_search_best('lasso_regression', X, Y)
        regressor = Lasso(alpha = best_parameters['regressor__alpha'])
        pipeline = Pipeline([('regressor', regressor)])
    elif reg_type == 'ridge_regression':
        # Fitting Ridge Regression to the Training set
        best_parameters = grid_search_best('ridge_regression', X, Y)
        regressor = Ridge(alpha = best_parameters['regressor__alpha'])
        pipeline = Pipeline([('regressor', regressor)])
    elif reg_type == 'svr_linear':
        # Fitting linear SVR to the dataset
        best_parameters = grid_search_best('svr_linear', X, Y)
        regressor = SVR(kernel = 'linear', C = best_parameters['regressor__C'])
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
    elif reg_type == 'svr_rbf':
        # Fitting SVR to the dataset
        best_parameters = grid_search_best('svr_rbf', X, Y)
        regressor = SVR(kernel = 'rbf', gamma = best_parameters['regressor__gamma'], C = best_parameters['regressor__C'])
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
    elif reg_type == 'random_forest':
        # Fitting Random Forest Regression to the dataset
        best_parameters = grid_search_best('random_forest', X, Y)
        regressor = RandomForestRegressor(n_estimators = best_parameters['regressor__n_estimators'], max_depth = best_parameters['regressor__max_depth'], random_state = 0)
        pipeline = Pipeline([('regressor', regressor)])
    elif reg_type == 'auto':
        # Fitting Multiple Linear Regression to the Training set
        regressor_linear = LinearRegression()
        # Fitting Lasso Regression to the Training set
        best_parameters = grid_search_best('lasso_regression', X, Y)
        regressor_lasso = Lasso(alpha = best_parameters['regressor__alpha'])
        # Fitting Ridge Regression to the Training set
        best_parameters = grid_search_best('ridge_regression', X, Y)
        regressor_ridge = Ridge(alpha = best_parameters['regressor__alpha'])
        # Fitting linear SVR to the dataset
        best_parameters = grid_search_best('svr_linear', X, Y)
        regressor_svr_linear = SVR(kernel = 'linear', C = best_parameters['regressor__C'])
        # Fitting SVR to the dataset
        best_parameters = grid_search_best('svr_rbf', X, Y)
        regressor_svr_rbf = SVR(kernel = 'rbf', gamma = best_parameters['regressor__gamma'], C = best_parameters['regressor__C'])
        # Fitting Random Forest Regression to the dataset
        best_parameters = grid_search_best('random_forest', X, Y)
        regressor_random_forest = RandomForestRegressor(n_estimators = best_parameters['regressor__n_estimators'], max_depth = best_parameters['regressor__max_depth'], random_state = 0)
        # Perform TimeSeriesSplit Validation and return best model
        pipes = [Pipeline([('regressor', regressor_linear)]), Pipeline([('regressor', regressor_lasso)]), Pipeline([('regressor', regressor_ridge)]), Pipeline([('scaler', scaler), ('regressor', regressor_svr_linear)]), Pipeline([('scaler', scaler), ('regressor', regressor_svr_rbf)]), Pipeline([('regressor', regressor_random_forest)])]
        pipeline = cross_validation_best(pipes, X, Y)
    
    # Fit selected model to the dataset
    pipeline.fit(X, Y.ravel())
    
    return pipeline

#===============================================================================
# build_and_train ()
#===============================================================================
def build_and_train(horizon_param, regressor_param):
    """
    Build forecasting models and return forecasts for an horizon specified by the user.
    Arguments:
        horizon_param: The forecasting horizon up to which forecasts will be produced.
        regressor_param: The regressor models that will be used to produce forecasts.
    Returns:
        A dictionary containing forecasted values for each intermediate step ahead up to the specified horizon.
    """
    
    # selecting indicators that will be used as model variables
    METRICS_TD = ['bugs', 'vulnerabilities', 'code_smells', 'sqale_index', 'reliability_remediation_effort', 'security_remediation_effort']
    # Select sliding window length
    WINDOW_SIZE = 2
    
    # Read dataset
    dataset = pd.read_csv('apache_kafka_measures.csv', sep=";", usecols = METRICS_TD)
    dataset['total_principal'] = dataset['reliability_remediation_effort'] + dataset['security_remediation_effort'] + dataset['sqale_index']
    dataset = dataset.drop(columns=['sqale_index', 'reliability_remediation_effort', 'security_remediation_effort'])
    
    # Read dataset date
    dataset_date = pd.read_csv('apache_kafka_measures.csv', sep=";", usecols = ['date'])
    
    dict_result = {}
    list_forecasts = []

    # Make forecasts using the Direct approach, i.e. train separate models for each forecasting horizon
    for intermediate_horizon in range (1, horizon_param+1):
        print('=========================== Horizon: %s ============================' % intermediate_horizon)
        
        # Add time-shifted prior and future period
        data = series_to_supervised(dataset, n_in = WINDOW_SIZE)
        
        # Append dependend variable column with value equal to total_principal of the target horizon's version
        data['forecasted_total_principal'] = data['total_principal(t)'].shift(-intermediate_horizon)
        data = data.drop(data.index[-intermediate_horizon:])
        
        # Remove TD as independent variable
        data = data.drop(columns=['total_principal(t-%s)' % (i) for i in range(WINDOW_SIZE, 0, -1)]) 
        
        # Define independent and dependent variables
        X = data.iloc[:, data.columns != 'forecasted_total_principal'].values
        Y = data.iloc[:, data.columns == 'forecasted_total_principal'].values
        
        #==============#
        #  Test model  #
        #==============#
        # Split data to training/test set to test model
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = horizon_param, random_state = 0, shuffle = False)
        # Make forecasts for training/test set
        regressor = create_regressor(regressor_param, X_train, Y_train)
        y_pred = regressor.predict(X_test)
        
        #==============#
        # Deploy model #
        #==============#
        # Define X to to deploy model for real forecasts
        # X_real = series_to_supervised(dataset, n_in = WINDOW_SIZE, dropnan=False)
        # X_real = X_real.drop(columns=['total_principal(t-%s)' % (i) for i in range(WINDOW_SIZE, 0, -1)]) 
        # X_real = X_real.iloc[-1, :].values
        # X_real = X_real.reshape(1, -1)        
        # Make real forecasts
        # regressor = create_regressor(regressor_param, X, Y)
        # y_pred = regressor.predict(X_real)
    
        # Fill dataframe with forecasts
        temp_dict = {
                        'x': dataset_date['date'].iloc[len(dataset_date['date'])-(horizon_param-intermediate_horizon+1)],
                        'y': y_pred[0]
                    }
        list_forecasts.append(temp_dict)
        
    # Fill results dictionary with forecasts
    dict_result['forecasts'] = list_forecasts
    
    print(dict_result)
    return(dict_result)
    
#build_and_train(5, 'auto')