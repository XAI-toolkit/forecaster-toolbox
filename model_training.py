# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import os
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
from utils import mean_absolute_percentage_error, root_mean_squared_error, series_to_supervised, read_from_td_toolbox_api, read_from_dependability_toolbox_api

debug = bool(os.environ.get('DEBUG'))

#===============================================================================
# grid_search_best ()
#===============================================================================
def grid_search_best(reg_type, x_array, y_array):
    """
    Perform Grid Search on a model and return best hyper-parameters based on R2
    error minimization.
    Arguments:
        reg_type: Type of regressor as a string.
        x_array: Independent variable values of observations as a NumPy array.
        y_array: Dependent variable values of observations as a NumPy array.
    Returns:
        The best model hyper-parameters as a dict.
    """

    # Chosing hyperparameters based on best score during TimeSeriesSplit Validation
    splits = int(len(x_array) / 30) if len(x_array) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=splits)
    scaler = StandardScaler()

    # Create the regressor model and parameters range
    if reg_type == 'lasso':
        regressor = Lasso()
        pipeline = Pipeline([('regressor', regressor)])
        parameters = {'regressor__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000, 1000000, 10000000]}
    elif reg_type == 'ridge':
        regressor = Ridge()
        pipeline = Pipeline([('regressor', regressor)])
        parameters = {'regressor__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000, 1000000, 10000000]}
    elif reg_type == 'svr_linear':
        # Fitting linear SVR to the dataset
        regressor = SVR(kernel='linear')
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        parameters = {'regressor__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    elif reg_type == 'svr_rbf':
        # Fitting SVR to the dataset
        regressor = SVR(kernel='rbf')
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
        parameters = {'regressor__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'regressor__gamma' : [1, 0.1, 0.01, 0.001]}
    elif reg_type == 'random_forest':
        # Fitting Random Forest Regression to the dataset
        regressor = RandomForestRegressor()
        pipeline = Pipeline([('regressor', regressor)])
        parameters = {'regressor__n_estimators' : [5, 10, 100, 500], 'regressor__max_depth': [5, 10]}

    # Perform Grid Search
    grid_search = GridSearchCV(pipeline, parameters, cv=tscv)
    grid_search = grid_search.fit(x_array, y_array.ravel())

    # best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    if debug:
        print('=========================== Grid Search ===========================')
        print(' - Regressor: ', reg_type)
        print(' - Best Parameters: ', best_parameters)

    return best_parameters

#===============================================================================
# arima_search_best ()
#===============================================================================
def arima_search_best(y_array):
    """
    Perform auto_arima and return the model with best (p,d,q) parameters based 
    on AIC score minimization.
    Arguments:
        y_array: Values of observations as a NumPy array.
    Returns:
        The best model as an object.
    """

    # Perform auto_arima
    try:
        stepwise_model = auto_arima(y_array, start_p=1, start_q=1, max_p=5, max_q=5, m=52, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
    except Exception as e:
        if debug:
            print(e)
        stepwise_model = auto_arima(y_array, start_p=1, start_q=1, max_p=5, max_q=5, m=12, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)

    if debug:
        print('============================ Auto ARIMA ===========================')
        print(stepwise_model.summary())

    return stepwise_model

#===============================================================================
# cross_validation_best ()
#===============================================================================
def cross_validation_best(pipes, x_array, y_array):
    """
    Perform TimeSeriesSplit Validation to a list of models and return best based
    on R2 error minimization.
    Arguments:
        pipelines: A list of models integrated into a Pipeline.
        x_array: Independent variable values of observations as a NumPy array.
        y_array: Dependent variable values of observations as a NumPy array.
    Returns:
        The best model integrated into a Pipeline.
    """

    # Chosing regressor based on best score during TimeSeriesSplit Validation
    splits = int(len(x_array) / 30) if len(x_array) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=splits)

    # Scores that will be computed during TimeSeriesSplit Validation
    scorer = {'neg_mean_absolute_error': 'neg_mean_absolute_error', 'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2', 'mean_absolute_percentage_error': make_scorer(mean_absolute_percentage_error, greater_is_better=False), 'root_mean_squared_error': make_scorer(root_mean_squared_error, greater_is_better=False)}

    # Perform TimeSeriesSplit Validation and compute metrics
    best_score = float('-inf')
    best_regressor = None
    for pipe in pipes:
        scores = cross_validate(estimator=pipe, X=x_array, y=y_array.ravel(), scoring=scorer, cv=tscv, return_train_score=False)
        if scores['test_r2'].mean() > best_score:
            best_score = scores['test_r2'].mean()
            best_regressor = pipe

    if debug:
        print('=================== TimeSeriesSplit Validation ====================')
        print(' - Best Regressor: ', best_regressor['regressor'])
        print(' - Best R2 Score: ', best_score)

    return best_regressor

#===============================================================================
# create_regressor ()
#===============================================================================
@ignore_warnings(category=ConvergenceWarning)
def create_regressor(reg_type, x_array, y_array):
    """
    Create and train a regressor based on given X and Y values. Regressor type
    can be provided manually by the user or selected automatically based on R2
    error minimization.
    Arguments:
        reg_type: Type of regressor as a string.
        x_array: Independent variable values of observations as a NumPy array.
        y_array: Dependent variable values of observations as a NumPy array.
    Returns:
        A fitted sklearn model integrated into a Pipeline.
    """

    scaler = StandardScaler()

    # Create the regressor model
    try:
        if reg_type == 'mlr':
            # Fitting Multiple Linear Regression to the Training set
            regressor = LinearRegression()
            pipeline = Pipeline([('regressor', regressor)])
            pipeline.fit(x_array, y_array.ravel())
        elif reg_type == 'lasso':
            # Fitting Lasso Regression to the Training set
            best_parameters = grid_search_best('lasso', x_array, y_array)
            regressor = Lasso(alpha=best_parameters['regressor__alpha'])
            pipeline = Pipeline([('regressor', regressor)])
            pipeline.fit(x_array, y_array.ravel())
        elif reg_type == 'ridge':
            # Fitting Ridge Regression to the Training set
            best_parameters = grid_search_best('ridge', x_array, y_array)
            regressor = Ridge(alpha=best_parameters['regressor__alpha'])
            pipeline = Pipeline([('regressor', regressor)])
            pipeline.fit(x_array, y_array.ravel())
        elif reg_type == 'svr_linear':
            # Fitting linear SVR to the dataset
            best_parameters = grid_search_best('svr_linear', x_array, y_array)
            regressor = SVR(kernel='linear', C=best_parameters['regressor__C'])
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
            pipeline.fit(x_array, y_array.ravel())
        elif reg_type == 'svr_rbf':
            # Fitting SVR to the dataset
            best_parameters = grid_search_best('svr_rbf', x_array, y_array)
            regressor = SVR(kernel='rbf', gamma=best_parameters['regressor__gamma'], C=best_parameters['regressor__C'])
            pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
            pipeline.fit(x_array, y_array.ravel())
        elif reg_type == 'random_forest':
            # Fitting Random Forest Regression to the dataset
            best_parameters = grid_search_best('random_forest', x_array, y_array)
            regressor = RandomForestRegressor(n_estimators=best_parameters['regressor__n_estimators'], max_depth=best_parameters['regressor__max_depth'], random_state=0)
            pipeline = Pipeline([('regressor', regressor)])
            pipeline.fit(x_array, y_array.ravel())
        elif reg_type == 'arima':
            # Fitting ARIMA Regression to the dataset
            regressor = arima_search_best(y_array)
            pipeline = regressor
            pipeline.fit(y_array)
        elif reg_type == 'auto':
            # Fitting Multiple Linear Regression to the Training set
            regressor_linear = LinearRegression()
            # Fitting Lasso Regression to the Training set
            best_parameters = grid_search_best('lasso', x_array, y_array)
            regressor_lasso = Lasso(alpha=best_parameters['regressor__alpha'])
            # Fitting Ridge Regression to the Training set
            best_parameters = grid_search_best('ridge', x_array, y_array)
            regressor_ridge = Ridge(alpha=best_parameters['regressor__alpha'])
            # Fitting linear SVR to the dataset
            best_parameters = grid_search_best('svr_linear', x_array, y_array)
            regressor_svr_linear = SVR(kernel='linear', C=best_parameters['regressor__C'])
            # Fitting SVR to the dataset
            best_parameters = grid_search_best('svr_rbf', x_array, y_array)
            regressor_svr_rbf = SVR(kernel='rbf', gamma=best_parameters['regressor__gamma'], C=best_parameters['regressor__C'])
            # Fitting Random Forest Regression to the dataset
            best_parameters = grid_search_best('random_forest', x_array, y_array)
            regressor_random_forest = RandomForestRegressor(n_estimators=best_parameters['regressor__n_estimators'], max_depth=best_parameters['regressor__max_depth'], random_state=0)
            # Perform TimeSeriesSplit Validation and return best model
            pipes = [Pipeline([('regressor', regressor_linear)]), Pipeline([('regressor', regressor_lasso)]), Pipeline([('regressor', regressor_ridge)]), Pipeline([('scaler', scaler), ('regressor', regressor_svr_linear)]), Pipeline([('scaler', scaler), ('regressor', regressor_svr_rbf)]), Pipeline([('regressor', regressor_random_forest)])]
            pipeline = cross_validation_best(pipes, x_array, y_array)
            pipeline.fit(x_array, y_array.ravel())
        return pipeline
    except ValueError as e:
        if debug:
            print(e)
        return -1

#===============================================================================
# build_and_train_td ()
#===============================================================================
def build_and_train_td(horizon_param, project_param, regressor_param, ground_truth_param, test_param):
    """
    Build TD forecasting models and return forecasts for an horizon specified by the user.
    Arguments:
        horizon_param: The forecasting horizon up to which forecasts will be produced.
        project_param: The project for which the forecasts will be produced.
        regressor_param: The regressor models that will be used to produce forecasts.
        ground_truth_param: If the model will return also ground truth values or not.
        test_param: If the model will produce Train-Test or unseen forecasts
    Returns:
        A dictionary containing forecasted values (and ground thruth values if
        ground_truth_param is set to yes) for each intermediate step ahead up
        to the specified horizon.
    """

    # selecting indicators that will be used as model variables
    metrics_td = ['bugs', 'vulnerabilities', 'code_smells', 'sqale_index', 'reliability_remediation_effort', 'security_remediation_effort']
    # Select sliding window length
    window_size = 2

    # Read dataset
    if project_param == 'HolisunArassistance' or project_param == 'Neurasmus' or project_param == 'Airbus':
        dataset_td = read_from_td_toolbox_api(project_param)
    else:
        dataset_td = pd.read_csv('data/%s.csv' % project_param, sep=";", usecols=metrics_td)
        # dataset = read_from_database('td_dummy', 'localhost', 27017, project_param, {'_id': 0, 'bugs': 1, 'vulnerabilities': 1, 'code_smells': 1, 'sqale_index': 1, 'reliability_remediation_effort': 1, 'security_remediation_effort': 1})
        dataset_td['total_principal'] = dataset_td['reliability_remediation_effort'] + dataset_td['security_remediation_effort'] + dataset_td['sqale_index']
        dataset_td = dataset_td.drop(columns=['sqale_index', 'reliability_remediation_effort', 'security_remediation_effort'])

    # Initialise variables
    dict_result = {
        'parameters': {
            'project': project_param,
            'horizon': horizon_param,
            'regressor': regressor_param,
            'ground_truth': ground_truth_param,
            'test': test_param
        }
    }
    list_forecasts = []
    list_ground_truth = []

    # Make forecasts using the ARIMA model
    if regressor_param == 'arima':
        # Test model
        if test_param == 'yes':
            # Split data to training/test set to test model
            y_array = dataset_td['total_principal'][0:-horizon_param]
        # Deploy model
        else:
            # Set Y to to deploy model for real forecasts
            y_array = dataset_td['total_principal']

        # Make forecasts for training/test set
        regressor = create_regressor(regressor_param, None, y_array)
        if regressor is -1:
            return -1
        y_pred = regressor.predict(n_periods=horizon_param)

        # Fill dataframe with forecasts
        for intermediate_horizon in range(1, horizon_param+1):
            version_counter = len(y_array)+intermediate_horizon
            temp_dict = {
                'version': version_counter,
                'value': float(y_pred[intermediate_horizon-1])
            }
            list_forecasts.append(temp_dict)

    # Make forecasts using the Direct approach, i.e. train separate ML models for each forecasting horizon
    else:
        for intermediate_horizon in range(1, horizon_param+1):
            if debug:
                print('=========================== Horizon: %s ============================' % intermediate_horizon)

            # Add time-shifted prior and future period
            data = series_to_supervised(dataset_td, n_in=window_size)

            # Append dependend variable column with value equal to total_principal of the target horizon's version
            data['forecasted_total_principal'] = data['total_principal(t)'].shift(-intermediate_horizon)
            data = data.drop(data.index[-intermediate_horizon:])

            # Remove TD as independent variable
            data = data.drop(columns=['total_principal(t-%s)' % (i) for i in range(window_size, 0, -1)]) 

            # Define independent and dependent variables
            x_array = data.iloc[:, data.columns != 'forecasted_total_principal'].values
            y_array = data.iloc[:, data.columns == 'forecasted_total_principal'].values

            # Test model
            if test_param == 'yes':
                # Assign version counter
                version_counter = len(dataset_td)-(horizon_param-intermediate_horizon)
                # Split data to training/test set to test model
                x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=horizon_param, random_state=0, shuffle=False)
                # Make forecasts for training/test set
                regressor = create_regressor(regressor_param, x_train, y_train)
                if regressor is -1:
                    return -1
                y_pred = regressor.predict(x_test)
            # Deploy model
            else:
                # Assign version counter
                version_counter = len(dataset_td)+intermediate_horizon
                # Define X to to deploy model for real forecasts
                x_real = series_to_supervised(dataset_td, n_in=window_size, dropnan=False)
                x_real = x_real.drop(columns=['total_principal(t-%s)' % (i) for i in range(window_size, 0, -1)])
                x_real = x_real.iloc[-1, :].values
                x_real = x_real.reshape(1, -1)
                # Make real forecasts
                regressor = create_regressor(regressor_param, x_array, y_array)
                if regressor is -1:
                    return -1
                y_pred = regressor.predict(x_real)

            # Fill dataframe with forecasts
            temp_dict = {
                'version': version_counter,
                'value': float(y_pred[0])
            }
            list_forecasts.append(temp_dict)

    # Fill results dictionary with forecasts
    dict_result['forecasts'] = list_forecasts

    # If the model will return also ground truth values
    if ground_truth_param == 'yes':
        # Fill dataframe with ground thruth
        for intermediate_horizon in range(0, len(dataset_td['total_principal'])):
            temp_dict = {
                'version': intermediate_horizon + 1,
                'value': float(dataset_td['total_principal'][intermediate_horizon])
            }
            list_ground_truth.append(temp_dict)
        # Fill results dictionary with ground thruth
        dict_result['ground_truth'] = list_ground_truth

    if debug:
        print(dict_result)

    return dict_result

#===============================================================================
# build_and_train_dependability ()
#===============================================================================
def build_and_train_dependability(horizon_param, project_param, regressor_param, ground_truth_param, test_param):
    """
    Build Dependability forecasting models and return forecasts for an horizon specified by the user.
    Arguments:
        horizon_param: The forecasting horizon up to which forecasts will be produced.
        project_param: The project for which the forecasts will be produced.
        regressor_param: The regressor models that will be used to produce forecasts.
        ground_truth_param: If the model will return also ground truth values or not.
        test_param: If the model will produce Train-Test or unseen forecasts
    Returns:
        A dictionary containing forecasted values (and ground thruth values if
        ground_truth_param is set to yes) for each intermediate step ahead up
        to the specified horizon.
    """

    # selecting indicators that will be used as model variables
    metrics_dependability = ['Resource_Handling', 'Assignment', 'Exception_Handling', 'Misused_Functionality', 'Security_Index']
    # Select sliding window length
    window_size = 2

    # Read dataset
    if project_param == 'HolisunArassistance' or project_param == 'Neurasmus' or project_param == 'Airbus':
        dataset_dep = read_from_dependability_toolbox_api(project_param)
    else:
        dataset_dep = pd.read_csv('data/%s.csv' % project_param, sep=";", usecols=metrics_dependability)
        # dataset = read_from_database('dependability_dummy', 'localhost', 27017, project_param, {'_id': 0, 'Resource_Handling': 1, 'Assignment': 1, 'Exception_Handling': 1, 'Misused_Functionality': 1, 'Security_Index': 1})

    # Initialise variables
    dict_result = {
        'parameters': {
            'project': project_param,
            'horizon': horizon_param,
            'regressor': regressor_param,
            'ground_truth': ground_truth_param,
            'test': test_param
        }
    }
    list_forecasts = []
    list_ground_truth = []

    # Make forecasts using the ARIMA model
    if regressor_param == 'arima':
        # Test model
        if test_param == 'yes':
            # Split data to training/test set to test model
            y_array = dataset_dep['Security_Index'][0:-horizon_param]
        # Deploy model
        else:
            # Set Y to to deploy model for real forecasts
            y_array = dataset_dep['Security_Index']

        # Make forecasts for training/test set
        regressor = create_regressor(regressor_param, None, y_array)
        if regressor is -1:
            return -1
        y_pred = regressor.predict(n_periods=horizon_param)

        # Fill dataframe with forecasts
        for intermediate_horizon in range(1, horizon_param+1):
            version_counter = len(y_array)+intermediate_horizon
            temp_dict = {
                'version': version_counter,
                'value': float(y_pred[intermediate_horizon-1])
            }
            list_forecasts.append(temp_dict)

    # Make forecasts using the Direct approach, i.e. train separate ML models for each forecasting horizon
    else:
        for intermediate_horizon in range(1, horizon_param+1):
            if debug:
                print('=========================== Horizon: %s ============================' % intermediate_horizon)

            # Add time-shifted prior and future period
            data = series_to_supervised(dataset_dep, n_in=window_size)

            # Append dependend variable column with value equal to Security_Index of the target horizon's version
            data['forecasted_Security_Index'] = data['Security_Index(t)'].shift(-intermediate_horizon)
            data = data.drop(data.index[-intermediate_horizon:])

            # Remove Security_Index as independent variable
            data = data.drop(columns=['Security_Index(t-%s)' % (i) for i in range(window_size, 0, -1)]) 

            # Define independent and dependent variables
            x_array = data.iloc[:, data.columns != 'forecasted_Security_Index'].values
            y_array = data.iloc[:, data.columns == 'forecasted_Security_Index'].values

            # Test model
            if test_param == 'yes':
                # Assign version counter
                version_counter = len(dataset_dep)-(horizon_param-intermediate_horizon)
                # Split data to training/test set to test model
                x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=horizon_param, random_state=0, shuffle=False)
                # Make forecasts for training/test set
                regressor = create_regressor(regressor_param, x_train, y_train)
                if regressor is -1:
                    return -1
                y_pred = regressor.predict(x_test)
            # Deploy model
            else:
                # Assign version counter
                version_counter = len(dataset_dep)+intermediate_horizon
                # Define X to to deploy model for real forecasts
                x_real = series_to_supervised(dataset_dep, n_in=window_size, dropnan=False)
                x_real = x_real.drop(columns=['Security_Index(t-%s)' % (i) for i in range(window_size, 0, -1)])
                x_real = x_real.iloc[-1, :].values
                x_real = x_real.reshape(1, -1)
                # Make real forecasts
                regressor = create_regressor(regressor_param, x_array, y_array)
                if regressor is -1:
                    return -1
                y_pred = regressor.predict(x_real)

            # Fill dataframe with forecasts
            temp_dict = {
                'version': version_counter,
                'value': float(y_pred[0])
            }
            list_forecasts.append(temp_dict)

    # Fill results dictionary with forecasts
    dict_result['forecasts'] = list_forecasts

    # If the model will return also ground truth values
    if ground_truth_param == 'yes':
        # Fill dataframe with ground thruth
        for intermediate_horizon in range(0, len(dataset_dep['Security_Index'])):
            temp_dict = {
                'version': intermediate_horizon + 1,
                'value': float(dataset_dep['Security_Index'][intermediate_horizon])
            }
            list_ground_truth.append(temp_dict)
        # Fill results dictionary with ground thruth
        dict_result['ground_truth'] = list_ground_truth

    if debug:
        print(dict_result)

    return dict_result

#===============================================================================
# build_and_train_energy ()
#===============================================================================
def build_and_train_energy(horizon_param, project_param, regressor_param, ground_truth_param, test_param):
    """
    Build Energy forecasting models and return forecasts for an horizon specified by the user.
    Arguments:
        horizon_param: The forecasting horizon up to which forecasts will be produced.
        project_param: The project for which the forecasts will be produced.
        regressor_param: The regressor models that will be used to produce forecasts.
        ground_truth_param: If the model will return also ground truth values or not.
        test_param: If the model will produce Train-Test or unseen forecasts
    Returns:
        A dictionary containing forecasted values (and ground thruth values if
        ground_truth_param is set to yes) for each intermediate step ahead up
        to the specified horizon.
    """

    # selecting indicators that will be used as model variables
    metrics_energy = ['cpu_cycles', 'cache_references', 'energy_CPU(J)']
    # Select sliding window length
    window_size = 2

    # Read dataset
    dataset_en = pd.read_csv('data/%s.csv' % project_param, sep=";", usecols=metrics_energy)
    # dataset = read_from_database('energy_dummy', 'localhost', 27017, project_param, {'_id': 0, 'cpu_cycles': 1, 'cache_references': 1, 'energy_CPU(J)': 1})

    # Initialise variables
    dict_result = {
        'parameters': {
            'project': project_param,
            'horizon': horizon_param,
            'regressor': regressor_param,
            'ground_truth': ground_truth_param,
            'test': test_param
        }
    }
    list_forecasts = []
    list_ground_truth = []

    # Make forecasts using the ARIMA model
    if regressor_param == 'arima':
        # Test model
        if test_param == 'yes':
            # Split data to training/test set to test model
            y_array = dataset_en['energy_CPU(J)'][0:-horizon_param]
        # Deploy model
        else:
            # Set Y to to deploy model for real forecasts
            y_array = dataset_en['energy_CPU(J)']

        # Make forecasts for training/test set
        regressor = create_regressor(regressor_param, None, y_array)
        if regressor is -1:
            return -1
        y_pred = regressor.predict(n_periods=horizon_param)

        # Fill dataframe with forecasts
        for intermediate_horizon in range(1, horizon_param+1):
            version_counter = len(y_array)+intermediate_horizon
            temp_dict = {
                'version': version_counter,
                'value': float(y_pred[intermediate_horizon-1])
            }
            list_forecasts.append(temp_dict)

    # Make forecasts using the Direct approach, i.e. train separate ML models for each forecasting horizon
    else:
        for intermediate_horizon in range(1, horizon_param+1):
            if debug:
                print('=========================== Horizon: %s ============================' % intermediate_horizon)

            # Add time-shifted prior and future period
            data = series_to_supervised(dataset_en, n_in=window_size)

            # Append dependend variable column with value equal to energy_CPU(J) of the target horizon's version
            data['forecasted_energy_CPU(J)'] = data['energy_CPU(J)(t)'].shift(-intermediate_horizon)
            data = data.drop(data.index[-intermediate_horizon:])

            # Remove energy_CPU(J) as independent variable
            data = data.drop(columns=['energy_CPU(J)(t-%s)' % (i) for i in range(window_size, 0, -1)])

            # Define independent and dependent variables
            x_array = data.iloc[:, data.columns != 'forecasted_energy_CPU(J)'].values
            y_array = data.iloc[:, data.columns == 'forecasted_energy_CPU(J)'].values

            # Test model
            if test_param == 'yes':
                # Assign version counter
                version_counter = len(dataset_en)-(horizon_param-intermediate_horizon)
                # Split data to training/test set to test model
                x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=horizon_param, random_state=0, shuffle=False)
                # Make forecasts for training/test set
                regressor = create_regressor(regressor_param, x_train, y_train)
                if regressor is -1:
                    return -1
                y_pred = regressor.predict(x_test)
            # Deploy model
            else:
                # Assign version counter
                version_counter = len(dataset_en)+intermediate_horizon
                # Define X to to deploy model for real forecasts
                x_real = series_to_supervised(dataset_en, n_in=window_size, dropnan=False)
                x_real = x_real.drop(columns=['energy_CPU(J)(t-%s)' % (i) for i in range(window_size, 0, -1)])
                x_real = x_real.iloc[-1, :].values
                x_real = x_real.reshape(1, -1)
                # Make real forecasts
                regressor = create_regressor(regressor_param, x_array, y_array)
                if regressor is -1:
                    return -1
                y_pred = regressor.predict(x_real)

            # Fill dataframe with forecasts
            temp_dict = {
                'version': version_counter,
                'value': float(y_pred[0])
            }
            list_forecasts.append(temp_dict)

    # Fill results dictionary with forecasts
    dict_result['forecasts'] = list_forecasts

    # If the model will return also ground truth values
    if ground_truth_param == 'yes':
        # Fill dataframe with ground thruth
        for intermediate_horizon in range(0, len(dataset_en['energy_CPU(J)'])):
            temp_dict = {
                'version': intermediate_horizon + 1,
                'value': float(dataset_en['energy_CPU(J)'][intermediate_horizon])
            }
            list_ground_truth.append(temp_dict)
        # Fill results dictionary with ground thruth
        dict_result['ground_truth'] = list_ground_truth

    if debug:
        print(dict_result)

    return dict_result
