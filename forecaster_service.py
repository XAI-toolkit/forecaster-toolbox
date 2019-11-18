# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

from flask import Flask, jsonify, request
import json

# Create the Flask app
app = Flask(__name__)

# Routes
@app.route('/ForecasterToolbox/TDForecasting', methods=['GET'])
def TDForecasting(horizon=None):
    """API Call

    horizon (sent as URL query parameter) from API Call
    regressor (sent as URL query parameter) from API Call
    """
    horizon_param = request.args.get("horizon") # if key doesn't exist, returns None
    regressor_param = request.args.get("regressor") # if key doesn't exist, returns None
    
    if horizon_param is None:
        return(bad_request())
    else:
        if regressor_param is None: regressor_param = 'auto'
        
        results = build_and_train(int(horizon_param), regressor_param)
        
        message = {
                'status': 200,
                'message': 'The selected horizon is {}!'.format(horizon_param),
                'forecast': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200
        
        return(resp)

# Run app in debug mode on port 5000
if __name__ == '__main__':
    app.run(port = 5000, debug=True)

# Error Handling
@app.errorhandler(400)
def bad_request(error=None):
	message = {
            'status': 400,
            'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return(resp)








import numpy as np
import pandas as pd
    
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate mean absolute percentage error (MAPE) between 2 lists of observations.
    Arguments:
        y_true: Real value of observations as a list or NumPy array.
        y_pred: Forecasted value of observations as a list or NumPy array.
    Returns:
        A value indicating the MAPE as percentage.
    """
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def series_to_supervised(dataset, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    
    data = dataset.values
    labels = dataset.columns.tolist()
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (labels[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
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
    return agg

def create_regressor(reg_type, X, Y):   
    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
        
    # Create the regressor model
    if reg_type == 'linear_regression':
        # Fitting Multiple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        pipeline = Pipeline([('regressor', regressor)])
    elif reg_type == 'lasso_regression':
        # Fitting Lasso Regression to the Training set
        from sklearn.linear_model import Lasso
        regressor = Lasso(alpha = 100000)
        pipeline = Pipeline([('regressor', regressor)])
    elif reg_type == 'ridge_regression':
        # Fitting Ridge Regression to the Training set
        from sklearn.linear_model import Ridge
        regressor = Ridge(alpha = 1000000)
        pipeline = Pipeline([('regressor', regressor)])
    elif reg_type == 'svr_linear':
        # Fitting linear SVR to the dataset
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'linear', C = 10000)
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
    elif reg_type == 'svr_rbf':
        # Fitting SVR to the dataset
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf', gamma = 0.01, C = 10000)
        pipeline = Pipeline([('scaler', scaler), ('regressor', regressor)])
    elif reg_type == 'random_forest':
        # Fitting Random Forest Regression to the dataset
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
        pipeline = Pipeline([('regressor', regressor)])
    elif reg_type == 'auto':
        # Chosing regressor based on best score
        from sklearn.linear_model import LinearRegression, Lasso, Ridge
        from sklearn.svm import SVR
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_validate
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics.scorer import make_scorer
        from math import sqrt
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        def mean_absolute_percentage_error(y_true, y_pred): 
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
        def root_mean_squared_error(y_true, y_pred): 
            return sqrt(mean_squared_error(y_true, y_pred))
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Applying TimeSeriesSplit Validation
        scorer = {'neg_mean_absolute_error': 'neg_mean_absolute_error', 'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2', 'mean_absolute_percentage_error': make_scorer(mean_absolute_percentage_error, greater_is_better=False), 'root_mean_squared_error': make_scorer(root_mean_squared_error, greater_is_better=False)}
        
        scores = cross_validate(estimator = LinearRegression(), X = X, y = Y.ravel(), scoring = scorer, cv = tscv, return_train_score = False)
        best_score = scores['test_r2'].mean()
        best_regressor = LinearRegression()
        print(scores['test_r2'].mean())
        
        scores = cross_validate(estimator = Lasso(alpha = 100000), X = X, y = Y.ravel(), scoring = scorer, cv = tscv, return_train_score = False)
        if scores['test_r2'].mean() > best_score:
            best_score = scores['test_r2'].mean()
            best_regressor = Lasso(alpha = 100000)
        print(scores['test_r2'].mean())
        
        scores = cross_validate(estimator = Ridge(alpha = 1000000), X = X, y = Y.ravel(), scoring = scorer, cv = tscv, return_train_score = False)
        if scores['test_r2'].mean() > best_score:
            best_score = scores['test_r2'].mean()
            best_regressor = Ridge(alpha = 1000000)
        print(scores['test_r2'].mean())
        
        scores = cross_validate(estimator = RandomForestRegressor(n_estimators = 100, random_state = 0), X = X, y = Y.ravel(), scoring = scorer, cv = tscv, return_train_score = False)
        if scores['test_r2'].mean() > best_score:
            best_score = scores['test_r2'].mean()
            best_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
        print(scores['test_r2'].mean())
        
#        regressor = 
#        regressor = Ridge(alpha = 1000000)
#        regressor = SVR(kernel = 'linear', C = 10000)
#        regressor = SVR(kernel = 'rbf', gamma = 0.01, C = 10000)
#        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
        
        
        regressor = best_regressor
        
        pipeline = Pipeline([('regressor', regressor)])
        
    pipeline.fit(X, Y.ravel())
    
    return pipeline

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
        
        # Split data to training/test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = horizon_param, random_state = 0, shuffle = False)
        
        # Fit Random Forest Regression to the dataset
#        from sklearn.ensemble import RandomForestRegressor
#        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
#        regressor.fit(X_train, Y_train.ravel())
        
        # Make predictions
        regressor = create_regressor(regressor_param, X_train, Y_train)
        y_pred = regressor.predict(X_test)
    
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
    
build_and_train(5, 'auto')