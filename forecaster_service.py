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
    model  (sent as URL query parameter) from API Call
    """
    horizon = request.args.get("horizon") # if key doesn't exist, returns None
    model = request.args.get("model") # if key doesn't exist, returns None
    
    if horizon is None:
        return(bad_request())
    else:
        results = build_and_train(int(horizon))
        
        message = {
                'status': 200,
                'message': 'The selected horizon is {}!'.format(horizon),
                'forecast': json.loads(results),
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

def build_and_train(horizon):
    """
    Build forecasting models and return forecasts for an horizon specified by the user.
    Arguments:
        horizon: The forecasting horizon up to which forecasts will be produced.
    Returns:
        A JSON object containing forecasted values for each intermediate step ahead up to the specified horizon.
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
    
    df_forecast = pd.DataFrame()

    # Make forecasts using the Direct approach, i.e. train separate models for each forecasting horizon
    for intermediate_horizon in range (1, horizon+1):
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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = horizon, random_state = 0, shuffle = False)
        
        # Fit Random Forest Regression to the dataset
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
        regressor.fit(X_train, Y_train.ravel())
        
        # Make predictions
        y_pred = regressor.predict(X_test)
    
        # Fill dataframe with forecasts
        temp_dict = {
                        'x': dataset_date['date'].iloc[len(dataset_date['date'])-(horizon-intermediate_horizon+1)],
                        'y': y_pred[0]
                    }
        df_forecast = df_forecast.append(temp_dict, ignore_index = True)
        
    # Convert forecasts dataframe to JSON
    json_df_forecast = df_forecast.to_json(orient = 'records')
    
#    print(print_df_forecast)
    print(df_forecast)
    return(json_df_forecast)
    
build_and_train(5)