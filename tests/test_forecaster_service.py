# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import sys
sys.path.append('..')
import os
from flask import Flask, json
from forecaster_service import td_forecasting, dependability_forecasting, energy_forecasting, td_class_level_forecasting

# Create the Flask app
app = Flask(__name__)

# Set environmental variables
os.environ['MONGO_HOST'] = '0'
os.environ['MONGO_PORT'] = '0'
os.environ['MONGO_DBNAME'] = 'foo'

def test_td_forecasting_mlr(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/TDForecasting?horizon=%s&project=%s&regressor=mlr&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = td_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'mlr'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_td_forecasting_lasso(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/TDForecasting?horizon=%s&project=%s&regressor=lasso&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = td_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'lasso'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_td_forecasting_ridge(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/TDForecasting?horizon=%s&project=%s&regressor=ridge&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = td_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'ridge'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_td_forecasting_svr_linear(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/TDForecasting?horizon=%s&project=%s&regressor=svr_linear&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = td_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_linear'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_td_forecasting_svr_rbf(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/TDForecasting?horizon=%s&project=%s&regressor=svr_rbf&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = td_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_rbf'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_td_forecasting_random_forest(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/TDForecasting?horizon=%s&project=%s&regressor=random_forest&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = td_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'random_forest'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_td_forecasting_auto(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/TDForecasting?horizon=%s&project=%s&regressor=auto&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = td_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'auto'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_energy_forecasting_mlr(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/EnergyForecasting?horizon=%s&project=%s&regressor=mlr&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = energy_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'mlr'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_energy_forecasting_lasso(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/EnergyForecasting?horizon=%s&project=%s&regressor=lasso&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = energy_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'lasso'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_energy_forecasting_ridge(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/EnergyForecasting?horizon=%s&project=%s&regressor=ridge&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = energy_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'ridge'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_energy_forecasting_svr_linear(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/EnergyForecasting?horizon=%s&project=%s&regressor=svr_linear&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = energy_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_linear'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_energy_forecasting_svr_rbf(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/EnergyForecasting?horizon=%s&project=%s&regressor=svr_rbf&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = energy_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_rbf'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_energy_forecasting_random_forest(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/EnergyForecasting?horizon=%s&project=%s&regressor=random_forest&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = energy_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'random_forest'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_energy_forecasting_auto(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/EnergyForecasting?horizon=%s&project=%s&regressor=auto&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = energy_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'auto'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_dependability_forecasting_mlr(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=%s&regressor=mlr&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = dependability_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'mlr'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_dependability_forecasting_lasso(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=%s&regressor=lasso&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = dependability_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'lasso'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_dependability_forecasting_ridge(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=%s&regressor=ridge&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = dependability_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'ridge'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_dependability_forecasting_svr_linear(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=%s&regressor=svr_linear&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = dependability_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_linear'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_dependability_forecasting_svr_rbf(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=%s&regressor=svr_rbf&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = dependability_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_rbf'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_dependability_forecasting_random_forest(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=%s&regressor=random_forest&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = dependability_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'random_forest'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

def test_dependability_forecasting_auto(forecasting_project_input, forecasting_horizon_input):
    with app.test_request_context('/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=%s&regressor=auto&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_input)):
        r = dependability_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'auto'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0       

def test_td_forecasting_class_level_mlr(forecasting_project_class_level_input, forecasting_horizon_input, forecasting_class_number_input):
    with app.test_request_context('/ForecasterToolbox/TDClassLevelForecasting?horizon=%s&project=%s&project_classes=%s&regressor=mlr&ground_truth=no&test=no' % (forecasting_horizon_input, forecasting_project_class_level_input, forecasting_class_number_input)):
        r = td_class_level_forecasting()
        data = json.loads(r.data)

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'mlr'
        assert 'change_metrics' in data['results']
        assert 'forecasts' in data['results']
        assert len(data['results']['change_metrics']) != 0
        assert len(data['results']['forecasts']) != 0
