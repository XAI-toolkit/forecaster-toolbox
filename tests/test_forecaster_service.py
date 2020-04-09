# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import requests

class TestTDForecasterAPI():
    """
        A set of tests performing API Calls to the TD Forecaster service, using different
        algorithms. The Forecasting server and Forecasting DB need to be up and running
        in order to run these tests
    """

    def test_td_forecasts_db_insertion(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/TDForecasting?horizon=%s&project=apache_kafka_measures&regressor=mlr&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert '_id' in data['results']

    def test_td_forecasting_mlr(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/TDForecasting?horizon=%s&project=apache_kafka_measures&regressor=mlr&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'mlr'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_td_forecasting_lasso(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/TDForecasting?horizon=%s&project=apache_kafka_measures&regressor=lasso&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'lasso'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_td_forecasting_ridge(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/TDForecasting?horizon=%s&project=apache_kafka_measures&regressor=ridge&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'ridge'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_td_forecasting_svr_linear(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/TDForecasting?horizon=%s&project=apache_kafka_measures&regressor=svr_linear&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_linear'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_td_forecasting_svr_rbf(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/TDForecasting?horizon=%s&project=apache_kafka_measures&regressor=svr_rbf&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_rbf'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_td_forecasting_random_forest(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/TDForecasting?horizon=%s&project=apache_kafka_measures&regressor=random_forest&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'random_forest'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_td_forecasting_arima(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/TDForecasting?horizon=%s&project=apache_kafka_measures&regressor=arima&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'arima'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_td_forecasting_auto(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/TDForecasting?horizon=%s&project=apache_kafka_measures&regressor=auto&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'auto'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

class TestEnergyForecasterAPI():
    """
        A set of tests performing API Calls to the Energy Forecaster service, using different
        algorithms. The Forecasting server and Forecasting DB need to be up and running
        in order to run these tests
    """

    def test_energy_forecasts_db_insertion(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/EnergyForecasting?horizon=%s&project=sbeamer_gapbs_energy_measures&regressor=mlr&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert '_id' in data['results']

    def test_energy_forecasting_mlr(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/EnergyForecasting?horizon=%s&project=sbeamer_gapbs_energy_measures&regressor=mlr&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'mlr'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_energy_forecasting_lasso(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/EnergyForecasting?horizon=%s&project=sbeamer_gapbs_energy_measures&regressor=lasso&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'lasso'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_energy_forecasting_ridge(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/EnergyForecasting?horizon=%s&project=sbeamer_gapbs_energy_measures&regressor=ridge&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'ridge'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_energy_forecasting_svr_linear(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/EnergyForecasting?horizon=%s&project=sbeamer_gapbs_energy_measures&regressor=svr_linear&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_linear'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_energy_forecasting_svr_rbf(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/EnergyForecasting?horizon=%s&project=sbeamer_gapbs_energy_measures&regressor=svr_rbf&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_rbf'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_energy_forecasting_random_forest(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/EnergyForecasting?horizon=%s&project=sbeamer_gapbs_energy_measures&regressor=random_forest&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'random_forest'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_energy_forecasting_arima(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/EnergyForecasting?horizon=%s&project=sbeamer_gapbs_energy_measures&regressor=arima&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'arima'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_energy_forecasting_auto(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/EnergyForecasting?horizon=%s&project=sbeamer_gapbs_energy_measures&regressor=auto&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'auto'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

class TestDependabilityForecasterAPI():
    """
        A set of tests performing API Calls to the Dependability Forecaster service, using different
        algorithms. The Forecasting server and Forecasting DB need to be up and running
        in order to run these tests
    """

    def test_dependability_forecasts_db_insertion(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=square_retrofit_security_measures&regressor=mlr&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert '_id' in data['results']

    def test_dependability_forecasting_mlr(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=square_retrofit_security_measures&regressor=mlr&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'mlr'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_dependability_forecasting_lasso(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=square_retrofit_security_measures&regressor=lasso&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'lasso'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_dependability_forecasting_ridge(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=square_retrofit_security_measures&regressor=ridge&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'ridge'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_dependability_forecasting_svr_linear(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=square_retrofit_security_measures&regressor=svr_linear&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_linear'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_dependability_forecasting_svr_rbf(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=square_retrofit_security_measures&regressor=svr_rbf&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'svr_rbf'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_dependability_forecasting_random_forest(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=square_retrofit_security_measures&regressor=random_forest&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'random_forest'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_dependability_forecasting_arima(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=square_retrofit_security_measures&regressor=arima&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'arima'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0

    def test_dependability_forecasting_auto(self, forecaster_url_input, forecasting_horizon_input):
        url = '%s/ForecasterToolbox/DependabilityForecasting?horizon=%s&project=square_retrofit_security_measures&regressor=auto&ground_truth=no&test=no' % (forecaster_url_input, forecasting_horizon_input)
        r = requests.get(url)
        data = r.json()

        assert r.status_code == 200
        assert data['results']['parameters']['regressor'] == 'auto'
        assert 'forecasts' in data['results']
        assert len(data['results']['forecasts']) != 0
