# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import requests

def test_td_forecasting_status_code():
    url = 'http://localhost:5000/ForecasterToolbox/TDForecasting?horizon=5&project=apache_kafka_measures&regressor=auto&ground_truth=no&test=no'

    r = requests.get(url)

    assert r.status_code == 200

def test_energy_forecasting_status_code():
    url = 'http://localhost:5000/ForecasterToolbox/EnergyForecasting?horizon=5&project=sbeamer_gapbs_energy_measures&regressor=auto&ground_truth=no&test=no'

    r = requests.get(url)

    assert r.status_code == 200

def test_dependability_forecasting_status_code():
    url = 'http://localhost:5000/ForecasterToolbox/DependabilityForecasting?horizon=5&project=square_retrofit_security_measures&regressor=auto&ground_truth=no&test=no'

    r = requests.get(url)

    assert r.status_code == 200
