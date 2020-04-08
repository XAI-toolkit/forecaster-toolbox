# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import requests

def test_td_forecasting():
    url = 'http://160.40.51.206:5000/ForecasterToolbox/TDForecasting?horizon=5&project=apache_kafka_measures&regressor=auto&ground_truth=no&test=no'

    r = requests.get(url)

    assert r.status_code == 200