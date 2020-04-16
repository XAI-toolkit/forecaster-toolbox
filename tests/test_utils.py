# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import sys
sys.path.append('..')
from utils import mean_absolute_percentage_error, root_mean_squared_error, series_to_supervised, objectid_to_string

def test_mean_absolute_percentage_error(x_vector_input):
    assert mean_absolute_percentage_error(x_vector_input, x_vector_input) == 0

def test_root_mean_squared_error(x_vector_input):
    assert root_mean_squared_error(x_vector_input, x_vector_input) == 0

def test_series_to_supervised(x_dataframe_input, x_dataframe_output):
    assert series_to_supervised(x_dataframe_input, n_in=1, n_out=1, dropnan=False).equals(x_dataframe_output)

def test_objectid_to_string(dict_oid_input):
    dict_str = objectid_to_string(dict_oid_input)
    for key in dict_str:
        assert isinstance(dict_str[key], str)
