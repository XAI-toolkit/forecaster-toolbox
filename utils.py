# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import numpy as np
import pandas as pd
import os
import pymongo
from math import sqrt
from sklearn.metrics import mean_squared_error
from bson import ObjectId

DEBUG = bool(os.environ.get('DEBUG', False))

#===============================================================================
# mean_absolute_percentage_error ()
#===============================================================================
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate mean absolute percentage error (MAPE) between 2 lists of 
    observations.
    Arguments:
        y_true: Real value of observations as a list or NumPy array.
        y_pred: Forecasted value of observations as a list or NumPy array.
    Returns:
        A value indicating the MAPE as percentage.
    """
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#===============================================================================
# root_mean_squared_error ()
#===============================================================================
def root_mean_squared_error(y_true, y_pred):
    """
    Calculate root mean squared error (RMSE) between 2 lists of observations.
    Arguments:
        y_true: Real value of observations as a list or NumPy array.
        y_pred: Forecasted value of observations as a list or NumPy array.
    Returns:
        A value indicating the RMSE.
    """
    
    return sqrt(mean_squared_error(y_true, y_pred))

#===============================================================================
# series_to_supervised ()
#===============================================================================
def series_to_supervised(dataset, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        dataset: Sequence of observations as a list or NumPy array.
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

#===============================================================================
# objectid_to_string ()
#===============================================================================
def objectid_to_string(dict_obj):
    """
    Convert all fields of dict that are ObjectIds to str.
    Arguments:
        dict_obj: A dict object.
    Returns:
        A dict object with Pymongo ObjectId as string.
    """
    
    for key in dict_obj:
        if isinstance(dict_obj[key], ObjectId):
            dict_obj[key] = str(dict_obj[key])
    
    return dict_obj

#===============================================================================
# import_to_database ()
#===============================================================================
def import_to_database(dict_obj, collection_name):
    """
    Insert a dict object into a Mongo database collection.
    Arguments:
        dict_obj: The dict object to be inserted into the database.
        collection_name: The name of the collection of the Mongo database.
    Returns:
        A field insertedId with the _id value of the inserted document.
    """
    
    # Read settings from environment variables
    MONGO_HOST = os.environ.get('MONGO_HOST', 'localhost')
    MONGO_PORT = int(os.environ.get('MONGO_PORT', 27017))
    DB_NAME = os.environ.get('MONGO_DBNAME', 'forecasting_toolbox')
    
    client = pymongo.MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    forecasts_collection = db[collection_name]
    
    try:
        result = forecasts_collection.insert_one(dict_obj)
    except Exception as e:
        result = e
    if DEBUG: print(result)
    
    return result