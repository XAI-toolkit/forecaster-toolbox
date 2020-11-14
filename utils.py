# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import os
import re
from math import sqrt
import numpy as np
import pandas as pd
import pymongo
import requests
from sklearn.metrics import mean_squared_error
from bson import ObjectId

debug = bool(os.environ.get('DEBUG', False))

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
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    d_f = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(d_f.shift(i))
        names += [('%s(t-%d)' % (labels[j], i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(d_f.shift(-i))
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
    mongo_host = os.environ.get('MONGO_HOST')
    mongo_port = int(os.environ.get('MONGO_PORT'))
    db_name = os.environ.get('MONGO_DBNAME')

    client = pymongo.MongoClient(mongo_host, mongo_port, serverSelectionTimeoutMS=2000)
    db_instance = client[db_name]
    forecasts_collection = db_instance[collection_name]

    try:
        result = forecasts_collection.insert_one(dict_obj)
    except Exception as e:
        result = e
    if debug:
        print(result)

    return result

#===============================================================================
# read_from_database ()
#===============================================================================
def read_from_database(db_name, db_url, db_port, collection_name, fields):
    """
    Read fields from a specific Mongo database collection and convert them to dataframe.
    Arguments:
        db_name: The name of the Mongo database.
        db_url: The URL of the Mongo database.
        db_port: The URL port of the Mongo database.
        collection_name: The name of the collection of the Mongo database.
        fields: The fields of the collection of the Mongo database in a form of
        {'_id': 0, 'field1': 1, 'field2': 1, ...}.
    Returns:
        A dataframe with the fields recovered from the Mongo database.
    """

    # Read settings from parameters
    mongo_host = db_url
    mongo_port = db_port
    db_name = db_name

    client = pymongo.MongoClient(mongo_host, mongo_port, serverSelectionTimeoutMS=2000)
    db_instance = client[db_name]
    collection = db_instance[collection_name]

    try:
        result = pd.DataFrame(list(collection.find({}, fields)))
    except Exception as e:
        result = e
    if debug:
        print(result)

    return result

#===============================================================================
# read_from_td_toolbox_api ()
#===============================================================================
def read_from_td_toolbox_api(project_param):
    """
    Read TD related data from the TD Toolbox API.
    Arguments:
        project_param: The project for which the data will be fetched.
    Returns:
        A dataframe with TD related data recovered from the TD Toolbox API.
    """

    # selecting indicators that will be used as model variables
    metrics_td = ['bugs', 'vulnerabilities', 'code_smells', 'sqale_index', 'reliability_remediation_effort', 'security_remediation_effort']

    # Check wether TD Toolbox API is up and running else return -1
    try:
        # Call TD Toolbox API
        td_toolbox_url = 'http://195.251.210.147:8989/api/sdk4ed/certh/metrics/%s?' % project_param
        response = requests.get(td_toolbox_url)
        # Check wether TD Toolbox API returns status code 200 else return -1
        if response.status_code == 200:
            # Create dataframe with TD related data
            td_data_df = pd.DataFrame.from_dict(response.json())
            # Rename columns
            td_data_df.rename(columns={'sqaleIndex': 'sqale_index', 'reliabilityRemediationEffort': 'reliability_remediation_effort', 'securityRemediationEffort': 'security_remediation_effort', 'codeSmells': 'code_smells'}, inplace=True)
            # Drop columns not present in TD metrics list
            td_data_df = td_data_df[td_data_df.columns.intersection(metrics_td)]
            result = td_data_df
        else:
            if debug:
                print(response.status_code)
            result = -1
    except requests.exceptions.RequestException as e:
        if debug:
            print(e)
        result = -1
    if debug:
        print(result)

    return result

#===============================================================================
# read_from_dependability_toolbox_api ()
#===============================================================================
def read_from_dependability_toolbox_api(project_param):
    """
    Read Dependability related data from the Dependability Toolbox API.
    Arguments:
        project_param: The project for which the data will be fetched.
    Returns:
        A dataframe with Dependability related data recovered from the Dependability Toolbox API.
    """

    # TODO: This function now reads Dependability data directly from Dependability Toolbox DB. Later will be an API
    # selecting indicators that will be used as model variables
    metrics_dependability = ['Resource_Handling', 'Assignment', 'Exception_Handling', 'Misused_Functionality', 'Security_Index']

    # Set Dependability DB parameters
    mongo_host = '160.40.52.130'
    mongo_port = 27017
    db_name = 'dependabilityToolbox'
    collection_name = 'securityAssessment'
    find_query = {'project_name': project_param}
    sort_query = [('commit_timestamp',1)]

    client = pymongo.MongoClient(mongo_host, mongo_port, serverSelectionTimeoutMS=2000)
    db_instance = client[db_name]
    collection = db_instance[collection_name]

    try:
        # Execute search query
        cursor_projects = collection.find(find_query).sort(sort_query)

        # If cursor does not contain results
        if cursor_projects.count() == 0:
            result = -1
        else:
            # Create dataframe with Dependability related data
            dependability_data_df = pd.DataFrame()
            for project in cursor_projects:
                temp_df = pd.DataFrame()
                for prop in project['report']['properties']['properties']:
                    temp_df[prop['name']] = [prop['measure']['normValue']]
                temp_df['Security_Index'] = [project['report']['security_index']['eval']]
                dependability_data_df = pd.concat([dependability_data_df, temp_df])
            # Drop columns not present in dependability metrics list
            dependability_data_df = dependability_data_df[dependability_data_df.columns.intersection(metrics_dependability)]
            # Reset index
            dependability_data_df.reset_index(drop=True, inplace=True)
            result = dependability_data_df
    except Exception as e:
        if debug:
            print(e)
        result = -1
    if debug:
        print(result)

    return result

#===============================================================================
# read_from_td_toolbox_api ()
#===============================================================================
def read_from_energy_toolbox_api(project_param):
    """
    Read Energy related data from the Energy Toolbox API.
    Arguments:
        project_param: The project for which the data will be fetched.
    Returns:
        A dataframe with Energy related data recovered from the Energy Toolbox API.
    """

    # Check wether Energy Toolbox API is up and running else return -1
    try:
        # Call Energy Toolbox API
        energy_toolbox_url = 'http://147.102.37.20:3002/analysis?new=T&user=&token=&url=%s&commit=&type=history' % project_param
        response = requests.get(energy_toolbox_url)
        # Check wether Energy Toolbox API returns status code 200 else return -1
        if response.status_code == 200:
            # Parse output
            parsed_response = response.json()['history_energy']['rows']
            # Make output dictionary key numerical
            parsed_response = {int(k) : v for k, v in parsed_response.items()}
            # Sort output dictionary
            sorted(parsed_response, reverse=False)
            # Use regex to keep only commits in the form 'vXX'
            pattern = '(^v[0-9]$|^v[0-9][0-9]$)'
            # Create dataframe with Energy related data
            energy_data_df = pd.DataFrame()
            for key in sorted(parsed_response.keys(), reverse=False):
                if re.match(pattern, parsed_response[key]['commit']):
                    temp_df = pd.DataFrame()
                    temp_df['cpu_cycles'] = ['0']
                    temp_df['cache_references'] = ['0']
                    temp_df['energy_CPU(J)'] = [parsed_response[key]['mainplatform1']]
                    energy_data_df = pd.concat([energy_data_df, temp_df])
            # Reset index
            energy_data_df.reset_index(drop=True, inplace=True)
            result = energy_data_df
        else:
            if debug:
                print(response.status_code)
            result = -1
    except requests.exceptions.RequestException as e:
        if debug:
            print(e)
        result = -1
    if debug:
        print(result)

    return result
