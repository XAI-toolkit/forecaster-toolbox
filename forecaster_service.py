# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import argparse
import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve
from model_training import build_and_train_td, build_and_train_dependability, build_and_train_energy, build_and_train_td_class_level
from utils import import_to_database, objectid_to_string

# Create the Flask app
app = Flask(__name__)
# Enable CORS
CORS(app)

#===============================================================================
# td_forecasting ()
#===============================================================================
@app.route('/ForecasterToolbox/TDForecasting', methods=['GET'])
def td_forecasting(horizon_param_td=None, project_param_td=None, regressor_param_td=None, ground_truth_param_td=None, test_param_td=None):
    """
    API Call to TDForecasting service
    Arguments:
        horizon_param_td: Required (sent as URL query parameter from API Call)
        project_param_td: Required (sent as URL query parameter from API Call)
        regressor_param_td: Optional (sent as URL query parameter from API Call)
        ground_truth_param_td: Optional (sent as URL query parameter from API Call)
        test_param_td: Optional (sent as URL query parameter from API Call)
    Returns:
        A JSON containing the forecasting results, status code and a message.
    """

    # Parse URL-encoded parameters
    horizon_param_td = request.args.get('horizon', type=int) # Required: if key doesn't exist, returns None
    project_param_td = request.args.get('project', type=str) # Required: if key doesn't exist, returns None
    regressor_param_td = request.args.get('regressor', default='auto', type=str) # Optional: if key doesn't exist, returns auto
    ground_truth_param_td = request.args.get('ground_truth', default='no', type=str) # Optional: if key doesn't exist, returns no
    test_param_td = request.args.get('test', default='no', type=str) # Optional: if key doesn't exist, returns no

    # If required parameters are missing from URL
    if horizon_param_td is None or project_param_td is None or regressor_param_td is None or ground_truth_param_td is None or test_param_td is None:
        return unprocessable_entity()
    else:
        # Call build_and_train() function and retrieve forecasts
        results = build_and_train_td(horizon_param_td, project_param_td, regressor_param_td, ground_truth_param_td, test_param_td)
        # Handle errors
        if results is -1:
            return internal_server_error('%s steps-ahead forecasting cannot provide reliable results for this project. Please ensure that a sufficient number of commits is already analysed by the TD Management Toolbox and reduce forecasting horizon.' % horizon_param_td)
        if results is -2:
            return internal_server_error('Cannot provide forecasts for the selected %s project. Please ensure that a sufficient number of commits is already analysed by the TD Management Toolbox.' % project_param_td)

        # Add to database
        import_to_database(results, 'td_forecasts')
        results = objectid_to_string(results)

        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# td_class_level_forecasting ()
#===============================================================================
@app.route('/ForecasterToolbox/TDClassLevelForecasting', methods=['GET'])
def td_class_level_forecasting(horizon_param_td=None, project_param_td=None, project_classes_param_td=None, regressor_param_td=None, ground_truth_param_td=None, test_param_td=None):
    """
    API Call to TDForecasting service
    Arguments:
        horizon_param_td: Required (sent as URL query parameter from API Call)
        project_param_td: Required (sent as URL query parameter from API Call)
        project_classes_param_td: Required (sent as URL query parameter from API Call)
        regressor_param_td: Optional (sent as URL query parameter from API Call)
        ground_truth_param_td: Optional (sent as URL query parameter from API Call)
        test_param_td: Optional (sent as URL query parameter from API Call)
    Returns:
        A JSON containing the forecasting results, status code and a message.
    """

    # Parse URL-encoded parameters
    horizon_param_td = request.args.get('horizon', type=int) # Required: if key doesn't exist, returns None
    project_param_td = request.args.get('project', type=str) # Required: if key doesn't exist, returns None
    project_classes_param_td = request.args.get('project_classes', type=int) # Required: if key doesn't exist, returns None
    regressor_param_td = request.args.get('regressor', default='auto', type=str) # Optional: if key doesn't exist, returns auto
    ground_truth_param_td = request.args.get('ground_truth', default='no', type=str) # Optional: if key doesn't exist, returns no
    test_param_td = request.args.get('test', default='no', type=str) # Optional: if key doesn't exist, returns no

    # If required parameters are missing from URL
    if horizon_param_td is None or project_param_td is None or project_classes_param_td is None or regressor_param_td is None or ground_truth_param_td is None or test_param_td is None:
        return unprocessable_entity()
    else:
        # Call build_and_train_td_class_level() function and retrieve forecasts
        results = build_and_train_td_class_level(horizon_param_td, project_param_td, project_classes_param_td, regressor_param_td, ground_truth_param_td, test_param_td)
        # Handle errors
        if results is -1:
            return internal_server_error('%s steps-ahead forecasting cannot provide reliable results for this project. Please reduce forecasting horizon.' % horizon_param_td)
        if results is -2:
            return internal_server_error('Class-level analysis is currently unavailable for this project.')

#        # Add to database
#        import_to_database(results, 'td_forecasts')
#        results = objectid_to_string(results)

        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# dependability_forecasting ()
#===============================================================================
@app.route('/ForecasterToolbox/DependabilityForecasting', methods=['GET'])
def dependability_forecasting(horizon_param_dep=None, project_param_dep=None, regressor_param_dep=None, ground_truth_param_dep=None, test_param_dep=None):
    """
    API Call to DependabilityForecasting service
    Arguments:
        horizon_param_dep: Required (sent as URL query parameter from API Call)
        project_param_dep: Required (sent as URL query parameter from API Call)
        regressor_param_dep: Optional (sent as URL query parameter from API Call)
        ground_truth_param_dep: Optional (sent as URL query parameter from API Call)
        test_param_dep: Optional (sent as URL query parameter from API Call)
    Returns:
        A JSON containing the forecasting results, status code and a message.
    """

    # Parse URL-encoded parameters
    horizon_param_dep = request.args.get('horizon', type=int) # Required: if key doesn't exist, returns None
    project_param_dep = request.args.get('project', type=str) # Required: if key doesn't exist, returns None
    regressor_param_dep = request.args.get('regressor', default='auto', type=str) # Optional: if key doesn't exist, returns auto
    ground_truth_param_dep = request.args.get('ground_truth', default='no', type=str) # Optional: if key doesn't exist, returns no
    test_param_dep = request.args.get('test', default='no', type=str) # Optional: if key doesn't exist, returns no

    # If required parameters are missing from URL
    if horizon_param_dep is None or project_param_dep is None or regressor_param_dep is None or ground_truth_param_dep is None or test_param_dep is None:
        return unprocessable_entity()
    else:
        # Call build_and_train() function and retrieve forecasts
        results = build_and_train_dependability(horizon_param_dep, project_param_dep, regressor_param_dep, ground_truth_param_dep, test_param_dep)
        # Handle errors
        if results is -1:
            return internal_server_error('%s steps-ahead forecasting cannot provide reliable results for this project. Please ensure that a sufficient number of commits is already analysed by the Dependability Toolbox and reduce forecasting horizon.' % horizon_param_dep)
        if results is -2:
            return internal_server_error('Cannot provide forecasts for the selected %s project. Please ensure that a sufficient number of commits is already analysed by the Dependability Toolbox.' % project_param_dep)

        # Add to database
        import_to_database(results, 'dependability_forecasts')
        results = objectid_to_string(results)

        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# energy_forecasting ()
#===============================================================================
@app.route('/ForecasterToolbox/EnergyForecasting', methods=['GET'])
def energy_forecasting(horizon_param_en=None, project_param_en=None, regressor_param_en=None, ground_truth_param_en=None, test_param_en=None):
    """
    API Call to EnergyForecasting service
    Arguments:
        horizon_param_en: Required (sent as URL query parameter from API Call)
        project_param_en: Required (sent as URL query parameter from API Call)
        regressor_param_en: Optional (sent as URL query parameter from API Call)
        ground_truth_param_en: Optional (sent as URL query parameter from API Call)
        test_param_en: Optional (sent as URL query parameter from API Call)
    Returns:
        A JSON containing the forecasting results, status code and a message.
    """

    # Parse URL-encoded parameters
    horizon_param_en = request.args.get('horizon', type=int) # Required: if key doesn't exist, returns None
    project_param_en = request.args.get('project', type=str) # Required: if key doesn't exist, returns None
    regressor_param_en = request.args.get('regressor', default='auto', type=str) # Optional: if key doesn't exist, returns auto
    ground_truth_param_en = request.args.get('ground_truth', default='no', type=str) # Optional: if key doesn't exist, returns no
    test_param_en = request.args.get('test', default='no', type=str) # Optional: if key doesn't exist, returns no

    # If required parameters are missing from URL
    if horizon_param_en is None or project_param_en is None or regressor_param_en is None or ground_truth_param_en is None or test_param_en is None:
        return unprocessable_entity()
    else:
        # Call build_and_train() function and retrieve forecasts
        results = build_and_train_energy(horizon_param_en, project_param_en, regressor_param_en, ground_truth_param_en, test_param_en)
        # Handle errors
        if results is -1:
            return internal_server_error('%s steps-ahead forecasting cannot provide reliable results for this project. Please ensure that a sufficient number of commits is already analysed by the Energy Toolbox and reduce forecasting horizon.' % horizon_param_en)
        if results is -2:
            return internal_server_error('Cannot provide forecasts for the selected %s project. Please ensure that a sufficient number of commits is already analysed by the Energy Toolbox.' % project_param_en)

        # Add to database
        import_to_database(results, 'energy_forecasts')
        results = objectid_to_string(results)

        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# errorhandler ()
#===============================================================================
@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + ' --> Please check your data payload.',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp
@app.errorhandler(422)
def unprocessable_entity(error=None):
    message = {
        'status': 400,
        'message': 'Unprocessable Entity: ' + request.url + ' --> Missing or invalid parameters.',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp
@app.errorhandler(500)
def internal_server_error(error=None):
    message = {
        'status': 500,
        'message': 'The server encountered an internal error and was unable to complete your request. ' + error,
    }
    resp = jsonify(message)
    resp.status_code = 500

    return resp

#===============================================================================
# run_server ()
#===============================================================================
def run_server(host, port, mode, dbhost, dbport, dbname, debug_mode):
    """
    Executes the command to start the server
    Arguments:
        host: retrieved from create_arg_parser() as a string
        port: retrieved from create_arg_parser() as a int
        mode: retrieved from create_arg_parser() as a string
        dbhost: retrieved from create_arg_parser() as a string
        dbport: retrieved from create_arg_parser() as a string
        dbname: retrieved from create_arg_parser() as a string
        debug_mode: retrieved from create_arg_parser() as a bool
    """

    print('server:      %s:%s' % (host, port))
    print('mode:        %s' % (mode))
    print('db server:      %s:%s' % (dbhost, dbport))
    print('db name:      %s' % (dbname))
    print('debug_mode:  %s' % (debug_mode))

    # Store settings in environment variables
    if debug_mode:
        print(" *** Debug enabled! ***")

    os.environ['DEBUG'] = str(debug_mode)
    os.environ['MONGO_HOST'] = dbhost
    os.environ['MONGO_PORT'] = dbport
    os.environ['MONGO_DBNAME'] = dbname

    if mode == 'builtin':
        # Run app in debug mode using flask
        app.run(host, port, debug_mode)
    elif mode == 'waitress':
        # Run app in production mode using waitress
        serve(app, host=host, port=port)
    else:
        print('Server mode "%s" not yet implemented' % mode)
        sys.exit(1)

#===============================================================================
# create_arg_parser ()
#===============================================================================
def create_arg_parser():
    """
    Creates the parser to retrieve arguments from the command line
    Returns:
        A Parser object
    """
    server_modes = ['builtin', 'waitress']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('h', metavar='HOST', help='Server HOST (e.g. "localhost")', type=str)
    parser.add_argument('p', metavar='PORT', help='Server PORT (e.g. "5000")', type=int)
    parser.add_argument('m', metavar='SERVER_MODE', help=", ".join(server_modes), choices=server_modes, type=str)
    parser.add_argument('-dh', metavar='DB_HOST', help='MongoDB HOST (e.g. "localhost")', type=str, default='localhost')
    parser.add_argument('-dp', metavar='DB_PORT', help='MongoDB PORT (e.g. "27017")', type=str, default='27017')
    parser.add_argument('-dn', metavar='DB_DBNAME', help="Database NAME", type=str, default='forecaster_service')
    parser.add_argument('--debug', help="Run builtin server in debug mode", action='store_true', default=False)

    return parser

#===============================================================================
# main ()
#===============================================================================
def main():
    """
    The main() function of the script acting as the entry point
    """
    parser = create_arg_parser()

    # If script run without arguments, print syntax
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse arguments
    args = parser.parse_args()
    host = args.h
    mode = args.m
    port = args.p
    dbhost = args.dh
    dbport = args.dp
    dbname = args.dn
    debug_mode = args.debug

    # Run server with user-given arguments
    run_server(host, port, mode, dbhost, dbport, dbname, debug_mode)

if __name__ == '__main__':
    main()
