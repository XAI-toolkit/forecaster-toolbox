# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import argparse
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from model_training import build_and_train
from waitress import serve

# Create the Flask app
app = Flask(__name__)
# Enable CORS
CORS(app)

#===============================================================================
# TDForecasting ()
#===============================================================================
@app.route('/ForecasterToolbox/TDForecasting', methods=['GET'])
def TDForecasting(horizon_param = None, project_param = None, regressor_param = None, ground_truth_param = None, test_param = None):
    """
    API Call to TDForecasting service
    Arguments:
        horizon_param: Required (sent as URL query parameter from API Call)
        project_param: Required (sent as URL query parameter from API Call)
        regressor_param: Optional (sent as URL query parameter from API Call)
        ground_truth_param: Optional (sent as URL query parameter from API Call)
        test_param: Optional (sent as URL query parameter from API Call)
    Returns:
        A JSON containing the forecasting results, status code and a message.
    """
    
    # Parse URL-encoded parameters
    horizon_param = request.args.get('horizon', type = int) # Required: if key doesn't exist, returns None
    project_param = request.args.get('project', type = str) # Required: if key doesn't exist, returns None
    regressor_param = request.args.get('regressor', default = 'auto', type = str) # Optional: if key doesn't exist, returns auto
    ground_truth_param = request.args.get('ground_truth', default = 'no', type = str) # Optional: if key doesn't exist, returns no
    test_param = request.args.get('test', default = 'no', type = str) # Optional: if key doesn't exist, returns no
        
    # If required parameters are missing from URL
    if horizon_param is None or project_param is None or regressor_param is None or ground_truth_param is None or test_param is None:
        return(unprocessable_entity())
    else:
        # Call build_and_train() function and retrieve forecasts
        results = build_and_train(horizon_param, project_param, regressor_param, ground_truth_param, test_param)
        
        # Compose and jsonify respond
        message = {
                'status': 200,
                'message': 'The selected horizon is {}!'.format(horizon_param),
                'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200
        
        return(resp)
        
#===============================================================================
# errorhandler ()
#===============================================================================
@app.errorhandler(400)
def bad_request(error=None):
	message = {
            'status': 400,
            'message': 'Bad Request: ' + request.url + ' --> Please check your data payload',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return(resp)
@app.errorhandler(422)
def unprocessable_entity(error=None):
	message = {
            'status': 400,
            'message': 'Unprocessable Entity: ' + request.url + ' --> Missing or invalid parameters. Required: horizon, project. Optional: regressor, ground_truth, test',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return(resp)

#===============================================================================
# run_server ()
#===============================================================================
def run_server(host, port, mode, debug_mode):
    """
    Executes the command to start the server
    Arguments:
        host: retrieved from create_arg_parser() as a string
        port: retrieved from create_arg_parser() as a int
        mode: retrieved from create_arg_parser() as a string
        debug_mode: retrieved from create_arg_parser() as a bool
    """
    
    print('server:      %s:%s' % (host, port))
    print('mode:        %s' % (mode))
    print('debug_mode:  %s' % (debug_mode))

    if mode == 'builtin':
        # Run app in debug mode using flask
        app.run(host, port, debug_mode)
    elif mode == 'waitress':
        # Run app in production mode using waitress
        serve(app, host = host, port = port)
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
    MODES = [ 'builtin', 'waitress' ]
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('h', metavar = 'HOST', help = 'Server HOST (e.g. "localhost")', type = str)
    parser.add_argument('p', metavar = 'PORT', help = 'Server PORT (e.g. "5000")', type = int)
    parser.add_argument('m', metavar='SERVER_MODE', help = ", ".join(MODES), choices = MODES, type = str)
    parser.add_argument('--debug', help = "Run builtin server in debug mode", action = 'store_true', default = False)
    
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
    port = args.p
    mode = args.m
    debug_mode = args.debug
    
    # Run server with user-given arguments
    run_server(host, port, mode, debug_mode)
    
if __name__ == '__main__':
    main()