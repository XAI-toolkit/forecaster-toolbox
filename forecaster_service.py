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
def TDForecasting(horizon = None, regressor = None, project = None):
    """
    API Call to TDForecasting service
    Arguments:
        horizon (sent as URL query parameter) from API Call
        regressor (sent as URL query parameter) from API Call
        project (sent as URL query parameter) from API Call
    Returns:
        A JSON containing the forecasted values, status code and a message.
    """
    
    # Parse URL-encoded parameters
    horizon_param = request.args.get("horizon", type = int) # if key doesn't exist, returns None
    regressor_param = request.args.get("regressor", default = 'auto', type = str) # if key doesn't exist, returns None
    project_param = request.args.get("project", type = str) # if key doesn't exist, returns None
    
    # If required parameters are missing from URL
    if horizon_param is None or regressor_param is None or project_param is None:
        return(unprocessable_entity())
    else:
        # Call build_and_train() function and retrieve forecasts
        results = build_and_train(horizon_param, regressor_param, project_param)
        
        # Compose and jsonify respond
        message = {
                'status': 200,
                'message': 'The selected horizon is {}!'.format(horizon_param),
                'forecast': results,
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
            'message': 'Unprocessable Entity: ' + request.url + ' --> Missing or invalid parameters (required: horizon, regressor, project)',
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
        print("Server mode '%s' not yet implemented" % mode)
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
    parser.add_argument('h', metavar = 'HOST', help = "Server HOST (e.g. 'localhost')", type = str)
    parser.add_argument('p', metavar = 'PORT', help = "Server PORT (e.g. '5000')", type = int)
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