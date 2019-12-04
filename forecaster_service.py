# -*- coding: utf-8 -*-
"""
@author: tsoukj
"""

import json
from flask import Flask, jsonify, request
from model_training import build_and_train

# Create the Flask app
app = Flask(__name__)

# Routes
@app.route('/ForecasterToolbox/TDForecasting', methods=['GET'])
def TDForecasting(horizon = None, regressor = None, project = None):
    """API Call

    horizon (sent as URL query parameter) from API Call
    regressor (sent as URL query parameter) from API Call
    """
    horizon_param = request.args.get("horizon", type = int) # if key doesn't exist, returns None
    regressor_param = request.args.get("regressor", default = 'auto', type = str) # if key doesn't exist, returns None
    project_param = request.args.get("project", type = str) # if key doesn't exist, returns None
    
    if horizon_param is None or regressor_param is None or project_param is None:
        return(unprocessable_entity())
    else:
        results = build_and_train(horizon_param, regressor_param, project_param)
        
        message = {
                'status': 200,
                'message': 'The selected horizon is {}!'.format(horizon_param),
                'forecast': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200
        
        return(resp)

# Run app in debug mode on port 5000
if __name__ == '__main__':
    app.run(port = 5000, debug = True)

# Error Handling
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