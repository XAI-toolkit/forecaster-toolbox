# Forecaster Toolbox

## Description

This repository contains the source code of the **Forecasting Toolbox back-end**, which is part of the [SDK4ED Platform](https://sdk4ed.eu/). The purpose of the Forecasting Toolbox is to provide predictive forecasts regarding the evolution of the three core quality attributes targeted by the SDK4ED platform, namely Technical Debt, Energy and Dependability (Security). The entry point of the Forecasting Toolbox is a RESTful web server that uses the [Flask web framework](https://www.palletsprojects.com/p/flask/) wrapped inside [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/), a Python WSGI production-ready server. At a lower level, the server exposes three sub-modules, implemented as individual web services. Each web service plays the role of an end-point that allows the invocation of a set of forecasting models and returns the results, providing users with insightful information for the future evolution each of the three core quality attributes of a software application. The services supported by the Forecasting Toolbox are listed below:
- **TD Forecaster**: This web service is responsible for generating TD forecasts for a given software application. A TD forecast represents the predicted evolution of the total remediation effort (measured in minutes) to fix all code issues (e.g. code smells, bugs, code duplications, etc.) of a software application, up to a future point specified by the user.
- **Energy Forecaster**: This web service is responsible for generating Energy forecasts for a given software application. An Energy forecast represents the predicted evolution of the total energy consumption (measured in Joules) of a software application, up to a future point specified by the user.
- **Dependability Forecaster**: This web service is responsible for generating Security forecasts for a given software application. A Security forecast represents the predicted evolution of the Security Index (value between 0 and 1 that aggregates the entire program security characteristics) of a software application, up to a future point specified by the user.

The three web services allow the individual and remote invocation of the forecasting models developed for estimating the evolution of TD, Energy and Security. This is achieved through the dedicated API exposed by the RESTful web server, which allows the user to perform simple HTTP GET requests to the three web services. Several inputs need to be provided as URL-encoded parameters to these requests. These parameters are listed below:
- **horizon (required)**: an integer indicating the desired horizon of the forecast (i.e. versions ahead).
- **project (required)**: a unique identifier of the software project that should be analysed. This identifier facilitates the process of retrieving the analysis results for a particular project from the TD Management, Energy Optimization or Dependability Optimization Toolbox data store, in order to be used for model training.
- **regressor (optional, default value = ‘auto’)**: a string indicating the regressor model that will be used to perform forecasts. If this parameter is omitted, default value is set to ‘auto’ and the service selects automatically the best model based on validation error minimization.
- **ground_truth (optional, default value = ‘no’)**: a string that takes two possible values (yes/no) indicating whether the service should return ground truth values in addition to the forecasts.
- **test (optional, default value = ‘no’)**: a string that takes two possible values (yes/no) indicating whether the forecasting algorithm should return forecasts based on a train/test split of the data. If set to ‘no’, then the service uses the whole data to train a regressor and returns forecasts on unseen data. A value of ‘yes’ should be used only for model testing and not actual deployment into production.

Table below briefly presents valid inputs for each of the parameters supported by the three web services presented above:

|      Parameter     |                                                                                                                                                Description                                                                                                                                               |     Required     |                                                                                                                                                                 Valid Inputs                                                                                                                                                                |
|:------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    horizon         |    The forecasting horizon up to which forecasts will be produced.                                                                                                                                                                                                                                       |        Yes       |    An integer in range [1-N], where N depends on the volume of data used to train the regressor. Currently there is no upper limit and the service returns an error if this value is set too high.                                                                                                                                          |
|    project         |    The project ID for which the forecasts will be produced. This identifier facilitates the process of retrieving the analysis results for a particular project from the TD Management, Energy Optimization or Dependability Optimization Toolbox data store, in order to be used for model training.    |        Yes       |    Currently the following string values are supported for testing purposes:   · TD Forecaster: ‘apache_kafka_measures’   · Security Forecaster: ‘square_retrofit_security_measures’   · Energy Forecaster: ‘sbeamer_gapbs_energy_measures’   Later, this input will be the ID of an actual project integrated into the SDK4ED platform.    |
|    regressor       |    The regressor model that will be used to produce forecasts. If this parameter is omitted, default value is set to ‘auto’ and the service selects automatically the best model based on validation error minimization.                                                                                 |        No        |    One of the following string values: [‘auto’, ‘mlr’, ‘lasso’, ‘ridge’, ‘svr_linear’, ‘svr_rbf’, ‘random_forest’, ‘arima’].    Default value is ‘auto’.                                                                                                                                                                                    |
|    ground_truth    |    If the model will return also ground truth values or not.                                                                                                                                                                                                                                             |        No        |    One of the following string values: [‘yes’, ‘no’].   Default value is ‘no.                                                                                                                                                                                                                                                               |
|    test            |    If the model will produce Train-Test or unseen forecasts. If set to ‘no’, then the service uses the whole data to train a regressor and returns forecasts on unseen data. A value of ‘yes’ should be used only for model testing and not actual deployment into production.                           |        No        |    One of the following string values: [‘yes’, ‘no’].   Default value is ‘no’.                                                                                                                                                                                                                                                              |

The output of the three individual web services provided by the Forecasting Toolbox, namely TD Forecaster, Energy Forecaster and Dependability Forecaster is a JSON file containing the predicted values for a particular quality attribute of the selected application. This JSON actually contains i) a status code of the response, ii) a N-size array containing the forecasts, where N is equal to the ‘horizon’ parameter, iii) a recap on the given parameter values, and iv) a message informing the user if the request was fulfilled successfully or not.

## Requirements

The Forecaster Toolbox is developed to run on Unix and Windows systems with python 3.6.*  nstalled. We suggest installing python via the Anaconda distribution as it provides an easy way to create a virtual environment and install dependencies. The configuration steps needed, are described below:

- Download the latest Anaconda distribution from: https://www.anaconda.com/distribution/
- Follow the installation steps described in the Anaconda documentation: https://docs.anaconda.com/anaconda/install/windows/

## Installation

- Open Anaconda cmd. Running Anaconda cmd activates the base environment. We need to create a specific environment to run Forecaster Toolbox.

- Create a new python 3.6.4 environment by running the command below:
```bash
conda create --name forecaster_toolbox python=3.6.4
```

- Activate the environment:
```bash
conda activate forecaster_toolbox
```

- Install the needed libraries by running:
```bash
conda install -c anaconda numpy pandas scikit-learn waitress flask flask-cors pymongo
```
and
```bash
conda install -c saravji pmdarima
```

## Run Server

You can run the server in various modes using the `forecaster_service.py` script:

```
usage: forecaster_service.py [-h] [--debug] HOST PORT DBNAME SERVER_MODE

positional arguments:
  HOST         Server HOST (e.g. "localhost")
  PORT         Server PORT (e.g. "5000")
  DBNAME       Database name
  SERVER_MODE  builtin, waitress

optional arguments:
  -h, --help   show this help message and exit
  --debug      Run builtin server in debug mode (default: False)
```

You can change the HOST, PORT, DBNAME and SERVER_MODE according to your needs.

Built-in Flask server:

```
         127.0.0.1:5000
Client <----------------> Flask
```

Using Waitress:

```
         127.0.0.1:5000
Client <----------------> Waitress <---> Flask
```

### Run built-in server

```bash
python forecaster_service.py 127.0.0.1 5000 forecasting_toolbox builtin --debug
```

This mode is useful for development since it has debugging enabled (e.g. in case of error the client gets a full stack trace).

**Warning**: Single-threaded, debugging enabled. Do NOT use this mode in production!

### Run server using Waitress

```bash
python forecaster_service.py 127.0.0.1 5000 forecasting_toolbox waitress
```

## Usage

### Example

Once the server is running, open your web browser and navigate to the following URL:

http://127.0.0.1:5000/ForecasterToolbox/TDForecasting?horizon=5&project=apache_kafka_measures&regressor=ridge&ground_truth=yes&test=no

You will get a JSON response containing TD forecasts of a sample application (Apache Kafka) for an horizon of 5 versions ahead, using the Ridge regressor model.

**TODO: More details will be added soon regarding Forecaster Toolbox API usage and its parameters**