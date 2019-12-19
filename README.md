# Forecaster Toolbox

## Description

This repository contains the source code of the **Forecaster Toolbox back-end**, which is part of the [SDK4ED Platform](https://sdk4ed.eu/). This Toolbox consists of a backend server dedicated to the deployment of a set of forecasting models and dedicated web services that expose the server. The purpose of the Forecaster Toolbox is to provide predictive forecasts regarding the evolution of the three core quality attributes targeted by the SDK4ED platform, namely Technical Debt, Energy and Dependability. At the moment, the Forecaster Toolbox consists of three individual sub-modules, implemented as individual web services integrated into the back-end. These services are listed below:
- **TD Forecasting**: This web service is responsible for generating Technical Debt forecasts for a given software application, up to an horizon specified by the user.
- **Energy Forecasting**: This web service is responsible for generating Energy Consumption forecasts for a given software application, up to an horizon specified by the user.
- **Dependability Forecasting**: This web service is responsible for generating Dependability (Security) forecasts for a given software application, up to an horizon specified by the user.

Each service plays the role of an end-point that allows the invocation of a set of forecasting models and returns the results, providing users with insightful information for the future evolution each of the three core quality attributes of a software application.

The backend of the Forecaster Toolbox is developped as a Python-based web server that exposes the Forecaster Toolbox API. The server uses the [Flask](https://www.fullstackpython.com/flask.html) web framework wrapped inside [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/), a Python WSGI production-ready server. Forecaster Toolbox API accepts user input as URL-encoded parameters. These parameters include the forecasting horizon, the regressor model that will be used to perform forecasts (optional), and the ID of the project for which to perform forecasts for. If the regressor parameter is omitted, the service selects automatically the best model based on training error minimization. Behind the scenes, forecasting models developed for estimating the evolution of TD, Energy and Security are trained and deployed. The service returns forecasting results as JSON. For now, the service reads input from files stored locally. During the next period, it will be extended to collect required data from TD, Energy and Dependability Toolboxes databases.

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
conda install -c anaconda waitress flask numpy pandas scikit-learn
```
and
```bash
conda install -c saravji pmdarima
```

## Run Server

You can run the server in various modes using the `forecaster_service.py` script:

```
usage: forecaster_service.py [-h] [--debug] HOST PORT SERVER_MODE

positional arguments:
  HOST         Server HOST (e.g. "localhost")
  PORT         Server PORT (e.g. "5000")
  SERVER_MODE  builtin, waitress

optional arguments:
  -h, --help   show this help message and exit
  --debug      Run builtin server in debug mode (default: False)
```

You can change the HOST, PORT and SERVER_MODE according to your needs.

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
python forecaster_service.py 127.0.0.1 5000 builtin --debug
```

This mode is useful for development since it has debugging enabled (e.g. in case of error the client gets a full stack trace).

**Warning**: Single-threaded, debugging enabled. Do NOT use this mode in production!

### Run server using Waitress

```bash
python forecaster_service.py 127.0.0.1 5000 waitress
```

## Usage

### Example

Once the server is running, open your web browser and navigate to the following URL:

http://127.0.0.1:5000/ForecasterToolbox/TDForecasting?horizon=5&project=apache_kafka_measures&regressor=ridge&ground_truth=yes&test=no

You will get a JSON response containing TD forecasts of a sample application (Apache Kafka) for an horizon of 5 versions ahead, using the Ridge regressor model.

**TODO: More details will be added soon regarding Forecaster Toolbox API usage and its parameters**